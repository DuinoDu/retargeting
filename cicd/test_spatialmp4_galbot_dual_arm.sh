#!/usr/bin/env bash
# End-to-end offline test for the pure dual-arm eepose retargeter:
# SpatialMP4 body_joints -> dual_arm_eepose -> Galbot G1 arm joint_q
# -> side-by-side visualization (left: VR pose, right: retargeted pose).
#
# Usage:
#   cicd/test_spatialmp4_galbot_dual_arm.sh [session_mp4] [out_dir]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MP4="${1:-/Users/duino/ws/operator/tmpdir/20260630_090309.mp4}"
OUT_DIR="${2:-build/galbot_dual_arm_test}"
ROBOT_XML="${ROBOT_XML:-/Users/duino/ws/operator/.conductor/worktrees/035b53/xr/assets/mujoco/galbot_g1.xml}"
EE_CONFIG="${EE_CONFIG:-data/ik_configs/dual_arm_eepose_galbot_g1.json}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.cache/install-x/GMR/.venv/bin/python}"
MAX_JOINT_VEL_DEG_S="${MAX_JOINT_VEL_DEG_S:-240}"
JOINT_LOWPASS_CUTOFF="${JOINT_LOWPASS_CUTOFF:-0}"
INPUT_ONE_EURO_MIN_CUTOFF="${INPUT_ONE_EURO_MIN_CUTOFF:-0.6}"
INPUT_ONE_EURO_BETA="${INPUT_ONE_EURO_BETA:-0.03}"
INPUT_ONE_EURO_D_CUTOFF="${INPUT_ONE_EURO_D_CUTOFF:-1.0}"
WORKSPACE_SCALE="${WORKSPACE_SCALE:-1.0}"
ARM_REACH_SCALE="${ARM_REACH_SCALE:-0.82}"
TARGET_MODE="${TARGET_MODE:-shoulder_delta}"
ORIENTATION_MODE="${ORIENTATION_MODE:-neutral}"
FPS="${FPS:-}"
VIS_MAX_FRAMES="${VIS_MAX_FRAMES:-0}"

[[ -f "$MP4" ]] || { echo "ERROR: no such mp4: $MP4" >&2; exit 1; }
[[ -f "$ROBOT_XML" ]] || { echo "ERROR: no such robot xml: $ROBOT_XML" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: python not found: $PYTHON_BIN" >&2; exit 1; }
for bin in ffprobe ffmpeg; do
  command -v "$bin" >/dev/null || { echo "ERROR: $bin not on PATH" >&2; exit 1; }
done

mkdir -p "$OUT_DIR"
BODY_JSONL="$OUT_DIR/body.jsonl"
TARGET_JSONL="$OUT_DIR/eepose_targets.jsonl"
SOLUTION_JSONL="$OUT_DIR/dual_arm_solution.jsonl"
VIS_MP4="${VIS_MP4:-$OUT_DIR/vr_pose_vs_retargeted.mp4}"

cmake -S app -B build -DCMAKE_BUILD_TYPE=Release -DRETARGETING_BUILD_RERUN=OFF >/dev/null
cmake --build build --target spatialmp4_to_galbot_dual_arm -j >/dev/null
echo "[build] spatialmp4_to_galbot_dual_arm ready"

echo "[1/4] extract body pose: $MP4"
"$PYTHON_BIN" tools/extract_spatialmp4_body_joints.py "$MP4" --output "$BODY_JSONL"
N_FRAMES="$(wc -l < "$BODY_JSONL" | tr -d ' ')"
[[ "${N_FRAMES:-0}" -gt 0 ]] || { echo "ERROR: extractor produced 0 frames" >&2; exit 1; }

echo "[2/4] retarget dual-arm eepose to Galbot"
./build/spatialmp4_to_galbot_dual_arm "$BODY_JSONL" \
  --robot_xml "$ROBOT_XML" \
  --ee_config "$EE_CONFIG" \
  --save_jsonl "$SOLUTION_JSONL" \
  --target_jsonl "$TARGET_JSONL" \
  --workspace_scale "$WORKSPACE_SCALE" \
  --arm_reach_scale "$ARM_REACH_SCALE" \
  --target_mode "$TARGET_MODE" \
  --orientation_mode "$ORIENTATION_MODE" \
  --max_joint_velocity_deg_s "$MAX_JOINT_VEL_DEG_S" \
  --joint_lowpass_cutoff "$JOINT_LOWPASS_CUTOFF" \
  --input_one_euro_min_cutoff "$INPUT_ONE_EURO_MIN_CUTOFF" \
  --input_one_euro_beta "$INPUT_ONE_EURO_BETA" \
  --input_one_euro_d_cutoff "$INPUT_ONE_EURO_D_CUTOFF"

echo "[3/4] replay collision check"
"$PYTHON_BIN" - "$ROBOT_XML" "$SOLUTION_JSONL" <<'PY'
import json
import sys

import mujoco

model_path, solution_path = sys.argv[1], sys.argv[2]
left = [
    "left_arm_link3_proxy",
    "left_arm_link4_proxy",
    "left_arm_link5_proxy",
    "left_arm_link6_proxy",
    "left_arm_link7_proxy",
    "left_arm_end_effector_mount_link_proxy",
    "left_gripper_base_link_proxy",
    "left_gripper_tcp_link_proxy",
]
right = [
    "right_arm_link3_proxy",
    "right_arm_link4_proxy",
    "right_arm_link5_proxy",
    "right_arm_link6_proxy",
    "right_arm_link7_proxy",
    "right_arm_end_effector_mount_link_proxy",
    "right_gripper_base_link_proxy",
    "right_gripper_tcp_link_proxy",
]
torso = ["torso_base_link_proxy", "torso_head_mount_link_proxy"]

m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)
gid = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)

with open(solution_path) as f:
    first = json.loads(next(f))
joint_names = first["joint_names"]
qadr = [
    m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]
    for name in joint_names
]

worst = (float("inf"), -1, None)
below = 0
groups = [(left, right), (left, torso), (right, torso)]
with open(solution_path) as f:
    for fi, line in enumerate(f):
        rec = json.loads(line)
        mujoco.mj_resetData(m, d)
        for qi, q in zip(qadr, rec["joint_q"]):
            d.qpos[qi] = q
        mujoco.mj_forward(m, d)
        best = float("inf")
        pair = None
        for a, b in groups:
            for ga_name in a:
                ga = gid(ga_name)
                for gb_name in b:
                    dist = float(mujoco.mj_geomDistance(m, d, ga, gid(gb_name), 1.0, None))
                    if dist < best:
                        best = dist
                        pair = (ga_name, gb_name)
        if best < 0.06:
            below += 1
        if best < worst[0]:
            worst = (best, fi, pair)

print(
    f"min_collision_distance={worst[0]:.6f} "
    f"worst_frame={worst[1]} pair={worst[2]} below_0.06={below}"
)
if below:
    raise SystemExit(1)
PY

echo "[4/4] render visualization"
RENDER_ARGS=(
  tools/render_galbot_dual_arm.py
  --body_jsonl "$BODY_JSONL"
  --solution_jsonl "$SOLUTION_JSONL"
  --model "$ROBOT_XML"
  --out "$VIS_MP4"
  --max_frames "$VIS_MAX_FRAMES"
)
if [[ -n "$FPS" ]]; then
  RENDER_ARGS+=(--fps "$FPS")
fi
"$PYTHON_BIN" "${RENDER_ARGS[@]}"

echo
echo "OK: $SOLUTION_JSONL"
echo "OK: $TARGET_JSONL"
echo "OK: $VIS_MP4"
