#!/usr/bin/env bash
# End-to-end offline test: SpatialMP4 VR body pose -> retarget to Unitree G1 ->
# saved MP4 visualization. This is the full, headless replica of the on-device
# overlay path, runnable without a headset.
#
# Stages:
#   1. extract  : pull the body_joints track out of the .mp4 -> body.jsonl
#                 (raw OpenXR joints; tools/extract_spatialmp4_body_joints.py)
#   2. retarget : body.jsonl -> robot_solution.jsonl (per-frame G1 joint_q)
#                 (build/spatialmp4_to_g1 — the C++ retargeting pipeline)
#   3. render   : robot_solution.jsonl -> g1.mp4 (MuJoCo offscreen renderer)
#                 (tools/render_g1.py, run under the GMR venv python)
#
# Prerequisites: ffmpeg/ffprobe on PATH, and a Python with mujoco + imageio
# (defaults to the GMR venv; override with RENDER_PY=/path/to/python).
#
# Usage: cicd/test_spatialmp4_g1.sh [session_mp4] [out_dir]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MP4="${1:-data/spatialmp4/20260622_083748.mp4}"
OUT_DIR="${2:-build/spatialmp4_g1}"
RENDER_PY="${RENDER_PY:-$HOME/.cache/install-x/GMR/.venv/bin/python}"
FPS="${FPS:-}"
RETARGET_MAX_JOINT_VEL_DEG_S="${RETARGET_MAX_JOINT_VEL_DEG_S:-180}"

ROBOT_XML="data/robot/unitree_g1/g1_mocap_29dof_nomesh.xml"
IK_CONFIG="data/ik_configs/quest3_upper_to_g1.json"
# The repo ships the meshed XML but not the 52MB STLs; render with the meshed
# G1 model that carries them (GMR assets). Override with RENDER_MODEL=.
RENDER_MODEL="${RENDER_MODEL:-$HOME/.cache/install-x/GMR/assets/unitree_g1/g1_mocap_29dof.xml}"

[[ -f "$MP4" ]] || { echo "ERROR: no such mp4: $MP4" >&2; exit 1; }
for bin in ffprobe ffmpeg; do
  command -v "$bin" >/dev/null || { echo "ERROR: $bin not on PATH" >&2; exit 1; }
done
[[ -x "$RENDER_PY" ]] || { echo "ERROR: render python not found: $RENDER_PY (set RENDER_PY)" >&2; exit 1; }
[[ -f "$RENDER_MODEL" ]] || { echo "ERROR: render model not found: $RENDER_MODEL (set RENDER_MODEL=)" >&2; exit 1; }

mkdir -p "$OUT_DIR"
BODY_JSONL="$OUT_DIR/body.jsonl"
SOLUTION_JSONL="$OUT_DIR/robot_solution.jsonl"
OUT_MP4="$OUT_DIR/g1.mp4"

# --- 0. build the retargeting binary --------------------------------------
cmake -S app -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build --target spatialmp4_to_g1 -j >/dev/null
echo "[build] spatialmp4_to_g1 ready"

# --- 1. extract VR body pose from the mp4 ---------------------------------
echo "[1/3] extract body pose: $MP4"
"$RENDER_PY" tools/extract_spatialmp4_body_joints.py "$MP4" --output "$BODY_JSONL"
N_FRAMES="$(wc -l < "$BODY_JSONL" | tr -d ' ')"
echo "       -> $BODY_JSONL ($N_FRAMES frames)"
# The extractor silently skips frames with no mapped upper-body joints; an empty
# body.jsonl would otherwise sail through retarget/render and report a bogus OK.
[[ "${N_FRAMES:-0}" -gt 0 ]] || { echo "ERROR: extractor produced 0 mapped frames from $MP4 (no body_joints track, or HJNT mismatch)" >&2; exit 1; }

# --- 2. retarget to G1 ----------------------------------------------------
echo "[2/3] retarget to G1"
./build/spatialmp4_to_g1 "$BODY_JSONL" \
  --save_jsonl "$SOLUTION_JSONL" \
  --robot_xml "$ROBOT_XML" --ik_config "$IK_CONFIG" \
  --max_joint_velocity_deg_s "$RETARGET_MAX_JOINT_VEL_DEG_S"
echo "       -> $SOLUTION_JSONL"

# --- 3. render the G1 visualization ---------------------------------------
echo "[3/3] render G1 visualization"
RENDER_ARGS=("$SOLUTION_JSONL" --model "$RENDER_MODEL" --out "$OUT_MP4")
if [[ -n "$FPS" ]]; then
  RENDER_ARGS+=(--fps "$FPS")
fi
"$RENDER_PY" tools/render_g1.py "${RENDER_ARGS[@]}"

echo
echo "OK: $OUT_MP4"
