# Offline spatialmp4 → G1 pipeline

A headless, fully-offline replica of GMR's `scripts/spatialmp4_body_to_g1.py`,
used to develop and validate the on-device G1 retargeting overlay without a
headset. It runs the exact same chain the overlay does at runtime:

```
SpatialMP4/Quest body JSONL (raw OpenXR joints)
  -> OpenXR→GMR convert + heading-align + anchor (preprocess)
  -> retarget to Unitree G1 (legs + wrists + waist roll/pitch held; arms + waist-yaw driven)
  -> robot_solution.jsonl (per-frame joint_q)        [apps/spatialmp4_to_g1]
  -> MP4 visualization                                [tools/render_g1.py]
```

## Build + run the pipeline (C++)

```bash
cmake -S . -B build && cmake --build build --target spatialmp4_to_g1 -j

./build/spatialmp4_to_g1 <body.jsonl> \
    --save_jsonl out_robot_solution.jsonl \
    --input_one_euro_min_cutoff 2.0 \
    --input_one_euro_beta 0.20 \
    --input_one_euro_d_cutoff 1.0 \
    --max_joint_velocity_deg_s 90 \
    --normalized out_gmr.jsonl \
    --compare <python_robot_solution.jsonl>
```

`--compare` prints the max abs joint-angle difference vs a Python reference.
With `posture_weight` disabled and the same velocity-limit settings, the C++
chain reproduces the Python pipeline to machine precision end to end from the
raw VR pose. The default spatialmp4 config now keeps a small posture regularizer
enabled because it gives a visibly steadier G1 right arm on noisy body-pose
captures.

Key details replicated from Python (all verified against the reference):
- `OPENXR_TO_GMR` position map = `(-z, -x, y)`, orientation = `q_rot ⊗ q` with
  `q_rot = (0.5, 0.5, -0.5, -0.5)` (wxyz).
- `anchor_and_align_frame`: fixed initial heading yaw + hips pinned to
  `pelvis_height` (0.88).
- Freeze set: legs (locked prefix 19) + wrists + waist roll/pitch (clamp-after),
  leaving the arms + waist-yaw free.
- Optional input OneEuro filtering runs after OpenXR->GMR normalization and
  before IK. It filters both joint positions and quaternions in the same space
  that is written to `--normalized`.
- `posture_weight` in `quest3_upper_to_g1.json` regularizes the free upper-body
  joints toward rest in the IK null space. This reduces right-arm shoulder/elbow
  branch jitter without changing the tracked wrist target appreciably.
- Optional `--max_joint_velocity_deg_s` limits per-frame robot joint jumps using
  the body-pose timestamps and feeds the limited qpos back into the next IK
  frame. The CI spatialmp4 test enables `90` by default to suppress visible arm
  jitter on noisy Quest body-pose frames.

> Note: the **orientation matters** even though the IK config uses position-only
> tasks — the SE3 IK error couples the position tangent to the target rotation
> (`v = V⁻¹(ω)·t`). Feeding identity quaternions gives visibly wrong arm angles.

## Visualize (Python + MuJoCo)

```bash
~/.cache/install-x/GMR/.venv/bin/python tools/render_g1.py \
    out_robot_solution.jsonl --out g1.mp4
```

Renders the G1 (g1_mocap_29dof model) from the per-frame `joint_q`. When
`timestamp_ns` is present, the renderer infers FPS from the solution timestamps;
pass `--fps` only to force a specific playback rate.
