# retargeting — motion retargeting toolkit (C++)

A modular C++ toolkit for retargeting tracked human motion onto robot joint
configurations, built for VR teleoperation. It is structured in **two layers**
so that the teleoperation *use case* and the underlying *algorithm* evolve
independently, and so it can be driven from a Godot GDExtension that feeds the
headset's body/hand poses in each frame.

Current release: `0.1.2`.

The first algorithm is a C++ port of **GMR** (General Motion Retargeting),
carried over from the validated standalone port. Its output is **byte-for-byte
identical** to that port on both kinematics backends, and therefore aligned with
the original Python (`mink` + MuJoCo) reference to ~1e-13.

## Two layers

```
                 SkeletonFrame  (VR body/hand poses, keyed by joint name)
                        │
   ┌────────────────────┴─ business layer (include/retargeting/retargeter.hpp)
   │   WholeBodyRetargeter │ UpperBodyRetargeter │ HandRetargeter
   │   - picks an algorithm by name + builds a ScenarioSpec
   │   - drives the per-frame protocol (begin → solve → end)
   │
   └────────────────────┬─ algorithm layer (include/retargeting/algorithm.hpp)
                        │   RetargetingAlgorithm  (interface)
                        │     └─ gmr::GmrAlgorithm  ──▶  gmr::GmrSolver
                        │   AlgorithmRegistry  (factory by name; "gmr", ...)
                        ▼
                 robot qpos  (MuJoCo convention)
```

### Business layer — `src/business/`, `include/retargeting/retargeter.hpp`
Distinguishes the three teleoperation scenarios. Each owns an algorithm and a
`ScenarioSpec` and exposes a single `step(SkeletonFrame) -> qpos`:

| Retargeter            | Scenario    | Specialization                                   |
|-----------------------|-------------|--------------------------------------------------|
| `WholeBodyRetargeter` | whole-body  | nothing locked; full robot is driven             |
| `UpperBodyRetargeter` | upper-body  | floating base + lower body held fixed (qpos lock)|
| `HandRetargeter`      | hand        | finger/hand pose → dexterous-hand joints         |

The scenario differences are expressed as **data** (`ScenarioSpec`), not as
algorithm subclasses, so any algorithm can serve any scenario.

### Algorithm layer — `src/algorithms/`, `include/retargeting/algorithm.hpp`
`RetargetingAlgorithm` is the extension point. Adding an algorithm is a new
folder under `src/algorithms/<name>/` implementing the interface, plus one line
in `src/algorithms/builtins.cpp` to register a factory. The business layer never
changes.

- `algorithms/gmr/` — GMR.
  - `gmr_solver.{hpp,cpp}` — the validated numerical core (QP-based differential
    IK; lie ops + box-QP solver in `src/core/`).
  - `gmr_algorithm.{hpp,cpp}` — adapts the solver to `RetargetingAlgorithm`.
- `algorithms/dual_arm_eepose/` — pure dual-arm end-effector pose IK.
  - Registered as `"dual_arm_eepose"`.
  - Supports robot-world absolute EE targets and shoulder-relative EE source
    modes from `SkeletonFrame` keys (e.g. `LeftShoulder` / `LeftWrist`),
    pins all non-arm DoFs, and uses MuJoCo geom-distance collision constraints
    for arm-arm and arm-torso clearance.
  - The Galbot G1 mapping lives in
    `data/ik_configs/dual_arm_eepose_galbot_g1.json`.

### Shared kinematics core — `src/core/`
Carried over verbatim from the validated GMR port:
- `lie.{hpp,cpp}`, `box_qp.{hpp,cpp}` — SE3 lie ops + box-constrained QP.
- `kinematics_backend.hpp` — pluggable FK / Jacobian / integration interface.
- `pinocchio_backend.{hpp,cpp}` — **Pinocchio** backend (no MuJoCo runtime
  dependency; the VR/Android target).
- `mujoco_backend.{hpp,cpp}` — **MuJoCo** backend (desktop reference cross-check;
  optional via `-DRETARGETING_WITH_MUJOCO=OFF`).

## Test data — `data/`

The toolkit is self-contained: it ships the test data the demo needs.

```
data/
├── robot/unitree_g1/g1_mocap_29dof.xml      # target robot (MJCF)
├── ik_configs/quest3_upper_to_g1.json       # GMR mapping/config
└── reference/qpos_*_f200_fps30_h1.75.csv     # golden output (Pinocchio == Python ~1e-13)
```

Two MJCFs ship: the original `g1_mocap_29dof.xml` and a **meshless**
`g1_mocap_29dof_nomesh.xml` (mesh assets + geoms stripped). Retargeting only
needs the kinematic tree, so neither backend needs the 52MB STL meshes —
Pinocchio builds its model from the MJCF directly, and MuJoCo loads the meshless
MJCF. The reference CSV is the per-frame robot `qpos` and is the contract both
the desktop and on-device runs verify against (PASS tolerance 1e-6; actual
residual ~5e-14).

## Build & validate (desktop)

Requires Eigen3, nlohmann-json, Pinocchio, and (optionally, for the cross-check)
the MuJoCo dylib.

```bash
./build_desktop.sh 200 30 1.75
```

Builds `libretargeting.a` + `upper_body_demo`, runs the synthetic Quest3
upper-body source through `UpperBodyRetargeter`, and verifies against the
checked-in reference. The validation scripts set `GMR_POSTURE_WEIGHT=0` so this
golden test stays aligned with the original Python reference; the SpatialMP4 G1
path keeps the default posture regularizer for steadier real body-pose captures:

```
[run] backend=pinocchio ...
VERIFY: max abs diff vs reference = 0  -> PASS ✓
```

## Galbot Dual-Arm EePose Test

The pure dual-arm eepose path is MuJoCo-only because the collision constraint
uses `mj_geomDistance` / `mj_jacGeom`. The test script below extracts body
joints from a SpatialMP4 capture, aligns them into a `SkeletonFrame`, and lets
`dual_arm_eepose` retarget to the Galbot G1 arm joints before replaying the
result to check the configured collision clearance. The default Galbot mapping
uses `TARGET_MODE=shoulder_delta` inside the algorithm: each frame first
expresses the wrist pose relative to the corresponding shoulder in the current
shoulder/body frame, then applies only that relative end-effector delta to the
previous Galbot TCP target. This keeps body motion separate from arm motion: if
the operator walks or turns without changing the arm pose relative to the
shoulders, the robot TCP target does not move. For experiments that need strict
local-frame SE(3) deltas, use `TARGET_MODE=se3_delta`; `TARGET_MODE=frame_delta`
keeps the old aligned-frame wrist delta behavior for comparison, and
`TARGET_MODE=shoulder_absolute` maps each frame's shoulder-relative wrist
position directly to the robot base.

```bash
cicd/test_spatialmp4_galbot_dual_arm.sh \
  /Users/duino/ws/operator/tmpdir/20260630_090309.mp4 \
  build/galbot_dual_arm_test
```

Override the test robot with `ROBOT_XML=/path/to/galbot_g1.xml` if needed.
The main outputs are `dual_arm_solution.jsonl` (14 arm joint commands),
`eepose_targets.jsonl` (robot-world TCP targets), and
`vr_pose_vs_retargeted.mp4` (left: aligned VR pose; right: retargeted Galbot
skeleton). The VR panel also renders the exported palm and finger keypoints
when the SpatialMP4 body track contains the Godot/Quest hand joints.
Set `VIS_MAX_FRAMES=120` for a short smoke-test render, or leave it unset to
render the full capture.

The Galbot test path keeps the joint output low-pass disabled by default for
delta-pose control: `MAX_JOINT_VEL_DEG_S=240`, `JOINT_LOWPASS_CUTOFF=0`,
and `INPUT_ONE_EURO_MIN_CUTOFF=0`. The OneEuro filter can still be enabled
with `INPUT_ONE_EURO_MIN_CUTOFF>0`; when enabled it is applied after converting
the capture to the selected EE source pose, so shoulder/body motion is not
filtered into arm motion. The default
`ORIENTATION_MODE=relative_wrist_roll` tracks only the wrist roll component
around the VR forearm axis and applies that roll around the configured Galbot
TCP/gripper axis. It does not send the full wrist quaternion to IK. Use
`ORIENTATION_MODE=relative_wrist` only when the input capture provides a
reliable wrist/tool orientation; full wrist quaternion delta tracking can
create unreachable 6D targets when the capture orientation is noisy or not
calibrated to the robot tool frame. Use `ORIENTATION_MODE=neutral` to ignore
wrist orientation and keep the configured neutral Galbot TCP orientation.

Each arm task can also enable an elbow direction soft constraint:
`elbow_constraint.enabled=true` requires a VR elbow source key such as
`LeftArmLower` and a robot elbow body such as `left_arm_link4`. The constraint
aligns only the shoulder-to-elbow direction, not the segment length, so it is
robust to human/robot upper-arm length differences.

The Galbot G1 config also masks the EE rotation task to the wrist joints:
`rotation_joint_names` is set to arm joints 5/6/7 and
`rotation_leak_weight=0`, so EE position can still use the whole arm while EE
orientation is solved by the wrist instead of pulling the shoulder/upper arm.
For roll-only wrist tracking, `rotation_roll_axis` specifies the Galbot local
TCP/gripper axis used by `relative_wrist_roll`, and `rotation_roll_scale` can
be set to `-1` if a robot model needs the opposite roll sign.

`wrist_forearm_constraint.enabled=true` is available as an experimental
naturalness constraint for grasping poses: it compares the VR forearm direction
with a configured VR hand axis, then discourages solutions where the robot TCP
axis bends much farther away from the robot forearm. It is not enabled in the
default Galbot G1 config because even weak wrist-axis constraints can pull the
shoulder and upper-arm joints away from the elbow/EE tasks.

## Build & run on device (Android / NDK)

The Android target uses the **MuJoCo** backend. MuJoCo loads MJCF natively,
whereas the `pinocchio-android` build is core-only (no MJCF parser), so it
cannot construct the model on device. MuJoCo for Android is built by the
`mujoco-android` wrapper (see `make -C xr build-mujoco-android`). Eigen is
header-only (host), nlohmann-json is vendored under `third_party/` — so the
on-device build needs neither Pinocchio nor vcpkg.

```bash
export ANDROID_NDK=$ANDROID_NDK_HOME          # NDK r26+
make -C xr build-mujoco-android               # once: builds libmujoco.so for arm64-v8a
./build_android.sh                            # build + push + run + verify on device
# ./build_android.sh --no-run                 # build only
```

`build_android.sh` cross-compiles the demo (`-DRETARGETING_WITH_PINOCCHIO=OFF
-DRETARGETING_WITH_MUJOCO=ON`), stages `libmujoco.so` + `libc++_shared.so` +
`data/`, pushes to `/data/local/tmp/retargeting`, and runs with the meshless
MJCF. Verified on a physical arm64 device:

```
scenario=upper_body algorithm=gmr nq=36
VERIFY: max abs diff vs reference = 4.92661e-14  -> PASS ✓
```

i.e. the arm64 device reproduces the desktop/Python result to machine precision.

## Versioning

The root `VERSION` file is the single source of truth. CMake reads it into
`project(retargeting VERSION ...)` and generates `retargeting/version.hpp`, so
library users and command-line tools report the same version:

```bash
./build/upper_body_demo --version
./build/spatialmp4_to_g1 --version
```

Create a local annotated release tag from the checked-in version with:

```bash
cicd/release.sh
```

### Backends

| | Pinocchio | MuJoCo |
|---|---|---|
| Desktop | ✅ default (`find_package`, homebrew) | ✅ cross-check (GMR `.venv` wheel) |
| Android | ❌ core-only build lacks MJCF parser | ✅ `mujoco-android` (`libmujoco.so`) |

Toggle with `-DRETARGETING_WITH_PINOCCHIO=ON/OFF` and
`-DRETARGETING_WITH_MUJOCO=ON/OFF` (at least one required). The Android
`libmujoco.so` is produced by the **mujoco-android** wrapper
(<https://github.com/DuinoDu/mujoco-android>), synced into
`.deps/src/mujoco-android` by `make -C xr deps` and built by
`xr/makefiles/Makefile.mujoco-android` (`make -C xr build-mujoco-android`) —
mirroring exactly how `pinocchio-android` is integrated.

## Toward Godot / VR

The toolkit is intentionally free of any Godot dependency. The integration is a
thin GDExtension wrapper (separate `build.sh`, like the sibling `pinocchio`
module) that:
1. converts the headset's body/hand tracking into a `SkeletonFrame`,
2. holds a `*Retargeter` for the active scenario, and
3. calls `step()` per frame and pushes the resulting `qpos` to the robot / sim.

Because the algorithms register through `AlgorithmRegistry` and the business
layer is data-driven, the GDExtension chooses scenario + algorithm by string at
runtime. Note: when linking the static lib into a shared object, force the
algorithm TUs in (`-Wl,-force_load` / `--whole-archive`) or call
`retargeting::register_builtin_algorithms()` — the business-layer factories
already call it for you.

## Status

- whole-body / upper-body: working on GMR; upper-body validated against the
  Python reference via the GMR port.
- dual-arm eepose: working on MuJoCo with collision-constrained Galbot G1
  configuration and SpatialMP4 offline test coverage.
- hand: wired through the same interface; needs a dexterous-hand MJCF +
  hand `ik_config` (and likely a hand-specific algorithm) to produce motion.
