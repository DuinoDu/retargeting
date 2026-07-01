# Changelog

All notable changes to this project are tracked here.

## [0.1.2] - 2026-07-01

### Added
- Added Galbot dual-arm `relative_wrist_roll` orientation retargeting, which
  extracts only the wrist roll around the VR forearm axis and applies it around
  the configured robot TCP/gripper axis.
- Added per-task `rotation_joint_names`, `rotation_leak_weight`,
  `rotation_roll_axis`, and `rotation_roll_scale` controls so Galbot EE
  rotation can be solved by wrist joints 5/6/7 without pulling the upper arm.
- Added elbow direction soft constraints for dual-arm EE pose retargeting.
- Added SpatialMP4 hand keypoint extraction and Galbot skeleton/gripper
  visualization for the dual-arm validation video.

### Changed
- The Galbot dual-arm test path now defaults to shoulder-relative delta pose
  retargeting with roll-only wrist orientation and input/joint low-pass filters
  disabled.
- The Galbot dual-arm config starts from a VR-like initial pose and keeps
  collision constraints enabled for offline validation.

## [0.1.1] - 2026-07-01

### Added
- Added a MuJoCo-based `dual_arm_eepose` retargeting algorithm for direct
  left/right end-effector pose IK with arm-arm and arm-torso collision
  constraints.
- Added a Galbot G1 dual-arm SpatialMP4 test path, retargeting config, offline
  converter, and skeleton visualization output.

### Changed
- The Galbot dual-arm path now drives IK from frame-to-frame relative end
  effector motion by default, with joint output low-pass filtering disabled for
  delta-pose control.

## [0.1.0] - 2026-06-23

### Added
- Added project-wide version management via the root `VERSION` file.
- Added generated C++ version metadata at `retargeting/version.hpp`.
- Added `--version` support to the C++ command-line tools.
- Added input OneEuro filtering for SpatialMP4 to G1 retargeting.

### Changed
- Tuned the SpatialMP4 G1 CI path for steadier right-arm output on noisy body-pose captures.
