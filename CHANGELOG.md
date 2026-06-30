# Changelog

All notable changes to this project are tracked here.

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
