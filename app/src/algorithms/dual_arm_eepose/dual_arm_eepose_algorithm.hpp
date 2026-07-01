// Pure dual-arm end-effector pose retargeting.
//
// This algorithm supports robot-world absolute end-effector targets and
// shoulder-relative source modes from SkeletonFrame keys (for example
// "LeftShoulder" / "LeftWrist"). It solves only the configured arm joints. All
// other robot DoFs are pinned in the QP, making it suitable for dual-arm
// control tasks where base/leg/head/gripper states are controlled by other
// modules. The MuJoCo implementation also adds geom-distance collision
// avoidance constraints for arm-arm and arm-torso pairs.
#pragma once

#include <memory>

#include "retargeting/algorithm.hpp"

namespace retargeting {
namespace dual_arm_eepose {

class DualArmEePoseAlgorithm : public RetargetingAlgorithm {
 public:
  DualArmEePoseAlgorithm();
  ~DualArmEePoseAlgorithm() override;

  const char* name() const override { return "dual_arm_eepose"; }

  void configure(const RetargetConfig& config, const ScenarioSpec& spec) override;
  void begin_frame() override;
  Eigen::VectorXd solve(const SkeletonFrame& frame) override;
  void end_frame(Eigen::VectorXd& qpos) override;

  int nq() const override;
  int nv() const override;
  void set_configuration(const Eigen::VectorXd& qpos) override;
  SkeletonFrame last_targets() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dual_arm_eepose
}  // namespace retargeting
