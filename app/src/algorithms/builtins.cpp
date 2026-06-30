#include "retargeting/builtins.hpp"

#include <memory>

#include "dual_arm_eepose/dual_arm_eepose_algorithm.hpp"
#include "gmr/gmr_algorithm.hpp"
#include "retargeting/algorithm_registry.hpp"

namespace retargeting {

void register_builtin_algorithms() {
  static bool done = false;
  if (done) return;
  done = true;

  AlgorithmRegistry::instance().register_algorithm(
      "gmr", []() -> std::unique_ptr<RetargetingAlgorithm> {
        return std::make_unique<gmr::GmrAlgorithm>();
      });

  AlgorithmRegistry::instance().register_algorithm(
      "dual_arm_eepose", []() -> std::unique_ptr<RetargetingAlgorithm> {
        return std::make_unique<dual_arm_eepose::DualArmEePoseAlgorithm>();
      });
}

}  // namespace retargeting
