#pragma once

#include <map>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace retargeting {
namespace utils {

struct NamedPose {
  std::string name;
  Eigen::Vector3d pos;
  Eigen::Vector4d quat;  // w, x, y, z
};

class OneEuroPoseFilter {
 public:
  OneEuroPoseFilter(double min_cutoff, double beta, double d_cutoff);

  bool enabled() const;
  void reset();
  void filter(double timestamp_s, std::vector<NamedPose>& poses);

 private:
  struct Vector3State {
    bool initialized = false;
    Eigen::Vector3d raw_prev = Eigen::Vector3d::Zero();
    Eigen::Vector3d value = Eigen::Vector3d::Zero();
    Eigen::Vector3d dx = Eigen::Vector3d::Zero();

    Eigen::Vector3d filter(const Eigen::Vector3d& raw, double dt,
                           double min_cutoff, double beta, double d_cutoff);
  };

  struct QuatState {
    bool initialized = false;
    Eigen::Quaterniond raw_prev = Eigen::Quaterniond::Identity();
    Eigen::Quaterniond value = Eigen::Quaterniond::Identity();
    double angular_velocity = 0.0;

    Eigen::Vector4d filter(const Eigen::Vector4d& raw_vec, double dt,
                           double min_cutoff, double beta, double d_cutoff);
  };

  double min_cutoff_ = 0.0;
  double beta_ = 0.0;
  double d_cutoff_ = 1.0;
  bool has_timestamp_ = false;
  double prev_timestamp_s_ = 0.0;
  std::map<std::string, Vector3State> pos_filters_;
  std::map<std::string, QuatState> quat_filters_;
};

}  // namespace utils
}  // namespace retargeting
