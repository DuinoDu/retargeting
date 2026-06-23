#include "one_euro_filter.hpp"

#include <cmath>

namespace retargeting {
namespace utils {

namespace {

constexpr double kFallbackFrameDt = 1.0 / 30.0;
constexpr double kTwoPi = 6.28318530717958647692;

double one_euro_alpha(double cutoff_hz, double dt) {
  if (!(cutoff_hz > 0.0) || !(dt > 0.0) || !std::isfinite(dt)) return 1.0;
  const double tau = 1.0 / (kTwoPi * cutoff_hz);
  return 1.0 / (1.0 + tau / dt);
}

Eigen::Quaterniond normalized_quat(const Eigen::Vector4d& q) {
  Eigen::Quaterniond out(q[0], q[1], q[2], q[3]);
  if (out.norm() > 1e-12) out.normalize();
  else out = Eigen::Quaterniond::Identity();
  return out;
}

double quat_angle(const Eigen::Quaterniond& a, const Eigen::Quaterniond& b) {
  double d = std::fabs(a.dot(b));
  if (d > 1.0) d = 1.0;
  return 2.0 * std::acos(d);
}

}  // namespace

Eigen::Vector3d OneEuroPoseFilter::Vector3State::filter(
    const Eigen::Vector3d& raw, double dt,
    double min_cutoff, double beta, double d_cutoff) {
  if (!initialized) {
    initialized = true;
    raw_prev = raw;
    value = raw;
    dx.setZero();
    return value;
  }
  if (!(dt > 0.0) || !std::isfinite(dt)) dt = kFallbackFrameDt;
  const Eigen::Vector3d raw_dx = (raw - raw_prev) / dt;
  const double dx_alpha = one_euro_alpha(d_cutoff, dt);
  dx = dx_alpha * raw_dx + (1.0 - dx_alpha) * dx;
  const double cutoff = min_cutoff + beta * dx.norm();
  const double a = one_euro_alpha(cutoff, dt);
  value = a * raw + (1.0 - a) * value;
  raw_prev = raw;
  return value;
}

Eigen::Vector4d OneEuroPoseFilter::QuatState::filter(
    const Eigen::Vector4d& raw_vec, double dt,
    double min_cutoff, double beta, double d_cutoff) {
  Eigen::Quaterniond raw = normalized_quat(raw_vec);
  if (!initialized) {
    initialized = true;
    raw_prev = raw;
    value = raw;
    angular_velocity = 0.0;
    return Eigen::Vector4d(value.w(), value.x(), value.y(), value.z());
  }
  if (!(dt > 0.0) || !std::isfinite(dt)) dt = kFallbackFrameDt;
  if (raw.dot(raw_prev) < 0.0) raw.coeffs() *= -1.0;
  const double raw_angular_velocity = quat_angle(raw_prev, raw) / dt;
  const double dx_alpha = one_euro_alpha(d_cutoff, dt);
  angular_velocity = dx_alpha * raw_angular_velocity + (1.0 - dx_alpha) * angular_velocity;
  const double cutoff = min_cutoff + beta * angular_velocity;
  const double a = one_euro_alpha(cutoff, dt);
  if (raw.dot(value) < 0.0) raw.coeffs() *= -1.0;
  value = value.slerp(a, raw);
  value.normalize();
  raw_prev = raw;
  return Eigen::Vector4d(value.w(), value.x(), value.y(), value.z());
}

OneEuroPoseFilter::OneEuroPoseFilter(double min_cutoff, double beta, double d_cutoff)
    : min_cutoff_(min_cutoff), beta_(beta), d_cutoff_(d_cutoff) {}

bool OneEuroPoseFilter::enabled() const {
  return min_cutoff_ > 0.0;
}

void OneEuroPoseFilter::reset() {
  has_timestamp_ = false;
  prev_timestamp_s_ = 0.0;
  pos_filters_.clear();
  quat_filters_.clear();
}

void OneEuroPoseFilter::filter(double timestamp_s, std::vector<NamedPose>& poses) {
  if (!enabled()) return;
  double dt = kFallbackFrameDt;
  if (has_timestamp_) {
    dt = timestamp_s - prev_timestamp_s_;
    if (!(dt > 0.0) || !std::isfinite(dt)) dt = kFallbackFrameDt;
  }
  for (auto& pose : poses) {
    pose.pos = pos_filters_[pose.name].filter(pose.pos, dt, min_cutoff_, beta_, d_cutoff_);
    pose.quat = quat_filters_[pose.name].filter(pose.quat, dt, min_cutoff_, beta_, d_cutoff_);
  }
  prev_timestamp_s_ = timestamp_s;
  has_timestamp_ = true;
}

}  // namespace utils
}  // namespace retargeting
