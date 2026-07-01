#include "dual_arm_eepose_algorithm.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <nlohmann/json.hpp>

#include "box_qp.hpp"
#include "lie.hpp"
#include "utils/one_euro_filter.hpp"

#ifndef RETARGETING_NO_MUJOCO
#include <mujoco/mujoco.h>
#endif

namespace retargeting {
namespace dual_arm_eepose {

using json = nlohmann::json;

#ifdef RETARGETING_NO_MUJOCO

struct DualArmEePoseAlgorithm::Impl {};

DualArmEePoseAlgorithm::DualArmEePoseAlgorithm() : impl_(std::make_unique<Impl>()) {}
DualArmEePoseAlgorithm::~DualArmEePoseAlgorithm() = default;

void DualArmEePoseAlgorithm::configure(const RetargetConfig&, const ScenarioSpec&) {
  throw std::runtime_error(
      "dual_arm_eepose requires the MuJoCo backend; rebuild with RETARGETING_WITH_MUJOCO=ON");
}
void DualArmEePoseAlgorithm::begin_frame() {}
Eigen::VectorXd DualArmEePoseAlgorithm::solve(const SkeletonFrame&) { return Eigen::VectorXd(); }
void DualArmEePoseAlgorithm::end_frame(Eigen::VectorXd&) {}
int DualArmEePoseAlgorithm::nq() const { return 0; }
int DualArmEePoseAlgorithm::nv() const { return 0; }
void DualArmEePoseAlgorithm::set_configuration(const Eigen::VectorXd&) {}
SkeletonFrame DualArmEePoseAlgorithm::last_targets() const { return {}; }

#else

namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<std::string> string_array(const json& value) {
  std::vector<std::string> out;
  for (const auto& item : value) out.push_back(item.get<std::string>());
  return out;
}

Eigen::Vector4d normalized_quat(const Eigen::Vector4d& q) {
  double n = q.norm();
  if (n <= 0.0 || !std::isfinite(n)) return Eigen::Vector4d(1, 0, 0, 0);
  return q / n;
}

Eigen::Vector4d qmul(const Eigen::Vector4d& a, const Eigen::Vector4d& b) {
  return Eigen::Vector4d(
      a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
      a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
      a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
      a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]);
}

Eigen::Vector4d qconj(const Eigen::Vector4d& q) {
  return Eigen::Vector4d(q[0], -q[1], -q[2], -q[3]);
}

Eigen::Vector3d qrotate(const Eigen::Vector4d& q, const Eigen::Vector3d& v) {
  Eigen::Vector4d p(0, v.x(), v.y(), v.z());
  Eigen::Vector4d out = qmul(qmul(normalized_quat(q), p), qconj(normalized_quat(q)));
  return Eigen::Vector3d(out[1], out[2], out[3]);
}

Eigen::Vector4d quat_from_rotation(const Eigen::Matrix3d& R) {
  Eigen::Quaterniond q(R);
  return normalized_quat(Eigen::Vector4d(q.w(), q.x(), q.y(), q.z()));
}

bool find_pose(const SkeletonFrame& frame, const std::vector<std::string>& names, Pose& out) {
  for (const std::string& name : names) {
    auto it = frame.find(name);
    if (it == frame.end()) continue;
    out = it->second;
    return true;
  }
  return false;
}

std::string option_string(const RetargetConfig& config, const std::string& key,
                          const std::string& fallback) {
  auto it = config.options.find(key);
  return it == config.options.end() ? fallback : it->second;
}

double option_double(const RetargetConfig& config, const std::string& key, double fallback) {
  auto it = config.options.find(key);
  if (it == config.options.end()) return fallback;
  return std::stod(it->second);
}

double clamp_unit(double value) {
  return std::max(-1.0, std::min(1.0, value));
}

Eigen::Vector3d parse_axis(const json& value, const std::string& field) {
  if (!value.is_array() || value.size() != 3)
    throw std::runtime_error(field + " must be a 3-element array");
  Eigen::Vector3d axis(value[0].get<double>(), value[1].get<double>(),
                       value[2].get<double>());
  const double norm = axis.norm();
  if (!(norm > 1e-9) || !std::isfinite(norm))
    throw std::runtime_error(field + " must be a finite non-zero vector");
  return axis / norm;
}

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d S;
  S << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
       -v.y(), v.x(), 0.0;
  return S;
}

double normalize_angle(double angle) {
  while (angle > kPi) angle -= 2.0 * kPi;
  while (angle < -kPi) angle += 2.0 * kPi;
  return angle;
}

Eigen::Vector4d axis_angle_quat(const Eigen::Vector3d& axis, double angle) {
  const double norm = axis.norm();
  if (!(norm > 1e-9) || !std::isfinite(norm) || !std::isfinite(angle))
    return Eigen::Vector4d(1, 0, 0, 0);
  const Eigen::Vector3d a = axis / norm;
  const double half = 0.5 * angle;
  const double s = std::sin(half);
  return normalized_quat(Eigen::Vector4d(std::cos(half), a.x() * s, a.y() * s, a.z() * s));
}

double signed_twist_angle(const Eigen::Vector4d& rotation,
                          const Eigen::Vector3d& axis) {
  const double norm = axis.norm();
  if (!(norm > 1e-9) || !std::isfinite(norm)) return 0.0;
  Eigen::Vector4d q = normalized_quat(rotation);
  if (q[0] < 0.0) q = -q;
  const Eigen::Vector3d a = axis / norm;
  const double projected = q.tail<3>().dot(a);
  return normalize_angle(2.0 * std::atan2(projected, q[0]));
}

}  // namespace

struct DualArmEePoseAlgorithm::Impl {
  enum class SourceMode {
    RobotWorldAbsolute,
    ShoulderDelta,
    FrameDelta,
    Se3Delta,
    ShoulderAbsolute,
  };

  enum class OrientationMode {
    Neutral,
    RelativeWrist,
    RelativeWristRoll,
  };

  struct SourcePose {
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Vector4d quat = Eigen::Vector4d(1, 0, 0, 0);
    Eigen::Vector3d roll_axis = Eigen::Vector3d::UnitX();
    bool has_roll_axis = false;
  };

  struct TargetState {
    bool initialized = false;
    Eigen::Vector3d prev_source_pos = Eigen::Vector3d::Zero();
    Eigen::Vector4d prev_source_quat = Eigen::Vector4d(1, 0, 0, 0);
    Eigen::Vector4d initial_source_quat = Eigen::Vector4d(1, 0, 0, 0);
    Eigen::Matrix<double, 7, 1> robot_target = Eigen::Matrix<double, 7, 1>::Zero();
  };

  struct Task {
    std::string name;
    std::vector<std::string> target_names;
    std::vector<std::string> reference_names;
    std::string body_name;
    std::string base_body_name;
    int body = -1;
    int base_body = -1;
    std::vector<int> joint_qpos;
    std::vector<int> joint_dofs;
    std::vector<bool> rotation_dof_mask;
    double rotation_leak_weight = 0.0;
    Eigen::Vector3d rotation_roll_axis = Eigen::Vector3d::UnitX();
    double rotation_roll_scale = 1.0;
    Eigen::Matrix<double, 6, 1> cost = Eigen::Matrix<double, 6, 1>::Ones();
    double gain = 1.0;
    Eigen::Matrix<double, 7, 1> target = Eigen::Matrix<double, 7, 1>::Zero();
    bool has_target = false;
    Eigen::Vector3d base_pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d straight_tcp_pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d neutral_tcp_pos = Eigen::Vector3d::Zero();
    Eigen::Vector4d neutral_tcp_quat = Eigen::Vector4d(1, 0, 0, 0);
    double reach = 0.0;
    TargetState state;

    bool elbow_enabled = false;
    std::vector<std::string> elbow_source_names;
    std::string elbow_body_name;
    int elbow_body = -1;
    double elbow_weight = 0.0;
    double elbow_gain = 0.0;
    Eigen::Vector3d elbow_target_dir = Eigen::Vector3d::Zero();
    bool has_elbow_target = false;

    bool wrist_forearm_enabled = false;
    std::vector<std::string> wrist_forearm_source_elbow_names;
    std::vector<std::string> wrist_forearm_source_wrist_names;
    std::string wrist_forearm_robot_elbow_body_name;
    std::string wrist_forearm_robot_wrist_body_name;
    std::string wrist_forearm_robot_ee_body_name;
    int wrist_forearm_robot_elbow_body = -1;
    int wrist_forearm_robot_wrist_body = -1;
    int wrist_forearm_robot_ee_body = -1;
    Eigen::Vector3d wrist_forearm_source_hand_axis = Eigen::Vector3d::UnitY();
    Eigen::Vector3d wrist_forearm_robot_hand_axis = Eigen::Vector3d::UnitX();
    double wrist_forearm_weight = 0.0;
    double wrist_forearm_gain = 0.0;
    double wrist_forearm_max_extra_bend_rad = -1.0;
    double wrist_forearm_target_dot = 1.0;
    bool has_wrist_forearm_target = false;
  };

  struct CollisionPair {
    int geom1 = -1;
    int geom2 = -1;
    std::string name1;
    std::string name2;
  };

  struct CollisionConstraint {
    Eigen::VectorXd row;
    double min_step = 0.0;
  };

  ~Impl() {
    if (d_) mj_deleteData(d_);
    if (m_) mj_deleteModel(m_);
  }

  void configure(const RetargetConfig& config, const ScenarioSpec& spec) {
    if (config.backend != KinematicsBackendKind::Mujoco)
      throw std::runtime_error("dual_arm_eepose requires config.backend = Mujoco");
    spec_ = spec;
    char error[1000] = "";
    m_ = mj_loadXML(config.robot_xml.c_str(), nullptr, error, sizeof(error));
    if (!m_) throw std::runtime_error(std::string("mj_loadXML failed: ") + error);
    d_ = mj_makeData(m_);
    if (!d_) throw std::runtime_error("mj_makeData failed for " + config.robot_xml);

    qpos_.resize(m_->nq);
    qpos0_.resize(m_->nq);
    for (int i = 0; i < m_->nq; ++i) {
      qpos_[i] = m_->qpos0[i];
      qpos0_[i] = m_->qpos0[i];
    }
    dt_ = m_->opt.timestep > 0.0 ? m_->opt.timestep : 1.0 / 120.0;
    build_qpos_to_dof_map();
    set_qpos(qpos_);

    json cfg;
    {
      std::ifstream f(config.ik_config_json);
      if (!f) throw std::runtime_error("cannot open dual-arm eepose config: " +
                                       config.ik_config_json);
      f >> cfg;
    }

    damping_ = cfg.value("damping", config.damping);
    if (!(damping_ > 0.0)) damping_ = 1e-4;
    max_iterations_ = cfg.value("max_iterations", 8);
    config_gain_ = cfg.value("configuration_limit_gain", 0.95);
    posture_weight_ = cfg.value("posture_weight", 0.02);
    max_iteration_update_ = cfg.value("max_iteration_update", 0.0);
    max_frame_update_ = cfg.value("max_frame_update", 0.0);
    load_source_config(config, cfg);

    active_dof_.assign(m_->nv, false);
    load_tasks(cfg);
    capture_task_robot_references(/*neutral=*/false);
    load_posture_reference(cfg);
    set_qpos(qpos_);
    capture_task_robot_references(/*neutral=*/true);
    load_collision(cfg);

    locked_qpos_count_ = spec.locked_qpos_prefix;
    if (locked_qpos_count_ > 0) locked_qpos_ = qpos_.head(locked_qpos_count_);
    clamp_qpos_indices_ = spec.clamp_qpos_indices;
    clamp_qpos_values_.resize(static_cast<int>(clamp_qpos_indices_.size()));
    for (size_t i = 0; i < clamp_qpos_indices_.size(); ++i) {
      int qi = clamp_qpos_indices_[i];
      if (qi < 0 || qi >= m_->nq) throw std::runtime_error("clamp qpos index out of range");
      clamp_qpos_values_[static_cast<int>(i)] = qpos_[qi];
    }
  }

  int nq() const { return m_ ? m_->nq : 0; }
  int nv() const { return m_ ? m_->nv : 0; }

  void begin_frame() {
    if (!m_) return;
    Eigen::VectorXd q = qpos_;
    freeze_locked_region(q);
    set_qpos(q);
  }

  Eigen::VectorXd solve(const SkeletonFrame& frame) {
    update_targets(frame);
    bool any_target = false;
    for (const auto& task : tasks_) any_target = any_target || task.has_target;
    if (!any_target) return qpos_;

    solve_start_qpos_ = qpos_;
    const int nv = m_->nv;
    for (int iter = 0; iter < max_iterations_; ++iter) {
      Eigen::MatrixXd H = damping_ * Eigen::MatrixXd::Identity(nv, nv);
      Eigen::VectorXd c = Eigen::VectorXd::Zero(nv);

      for (const auto& task : tasks_) {
        if (task.has_target) add_pose_task(task, H, c);
        if (task.has_elbow_target) add_elbow_direction_task(task, H, c);
        if (task.has_wrist_forearm_target) add_wrist_forearm_task(task, H, c);
      }
      add_posture_task(H, c);

      std::vector<CollisionConstraint> collision_constraints;
      add_collision_tasks(H, c, collision_constraints);

      Eigen::VectorXd lb, ub;
      compute_box_bounds(lb, ub);
      Eigen::VectorXd dq = box_qp::solve(H, c, lb, ub);
      project_collision_constraints(dq, collision_constraints, lb, ub);
      integrate(dq / dt_);

      if (dq.lpNorm<Eigen::Infinity>() < 1e-6) break;
    }
    return qpos_;
  }

  void end_frame(Eigen::VectorXd& qpos) {
    freeze_locked_region(qpos);
    set_qpos(qpos);
  }

  void set_configuration(const Eigen::VectorXd& qpos) {
    if (!m_) return;
    if (qpos.size() != m_->nq)
      throw std::runtime_error("dual_arm_eepose qpos size mismatch");
    set_qpos(qpos);
    if (locked_qpos_count_ > 0 && locked_qpos_count_ <= static_cast<int>(qpos_.size()))
      locked_qpos_ = qpos_.head(locked_qpos_count_);
    for (size_t i = 0; i < clamp_qpos_indices_.size(); ++i) {
      const int qi = clamp_qpos_indices_[i];
      if (qi >= 0 && qi < qpos_.size())
        clamp_qpos_values_[static_cast<int>(i)] = qpos_[qi];
    }
  }

  SkeletonFrame last_targets() const { return last_targets_; }

 private:
  int body_id(const std::string& name) const {
    int id = mj_name2id(m_, mjOBJ_BODY, name.c_str());
    if (id < 0) throw std::runtime_error("unknown body in dual_arm_eepose config: " + name);
    return id;
  }

  int geom_id(const std::string& name) const {
    int id = mj_name2id(m_, mjOBJ_GEOM, name.c_str());
    if (id < 0) throw std::runtime_error("unknown geom in dual_arm_eepose config: " + name);
    return id;
  }

  static SourceMode parse_source_mode(const std::string& mode) {
    if (mode == "robot_world_absolute" || mode == "robot_world" || mode == "absolute")
      return SourceMode::RobotWorldAbsolute;
    if (mode == "shoulder_delta") return SourceMode::ShoulderDelta;
    if (mode == "frame_delta") return SourceMode::FrameDelta;
    if (mode == "se3_delta") return SourceMode::Se3Delta;
    if (mode == "shoulder_absolute") return SourceMode::ShoulderAbsolute;
    throw std::runtime_error("unknown dual_arm_eepose source mode: " + mode);
  }

  static OrientationMode parse_orientation_mode(const std::string& mode) {
    if (mode == "neutral") return OrientationMode::Neutral;
    if (mode == "relative_wrist") return OrientationMode::RelativeWrist;
    if (mode == "relative_wrist_roll" || mode == "wrist_roll")
      return OrientationMode::RelativeWristRoll;
    throw std::runtime_error("unknown dual_arm_eepose orientation mode: " + mode);
  }

  void load_source_config(const RetargetConfig& config, const json& cfg) {
    const json source = cfg.contains("source") && cfg["source"].is_object()
        ? cfg["source"]
        : json::object();
    const json filter = source.contains("filter") && source["filter"].is_object()
        ? source["filter"]
        : json::object();

    const std::string mode = option_string(
        config, "source_mode",
        source.value("mode", cfg.value("source_mode", "robot_world_absolute")));
    const std::string orientation = option_string(
        config, "orientation_mode",
        source.value("orientation_mode", cfg.value("orientation_mode", "neutral")));
    source_mode_ = parse_source_mode(mode);
    orientation_mode_ = parse_orientation_mode(orientation);
    workspace_scale_ = option_double(
        config, "workspace_scale",
        source.value("workspace_scale", cfg.value("workspace_scale", 1.0)));
    arm_reach_scale_ = option_double(
        config, "arm_reach_scale",
        source.value("arm_reach_scale", cfg.value("arm_reach_scale", 0.82)));
    source_fps_ = option_double(
        config, "source_fps",
        source.value("fps", cfg.value("source_fps", 30.0)));
    if (!(source_fps_ > 0.0) || !std::isfinite(source_fps_)) source_fps_ = 30.0;

    const double min_cutoff = option_double(
        config, "source_filter_min_cutoff",
        filter.value("min_cutoff", cfg.value("source_filter_min_cutoff", 0.0)));
    const double beta = option_double(
        config, "source_filter_beta",
        filter.value("beta", cfg.value("source_filter_beta", 0.0)));
    const double d_cutoff = option_double(
        config, "source_filter_d_cutoff",
        filter.value("d_cutoff", cfg.value("source_filter_d_cutoff", 1.0)));
    source_filter_ = utils::OneEuroPoseFilter(min_cutoff, beta, d_cutoff);
    source_time_s_ = 0.0;
  }

  void build_qpos_to_dof_map() {
    qpos_to_dof_.assign(m_->nq, -1);
    for (int j = 0; j < m_->njnt; ++j) {
      const int qadr = m_->jnt_qposadr[j];
      const int dadr = m_->jnt_dofadr[j];
      const int type = m_->jnt_type[j];
      if (type == mjJNT_FREE) {
        for (int k = 0; k < 3 && qadr + k < m_->nq; ++k) qpos_to_dof_[qadr + k] = dadr + k;
        for (int k = 0; k < 4 && qadr + 3 + k < m_->nq; ++k) qpos_to_dof_[qadr + 3 + k] = dadr + 3;
      } else if (type == mjJNT_HINGE || type == mjJNT_SLIDE) {
        qpos_to_dof_[qadr] = dadr;
      }
    }
  }

  void load_tasks(const json& cfg) {
    if (!cfg.contains("tasks") || !cfg["tasks"].is_array())
      throw std::runtime_error("dual_arm_eepose config requires a tasks array");

    const double default_pos_weight = cfg.value("position_weight", 60.0);
    const double default_rot_weight = cfg.value("rotation_weight", 5.0);
    for (const auto& entry : cfg["tasks"]) {
      Task task;
      task.name = entry.value("name", "");
      task.body_name = entry.at("body").get<std::string>();
      task.body = body_id(task.body_name);
      if (entry.contains("base_body")) {
        task.base_body_name = entry.at("base_body").get<std::string>();
      } else if (task.name.find("right") != std::string::npos) {
        task.base_body_name = "right_arm_base_link";
      } else {
        task.base_body_name = "left_arm_base_link";
      }
      task.base_body = body_id(task.base_body_name);
      task.target_names.push_back(entry.at("target").get<std::string>());
      if (entry.contains("target_aliases")) {
        auto aliases = string_array(entry["target_aliases"]);
        task.target_names.insert(task.target_names.end(), aliases.begin(), aliases.end());
      }
      if (entry.contains("reference_target")) {
        task.reference_names.push_back(entry.at("reference_target").get<std::string>());
      } else if (task.name.find("right") != std::string::npos) {
        task.reference_names.push_back("RightShoulder");
      } else {
        task.reference_names.push_back("LeftShoulder");
      }
      if (entry.contains("reference_aliases")) {
        auto aliases = string_array(entry["reference_aliases"]);
        task.reference_names.insert(task.reference_names.end(), aliases.begin(), aliases.end());
      }
      if (entry.contains("elbow_constraint")) {
        const auto& elbow = entry["elbow_constraint"];
        task.elbow_enabled = elbow.value("enabled", false);
        if (task.elbow_enabled) {
          if (!elbow.contains("source_target") || !elbow.contains("robot_body")) {
            throw std::runtime_error(
                "enabled elbow_constraint requires source_target and robot_body for task " +
                task.name);
          }
          task.elbow_source_names.push_back(elbow.at("source_target").get<std::string>());
          if (elbow.contains("source_aliases")) {
            auto aliases = string_array(elbow["source_aliases"]);
            task.elbow_source_names.insert(task.elbow_source_names.end(),
                                           aliases.begin(), aliases.end());
          }
          task.elbow_body_name = elbow.at("robot_body").get<std::string>();
          task.elbow_body = body_id(task.elbow_body_name);
          task.elbow_weight = elbow.value("weight", 8.0);
          task.elbow_gain = elbow.value("gain", 0.5);
          if (!(task.elbow_weight > 0.0) || !std::isfinite(task.elbow_weight))
            throw std::runtime_error("elbow_constraint weight must be positive for task " +
                                     task.name);
          if (!(task.elbow_gain > 0.0) || !std::isfinite(task.elbow_gain))
            throw std::runtime_error("elbow_constraint gain must be positive for task " +
                                     task.name);
        }
      }
      if (entry.contains("wrist_forearm_constraint")) {
        const auto& wrist = entry["wrist_forearm_constraint"];
        task.wrist_forearm_enabled = wrist.value("enabled", false);
        if (task.wrist_forearm_enabled) {
          if (!wrist.contains("source_elbow") || !wrist.contains("source_wrist") ||
              !wrist.contains("robot_elbow_body") ||
              !wrist.contains("robot_wrist_body") ||
              !wrist.contains("robot_ee_body")) {
            throw std::runtime_error(
                "enabled wrist_forearm_constraint requires source_elbow, source_wrist, "
                "robot_elbow_body, robot_wrist_body, and robot_ee_body for task " +
                task.name);
          }
          task.wrist_forearm_source_elbow_names.push_back(
              wrist.at("source_elbow").get<std::string>());
          task.wrist_forearm_source_wrist_names.push_back(
              wrist.at("source_wrist").get<std::string>());
          if (wrist.contains("source_elbow_aliases")) {
            auto aliases = string_array(wrist["source_elbow_aliases"]);
            task.wrist_forearm_source_elbow_names.insert(
                task.wrist_forearm_source_elbow_names.end(), aliases.begin(), aliases.end());
          }
          if (wrist.contains("source_wrist_aliases")) {
            auto aliases = string_array(wrist["source_wrist_aliases"]);
            task.wrist_forearm_source_wrist_names.insert(
                task.wrist_forearm_source_wrist_names.end(), aliases.begin(), aliases.end());
          }

          task.wrist_forearm_robot_elbow_body_name =
              wrist.at("robot_elbow_body").get<std::string>();
          task.wrist_forearm_robot_wrist_body_name =
              wrist.at("robot_wrist_body").get<std::string>();
          task.wrist_forearm_robot_ee_body_name =
              wrist.at("robot_ee_body").get<std::string>();
          task.wrist_forearm_robot_elbow_body =
              body_id(task.wrist_forearm_robot_elbow_body_name);
          task.wrist_forearm_robot_wrist_body =
              body_id(task.wrist_forearm_robot_wrist_body_name);
          task.wrist_forearm_robot_ee_body =
              body_id(task.wrist_forearm_robot_ee_body_name);

          if (wrist.contains("source_hand_axis")) {
            task.wrist_forearm_source_hand_axis = parse_axis(
                wrist["source_hand_axis"],
                "wrist_forearm_constraint.source_hand_axis");
          }
          if (wrist.contains("robot_hand_axis")) {
            task.wrist_forearm_robot_hand_axis = parse_axis(
                wrist["robot_hand_axis"],
                "wrist_forearm_constraint.robot_hand_axis");
          }
          task.wrist_forearm_weight = wrist.value("weight", 6.0);
          task.wrist_forearm_gain = wrist.value("gain", 0.5);
          if (!(task.wrist_forearm_weight > 0.0) ||
              !std::isfinite(task.wrist_forearm_weight))
            throw std::runtime_error(
                "wrist_forearm_constraint weight must be positive for task " +
                task.name);
          if (!(task.wrist_forearm_gain > 0.0) ||
              !std::isfinite(task.wrist_forearm_gain))
            throw std::runtime_error(
                "wrist_forearm_constraint gain must be positive for task " +
                task.name);
          if (wrist.contains("max_extra_bend_deg")) {
            const double deg = wrist.at("max_extra_bend_deg").get<double>();
            if (!(deg >= 0.0) || !std::isfinite(deg))
              throw std::runtime_error(
                  "wrist_forearm_constraint max_extra_bend_deg must be non-negative for task " +
                  task.name);
            task.wrist_forearm_max_extra_bend_rad = deg * kPi / 180.0;
          }
        }
      }
      const double pos_w = entry.value("position_weight", default_pos_weight);
      const double rot_w = entry.value("rotation_weight", default_rot_weight);
      task.cost << pos_w, pos_w, pos_w, rot_w, rot_w, rot_w;
      task.gain = entry.value("gain", 1.0);

      for (const std::string& joint_name : string_array(entry.at("joint_names"))) {
        int jid = mj_name2id(m_, mjOBJ_JOINT, joint_name.c_str());
        if (jid < 0) throw std::runtime_error("unknown joint in dual_arm_eepose config: " +
                                              joint_name);
        if (m_->jnt_type[jid] != mjJNT_HINGE && m_->jnt_type[jid] != mjJNT_SLIDE)
          throw std::runtime_error("dual_arm_eepose only supports 1-DoF arm joints: " +
                                   joint_name);
        int qadr = m_->jnt_qposadr[jid];
        int dadr = m_->jnt_dofadr[jid];
        task.joint_qpos.push_back(qadr);
        task.joint_dofs.push_back(dadr);
        active_dof_[dadr] = true;
      }
      task.rotation_dof_mask.assign(m_->nv, false);
      if (entry.contains("rotation_joint_names")) {
        const auto rotation_joint_names = string_array(entry["rotation_joint_names"]);
        if (rotation_joint_names.empty())
          throw std::runtime_error("rotation_joint_names must not be empty for task " +
                                   task.name);
        for (const std::string& joint_name : rotation_joint_names) {
          int jid = mj_name2id(m_, mjOBJ_JOINT, joint_name.c_str());
          if (jid < 0)
            throw std::runtime_error("unknown rotation joint in dual_arm_eepose config: " +
                                     joint_name);
          if (m_->jnt_type[jid] != mjJNT_HINGE && m_->jnt_type[jid] != mjJNT_SLIDE)
            throw std::runtime_error(
                "dual_arm_eepose rotation_joint_names only supports 1-DoF joints: " +
                joint_name);
          const int dadr = m_->jnt_dofadr[jid];
          if (std::find(task.joint_dofs.begin(), task.joint_dofs.end(), dadr) ==
              task.joint_dofs.end()) {
            throw std::runtime_error("rotation joint is not part of task joint_names: " +
                                     joint_name);
          }
          task.rotation_dof_mask[dadr] = true;
        }
      } else {
        for (int dadr : task.joint_dofs) task.rotation_dof_mask[dadr] = true;
      }
      task.rotation_leak_weight = entry.value("rotation_leak_weight", 0.0);
      if (!(task.rotation_leak_weight >= 0.0 && task.rotation_leak_weight <= 1.0) ||
          !std::isfinite(task.rotation_leak_weight)) {
        throw std::runtime_error(
            "rotation_leak_weight must be finite in [0, 1] for task " + task.name);
      }
      if (entry.contains("rotation_roll_axis")) {
        task.rotation_roll_axis =
            parse_axis(entry["rotation_roll_axis"], "rotation_roll_axis");
      }
      task.rotation_roll_scale = entry.value("rotation_roll_scale", 1.0);
      if (!std::isfinite(task.rotation_roll_scale)) {
        throw std::runtime_error("rotation_roll_scale must be finite for task " + task.name);
      }
      tasks_.push_back(std::move(task));
    }
  }

  void apply_joint_positions(const json& positions, bool update_posture_ref,
                             bool update_initial_qpos) {
    if (!positions.is_object())
      throw std::runtime_error("joint position block must be a JSON object");
    for (auto& [joint_name, value] : positions.items()) {
      int jid = mj_name2id(m_, mjOBJ_JOINT, joint_name.c_str());
      if (jid < 0) throw std::runtime_error("unknown joint in joint position block: " +
                                            joint_name);
      if (m_->jnt_type[jid] != mjJNT_HINGE && m_->jnt_type[jid] != mjJNT_SLIDE)
        throw std::runtime_error("joint position block only supports 1-DoF joints: " +
                                 joint_name);
      const double q = value.get<double>();
      if (!std::isfinite(q))
        throw std::runtime_error("non-finite joint position for " + joint_name);
      if (m_->jnt_limited[jid]) {
        const double lo = m_->jnt_range[2 * jid + 0];
        const double hi = m_->jnt_range[2 * jid + 1];
        if (q < lo - 1e-9 || q > hi + 1e-9)
          throw std::runtime_error("joint position outside limits for " + joint_name);
      }
      const int qadr = m_->jnt_qposadr[jid];
      if (update_posture_ref) qpos0_[qadr] = q;
      if (update_initial_qpos) qpos_[qadr] = q;
    }
  }

  void load_posture_reference(const json& cfg) {
    if (cfg.contains("posture_reference")) {
      has_neutral_reference_ = true;
      apply_joint_positions(cfg["posture_reference"], true, true);
    }
    if (cfg.contains("initial_joint_positions")) {
      has_neutral_reference_ = true;
      apply_joint_positions(cfg["initial_joint_positions"], false, true);
    }
  }

  void capture_task_robot_references(bool neutral) {
    for (auto& task : tasks_) {
      Eigen::Matrix<double, 7, 1> tcp;
      frame_pose(task.body, tcp);
      if (neutral) {
        task.neutral_tcp_quat = tcp.head<4>();
        task.neutral_tcp_pos = tcp.tail<3>();
        continue;
      }

      Eigen::Matrix<double, 7, 1> base;
      frame_pose(task.base_body, base);
      task.base_pos = base.tail<3>();
      task.straight_tcp_pos = tcp.tail<3>();
      task.reach = (task.straight_tcp_pos - task.base_pos).norm();
    }
  }

  std::vector<int> resolve_geom_group(const json& group) const {
    std::vector<int> out;
    for (const auto& name_json : group) out.push_back(geom_id(name_json.get<std::string>()));
    return out;
  }

  void load_collision(const json& cfg) {
    if (!cfg.contains("collision")) return;
    const auto& coll = cfg["collision"];
    collision_enabled_ = coll.value("enabled", true);
    if (!collision_enabled_) return;
    collision_min_distance_ = coll.value("min_distance", 0.06);
    collision_activation_distance_ =
        std::max(coll.value("activation_distance", collision_min_distance_),
                 collision_min_distance_);
    collision_gain_ = coll.value("gain", 0.75);
    collision_weight_ = coll.value("weight", 100.0);
    collision_projection_iterations_ = coll.value("projection_iterations", 8);

    std::set<std::pair<int, int>> seen;
    auto add_pair = [&](int a, int b) {
      if (a == b) return;
      auto key = std::minmax(a, b);
      if (!seen.insert(key).second) return;
      const char* n1 = mj_id2name(m_, mjOBJ_GEOM, a);
      const char* n2 = mj_id2name(m_, mjOBJ_GEOM, b);
      collision_pairs_.push_back({a, b, n1 ? n1 : "", n2 ? n2 : ""});
    };

    std::map<std::string, std::vector<int>> groups;
    if (coll.contains("geom_groups")) {
      for (auto& [name, group] : coll["geom_groups"].items())
        groups[name] = resolve_geom_group(group);
    }
    if (coll.contains("avoid_group_pairs")) {
      for (const auto& pair : coll["avoid_group_pairs"]) {
        const std::string a = pair.at(0).get<std::string>();
        const std::string b = pair.at(1).get<std::string>();
        auto ia = groups.find(a);
        auto ib = groups.find(b);
        if (ia == groups.end() || ib == groups.end())
          throw std::runtime_error("unknown collision geom group pair: " + a + ", " + b);
        for (int ga : ia->second)
          for (int gb : ib->second) add_pair(ga, gb);
      }
    }
    if (coll.contains("pairs")) {
      for (const auto& pair : coll["pairs"])
        add_pair(geom_id(pair.at(0).get<std::string>()), geom_id(pair.at(1).get<std::string>()));
    }
  }

  void set_task_target(Task& task, const Eigen::Vector3d& pos,
                       const Eigen::Vector4d& quat) {
    const Eigen::Vector4d q = normalized_quat(quat);
    task.target << q[0], q[1], q[2], q[3], pos[0], pos[1], pos[2];
    task.has_target = true;

    if (!task.target_names.empty()) {
      Pose out;
      out.pos = pos;
      out.quat = q;
      last_targets_[task.target_names.front()] = out;
    }
  }

  Eigen::Vector3d clamp_reach(const Task& task, const Eigen::Vector3d& target) const {
    Eigen::Vector3d v = target - task.base_pos;
    double n = v.norm();
    const double max_reach = task.reach * 0.995;
    if (!(max_reach > 0.0) || !(n > max_reach)) return target;
    return task.base_pos + v * (max_reach / n);
  }

  bool find_side_reference(const SkeletonFrame& frame, const char* side, Pose& out) const {
    for (const auto& task : tasks_) {
      if (task.name.find(side) == std::string::npos) continue;
      if (find_pose(frame, task.reference_names, out)) return true;
    }
    return false;
  }

  bool make_shoulder_frame(const SkeletonFrame& frame, Eigen::Matrix3d& R_world_body,
                           Eigen::Vector4d& q_world_body) const {
    Pose left_shoulder;
    Pose right_shoulder;
    if (!find_side_reference(frame, "left", left_shoulder) ||
        !find_side_reference(frame, "right", right_shoulder)) {
      return false;
    }

    Eigen::Vector3d y_axis = left_shoulder.pos - right_shoulder.pos;
    y_axis.z() = 0.0;
    if (y_axis.norm() < 1e-5) return false;
    y_axis.normalize();

    const Eigen::Vector3d z_axis = Eigen::Vector3d::UnitZ();
    Eigen::Vector3d x_axis = y_axis.cross(z_axis);
    if (x_axis.norm() < 1e-5) return false;
    x_axis.normalize();
    y_axis = z_axis.cross(x_axis).normalized();

    R_world_body.col(0) = x_axis;
    R_world_body.col(1) = y_axis;
    R_world_body.col(2) = z_axis;
    q_world_body = quat_from_rotation(R_world_body);
    return true;
  }

  SourcePose to_body_relative_pose(const Pose& origin, const Pose& child,
                                   const Eigen::Matrix3d& R_world_body,
                                   const Eigen::Vector4d& q_world_body) const {
    SourcePose out;
    out.pos = R_world_body.transpose() * (child.pos - origin.pos);
    out.quat = normalized_quat(qmul(qconj(q_world_body), normalized_quat(child.quat)));
    return out;
  }

  void initialize_source_target(Task& task, const SourcePose& source) {
    TargetState& state = task.state;
    state.initialized = true;
    state.prev_source_pos = source.pos;
    state.prev_source_quat = source.quat;
    state.initial_source_quat = source.quat;

    Eigen::Vector3d pos;
    if (source_mode_ == SourceMode::ShoulderAbsolute) {
      pos = task.base_pos + workspace_scale_ * source.pos;
    } else if (has_neutral_reference_) {
      pos = task.neutral_tcp_pos;
    } else {
      pos = task.base_pos + arm_reach_scale_ * (task.straight_tcp_pos - task.base_pos);
    }
    pos = clamp_reach(task, pos);

    state.robot_target << task.neutral_tcp_quat[0], task.neutral_tcp_quat[1],
        task.neutral_tcp_quat[2], task.neutral_tcp_quat[3], pos.x(), pos.y(), pos.z();
    set_task_target(task, pos, task.neutral_tcp_quat);
  }

  void update_source_target(Task& task, const SourcePose& source) {
    TargetState& state = task.state;
    if (!state.initialized) {
      initialize_source_target(task, source);
      return;
    }

    Eigen::Vector3d pos = state.robot_target.tail<3>();
    Eigen::Vector4d quat = state.robot_target.head<4>();
    if (source_mode_ == SourceMode::ShoulderAbsolute) {
      pos = task.base_pos + workspace_scale_ * source.pos;
      if (orientation_mode_ == OrientationMode::RelativeWrist) {
        quat = normalized_quat(qmul(
            task.neutral_tcp_quat, qmul(qconj(state.initial_source_quat), source.quat)));
      } else if (orientation_mode_ == OrientationMode::RelativeWristRoll &&
                 source.has_roll_axis) {
        const Eigen::Vector4d delta_body =
            normalized_quat(qmul(source.quat, qconj(state.initial_source_quat)));
        const double roll =
            task.rotation_roll_scale * signed_twist_angle(delta_body, source.roll_axis);
        quat = normalized_quat(
            qmul(task.neutral_tcp_quat, axis_angle_quat(task.rotation_roll_axis, roll)));
      } else {
        quat = task.neutral_tcp_quat;
      }
    } else {
      const Eigen::Vector4d delta_quat =
          normalized_quat(qmul(qconj(state.prev_source_quat), source.quat));
      const Eigen::Vector4d delta_body_quat =
          normalized_quat(qmul(source.quat, qconj(state.prev_source_quat)));
      Eigen::Vector3d delta_pos = source.pos - state.prev_source_pos;
      if (source_mode_ == SourceMode::Se3Delta) {
        delta_pos = qrotate(qconj(state.prev_source_quat), delta_pos);
        pos += qrotate(quat, workspace_scale_ * delta_pos);
      } else {
        pos += workspace_scale_ * delta_pos;
      }

      if (orientation_mode_ == OrientationMode::RelativeWrist) {
        quat = normalized_quat(qmul(quat, delta_quat));
      } else if (orientation_mode_ == OrientationMode::RelativeWristRoll &&
                 source.has_roll_axis) {
        const double roll =
            task.rotation_roll_scale * signed_twist_angle(delta_body_quat, source.roll_axis);
        quat = normalized_quat(qmul(quat, axis_angle_quat(task.rotation_roll_axis, roll)));
      } else {
        quat = task.neutral_tcp_quat;
      }
    }

    pos = clamp_reach(task, pos);
    state.robot_target << quat[0], quat[1], quat[2], quat[3], pos.x(), pos.y(), pos.z();
    state.prev_source_pos = source.pos;
    state.prev_source_quat = source.quat;
    set_task_target(task, pos, quat);
  }

  void update_absolute_targets(const SkeletonFrame& frame) {
    for (auto& task : tasks_) {
      task.has_target = false;
      task.has_elbow_target = false;
      task.has_wrist_forearm_target = false;
      Pose pose;
      if (find_pose(frame, task.target_names, pose)) {
        set_task_target(task, pose.pos, pose.quat);
      }
    }
  }

  void update_source_targets(const SkeletonFrame& frame) {
    Eigen::Matrix3d R_world_body;
    Eigen::Vector4d q_world_body;
    const bool any_elbow_enabled = std::any_of(
        tasks_.begin(), tasks_.end(), [](const Task& task) { return task.elbow_enabled; });
    const bool any_wrist_forearm_enabled = std::any_of(
        tasks_.begin(), tasks_.end(),
        [](const Task& task) { return task.wrist_forearm_enabled; });
    const bool need_shoulder_frame =
        source_mode_ != SourceMode::FrameDelta || any_elbow_enabled ||
        any_wrist_forearm_enabled;
    if (need_shoulder_frame &&
        !make_shoulder_frame(frame, R_world_body, q_world_body)) {
      for (auto& task : tasks_) {
        task.has_target = false;
        task.has_elbow_target = false;
        task.has_wrist_forearm_target = false;
      }
      return;
    }

    std::vector<utils::NamedPose> filter_inputs;
    std::vector<SourcePose> source_inputs;
    std::vector<size_t> filter_task_indices;
    filter_inputs.reserve(tasks_.size());
    source_inputs.reserve(tasks_.size());
    filter_task_indices.reserve(tasks_.size());
    for (size_t i = 0; i < tasks_.size(); ++i) {
      Task& task = tasks_[i];
      task.has_target = false;
      task.has_elbow_target = false;
      task.has_wrist_forearm_target = false;

      Pose target_pose;
      if (!find_pose(frame, task.target_names, target_pose)) continue;

      Pose reference_pose;
      bool has_reference = false;
      if (need_shoulder_frame || source_mode_ != SourceMode::FrameDelta) {
        has_reference = find_pose(frame, task.reference_names, reference_pose);
        if (!has_reference && source_mode_ != SourceMode::FrameDelta) continue;
      }

      if (task.elbow_enabled && has_reference) {
        Pose elbow_pose;
        if (find_pose(frame, task.elbow_source_names, elbow_pose)) {
          Eigen::Vector3d dir =
              R_world_body.transpose() * (elbow_pose.pos - reference_pose.pos);
          const double norm = dir.norm();
          if (norm > 1e-6 && std::isfinite(norm)) {
            task.elbow_target_dir = dir / norm;
            task.has_elbow_target = true;
          }
        }
      }

      if (task.wrist_forearm_enabled) {
        Pose source_elbow_pose;
        Pose source_wrist_pose;
        if (find_pose(frame, task.wrist_forearm_source_elbow_names, source_elbow_pose) &&
            find_pose(frame, task.wrist_forearm_source_wrist_names, source_wrist_pose)) {
          Eigen::Vector3d forearm =
              source_wrist_pose.pos - source_elbow_pose.pos;
          const double forearm_norm = forearm.norm();
          Eigen::Vector3d hand_axis = qrotate(
              normalized_quat(source_wrist_pose.quat),
              task.wrist_forearm_source_hand_axis);
          const double hand_axis_norm = hand_axis.norm();
          if (forearm_norm > 1e-6 && std::isfinite(forearm_norm) &&
              hand_axis_norm > 1e-6 && std::isfinite(hand_axis_norm)) {
            forearm /= forearm_norm;
            hand_axis /= hand_axis_norm;
            task.wrist_forearm_target_dot = clamp_unit(forearm.dot(hand_axis));
            task.has_wrist_forearm_target = true;
          }
        }
      }

      SourcePose source;
      if (source_mode_ == SourceMode::FrameDelta) {
        source.pos = target_pose.pos;
        source.quat = normalized_quat(target_pose.quat);
      } else {
        source = to_body_relative_pose(reference_pose, target_pose, R_world_body, q_world_body);
      }

      if (orientation_mode_ == OrientationMode::RelativeWristRoll) {
        std::vector<std::string> roll_elbow_names = task.elbow_source_names;
        if (roll_elbow_names.empty()) {
          if (task.name.find("right") != std::string::npos) {
            roll_elbow_names.push_back("RightArmLower");
          } else {
            roll_elbow_names.push_back("LeftArmLower");
          }
        }
        Pose roll_elbow_pose;
        if (find_pose(frame, roll_elbow_names, roll_elbow_pose)) {
          Eigen::Vector3d axis = target_pose.pos - roll_elbow_pose.pos;
          if (source_mode_ != SourceMode::FrameDelta) axis = R_world_body.transpose() * axis;
          const double axis_norm = axis.norm();
          if (axis_norm > 1e-6 && std::isfinite(axis_norm)) {
            source.roll_axis = axis / axis_norm;
            source.has_roll_axis = true;
          }
        }
      }

      utils::NamedPose named;
      named.name = task.target_names.empty() ? task.name : task.target_names.front();
      named.pos = source.pos;
      named.quat = source.quat;
      filter_inputs.push_back(std::move(named));
      source_inputs.push_back(source);
      filter_task_indices.push_back(i);
    }

    if (filter_inputs.empty()) return;
    source_filter_.filter(source_time_s_, filter_inputs);
    source_time_s_ += 1.0 / source_fps_;

    for (size_t i = 0; i < filter_inputs.size(); ++i) {
      SourcePose source = source_inputs[i];
      source.pos = filter_inputs[i].pos;
      source.quat = normalized_quat(filter_inputs[i].quat);
      update_source_target(tasks_[filter_task_indices[i]], source);
    }
  }

  void update_targets(const SkeletonFrame& frame) {
    last_targets_.clear();
    if (source_mode_ == SourceMode::RobotWorldAbsolute) {
      update_absolute_targets(frame);
    } else {
      update_source_targets(frame);
    }
  }

  void set_qpos(const Eigen::VectorXd& qpos) {
    qpos_ = qpos;
    for (int i = 0; i < m_->nq; ++i) d_->qpos[i] = qpos_[i];
    mj_kinematics(m_, d_);
    mj_comPos(m_, d_);
  }

  void frame_pose(int body, Eigen::Matrix<double, 7, 1>& out7) const {
    double frame7[7];
    lie::xmat_xpos_to_wxyz_xyz(d_->xmat + 9 * body, d_->xpos + 3 * body, frame7);
    for (int i = 0; i < 7; ++i) out7[i] = frame7[i];
  }

  void frame_jacobian(int body, Eigen::MatrixXd& jac6) const {
    const int nv = m_->nv;
    std::vector<double> jacp(3 * nv), jacr(3 * nv);
    mj_jacBody(m_, d_, jacp.data(), jacr.data(), body);
    Eigen::MatrixXd jac(6, nv);
    for (int c = 0; c < nv; ++c) {
      for (int r = 0; r < 3; ++r) jac(r, c) = jacp[r * nv + c];
      for (int r = 0; r < 3; ++r) jac(3 + r, c) = jacr[r * nv + c];
    }
    double adj[36];
    lie::se3_rotation_adjoint_from_xmat(d_->xmat + 9 * body, adj);
    Eigen::Matrix<double, 6, 6> A;
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) A(i, j) = adj[6 * i + j];
    jac6 = A * jac;
  }

  void add_pose_task(const Task& task, Eigen::MatrixXd& H, Eigen::VectorXd& c) const {
    Eigen::Matrix<double, 7, 1> current;
    frame_pose(task.body, current);
    const int nv = m_->nv;

    std::vector<double> jacp(3 * nv), jacr(3 * nv);
    mj_jacBody(m_, d_, jacp.data(), jacr.data(), task.body);
    Eigen::MatrixXd pos_jacobian = Eigen::MatrixXd::Zero(3, nv);
    for (int col = 0; col < nv; ++col) {
      for (int row = 0; row < 3; ++row)
        pos_jacobian(row, col) = jacp[row * nv + col];
    }

    const Eigen::Vector3d pos_error = task.target.tail<3>() - current.tail<3>();
    const double pos_weight = task.cost[0];
    if (pos_weight > 0.0) {
      Eigen::MatrixXd weighted_pos_jacobian = pos_weight * pos_jacobian;
      Eigen::Vector3d weighted_pos_error = pos_weight * task.gain * pos_error;
      H.noalias() += weighted_pos_jacobian.transpose() * weighted_pos_jacobian;
      c.noalias() -= weighted_pos_jacobian.transpose() * weighted_pos_error;
    }

    if (!(task.cost.tail<3>().maxCoeff() > 0.0)) return;

    Eigen::Matrix<double, 6, 1> error = lie::se3_rminus(task.target, current);
    double T_tb[7];
    lie::se3_inverse_multiply(task.target.data(), current.data(), T_tb);
    double jlog[36];
    lie::se3_jlog(T_tb, jlog);
    Eigen::Matrix<double, 6, 6> Jlog;
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) Jlog(i, j) = jlog[6 * i + j];

    Eigen::MatrixXd jac6;
    frame_jacobian(task.body, jac6);
    Eigen::MatrixXd jacobian = -(Jlog * jac6);
    Eigen::Matrix<double, 6, 1> weighted_error =
        task.cost.array() * (-task.gain * error).array();
    Eigen::MatrixXd weighted_jacobian = task.cost.asDiagonal() * jacobian;
    weighted_jacobian.topRows<3>().setZero();
    weighted_error.head<3>().setZero();
    if (task.rotation_dof_mask.size() == static_cast<size_t>(weighted_jacobian.cols())) {
      for (int row = 3; row < 6; ++row) {
        for (int col = 0; col < weighted_jacobian.cols(); ++col) {
          if (!task.rotation_dof_mask[col])
            weighted_jacobian(row, col) *= task.rotation_leak_weight;
        }
      }
    }
    H.noalias() += weighted_jacobian.transpose() * weighted_jacobian;
    c.noalias() -= weighted_jacobian.transpose() * weighted_error;
  }

  void add_elbow_direction_task(const Task& task, Eigen::MatrixXd& H,
                                Eigen::VectorXd& c) const {
    if (!task.elbow_enabled || !task.has_elbow_target) return;
    const double* base_xpos = d_->xpos + 3 * task.base_body;
    const double* elbow_xpos = d_->xpos + 3 * task.elbow_body;
    Eigen::Vector3d v(elbow_xpos[0] - base_xpos[0],
                      elbow_xpos[1] - base_xpos[1],
                      elbow_xpos[2] - base_xpos[2]);
    const double len = v.norm();
    if (!(len > 1e-6) || !std::isfinite(len)) return;

    Eigen::Vector3d u_robot = v / len;
    Eigen::Vector3d u_target = task.elbow_target_dir;
    const double target_norm = u_target.norm();
    if (!(target_norm > 1e-6) || !std::isfinite(target_norm)) return;
    u_target /= target_norm;

    const int nv = m_->nv;
    std::vector<double> j_base(3 * nv), jr_base(3 * nv);
    std::vector<double> j_elbow(3 * nv), jr_elbow(3 * nv);
    mj_jacBody(m_, d_, j_base.data(), jr_base.data(), task.base_body);
    mj_jacBody(m_, d_, j_elbow.data(), jr_elbow.data(), task.elbow_body);

    Eigen::MatrixXd j_v = Eigen::MatrixXd::Zero(3, nv);
    for (int col = 0; col < nv; ++col) {
      if (!active_dof_[col]) continue;
      for (int row = 0; row < 3; ++row)
        j_v(row, col) = j_elbow[row * nv + col] - j_base[row * nv + col];
    }

    const Eigen::Matrix3d projector =
        Eigen::Matrix3d::Identity() - u_robot * u_robot.transpose();
    Eigen::MatrixXd j_dir = (projector / len) * j_v;
    if (j_dir.squaredNorm() < 1e-12) return;

    const Eigen::Vector3d error = u_robot - u_target;
    const Eigen::Vector3d desired = -task.elbow_gain * error;
    H.noalias() += task.elbow_weight * (j_dir.transpose() * j_dir);
    c.noalias() -= task.elbow_weight * (j_dir.transpose() * desired);
  }

  void add_wrist_forearm_task(const Task& task, Eigen::MatrixXd& H,
                              Eigen::VectorXd& c) const {
    if (!task.wrist_forearm_enabled || !task.has_wrist_forearm_target) return;

    const double* elbow_xpos = d_->xpos + 3 * task.wrist_forearm_robot_elbow_body;
    const double* wrist_xpos = d_->xpos + 3 * task.wrist_forearm_robot_wrist_body;
    Eigen::Vector3d forearm(wrist_xpos[0] - elbow_xpos[0],
                            wrist_xpos[1] - elbow_xpos[1],
                            wrist_xpos[2] - elbow_xpos[2]);
    const double forearm_len = forearm.norm();
    if (!(forearm_len > 1e-6) || !std::isfinite(forearm_len)) return;
    Eigen::Vector3d u_robot = forearm / forearm_len;

    const double* ee_xmat = d_->xmat + 9 * task.wrist_forearm_robot_ee_body;
    Eigen::Matrix3d R_ee;
    for (int row = 0; row < 3; ++row)
      for (int col = 0; col < 3; ++col) R_ee(row, col) = ee_xmat[3 * row + col];
    Eigen::Vector3d a_robot = R_ee * task.wrist_forearm_robot_hand_axis;
    const double axis_norm = a_robot.norm();
    if (!(axis_norm > 1e-6) || !std::isfinite(axis_norm)) return;
    a_robot /= axis_norm;

    const double source_dot = clamp_unit(task.wrist_forearm_target_dot);
    const double robot_dot = clamp_unit(u_robot.dot(a_robot));
    double target_dot = source_dot;
    if (task.wrist_forearm_max_extra_bend_rad >= 0.0) {
      const double source_angle = std::acos(source_dot);
      const double limit_angle =
          std::min(kPi, source_angle + task.wrist_forearm_max_extra_bend_rad);
      target_dot = std::cos(limit_angle);
    }

    const int nv = m_->nv;
    std::vector<double> j_elbow(3 * nv), jr_elbow(3 * nv);
    std::vector<double> j_wrist(3 * nv), jr_wrist(3 * nv);
    std::vector<double> j_ee(3 * nv), jr_ee(3 * nv);
    mj_jacBody(m_, d_, j_elbow.data(), jr_elbow.data(),
               task.wrist_forearm_robot_elbow_body);
    mj_jacBody(m_, d_, j_wrist.data(), jr_wrist.data(),
               task.wrist_forearm_robot_wrist_body);
    mj_jacBody(m_, d_, j_ee.data(), jr_ee.data(),
               task.wrist_forearm_robot_ee_body);

    Eigen::MatrixXd j_v = Eigen::MatrixXd::Zero(3, nv);
    Eigen::MatrixXd j_rot = Eigen::MatrixXd::Zero(3, nv);
    for (int col = 0; col < nv; ++col) {
      if (!active_dof_[col]) continue;
      for (int row = 0; row < 3; ++row) {
        j_v(row, col) = j_wrist[row * nv + col] - j_elbow[row * nv + col];
        j_rot(row, col) = jr_ee[row * nv + col];
      }
    }

    const Eigen::Matrix3d projector =
        Eigen::Matrix3d::Identity() - u_robot * u_robot.transpose();
    const Eigen::MatrixXd j_u = (projector / forearm_len) * j_v;
    const Eigen::MatrixXd j_a = -skew(a_robot) * j_rot;

    if (task.wrist_forearm_max_extra_bend_rad >= 0.0 && robot_dot >= target_dot) return;
    Eigen::RowVectorXd row = a_robot.transpose() * j_u + u_robot.transpose() * j_a;
    if (row.squaredNorm() < 1e-12) return;

    const double error = robot_dot - target_dot;
    const double desired = -task.wrist_forearm_gain * error;
    H.noalias() += task.wrist_forearm_weight * (row.transpose() * row);
    c.noalias() -= task.wrist_forearm_weight * desired * row.transpose();
  }

  void add_posture_task(Eigen::MatrixXd& H, Eigen::VectorXd& c) const {
    if (!(posture_weight_ > 0.0)) return;
    for (int d = 0; d < m_->nv; ++d) {
      if (!active_dof_[d]) continue;
      int qi = dof_to_qpos(d);
      if (qi < 0) continue;
      H(d, d) += posture_weight_;
      c[d] -= posture_weight_ * (qpos0_[qi] - qpos_[qi]);
    }
  }

  int dof_to_qpos(int dof) const {
    for (int j = 0; j < m_->njnt; ++j) {
      if (m_->jnt_dofadr[j] == dof &&
          (m_->jnt_type[j] == mjJNT_HINGE || m_->jnt_type[j] == mjJNT_SLIDE)) {
        return m_->jnt_qposadr[j];
      }
    }
    return -1;
  }

  void add_collision_tasks(Eigen::MatrixXd& H, Eigen::VectorXd& c,
                           std::vector<CollisionConstraint>& constraints) const {
    if (!collision_enabled_) return;
    const int nv = m_->nv;
    for (const CollisionPair& pair : collision_pairs_) {
      double fromto[6] = {0, 0, 0, 0, 0, 0};
      double dist = mj_geomDistance(m_, d_, pair.geom1, pair.geom2,
                                    collision_activation_distance_, fromto);
      if (!std::isfinite(dist) || dist >= collision_activation_distance_) continue;

      Eigen::Vector3d p1(fromto[0], fromto[1], fromto[2]);
      Eigen::Vector3d p2(fromto[3], fromto[4], fromto[5]);
      Eigen::Vector3d normal = p2 - p1;
      if (normal.norm() < 1e-9) {
        const double* x1 = d_->geom_xpos + 3 * pair.geom1;
        const double* x2 = d_->geom_xpos + 3 * pair.geom2;
        normal = Eigen::Vector3d(x2[0] - x1[0], x2[1] - x1[1], x2[2] - x1[2]);
      }
      if (normal.norm() < 1e-9) normal = Eigen::Vector3d::UnitX();
      normal.normalize();

      std::vector<double> j1(3 * nv), jr1(3 * nv), j2(3 * nv), jr2(3 * nv);
      mj_jacGeom(m_, d_, j1.data(), jr1.data(), pair.geom1);
      mj_jacGeom(m_, d_, j2.data(), jr2.data(), pair.geom2);
      Eigen::VectorXd row = Eigen::VectorXd::Zero(nv);
      for (int d = 0; d < nv; ++d) {
        if (!active_dof_[d]) continue;
        double v = 0.0;
        for (int r = 0; r < 3; ++r) v += normal[r] * (j2[r * nv + d] - j1[r * nv + d]);
        row[d] = v;
      }
      if (row.squaredNorm() < 1e-12) continue;

      const double desired_soft = collision_gain_ * (collision_activation_distance_ - dist);
      if (desired_soft > 0.0 && collision_weight_ > 0.0) {
        H.noalias() += collision_weight_ * (row * row.transpose());
        c.noalias() -= collision_weight_ * desired_soft * row;
      }
      const double desired_hard = collision_gain_ * (collision_min_distance_ - dist);
      if (desired_hard > 0.0) constraints.push_back({std::move(row), desired_hard});
    }
  }

  void project_collision_constraints(Eigen::VectorXd& dq,
                                     const std::vector<CollisionConstraint>& constraints,
                                     const Eigen::VectorXd& lb,
                                     const Eigen::VectorXd& ub) const {
    if (constraints.empty()) return;
    auto clip = [&]() {
      for (int i = 0; i < dq.size(); ++i) {
        if (dq[i] < lb[i]) dq[i] = lb[i];
        if (dq[i] > ub[i]) dq[i] = ub[i];
      }
    };
    clip();
    for (int iter = 0; iter < collision_projection_iterations_; ++iter) {
      double max_violation = 0.0;
      for (const CollisionConstraint& con : constraints) {
        double lhs = con.row.dot(dq);
        double violation = con.min_step - lhs;
        if (violation <= 1e-8) continue;
        double denom = con.row.squaredNorm();
        if (denom <= 1e-12) continue;
        dq.noalias() += (violation / denom) * con.row;
        clip();
        max_violation = std::max(max_violation, violation);
      }
      if (max_violation <= 1e-8) break;
    }
  }

  void compute_box_bounds(Eigen::VectorXd& lb, Eigen::VectorXd& ub) const {
    const int nv = m_->nv;
    const double kInf = std::numeric_limits<double>::infinity();
    lb = Eigen::VectorXd::Constant(nv, -kInf);
    ub = Eigen::VectorXd::Constant(nv, kInf);

    for (int d = 0; d < nv; ++d) {
      if (!active_dof_[d]) {
        lb[d] = 0.0;
        ub[d] = 0.0;
      }
    }

    for (int j = 0; j < m_->njnt; ++j) {
      if (!m_->jnt_limited[j]) continue;
      const int type = m_->jnt_type[j];
      if (type != mjJNT_HINGE && type != mjJNT_SLIDE) continue;
      const int qadr = m_->jnt_qposadr[j];
      const int dadr = m_->jnt_dofadr[j];
      double q = qpos_[qadr];
      double lo = m_->jnt_range[2 * j + 0];
      double hi = m_->jnt_range[2 * j + 1];
      double p_max = config_gain_ * (hi - q);
      double p_min = config_gain_ * (q - lo);
      ub[dadr] = std::min(ub[dadr], p_max);
      lb[dadr] = std::max(lb[dadr], -p_min);
    }

    for (int d = 0; d < nv; ++d) {
      if (!active_dof_[d]) continue;
      if (max_iteration_update_ > 0.0) {
        ub[d] = std::min(ub[d], max_iteration_update_);
        lb[d] = std::max(lb[d], -max_iteration_update_);
      }
      if (max_frame_update_ > 0.0 && solve_start_qpos_.size() == m_->nq) {
        int qi = dof_to_qpos(d);
        if (qi >= 0) {
          ub[d] = std::min(ub[d], solve_start_qpos_[qi] + max_frame_update_ - qpos_[qi]);
          lb[d] = std::max(lb[d], solve_start_qpos_[qi] - max_frame_update_ - qpos_[qi]);
        }
      }
    }

    if (spec_.freeze_locked_in_solve && locked_qpos_count_ > 0) {
      const int locked_nv = locked_qpos_count_ - 1;
      for (int d = 0; d < locked_nv && d < nv; ++d) {
        lb[d] = 0.0;
        ub[d] = 0.0;
      }
    }
    for (int qi : clamp_qpos_indices_) {
      if (qi < 0 || qi >= static_cast<int>(qpos_to_dof_.size())) continue;
      int d = qpos_to_dof_[qi];
      if (d >= 0 && d < nv) {
        lb[d] = 0.0;
        ub[d] = 0.0;
      }
    }
  }

  void integrate(const Eigen::VectorXd& v) {
    std::vector<double> q(m_->nq), vel(m_->nv);
    for (int i = 0; i < m_->nq; ++i) q[i] = qpos_[i];
    for (int i = 0; i < m_->nv; ++i) vel[i] = v[i];
    mj_integratePos(m_, q.data(), vel.data(), dt_);
    Eigen::VectorXd out(m_->nq);
    for (int i = 0; i < m_->nq; ++i) out[i] = q[i];
    freeze_locked_region(out);
    set_qpos(out);
  }

  void freeze_locked_region(Eigen::VectorXd& qpos) const {
    if (locked_qpos_count_ > 0 && locked_qpos_.size() == locked_qpos_count_)
      qpos.head(locked_qpos_count_) = locked_qpos_;
    for (size_t i = 0; i < clamp_qpos_indices_.size(); ++i)
      qpos[clamp_qpos_indices_[i]] = clamp_qpos_values_[static_cast<int>(i)];
  }

  mjModel* m_ = nullptr;
  mjData* d_ = nullptr;
  ScenarioSpec spec_;
  Eigen::VectorXd qpos_;
  Eigen::VectorXd qpos0_;
  double dt_ = 1.0 / 120.0;
  double damping_ = 1e-4;
  double config_gain_ = 0.95;
  double posture_weight_ = 0.02;
  double max_iteration_update_ = 0.0;
  double max_frame_update_ = 0.0;
  int max_iterations_ = 8;
  Eigen::VectorXd solve_start_qpos_;

  std::vector<Task> tasks_;
  std::vector<bool> active_dof_;
  std::vector<int> qpos_to_dof_;

  SourceMode source_mode_ = SourceMode::RobotWorldAbsolute;
  OrientationMode orientation_mode_ = OrientationMode::Neutral;
  double workspace_scale_ = 1.0;
  double arm_reach_scale_ = 0.82;
  double source_fps_ = 30.0;
  double source_time_s_ = 0.0;
  bool has_neutral_reference_ = false;
  utils::OneEuroPoseFilter source_filter_{0.0, 0.0, 1.0};
  SkeletonFrame last_targets_;

  int locked_qpos_count_ = 0;
  Eigen::VectorXd locked_qpos_;
  std::vector<int> clamp_qpos_indices_;
  Eigen::VectorXd clamp_qpos_values_;

  bool collision_enabled_ = false;
  double collision_min_distance_ = 0.06;
  double collision_activation_distance_ = 0.10;
  double collision_gain_ = 0.75;
  double collision_weight_ = 100.0;
  int collision_projection_iterations_ = 8;
  std::vector<CollisionPair> collision_pairs_;
};

DualArmEePoseAlgorithm::DualArmEePoseAlgorithm() : impl_(std::make_unique<Impl>()) {}
DualArmEePoseAlgorithm::~DualArmEePoseAlgorithm() = default;

void DualArmEePoseAlgorithm::configure(const RetargetConfig& config,
                                       const ScenarioSpec& spec) {
  impl_->configure(config, spec);
}

void DualArmEePoseAlgorithm::begin_frame() { impl_->begin_frame(); }

Eigen::VectorXd DualArmEePoseAlgorithm::solve(const SkeletonFrame& frame) {
  return impl_->solve(frame);
}

void DualArmEePoseAlgorithm::end_frame(Eigen::VectorXd& qpos) {
  impl_->end_frame(qpos);
}

int DualArmEePoseAlgorithm::nq() const { return impl_->nq(); }

int DualArmEePoseAlgorithm::nv() const { return impl_->nv(); }

void DualArmEePoseAlgorithm::set_configuration(const Eigen::VectorXd& qpos) {
  impl_->set_configuration(qpos);
}

SkeletonFrame DualArmEePoseAlgorithm::last_targets() const {
  return impl_->last_targets();
}

#endif  // RETARGETING_NO_MUJOCO

}  // namespace dual_arm_eepose
}  // namespace retargeting
