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

#include <nlohmann/json.hpp>

#include "box_qp.hpp"
#include "lie.hpp"

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

#else

namespace {

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

}  // namespace

struct DualArmEePoseAlgorithm::Impl {
  struct Task {
    std::string name;
    std::vector<std::string> target_names;
    std::string body_name;
    int body = -1;
    std::vector<int> joint_qpos;
    std::vector<int> joint_dofs;
    Eigen::Matrix<double, 6, 1> cost = Eigen::Matrix<double, 6, 1>::Ones();
    double gain = 1.0;
    Eigen::Matrix<double, 7, 1> target = Eigen::Matrix<double, 7, 1>::Zero();
    bool has_target = false;
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

    active_dof_.assign(m_->nv, false);
    load_tasks(cfg);
    load_posture_reference(cfg);
    set_qpos(qpos_);
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
  }

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
      task.target_names.push_back(entry.at("target").get<std::string>());
      if (entry.contains("target_aliases")) {
        auto aliases = string_array(entry["target_aliases"]);
        task.target_names.insert(task.target_names.end(), aliases.begin(), aliases.end());
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
    if (cfg.contains("posture_reference"))
      apply_joint_positions(cfg["posture_reference"], true, true);
    if (cfg.contains("initial_joint_positions"))
      apply_joint_positions(cfg["initial_joint_positions"], false, true);
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

  void update_targets(const SkeletonFrame& frame) {
    for (auto& task : tasks_) {
      task.has_target = false;
      for (const std::string& name : task.target_names) {
        auto it = frame.find(name);
        if (it == frame.end()) continue;
        const Pose& pose = it->second;
        Eigen::Vector4d q = normalized_quat(pose.quat);
        task.target << q[0], q[1], q[2], q[3], pose.pos[0], pose.pos[1], pose.pos[2];
        task.has_target = true;
        break;
      }
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
    Eigen::MatrixXd jac6;
    frame_jacobian(task.body, jac6);

    Eigen::Matrix<double, 6, 1> error = lie::se3_rminus(task.target, current);
    double T_tb[7];
    lie::se3_inverse_multiply(task.target.data(), current.data(), T_tb);
    double jlog[36];
    lie::se3_jlog(T_tb, jlog);
    Eigen::Matrix<double, 6, 6> Jlog;
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) Jlog(i, j) = jlog[6 * i + j];

    Eigen::MatrixXd jacobian = -(Jlog * jac6);
    Eigen::Matrix<double, 6, 1> weighted_error =
        task.cost.array() * (-task.gain * error).array();
    Eigen::MatrixXd weighted_jacobian = task.cost.asDiagonal() * jacobian;
    H.noalias() += weighted_jacobian.transpose() * weighted_jacobian;
    c.noalias() -= weighted_jacobian.transpose() * weighted_error;
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

#endif  // RETARGETING_NO_MUJOCO

}  // namespace dual_arm_eepose
}  // namespace retargeting
