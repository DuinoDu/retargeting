// spatialmp4_to_galbot_dual_arm
//
// Offline adapter for the pure dual-arm eepose algorithm:
//   body.jsonl (raw OpenXR body joints) -> frame-to-frame left/right wrist
//   eepose deltas -> Galbot G1 arm joint commands.
//
// The retargeting algorithm itself consumes only end-effector poses. This file
// is test glue for recorded SpatialMP4 body tracks.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <mujoco/mujoco.h>
#include <nlohmann/json.hpp>

#include "lie.hpp"
#include "retargeting/retargeter.hpp"
#include "retargeting/version.hpp"
#include "utils/one_euro_filter.hpp"

using json = nlohmann::json;
using retargeting::Pose;
using retargeting::SkeletonFrame;
using retargeting::utils::NamedPose;
using retargeting::utils::OneEuroPoseFilter;

namespace {

const std::vector<std::string> kArmJointNames = {
    "left_arm_joint1",  "left_arm_joint2",  "left_arm_joint3",  "left_arm_joint4",
    "left_arm_joint5",  "left_arm_joint6",  "left_arm_joint7",  "right_arm_joint1",
    "right_arm_joint2", "right_arm_joint3", "right_arm_joint4", "right_arm_joint5",
    "right_arm_joint6", "right_arm_joint7"};

constexpr double kFallbackFrameDt = 1.0 / 30.0;

struct RawFrame {
  double timestamp_s = 0.0;
  std::vector<NamedPose> joints;
};

struct RobotRest {
  Eigen::Vector3d left_base_pos = Eigen::Vector3d::Zero();
  Eigen::Vector3d right_base_pos = Eigen::Vector3d::Zero();
  Eigen::Vector3d left_tcp_pos = Eigen::Vector3d::Zero();
  Eigen::Vector3d right_tcp_pos = Eigen::Vector3d::Zero();
  Eigen::Vector3d left_straight_tcp_pos = Eigen::Vector3d::Zero();
  Eigen::Vector3d right_straight_tcp_pos = Eigen::Vector3d::Zero();
  double left_reach = 0.0;
  double right_reach = 0.0;
  Eigen::Vector4d left_tcp_quat = Eigen::Vector4d(1, 0, 0, 0);
  Eigen::Vector4d right_tcp_quat = Eigen::Vector4d(1, 0, 0, 0);
  bool has_neutral_reference = false;
  std::vector<int> arm_qpos_indices;
};

struct TargetState {
  bool initialized = false;
  Eigen::Vector3d prev_left_pos = Eigen::Vector3d::Zero();
  Eigen::Vector3d prev_right_pos = Eigen::Vector3d::Zero();
  Eigen::Vector4d prev_left_quat = Eigen::Vector4d(1, 0, 0, 0);
  Eigen::Vector4d prev_right_quat = Eigen::Vector4d(1, 0, 0, 0);
  Eigen::Vector4d initial_left_quat = Eigen::Vector4d(1, 0, 0, 0);
  Eigen::Vector4d initial_right_quat = Eigen::Vector4d(1, 0, 0, 0);
  Pose left_target;
  Pose right_target;
};

enum class TargetMode {
  FrameDelta,
  Se3Delta,
  ShoulderAbsolute,
};

enum class OrientationMode {
  Neutral,
  RelativeWrist,
};

Eigen::Vector3d openxr_to_gmr(const Eigen::Vector3d& p) {
  return Eigen::Vector3d(-p.z(), -p.x(), p.y());
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

Eigen::Vector4d qnormalize(const Eigen::Vector4d& q) {
  double n = q.norm();
  if (!(n > 0.0) || !std::isfinite(n)) return Eigen::Vector4d(1, 0, 0, 0);
  return q / n;
}

Eigen::Vector3d qrotate(const Eigen::Vector4d& q, const Eigen::Vector3d& v) {
  Eigen::Vector4d p(0, v.x(), v.y(), v.z());
  Eigen::Vector4d out = qmul(qmul(qnormalize(q), p), qconj(qnormalize(q)));
  return Eigen::Vector3d(out[1], out[2], out[3]);
}

const Eigen::Vector4d kOpenxrToGmrQuat(0.5, 0.5, -0.5, -0.5);

bool find_pose(const std::vector<NamedPose>& joints, const std::string& name, NamedPose& out) {
  for (const auto& joint : joints) {
    if (joint.name == name) {
      out = joint;
      return true;
    }
  }
  return false;
}

bool get_pos(const RawFrame& f, const std::string& name, Eigen::Vector3d& out) {
  for (const auto& joint : f.joints) {
    if (joint.name == name) {
      out = joint.pos;
      return true;
    }
  }
  return false;
}

std::vector<RawFrame> load_body_jsonl(const std::string& path) {
  std::vector<RawFrame> frames;
  std::ifstream in(path);
  std::string line;
  int idx = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    json r = json::parse(line);
    RawFrame f;
    f.timestamp_s = r.value("timestamp_s", static_cast<double>(idx) / 30.0);
    for (auto& [name, rec] : r["joints"].items()) {
      auto pos = rec["position"];
      auto rot = rec["rotation"];
      NamedPose j;
      j.name = name;
      j.pos = Eigen::Vector3d(pos[0].get<double>(), pos[1].get<double>(), pos[2].get<double>());
      j.quat = Eigen::Vector4d(rot[3].get<double>(), rot[0].get<double>(),
                               rot[1].get<double>(), rot[2].get<double>());
      f.joints.push_back(std::move(j));
    }
    frames.push_back(std::move(f));
    ++idx;
  }
  return frames;
}

void body_pose(const mjModel* m, const mjData* d, const std::string& name,
               Eigen::Vector3d& pos, Eigen::Vector4d& quat) {
  int id = mj_name2id(m, mjOBJ_BODY, name.c_str());
  if (id < 0) throw std::runtime_error("unknown robot body: " + name);
  double frame7[7];
  lie::xmat_xpos_to_wxyz_xyz(d->xmat + 9 * id, d->xpos + 3 * id, frame7);
  quat = Eigen::Vector4d(frame7[0], frame7[1], frame7[2], frame7[3]);
  pos = Eigen::Vector3d(frame7[4], frame7[5], frame7[6]);
}

bool apply_config_joint_positions(const std::string& ee_config, mjModel* m, mjData* d) {
  if (ee_config.empty()) return false;
  std::ifstream in(ee_config);
  if (!in) return false;
  json cfg = json::parse(in);
  bool applied = false;
  auto apply_block = [&](const char* key) {
    if (!cfg.contains(key)) return;
    for (auto& [joint_name, value] : cfg[key].items()) {
      int jid = mj_name2id(m, mjOBJ_JOINT, joint_name.c_str());
      if (jid < 0) throw std::runtime_error("unknown joint in " + std::string(key) + ": " +
                                            joint_name);
      if (m->jnt_type[jid] != mjJNT_HINGE && m->jnt_type[jid] != mjJNT_SLIDE)
        throw std::runtime_error("non 1-DoF joint in " + std::string(key) + ": " +
                                 joint_name);
      d->qpos[m->jnt_qposadr[jid]] = value.get<double>();
      applied = true;
    }
  };
  apply_block("posture_reference");
  apply_block("initial_joint_positions");
  if (applied) mj_forward(m, d);
  return applied;
}

RobotRest load_robot_rest(const std::string& robot_xml, const std::string& ee_config) {
  char error[1000] = "";
  mjModel* m = mj_loadXML(robot_xml.c_str(), nullptr, error, sizeof(error));
  if (!m) throw std::runtime_error(std::string("mj_loadXML failed: ") + error);
  mjData* d = mj_makeData(m);
  if (!d) {
    mj_deleteModel(m);
    throw std::runtime_error("mj_makeData failed");
  }
  mj_forward(m, d);

  RobotRest rest;
  Eigen::Vector4d unused;
  body_pose(m, d, "left_arm_base_link", rest.left_base_pos, unused);
  body_pose(m, d, "right_arm_base_link", rest.right_base_pos, unused);
  body_pose(m, d, "left_gripper_tcp_link", rest.left_straight_tcp_pos, rest.left_tcp_quat);
  body_pose(m, d, "right_gripper_tcp_link", rest.right_straight_tcp_pos, rest.right_tcp_quat);
  rest.left_reach = (rest.left_straight_tcp_pos - rest.left_base_pos).norm();
  rest.right_reach = (rest.right_straight_tcp_pos - rest.right_base_pos).norm();

  rest.has_neutral_reference = apply_config_joint_positions(ee_config, m, d);
  body_pose(m, d, "left_gripper_tcp_link", rest.left_tcp_pos, rest.left_tcp_quat);
  body_pose(m, d, "right_gripper_tcp_link", rest.right_tcp_pos, rest.right_tcp_quat);

  for (const std::string& joint_name : kArmJointNames) {
    int jid = mj_name2id(m, mjOBJ_JOINT, joint_name.c_str());
    if (jid < 0) throw std::runtime_error("unknown robot joint: " + joint_name);
    rest.arm_qpos_indices.push_back(m->jnt_qposadr[jid]);
  }

  mj_deleteData(d);
  mj_deleteModel(m);
  return rest;
}

double clamp_delta(double prev, double next, double max_delta) {
  double delta = next - prev;
  if (delta > max_delta) return prev + max_delta;
  if (delta < -max_delta) return prev - max_delta;
  return next;
}

double low_pass_alpha(double cutoff_hz, double dt) {
  if (!(cutoff_hz > 0.0) || !(dt > 0.0) || !std::isfinite(dt)) return 1.0;
  const double tau = 1.0 / (2.0 * M_PI * cutoff_hz);
  return dt / (tau + dt);
}

Eigen::Vector3d clamp_reach(const Eigen::Vector3d& base, const Eigen::Vector3d& target,
                            double max_reach) {
  Eigen::Vector3d v = target - base;
  double n = v.norm();
  if (!(max_reach > 0.0) || !(n > max_reach)) return target;
  return base + v * (max_reach / n);
}

bool make_eepose_targets(const std::vector<NamedPose>& aligned, const RobotRest& rest,
                         TargetState& state, double workspace_scale,
                         double arm_reach_scale, TargetMode target_mode,
                         OrientationMode orientation_mode, SkeletonFrame& out) {
  NamedPose left_shoulder, right_shoulder, left_wrist, right_wrist;
  if (!find_pose(aligned, "LeftShoulder", left_shoulder) ||
      !find_pose(aligned, "RightShoulder", right_shoulder) ||
      !find_pose(aligned, "LeftWrist", left_wrist) ||
      !find_pose(aligned, "RightWrist", right_wrist)) {
    return false;
  }

  Eigen::Vector3d left_rel = left_wrist.pos - left_shoulder.pos;
  Eigen::Vector3d right_rel = right_wrist.pos - right_shoulder.pos;
  const Eigen::Vector4d left_wrist_quat = qnormalize(left_wrist.quat);
  const Eigen::Vector4d right_wrist_quat = qnormalize(right_wrist.quat);

  if (!state.initialized) {
    state.initialized = true;
    state.prev_left_pos = left_wrist.pos;
    state.prev_right_pos = right_wrist.pos;
    state.prev_left_quat = left_wrist_quat;
    state.prev_right_quat = right_wrist_quat;
    state.initial_left_quat = left_wrist_quat;
    state.initial_right_quat = right_wrist_quat;

    if (target_mode == TargetMode::ShoulderAbsolute) {
      state.left_target.pos = rest.left_base_pos + workspace_scale * left_rel;
      state.right_target.pos = rest.right_base_pos + workspace_scale * right_rel;
    } else {
      state.left_target.pos = rest.has_neutral_reference
          ? rest.left_tcp_pos
          : rest.left_base_pos + arm_reach_scale *
                                    (rest.left_straight_tcp_pos - rest.left_base_pos);
      state.right_target.pos = rest.has_neutral_reference
          ? rest.right_tcp_pos
          : rest.right_base_pos + arm_reach_scale *
                                     (rest.right_straight_tcp_pos - rest.right_base_pos);
    }
    state.left_target.quat = rest.left_tcp_quat;
    state.right_target.quat = rest.right_tcp_quat;
  } else {
    if (target_mode == TargetMode::ShoulderAbsolute) {
      state.left_target.pos = rest.left_base_pos + workspace_scale * left_rel;
      state.right_target.pos = rest.right_base_pos + workspace_scale * right_rel;
      if (orientation_mode == OrientationMode::RelativeWrist) {
        state.left_target.quat = qnormalize(qmul(
            rest.left_tcp_quat, qmul(qconj(state.initial_left_quat), left_wrist_quat)));
        state.right_target.quat = qnormalize(qmul(
            rest.right_tcp_quat, qmul(qconj(state.initial_right_quat), right_wrist_quat)));
      } else {
        state.left_target.quat = rest.left_tcp_quat;
        state.right_target.quat = rest.right_tcp_quat;
      }
    } else {
      const Eigen::Vector4d left_delta_quat =
          qnormalize(qmul(qconj(state.prev_left_quat), left_wrist_quat));
      const Eigen::Vector4d right_delta_quat =
          qnormalize(qmul(qconj(state.prev_right_quat), right_wrist_quat));
      Eigen::Vector3d left_delta_pos = left_wrist.pos - state.prev_left_pos;
      Eigen::Vector3d right_delta_pos = right_wrist.pos - state.prev_right_pos;

      if (target_mode == TargetMode::Se3Delta) {
        left_delta_pos = qrotate(qconj(state.prev_left_quat), left_delta_pos);
        right_delta_pos = qrotate(qconj(state.prev_right_quat), right_delta_pos);
        state.left_target.pos +=
            qrotate(state.left_target.quat, workspace_scale * left_delta_pos);
        state.right_target.pos +=
            qrotate(state.right_target.quat, workspace_scale * right_delta_pos);
      } else {
        state.left_target.pos += workspace_scale * left_delta_pos;
        state.right_target.pos += workspace_scale * right_delta_pos;
      }

      if (orientation_mode == OrientationMode::RelativeWrist) {
        state.left_target.quat = qnormalize(qmul(state.left_target.quat, left_delta_quat));
        state.right_target.quat = qnormalize(qmul(state.right_target.quat, right_delta_quat));
      } else {
        state.left_target.quat = rest.left_tcp_quat;
        state.right_target.quat = rest.right_tcp_quat;
      }
    }

    state.prev_left_pos = left_wrist.pos;
    state.prev_right_pos = right_wrist.pos;
    state.prev_left_quat = left_wrist_quat;
    state.prev_right_quat = right_wrist_quat;
  }

  state.left_target.pos =
      clamp_reach(rest.left_base_pos, state.left_target.pos, rest.left_reach * 0.995);
  state.right_target.pos =
      clamp_reach(rest.right_base_pos, state.right_target.pos, rest.right_reach * 0.995);

  out["LeftWrist"] = state.left_target;
  out["RightWrist"] = state.right_target;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--version") {
    printf("retargeting %s\n", retargeting::kVersion);
    return 0;
  }
  if (argc < 2) {
    fprintf(stderr,
            "usage: %s <body.jsonl> --robot_xml <galbot_g1.xml> --save_jsonl <out> "
            "[--target_mode frame_delta|se3_delta|shoulder_absolute] "
            "[--orientation_mode neutral|relative_wrist] [opts]\n",
            argv[0]);
    return 2;
  }

  std::string body_path = argv[1];
  std::string save_path, target_path;
  std::string robot_xml;
  std::string ee_config = "data/ik_configs/dual_arm_eepose_galbot_g1.json";
  double pelvis_height = 0.88;
  double workspace_scale = 1.0;
  double arm_reach_scale = 0.82;
  double max_joint_velocity_deg_s = 240.0;
  double joint_lowpass_cutoff = 0.0;
  double input_one_euro_min_cutoff = 0.6;
  double input_one_euro_beta = 0.03;
  double input_one_euro_d_cutoff = 1.0;
  bool align_heading = true;
  TargetMode target_mode = TargetMode::FrameDelta;
  std::string target_mode_name = "frame_delta";
  OrientationMode orientation_mode = OrientationMode::Neutral;
  std::string orientation_mode_name = "neutral";
  int max_frames = 0;

  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--save_jsonl") save_path = next();
    else if (a == "--target_jsonl") target_path = next();
    else if (a == "--robot_xml") robot_xml = next();
    else if (a == "--ee_config") ee_config = next();
    else if (a == "--pelvis_height") pelvis_height = std::stod(next());
    else if (a == "--workspace_scale") workspace_scale = std::stod(next());
    else if (a == "--arm_reach_scale") arm_reach_scale = std::stod(next());
    else if (a == "--max_joint_velocity_deg_s") max_joint_velocity_deg_s = std::stod(next());
    else if (a == "--joint_lowpass_cutoff") joint_lowpass_cutoff = std::stod(next());
    else if (a == "--input_one_euro_min_cutoff") input_one_euro_min_cutoff = std::stod(next());
    else if (a == "--input_one_euro_beta") input_one_euro_beta = std::stod(next());
    else if (a == "--input_one_euro_d_cutoff") input_one_euro_d_cutoff = std::stod(next());
    else if (a == "--target_mode") {
      target_mode_name = next();
      if (target_mode_name == "frame_delta") {
        target_mode = TargetMode::FrameDelta;
      } else if (target_mode_name == "se3_delta") {
        target_mode = TargetMode::Se3Delta;
      } else if (target_mode_name == "shoulder_absolute" || target_mode_name == "absolute") {
        target_mode = TargetMode::ShoulderAbsolute;
        target_mode_name = "shoulder_absolute";
      } else {
        fprintf(stderr, "unknown --target_mode: %s\n", target_mode_name.c_str());
        return 2;
      }
    }
    else if (a == "--orientation_mode") {
      orientation_mode_name = next();
      if (orientation_mode_name == "neutral") {
        orientation_mode = OrientationMode::Neutral;
      } else if (orientation_mode_name == "relative_wrist") {
        orientation_mode = OrientationMode::RelativeWrist;
      } else {
        fprintf(stderr, "unknown --orientation_mode: %s\n", orientation_mode_name.c_str());
        return 2;
      }
    }
    else if (a == "--max_frames") max_frames = std::stoi(next());
    else if (a == "--no_heading_align") align_heading = false;
    else if (a == "--absolute_position") {
      target_mode = TargetMode::ShoulderAbsolute;
      target_mode_name = "shoulder_absolute";
    } else {
      fprintf(stderr, "unknown argument: %s\n", a.c_str());
      return 2;
    }
  }
  if (save_path.empty() || robot_xml.empty()) {
    fprintf(stderr, "missing required --robot_xml <galbot_g1.xml> or --save_jsonl <out>\n");
    return 2;
  }

  std::vector<RawFrame> raw = load_body_jsonl(body_path);
  if (raw.empty()) {
    fprintf(stderr, "no frames in %s\n", body_path.c_str());
    return 2;
  }
  if (max_frames > 0 && max_frames < static_cast<int>(raw.size()))
    raw.resize(static_cast<size_t>(max_frames));

  double heading_yaw = 0.0;
  if (align_heading) {
    Eigen::Vector3d ls, rs;
    if (get_pos(raw[0], "LeftShoulder", ls) && get_pos(raw[0], "RightShoulder", rs)) {
      Eigen::Vector3d side = openxr_to_gmr(ls) - openxr_to_gmr(rs);
      side.z() = 0.0;
      if (side.norm() > 1e-5)
        heading_yaw = M_PI / 2.0 - std::atan2(side.y(), side.x());
    }
  }
  const double cz = std::cos(heading_yaw), sz = std::sin(heading_yaw);
  const Eigen::Vector4d rz_quat(std::cos(heading_yaw / 2.0), 0, 0,
                                std::sin(heading_yaw / 2.0));
  const Eigen::Vector4d combined_quat = qmul(rz_quat, kOpenxrToGmrQuat);

  auto anchor = [&](const RawFrame& f, std::vector<NamedPose>& out) {
    Eigen::Vector3d hips_xr;
    if (!get_pos(f, "Hips", hips_xr)) return false;
    Eigen::Vector3d root = openxr_to_gmr(hips_xr);
    for (const auto& jin : f.joints) {
      Eigen::Vector3d r = openxr_to_gmr(jin.pos) - root;
      NamedPose jo;
      jo.name = jin.name;
      jo.pos = Eigen::Vector3d(r.x() * cz - r.y() * sz, r.x() * sz + r.y() * cz,
                               r.z() + pelvis_height);
      jo.quat = qnormalize(qmul(combined_quat, jin.quat));
      out.push_back(std::move(jo));
    }
    return true;
  };

  RobotRest rest = load_robot_rest(robot_xml, ee_config);

  retargeting::RetargetConfig cfg;
  cfg.robot_xml = robot_xml;
  cfg.ik_config_json = ee_config;
  cfg.backend = retargeting::KinematicsBackendKind::Mujoco;
  cfg.damping = 0.001;
  auto rt = retargeting::UpperBodyRetargeter::create(cfg, 14, "dual_arm_eepose", true);

  printf("loaded %zu frames; heading_yaw=%.2f deg; workspace_scale=%.2f; arm_reach_scale=%.2f; mode=%s; orientation=%s\n",
         raw.size(), heading_yaw * 180.0 / M_PI, workspace_scale, arm_reach_scale,
         target_mode_name.c_str(), orientation_mode_name.c_str());
  printf("input_filter=(min_cutoff=%.3f beta=%.3f d_cutoff=%.3f) joint_lowpass=%.3fHz\n",
         input_one_euro_min_cutoff, input_one_euro_beta, input_one_euro_d_cutoff,
         joint_lowpass_cutoff);
  printf("robot=%s\nconfig=%s\n", robot_xml.c_str(), ee_config.c_str());

  OneEuroPoseFilter input_filter(input_one_euro_min_cutoff, input_one_euro_beta,
                                 input_one_euro_d_cutoff);
  std::ofstream out(save_path);
  std::ofstream target_out;
  if (!target_path.empty()) target_out.open(target_path);

  TargetState target_state;
  const double max_joint_velocity_rad_s = max_joint_velocity_deg_s * M_PI / 180.0;
  Eigen::VectorXd prev_q;
  double prev_timestamp_s = std::numeric_limits<double>::quiet_NaN();
  int written = 0;

  for (size_t fi = 0; fi < raw.size(); ++fi) {
    std::vector<NamedPose> aligned;
    if (!anchor(raw[fi], aligned)) continue;
    input_filter.filter(raw[fi].timestamp_s, aligned);

    SkeletonFrame targets;
    if (!make_eepose_targets(aligned, rest, target_state, workspace_scale, arm_reach_scale,
                             target_mode, orientation_mode, targets)) {
      continue;
    }

    Eigen::VectorXd q = rt->step(targets);
    if (prev_q.size() == q.size()) {
      double frame_dt = raw[fi].timestamp_s - prev_timestamp_s;
      if (!(frame_dt > 0.0) || !std::isfinite(frame_dt)) frame_dt = kFallbackFrameDt;
      const double alpha = low_pass_alpha(joint_lowpass_cutoff, frame_dt);
      for (int qi : rest.arm_qpos_indices) {
        q[qi] = prev_q[qi] + alpha * (q[qi] - prev_q[qi]);
      }
      if (max_joint_velocity_rad_s > 0.0) {
        const double max_delta = max_joint_velocity_rad_s * frame_dt;
        for (int qi : rest.arm_qpos_indices) q[qi] = clamp_delta(prev_q[qi], q[qi], max_delta);
      }
      rt->set_configuration(q);
    }
    prev_q = q;
    prev_timestamp_s = raw[fi].timestamp_s;

    json rec;
    rec["timestamp_ns"] = static_cast<long long>(std::llround(raw[fi].timestamp_s * 1e9));
    rec["joint_names"] = kArmJointNames;
    std::vector<double> joint_q;
    joint_q.reserve(rest.arm_qpos_indices.size());
    for (int qi : rest.arm_qpos_indices) joint_q.push_back(q[qi]);
    rec["joint_q"] = joint_q;
    out << rec.dump() << "\n";
    ++written;

    if (target_out.is_open()) {
      json tr;
      tr["timestamp_s"] = raw[fi].timestamp_s;
      tr["coordinate_space"] = "robot_world";
      tr["quat_order"] = "wxyz";
      for (const auto& [name, pose] : targets) {
        tr["targets"][name] = {
            {"position", {pose.pos.x(), pose.pos.y(), pose.pos.z()}},
            {"rotation", {pose.quat[0], pose.quat[1], pose.quat[2], pose.quat[3]}}};
      }
      target_out << tr.dump() << "\n";
    }
  }

  out.close();
  printf("saved %d dual-arm frames -> %s\n", written, save_path.c_str());
  if (!target_path.empty()) printf("saved eepose targets -> %s\n", target_path.c_str());
  return written > 0 ? 0 : 2;
}
