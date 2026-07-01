// spatialmp4_to_galbot_dual_arm
//
// Offline adapter for the pure dual-arm eepose algorithm:
//   body.jsonl (raw OpenXR body joints) -> aligned SkeletonFrame
//   -> dual_arm_eepose retargeting -> Galbot G1 arm joint commands.
//
// The retargeting algorithm owns the shoulder-relative EE source semantics.
// This file is test glue for recorded SpatialMP4 body tracks.
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

#include "algorithms/dual_arm_eepose/dual_arm_eepose_algorithm.hpp"
#include "retargeting/retargeter.hpp"
#include "retargeting/version.hpp"

using json = nlohmann::json;
using retargeting::Pose;
using retargeting::SkeletonFrame;

namespace {

const std::vector<std::string> kArmJointNames = {
    "left_arm_joint1",  "left_arm_joint2",  "left_arm_joint3",  "left_arm_joint4",
    "left_arm_joint5",  "left_arm_joint6",  "left_arm_joint7",  "right_arm_joint1",
    "right_arm_joint2", "right_arm_joint3", "right_arm_joint4", "right_arm_joint5",
    "right_arm_joint6", "right_arm_joint7"};

constexpr double kFallbackFrameDt = 1.0 / 30.0;

struct NamedPose {
  std::string name;
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  Eigen::Vector4d quat = Eigen::Vector4d(1, 0, 0, 0);
};

struct RawFrame {
  double timestamp_s = 0.0;
  std::vector<NamedPose> joints;
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

Eigen::Vector4d qnormalize(const Eigen::Vector4d& q) {
  double n = q.norm();
  if (!(n > 0.0) || !std::isfinite(n)) return Eigen::Vector4d(1, 0, 0, 0);
  return q / n;
}

const Eigen::Vector4d kOpenxrToGmrQuat(0.5, 0.5, -0.5, -0.5);

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

struct RobotRuntime {
  mjModel* model = nullptr;
  mjData* data = nullptr;
  std::vector<int> arm_qpos_indices;

  RobotRuntime() = default;
  RobotRuntime(const RobotRuntime&) = delete;
  RobotRuntime& operator=(const RobotRuntime&) = delete;
  RobotRuntime(RobotRuntime&& other) noexcept
      : model(other.model), data(other.data), arm_qpos_indices(std::move(other.arm_qpos_indices)) {
    other.model = nullptr;
    other.data = nullptr;
  }
  ~RobotRuntime() {
    if (data) mj_deleteData(data);
    if (model) mj_deleteModel(model);
  }

};

RobotRuntime load_robot_runtime(const std::string& robot_xml) {
  char error[1000] = "";
  RobotRuntime rt;
  rt.model = mj_loadXML(robot_xml.c_str(), nullptr, error, sizeof(error));
  if (!rt.model) throw std::runtime_error(std::string("mj_loadXML failed: ") + error);
  rt.data = mj_makeData(rt.model);
  if (!rt.data) throw std::runtime_error("mj_makeData failed");
  for (const std::string& joint_name : kArmJointNames) {
    int jid = mj_name2id(rt.model, mjOBJ_JOINT, joint_name.c_str());
    if (jid < 0) throw std::runtime_error("unknown robot joint: " + joint_name);
    rt.arm_qpos_indices.push_back(rt.model->jnt_qposadr[jid]);
  }
  return rt;
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
            "[--target_mode shoulder_delta|frame_delta|se3_delta|shoulder_absolute] "
            "[--orientation_mode neutral|relative_wrist|relative_wrist_roll|wrist_roll] [opts]\n",
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
  double input_one_euro_min_cutoff = 0.0;
  double input_one_euro_beta = 0.03;
  double input_one_euro_d_cutoff = 1.0;
  bool align_heading = true;
  std::string target_mode_name = "shoulder_delta";
  std::string orientation_mode_name = "relative_wrist_roll";
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
      if (target_mode_name == "absolute") {
        target_mode_name = "shoulder_absolute";
      }
      if (target_mode_name != "shoulder_delta" && target_mode_name != "frame_delta" &&
          target_mode_name != "se3_delta" && target_mode_name != "shoulder_absolute") {
        fprintf(stderr, "unknown --target_mode: %s\n", target_mode_name.c_str());
        return 2;
      }
    }
    else if (a == "--orientation_mode") {
      orientation_mode_name = next();
      if (orientation_mode_name == "wrist_roll") {
        orientation_mode_name = "relative_wrist_roll";
      }
      if (orientation_mode_name != "neutral" &&
          orientation_mode_name != "relative_wrist" &&
          orientation_mode_name != "relative_wrist_roll") {
        fprintf(stderr, "unknown --orientation_mode: %s\n", orientation_mode_name.c_str());
        return 2;
      }
    }
    else if (a == "--max_frames") max_frames = std::stoi(next());
    else if (a == "--no_heading_align") align_heading = false;
    else if (a == "--absolute_position") {
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
  double source_fps = 30.0;
  if (raw.size() > 1) {
    const double duration = raw.back().timestamp_s - raw.front().timestamp_s;
    if (duration > 0.0 && std::isfinite(duration))
      source_fps = static_cast<double>(raw.size() - 1) / duration;
  }

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

  RobotRuntime robot = load_robot_runtime(robot_xml);

  retargeting::RetargetConfig cfg;
  cfg.robot_xml = robot_xml;
  cfg.ik_config_json = ee_config;
  cfg.backend = retargeting::KinematicsBackendKind::Mujoco;
  cfg.damping = 0.001;
  cfg.options["source_mode"] = target_mode_name;
  cfg.options["orientation_mode"] = orientation_mode_name;
  cfg.options["workspace_scale"] = std::to_string(workspace_scale);
  cfg.options["arm_reach_scale"] = std::to_string(arm_reach_scale);
  cfg.options["source_fps"] = std::to_string(source_fps);
  cfg.options["source_filter_min_cutoff"] = std::to_string(input_one_euro_min_cutoff);
  cfg.options["source_filter_beta"] = std::to_string(input_one_euro_beta);
  cfg.options["source_filter_d_cutoff"] = std::to_string(input_one_euro_d_cutoff);
  auto rt = retargeting::UpperBodyRetargeter::create(cfg, 14, "dual_arm_eepose", true);
  auto* ee_algo = dynamic_cast<retargeting::dual_arm_eepose::DualArmEePoseAlgorithm*>(
      &rt->algorithm());

  printf("loaded %zu frames; heading_yaw=%.2f deg; source_fps=%.3f; workspace_scale=%.2f; arm_reach_scale=%.2f; mode=%s; orientation=%s\n",
         raw.size(), heading_yaw * 180.0 / M_PI, source_fps, workspace_scale,
         arm_reach_scale, target_mode_name.c_str(), orientation_mode_name.c_str());
  printf("source_filter=(min_cutoff=%.3f beta=%.3f d_cutoff=%.3f) joint_lowpass=%.3fHz\n",
         input_one_euro_min_cutoff, input_one_euro_beta, input_one_euro_d_cutoff,
         joint_lowpass_cutoff);
  printf("robot=%s\nconfig=%s\n", robot_xml.c_str(), ee_config.c_str());

  std::ofstream out(save_path);
  std::ofstream target_out;
  if (!target_path.empty()) target_out.open(target_path);

  const double max_joint_velocity_rad_s = max_joint_velocity_deg_s * M_PI / 180.0;
  Eigen::VectorXd prev_q;
  double prev_timestamp_s = std::numeric_limits<double>::quiet_NaN();
  int written = 0;

  for (size_t fi = 0; fi < raw.size(); ++fi) {
    std::vector<NamedPose> aligned;
    if (!anchor(raw[fi], aligned)) continue;

    SkeletonFrame frame;
    for (const auto& pose : aligned) {
      Pose out_pose;
      out_pose.pos = pose.pos;
      out_pose.quat = pose.quat;
      frame[pose.name] = out_pose;
    }

    Eigen::VectorXd q = rt->step(frame);
    if (prev_q.size() == q.size()) {
      double frame_dt = raw[fi].timestamp_s - prev_timestamp_s;
      if (!(frame_dt > 0.0) || !std::isfinite(frame_dt)) frame_dt = kFallbackFrameDt;
      const double alpha = low_pass_alpha(joint_lowpass_cutoff, frame_dt);
      for (int qi : robot.arm_qpos_indices) {
        q[qi] = prev_q[qi] + alpha * (q[qi] - prev_q[qi]);
      }
      if (max_joint_velocity_rad_s > 0.0) {
        const double max_delta = max_joint_velocity_rad_s * frame_dt;
        for (int qi : robot.arm_qpos_indices) q[qi] = clamp_delta(prev_q[qi], q[qi], max_delta);
      }
      rt->set_configuration(q);
    }
    prev_q = q;
    prev_timestamp_s = raw[fi].timestamp_s;

    json rec;
    rec["timestamp_ns"] = static_cast<long long>(std::llround(raw[fi].timestamp_s * 1e9));
    rec["joint_names"] = kArmJointNames;
    std::vector<double> joint_q;
    joint_q.reserve(robot.arm_qpos_indices.size());
    for (int qi : robot.arm_qpos_indices) joint_q.push_back(q[qi]);
    rec["joint_q"] = joint_q;
    out << rec.dump() << "\n";
    ++written;

    if (target_out.is_open()) {
      json tr;
      tr["timestamp_s"] = raw[fi].timestamp_s;
      tr["coordinate_space"] = "robot_world";
      tr["quat_order"] = "wxyz";
      SkeletonFrame targets = ee_algo ? ee_algo->last_targets() : SkeletonFrame{};
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
