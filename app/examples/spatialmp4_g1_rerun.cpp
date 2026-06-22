// Log normalized SpatialMP4 VR pose and retargeted G1 pose to Rerun.
//
// The recording contains two independent 3D roots:
//   /vr_pose
//   /retargeted_robot
//
// Usage:
//   spatialmp4_g1_rerun <normalized_gmr.jsonl> <robot_solution.jsonl>
//       --model <g1_mocap_29dof.xml> [--out <recording.rrd>] [--spawn] [--connect]
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>
#include <nlohmann/json.hpp>
#include <rerun.hpp>

using json = nlohmann::json;

namespace {

const std::vector<std::pair<std::string, std::string>> kVrBones = {
    {"Hips", "Chest"},
    {"Chest", "Neck"},
    {"Neck", "Head"},
    {"Chest", "LeftShoulder"},
    {"LeftShoulder", "LeftArmUpper"},
    {"LeftArmUpper", "LeftArmLower"},
    {"LeftArmLower", "LeftWrist"},
    {"Chest", "RightShoulder"},
    {"RightShoulder", "RightArmUpper"},
    {"RightArmUpper", "RightArmLower"},
    {"RightArmLower", "RightWrist"},
};

struct VrFrame {
  double timestamp_s = 0.0;
  std::map<std::string, rerun::Position3D> joints;
};

struct RobotFrame {
  double timestamp_s = 0.0;
  std::vector<double> joint_q;
};

bool is_left_joint(const std::string& name) {
  return name.rfind("Left", 0) == 0;
}

bool is_right_joint(const std::string& name) {
  return name.rfind("Right", 0) == 0;
}

rerun::Color vr_color(const std::string& name) {
  if (is_left_joint(name)) return rerun::Color(44, 128, 188);
  if (is_right_joint(name)) return rerun::Color(207, 96, 84);
  return rerun::Color(58, 64, 74);
}

std::vector<VrFrame> load_vr_frames(const std::string& path) {
  std::vector<VrFrame> frames;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    json rec = json::parse(line);
    VrFrame frame;
    frame.timestamp_s = rec.value("timestamp_s", static_cast<double>(frames.size()) / 30.0);
    for (const auto& [name, value] : rec["joints"].items()) {
      const auto& p = value["position"];
      frame.joints.emplace(
          name,
          rerun::Position3D{
              p[0].get<float>(),
              p[1].get<float>(),
              p[2].get<float>(),
          });
    }
    frames.push_back(std::move(frame));
  }
  return frames;
}

std::vector<RobotFrame> load_robot_frames(const std::string& path) {
  std::vector<RobotFrame> frames;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    json rec = json::parse(line);
    RobotFrame frame;
    frame.timestamp_s = static_cast<double>(rec.value("timestamp_ns", 0LL)) * 1e-9;
    frame.joint_q = rec["joint_q"].get<std::vector<double>>();
    frames.push_back(std::move(frame));
  }
  return frames;
}

void log_vr_frame(const rerun::RecordingStream& rec, const VrFrame& frame) {
  std::vector<rerun::Position3D> points;
  std::vector<rerun::Color> colors;
  for (const auto& [name, point] : frame.joints) {
    points.push_back(point);
    colors.push_back(vr_color(name));
  }
  rec.log(
      "/vr_pose/joints",
      rerun::Points3D(points)
          .with_colors(colors)
          .with_radii(rerun::Radius(0.025f)));

  std::vector<rerun::LineStrip3D> strips;
  std::vector<rerun::Color> strip_colors;
  for (const auto& [a, b] : kVrBones) {
    auto ia = frame.joints.find(a);
    auto ib = frame.joints.find(b);
    if (ia == frame.joints.end() || ib == frame.joints.end()) continue;
    strips.emplace_back(rerun::Collection<rerun::Vec3D>{ia->second, ib->second});
    if (is_left_joint(a) || is_left_joint(b)) {
      strip_colors.emplace_back(44, 128, 188);
    } else if (is_right_joint(a) || is_right_joint(b)) {
      strip_colors.emplace_back(207, 96, 84);
    } else {
      strip_colors.emplace_back(58, 64, 74);
    }
  }
  if (!strips.empty()) {
    rec.log(
        "/vr_pose/bones",
        rerun::LineStrips3D(strips)
            .with_colors(strip_colors)
            .with_radii(rerun::Radius(0.012f)));
  }
}

class MujocoRobotPose {
 public:
  explicit MujocoRobotPose(const std::string& model_path) {
    char error[1000] = "";
    model_ = mj_loadXML(model_path.c_str(), nullptr, error, sizeof(error));
    if (!model_) {
      throw std::runtime_error(std::string("mj_loadXML failed: ") + error);
    }
    data_ = mj_makeData(model_);
    mj_resetData(model_, data_);
    base_qpos_.assign(model_->qpos0, model_->qpos0 + model_->nq);
    for (int body_id = 1; body_id < model_->nbody; ++body_id) {
      const int parent = model_->body_parentid[body_id];
      if (parent > 0) {
        body_edges_.emplace_back(parent, body_id);
      }
    }
  }

  ~MujocoRobotPose() {
    if (data_) mj_deleteData(data_);
    if (model_) mj_deleteModel(model_);
  }

  std::vector<rerun::Position3D> body_positions(const std::vector<double>& joint_q) {
    std::vector<double> qpos = base_qpos_;
    const int n = std::min<int>(model_->nq - 7, static_cast<int>(joint_q.size()));
    for (int i = 0; i < n; ++i) {
      qpos[7 + i] = joint_q[i];
    }
    mju_copy(data_->qpos, qpos.data(), model_->nq);
    mj_forward(model_, data_);

    std::vector<rerun::Position3D> positions;
    positions.reserve(model_->nbody);
    for (int body_id = 0; body_id < model_->nbody; ++body_id) {
      const double* p = data_->xpos + 3 * body_id;
      positions.emplace_back(
          static_cast<float>(p[0]),
          static_cast<float>(p[1]),
          static_cast<float>(p[2]));
    }
    return positions;
  }

  const std::vector<std::pair<int, int>>& body_edges() const {
    return body_edges_;
  }

 private:
  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  std::vector<double> base_qpos_;
  std::vector<std::pair<int, int>> body_edges_;
};

void log_robot_frame(
    const rerun::RecordingStream& rec,
    const std::vector<rerun::Position3D>& body_positions,
    const std::vector<std::pair<int, int>>& body_edges) {
  if (body_positions.size() > 1) {
    std::vector<rerun::Position3D> body_points(body_positions.begin() + 1, body_positions.end());
    rec.log(
        "/retargeted_robot/body_points",
        rerun::Points3D(body_points)
            .with_colors(rerun::Color(210, 210, 210))
            .with_radii(rerun::Radius(0.018f)));
  }

  std::vector<rerun::LineStrip3D> strips;
  strips.reserve(body_edges.size());
  for (const auto& [parent, child] : body_edges) {
    if (parent >= static_cast<int>(body_positions.size()) ||
        child >= static_cast<int>(body_positions.size())) {
      continue;
    }
    strips.emplace_back(
        rerun::Collection<rerun::Vec3D>{body_positions[parent], body_positions[child]});
  }
  if (!strips.empty()) {
    rec.log(
        "/retargeted_robot/body_graph",
        rerun::LineStrips3D(strips)
            .with_colors(rerun::Color(90, 160, 230))
            .with_radii(rerun::Radius(0.010f)));
  }
}

struct Args {
  std::string normalized_jsonl;
  std::string solution_jsonl;
  std::string model;
  std::string out;
  bool spawn = false;
  bool connect = false;
  int max_frames = 0;
};

Args parse_args(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr,
            "usage: %s <normalized_gmr.jsonl> <robot_solution.jsonl> "
            "--model <g1.xml> [--out <recording.rrd>] [--spawn] [--connect]\n",
            argv[0]);
    std::exit(2);
  }
  Args args;
  args.normalized_jsonl = argv[1];
  args.solution_jsonl = argv[2];
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") args.model = next();
    else if (a == "--out") args.out = next();
    else if (a == "--spawn") args.spawn = true;
    else if (a == "--connect") args.connect = true;
    else if (a == "--max_frames") args.max_frames = std::stoi(next());
    else {
      fprintf(stderr, "unknown argument: %s\n", a.c_str());
      std::exit(2);
    }
  }
  if (args.model.empty()) {
    fprintf(stderr, "--model is required\n");
    std::exit(2);
  }
  if (args.out.empty() && !args.spawn && !args.connect) {
    fprintf(stderr, "one of --out, --spawn, or --connect is required\n");
    std::exit(2);
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  auto vr_frames = load_vr_frames(args.normalized_jsonl);
  auto robot_frames = load_robot_frames(args.solution_jsonl);
  size_t n_frames = std::min(vr_frames.size(), robot_frames.size());
  if (args.max_frames > 0) {
    n_frames = std::min(n_frames, static_cast<size_t>(args.max_frames));
  }
  if (n_frames == 0) {
    fprintf(stderr, "no frames to log\n");
    return 2;
  }

  rerun::RecordingStream rec("spatialmp4_gmr_retargeting");
  if (!args.out.empty()) {
    rec.save(args.out).exit_on_failure();
  }
  if (args.spawn) {
    rec.spawn().exit_on_failure();
  }
  if (args.connect) {
    rec.connect_grpc().exit_on_failure();
  }

  rec.log_static("/vr_pose", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);
  rec.log_static("/retargeted_robot", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);

  MujocoRobotPose robot_pose(args.model);
  for (size_t i = 0; i < n_frames; ++i) {
    rec.set_time_sequence("frame", static_cast<int64_t>(i));
    rec.set_time_duration_secs("time", vr_frames[i].timestamp_s);
    log_vr_frame(rec, vr_frames[i]);
    log_robot_frame(rec, robot_pose.body_positions(robot_frames[i].joint_q), robot_pose.body_edges());
  }

  rec.flush_blocking().exit_on_failure();
  if (!args.out.empty()) {
    printf("wrote %zu frames -> %s\n", n_frames, args.out.c_str());
  } else {
    printf("logged %zu frames to Rerun\n", n_frames);
  }
  return 0;
}
