#!/usr/bin/env python3
"""Render Galbot dual-arm retargeting as a side-by-side MP4.

Left panel: aligned VR upper-body pose from body.jsonl.
Right panel: retargeted Galbot skeleton from dual_arm_solution.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


JOINT_ORDER = [
    "Hips",
    "Chest",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArmUpper",
    "LeftArmLower",
    "LeftWrist",
    "RightShoulder",
    "RightArmUpper",
    "RightArmLower",
    "RightWrist",
]

EDGES = [
    ("Hips", "Chest"),
    ("Chest", "Neck"),
    ("Neck", "Head"),
    ("Chest", "LeftShoulder"),
    ("LeftShoulder", "LeftArmUpper"),
    ("LeftArmUpper", "LeftArmLower"),
    ("LeftArmLower", "LeftWrist"),
    ("Chest", "RightShoulder"),
    ("RightShoulder", "RightArmUpper"),
    ("RightArmUpper", "RightArmLower"),
    ("RightArmLower", "RightWrist"),
]

ROBOT_BODY_ORDER = [
    "torso_base_link",
    "torso_head_mount_link",
    "head_base_link",
    "head_end_effector_mount_link",
    "torso_left_arm_mount_link",
    "left_arm_base_link",
    "left_arm_link1",
    "left_arm_link2",
    "left_arm_link3",
    "left_arm_link4",
    "left_arm_link5",
    "left_arm_link6",
    "left_arm_link7",
    "left_arm_end_effector_mount_link",
    "left_gripper_tcp_link",
    "torso_right_arm_mount_link",
    "right_arm_base_link",
    "right_arm_link1",
    "right_arm_link2",
    "right_arm_link3",
    "right_arm_link4",
    "right_arm_link5",
    "right_arm_link6",
    "right_arm_link7",
    "right_arm_end_effector_mount_link",
    "right_gripper_tcp_link",
]

ROBOT_EDGES = [
    ("torso_base_link", "torso_head_mount_link"),
    ("torso_head_mount_link", "head_base_link"),
    ("head_base_link", "head_end_effector_mount_link"),
    ("torso_base_link", "torso_left_arm_mount_link"),
    ("torso_left_arm_mount_link", "left_arm_base_link"),
    ("left_arm_base_link", "left_arm_link1"),
    ("left_arm_link1", "left_arm_link2"),
    ("left_arm_link2", "left_arm_link3"),
    ("left_arm_link3", "left_arm_link4"),
    ("left_arm_link4", "left_arm_link5"),
    ("left_arm_link5", "left_arm_link6"),
    ("left_arm_link6", "left_arm_link7"),
    ("left_arm_link7", "left_arm_end_effector_mount_link"),
    ("left_arm_end_effector_mount_link", "left_gripper_tcp_link"),
    ("torso_base_link", "torso_right_arm_mount_link"),
    ("torso_right_arm_mount_link", "right_arm_base_link"),
    ("right_arm_base_link", "right_arm_link1"),
    ("right_arm_link1", "right_arm_link2"),
    ("right_arm_link2", "right_arm_link3"),
    ("right_arm_link3", "right_arm_link4"),
    ("right_arm_link4", "right_arm_link5"),
    ("right_arm_link5", "right_arm_link6"),
    ("right_arm_link6", "right_arm_link7"),
    ("right_arm_link7", "right_arm_end_effector_mount_link"),
    ("right_arm_end_effector_mount_link", "right_gripper_tcp_link"),
]


def openxr_to_gmr(p: np.ndarray) -> np.ndarray:
    return np.array([-p[2], -p[0], p[1]], dtype=float)


def load_body_frames(path: Path) -> Tuple[List[float], List[Dict[str, np.ndarray]]]:
    timestamps: List[float] = []
    frames: List[Dict[str, np.ndarray]] = []
    with path.open() as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            joints = {}
            for name, item in rec["joints"].items():
                joints[name] = np.asarray(item["position"], dtype=float)
            timestamps.append(float(rec.get("timestamp_s", idx / 30.0)))
            frames.append(joints)
    return timestamps, frames


def load_solution(path: Path) -> Tuple[List[Optional[float]], List[str], List[np.ndarray]]:
    timestamps: List[Optional[float]] = []
    joint_names: List[str] = []
    qs: List[np.ndarray] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not joint_names:
                joint_names = list(rec["joint_names"])
            if "timestamp_ns" in rec:
                timestamps.append(float(rec["timestamp_ns"]) * 1e-9)
            elif "timestamp_s" in rec:
                timestamps.append(float(rec["timestamp_s"]))
            else:
                timestamps.append(None)
            qs.append(np.asarray(rec["joint_q"], dtype=float))
    return timestamps, joint_names, qs


def infer_fps(timestamps: Sequence[Optional[float]]) -> float:
    valid = [t for t in timestamps if t is not None]
    if len(valid) == len(timestamps) and len(valid) > 1:
        duration = valid[-1] - valid[0]
        if duration > 0:
            return (len(valid) - 1) / duration
    return 30.0


def heading_yaw(frames: Sequence[Dict[str, np.ndarray]]) -> float:
    if not frames:
        return 0.0
    first = frames[0]
    if "LeftShoulder" not in first or "RightShoulder" not in first:
        return 0.0
    side = openxr_to_gmr(first["LeftShoulder"]) - openxr_to_gmr(first["RightShoulder"])
    side[2] = 0.0
    n = np.linalg.norm(side)
    if n < 1e-5:
        return 0.0
    return math.pi / 2.0 - math.atan2(float(side[1]), float(side[0]))


def aligned_body_frames(
    raw_frames: Sequence[Dict[str, np.ndarray]], pelvis_height: float
) -> List[Dict[str, np.ndarray]]:
    yaw = heading_yaw(raw_frames)
    cz, sz = math.cos(yaw), math.sin(yaw)
    out: List[Dict[str, np.ndarray]] = []
    for frame in raw_frames:
        if "Hips" not in frame:
            out.append({})
            continue
        root = openxr_to_gmr(frame["Hips"])
        aligned: Dict[str, np.ndarray] = {}
        for name, pos in frame.items():
            r = openxr_to_gmr(pos) - root
            aligned[name] = np.array(
                [r[0] * cz - r[1] * sz, r[0] * sz + r[1] * cz, r[2] + pelvis_height],
                dtype=float,
            )
        out.append(aligned)
    return out


def fit_yz_bounds(
    frames: Sequence[Dict[str, np.ndarray]], names: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    pts = []
    for frame in frames:
        for name in names:
            if name in frame:
                p = frame[name]
                pts.append([p[1], p[2]])
    if not pts:
        return np.array([-0.8, 0.5]), np.array([0.8, 2.1])
    arr = np.asarray(pts, dtype=float)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    span = np.maximum(hi - lo, 0.25)
    return lo - 0.15 * span, hi + 0.15 * span


def draw_line(img: np.ndarray, p0: Tuple[int, int], p1: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < img.shape[1] and 0 <= y0 < img.shape[0]:
            img[max(0, y0 - 1) : min(img.shape[0], y0 + 2), max(0, x0 - 1) : min(img.shape[1], x0 + 2)] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_circle(img: np.ndarray, center: Tuple[int, int], radius: int, color: Tuple[int, int, int]) -> None:
    cx, cy = center
    y0, y1 = max(0, cy - radius), min(img.shape[0], cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(img.shape[1], cx + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    img[y0:y1, x0:x1][mask] = color


def side_color(name: str) -> Tuple[int, int, int]:
    lower = name.lower()
    if lower.startswith("left"):
        return (86, 170, 255)
    if lower.startswith("right"):
        return (98, 220, 152)
    return (230, 232, 235)


def edge_color(a: str, b: str) -> Tuple[int, int, int]:
    la = a.lower()
    lb = b.lower()
    if la.startswith("left") or lb.startswith("left"):
        return (86, 170, 255)
    if la.startswith("right") or lb.startswith("right"):
        return (98, 220, 152)
    return (210, 214, 220)


def render_skeleton_panel(
    frame: Dict[str, np.ndarray],
    width: int,
    height: int,
    lo: np.ndarray,
    hi: np.ndarray,
    names: Sequence[str],
    edges: Sequence[Tuple[str, str]],
) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = np.array([18, 21, 25], dtype=np.uint8)
    margin = 42
    span = np.maximum(hi - lo, 1e-6)

    def project(pos: np.ndarray) -> Tuple[int, int]:
        yz = np.array([pos[1], pos[2]], dtype=float)
        u = (yz[0] - lo[0]) / span[0]
        v = (yz[1] - lo[1]) / span[1]
        x = int(round(margin + u * (width - 2 * margin)))
        y = int(round(height - margin - v * (height - 2 * margin)))
        return x, y

    for gx in np.linspace(lo[0], hi[0], 5):
        x, _ = project(np.array([0.0, gx, lo[1]]))
        img[:, max(0, x) : min(width, x + 1)] = [32, 36, 42]
    for gz in np.linspace(lo[1], hi[1], 5):
        _, y = project(np.array([0.0, lo[0], gz]))
        img[max(0, y) : min(height, y + 1), :] = [32, 36, 42]

    for a, b in edges:
        if a not in frame or b not in frame:
            continue
        draw_line(img, project(frame[a]), project(frame[b]), edge_color(a, b))

    for name in names:
        if name not in frame:
            continue
        draw_circle(img, project(frame[name]), 5, side_color(name))
    return img


def render_vr_panel(
    frame: Dict[str, np.ndarray],
    width: int,
    height: int,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    return render_skeleton_panel(frame, width, height, lo, hi, JOINT_ORDER, EDGES)


def render_robot_panel(
    frame: Dict[str, np.ndarray],
    width: int,
    height: int,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    return render_skeleton_panel(frame, width, height, lo, hi, ROBOT_BODY_ORDER, ROBOT_EDGES)


def collect_robot_frames(
    model,
    data,
    base_qpos: np.ndarray,
    qpos_indices: Sequence[int],
    qs: Sequence[np.ndarray],
    n: int,
) -> List[Dict[str, np.ndarray]]:
    import mujoco

    body_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ROBOT_BODY_ORDER
    }
    missing = [name for name, body_id in body_ids.items() if body_id < 0]
    if missing:
        sys.exit(f"missing robot bodies for skeleton render: {', '.join(missing)}")

    frames: List[Dict[str, np.ndarray]] = []
    for i in range(n):
        qpos = base_qpos.copy()
        for qi, q in zip(qpos_indices, qs[i]):
            qpos[qi] = q
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        frames.append({name: data.xpos[body_id].copy() for name, body_id in body_ids.items()})
    return frames


def iter_frames(args: argparse.Namespace) -> Iterable[np.ndarray]:
    import mujoco

    body_timestamps, raw_body = load_body_frames(args.body_jsonl)
    sol_timestamps, joint_names, qs = load_solution(args.solution_jsonl)
    if not raw_body:
        sys.exit(f"no frames in {args.body_jsonl}")
    if not qs:
        sys.exit(f"no frames in {args.solution_jsonl}")

    body = aligned_body_frames(raw_body, args.pelvis_height)
    lo, hi = fit_yz_bounds(body, JOINT_ORDER)
    n = min(len(body), len(qs))
    if args.max_frames > 0:
        n = min(n, args.max_frames)

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    base_qpos = data.qpos.copy()
    qpos_indices = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in joint_names
    ]
    robot_frames = collect_robot_frames(model, data, base_qpos, qpos_indices, qs, n)
    robot_lo, robot_hi = fit_yz_bounds(robot_frames, ROBOT_BODY_ORDER)

    for i in range(n):
        left = render_vr_panel(body[i], args.width, args.height, lo, hi)
        right = render_robot_panel(robot_frames[i], args.width, args.height, robot_lo, robot_hi)
        separator = np.full((args.height, 4, 3), 28, dtype=np.uint8)
        yield np.concatenate([left, separator, right], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--body_jsonl", type=Path, required=True)
    ap.add_argument("--solution_jsonl", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--pelvis_height", type=float, default=0.88)
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    import imageio.v2 as imageio

    sol_timestamps, _, _ = load_solution(args.solution_jsonl)
    fps = args.fps if args.fps is not None else infer_fps(sol_timestamps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with imageio.get_writer(args.out, fps=fps, macro_block_size=1) as writer:
        for frame in iter_frames(args):
            writer.append_data(frame)
            count += 1
    print(
        f"wrote {count} frames -> {args.out} "
        f"({args.width * 2 + 4}x{args.height} @ {fps:.3f}fps)"
    )


if __name__ == "__main__":
    main()
