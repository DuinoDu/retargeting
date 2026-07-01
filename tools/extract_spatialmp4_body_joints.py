#!/usr/bin/env python3
"""Extract Quest/Godot body_joints metadata from a SpatialMP4 capture.

The SpatialMP4 Python binding is used when it is importable in the current
Python. A small HJNT demux fallback is kept here because the binding may be
compiled for a different Python ABI than the GMR environment.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence


HJNT_MAGIC = 0x544E4A48
HJNT_HEADER = struct.Struct("<IHH")
HJNT_JOINT = struct.Struct("<HH8f")


# Godot XRBodyTracker / Meta Quest body-joint ids that should feed the existing
# quest3_upper GMR configuration. The *_UPPER_ARM ids are used as shoulder
# targets because Godot's LEFT/RIGHT_SHOULDER ids are clavicle/scapula points
# and collapse shoulder width for IK.
GODOT_QUEST_UPPER_JOINTS: Dict[str, Sequence[int]] = {
    "Hips": (1,),
    "Chest": (4, 3, 76),
    "Neck": (5,),
    "Head": (6,),
    "LeftShoulder": (9, 8, 77),
    "LeftArmUpper": (9,),
    "LeftArmLower": (10,),
    "LeftWrist": (24, 22, 78),
    "RightShoulder": (12, 11, 79),
    "RightArmUpper": (12,),
    "RightArmLower": (13,),
    "RightWrist": (51, 49, 80),
}

GODOT_QUEST_HAND_JOINTS: Dict[str, Sequence[int]] = {
    "LeftHand": (22,),
    "LeftPalm": (23,),
    "LeftThumbMetacarpal": (25,),
    "LeftThumbProximal": (26,),
    "LeftThumbDistal": (27,),
    "LeftThumbTip": (28,),
    "LeftIndexMetacarpal": (29,),
    "LeftIndexProximal": (30,),
    "LeftIndexIntermediate": (31,),
    "LeftIndexDistal": (32,),
    "LeftIndexTip": (33,),
    "LeftMiddleMetacarpal": (34,),
    "LeftMiddleProximal": (35,),
    "LeftMiddleIntermediate": (36,),
    "LeftMiddleDistal": (37,),
    "LeftMiddleTip": (38,),
    "LeftRingMetacarpal": (39,),
    "LeftRingProximal": (40,),
    "LeftRingIntermediate": (41,),
    "LeftRingDistal": (42,),
    "LeftRingTip": (43,),
    "LeftPinkyMetacarpal": (44,),
    "LeftPinkyProximal": (45,),
    "LeftPinkyIntermediate": (46,),
    "LeftPinkyDistal": (47,),
    "LeftPinkyTip": (48,),
    "RightHand": (49,),
    "RightPalm": (50,),
    "RightThumbMetacarpal": (52,),
    "RightThumbProximal": (53,),
    "RightThumbDistal": (54,),
    "RightThumbTip": (55,),
    "RightIndexMetacarpal": (56,),
    "RightIndexProximal": (57,),
    "RightIndexIntermediate": (58,),
    "RightIndexDistal": (59,),
    "RightIndexTip": (60,),
    "RightMiddleMetacarpal": (61,),
    "RightMiddleProximal": (62,),
    "RightMiddleIntermediate": (63,),
    "RightMiddleDistal": (64,),
    "RightMiddleTip": (65,),
    "RightRingMetacarpal": (66,),
    "RightRingProximal": (67,),
    "RightRingIntermediate": (68,),
    "RightRingDistal": (69,),
    "RightRingTip": (70,),
    "RightPinkyMetacarpal": (71,),
    "RightPinkyProximal": (72,),
    "RightPinkyIntermediate": (73,),
    "RightPinkyDistal": (74,),
    "RightPinkyTip": (75,),
}


@dataclass
class BodyJoint:
    joint_id: int
    flags: int
    radius_m: float
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass
class BodyFrame:
    timestamp_s: float
    track_id: str
    joints: List[BodyJoint]


def run_json(cmd: Sequence[str]) -> dict:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def find_body_stream_index(mp4_path: Path, track_id: str) -> int:
    data = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            str(mp4_path),
        ]
    )
    body_streams: List[tuple[int, str]] = []
    for stream in data.get("streams", []):
        tags = stream.get("tags", {}) if isinstance(stream.get("tags"), dict) else {}
        handler_name = str(tags.get("handler_name", ""))
        if "body_joints" not in handler_name:
            continue
        index = int(stream["index"])
        body_streams.append((index, handler_name))
        if handler_name.endswith(f":{track_id}") or handler_name == track_id:
            return index

    if len(body_streams) == 1:
        return body_streams[0][0]
    if not body_streams:
        raise RuntimeError(f"No body_joints data track found in {mp4_path}")
    available = ", ".join(f"{idx}:{name}" for idx, name in body_streams)
    raise RuntimeError(f"Could not select body track {track_id!r}; available: {available}")


def read_body_frames_with_ffmpeg(mp4_path: Path, track_id: str) -> List[BodyFrame]:
    stream_index = find_body_stream_index(mp4_path, track_id)
    packets = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            str(stream_index),
            "-show_packets",
            "-of",
            "json",
            str(mp4_path),
        ]
    ).get("packets", [])
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(mp4_path),
            "-map",
            f"0:{stream_index}",
            "-c",
            "copy",
            "-f",
            "data",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout

    frames: List[BodyFrame] = []
    offset = 0
    for packet in packets:
        size = int(packet["size"])
        payload = raw[offset : offset + size]
        offset += size
        if len(payload) < HJNT_HEADER.size:
            continue
        magic, version, count = HJNT_HEADER.unpack_from(payload, 0)
        if magic != HJNT_MAGIC or version != 1:
            continue
        expected = HJNT_HEADER.size + count * HJNT_JOINT.size
        if len(payload) < expected:
            raise RuntimeError(
                f"Truncated HJNT packet at {packet.get('pts_time')}: "
                f"got {len(payload)} bytes, expected {expected}"
            )
        joints: List[BodyJoint] = []
        for i in range(count):
            joint_id, flags, radius_m, x, y, z, qx, qy, qz, qw = HJNT_JOINT.unpack_from(
                payload, HJNT_HEADER.size + i * HJNT_JOINT.size
            )
            joints.append(BodyJoint(joint_id, flags, radius_m, x, y, z, qx, qy, qz, qw))
        frames.append(
            BodyFrame(
                timestamp_s=float(packet.get("pts_time", "0")),
                track_id=track_id,
                joints=joints,
            )
        )

    if offset != len(raw):
        raise RuntimeError(f"Packet/raw byte mismatch: consumed {offset}, raw has {len(raw)}")
    return frames


def read_body_frames_with_spatialmp4(mp4_path: Path, track_id: str) -> Optional[List[BodyFrame]]:
    try:
        import spatialmp4  # type: ignore
    except Exception:
        return None

    reader = spatialmp4.Reader(str(mp4_path))
    frames = []
    for frame in reader.get_body_joint_frames(track_id):
        joints = [
            BodyJoint(
                int(j.joint_id),
                int(j.flags),
                float(j.radius_m),
                float(j.x),
                float(j.y),
                float(j.z),
                float(j.qx),
                float(j.qy),
                float(j.qz),
                float(j.qw),
            )
            for j in frame.joints
        ]
        frames.append(BodyFrame(float(frame.timestamp), str(frame.track_id), joints))
    return frames


def read_body_frames(mp4_path: Path, track_id: str) -> tuple[List[BodyFrame], str]:
    frames = read_body_frames_with_spatialmp4(mp4_path, track_id)
    if frames is not None:
        return frames, "spatialmp4-python"
    return read_body_frames_with_ffmpeg(mp4_path, track_id), "hjnt-ffmpeg-fallback"


def mapped_upper_body_joints(frame: BodyFrame) -> dict:
    by_id = {joint.joint_id: joint for joint in frame.joints}
    out = {}
    mapped_joints = {**GODOT_QUEST_UPPER_JOINTS, **GODOT_QUEST_HAND_JOINTS}
    for name, candidates in mapped_joints.items():
        joint = next((by_id[joint_id] for joint_id in candidates if joint_id in by_id), None)
        if joint is None:
            continue
        out[name] = {
            "joint_id": joint.joint_id,
            "flags": joint.flags,
            "position": [joint.x, joint.y, joint.z],
            "rotation": [joint.qx, joint.qy, joint.qz, joint.qw],
        }
    return out


def iter_output_records(frames: Iterable[BodyFrame]) -> Iterator[dict]:
    for frame in frames:
        joints = mapped_upper_body_joints(frame)
        if not joints:
            continue
        yield {
            "timestamp_s": frame.timestamp_s,
            "track_id": frame.track_id,
            "source_joint_set": "godot_xr_body_tracker_quest",
            "raw_joint_count": len(frame.joints),
            "joints": joints,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Quest/Godot body and hand joints from a SpatialMP4 body_joints track.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mp4", type=Path)
    parser.add_argument("--track", default="body")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_frames", type=int, default=0, help="0 means export all frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames, reader_name = read_body_frames(args.mp4, args.track)
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as f:
        for record in iter_output_records(frames):
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
    print(
        f"Extracted {count} mapped body/hand frames from {args.mp4} "
        f"using {reader_name}; wrote {args.output}"
    )


if __name__ == "__main__":
    main()
