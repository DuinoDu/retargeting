#!/usr/bin/env python3
"""Analyze a g1_debug.jsonl captured by the in-headset G1 overlay (#1 dump).

The overlay writes, per frame:
  canon : raw canonical joints per GMR slot {name: {p:[xyz], q:[xyzw], matched}}
  src   : the GMR source pose actually fed to the solver {name: {gmr_pos, gmr_quat}}
  qpos  : the full 36-entry qpos the native (on-device) retargeter returned
  heading_deg

This splits the on-device pipeline into offline-comparable artifacts:

  1. ondevice_qpos.jsonl  — the on-device solver output, ready for render_g1.py.
       Render it: if it looks CORRECT but the headset looked wrong, the bug is
       downstream in the GLB FK / render. If it looks WRONG, the native solver
       produced bad qpos from the live input (input convention or solver).

  2. body_from_dump.jsonl — the raw canonical joints in the offline tool's
       body.jsonl format. Feed it to spatialmp4_to_g1 and compare its qpos to
       the on-device qpos: a mismatch isolates the native solver / input-prep;
       a match means the on-device qpos is trustworthy and the bug is in render.

Usage:
  analyze_g1_debug.py g1_debug.jsonl --out_dir build/g1_debug
"""
import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out_dir", default="build/g1_debug")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    recs = [json.loads(l) for l in open(args.dump) if l.strip()]
    if not recs:
        raise SystemExit("empty dump")

    ondevice = os.path.join(args.out_dir, "ondevice_qpos.jsonl")
    body = os.path.join(args.out_dir, "body_from_dump.jsonl")

    headings = [r.get("heading_deg") for r in recs if "heading_deg" in r]
    n_q = n_b = 0
    with open(ondevice, "w") as fq, open(body, "w") as fb:
        for i, r in enumerate(recs):
            qpos = r.get("qpos", [])
            if len(qpos) >= 36:
                fq.write(json.dumps({"joint_q": qpos[7:36]}) + "\n")
                n_q += 1
            canon = r.get("canon", {})
            if canon:
                joints = {
                    name: {"position": j["p"], "rotation": j["q"]}
                    for name, j in canon.items()
                    if "p" in j and "q" in j
                }
                fb.write(json.dumps({"timestamp_s": i / args.fps, "joints": joints}) + "\n")
                n_b += 1

    print(f"frames: {len(recs)}")
    if headings:
        h0 = headings[0]
        spread = max(headings) - min(headings)
        print(f"heading_deg: first={h0:.2f}  spread={spread:.4f} (should be ~0 — it's latched once)")
    print(f"wrote {n_q} on-device qpos -> {ondevice}")
    print(f"wrote {n_b} canonical body frames -> {body}")
    print()
    print("Next (from the retargeting repo root):")
    print("  # A) render what the ON-DEVICE solver produced:")
    print(f"  $RENDER_PY tools/render_g1.py {ondevice} \\")
    print("      --model $GMR/assets/unitree_g1/g1_mocap_29dof.xml --out build/g1_debug/ondevice.mp4")
    print("  # B) re-solve the same canonical input offline + compare to on-device qpos:")
    print(f"  ./build/spatialmp4_to_g1 {body} --save_jsonl build/g1_debug/offline_qpos.jsonl \\")
    print("      --robot_xml data/robot/unitree_g1/g1_mocap_29dof_nomesh.xml \\")
    print("      --ik_config data/ik_configs/quest3_upper_to_g1.json \\")
    print(f"      --compare {ondevice}")


if __name__ == "__main__":
    main()
