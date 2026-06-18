#!/usr/bin/env python3
"""Compare Godot's runtime GLB link rest transforms against the raw glTF matrices.

The G1 overlay (debug_dump_glb_rest) writes user://g1_glb_rest.json:
  { link_name: {"parent": str, "cols": [9 floats], "origin": [3]} }
where `cols` are the three COLUMN vectors of the Godot node's local basis
(local->parent) and `origin` is its local translation.

This loads the same GLB with trimesh (the raw glTF node matrices the FK math was
validated against, == MuJoCo to 0) and reports, per link, the basis-angle and
origin difference between Godot's imported transform and the glTF matrix. A
nonzero diff on the torso/waist chain explains why a known-good qpos renders
tilted on-device while it renders upright in MuJoCo/trimesh.

Usage:
  compare_glb_rest.py g1_glb_rest.json /path/to/g1_29dof.glb [g1_mocap_29dof_nomesh.xml]
"""
import argparse
import json
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def basis_angle_deg(Ra, Rb):
    R = Ra.T @ Rb
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("godot_json")
    ap.add_argument("glb")
    ap.add_argument("mjcf", nargs="?", help="MJCF to cross-check the runtime joint table")
    args = ap.parse_args()

    doc = json.load(open(args.godot_json))
    # New dump: {"links": {...}, "joints": [...]}; old dump: flat link dict.
    godot = doc.get("links", doc)
    joints = doc.get("joints", [])

    # raw glTF local matrices via trimesh
    s = trimesh.load(args.glb, process=False)
    tf = s.graph.transforms
    gltf_local = {}
    gltf_parent = {}
    for (u, v), d in tf.edge_data.items():
        if u == v:
            continue
        m = d.get("matrix")
        gltf_local[v] = np.array(m, float) if m is not None else np.eye(4)
        gltf_parent[v] = u

    rows = []
    for name, rec in godot.items():
        if name not in gltf_local:
            rows.append((name, None, None, "not in glTF graph"))
            continue
        c = rec["cols"]
        M = np.array([[c[0], c[3], c[6]],
                      [c[1], c[4], c[7]],
                      [c[2], c[5], c[8]]], float)  # columns = basis vectors
        o = np.array(rec["origin"], float)
        gl = gltf_local[name]
        ang = basis_angle_deg(M, gl[:3, :3])
        doff = float(np.linalg.norm(o - gl[:3, 3]))
        pnote = ""
        if rec.get("parent", "") != gltf_parent.get(name, ""):
            pnote = " PARENT godot=%s gltf=%s" % (rec.get("parent"), gltf_parent.get(name))
        rows.append((name, ang, doff, pnote))

    rows.sort(key=lambda r: (-(r[1] or 0), -(r[2] or 0)))
    print(f"{'link':28s} {'basis_diff_deg':>14s} {'origin_diff_m':>13s}  note")
    for name, ang, doff, note in rows:
        if ang is None:
            print(f"{name:28s} {'--':>14s} {'--':>13s}  {note}")
        else:
            print(f"{name:28s} {ang:14.4f} {doff:13.5f}  {note}")
    bad = [r for r in rows if r[1] is not None and (r[1] > 0.5 or r[2] > 0.002)]
    print(f"\n{len(bad)} link(s) differ beyond tolerance "
          f"(basis>0.5deg or origin>2mm) -> Godot import != glTF where listed.")

    # Cross-check the runtime joint table (name, body, axis in qpos order) vs the
    # MJCF, to rule out a joint-order / body / axis mismatch in _parse_mocap_joints.
    if joints and args.mjcf:
        ref = []
        root = ET.fromstring(open(args.mjcf).read())

        def walk(el, body):
            for ch in el:
                if ch.tag == "body":
                    walk(ch, ch.get("name"))
                elif ch.tag == "joint" and ch.get("type", "hinge") not in ("free", "ball"):
                    ax = [float(v) for v in ch.get("axis", "0 0 1").split()]
                    ref.append((ch.get("name"), body, ax)); walk(ch, body)
                else:
                    walk(ch, body)
        walk(root.find("worldbody"), None)
        print(f"\njoint table: runtime={len(joints)}  mjcf={len(ref)}")
        mism = 0
        for i, jr in enumerate(joints):
            if i >= len(ref):
                break
            rn, rb, rax = ref[i]
            ax = jr["axis"]
            if jr["name"] != rn or jr["body"] != rb or any(abs(a - b) > 1e-6 for a, b in zip(ax, rax)):
                mism += 1
                print(f"  [{i}] runtime {jr['name']}/{jr['body']}/{ax}  !=  mjcf {rn}/{rb}/{rax}")
        print(f"joint-table mismatches: {mism} (0 = runtime parse matches MJCF qpos order)")


if __name__ == "__main__":
    main()
