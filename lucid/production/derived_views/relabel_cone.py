#!/usr/bin/env python3
"""Derived-view demo: relabel each hit PMT as inside/outside the Cherenkov cone.

Takes a dataset, and WITHOUT re-simulating, recomputes a per-hit label from the
stored truth (interaction vertex + primary-track direction + sensor positions):

    angle(PMT) = angle between (sensor_pos - vertex) and primary track direction
    label = 1 (inside cone)  if angle <= theta_cut   else 0 (outside)

The label is written into the `emission_process` column of hits/ and step/sensor_hits/
so the existing viewer renders it via its emission toggle. Same events, different
theta_cut -> a different downstream label definition. No propagation is rerun.

Usage: relabel_cone.py SRC_DIR DST_DIR THETA_DEG
"""
import sys, os, shutil, glob
import numpy as np
import h5py

src, dst, theta_deg = sys.argv[1], sys.argv[2], float(sys.argv[3])
cos_cut = np.cos(np.deg2rad(theta_deg))

if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)

def first(name):
    return sorted(glob.glob(os.path.join(dst, name)))[0]

sensor_f = first("sensor/*.h5")
hits_f   = first("hits/*.h5")
edep_f   = first("step/*.h5")
labl_f   = first("labl/*.h5")

with h5py.File(sensor_f, "r") as h:
    sensor_pos = h["config/sensor_positions"][:].astype(np.float64)   # (n_sensors, 3) meters

def event_keys(path):
    with h5py.File(path, "r") as h:
        return [k for k in h.keys() if k.startswith("event_")]

evs = event_keys(hits_f)
print(f"[theta={theta_deg:.0f}deg] {len(evs)} events; sensors={sensor_pos.shape[0]}")

# Per-event vertex + primary direction from labl/edep truth.
def event_axis(ev):
    with h5py.File(labl_f, "r") as h:
        pi = h[ev]["per_interaction"]
        vertex = np.array([pi["vertex_x"][0], pi["vertex_y"][0], pi["vertex_z"][0]], np.float64)
        pt = h[ev]["per_track"]
        parent = pt["parent_id"][:]
    primary_rows = np.where(parent == 0)[0]
    with h5py.File(edep_f, "r") as h:
        g = h[ev]
        tidx = g["track_idx"][:]
        dirs = np.stack([g["dir_x"][:], g["dir_y"][:], g["dir_z"][:]], axis=1).astype(np.float64)
        tseg = g["time"][:]
    mask = np.isin(tidx, primary_rows)
    if mask.any():
        cand = np.where(mask)[0]
        d = dirs[cand[np.argmin(tseg[cand])]]            # earliest primary segment direction
    else:
        d = dirs.mean(axis=0)                             # fallback
    n = np.linalg.norm(d)
    d = d / n if n > 0 else np.array([0., 0., 1.])
    return vertex, d

def cone_labels(sensor_idx, vertex, axis):
    v = sensor_pos[sensor_idx] - vertex[None, :]
    vn = np.linalg.norm(v, axis=1)
    cos = np.where(vn > 0, (v @ axis) / np.maximum(vn, 1e-12), 0.0)
    return (cos >= cos_cut).astype(np.int8)              # 1 = inside cone

tot_in = tot = 0
for ev in evs:
    vertex, axis = event_axis(ev)
    for path in (hits_f, edep_f):
        with h5py.File(path, "r+") as h:
            grp = h[ev]["sensor_hits"] if path == edep_f else h[ev]
            sidx = grp["sensor_idx"][:]
            lab = cone_labels(sidx, vertex, axis)
            grp["emission_process"][...] = lab
            if path == hits_f:
                tot_in += int(lab.sum()); tot += lab.size
print(f"[theta={theta_deg:.0f}deg] hits inside cone: {tot_in}/{tot} ({100*tot_in/max(tot,1):.1f}%)")
print(f"[theta={theta_deg:.0f}deg] wrote {dst}")
