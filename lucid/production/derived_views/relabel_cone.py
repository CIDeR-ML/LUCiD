#!/usr/bin/env python3
"""Derived-view demo: relabel each hit PMT as inside/outside the Cherenkov cone.

Takes a dataset, and WITHOUT re-simulating, recomputes a per-hit label from the
stored truth (interaction vertex + primary-track direction + sensor positions):

    angle(PMT) = angle between (sensor_pos - vertex) and primary track direction
    label = 1 (inside cone)  if angle <= theta_cut   else 0 (outside)

The label is written into the `emission_process` column of hits/ and step/sensor_hits/
so the existing viewer renders it via its emission toggle; dark-noise rows keep their
code (2). Same events, different theta_cut -> a different downstream label definition.
No propagation is rerun. Every batch of SRC is relabelled.

Usage: relabel_cone.py SRC_DIR DST_DIR THETA_DEG
"""

import os
import shutil
import sys

import h5py
import numpy as np

from lucid.dataset.layout import batch_paths, list_batches
from lucid.dataset.schema import EMISSION_PROCESS_DARK


def event_keys(path):
    with h5py.File(path, "r") as h:
        return [k for k in h.keys() if k.startswith("event_")]


def event_axis(labl_f, step_f, ev):
    """Vertex and primary direction of one event, from labl/step truth."""
    with h5py.File(labl_f, "r") as h:
        pi = h[ev]["per_interaction"]
        vertex = np.array([pi["vertex_x"][0], pi["vertex_y"][0], pi["vertex_z"][0]], np.float64)
        parent = h[ev]["per_track"]["parent_id"][:]
    primary_rows = np.where(parent == 0)[0]
    with h5py.File(step_f, "r") as h:
        g = h[ev]
        tidx = g["track_idx"][:]
        dirs = np.stack([g["dir_x"][:], g["dir_y"][:], g["dir_z"][:]], axis=1).astype(np.float64)
        tseg = g["time"][:]
    mask = np.isin(tidx, primary_rows)
    if mask.any():
        cand = np.where(mask)[0]
        d = dirs[cand[np.argmin(tseg[cand])]]  # earliest primary segment direction
    else:
        d = dirs.mean(axis=0)  # fallback
    n = np.linalg.norm(d)
    d = d / n if n > 0 else np.array([0.0, 0.0, 1.0])
    return vertex, d


def cone_labels(sensor_pos, sensor_idx, vertex, axis, cos_cut):
    v = sensor_pos[sensor_idx] - vertex[None, :]
    vn = np.linalg.norm(v, axis=1)
    cos = np.where(vn > 0, (v @ axis) / np.maximum(vn, 1e-12), 0.0)
    return (cos >= cos_cut).astype(np.int8)  # 1 = inside cone


def relabel_batch(paths, cos_cut):
    """Rewrite one batch in place. Returns (events, light hits inside the cone, light hits)."""
    with h5py.File(paths["sensor"], "r") as h:
        sensor_pos = h["config/sensor_positions"][:].astype(np.float64)  # (n_sensors, 3) m
    evs = event_keys(paths["hits"])
    tot_in = tot = 0
    for ev in evs:
        vertex, axis = event_axis(paths["labl"], paths["step"], ev)
        for sub in ("hits", "step"):
            with h5py.File(paths[sub], "r+") as h:
                grp = h[ev]["sensor_hits"] if sub == "step" else h[ev]
                old = grp["emission_process"][:]
                light = old != EMISSION_PROCESS_DARK  # dark noise is not light from the vertex
                lab = np.where(
                    light, cone_labels(sensor_pos, grp["sensor_idx"][:], vertex, axis, cos_cut), old
                ).astype(old.dtype)
                grp["emission_process"][...] = lab
                if sub == "hits":
                    tot_in += int(lab[light].sum())
                    tot += int(light.sum())
    return len(evs), tot_in, tot


def main():
    src, dst, theta_deg = sys.argv[1], sys.argv[2], float(sys.argv[3])
    cos_cut = np.cos(np.deg2rad(theta_deg))
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    n_events = tot_in = tot = 0
    for batch in list_batches(dst):
        n, i, t = relabel_batch(batch_paths(dst, batch), cos_cut)
        n_events += n
        tot_in += i
        tot += t
    print(
        f"[theta={theta_deg:.0f}deg] {n_events} events; light hits inside cone: "
        f"{tot_in}/{tot} ({100 * tot_in / max(tot, 1):.1f}%)"
    )
    print(f"[theta={theta_deg:.0f}deg] wrote {dst}")


if __name__ == "__main__":
    main()
