#!/usr/bin/env python3
"""Genuine segment-level downstream re-categorization (ROOT-FREE).

Redefines what a 'particle' is, purely from the persisted dataset (edep segment
geometry + per_track ancestry). A particle = a connected group of meaningful tracks
under this rule, walked child->parent:

  * EM children (e+-, gamma)         -> absorbed into the parent's particle
  * hadron children (pi/p/n/...)     -> SAME particle as parent IF the junction
                                        deflection angle < THETA, else a NEW particle
  * primary (parent_id==0)           -> root of its own particle

THETA is the tunable "number of degrees" knob. Small theta -> a pion splits into
many particles at every kink; large theta -> the cascade stays one particle.
The number of particles changes with theta -> per_particle is rebuilt to match.

Rewrites: labl/per_track/particle_idx, labl/per_particle (full rebuild),
and re-aggregates hits/ from step/sensor_hits under the new definition.

Usage: relabel_segments.py SRC DST THETA_DEG
"""
import sys, os, shutil, glob
import numpy as np
import h5py

EMISSION_PROCESS_DARK = 2   # schema constant (lucid.sources.writer): dark-noise hit

src, dst, theta = sys.argv[1], sys.argv[2], float(sys.argv[3])
cos_thr = np.cos(np.deg2rad(theta))
EM = {11, -11, 22}

if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
labl_f = sorted(glob.glob(os.path.join(dst, "labl/*.h5")))[0]
edep_f = sorted(glob.glob(os.path.join(dst, "step/*.h5")))[0]
hits_f = sorted(glob.glob(os.path.join(dst, "hits/*.h5")))[0]

class UF:
    def __init__(s, n): s.p = list(range(n))
    def find(s, x):
        while s.p[x] != x: s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def union(s, a, b): s.p[s.find(a)] = s.find(b)

with h5py.File(labl_f, "r") as h:
    evs = [k for k in h if k.startswith("event_")]

counts = []
for ev in evs:
    with h5py.File(edep_f, "r") as h:
        g = h[ev]
        seg_track = g["track_idx"][:]                     # per-segment -> per_track row
        seg_dir = np.stack([g["dir_x"][:], g["dir_y"][:], g["dir_z"][:]], 1).astype(np.float64)
        seg_time = g["time"][:]
        seg_contained = g["contained"][:]
        sh = g["sensor_hits"]
        sh_seg = sh["segment_idx"][:]; sh_sensor = sh["sensor_idx"][:]
        sh_pe = sh["PE"][:]; sh_t = sh["T"][:]
        sh_treco = sh["T_reco"][:] if "T_reco" in sh else sh["T"][:]
        sh_emis = sh["emission_process"][:]
        sh_digit = sh["digit_idx"][:]
    with h5py.File(labl_f, "r") as h:
        pt = h[ev]["per_track"]
        track_id = pt["track_id"][:]; parent_id = pt["parent_id"][:]; pdg = pt["pdg"][:]
        old_pid = pt["particle_idx"][:]
        pp = h[ev]["per_particle"]
        old_cat = pp["category"][:]

    nT = len(track_id)
    row_of_tid = {int(track_id[i]): i for i in range(nT)}

    # per-track start/end direction (earliest / latest segment by time)
    start_dir = np.zeros((nT, 3)); end_dir = np.zeros((nT, 3)); has_seg = np.zeros(nT, bool)
    for t in range(nT):
        m = np.where(seg_track == t)[0]
        if m.size == 0: continue
        has_seg[t] = True
        order = m[np.argsort(seg_time[m])]
        start_dir[t] = seg_dir[order[0]]; end_dir[t] = seg_dir[order[-1]]

    # union-find grouping under the theta rule
    uf = UF(nT)
    for t in range(nT):
        p = row_of_tid.get(int(parent_id[t]))
        if p is None:                      # primary / parent not meaningful -> own particle
            continue
        if int(pdg[t]) in EM:
            uf.union(t, p)                 # absorb EM into source
        else:                              # hadron: split on deflection angle
            if has_seg[t] and has_seg[p]:
                cosang = float(np.clip(end_dir[p] @ start_dir[t], -1, 1))
                if cosang >= cos_thr:      # junction angle < theta -> continuation
                    uf.union(t, p)
            # else: leave as its own particle (a new split)

    roots = np.array([uf.find(t) for t in range(nT)])
    uniq = {r: i for i, r in enumerate(sorted(set(roots.tolist())))}
    new_pid = np.array([uniq[r] for r in roots], dtype=np.int64)
    N = len(uniq)
    counts.append(N)

    # representative track per new particle (prefer a primary, else lowest track_id)
    rep = {}
    for t in range(nT):
        c = new_pid[t]
        if c not in rep or (parent_id[t] == 0) or (track_id[t] < track_id[rep[c]]):
            if c not in rep or parent_id[t] == 0 or track_id[t] < track_id[rep[c]]:
                rep[c] = t
    # contained per new particle = AND over member segments
    contained_new = np.ones(N, bool)
    for s in range(len(seg_track)):
        contained_new[new_pid[seg_track[s]]] &= bool(seg_contained[s])
    # category: carry the representative's old category if available
    cat_new = np.zeros(N, dtype=old_cat.dtype)
    for c in range(N):
        op = old_pid[rep[c]]
        cat_new[c] = old_cat[op] if 0 <= op < len(old_cat) else 0

    # ---- write per_track.particle_idx ----
    with h5py.File(labl_f, "r+") as h:
        h[ev]["per_track"]["particle_idx"][...] = new_pid.astype(old_pid.dtype)
        # ---- rebuild per_particle ----
        del h[ev]["per_particle"]
        pp = h[ev].create_group("per_particle")
        pp.create_dataset("category", data=cat_new)
        pp.create_dataset("contained", data=contained_new)
        pp.create_dataset("interaction_idx", data=np.zeros(N, np.int32))
        gen = np.array([track_id[rep[c]] for c in range(N)], dtype=np.int32)   # one-entry genealogy = rep track_id
        off = np.arange(N + 1, dtype=np.uint32)
        for nm in ("genealogy", "ext_genealogy"):
            pp.create_dataset(f"{nm}_data", data=gen)
            pp.create_dataset(f"{nm}_offsets", data=off)
        h[ev].attrs["n_particles"] = np.int64(N)        # <-- keep the count attr in sync

    # ---- re-aggregate hits from sensor_hits under new particle def ----
    seg_to_newp = new_pid[seg_track]                       # per-segment -> new particle
    hit_newp = seg_to_newp[sh_seg]
    key = {}
    for i in range(len(sh_seg)):
        k = (int(hit_newp[i]), int(sh_sensor[i]), int(sh_digit[i]), int(sh_emis[i]))
        a = key.get(k)
        if a is None: key[k] = [sh_pe[i], sh_t[i], sh_treco[i]]
        else:
            a[0] += sh_pe[i]; a[1] = min(a[1], sh_t[i]); a[2] = min(a[2], sh_treco[i])
    ks = list(key.keys())
    hp = np.array([k[0] for k in ks], np.int32)
    hs = np.array([k[1] for k in ks], np.uint16)
    hd = np.array([k[2] for k in ks], np.int32)          # digit_idx (FK preserved)
    he = np.array([k[3] for k in ks], np.int8)
    hpe = np.array([key[k][0] for k in ks], np.float32)
    ht  = np.array([key[k][1] for k in ks], np.float64)   # absolute time stays float64
    htr = np.array([key[k][2] for k in ks], np.float64)
    _cols = ("particle_idx", "digit_idx", "sensor_idx", "PE", "T", "T_reco", "emission_process")
    _phys = {"particle_idx": hp, "digit_idx": hd, "sensor_idx": hs,
             "PE": hpe, "T": ht, "T_reco": htr, "emission_process": he}
    with h5py.File(hits_f, "r+") as h:
        g = h[ev]
        # Dark-noise hits (emission_process==2) exist only in hits/ — sensor_hits
        # carries no dark — so preserve them unchanged; relabeling reassigns
        # physics particle_idx only, it must not drop dark.
        dark = g["emission_process"][:] == EMISSION_PROCESS_DARK
        keep = {nm: g[nm][:][dark] for nm in _cols}
        for nm in _cols:
            if nm in g: del g[nm]
        for nm in _cols:
            g.create_dataset(nm, data=np.concatenate([_phys[nm], keep[nm]]))
        g.attrs["n_particle_hits"] = np.int64(len(hp) + int(dark.sum()))
        if "n_particles" in g.attrs:
            g.attrs["n_particles"] = np.int64(N)

counts = np.array(counts)
print(f"[theta={theta:.0f}] particles/event min/med/max = {counts.min()}/{int(np.median(counts))}/{counts.max()}; wrote {dst}")
