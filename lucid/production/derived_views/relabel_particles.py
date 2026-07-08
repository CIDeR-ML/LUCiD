#!/usr/bin/env python3
"""Derived-view demo (ROOT-FREE): re-group tracks into 'particles' using only the
stored dataset (labl/per_track ancestry). No ROOT, no re-simulation.

The viewer colors LABEL=Particle from `labl/per_track/particle_idx` (segment->track->
particle) and `hits/particle_idx`. We just rewrite those columns under a chosen
grouping criterion. Same events, different particle definition.

Modes:
  split  : keep the stored per-track categorization (secondary pions are separate
           particles -> "one pion splits into many"). Output == input grouping.
  merge  : fold every track into its PRIMARY (group by per_track.ancestor) -> one
           particle per primary ("the whole cascade is one pion").

Usage: relabel_particles.py SRC DST {split|merge}
"""
import sys, os, shutil, glob
import numpy as np
import h5py

src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
assert mode in ("split", "merge")
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)

labl_f = sorted(glob.glob(os.path.join(dst, "labl/*.h5")))[0]
hits_f = sorted(glob.glob(os.path.join(dst, "hits/*.h5")))[0]

def event_keys(path):
    with h5py.File(path, "r") as h:
        return [k for k in h if k.startswith("event_")]

evs = event_keys(labl_f)
changed = 0
for ev in evs:
    with h5py.File(labl_f, "r+") as h:
        pt = h[ev]["per_track"]
        track_id   = pt["track_id"][:]
        parent_id  = pt["parent_id"][:]
        ancestor   = pt["ancestor"][:]
        old_pid    = pt["particle_idx"][:]

        if mode == "split":
            new_pid = old_pid.copy()                      # stored grouping
        else:  # merge: every track -> its primary's particle id
            # primary tracks have parent_id == 0; map their track_id -> particle_idx
            prim_pid = {int(track_id[t]): int(old_pid[t])
                        for t in range(len(track_id)) if parent_id[t] == 0 and old_pid[t] >= 0}
            new_pid = old_pid.copy()
            for t in range(len(track_id)):
                tgt = prim_pid.get(int(ancestor[t]))      # the primary this track descends from
                if tgt is not None and old_pid[t] >= 0:
                    new_pid[t] = tgt
        pt["particle_idx"][...] = new_pid.astype(old_pid.dtype)

        # old particle id -> new particle id (for the hits table)
        remap = {}
        for t in range(len(old_pid)):
            if old_pid[t] >= 0:
                remap[int(old_pid[t])] = int(new_pid[t])

    with h5py.File(hits_f, "r+") as h:
        hp = h[ev]["particle_idx"]
        a = hp[:]
        b = np.array([remap.get(int(x), int(x)) if x >= 0 else int(x) for x in a], dtype=a.dtype)
        hp[...] = b
        if not np.array_equal(a, b):
            changed += 1

# report distinct-particle counts per event for a couple of rich events
with h5py.File(labl_f, "r") as h:
    dist = []
    for ev in evs:
        pid = h[ev]["per_track"]["particle_idx"][:]
        dist.append(len(set(int(x) for x in pid if x >= 0)))
    dist = np.array(dist)
print(f"[{mode}] {len(evs)} events; distinct particles per event "
      f"min/med/max = {dist.min()}/{int(np.median(dist))}/{dist.max()}; hits events changed={changed}")
print(f"[{mode}] wrote {dst}")
