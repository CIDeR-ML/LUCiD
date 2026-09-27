#!/usr/bin/env python3
"""Vertex, direction and momentum resolution of a fiTQun reconstruction.

Reads fiTQun's own output tree and LUCiD's truth, matches them by event index,
and reports the single-ring fit's performance under one particle hypothesis.

fiTQun stores the single-ring results as ``fq1r*[sub-event][hypothesis]``, where
the hypothesis axis follows ``fiTQun_shared::PIDarr = {22,11,13,211,321,2212,48}``
-- so the muon is index 2, not 1. Sub-event 0 is the primary trigger cluster.

    ./fq_resolution.py --fq out.root --labl wc_labl_0000.h5 --step wc_step_0000.h5

Resolutions are quoted as the 68th percentile of the absolute error, which is
the usual water-Cherenkov convention and does not assume the error is Gaussian;
the momentum line also carries the median fractional bias, because a mistuned
charge scale shows up there first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

LUCID_ROOT = Path(__file__).resolve().parents[2]
if str(LUCID_ROOT) not in sys.path:
    sys.path.insert(0, str(LUCID_ROOT))

from lucid.production.fitqun import truth as truth_mod  # noqa: E402
from lucid.production.fitqun.particles import PDG_NAMES  # noqa: E402

PID_INDEX = {22: 0, 11: 1, 13: 2, 211: 3, 321: 4, 2212: 5}
M_TO_CM = 100.0


def pct68(x):
    return float(np.percentile(np.abs(x), 68)) if len(x) else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fq", type=Path, required=True, help="fiTQun output .root")
    p.add_argument("--labl", type=Path, required=True)
    p.add_argument("--step", type=Path, required=True)
    p.add_argument("--pdg", type=int, default=13, help="hypothesis to score (default mu-)")
    p.add_argument("--require-contained", action="store_true",
                   help="keep only events LUCiD marks as contained")
    p.add_argument("--require-fc", action="store_true",
                   help="keep only events fiTQun fits as fully contained "
                        "(fq1rpcflg==0). Resolutions are conventionally quoted "
                        "for FC events; a PC track leaves the detector and its "
                        "momentum is a lower bound, not a measurement")
    a = p.parse_args(argv)

    import uproot
    ip = PID_INDEX[a.pdg]
    with uproot.open(a.fq) as f:
        t = f["fiTQun"]
        pos = t["fq1rpos"].array(library="np")
        dir_ = t["fq1rdir"].array(library="np")
        mom = t["fq1rmom"].array(library="np")
        pcflg = t["fq1rpcflg"].array(library="np")

    tracks = truth_mod.read_tracks(a.labl, a.step, max_events=len(mom))
    if not tracks:
        raise SystemExit("no single-primary truth events found")

    dv, dth, dp, frac = [], [], [], []
    n_pc = 0
    for tr in tracks:
        if tr.event >= len(mom):
            continue
        if a.require_contained and not tr.contained:
            continue
        # fq1r* are [sub-event][hypothesis]; take the first sub-event.
        rp = np.asarray(pos[tr.event][0][ip], dtype=np.float64)     # cm
        rd = np.asarray(dir_[tr.event][0][ip], dtype=np.float64)
        rm = float(mom[tr.event][0][ip])
        is_pc = int(pcflg[tr.event][0][ip]) != 0
        if is_pc:
            n_pc += 1
            if a.require_fc:
                continue
        nd = np.linalg.norm(rd)
        if not (nd > 0 and rm > 0):
            continue
        rd /= nd
        dv.append(np.linalg.norm(rp - tr.vertex_m * M_TO_CM))
        dth.append(np.degrees(np.arccos(np.clip(np.dot(rd, tr.direction), -1, 1))))
        dp.append(rm - tr.momentum_mev)
        frac.append((rm - tr.momentum_mev) / tr.momentum_mev)

    dv, dth, dp, frac = map(np.asarray, (dv, dth, dp, frac))
    name = PDG_NAMES.get(a.pdg, str(a.pdg))
    print(f"fiTQun single-ring {name} hypothesis: {len(dv)} events scored "
          f"({n_pc} flagged not fully contained)")
    if not len(dv):
        return 1
    print(f"  vertex     68% = {pct68(dv):8.1f} cm     median {np.median(dv):8.1f} cm")
    print(f"  direction  68% = {pct68(dth):8.2f} deg    median {np.median(dth):8.2f} deg")
    print(f"  momentum   68% = {pct68(frac)*100:8.2f} %      median bias "
          f"{np.median(frac)*100:+7.2f} %")
    print(f"  truth momentum range: {min(t.momentum_mev for t in tracks):.0f}"
          f"-{max(t.momentum_mev for t in tracks):.0f} MeV/c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
