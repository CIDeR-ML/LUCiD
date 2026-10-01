#!/usr/bin/env python3
"""Per-event fiTQun residuals, with the vertex error split along the track.

A single |dv| hides which way the fit is wrong. Cherenkov reconstruction is
intrinsically weaker ALONG the track than across it -- the ring barely changes
as the vertex slides up and down the direction of travel -- so the longitudinal
and transverse components have different resolutions and, more importantly,
different biases. Quoting only |dv| averages the two together.

Writes one row per FC event: dL dT dth dpfrac
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

LUCID_ROOT = Path(__file__).resolve().parents[2]
if str(LUCID_ROOT) not in sys.path:
    sys.path.insert(0, str(LUCID_ROOT))
from lucid.production.fitqun import truth as truth_mod  # noqa: E402

PID_INDEX = {22: 0, 11: 1, 13: 2, 211: 3, 321: 4, 2212: 5, 48: 6}
M_TO_CM = 100.0

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--fq", required=True)
    p.add_argument("--labl", required=True)
    p.add_argument("--step", required=True)
    p.add_argument("--pdg", type=int, default=13)
    p.add_argument("--require-fc", action="store_true")
    p.add_argument("--all", action="store_true", help="keep PC events too, tagging them")
    p.add_argument("-o", "--out", required=True)
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
    rows = []
    for tr in tracks:
        if tr.event >= len(mom):
            continue
        rp = np.asarray(pos[tr.event][0][ip], dtype=np.float64)
        rd = np.asarray(dir_[tr.event][0][ip], dtype=np.float64)
        rm = float(mom[tr.event][0][ip])
        is_pc = int(pcflg[tr.event][0][ip]) != 0
        if a.require_fc and is_pc:
            continue
        nd = np.linalg.norm(rd)
        if not (nd > 0 and rm > 0):
            continue
        rd /= nd
        u = np.asarray(tr.direction, dtype=np.float64)      # true unit direction
        u /= np.linalg.norm(u)
        d = rp - tr.vertex_m * M_TO_CM                      # vertex error vector
        dL = float(np.dot(d, u))                            # signed, along track
        dT = float(np.linalg.norm(d - dL * u))               # perpendicular
        dth = float(np.degrees(np.arccos(np.clip(np.dot(rd, u), -1, 1))))
        # distance from the true vertex to the nearest wall of the SK_WAND
        # cylinder (R=1696.23 cm, half-length 1818.92 cm, axis = z) -- the geometry
        # fiTQun prints at load. dwall<0 would mean outside the tank.
        v = np.asarray(tr.vertex_m, dtype=np.float64) * M_TO_CM
        r_perp = float(np.hypot(v[0], v[1]))
        dwall = min(1696.23 - r_perp, 1818.92 - abs(float(v[2])))
        rows.append((dL, dT, dth, (rm - tr.momentum_mev) / tr.momentum_mev,
                     dwall, 1.0 if is_pc else 0.0,
                     1.0 if getattr(tr, "contained", True) else 0.0))

    np.savetxt(a.out, np.array(rows), fmt="%.6g",
               header="dL_cm dT_cm dtheta_deg dp_frac dwall_cm fq_pc truth_contained")
    print(f"  wrote {len(rows)} FC events -> {a.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
