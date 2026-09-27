#!/usr/bin/env python3
"""Emit one production config per momentum point of the reference time-PDF grid.

``Utilities/timepdf/chart_<pdg>.txt`` is the reference's own job table: columns
are (sequential ops, parallel threads, events each, p_lo, p_hi, -, -, walltime
estimate), so a row is ops*threads*events events at p_lo, or spread over
p_lo..p_hi when p_hi is non-zero. Rows repeat a momentum, so they are summed.

The reference's relative weighting across momentum is preserved and the whole
grid scaled by ``--scale``: histogram statistics are additive (mergehists.pl
sums, combhists re-bins the sum), so a low-statistics pass over the *full*
momentum range can be topped up later by submitting more jobs at the same
points, and nothing has to be regenerated. Momentum range matters more than
depth for a first tune -- a gap in momentum cannot be patched afterwards.

PhotonSim's gun takes kinetic energy, so p is converted.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

MASS_MEV = {13: 105.6583755, 11: 0.51099895, 211: 139.57039}
PARTICLE = {13: "mu-", 11: "e-", 211: "pi+"}
# Cost grows steeply with momentum -- a high-momentum track makes proportionally
# more photons and LUCiD propagates every one. Two measurements: 12.1 s/event at
# ~1000 MeV/c (the mu_metrics campaign) and 153 s/event at 6500 MeV/c (a grid
# job), giving an exponent of 1.36. Using one flat number instead sizes every
# job for 500 events and the high-momentum ones then blow through the walltime
# cap and write NOTHING -- the reference avoids this too, dropping from 5000
# events/file at low momentum to 625 at 5500-7000 (chart_13.txt).
SECONDS_PER_EVENT_AT_1GEV = 12.1
COST_EXPONENT = 1.36
TARGET_SECONDS_PER_JOB = 6050.0   # ~1.7 h, comfortably inside workday


def seconds_per_event(p_mev: float) -> float:
    return SECONDS_PER_EVENT_AT_1GEV * (max(p_mev, 100.0) / 1000.0) ** COST_EXPONENT


def kinetic(p_mev: float, pdg: int) -> float:
    m = MASS_MEV[abs(pdg)]
    return math.sqrt(p_mev * p_mev + m * m) - m


def read_chart(path: Path) -> dict[tuple[float, float], int]:
    """``{(p_lo, p_hi): n_events}``, summing rows that repeat a momentum."""
    out: dict[tuple[float, float], int] = {}
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        n = int(f[0]) * int(f[1]) * int(f[2])
        lo, hi = float(f[3]), float(f[4])
        key = (lo, hi if hi > 0 else lo)
        out[key] = out.get(key, 0) + n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chart", type=Path, required=True)
    ap.add_argument("--pdg", type=int, required=True)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="fraction of the reference statistics to generate")
    ap.add_argument("--min-events", type=int, default=250)
    ap.add_argument("--max-momentum", type=float, default=None,
                    help="skip points above this (MeV/c). Cost grows as p^1.36, "
                         "so the high-momentum tail dominates; a PDF only has to "
                         "span the momenta being reconstructed, since "
                         "GetTimeCoeff clamps outside its range.")
    ap.add_argument("-o", "--out-dir", type=Path, required=True)
    ap.add_argument("--start-index", type=int, default=1,
                    help="dataprod_fanout requires an 'NN_' filename prefix")
    a = ap.parse_args()

    grid = read_chart(a.chart)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    rows = []
    index = a.start_index
    for (lo, hi), n_ref in sorted(grid.items()):
        if a.max_momentum is not None and hi > a.max_momentum:
            continue
        n = max(a.min_events, int(round(n_ref * a.scale)))
        tag = f"p{lo:.0f}" if hi == lo else f"p{lo:.0f}_{hi:.0f}"
        cfg = {
            "name": f"time-PDF training, {PARTICLE[a.pdg]} {tag}",
            "description": (
                f"Reference time-PDF grid point {tag} MeV/c, "
                f"{n} of the reference's {n_ref} events (scale={a.scale}). "
                "Statistics are additive -- top up by resubmitting this config."),
            "material": "water",
            "nominal_train": 0,
            "nominal_test": n,
            "seconds_per_event": round(seconds_per_event(hi), 2),
            "target_seconds_per_job": TARGET_SECONDS_PER_JOB,
            "selection": {"mode": "trigger"},
            "energy_distribution": "uniform",
            "store_individual_photons": True,
            "run_lucid": True,
            "disable_decays": False,
            "particles": [{
                "type": PARTICLE[a.pdg],
                "energy_min_MeV": round(kinetic(lo, a.pdg), 3),
                "energy_max_MeV": round(kinetic(hi, a.pdg), 3),
            }],
            "lucid_options": {"apply_smearing": True, "apply_translation": True},
            "cleanup_root_files": True,
        }
        path = a.out_dir / f"{index:02d}_tpdf_{a.pdg}_{tag}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        index += 1
        total += n
        spe = seconds_per_event(hi)
        rows.append((tag, n, math.ceil(n / max(TARGET_SECONDS_PER_JOB / spe, 1)), spe))

    for tag, n, njob, spe in rows:
        print(f"  {tag:>12}  {n:>7} events  {njob:>3} jobs  "
              f"{spe:>6.1f} s/ev  {n*spe/3600/njob:>4.1f} h/job")
    print(f"next --start-index {index}")
    print(f"{len(rows)} momentum points, {total:,} events, "
          f"{sum(r[2] for r in rows)} jobs, "
          f"{sum(r[1]*r[3] for r in rows) / 3600:.0f} CPU-h")


if __name__ == "__main__":
    main()
