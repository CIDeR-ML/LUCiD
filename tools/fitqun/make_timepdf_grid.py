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
# Measured on the mu_metrics campaign: 242 events in ~49 min of wall clock.
SECONDS_PER_EVENT = 12.1
TARGET_SECONDS_PER_JOB = 6050.0   # -> 500 events/job, safe inside workday


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
            "seconds_per_event": SECONDS_PER_EVENT,
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
        rows.append((tag, n, math.ceil(n / (TARGET_SECONDS_PER_JOB / SECONDS_PER_EVENT))))

    for tag, n, njob in rows:
        print(f"  {tag:>12}  {n:>7} events  {njob:>3} jobs")
    print(f"next --start-index {index}")
    print(f"{len(rows)} momentum points, {total:,} events, "
          f"{sum(r[2] for r in rows)} jobs, "
          f"{total * SECONDS_PER_EVENT / 3600:.0f} CPU-h")


if __name__ == "__main__":
    main()
