"""Cluster-agnostic planning for the s_max parametrisation scan (Stage 0).

Walks an energy grid for each (material, particle), with per-cell event
count following an energy-dependent schedule:

    `base` events at and below the anchor energy, halved per doubling
    above, never below `floor`.

Optional fan-out for very-high-energy cells: when E > `split_above_MeV`,
each cell is split into ceil(n_events / target_events_per_job) sub-jobs.

The output config carries no `smax_mm` — at this stage we are *computing*
s_max, not consuming it. The Stage-0 invariants (`fixed_direction_z`,
`store_individual_photons=false`, `disable_decays`) match those of
`PhotonSim/tools/smax/scan_smax.py`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Tuple


def events_for(energy_mev: int, base: int, anchor_mev: int, floor: int) -> int:
    """Energy-dependent event count (mirrors scan_smax.py's default schedule)."""
    if energy_mev <= anchor_mev:
        return max(floor, base)
    raw = base * anchor_mev / energy_mev
    return max(floor, int(round(raw)))


def split_plan(*, n_events: int, energy_mev: int,
               split_above_mev: Optional[int],
               target_per_job: Optional[int]) -> Tuple[int, int]:
    """Return (n_jobs, events_per_job) for one cell.

    No split if `split_above_MeV` / `target_events_per_job` aren't set, or
    if `energy_mev` is at/below the threshold. The final job may have
    fewer events than `events_per_job` (ceil(n/target) jobs of `target`
    events each — the last one truncates to whatever's left).
    """
    if (split_above_mev is None or target_per_job is None
            or energy_mev <= split_above_mev or n_events <= target_per_job):
        return 1, n_events
    n_jobs = math.ceil(n_events / target_per_job)
    return n_jobs, target_per_job


def write_per_cell_config(*, cell_dir: Path, material: str, particle: str,
                          energy_mev: int, events_per_job: int, n_jobs: int,
                          name: str) -> Path:
    """Write the dataprod-style JSON config consumed by lucid-run-job.

    Note the absence of `smax_mm` — the macro must not emit `/output/smax`
    because we are computing it from this scan's output.
    """
    cfg = {
        "config_number": -1,
        "use_config_number": False,
        "name": name,
        "description": (f"s_max scan cell: {particle} @ {energy_mev} MeV in "
                        f"{material}. Fills PhotonHist_Distance for s_max fit."),

        "material": material,
        "energy_distribution": "monoenergetic",
        "energy_MeV": energy_mev,

        # Stage-0 invariants:
        "fixed_direction_z": True,
        "store_individual_photons": False,
        "run_lucid": False,
        "disable_decays": True,

        "particles": [{"type": particle}],

        "n_jobs": int(n_jobs),
        "n_events_per_job": int(events_per_job),
    }
    out = cell_dir / "photonsim_config.json"
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    return out


def parse_schedule(cfg: dict
                   ) -> Tuple[int, int, int, Optional[int], Optional[int]]:
    """Extract (base, anchor_MeV, floor, split_above_MeV, target_per_job)."""
    sched = cfg.get("events_schedule")
    if sched:
        base = int(sched["base"])
        anchor = int(sched["anchor_MeV"])
        floor = int(sched.get("floor", 10))
        split_above = sched.get("split_above_MeV")
        target_per_job = sched.get("target_events_per_job")
        split_above = int(split_above) if split_above is not None else None
        target_per_job = int(target_per_job) if target_per_job is not None else None
    else:
        flat = int(cfg["n_events_per_job"])
        base = anchor = floor = flat
        split_above = None
        target_per_job = None
    return base, anchor, floor, split_above, target_per_job
