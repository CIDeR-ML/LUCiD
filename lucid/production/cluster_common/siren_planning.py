"""Cluster-agnostic planning for SIREN-input scans (Stage 1).

For each (particle, energy) cell:

  1. Read the `s_max(E)` parametrisation from
     `PhotonSim/data/<material>/<particle>/smax_fit.csv` and evaluate it.
  2. Split the cell's `events_per_cell` budget across N sub-jobs so each
     job targets ~`target_seconds_per_job` of wall time, using a linear
     time model `t/event = a + b * E_MeV`.
  3. Write the per-cell `photonsim_config.json` (consumed unchanged by
     `lucid-run-job` inside the container).

The cluster-specific part (which submit-description file to write per
sub-job, which command to use to submit it) lives in `cluster.py`.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Empirical time model fit from the v6 SIREN-input test (100 events/cell,
# 1 km world box, 500 MeV..100 GeV on roma partition):
#   t_per_event(E_MeV) ≈ 0.06 + 4.22e-4 * E_MeV   [seconds]
# Override per-config via events_schedule.time_model if needed.
DEFAULT_TIME_MODEL = {
    "a_seconds_per_event":         0.06,
    "b_seconds_per_event_per_mev": 4.22e-4,
}


# --- PhotonSim checkout discovery --------------------------------------------

def find_photonsim_dir(env: Dict[str, str], override: Optional[Path],
                       script_dir: Path) -> Path:
    """Locate the PhotonSim checkout that holds data/<m>/<p>/smax_fit.csv.

    Search order: explicit override, $PHOTONSIM_DEV_PATH (from user_paths.sh),
    then walk up from `script_dir` looking for a sibling PhotonSim/.
    """
    if override is not None:
        return override.resolve()
    if env.get("PHOTONSIM_DEV_PATH"):
        return Path(env["PHOTONSIM_DEV_PATH"]).resolve()
    for p in script_dir.parents:
        cand = p / "PhotonSim"
        if (cand / "data").is_dir():
            return cand.resolve()
    raise FileNotFoundError(
        "Could not locate a PhotonSim checkout with data/. Set "
        "PHOTONSIM_DEV_PATH in user_paths.sh or pass --photonsim-dir."
    )


# --- s_max parametrisation ----------------------------------------------------

def load_smax_row(photonsim_dir: Path, material: str, particle: str
                  ) -> Tuple[dict, int]:
    """Return (csv_row_dict, fit_min_mev) from smax_fit.csv. Raises if missing."""
    path = photonsim_dir / "data" / material / particle / "smax_fit.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"No s_max fit at {path}. Run PhotonSim's tools/smax/analyze_smax.py "
            f"for ({material!r}, {particle!r}) first."
        )
    with path.open() as fh:
        row = next(csv.DictReader(fh), None)
    if not row or not row.get("form"):
        raise ValueError(f"s_max fit at {path} is empty/incomplete.")
    return row, int(row["fit_min_mev"])


def eval_smax(row: dict, energy_mev: float) -> float:
    """Evaluate s_max(E) given a CSV row from smax_fit.csv.

    Dispatches on `row["form"]`. Keep in sync with the canonical FORMS dict
    in PhotonSim/tools/smax/analyze_smax.py.
    """
    form = row["form"]
    if form == "A*E^B":
        A, B = float(row["A"]), float(row["B"])
        return A * energy_mev ** B
    if form == "smooth_two_power":
        a, b1, b2, E0 = (float(row[k]) for k in ("a", "b1", "b2", "E0"))
        return a * energy_mev ** b1 / (1.0 + (energy_mev / E0) ** (b1 - b2))
    if form == "piecewise":
        # Two smooth_two_power pieces joined at e_join_mev with C⁰+C¹
        # continuity. Low piece in (a, b1, b2, E0); high piece in
        # (a_hi, b1_hi, b2_hi, E0_hi).
        ej = float(row["e_join_mev"])
        if energy_mev < ej:
            a, b1, b2, E0 = (float(row[k]) for k in ("a", "b1", "b2", "E0"))
        else:
            a, b1, b2, E0 = (float(row[k]) for k in ("a_hi", "b1_hi", "b2_hi", "E0_hi"))
        return a * energy_mev ** b1 / (1.0 + (energy_mev / E0) ** (b1 - b2))
    raise ValueError(f"unknown smax form: {form!r}")


# --- Time-based split planning -----------------------------------------------

def t_per_event(energy_mev: float, a_s: float, b_s_per_mev: float) -> float:
    return a_s + b_s_per_mev * energy_mev


def split_plan_by_time(*, events_per_cell: int, energy_mev: int,
                        target_seconds_per_job: Optional[float],
                        a_s: float, b_s_per_mev: float) -> Tuple[int, int]:
    """Return (n_jobs, events_per_job) for one cell.

    If `target_seconds_per_job` is None or the cell would fit in one job
    already, return (1, events_per_cell). Otherwise pick `n_jobs` so each
    job runs for ≈ target wall-time, then balance events evenly across
    them (`ceil` per-job count — the last sub-job may run slightly short).
    """
    if target_seconds_per_job is None or events_per_cell <= 0:
        return 1, events_per_cell
    t_per = t_per_event(energy_mev, a_s, b_s_per_mev)
    if t_per <= 0:
        return 1, events_per_cell
    cell_wall_s = events_per_cell * t_per
    if cell_wall_s <= target_seconds_per_job:
        return 1, events_per_cell
    n_jobs = max(1, math.ceil(cell_wall_s / target_seconds_per_job))
    events_per_job = math.ceil(events_per_cell / n_jobs)
    return n_jobs, events_per_job


# --- Per-cell config emission ------------------------------------------------

def write_per_cell_config(*, cell_dir: Path, material: str, particle: str,
                          energy_mev: int, smax_mm: float,
                          events_per_job: int, n_jobs: int,
                          name: str) -> Path:
    """Write the dataprod-style JSON config that lucid-run-job will read.

    The same config is reused by every sub-job in the cell — only the
    submit-description's `--job-id` differs, which makes each sub-job's
    output ROOT a unique `output_job_NNNNNN.root` ready to hadd.
    """
    cfg = {
        "config_number": -1,
        "use_config_number": False,
        "name": name,
        "description": (f"SIREN-input cell: {particle} @ {energy_mev} MeV in "
                        f"{material}, s_max baked in for AngleDistanceNorm."),

        "material": material,
        "energy_distribution": "monoenergetic",
        "energy_MeV": energy_mev,

        # SIREN-input invariants:
        "fixed_direction_z": True,
        "smax_mm": smax_mm,
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


# --- Schedule parsing --------------------------------------------------------

def parse_schedule(cfg: dict
                   ) -> Tuple[int, List[Tuple[int, int]], Optional[float], float, float]:
    """Extract (events_default, events_ranges, target_seconds, a, b)
    from a SIREN-input config.

    Two forms accepted:

    * `events_schedule` block:
        - `events_per_cell` (required): default per-cell event budget.
        - `events_per_cell_ranges` (optional): list of
          `{"e_max": <MeV>, "events_per_cell": <int>}` entries. Evaluated in
          order against each cell's energy; the first whose `e_max` strictly
          exceeds the energy wins. Cells above the largest `e_max` fall back
          to the top-level `events_per_cell`.
        - `target_seconds_per_job` (optional): wall-time target per sub-job.
        - `time_model` (optional): `t_per_event = a + b * E_MeV` for splitting.
    * A flat `n_events_per_job` at top level (smoke-test configs).

    `events_ranges` is returned as a list of `(e_max_excl, events_per_cell)`
    tuples, sorted by `e_max_excl`. Empty list when no ranges are defined.
    """
    sched = cfg.get("events_schedule")
    if sched:
        events_per_cell = int(sched["events_per_cell"])
        target_s = sched.get("target_seconds_per_job")
        target_s = float(target_s) if target_s is not None else None
        tm = {**DEFAULT_TIME_MODEL, **(sched.get("time_model") or {})}
        ranges_raw = sched.get("events_per_cell_ranges") or []
        ranges: List[Tuple[int, int]] = sorted(
            (int(r["e_max"]), int(r["events_per_cell"])) for r in ranges_raw
        )
    else:
        events_per_cell = int(cfg["n_events_per_job"])
        target_s = None
        tm = DEFAULT_TIME_MODEL
        ranges = []
    a_s = float(tm["a_seconds_per_event"])
    b_s_per_mev = float(tm["b_seconds_per_event_per_mev"])
    return events_per_cell, ranges, target_s, a_s, b_s_per_mev


def events_per_cell_for_energy(default_events: int,
                               ranges: List[Tuple[int, int]],
                               energy_mev: int) -> int:
    """Resolve per-cell event budget via the per-range schedule, falling
    back to ``default_events`` when no range matches."""
    for e_max, n in ranges:
        if energy_mev < e_max:
            return n
    return default_events


# --- Cell enumeration --------------------------------------------------------

class Cell(tuple):
    """A planned (particle, energy, smax_mm, n_jobs, events_per_job) tuple."""
    __slots__ = ()
    def __new__(cls, particle, energy_mev, smax_mm, n_jobs, events_per_job):
        return super().__new__(cls, (particle, energy_mev, smax_mm,
                                     n_jobs, events_per_job))
    @property
    def particle(self) -> str: return self[0]
    @property
    def energy_mev(self) -> int: return self[1]
    @property
    def smax_mm(self) -> float: return self[2]
    @property
    def n_jobs(self) -> int: return self[3]
    @property
    def events_per_job(self) -> int: return self[4]


def build_cells(*, particles: List[str], energies: List[int],
                photonsim_dir: Path, material: str,
                events_per_cell: int,
                events_per_cell_ranges: Optional[List[Tuple[int, int]]],
                target_seconds_per_job: Optional[float],
                a_s: float, b_s_per_mev: float,
                include_extrapolated: bool) -> Tuple[List[Cell], List[str]]:
    """Enumerate every (particle, energy) cell with its full plan.

    Returns (cells, skipped_messages). Energies below `fit_min_mev` are
    skipped (with a human-readable message in the second return value)
    unless `include_extrapolated` is True.

    Per-cell event budget is resolved from `events_per_cell_ranges` (if
    set), falling back to `events_per_cell` for energies above the largest
    range cutoff.
    """
    ranges = events_per_cell_ranges or []
    cells: List[Cell] = []
    skipped: List[str] = []
    for particle in particles:
        row, fit_min = load_smax_row(photonsim_dir, material, particle)
        for energy in energies:
            if energy < fit_min and not include_extrapolated:
                skipped.append(f"{particle} @ {energy} MeV "
                               f"(< fit_min={fit_min}; pass --include-extrapolated to force)")
                continue
            cell_budget = events_per_cell_for_energy(events_per_cell, ranges, energy)
            n_jobs, evt_per_job = split_plan_by_time(
                events_per_cell=cell_budget, energy_mev=energy,
                target_seconds_per_job=target_seconds_per_job,
                a_s=a_s, b_s_per_mev=b_s_per_mev,
            )
            cells.append(Cell(particle, energy, eval_smax(row, energy),
                              n_jobs, evt_per_job))
    return cells, skipped
