#!/usr/bin/env python3
"""SLURM fan-out for SIREN-input PhotonSim jobs (s/s_max histogram axis).

Reads a SIREN-input JSON config (material + particles + non-uniform energy
list) and submits one or more SLURM jobs per (particle, energy) cell.

For each cell:

  1. Reads the s_max power-law fit from
     PhotonSim/data/<material>/<particle>/smax_fit.csv (s_max = A * E^B),
     skips energies below `fit_min_mev` unless --include-extrapolated.
  2. Writes a per-cell dataprod-style JSON config that lucid-run-job
     consumes, with `smax_mm` set so the macro emits `/output/smax <val> mm`
     and PhotonSim writes the `PhotonHist_AngleDistanceNorm` histogram.
  3. Splits the total `events_per_cell` across N sub-jobs so each job
     targets ~`target_seconds_per_job` of wall time (linear time model
     fit empirically from v6: t/event ≈ 0.06 + 4.22e-4 * E_MeV).
     Each sub-job gets a unique --job-id and writes
     `output_job_NNNNNN.root` into the cell dir.
  4. Emits one sbatch per sub-job. After all jobs finish, run merge.sh to
     hadd them into a single `photonsim.root` per cell — what
     `plot_norm_hists.py` and the training-table builder consume.

Output layout (PhotonSim-compatible, after merge.sh):

    <OUTPUT_BASE>/<material>/<particle>/<E>MeV/photonsim.root

Config schema (events_schedule is optional; if absent the script falls back
to a flat `n_events_per_job` at top level for backward-compat with smoke
tests):

    {
      "name": "siren_water_mu",
      "material": "water",
      "particles": [{"type": "mu-"}],
      "energy_list_MeV": [...],
      "events_schedule": {
        "events_per_cell": 10000,
        "target_seconds_per_job": 3600,
        "time_model": {
          "a_seconds_per_event": 0.06,
          "b_seconds_per_event_per_mev": 4.22e-4
        }
      },
      "include_extrapolated": false
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
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


# --- Paths --------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
S3DF_JOBS_DIR = SCRIPT_DIR.parent
USER_PATHS_DEFAULT = S3DF_JOBS_DIR / "user_paths.sh"


def load_user_paths(path: Path) -> Dict[str, str]:
    """Source user_paths.sh in a subshell and extract the env it sets."""
    if not path.is_file():
        raise FileNotFoundError(
            f"user_paths.sh not found at {path}. "
            f"Copy {path.parent / 'user_paths.sh.template'} and configure it."
        )
    proc = subprocess.run(
        ["bash", "-c", f"set -e; source {shlex.quote(str(path))} && env"],
        capture_output=True, text=True, check=True,
    )
    out: Dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


# --- s_max parametrisation ---------------------------------------------------

def find_photonsim_dir(env: Dict[str, str], override: Optional[Path]) -> Path:
    """Locate the PhotonSim checkout that holds data/<m>/<p>/smax_fit.csv.

    Search order: explicit --photonsim-dir, $PHOTONSIM_DEV_PATH (from
    user_paths.sh), then walk up from this script's path.
    """
    if override is not None:
        return override.resolve()
    if env.get("PHOTONSIM_DEV_PATH"):
        return Path(env["PHOTONSIM_DEV_PATH"]).resolve()
    # Walk up from this script: .../LUCiD/lucid/production/s3df_jobs/siren_inputs/
    # and look for a sibling PhotonSim/ checkout.
    for p in SCRIPT_DIR.parents:
        cand = p / "PhotonSim"
        if (cand / "data").is_dir():
            return cand.resolve()
    raise FileNotFoundError(
        "Could not locate a PhotonSim checkout with data/. Set "
        "PHOTONSIM_DEV_PATH in user_paths.sh or pass --photonsim-dir."
    )


def load_smax_fit(photonsim_dir: Path, material: str, particle: str
                  ) -> Tuple[float, float, int]:
    """Return (A, B, fit_min_mev) from smax_fit.csv. Raises if missing."""
    path = photonsim_dir / "data" / material / particle / "smax_fit.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"No s_max fit at {path}. Run PhotonSim's tools/smax/analyze_smax.py "
            f"for ({material!r}, {particle!r}) first."
        )
    with path.open() as fh:
        row = next(csv.DictReader(fh), None)
    if not row or not row.get("A") or not row.get("B"):
        raise ValueError(f"s_max fit at {path} is empty/incomplete.")
    return float(row["A"]), float(row["B"]), int(row["fit_min_mev"])


def smax_at(A: float, B: float, energy_mev: float) -> float:
    return A * (energy_mev ** B)


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


# --- Per-cell artifacts ------------------------------------------------------

def write_per_cell_config(*, cell_dir: Path, material: str, particle: str,
                          energy_mev: int, smax_mm: float,
                          events_per_job: int, n_jobs: int,
                          name: str) -> Path:
    """Write the dataprod-style JSON config that lucid-run-job will read.

    The same config is reused by every sub-job in the cell — only the
    sbatch's `--job-id` differs, which makes each sub-job's output ROOT
    a unique `output_job_NNNNNN.root` ready for merge.sh to hadd.
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


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --job-name={job_name}
#SBATCH --output={cell_dir}/job-{job_id:03d}-%j.out
#SBATCH --error={cell_dir}/job-{job_id:03d}-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
{gpu_line}#SBATCH --mem={memory}
#SBATCH --time={time}

set -eu -o pipefail
echo "SLURM Job ID: ${{SLURM_JOB_ID}}"
echo "Job started:  $(date)"
echo "Node:         $(hostname)"

export APPTAINERENV_GENIE_XSEC_FILE={genie_xsec}
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch,/cvmfs {dev_binds} \\
    {image} \\
    lucid-run-job \\
        --config "{cell_cfg}" \\
        --output-dir "{cell_dir}" \\
        --job-id {job_id} \\
        --skip-lucid \\
        --override-energy-MeV {energy_mev}

# Each sub-job writes output_job_<job_id:06>.root; merge.sh hadd's them all
# into photonsim.root per cell after the queue drains.

echo "Job ended: $(date)"
"""


def write_sbatch(*, cell_dir: Path, cell_cfg: Path, energy_mev: int,
                 job_name: str, job_id: int, sbatch_name: str,
                 env: Dict[str, str], partition: str,
                 use_gpu: bool) -> Path:
    """Emit submit_job_<id>.sbatch (or submit.sbatch when n_jobs=1) in the
    cell dir."""
    image = env["LUCID_IMAGE_PATH"]
    gpus = "1" if use_gpu else env.get("DEFAULT_GPUS", "0")
    gpu_line = f"#SBATCH --gpus={gpus}\n" if gpus != "0" else ""

    dev_binds = ""
    if env.get("LUCID_DEV_PATH"):
        dev_binds += f" -B {env['LUCID_DEV_PATH']}:/opt/LUCiD"
    if env.get("PHOTONSIM_DEV_PATH"):
        dev_binds += f" -B {env['PHOTONSIM_DEV_PATH']}:/opt/PhotonSim"

    body = SBATCH_TEMPLATE.format(
        partition=partition,
        account=env["SLURM_ACCOUNT"],
        job_name=job_name,
        cell_dir=str(cell_dir),
        cpus=env.get("DEFAULT_CPUS", "1"),
        gpu_line=gpu_line,
        memory=env.get("DEFAULT_MEMORY", "16000"),
        time=env.get("DEFAULT_TIME", "08:00:00"),
        genie_xsec=env.get("GENIE_XSEC_FILE", ""),
        dev_binds=dev_binds.strip(),
        image=image,
        cell_cfg=str(cell_cfg),
        energy_mev=energy_mev,
        job_id=job_id,
    )
    out = cell_dir / sbatch_name
    out.write_text(body)
    out.chmod(0o755)
    return out


# --- Driver ------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", type=Path, required=True,
                   help="SIREN-input JSON config "
                        "(see configs/water_mu_test.json for an example).")
    p.add_argument("-s", "--submit", action="store_true",
                   help="Submit jobs to SLURM (default: prepare sbatch only).")
    p.add_argument("-t", "--test", action="store_true",
                   help="Test mode: only the first (particle, energy) cell.")
    p.add_argument("-o", "--output-base", type=Path, default=None,
                   help="Override OUTPUT_BASE_PATH from user_paths.sh.")
    p.add_argument("-P", "--partition", type=str, default=None,
                   help="Override SLURM_PARTITION from user_paths.sh.")
    p.add_argument("-g", "--gpu", action="store_true",
                   help="Request 1 GPU per job (default: DEFAULT_GPUS).")
    p.add_argument("--user-paths", type=Path, default=USER_PATHS_DEFAULT,
                   help="Path to user_paths.sh (default: %(default)s).")
    p.add_argument("--photonsim-dir", type=Path, default=None,
                   help="PhotonSim checkout (overrides PHOTONSIM_DEV_PATH).")
    p.add_argument("--include-extrapolated", action="store_true",
                   help="Process energies below the fit's fit_min_mev "
                        "(default: skip with a warning).")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-run cells that already have photonsim.root "
                        "(default: skip them).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Tooling check
    if shutil.which("sbatch") is None and args.submit:
        print("error: sbatch not found on PATH; submission requires SLURM.",
              file=sys.stderr)
        return 2

    # Config
    if not args.config.is_file():
        print(f"error: config not found: {args.config}", file=sys.stderr)
        return 2
    cfg = json.loads(args.config.read_text())

    name = cfg["name"]
    material = cfg["material"]
    particles = [p["type"] for p in cfg["particles"]]
    energies: List[int] = list(cfg["energy_list_MeV"])
    include_extrapolated = bool(
        args.include_extrapolated or cfg.get("include_extrapolated", False)
    )

    # Events budget. Either an `events_schedule` block (with optional
    # time_model + target_seconds_per_job for splitting), or fall back to a
    # flat `n_events_per_job` at top level for smoke-test configs.
    sched = cfg.get("events_schedule")
    if sched:
        events_per_cell = int(sched["events_per_cell"])
        target_s = sched.get("target_seconds_per_job")
        target_s = float(target_s) if target_s is not None else None
        tm = {**DEFAULT_TIME_MODEL, **(sched.get("time_model") or {})}
    else:
        events_per_cell = int(cfg["n_events_per_job"])
        target_s = None
        tm = DEFAULT_TIME_MODEL
    a_s = float(tm["a_seconds_per_event"])
    b_s_per_mev = float(tm["b_seconds_per_event_per_mev"])

    # user_paths
    env = load_user_paths(args.user_paths)
    photonsim_dir = find_photonsim_dir(env, args.photonsim_dir)

    output_base = (args.output_base.resolve() if args.output_base
                   else Path(env["OUTPUT_BASE_PATH"]).resolve())
    partition = args.partition or env.get("SLURM_PARTITION", "")
    if not partition:
        print("error: SLURM partition not set (user_paths.sh or -P).",
              file=sys.stderr)
        return 2

    print(f"=== SIREN-input fan-out: {name} ===")
    print(f"material:     {material}")
    print(f"particles:    {particles}")
    print(f"energies:     {len(energies)} cells, "
          f"{energies[0]}..{energies[-1]} MeV")
    print(f"events/cell:  {events_per_cell:,}")
    if target_s is not None:
        print(f"target/job:   {target_s:.0f} s "
              f"(time model: t/event = {a_s:g} + {b_s_per_mev:g} * E_MeV)")
    print(f"output base:  {output_base}")
    print(f"PhotonSim:    {photonsim_dir}")
    print(f"partition:    {partition}")
    print("")

    # Build cell list. Each cell: (particle, energy, smax_mm, n_jobs, evt_per_job).
    cells: List[Tuple[str, int, float, int, int]] = []
    skipped: List[str] = []
    for particle in particles:
        A, B, fit_min = load_smax_fit(photonsim_dir, material, particle)
        for energy in energies:
            if energy < fit_min and not include_extrapolated:
                skipped.append(f"{particle} @ {energy} MeV "
                               f"(< fit_min={fit_min}; pass --include-extrapolated to force)")
                continue
            n_jobs, evt_per_job = split_plan_by_time(
                events_per_cell=events_per_cell, energy_mev=energy,
                target_seconds_per_job=target_s,
                a_s=a_s, b_s_per_mev=b_s_per_mev,
            )
            cells.append((particle, energy, smax_at(A, B, energy),
                          n_jobs, evt_per_job))

    if args.test and cells:
        cells = cells[:1]

    if not cells:
        print("error: no cells to submit "
              f"(skipped {len(skipped)}). Use --include-extrapolated to force.",
              file=sys.stderr)
        for s in skipped:
            print(f"  SKIP {s}", file=sys.stderr)
        return 1

    total_jobs = sum(c[3] for c in cells)
    total_events = sum(c[3] * c[4] for c in cells)
    print(f"total cells:   {len(cells)}")
    print(f"total jobs:    {total_jobs}")
    print(f"total events:  {total_events:,}")
    print("")

    prepared = submitted = skipped_existing = skipped_jobs = 0
    for particle, energy, smax_mm, n_jobs, evt_per_job in cells:
        cell_dir = output_base / material / particle / f"{energy}MeV"
        cell_dir.mkdir(parents=True, exist_ok=True)

        merged = cell_dir / "photonsim.root"
        if merged.is_file() and not args.no_skip_existing:
            print(f"  skip (merged exists): {merged}")
            skipped_existing += 1
            continue

        cell_name = f"siren_{material}_{particle}_{energy}MeV"
        cell_cfg = write_per_cell_config(
            cell_dir=cell_dir, material=material, particle=particle,
            energy_mev=energy, smax_mm=smax_mm,
            events_per_job=evt_per_job, n_jobs=n_jobs, name=cell_name,
        )

        for job_id in range(1, n_jobs + 1):
            if n_jobs == 1:
                sb_name = "submit.sbatch"
                job_label = cell_name
            else:
                sb_name = f"submit_job_{job_id:03d}.sbatch"
                job_label = f"{cell_name}_j{job_id:03d}"

            out_root = cell_dir / f"output_job_{job_id:06d}.root"
            if out_root.is_file() and not args.no_skip_existing:
                print(f"  skip (job exists): {out_root.name}")
                skipped_jobs += 1
                continue

            sb = write_sbatch(
                cell_dir=cell_dir, cell_cfg=cell_cfg, energy_mev=energy,
                job_name=job_label, job_id=job_id, sbatch_name=sb_name,
                env=env, partition=partition, use_gpu=args.gpu,
            )
            prepared += 1

            if args.submit:
                r = subprocess.run(["sbatch", str(sb)], capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"  FAILED sbatch for {job_label}: {r.stderr.strip()}",
                          file=sys.stderr)
                    continue
                submitted += 1
                print(f"  submitted {job_label}  (s_max={smax_mm:.1f} mm, "
                      f"{evt_per_job} evts)  -> {r.stdout.strip()}")
            else:
                print(f"  prepared  {job_label}  (s_max={smax_mm:.1f} mm, "
                      f"{evt_per_job} evts)  -> {sb}")

    for s in skipped:
        print(f"  SKIP {s}")

    print("")
    print("=== Fan-out complete ===")
    print(f"Prepared:        {prepared}")
    print(f"Submitted:       {submitted}")
    print(f"Skipped cells:   {skipped_existing}")
    print(f"Skipped jobs:    {skipped_jobs}")
    print(f"Skipped (fit):   {len(skipped)}")
    print(f"Output root:     {output_base / material}")
    print("")
    print("Next: once all jobs finish, run merge.sh to hadd per-cell")
    print("      output_job_*.root into photonsim.root for each cell.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
