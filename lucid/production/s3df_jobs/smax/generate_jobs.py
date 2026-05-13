#!/usr/bin/env python3
"""SLURM fan-out for the s_max parametrisation scan (Stage 0).

Prerequisite for SIREN-input generation: walks an energy grid for each
particle, runs PhotonSim with `/photon/storeIndividual false` + fixed +Z
direction, and writes `photonsim.root` files containing the
`PhotonHist_Distance` 1D histogram. Those ROOTs are then fed to
`analyze.sh` (which calls `PhotonSim/tools/smax/analyze_smax.py`) to
produce the per-particle `smax_data.csv` + `smax_fit.csv` under
`PhotonSim/data/<material>/<particle>/`.

`/output/smax` is intentionally **not** emitted here — at this stage we
are *computing* s_max, not consuming it.

Output layout (PhotonSim-compatible, matches scan_smax.py):

    <OUTPUT_BASE>/<material>/<particle>/<E>MeV/photonsim.root

For cells where `events_schedule` requests splitting (E above
`split_above_MeV`), each cell is fanned out into several smaller jobs:

    <OUTPUT_BASE>/<material>/<particle>/<E>MeV/output_job_NNNNNN.root
    <OUTPUT_BASE>/<material>/<particle>/<E>MeV/submit_job_NNN.sbatch

Run `merge.sh <OUTPUT_BASE>` after all jobs finish to `hadd` the
per-job ROOTs into `photonsim.root` for each cell.

Per-cell event count follows an energy-dependent schedule: `base` events
at and below the anchor energy, halved per doubling above, never below
`floor`. The default config reproduces (10x stats) the schedule that
generated the existing `PhotonSim/data/water/{mu-,e-}/smax_data.csv`.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
S3DF_JOBS_DIR = SCRIPT_DIR.parent
USER_PATHS_DEFAULT = S3DF_JOBS_DIR / "user_paths.sh"


def load_user_paths(path: Path) -> Dict[str, str]:
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


def events_for(energy_mev: int, base: int, anchor_mev: int, floor: int) -> int:
    """Energy-dependent event count (mirrors scan_smax.py's default schedule)."""
    if energy_mev <= anchor_mev:
        return max(floor, base)
    raw = base * anchor_mev / energy_mev
    return max(floor, int(round(raw)))


def split_plan(*, n_events: int, energy_mev: int, split_above_mev: Optional[int],
               target_per_job: Optional[int]) -> Tuple[int, int]:
    """Return (n_jobs, events_per_job) for one cell.

    No split if `split_above_MeV`/`target_events_per_job` aren't set, or if
    `energy_mev` is at/below the threshold. The final job may have fewer
    events than `events_per_job` (we just use ceil(n/target) jobs of
    `target` events each — the last one truncates to whatever's left).
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

echo "Job ended: $(date)"
"""


def write_sbatch(*, cell_dir: Path, cell_cfg: Path, energy_mev: int,
                 job_name: str, job_id: int, sbatch_name: str,
                 env: Dict[str, str], partition: str, use_gpu: bool) -> Path:
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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", type=Path, required=True,
                   help="s_max-scan JSON config (see configs/water_mu.json).")
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
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-run cells that already have photonsim.root.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.submit and shutil.which("sbatch") is None:
        print("error: sbatch not found on PATH.", file=sys.stderr)
        return 2

    if not args.config.is_file():
        print(f"error: config not found: {args.config}", file=sys.stderr)
        return 2
    cfg = json.loads(args.config.read_text())

    name = cfg["name"]
    material = cfg["material"]
    particles = [p["type"] for p in cfg["particles"]]
    energies: List[int] = list(cfg["energy_list_MeV"])

    # Event schedule: either a {base, anchor_MeV, floor} dict, or fall back to
    # a flat n_events_per_job. Optional split fields control fan-out for the
    # very-high-energy cells.
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
        base = anchor = flat
        floor = flat
        split_above = None
        target_per_job = None

    env = load_user_paths(args.user_paths)
    output_base = (args.output_base.resolve() if args.output_base
                   else Path(env["OUTPUT_BASE_PATH"]).resolve())
    partition = args.partition or env.get("SLURM_PARTITION", "")
    if not partition:
        print("error: SLURM partition not set.", file=sys.stderr)
        return 2

    print(f"=== s_max scan fan-out: {name} ===")
    print(f"material:   {material}")
    print(f"particles:  {particles}")
    print(f"energies:   {len(energies)} cells, {energies[0]}..{energies[-1]} MeV")
    print(f"schedule:   base={base} anchor={anchor} MeV floor={floor}")
    if split_above is not None and target_per_job is not None:
        print(f"split:      E > {split_above} MeV → ~{target_per_job} events/job")
    print(f"output:     {output_base}")
    print(f"partition:  {partition}")
    print("")

    # (particle, energy, total_events, n_jobs, events_per_job)
    cells: List[Tuple[str, int, int, int, int]] = []
    for particle in particles:
        for energy in energies:
            n_events = events_for(energy, base, anchor, floor)
            n_jobs, evt_per_job = split_plan(
                n_events=n_events, energy_mev=energy,
                split_above_mev=split_above, target_per_job=target_per_job,
            )
            cells.append((particle, energy, n_events, n_jobs, evt_per_job))

    if args.test and cells:
        cells = cells[:1]

    total_events = sum(c[2] for c in cells)
    total_jobs = sum(c[3] for c in cells)
    print(f"total cells:   {len(cells)}")
    print(f"total jobs:    {total_jobs}")
    print(f"total events:  {total_events:,}")
    print("")

    prepared = submitted = skipped_existing = skipped_jobs = 0
    for particle, energy, n_events, n_jobs, evt_per_job in cells:
        cell_dir = output_base / material / particle / f"{energy}MeV"
        cell_dir.mkdir(parents=True, exist_ok=True)

        merged = cell_dir / "photonsim.root"
        if merged.is_file() and not args.no_skip_existing:
            print(f"  skip (merged exists): {merged}")
            skipped_existing += 1
            continue

        cell_name = f"smax_{material}_{particle}_{energy}MeV"
        cell_cfg = write_per_cell_config(
            cell_dir=cell_dir, material=material, particle=particle,
            energy_mev=energy, events_per_job=evt_per_job, n_jobs=n_jobs,
            name=cell_name,
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
                print(f"  submitted {job_label} ({evt_per_job} evts)  -> {r.stdout.strip()}")
            else:
                print(f"  prepared  {job_label} ({evt_per_job} evts)  -> {sb}")

    print("")
    print("=== Fan-out complete ===")
    print(f"Prepared:        {prepared}")
    print(f"Submitted:       {submitted}")
    print(f"Skipped cells:   {skipped_existing}")
    print(f"Skipped jobs:    {skipped_jobs}")
    print(f"Output root:     {output_base / material}")
    print("")
    print("Next: once all jobs finish, run merge.sh to hadd per-cell output_job_*.root")
    print("      into photonsim.root, then analyze.sh to fit s_max(E) and write")
    print("      PhotonSim/data/<material>/<particle>/smax_{data,fit}.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
