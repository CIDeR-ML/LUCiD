#!/usr/bin/env python3
"""SLURM fan-out for the s_max parametrisation scan (Stage 0).

Prerequisite for SIREN-input generation: walks an energy grid for each
particle, runs PhotonSim with `/photon/storeIndividual false` + fixed +Z
direction, and writes `photonsim.root` files containing the
`PhotonHist_Distance` 1D histogram. Those ROOTs are then fed to
`analyze.sh` (which calls `PhotonSim/tools/smax/analyze_smax.py`) to
produce the per-particle `smax_data.csv` + `smax_fit.csv` under
`PhotonSim/data/<material>/<particle>/`.

The planning math (event-count schedule, optional split for high-E cells)
lives in `lucid.production.cluster_common.smax_planning`; this script
only owns the SLURM-specific submit-script writing and `sbatch` call.

`/output/smax` is intentionally **not** emitted here — at this stage we
are *computing* s_max, not consuming it.

Output layout (PhotonSim-compatible, matches scan_smax.py):

    <OUTPUT_BASE>/<material>/<particle>/<E>MeV/photonsim.root

Run `merge.sh <OUTPUT_BASE>` after all jobs finish to `hadd` the
per-job ROOTs into `photonsim.root` for each cell.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Host-side entrypoint: ensure the LUCiD checkout root is on sys.path.
SCRIPT_DIR = Path(__file__).parent
LUCID_ROOT = SCRIPT_DIR.resolve().parents[3]
if str(LUCID_ROOT) not in sys.path:
    sys.path.insert(0, str(LUCID_ROOT))

from lucid.production.cluster_common import htcondor  # noqa: E402, F401
from lucid.production.cluster_common.cluster import get_adapter  # noqa: E402
from lucid.production.cluster_common.smax_planning import (  # noqa: E402
    events_for, parse_schedule, split_plan, write_per_cell_config,
)
from lucid.production.cluster_common.user_paths import load_user_paths  # noqa: E402


JOBS_DIR = SCRIPT_DIR.parent       # the cluster-portable jobs/ dir
USER_PATHS_DEFAULT = JOBS_DIR / "user_paths.sh"


def write_submit(*, adapter, cell_dir: Path, cell_cfg: Path, energy_mev: int,
                 job_name: str, job_id: int, sb_basename: str,
                 partition: str, use_gpu: bool) -> Path:
    body = adapter.render_smax_cell(
        cell_dir=cell_dir, cell_cfg=cell_cfg, energy_mev=energy_mev,
        job_name=job_name, job_id=job_id, partition=partition,
        use_gpu=use_gpu,
    )
    out = cell_dir / f"{sb_basename}.{adapter.submit_extension}"
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
                   help="Output root override. Default: "
                        "<SIREN_OUTPUT_BASE_PATH>/smax_parametrization if set, "
                        "else OUTPUT_BASE_PATH.")
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

    if not args.config.is_file():
        print(f"error: config not found: {args.config}", file=sys.stderr)
        return 2
    cfg = json.loads(args.config.read_text())

    name = cfg["name"]
    material = cfg["material"]
    particles = [p["type"] for p in cfg["particles"]]
    energies: List[int] = list(cfg["energy_list_MeV"])
    base, anchor, floor, split_above, target_per_job = parse_schedule(cfg)

    env = load_user_paths(args.user_paths)
    adapter = get_adapter(env)
    if args.submit and shutil.which(adapter.submit_cmd) is None:
        print(f"error: {adapter.submit_cmd} not found on PATH.", file=sys.stderr)
        return 2

    if args.output_base:
        output_base = args.output_base.resolve()
    elif env.get("SIREN_OUTPUT_BASE_PATH"):
        # Stage-specific subdir under the SIREN root keeps smax outputs
        # disjoint from siren_inputs outputs (both layer <material>/...).
        output_base = (Path(env["SIREN_OUTPUT_BASE_PATH"])
                       / "smax_parametrization").resolve()
    else:
        output_base = Path(env["OUTPUT_BASE_PATH"]).resolve()
    partition = args.partition or env.get("SLURM_PARTITION") or env.get("CONDOR_JOB_FLAVOUR", "")
    if not partition:
        print("error: queue partition/flavour not set "
              "(SLURM_PARTITION or CONDOR_JOB_FLAVOUR).", file=sys.stderr)
        return 2

    print(f"=== s_max scan fan-out: {name} ===")
    print(f"cluster:    {adapter.name}")
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
                sb_basename = "submit"
                job_label = cell_name
            else:
                sb_basename = f"submit_job_{job_id:03d}"
                job_label = f"{cell_name}_j{job_id:03d}"

            out_root = cell_dir / f"output_job_{job_id:06d}.root"
            if out_root.is_file() and not args.no_skip_existing:
                print(f"  skip (job exists): {out_root.name}")
                skipped_jobs += 1
                continue

            sb = write_submit(
                adapter=adapter, cell_dir=cell_dir, cell_cfg=cell_cfg,
                energy_mev=energy, job_name=job_label, job_id=job_id,
                sb_basename=sb_basename, partition=partition, use_gpu=args.gpu,
            )
            # Stable marker for the bash shim's host-side submission pass.
            print(f"[PREPARED] {sb}")
            prepared += 1

            if args.submit:
                r = subprocess.run([adapter.submit_cmd, str(sb)],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"  FAILED {adapter.submit_cmd} for {job_label}: "
                          f"{r.stderr.strip()}", file=sys.stderr)
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
