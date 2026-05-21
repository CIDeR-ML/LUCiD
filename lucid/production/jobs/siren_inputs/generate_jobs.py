#!/usr/bin/env python3
"""Cluster fan-out for SIREN-input PhotonSim jobs (s/s_max histogram axis).

Reads a SIREN-input JSON config (material + particles + non-uniform energy
list) and submits one or more sub-jobs per (particle, energy) cell. The
planning math (s_max parametrisation, time-based split, per-cell config
emission) lives in `lucid.production.cluster_common.siren_planning`; the
cluster-specific submit-description template lives in
`lucid.production.cluster_common.cluster.ClusterAdapter` subclasses
(`SlurmAdapter`, `HTCondorAdapter`), selected at runtime from `CLUSTER=`
in the active `user_paths.sh`.

Output layout (PhotonSim-compatible, after merge.sh):

    <OUTPUT_BASE>/<material>/<particle>/<E>MeV/photonsim.root

See README.md for the config schema and the smoke-test → full-scan loop.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# This script is also used as a host-side entrypoint (sbatch / condor_submit
# lives on the host, not inside the container), so `lucid` may not be on
# sys.path. Add the LUCiD checkout root explicitly.
SCRIPT_DIR = Path(__file__).parent
LUCID_ROOT = SCRIPT_DIR.resolve().parents[3]   # .../LUCiD/  (real path)
if str(LUCID_ROOT) not in sys.path:
    sys.path.insert(0, str(LUCID_ROOT))

from lucid.production.cluster_common import htcondor  # noqa: E402, F401  registers adapter
from lucid.production.cluster_common.cluster import get_adapter  # noqa: E402
from lucid.production.cluster_common.siren_planning import (  # noqa: E402
    build_cells, find_photonsim_dir, parse_schedule, write_per_cell_config,
)
from lucid.production.cluster_common.user_paths import load_user_paths  # noqa: E402


JOBS_DIR = SCRIPT_DIR.parent       # the cluster-portable jobs/ dir
USER_PATHS_DEFAULT = JOBS_DIR / "user_paths.sh"


def write_submit(*, adapter, cell_dir: Path, cell_cfg: Path, energy_mev: int,
                 job_name: str, job_id: int, sb_basename: str,
                 partition: str, use_gpu: bool) -> Path:
    """Render and write the cluster's submit-description file for one sub-job."""
    body = adapter.render_siren_cell(
        cell_dir=cell_dir, cell_cfg=cell_cfg, energy_mev=energy_mev,
        job_name=job_name, job_id=job_id, partition=partition,
        use_gpu=use_gpu,
    )
    out = cell_dir / f"{sb_basename}.{adapter.submit_extension}"
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
                   help="Output root override. Default: "
                        "<SIREN_OUTPUT_BASE_PATH>/training_inputs if set, "
                        "else OUTPUT_BASE_PATH.")
    p.add_argument("-P", "--partition", type=str, default=None,
                   help="Override SLURM_PARTITION from user_paths.sh. "
                        "Comma-separated values (e.g. 'roma,milano') round-"
                        "robin sub-jobs across partitions to drain faster.")
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
    p.add_argument("--min-energy", type=int, default=None,
                   help="Filter the config's energy_list to E >= this value "
                        "(MeV). Useful for restricted re-runs.")
    p.add_argument("--max-energy", type=int, default=None,
                   help="Filter the config's energy_list to E < this value "
                        "(MeV). Useful for restricted re-runs (e.g. when only "
                        "low-E cells need regenerating after a piecewise "
                        "smax refit).")
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
    if args.min_energy is not None:
        energies = [e for e in energies if e >= args.min_energy]
    if args.max_energy is not None:
        energies = [e for e in energies if e < args.max_energy]
    if not energies:
        print("error: no energies left after --min-energy/--max-energy filter",
              file=sys.stderr)
        return 2
    include_extrapolated = bool(
        args.include_extrapolated or cfg.get("include_extrapolated", False)
    )
    events_per_cell, events_ranges, target_s, a_s, b_s_per_mev = parse_schedule(cfg)

    env = load_user_paths(args.user_paths)
    adapter = get_adapter(env)
    if args.submit and shutil.which(adapter.submit_cmd) is None:
        print(f"error: {adapter.submit_cmd} not found on PATH; "
              f"submission requires this cluster.", file=sys.stderr)
        return 2

    # find_photonsim_dir walks up looking for a sibling PhotonSim/ checkout;
    # use the resolved location so the walk happens off the real path.
    photonsim_dir = find_photonsim_dir(env, args.photonsim_dir,
                                        SCRIPT_DIR.resolve())

    if args.output_base:
        output_base = args.output_base.resolve()
    elif env.get("SIREN_OUTPUT_BASE_PATH"):
        # Stage-specific subdir under the SIREN root keeps siren_inputs
        # outputs disjoint from smax outputs (both layer <material>/...).
        output_base = (Path(env["SIREN_OUTPUT_BASE_PATH"])
                       / "training_inputs").resolve()
    else:
        output_base = Path(env["OUTPUT_BASE_PATH"]).resolve()
    # `partition` is SLURM-flavoured terminology; HTCondor reads the equivalent
    # from CONDOR_JOB_FLAVOUR via the adapter. The CLI -P / env var is
    # interpreted by the adapter's render_* method.
    partition_spec = args.partition or env.get("SLURM_PARTITION") or env.get("CONDOR_JOB_FLAVOUR", "")
    if not partition_spec:
        print("error: queue partition/flavour not set "
              "(SLURM_PARTITION or CONDOR_JOB_FLAVOUR in user_paths.sh, or -P).",
              file=sys.stderr)
        return 2
    partitions = [p.strip() for p in partition_spec.split(",") if p.strip()]
    if not partitions:
        print(f"error: empty partition spec: {partition_spec!r}", file=sys.stderr)
        return 2

    print(f"=== SIREN-input fan-out: {name} ===")
    print(f"cluster:      {adapter.name}")
    print(f"material:     {material}")
    print(f"particles:    {particles}")
    print(f"energies:     {len(energies)} cells, "
          f"{energies[0]}..{energies[-1]} MeV")
    if events_ranges:
        ranges_desc = ", ".join(f"<{e}MeV→{n:,}" for e, n in events_ranges)
        print(f"events/cell:  {events_per_cell:,} (default; ranges: {ranges_desc})")
    else:
        print(f"events/cell:  {events_per_cell:,}")
    if target_s is not None:
        print(f"target/job:   {target_s:.0f} s "
              f"(time model: t/event = {a_s:g} + {b_s_per_mev:g} * E_MeV)")
    print(f"output base:  {output_base}")
    print(f"PhotonSim:    {photonsim_dir}")
    print(f"partition:    {partition_spec}"
          + (f"  (round-robin across {len(partitions)} partitions)"
             if len(partitions) > 1 else ""))
    print("")

    cells, skipped = build_cells(
        particles=particles, energies=energies, photonsim_dir=photonsim_dir,
        material=material, events_per_cell=events_per_cell,
        events_per_cell_ranges=events_ranges,
        target_seconds_per_job=target_s, a_s=a_s, b_s_per_mev=b_s_per_mev,
        include_extrapolated=include_extrapolated,
    )

    if args.test and cells:
        cells = cells[:1]

    if not cells:
        print("error: no cells to submit "
              f"(skipped {len(skipped)}). Use --include-extrapolated to force.",
              file=sys.stderr)
        for s in skipped:
            print(f"  SKIP {s}", file=sys.stderr)
        return 1

    total_jobs = sum(c.n_jobs for c in cells)
    total_events = sum(c.n_jobs * c.events_per_job for c in cells)
    print(f"total cells:   {len(cells)}")
    print(f"total jobs:    {total_jobs}")
    print(f"total events:  {total_events:,}")
    print("")

    prepared = submitted = skipped_existing = skipped_jobs = 0
    n_emitted = 0
    for cell in cells:
        cell_dir = output_base / material / cell.particle / f"{cell.energy_mev}MeV"
        cell_dir.mkdir(parents=True, exist_ok=True)

        merged = cell_dir / "photonsim.root"
        if merged.is_file() and not args.no_skip_existing:
            print(f"  skip (merged exists): {merged}")
            skipped_existing += 1
            continue

        cell_name = f"siren_{material}_{cell.particle}_{cell.energy_mev}MeV"
        cell_cfg = write_per_cell_config(
            cell_dir=cell_dir, material=material, particle=cell.particle,
            energy_mev=cell.energy_mev, smax_mm=cell.smax_mm,
            events_per_job=cell.events_per_job, n_jobs=cell.n_jobs,
            name=cell_name,
        )

        for job_id in range(1, cell.n_jobs + 1):
            if cell.n_jobs == 1:
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

            sub_partition = partitions[n_emitted % len(partitions)]
            sb = write_submit(
                adapter=adapter, cell_dir=cell_dir, cell_cfg=cell_cfg,
                energy_mev=cell.energy_mev, job_name=job_label, job_id=job_id,
                sb_basename=sb_basename, partition=sub_partition,
                use_gpu=args.gpu,
            )
            # Stable marker for the bash shim's host-side submission pass.
            print(f"[PREPARED] {sb}")
            n_emitted += 1
            prepared += 1

            if args.submit:
                r = subprocess.run([adapter.submit_cmd, str(sb)],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"  FAILED {adapter.submit_cmd} for {job_label}: "
                          f"{r.stderr.strip()}", file=sys.stderr)
                    continue
                submitted += 1
                print(f"  submitted {job_label}  (s_max={cell.smax_mm:.1f} mm, "
                      f"{cell.events_per_job} evts)  -> {r.stdout.strip()}")
            else:
                print(f"  prepared  {job_label}  (s_max={cell.smax_mm:.1f} mm, "
                      f"{cell.events_per_job} evts)  -> {sb}")

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
