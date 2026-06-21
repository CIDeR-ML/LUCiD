#!/usr/bin/env python3
"""Dataprod fan-out — Python port of the legacy bash `generate_jobs.sh`.

Reads a `dataprod_*.json` config and emits one submit description file per
sub-job under `<OUTPUT_BASE>/<DETECTOR>/config_NNNNNN/[<E>MeV/]/`. The
output layout, file naming, and CLI surface are preserved from the bash
version so existing s3df workflows keep working bit-for-bit.

CLI surface:

    -c, --config         dataprod JSON config (required)
    -s, --submit         submit jobs (default: prepare only)
    -t, --test           test mode — one job, 2 events
    -g, --gpu            request 1 GPU per job
    -P, --partition      override SLURM_PARTITION / CONDOR_JOB_FLAVOUR;
                         a comma list (e.g. "roma:130,milano:272") round-robins
                         jobs across targets, weighted by the optional :N
    -o, --output-base    override OUTPUT_BASE_PATH from user_paths.sh
    -D, --detector       detector geometry (default: SK_like)
    -j, --job-id-start   job_id offset (default: 1)
    -N, --n-jobs         override n_jobs for this invocation
    --user-paths         path to user_paths.sh

The cluster (slurm/htcondor) is read from `CLUSTER` in user_paths.sh.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .cluster import get_adapter
from .user_paths import load_user_paths
# Side-effect: registers the HTCondor and NERSC adapters in the cluster factory.
from . import htcondor  # noqa: F401
from . import nersc  # noqa: F401


def _slugify(name: str) -> str:
    """Filesystem-safe slug. Mirrors bash: + → plus, - → minus, space → _."""
    s = name.replace("+", "plus").replace("-", "minus").replace(" ", "_")
    return re.sub(r"_+", "_", s)


def _parse_partition_spec(spec: str) -> List[tuple]:
    """Parse a partition/flavour spec into ``[(name, weight), ...]``.

    Accepts ``"roma:130,milano:272"`` (node-weighted round-robin),
    ``"roma,milano"`` (equal round-robin, weight 1 each), or a bare
    ``"roma"`` / ``"workday"`` (single target). A ``:N`` suffix counts as a
    weight only when ``N`` is a positive integer; anything else is treated as
    part of the name, so single SLURM partitions, HTCondor flavours, and
    empty specs pass through untouched (the multi-target path never engages
    for them). Returns ``[]`` for an empty/blank spec.
    """
    out: List[tuple] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, weight = tok, 1
        if ":" in tok:
            head, tail = tok.rsplit(":", 1)
            if head and tail.isdigit() and int(tail) > 0:
                name, weight = head, int(tail)
        out.append((name, weight))
    return out


class _WeightedRoundRobin:
    """Smooth weighted round-robin (nginx-style), deterministic per process.

    Successive ``next()`` calls return target names interleaved in proportion
    to their integer weights — e.g. weights (130, 272) yield milano roughly
    twice as often as roma, spread out rather than blocked. A single target
    degenerates to always returning that name.
    """

    def __init__(self, weighted: List[tuple]):
        self.names = [n for n, _ in weighted]
        self.weights = [w for _, w in weighted]
        self.total = sum(self.weights)
        self._current = [0] * len(self.names)
        self.counts = {n: 0 for n in self.names}

    def next(self) -> str:
        if len(self.names) == 1:
            self.counts[self.names[0]] += 1
            return self.names[0]
        for i, w in enumerate(self.weights):
            self._current[i] += w
        best = max(range(len(self.names)), key=lambda i: self._current[i])
        self._current[best] -= self.total
        name = self.names[best]
        self.counts[name] += 1
        return name


def _events_schedule_plan(cfg: dict, n_events_default: int, n_jobs_default: int,
                          test: bool) -> tuple:
    """Return (n_events, n_jobs, note).

    If the config has an `events_schedule` block AND we're not in test mode,
    derive (events_per_job, n_jobs) from (events_per_dataset, target_seconds,
    seconds_per_event). Otherwise return the flat config values unchanged.
    """
    sched = cfg.get("events_schedule") or {}
    e_total = sched.get("events_per_dataset")
    t_target = sched.get("target_seconds_per_job")
    s_per_e = sched.get("seconds_per_event")
    if e_total is None or t_target is None or s_per_e is None or test:
        return n_events_default, n_jobs_default, ""
    e_per_job = max(1, int(float(t_target) / float(s_per_e)))
    n_jobs = max(1, (int(e_total) + e_per_job - 1) // e_per_job)
    note = (f"events_schedule: {e_total} evts / {t_target}s target "
            f"@ {s_per_e}s per event")
    return e_per_job, n_jobs, note


def _write_readme(config_dir: Path, *, name: str, desc: str, config_id: str,
                  material: str, energy_dist: str, n_jobs: int, n_events: int,
                  n_particles: int) -> None:
    body = (
        f"# Dataset: {name}\n\n"
        f"- **Config number**: {config_id}\n"
        f"- **Description**: {desc}\n"
        f"- **Material**: {material}\n"
        f"- **Energy distribution**: {energy_dist}\n"
        f"- **Jobs**: {n_jobs}  ×  **Events/job**: {n_events}  =  "
        f"**Total events**: {n_jobs * n_events}\n"
        f"- **Particles per event**: {n_particles}\n"
        f"- **Generated**: "
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"## Layout\n\n"
        f"This directory is one LUCiD dataset. Each job contributes one v3 "
        f"batch (`file_index = job_id - 1`) consisting of four files spread "
        f"across `sensor/`, `hits/`, `step/`, `labl/`. See "
        f"`LUCiD/docs/LUCID_DATASET.md` for the schema.\n"
    )
    (config_dir / "README.md").write_text(body)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", type=Path, required=True,
                   help="dataprod JSON config (required)")
    p.add_argument("-s", "--submit", action="store_true",
                   help="submit jobs (default: prepare only)")
    p.add_argument("-t", "--test", action="store_true",
                   help="test mode — one job, 2 events")
    p.add_argument("-g", "--gpu", action="store_true",
                   help="request 1 GPU per job")
    p.add_argument("-P", "--partition", type=str, default=None,
                   help="override SLURM_PARTITION / CONDOR_JOB_FLAVOUR. A "
                        "comma list (e.g. 'roma:130,milano:272') round-robins "
                        "jobs across targets, weighted by the optional :N.")
    p.add_argument("-o", "--output-base", type=Path, default=None,
                   help="override OUTPUT_BASE_PATH from user_paths.sh")
    p.add_argument("-D", "--detector", type=str, default="SK_like",
                   help="detector geometry (default: SK_like)")
    p.add_argument("-j", "--job-id-start", type=int, default=1,
                   help="job_id offset (default: 1). Set to N+1 for a top-up "
                        "wave on a dataset that already has N sub-jobs.")
    p.add_argument("-N", "--n-jobs-override", type=int, default=None,
                   help="override n_jobs for this invocation. With "
                        "events_schedule active, this keeps the schedule's "
                        "events_per_job but forces n_jobs = N.")
    p.add_argument("--user-paths", type=Path, default=None,
                   help="path to user_paths.sh (default: next to the "
                        "invoking jobs/ dir)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.config.is_file():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        return 1
    cfg = json.loads(args.config.read_text())

    # The bash shim resolves --user-paths to its own jobs/.. dir; if invoked
    # directly with no flag, fall back to looking for one near the config.
    user_paths = args.user_paths
    if user_paths is None:
        # Best-effort: <config>/../user_paths.sh, then env's default
        cand = args.config.parent.parent / "user_paths.sh"
        if cand.is_file():
            user_paths = cand
        else:
            print("Error: --user-paths not provided and no user_paths.sh "
                  "found near config.", file=sys.stderr)
            return 1
    env = load_user_paths(user_paths)
    adapter = get_adapter(env)
    if args.submit and shutil.which(adapter.submit_cmd) is None:
        print(f"Error: {adapter.submit_cmd} not found on PATH.",
              file=sys.stderr)
        return 1

    config_number = cfg.get("config_number", -1)
    name = cfg["name"]
    desc = cfg.get("description", "")
    material = cfg["material"]
    primary_source = cfg.get("primary_source", "particle_gun")
    energy_dist = cfg.get("energy_distribution", "uniform")
    run_lucid = bool(cfg.get("run_lucid", True))
    n_events = int(cfg["n_events_per_job"])
    n_jobs = int(cfg["n_jobs"])
    n_particles = len(cfg.get("particles") or [])
    use_config_number = bool(cfg.get("use_config_number", True))

    if primary_source != "genie" and energy_dist not in ("monoenergetic", "uniform"):
        print(f"Error: energy_distribution must be 'monoenergetic' or "
              f"'uniform' (got: {energy_dist})", file=sys.stderr)
        return 1
    if use_config_number and (config_number is None or config_number == -1):
        print("Error: config_number is required when use_config_number is true",
              file=sys.stderr)
        return 1

    if not env.get("LUCID_IMAGE_PATH"):
        print("Error: LUCID_IMAGE_PATH not set in user_paths.sh.",
              file=sys.stderr)
        return 1
    if not Path(env["LUCID_IMAGE_PATH"]).is_file():
        print(f"Error: LUCID_IMAGE_PATH={env['LUCID_IMAGE_PATH']} does not "
              f"exist.", file=sys.stderr)
        return 1

    n_events, n_jobs, schedule_note = _events_schedule_plan(
        cfg, n_events, n_jobs, args.test,
    )
    if args.test:
        n_events = 2
    if args.n_jobs_override is not None:
        n_jobs = args.n_jobs_override

    # Effective resources
    output_base = (args.output_base.resolve() if args.output_base
                   else Path(env["OUTPUT_BASE_PATH"]).resolve())
    partition = (args.partition or env.get("SLURM_PARTITION")
                 or env.get("CONDOR_JOB_FLAVOUR", ""))
    # A comma-separated spec round-robins jobs across targets, weighted by an
    # optional `:N` per target (e.g. node counts). A single target (the usual
    # case on NERSC/HTCondor) degenerates to the previous constant behaviour;
    # an empty spec is passed through verbatim.
    parsed_parts = _parse_partition_spec(partition)
    wrr = _WeightedRoundRobin(parsed_parts) if parsed_parts else None
    multi_partition = len(parsed_parts) > 1

    detector = args.detector
    output_base_dir = output_base / detector
    if use_config_number:
        config_dir = output_base_dir / f"config_{int(config_number):06d}"
        config_id = f"{int(config_number):06d}"
    else:
        config_dir = output_base_dir
        config_id = "N/A (use_config_number: false)"
    config_dir.mkdir(parents=True, exist_ok=True)
    if run_lucid:
        for sub in ("sensor", "hits", "step", "labl"):
            (config_dir / sub).mkdir(parents=True, exist_ok=True)

    # Print summary
    print(f"=== lucid-run-job {adapter.name} fan-out ===\n")
    if use_config_number:
        print(f"Configuration: {name} (config_{config_id})")
    else:
        print(f"Configuration: {name} (no config subfolder)")
    print(f"Description:   {desc}")
    print(f"Energy dist:   {energy_dist}")
    print(f"Particles:     {n_particles}")
    if schedule_note:
        print(f"Schedule:      {schedule_note}")
    print(f"Jobs:          {n_jobs} (events/job={n_events})")
    if multi_partition:
        plan = ", ".join(f"{n}:{w}" for n, w in parsed_parts)
        print(f"Partition:     {partition}  "
              f"(weighted round-robin → {plan})  "
              f"GPUs={'1' if args.gpu else env.get('DEFAULT_GPUS', '0')}")
    else:
        print(f"Partition:     {partition}  GPUs={'1' if args.gpu else env.get('DEFAULT_GPUS', '0')}")
    print(f"Detector:      {detector}")
    print(f"Output dir:    {config_dir}\n")

    slug = _slugify(name)
    submitted = 0
    prepared = 0
    skip_lucid_flag = not run_lucid

    # Energy list
    if energy_dist == "uniform":
        energies: List[Optional[float]] = [None]
        out_dirs = [config_dir]
    elif energy_dist == "monoenergetic":
        single_e = cfg.get("energy_MeV")
        scan = cfg.get("energy_scan") or {}
        if single_e is not None:
            energies = [float(single_e)]
        elif "start_MeV" in scan and "stop_MeV" in scan and "step_MeV" in scan:
            start = scan["start_MeV"]; stop = scan["stop_MeV"]; step = scan["step_MeV"]
            energies = []
            e = start
            # Match bash's `for ((e=start; e<=stop; e+=step))` semantics.
            while e <= stop:
                energies.append(float(e))
                e += step
        else:
            print("Error: monoenergetic requires 'energy_MeV' or "
                  "'energy_scan.{start,stop,step}_MeV'", file=sys.stderr)
            return 1
        out_dirs = []
        for energy in energies:
            e_int = int(round(energy))
            d = config_dir / f"{e_int}MeV"
            d.mkdir(parents=True, exist_ok=True)
            if run_lucid:
                for sub in ("sensor", "hits", "step", "labl"):
                    (d / sub).mkdir(parents=True, exist_ok=True)
            out_dirs.append(d)
    else:  # genie or other — fall back to uniform-style layout
        energies = [None]
        out_dirs = [config_dir]

    jobs_to_process = 1 if args.test else n_jobs

    for energy, out_dir in zip(energies, out_dirs):
        e_label = f"{int(round(energy))}MeV_" if energy is not None else ""
        j_end = args.job_id_start + jobs_to_process - 1
        for j in range(args.job_id_start, j_end + 1):
            job_id = f"{j:06d}"
            if use_config_number:
                job_name = f"photonsim_config{config_id}_{e_label}{job_id}"
            else:
                job_name = f"photonsim_{slug}_{e_label}{job_id}"

            job_partition = wrr.next() if wrr is not None else partition
            body = adapter.render_dataprod_job(
                cell_dir=out_dir, config_path=args.config, detector=detector,
                job_id=job_id, job_name=job_name, partition=job_partition,
                use_gpu=args.gpu, test=args.test, skip_lucid=skip_lucid_flag,
                n_events=n_events if schedule_note else None,
                override_energy_mev=energy,
            )
            sb_path = out_dir / f"submit_job_{job_id}.{adapter.submit_extension}"
            sb_path.write_text(body)
            sb_path.chmod(0o755)
            # Stable marker for the bash shim's host-side submission pass.
            print(f"[PREPARED] {sb_path}")
            prepared += 1

            if args.submit:
                r = subprocess.run([adapter.submit_cmd, str(sb_path)],
                                   capture_output=True, text=True)
                if r.returncode == 0:
                    submitted += 1
                else:
                    print(f"  FAILED {adapter.submit_cmd} for {job_name}: "
                          f"{r.stderr.strip()}", file=sys.stderr)
        if args.test:
            break  # one energy in test mode

    _write_readme(
        config_dir, name=name, desc=desc, config_id=config_id,
        material=material, energy_dist=energy_dist,
        n_jobs=n_jobs, n_events=n_events, n_particles=n_particles,
    )

    print("\n=== Fan-out complete ===")
    print(f"Prepared:  {prepared}")
    print(f"Submitted: {submitted}")
    if multi_partition and wrr is not None:
        split = ", ".join(f"{n}={wrr.counts[n]}" for n, _ in parsed_parts)
        print(f"Partition split: {split}")
    print(f"Config dir: {config_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
