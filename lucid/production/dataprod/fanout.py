#!/usr/bin/env python3
"""Dataprod fan-out: one dataset config into per-sub-job submit files.

Reads a `dataprod_*.json` config and emits one submit description file per
sub-job under `<OUTPUT_BASE>/<DETECTOR>/<BLOCK>/[<split>/]config_NN/[<E>MeV/]/`, or under
`<OUTPUT_BASE>/<DETECTOR>/_test/...` in test mode (`-t`). Run it through
`lucid/production/platform/run_in_container.sh`, which submits from the host.

CLI surface:

    -c, --config         dataprod JSON config (required)
    -s, --submit         submit jobs (default: prepare only)
    -t, --test           test mode — one job, 2 events, under
                         <OUTPUT_BASE>/<DETECTOR>/_test/ so it cannot
                         overwrite a production batch
    -g, --gpu            request 1 GPU per job
    -P, --partition      override SLURM_PARTITION / CONDOR_JOB_FLAVOUR;
                         a comma list (e.g. "roma:130,milano:272") round-robins
                         jobs across targets, weighted by the optional :N
    -o, --output-base    override OUTPUT_BASE_PATH from user_paths.sh
    -D, --detector       detector geometry; required unless the config
                         declares its own 'detector' (which then wins)
    -j, --job-id-start   job_id offset (default: 1)
    -N, --n-jobs-override  override n_jobs for this invocation
    --user-paths         path to user_paths.sh

The cluster (slurm/htcondor) is read from `CLUSTER` in user_paths.sh.
"""

from __future__ import annotations

import argparse
import datetime
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from lucid.production.dataprod.generators.supernova import sn_orderings
from lucid.production.dataprod.run_job import _load_config, _readout
from lucid.production.dataprod.run_job import primary_source as _primary_source
from lucid.production.platform.adapters import get_adapter
from lucid.production.platform.user_paths import load_user_paths


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
        f"This directory is one LUCiD dataset. Each job contributes one "
        f"batch (`file_index = job_id - 1`) consisting of four files spread "
        f"across `sensor/`, `hits/`, `step/`, `labl/`. See "
        f"`LUCiD/docs/reference/dataset-schema.md` for the schema.\n"
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
                   help="test mode — one job, 2 events, written under "
                        "<OUTPUT_BASE>/<DETECTOR>/_test/ instead of the "
                        "production tree. Before, -t reused job_id 000001 in "
                        "the production cell: it overwrote submit_job_000001 "
                        "and its batch with a 2-event version, and verify_jobs "
                        "then read the rewritten sbatch and accepted it.")
    p.add_argument("-g", "--gpu", action="store_true",
                   help="request 1 GPU per job")
    p.add_argument("-P", "--partition", type=str, default=None,
                   help="override SLURM_PARTITION / CONDOR_JOB_FLAVOUR. A "
                        "comma list (e.g. 'roma:130,milano:272') round-robins "
                        "jobs across targets, weighted by the optional :N.")
    p.add_argument("-o", "--output-base", type=Path, default=None,
                   help="override OUTPUT_BASE_PATH from user_paths.sh")
    p.add_argument("-D", "--detector", type=str, default=None,
                   help="detector geometry (e.g. SK_WAND, HK_WAND). Required "
                        "unless the dataset config declares its own 'detector'.")
    p.add_argument("--force-overwrite", action="store_true",
                   help="write into a cell that already holds submit files "
                        "generated from a *different* config. Four configs "
                        "whose names share a numeric prefix (02_mu, 02_mu_fast, "
                        "02_mu_metrics, 02_mu_smoke) all resolve to one "
                        "config_02/ cell, so without this the second one "
                        "refuses rather than overwriting the first one's "
                        "batch 0.")
    p.add_argument("-j", "--job-id-start", type=int, default=1,
                   help="job_id offset (default: 1). Set to N+1 for a top-up "
                        "wave on a dataset that already has N sub-jobs.")
    p.add_argument("-N", "--n-jobs-override", type=int, default=None,
                   help="override n_jobs for this invocation: each split "
                        "keeps its events_per_job but runs N jobs.")
    p.add_argument("--user-paths", type=Path, default=None,
                   help="path to user_paths.sh (default: "
                        "lucid/production/platform/user_paths.sh)")
    return p.parse_args(argv)


def _refuse_foreign_cell(config_dir: Path, config_path: Path,
                         force: bool) -> int:
    """Refuse a cell whose existing submit files came from another config.

    The cell name is ``config_<N>``, where ``N`` is the dataset config's
    numeric filename prefix, so ``02_mu``, ``02_mu_fast``, ``02_mu_metrics``
    and ``02_mu_smoke`` all resolve to ``config_02/`` and each fan-out
    overwrote the previous one's ``submit_job_000001`` and batch 0.

    Parameters
    ----------
    config_dir : Path
        The cell about to be written.
    config_path : Path
        The dataset config driving this fan-out.
    force : bool
        ``True`` writes anyway (``--force-overwrite``).

    Returns
    -------
    int
        ``0`` to proceed, ``1`` to abort.
    """
    if force:
        return 0
    want = f'--config "{config_path}"'
    for sb in sorted(config_dir.rglob("submit_job_*")):
        try:
            body = sb.read_text()
        except OSError:
            continue
        if "--config" not in body or want in body:
            continue
        other = re.search(r'--config "([^"]*)"', body)
        print(f"Error: {config_dir} already holds submit files generated from "
              f"{(other.group(1) if other else 'another config')!r}, not from "
              f"{config_path}. Two dataset configs whose filenames share a "
              f"numeric prefix resolve to the same config_<N> cell, and "
              f"writing here would overwrite that dataset's submit files and "
              f"its batches. Point -o somewhere else, or pass "
              f"--force-overwrite if that is what you mean.", file=sys.stderr)
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.config.is_file():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        return 1
    # The checks lucid-run-job makes at load (required keys, primary_source), here before
    # anything is submitted.
    try:
        cfg = _load_config(str(args.config))
    except (OSError, ValueError) as e:
        print(f"Error loading config {args.config}: {e}", file=sys.stderr)
        return 1

    # Without --user-paths, use the platform's own copy.
    user_paths = args.user_paths
    if user_paths is None:
        user_paths = Path(__file__).resolve().parents[1] / "platform" / "user_paths.sh"
        if not user_paths.is_file():
            print(f"Error: --user-paths not provided and no {user_paths}.", file=sys.stderr)
            return 1
    env = load_user_paths(user_paths)
    adapter = get_adapter(env)
    if args.submit and shutil.which(adapter.submit_cmd) is None:
        print(f"Error: {adapter.submit_cmd} not found on PATH.",
              file=sys.stderr)
        return 1

    # Config identity. A dataset config lives in a per-block folder (GeV / Solar /
    # SN / Test) and carries nominal_{train,test}; the block is the parent folder and
    # the 2-digit config number is the filename prefix (NN_name.json).
    if "nominal_test" not in cfg and "nominal_train" not in cfg:
        print(f"Error: {args.config} sets neither nominal_train nor nominal_test",
              file=sys.stderr)
        return 1
    block = args.config.parent.name
    m = re.match(r"^(\d+)_", args.config.stem)
    if not m:
        print(f"Error: block config filename must start with 'NN_' "
              f"(got {args.config.stem!r})", file=sys.stderr)
        return 1
    config_number = int(m.group(1))

    name = cfg["name"]
    desc = cfg.get("description", "")
    material = cfg["material"]
    primary_source = _primary_source(cfg)
    energy_dist = cfg.get("energy_distribution", "uniform")
    run_lucid = bool(cfg.get("run_lucid", True))
    n_particles = len(cfg.get("particles") or [])

    if primary_source not in ("genie", "supernova") and energy_dist not in ("monoenergetic", "uniform"):
        print(f"Error: energy_distribution must be 'monoenergetic' or "
              f"'uniform' (got: {energy_dist})", file=sys.stderr)
        return 1

    if not env.get("LUCID_IMAGE_PATH"):
        print("Error: LUCID_IMAGE_PATH not set in user_paths.sh.",
              file=sys.stderr)
        return 1
    if not Path(env["LUCID_IMAGE_PATH"]).is_file():
        print(f"Error: LUCID_IMAGE_PATH={env['LUCID_IMAGE_PATH']} does not "
              f"exist.", file=sys.stderr)
        return 1

    # Split plan. GeV / Solar carry a train/test layer (each from its own
    # nominal_*); SN / Test are flat (one dataset from nominal_test). Per-split sizing:
    #   n_jobs = ceil(nominal / (target_seconds_per_job / seconds_per_event));
    # supernova is one all-at-once burst per job. Train/test get disjoint master
    # seeds so the two datasets never share an event.
    SPLIT_BLOCKS = {"GeV", "Solar"}
    SEED_BASE = {"train": 10_000_000, "test": 20_000_000, None: 30_000_000}

    def _plan(n_total):
        if args.test:
            return 2, 1
        if primary_source == "supernova":
            return 1, max(1, int(n_total))
        spe = cfg.get("seconds_per_event")
        tgt = float(cfg.get("target_seconds_per_job", 3600))
        epj = (max(1, int(tgt / float(spe))) if spe
               else int(cfg.get("n_events_per_job", n_total) or n_total))
        return epj, max(1, (int(n_total) + epj - 1) // epj)

    # each entry: (split_name|None, path_component|None, n_events, n_jobs, seed_base)
    splits: List[tuple] = []
    if block in SPLIT_BLOCKS:
        for sp in ("train", "test"):
            n_tot = int(cfg.get(f"nominal_{sp}", 0) or 0)
            if n_tot <= 0:
                continue
            epj, njob = _plan(n_tot)
            splits.append((sp, sp, epj, njob, SEED_BASE[sp]))
    else:
        epj, njob = _plan(int(cfg.get("nominal_test", 0) or 0))
        splits.append((None, None, epj, njob, SEED_BASE[None]))
    if args.n_jobs_override is not None:
        splits = [(a, b, c, args.n_jobs_override, e) for (a, b, c, _, e) in splits]
    if not splits:
        print("Error: nothing to generate (no split has nominal > 0)", file=sys.stderr)
        return 1

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

    # The dataset config owns its detector (new-style); --detector is the fallback.
    # Neither is a silent default: picking the wrong detector changes the geometry,
    # the digitizer and the trigger, so an unspecified detector is a hard error.
    detector = cfg.get("detector", args.detector)
    # A config that declares its own detector wins over -D. Say so: it used to
    # win silently, so `-D HK_WAND` on an SK_WAND config produced an SK_WAND
    # dataset in an SK_WAND directory with no line of output mentioning it.
    if cfg.get("detector") and args.detector and args.detector != cfg["detector"]:
        print(f"Note: the dataset config declares detector "
              f"{cfg['detector']!r}, which overrides -D {args.detector!r}.")
    if not detector:
        print(f"Error: detector needs to be specified - the dataset config "
              f"{args.config} declares no 'detector' and no -D/--detector was "
              f"given (e.g. -D SK_WAND).", file=sys.stderr)
        return 1
    # The readout lucid-run-job will resolve (dataset config, else the detector physics config;
    # a block in neither is an error), so a config that cannot run is refused before submission.
    if run_lucid:
        try:
            _readout(cfg, detector)
        except (OSError, ValueError) as e:
            print(f"Error: {args.config} with detector {detector!r}: {e}", file=sys.stderr)
            return 1
    print(f"=== lucid-run-job {adapter.name} fan-out ===\n")
    print(f"Configuration: {name}  [{block}/config_{config_number:02d}]")
    print(f"Description:   {desc}")
    print(f"Detector:      {detector}")
    print("Splits:        " + ", ".join(
        f"{(s[0] or 'flat')}={s[3]}job×{s[2]}ev" for s in splits))
    if multi_partition:
        plan = ", ".join(f"{n}:{w}" for n, w in parsed_parts)
        print(f"Partition:     {partition}  (weighted round-robin → {plan})")
    else:
        print(f"Partition:     {partition}  "
              f"GPUs={'1' if args.gpu else env.get('DEFAULT_GPUS', '0')}")
    print()

    submitted = 0
    prepared = 0
    skip_lucid_flag = not run_lucid

    def _prep_subdir(d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        if run_lucid:
            for sub in ("sensor", "hits", "step", "labl"):
                (d / sub).mkdir(parents=True, exist_ok=True)

    for split_name, split_comp, n_events, n_jobs, seed_base in splits:
        # Output dir: OUTPUT_BASE/<detector>/<block>/[<split>]/config_NN
        out_base_dir = output_base / detector
        # Test mode mirrors the whole tree under _test/. It used to write into
        # the production cell with job_id 000001, overwriting the real
        # submit_job_000001 and its batch with a 2-event version.
        if args.test:
            out_base_dir = out_base_dir / "_test"
        out_base_dir = out_base_dir / block
        if split_comp:
            out_base_dir = out_base_dir / split_comp
        config_id = f"{config_number:02d}"
        config_dir = out_base_dir / f"config_{config_id}"
        _prep_subdir(config_dir)
        rc = _refuse_foreign_cell(config_dir, args.config, args.force_overwrite)
        if rc:
            return rc
        print(f"[{split_name or 'flat'}] {n_jobs} job(s) × {n_events} events → {config_dir}")

        # Subcases: supernova → <model>/<ordering>/, monoenergetic → <E>MeV/,
        # everything else → the config dir itself.
        subcases: List[tuple] = []
        if primary_source == "supernova":
            sn = cfg.get("supernova") or {}
            models = sn.get("models") or []
            orderings = sn_orderings(sn)     # _load_config has checked them
            if not models:
                print("Error: primary_source=supernova requires a non-empty "
                      "supernova.models list", file=sys.stderr)
                return 1
            for msn in models:
                mname = msn.get("name")
                if not mname:
                    print("Error: each supernova.models entry needs a 'name'",
                          file=sys.stderr)
                    return 1
                for ordering in orderings:
                    d = config_dir / _slugify(mname) / _slugify(str(ordering))
                    _prep_subdir(d)
                    subcases.append((d, f"{_slugify(mname)}_{_slugify(str(ordering))}_",
                                     None, mname, str(ordering)))
        elif energy_dist == "monoenergetic":
            single_e = cfg.get("energy_MeV")
            scan = cfg.get("energy_scan") or {}
            if single_e is not None:
                energies = [float(single_e)]
            elif "start_MeV" in scan and "stop_MeV" in scan and "step_MeV" in scan:
                start = scan["start_MeV"]; stop = scan["stop_MeV"]; step = scan["step_MeV"]
                energies = []
                e = start
                while e <= stop:
                    energies.append(float(e))
                    e += step
            else:
                print("Error: monoenergetic requires 'energy_MeV' or "
                      "'energy_scan.{start,stop,step}_MeV'", file=sys.stderr)
                return 1
            for energy in energies:
                d = config_dir / f"{int(round(energy))}MeV"
                _prep_subdir(d)
                subcases.append((d, f"{int(round(energy))}MeV_", float(energy), None, None))
        else:
            subcases.append((config_dir, "", None, None, None))

        jobs_to_process = 1 if args.test else n_jobs
        for out_dir, sub_label, override_energy, sn_model, sn_ordering in subcases:
            for j in range(args.job_id_start, args.job_id_start + jobs_to_process):
                job_id = f"{j:06d}"
                sp_tag = f"{split_name}_" if split_name else ""
                # the block disambiguates per-block config_NN
                job_name = f"photonsim_{block}_config{config_id}_{sp_tag}{sub_label}{job_id}"
                job_partition = wrr.next() if wrr is not None else partition
                master_seed = seed_base + j
                # Optional: decouple the LUCiD-side draws from the Geant4 seed.
                # With `lucid_seed_base` set, PhotonSim still runs on master_seed
                # (so the G4 sample matches a run without it) while the vertex /
                # rotation / propagation draws come from a different stream.
                lucid_seed = ((int(cfg["lucid_seed_base"]) + j)
                              if cfg.get("lucid_seed_base") is not None else None)
                # Supernova is one all-at-once burst per job — do NOT pass
                # --n-events, or run_job would read it as the interaction cap and
                # truncate the burst to n_events; let it use supernova.cap_events.
                job_n_events = None if primary_source == "supernova" else n_events
                body = adapter.render_dataprod_job(
                    cell_dir=out_dir, config_path=args.config, detector=detector,
                    job_id=job_id, job_name=job_name, partition=job_partition,
                    use_gpu=args.gpu, test=args.test, skip_lucid=skip_lucid_flag,
                    n_events=job_n_events, override_energy_mev=override_energy,
                    sn_model=sn_model, sn_ordering=sn_ordering,
                    master_seed=master_seed, lucid_seed=lucid_seed,
                )
                sb_path = out_dir / f"submit_job_{job_id}.{adapter.submit_extension}"
                sb_path.write_text(body)
                sb_path.chmod(0o755)
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
                break  # one subcase in test mode

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
