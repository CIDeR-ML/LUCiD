"""Single-job end-to-end runner for the LUCiD production chain.

One invocation generates a Geant4 macro from a dataprod JSON config,
runs PhotonSim (C++ binary, via subprocess), runs LUCiD's v3 writer
(in-process Python call), and verifies the resulting four-file batch.

Entry point for the `lucid-run-job` console script.

Required env vars:
    PHOTONSIM_BIN   — absolute path to built PhotonSim binary.

Optional env vars:
    JAX_PLATFORMS   — e.g. `cpu` for determinism (set before import).

Exit codes:
    0   success; all four v3 files present and valid
    10  config load / schema error
    20  PhotonSim failure
    30  LUCiD failure
    40  v3 verification failure
    1   generic
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


EXIT_OK = 0
EXIT_CONFIG = 10
EXIT_PHOTONSIM = 20
EXIT_LUCID = 30
EXIT_VERIFY = 40


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one job of the LUCiD production chain: PhotonSim → LUCiD v3 writer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to dataprod JSON config (e.g. from lucid/production/configs/).",
    )
    parser.add_argument(
        "--detector", type=str, default="SK_like",
        help="Detector geometry to simulate against. Selects "
             "config/<detector>_{geom,physics}_config.json.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Absolute path to the dataset directory for this config. "
             "PhotonSim ROOT and the v3 {sensor,hits,step,labl}/ subdirs are written here.",
    )
    parser.add_argument(
        "--job-id", type=int, required=True,
        help="1-based job id. Maps to v3 file_index = job_id - 1.",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: cap n_events to 2. Overrides --n-events.",
    )
    parser.add_argument(
        "--n-events", type=int, default=None,
        help="Override n_events_per_job from the config.",
    )
    parser.add_argument(
        "--master-seed", type=int, default=None,
        help="JAX PRNG seed for reproducibility. Default: random.",
    )
    parser.add_argument(
        "--override-energy-MeV", type=float, default=None,
        help="Force monoenergetic primaries at this energy (for energy-scan fan-outs).",
    )
    parser.add_argument(
        "--keep-root", action="store_true",
        help="Keep the intermediate PhotonSim ROOT file even if config sets cleanup_root_files=true.",
    )
    parser.add_argument(
        "--skip-genie", action="store_true",
        help="Skip the rooTracker-generator step (GENIE gevgen+gntpc, or "
             "sntools for supernova). Has no effect when primary_source is "
             "particle-gun/bomb.",
    )
    parser.add_argument(
        "--sn-model", type=str, default=None,
        help="Supernova model name; selects an entry in the config's "
             "supernova.models. The fan-out passes one per model×ordering "
             "subcase. Only used when primary_source == 'supernova'.",
    )
    parser.add_argument(
        "--sn-ordering", type=str, default=None,
        help="Mass-ordering label (NMO/IMO/noosc) → sntools --transformation. "
             "Only used when primary_source == 'supernova'.",
    )
    parser.add_argument(
        "--skip-photonsim", action="store_true",
        help="Skip macro generation + PhotonSim. Assumes the ROOT file is already present. "
             "Use in multi-process workflows where PhotonSim runs in a different environment "
             "(e.g., S3DF bare node) and LUCiD runs in a container.",
    )
    parser.add_argument(
        "--skip-lucid", action="store_true",
        help="Run macro generation + PhotonSim only; stop before LUCiD v3 writer. "
             "Use in multi-process workflows where LUCiD runs in a different environment "
             "(e.g., inside a singularity image) than PhotonSim.",
    )
    return parser.parse_args(argv)


def _read_digitizer_cfg(physics_config_path: str):
    """Read the optional 'digitizer' block from a detector physics config.

    Returns the block (str or dict) or None (-> 'basic') when absent/unreadable.
    """
    try:
        with open(physics_config_path) as f:
            return json.load(f).get("digitizer")
    except (OSError, ValueError):
        return None


def _load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    required = ["name", "material"]
    if cfg.get("pile_up"):
        required += ["vertices"]
        vertices = cfg.get("vertices")
        if not isinstance(vertices, list) or len(vertices) < 2:
            raise ValueError(
                f"Pile-up config {path!r} must declare at least 2 vertices; got {vertices!r}")
        for i, v in enumerate(vertices):
            src = v.get("primary_source")
            if src == "genie":
                if "genie" not in v:
                    raise ValueError(
                        f"Pile-up vertex {i} primary_source='genie' but missing 'genie' block")
            elif src == "bomb":
                if "bomb" not in v or not isinstance(v["bomb"].get("candidates"), list):
                    raise ValueError(
                        f"Pile-up vertex {i} primary_source='bomb' but missing "
                        f"'bomb.candidates' list")
            else:
                if "particles" not in v or "energy_distribution" not in v:
                    raise ValueError(
                        f"Pile-up vertex {i} missing 'particles'/'energy_distribution'")
    elif cfg.get("primary_source") == "genie":
        required.append("genie")
    elif cfg.get("primary_source") == "supernova":
        required.append("supernova")
    elif cfg.get("primary_source") == "bomb":
        required.append("bomb")
    else:
        required += ["particles", "energy_distribution"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Config {path!r} missing required field: {key!r}")
    return cfg


def _build_vertex_subconfig(top_config: dict, vertex: dict) -> dict:
    """Merge top-level shared fields with a per-vertex block.

    The per-vertex sub-config is what generate_macro/run_genie see as
    "the config" — so it must carry the usual top-level keys
    (``name``, ``material``, photon/decay switches, ...) plus the
    vertex-specific primary source.
    """
    sub = {
        "name": top_config["name"],
        "material": top_config["material"],
        "store_individual_photons": top_config.get("store_individual_photons", False),
        "disable_decays": top_config.get("disable_decays", False),
        "config_number": top_config.get("config_number", -1),
    }
    sub["primary_source"] = vertex.get("primary_source", "particles")
    if sub["primary_source"] == "genie":
        sub["genie"] = vertex["genie"]
    elif sub["primary_source"] == "bomb":
        sub["bomb"] = vertex["bomb"]
    else:
        sub["energy_distribution"] = vertex.get(
            "energy_distribution", top_config.get("energy_distribution", "uniform"))
        sub["particles"] = vertex["particles"]
    return sub


def _run_photonsim(photonsim_bin: str, macro_path: Path, cwd: Path) -> None:
    print(f"=== Step 2: Running PhotonSim ===", flush=True)
    print(f"    binary: {photonsim_bin}")
    print(f"    macro:  {macro_path}")
    print(f"    cwd:    {cwd}")
    t0 = time.time()
    rc = subprocess.call([photonsim_bin, str(macro_path)], cwd=str(cwd))
    elapsed = time.time() - t0
    print(f"PhotonSim exit code: {rc} (elapsed {elapsed:.1f}s)")
    if rc != 0:
        raise RuntimeError(f"PhotonSim returned {rc}")


def _run_lucid(
    *,
    root_file: Path,
    output_dir: Path,
    config: dict,
    file_index: int,
    n_events: int,
    master_seed: Optional[int],
    job_id: int,
    detector: str,
) -> None:
    """Import LUCiD and write the v3 four-file batch for this job.

    Imports are deferred so that `--help` works without jax/numpy/uproot
    installed (e.g. on the login node).
    """
    # Deferred imports
    import numpy as np
    from lucid.sources.event_generation import generate_events_from_photonsim_particles
    from lucid.simulation import setup_event_simulator
    from lucid.geometry.detector_geometry import DetectorGeometry
    from lucid.utils import base_dir_path

    detector_config_path = base_dir_path() + f"config/{detector}_geom_config.json"
    physics_config_path = base_dir_path() + f"config/{detector}_physics_config.json"

    print(f"=== Step 3: Running LUCiD v3 writer ===", flush=True)
    print(f"    ROOT file:  {root_file}")
    print(f"    Output dir: {output_dir}")
    print(f"    Dataset:    {config['name']}")
    print(f"    file_index: {file_index}")
    print(f"    n_events:   {n_events}")

    det_geom = DetectorGeometry.from_config(detector_config_path)
    sensor_positions = np.asarray(det_geom.sensor_points, dtype=np.float32)
    detector_type = str(det_geom.detector_type)
    material = str(det_geom.medium.material)
    print(f"    geometry:   {detector_type} / {material}, "
          f"{sensor_positions.shape[0]} sensors")

    lucid_opts = config.get("lucid_options", {})
    apply_smearing = bool(lucid_opts.get("apply_smearing", True))
    apply_translation = bool(lucid_opts.get("apply_translation", True))
    # step/event_NNN/sensor_hits/ is mandatory for data-mode datasets:
    # it's the per-(segment, sensor) ground truth that the hits file is now
    # aggregated from. Photon_SegmentIndex is unconditional in PhotonSim
    # post-Stage-5a, so no macro flag is needed. The simulator runs in
    # 'per_segment' mode, which produces an exact per-(segment, sensor)
    # decomposition with column-sums equal to the per-sensor totals
    # by construction (shared qe_weights — sensor_response.py:100-104).

    simulate_event = setup_event_simulator(
        detector_config_path,
        0,
        K=12,
        is_data=True,
        temperature=0.0,
        apply_smearing=False,  # per-particle smearing off; PE-sum smearing applied below
        physics_config=physics_config_path,
        default_detector_params=True,
        hit_mode='per_segment',
    )
    # PAD_SIZE bucketing (see lucid/sources/event_io.py for the full rationale).
    # `None` -> module default; an explicit empty list opts back into the
    # legacy single-PAD_SIZE-from-file-max path (one JIT compile, no chunking).
    pad_size_buckets = lucid_opts.get("pad_size_buckets", None)

    # Digitizer (sensor hit-making) model comes from the detector physics
    # config's optional "digitizer" block; absent -> "basic" (legacy behaviour).
    digitizer_cfg = _read_digitizer_cfg(physics_config_path)

    saved_files = generate_events_from_photonsim_particles(
        event_simulator=simulate_event,
        root_file_path=str(root_file),
        sensor_positions=sensor_positions,
        output_dir=str(output_dir),
        n_events=n_events,
        batch_size=n_events,   # one batch per job
        master_seed=master_seed,
        job_id=job_id,
        apply_smearing=apply_smearing,
        apply_rotation=False,  # PhotonSim already randomizes directions
        apply_translation=apply_translation,
        dataset_name=config["name"],
        run_id=None,           # auto-UUID4
        file_index_start=file_index,
        primary_source=config.get("primary_source", "particles"),
        pad_size_buckets=pad_size_buckets,
        digitizer=digitizer_cfg,
    )
    print(f"LUCiD wrote {len(saved_files)} files under {output_dir}/{{sensor,hits,step,labl}}/")


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # 1. Load config
    try:
        config = _load_config(args.config)
    except Exception as e:
        print(f"Error loading config {args.config!r}: {e}", file=sys.stderr)
        return EXIT_CONFIG

    if config.get("pile_up"):
        return _main_pileup(args, config)

    n_events = args.n_events
    if args.test:
        n_events = 2
    if n_events is None:
        n_events = int(config.get("n_events_per_job", 100))

    if args.skip_genie and args.skip_photonsim and args.skip_lucid:
        print("Error: all steps skipped; nothing to do.", file=sys.stderr)
        return EXIT_CONFIG

    photonsim_bin = os.environ.get("PHOTONSIM_BIN")
    if not args.skip_photonsim:
        if not photonsim_bin:
            print("Error: PHOTONSIM_BIN env var must point to the built PhotonSim binary.",
                  file=sys.stderr)
            return EXIT_CONFIG
        if not Path(photonsim_bin).is_file():
            print(f"Error: PHOTONSIM_BIN={photonsim_bin!r} does not exist.", file=sys.stderr)
            return EXIT_CONFIG

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # The sensor/hits/step/labl subdirs are pre-created by the host-side
    # fan-out (dataprod_fanout.py) before any sub-job is submitted, so the
    # worker doesn't race other sub-jobs to create them. event_generation.py
    # still mkdirs them defensively before opening h5 files, but by then
    # PhotonSim has run and any storage metadata lag has long since cleared.

    job_id_padded = f"{args.job_id:06d}"
    file_index = args.job_id - 1
    macro_path = output_dir / f"job_{job_id_padded}.mac"
    root_filename = f"output_job_{job_id_padded}.root"
    root_path = output_dir / root_filename

    print(f"=== lucid-run-job ===", flush=True)
    print(f"    config:       {args.config}")
    print(f"    dataset_name: {config['name']}")
    print(f"    config_number:{config.get('config_number', '?')}")
    print(f"    job_id:       {args.job_id} (file_index={file_index})")
    print(f"    n_events:     {n_events}")
    print(f"    output_dir:   {output_dir}")

    # Deterministic path for the GENIE rootracker file, regardless of
    # whether we produce it this invocation or a prior one did. Used by both
    # the GENIE step and the macro-generation step so the two can run in
    # separate processes and still agree on the path.
    gtrac_path = output_dir / f"gntp_job_{job_id_padded}.gtrac.root"
    uses_genie = config.get("primary_source") == "genie"
    uses_supernova = config.get("primary_source") == "supernova"
    # Both GENIE and supernova feed PhotonSim through the rooTracker /
    # /gun/genieInput path; the macro builder and beamOn-count handling are
    # shared, only the Step-0 generator differs.
    uses_rootracker = uses_genie or uses_supernova

    # Step 0: GENIE event generation (only when the config requests it).
    # GENIE configs use primary-by-primary injection in PhotonSim: gevgen
    # produces N interactions, run_genie counts the total status==1 primaries
    # across those, and we set /run/beamOn (and LUCiD's n_events) to that
    # primary count so the downstream v3 dataset has one entry per primary.
    photonsim_events = n_events

    # Derive per-subprocess seeds from the (master_seed, job_id, vertex_idx=0)
    # hierarchy. vertex_idx=0 is the only stream today; pile-up configs
    # (commit 3) will iterate this per vertex.
    from lucid.sources.seed_utils import derive_subprocess_seeds
    subproc_seeds = derive_subprocess_seeds(
        args.master_seed, args.job_id, vertex_idx=0
    )

    if uses_genie and not args.skip_genie:
        try:
            from lucid.production.run_genie import run_genie
            print("\n=== Step 0: GENIE event generation ===", flush=True)
            t_genie = time.time()
            produced, total_primaries = run_genie(
                config=config,
                output_dir=output_dir,
                job_id=args.job_id,
                n_events=n_events,
                seed=subproc_seeds['genie_seed'],
            )
            print(f"GENIE elapsed: {time.time() - t_genie:.1f}s")
            print(f"GENIE rootracker: {produced}")
            print(f"GENIE total primaries (one G4 event each): {total_primaries}")
            if produced != gtrac_path:
                print(f"Warning: GENIE output {produced} != expected {gtrac_path}")
            photonsim_events = total_primaries
            # Persist primary count so --skip-genie reruns know the right beamOn.
            (output_dir / f"gntp_job_{job_id_padded}.primaries.txt"
             ).write_text(str(total_primaries) + "\n")
        except Exception as e:
            print(f"GENIE step failed: {e}", file=sys.stderr)
            return EXIT_PHOTONSIM
    elif uses_genie and args.skip_genie and not args.skip_photonsim:
        # PhotonSim will consume the rootracker; it must already exist.
        if not gtrac_path.is_file():
            print(f"Error: --skip-genie set but {gtrac_path} does not exist "
                  f"(required for PhotonSim step with primary_source=genie).",
                  file=sys.stderr)
            return EXIT_PHOTONSIM
        marker = output_dir / f"gntp_job_{job_id_padded}.primaries.txt"
        if marker.is_file():
            photonsim_events = int(marker.read_text().strip())
            print(f"GENIE primary count from cache: {photonsim_events}")

    # Step 0 (supernova): sntools event generation → rooTracker. Mirrors the
    # GENIE path — one sntools event per rooTracker entry, and beamOn is set to
    # the entry count. The burst size is data-driven; --test / --n-events /
    # supernova.cap_events truncate it for validation runs.
    if uses_supernova and not args.skip_genie:
        try:
            from lucid.production.run_supernova import run_supernova
            print("\n=== Step 0: supernova (sntools) event generation ===", flush=True)
            sn_cap = None
            if args.test:
                sn_cap = 2
            elif args.n_events is not None:
                sn_cap = args.n_events
            elif config["supernova"].get("cap_events") is not None:
                sn_cap = int(config["supernova"]["cap_events"])
            t_sn = time.time()
            produced, n_entries = run_supernova(
                config=config,
                output_dir=output_dir,
                job_id=args.job_id,
                model_name=args.sn_model,
                ordering=args.sn_ordering,
                seed=subproc_seeds['genie_seed'],
                cap_events=sn_cap,
            )
            print(f"sntools elapsed: {time.time() - t_sn:.1f}s")
            print(f"supernova rootracker: {produced} ({n_entries} events)")
            if produced != gtrac_path:
                print(f"Warning: sntools output {produced} != expected {gtrac_path}")
            photonsim_events = n_entries
            (output_dir / f"gntp_job_{job_id_padded}.primaries.txt"
             ).write_text(str(n_entries) + "\n")
        except Exception as e:
            print(f"supernova step failed: {e}", file=sys.stderr)
            return EXIT_PHOTONSIM
    elif uses_supernova and args.skip_genie and not args.skip_photonsim:
        if not gtrac_path.is_file():
            print(f"Error: --skip-genie set but {gtrac_path} does not exist "
                  f"(required for PhotonSim step with primary_source=supernova).",
                  file=sys.stderr)
            return EXIT_PHOTONSIM
        marker = output_dir / f"gntp_job_{job_id_padded}.primaries.txt"
        if marker.is_file():
            photonsim_events = int(marker.read_text().strip())
            print(f"supernova event count from cache: {photonsim_events}")

    # Step 1+2: Generate macro and run PhotonSim
    if not args.skip_photonsim:
        from lucid.production.generate_macro import generate_macro
        print("\n=== Step 1: Generating Geant4 macro ===", flush=True)
        macro_text = generate_macro(
            config,
            output_root_file=root_filename,
            n_events=photonsim_events,
            override_energy_MeV=args.override_energy_MeV,
            genie_rootracker=str(gtrac_path) if uses_rootracker else None,
            photonsim_seeds=(subproc_seeds['photonsim_seed1'],
                             subproc_seeds['photonsim_seed2']),
        )
        macro_path.write_text(macro_text)
        print(f"Wrote {macro_path}")

        try:
            _run_photonsim(photonsim_bin, macro_path, cwd=output_dir)
        except Exception as e:
            print(f"PhotonSim failed: {e}", file=sys.stderr)
            return EXIT_PHOTONSIM
        if not root_path.is_file():
            print(f"Error: PhotonSim did not produce {root_path}", file=sys.stderr)
            return EXIT_PHOTONSIM

    if args.skip_lucid:
        print(f"\n--skip-lucid: stopping after PhotonSim. ROOT file: {root_path}")
        return EXIT_OK

    # Step 3: Run LUCiD (requires ROOT file to exist; verify)
    if not root_path.is_file():
        print(f"Error: expected ROOT file {root_path} does not exist (needed for LUCiD step).",
              file=sys.stderr)
        return EXIT_PHOTONSIM

    # For GENIE configs the LUCiD step should see one entry per injected
    # primary, matching the PhotonSim output. For particle-gun configs
    # photonsim_events == n_events, so this is a no-op.
    if uses_genie and photonsim_events == n_events:
        # Happens when --skip-genie was set and the marker was missing;
        # fall back to reading from PhotonSim's ROOT header.
        pass
    try:
        _run_lucid(
            root_file=root_path,
            output_dir=output_dir,
            config=config,
            file_index=file_index,
            n_events=photonsim_events,
            master_seed=args.master_seed,
            job_id=args.job_id,
            detector=args.detector,
        )
    except Exception as e:
        import traceback
        print(f"LUCiD failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return EXIT_LUCID

    # Step 4: Verify v3 output
    print("\n=== Step 4: Verifying v3 output ===", flush=True)
    from lucid.production.verify_output import verify_batch
    ok, messages = verify_batch(output_dir, file_index, expected_dataset_name=config["name"])
    for m in messages:
        print(f"    {m}")
    if not ok:
        return EXIT_VERIFY

    # Supernova: stamp the interaction channel (IBD/ES/o16e/o16eb) into the
    # labl truth from the sidecar run_supernova wrote next to the rooTracker.
    # Non-fatal: a valid dataset must not fail over truth annotation.
    if uses_supernova:
        try:
            from lucid.production.run_supernova import annotate_labl_channels
            labl_path = output_dir / "labl" / f"wc_labl_{file_index:04d}.h5"
            sidecar = output_dir / f"gntp_job_{job_id_padded}.channels.json"
            if labl_path.is_file() and sidecar.is_file():
                n_ann = annotate_labl_channels(labl_path, sidecar)
                print(f"Annotated interaction channel into {n_ann} labl events")
            else:
                print(f"Warning: channel sidecar/labl missing "
                      f"({sidecar.name} / {labl_path.name}); skipped channel annotation")
        except Exception as e:
            print(f"Warning: channel annotation failed: {e}")

    # Step 5: Cleanup ROOT if requested
    cleanup = bool(config.get("cleanup_root_files", False)) and not args.keep_root
    if cleanup:
        try:
            root_path.unlink()
            print(f"\nRemoved intermediate ROOT file: {root_path}")
        except OSError as e:
            print(f"\nWarning: could not remove {root_path}: {e}")

    # 7. One-line summary
    print(
        f"\nOK config_number={config.get('config_number','?')} "
        f"job_id={args.job_id} file_index={file_index} "
        f"dataset_name={config['name']!r}"
    )
    return EXIT_OK


def _main_pileup(args: argparse.Namespace, config: dict) -> int:
    """Run a pile-up job: N PhotonSim streams -> merged v3 event."""
    n_events = args.n_events
    if args.test:
        n_events = 2
    if n_events is None:
        n_events = int(config.get("n_events_per_job", 100))

    if args.skip_genie and args.skip_photonsim and args.skip_lucid:
        print("Error: all steps skipped; nothing to do.", file=sys.stderr)
        return EXIT_CONFIG

    photonsim_bin = os.environ.get("PHOTONSIM_BIN")
    if not args.skip_photonsim:
        if not photonsim_bin:
            print("Error: PHOTONSIM_BIN env var must point to the built PhotonSim binary.",
                  file=sys.stderr)
            return EXIT_CONFIG
        if not Path(photonsim_bin).is_file():
            print(f"Error: PHOTONSIM_BIN={photonsim_bin!r} does not exist.", file=sys.stderr)
            return EXIT_CONFIG

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # sensor/hits/step/labl subdirs are pre-created by dataprod_fanout.py;
    # see the corresponding comment in the main run_job entry above.

    vertices = config["vertices"]
    n_vertices = len(vertices)
    job_id_padded = f"{args.job_id:06d}"
    file_index = args.job_id - 1

    print(f"=== lucid-run-job (pile-up) ===", flush=True)
    print(f"    config:        {args.config}")
    print(f"    dataset_name:  {config['name']}")
    print(f"    config_number: {config.get('config_number', '?')}")
    print(f"    job_id:        {args.job_id} (file_index={file_index})")
    print(f"    n_events:      {n_events}")
    print(f"    n_vertices:    {n_vertices}")
    print(f"    output_dir:    {output_dir}")

    from lucid.sources.seed_utils import derive_subprocess_seeds
    from lucid.production.generate_macro import generate_macro

    root_paths: list[Path] = []
    vertex_sources: list[str] = []

    for vidx, vertex in enumerate(vertices):
        sub_cfg = _build_vertex_subconfig(config, vertex)
        vertex_sources.append(sub_cfg["primary_source"])
        uses_genie_v = sub_cfg["primary_source"] == "genie"
        subproc_seeds = derive_subprocess_seeds(
            args.master_seed, args.job_id, vertex_idx=vidx
        )
        gtrac_path = output_dir / f"gntp_job_{job_id_padded}_v{vidx}.gtrac.root"
        root_filename = f"output_job_{job_id_padded}_v{vidx}.root"
        root_path = output_dir / root_filename
        macro_path = output_dir / f"job_{job_id_padded}_v{vidx}.mac"

        print(f"\n--- vertex {vidx}: {sub_cfg['primary_source']} ---", flush=True)

        # Step 0 (per-vertex): GENIE if requested
        ps_events = n_events
        if uses_genie_v and not args.skip_genie:
            try:
                from lucid.production.run_genie import run_genie
                t_genie = time.time()
                produced, total_primaries = run_genie(
                    config=sub_cfg,
                    output_dir=output_dir,
                    job_id=args.job_id * 100 + vidx,  # make rootracker filenames distinct
                    n_events=n_events,
                    seed=subproc_seeds["genie_seed"],
                )
                print(f"GENIE vertex {vidx} elapsed: {time.time() - t_genie:.1f}s")
                # run_genie writes gntp_job_{job_id_padded_inner}.gtrac.root —
                # move/rename to our per-vertex path.
                if produced != gtrac_path:
                    try:
                        produced.rename(gtrac_path)
                    except OSError:
                        import shutil
                        shutil.copy2(str(produced), str(gtrac_path))
                ps_events = total_primaries
            except Exception as e:
                print(f"GENIE step for vertex {vidx} failed: {e}", file=sys.stderr)
                return EXIT_PHOTONSIM

        # Step 1+2: Generate macro, run PhotonSim
        if not args.skip_photonsim:
            macro_text = generate_macro(
                sub_cfg,
                output_root_file=root_filename,
                n_events=ps_events,
                override_energy_MeV=args.override_energy_MeV,
                genie_rootracker=str(gtrac_path) if uses_genie_v else None,
                photonsim_seeds=(subproc_seeds["photonsim_seed1"],
                                 subproc_seeds["photonsim_seed2"]),
            )
            macro_path.write_text(macro_text)
            print(f"    macro: {macro_path}")
            try:
                _run_photonsim(photonsim_bin, macro_path, cwd=output_dir)
            except Exception as e:
                print(f"PhotonSim vertex {vidx} failed: {e}", file=sys.stderr)
                return EXIT_PHOTONSIM
            if not root_path.is_file():
                print(f"Error: PhotonSim vertex {vidx} did not produce {root_path}",
                      file=sys.stderr)
                return EXIT_PHOTONSIM
        else:
            if not root_path.is_file():
                print(f"Error: --skip-photonsim but {root_path} missing", file=sys.stderr)
                return EXIT_PHOTONSIM

        root_paths.append(root_path)

    if args.skip_lucid:
        print(f"\n--skip-lucid: stopping after per-vertex PhotonSim. "
              f"ROOT files: {root_paths}")
        return EXIT_OK

    # Step 3: Run LUCiD pile-up merger
    import numpy as np
    from lucid.sources.event_generation import generate_events_from_photonsim_pileup
    from lucid.simulation import setup_event_simulator
    from lucid.geometry.detector_geometry import DetectorGeometry
    from lucid.utils import base_dir_path

    detector_config_path = base_dir_path() + f"config/{args.detector}_geom_config.json"
    physics_config_path  = base_dir_path() + f"config/{args.detector}_physics_config.json"

    det_geom = DetectorGeometry.from_config(detector_config_path)
    sensor_positions = np.asarray(det_geom.sensor_points, dtype=np.float32)

    simulate_event = setup_event_simulator(
        detector_config_path, 0, K=12, is_data=True, temperature=0.0,
        apply_smearing=False, physics_config=physics_config_path,
        default_detector_params=True,
        hit_mode='per_segment',
    )

    lucid_opts = config.get("lucid_options", {})
    apply_smearing = bool(lucid_opts.get("apply_smearing", True))
    apply_translation = bool(lucid_opts.get("apply_translation", True))

    print(f"\n=== Step 3: LUCiD pile-up merger ({n_vertices} streams) ===", flush=True)
    saved_files = generate_events_from_photonsim_pileup(
        event_simulator=simulate_event,
        root_file_paths=[str(p) for p in root_paths],
        vertex_primary_sources=vertex_sources,
        sensor_positions=sensor_positions,
        output_dir=str(output_dir),
        # Cap at the user-requested n_events. GENIE emits variable
        # primaries per interaction, so each vertex's PhotonSim ROOT
        # typically contains more events than requested; without this
        # cap the merger would use min(per_file_counts) and the job
        # would produce far more events than --n-events / --test asked for.
        n_events=n_events,
        batch_size=n_events,
        master_seed=args.master_seed,
        job_id=args.job_id,
        apply_smearing=apply_smearing,
        apply_translation=apply_translation,
        dataset_name=config["name"],
        run_id=None,
        file_index_start=file_index,
        digitizer=_read_digitizer_cfg(physics_config_path),
    )
    print(f"LUCiD wrote {len(saved_files)} files.")

    # Step 4: Verify
    print("\n=== Step 4: Verifying v3 output ===", flush=True)
    from lucid.production.verify_output import verify_batch
    ok, messages = verify_batch(output_dir, file_index, expected_dataset_name=config["name"])
    for m in messages:
        print(f"    {m}")
    if not ok:
        return EXIT_VERIFY

    # Step 5: Cleanup
    if bool(config.get("cleanup_root_files", False)) and not args.keep_root:
        for rp in root_paths:
            try:
                rp.unlink()
            except OSError as e:
                print(f"    warning: could not remove {rp}: {e}")

    print(f"\nOK pile-up config_number={config.get('config_number','?')} "
          f"job_id={args.job_id} file_index={file_index} "
          f"dataset_name={config['name']!r} n_vertices={n_vertices}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
