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
        "--output-dir", type=str, required=True,
        help="Absolute path to the dataset directory for this config. "
             "PhotonSim ROOT and the v3 {sensor,inst,seg,labl}/ subdirs are written here.",
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
        help="Skip the GENIE step (gevgen+gntpc). Has no effect when "
             "primary_source != 'genie'.",
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


def _load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    required = ["name", "material"]
    if cfg.get("primary_source") == "genie":
        required.append("genie")
    else:
        required += ["particles", "energy_distribution"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Config {path!r} missing required field: {key!r}")
    return cfg


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
) -> None:
    """Import LUCiD and write the v3 four-file batch for this job.

    Imports are deferred so that `--help` works without jax/numpy/uproot
    installed (e.g. on the login node).
    """
    # Deferred imports
    import numpy as np
    from lucid.sources.event_io import generate_events_from_photonsim_particles
    from lucid.simulation import setup_event_simulator
    from lucid.geometry.detector_geometry import DetectorGeometry
    from lucid.utils import base_dir_path

    detector_config_path = base_dir_path() + "config/SK_like_geom_config.json"
    physics_config_path = base_dir_path() + "config/SK_like_physics_config.json"

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

    simulate_event = setup_event_simulator(
        detector_config_path,
        0,
        K=12,
        is_data=True,
        temperature=0.0,
        apply_smearing=False,  # per-particle smearing off; PE-sum smearing applied below
        physics_config=physics_config_path,
        default_detector_params=True,
    )

    lucid_opts = config.get("lucid_options", {})
    apply_smearing = bool(lucid_opts.get("apply_smearing", True))
    apply_translation = bool(lucid_opts.get("apply_translation", True))

    saved_files = generate_events_from_photonsim_particles(
        event_simulator=simulate_event,
        root_file_path=str(root_file),
        sensor_positions=sensor_positions,
        output_dir=str(output_dir),
        n_events=n_events,
        batch_size=n_events,   # one batch per job
        master_seed=master_seed,
        apply_smearing=apply_smearing,
        apply_rotation=False,  # PhotonSim already randomizes directions
        apply_translation=apply_translation,
        detector_config_path=detector_config_path,
        dataset_name=config["name"],
        run_id=None,           # auto-UUID4
        file_index_start=file_index,
        detector_type=detector_type,
        material=material,
        include_track_segments=True,
    )
    print(f"LUCiD wrote {len(saved_files)} files under {output_dir}/{{sensor,inst,seg,labl}}/")


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # 1. Load config
    try:
        config = _load_config(args.config)
    except Exception as e:
        print(f"Error loading config {args.config!r}: {e}", file=sys.stderr)
        return EXIT_CONFIG

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
    for sub in ("sensor", "inst", "seg", "labl"):
        (output_dir / sub).mkdir(exist_ok=True)

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

    # Step 0: GENIE event generation (only when the config requests it)
    if uses_genie and not args.skip_genie:
        try:
            from lucid.production.run_genie import run_genie
            print("\n=== Step 0: GENIE event generation ===", flush=True)
            produced = run_genie(
                config=config,
                output_dir=output_dir,
                job_id=args.job_id,
                n_events=n_events,
                seed=args.master_seed,
            )
            print(f"GENIE rootracker: {produced}")
            # run_genie uses the same convention; sanity-check.
            if produced != gtrac_path:
                print(f"Warning: GENIE output {produced} != expected {gtrac_path}")
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

    # Step 1+2: Generate macro and run PhotonSim
    if not args.skip_photonsim:
        from lucid.production.generate_macro import generate_macro
        print("\n=== Step 1: Generating Geant4 macro ===", flush=True)
        macro_text = generate_macro(
            config,
            output_root_file=root_filename,
            n_events=n_events,
            override_energy_MeV=args.override_energy_MeV,
            genie_rootracker=str(gtrac_path) if uses_genie else None,
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

    try:
        _run_lucid(
            root_file=root_path,
            output_dir=output_dir,
            config=config,
            file_index=file_index,
            n_events=n_events,
            master_seed=args.master_seed,
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


if __name__ == "__main__":
    sys.exit(main())
