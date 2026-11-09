"""Single-job end-to-end runner for the LUCiD production chain.

One invocation generates a Geant4 macro from a dataprod JSON config,
runs PhotonSim (C++ binary, via subprocess), runs LUCiD's writer
(in-process Python call), and verifies the resulting four-file batch.

Entry point for the `lucid-run-job` console script.

Required env vars:
    PHOTONSIM_BIN   — absolute path to built PhotonSim binary.

Optional env vars:
    JAX_PLATFORMS   — e.g. `cpu` for determinism (set before import).

Exit codes:
    0   success; all four files present and valid
    10  config load / schema error
    20  PhotonSim failure
    30  LUCiD failure
    40  verification failure
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
        description="Run one job of the LUCiD production chain: PhotonSim → LUCiD writer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to dataprod JSON config (e.g. from lucid/production/dataprod/configs/).",
    )
    parser.add_argument(
        "--detector", type=str, default=None,
        help="Detector geometry to simulate against. Selects "
             "config/<detector>_{geom,physics}_config.json. Required unless the "
             "dataset config declares its own 'detector' (then it must match) "
             "or --skip-lucid (which simulates none).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Absolute path to the dataset directory for this config. "
             "PhotonSim ROOT and the {sensor,hits,step,labl}/ subdirs are written here.",
    )
    parser.add_argument(
        "--job-id", type=int, required=True,
        help="1-based job id. Maps to file_index = job_id - 1.",
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
        "--lucid-seed", type=int, default=None,
        help="Seed for the LUCiD-side per-event draws (vertex translation, "
             "optional rotation, t0, propagation, smearing). Defaults to "
             "--master-seed. Set it to a different value to reuse the SAME "
             "Geant4 sample (PhotonSim stays on --master-seed) under new "
             "transforms — e.g. an SK set matching HK's events but "
             "independently placed and oriented.")
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
        help="Mass ordering of this job: one entry of the config's supernova.orderings "
             "(e.g. NMO/IMO/noosc → sntools --transformation). The fan-out passes one per "
             "model×ordering subcase; required when the list has more than one entry. "
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
        help="Run macro generation + PhotonSim only; stop before LUCiD writer. "
             "Use in multi-process workflows where LUCiD runs in a different environment "
             "(e.g., inside a singularity image) than PhotonSim.",
    )
    return parser.parse_args(argv)


# Every primary source a dataset config can name; absent means a particle gun ("particles").
PRIMARY_SOURCES = ("particles", "bomb", "genie", "supernova")
# A pile-up vertex is one PhotonSim stream; a supernova burst is not one.
_VERTEX_SOURCES = ("particles", "bomb", "genie")


def primary_source(config: dict, allowed=PRIMARY_SOURCES) -> str:
    """The config's ``primary_source`` (``"particles"`` when absent), checked against ``allowed``.

    Raises
    ------
    ValueError
        If it names anything else. Every consumer compares the string against the names it
        knows and treats the rest as a particle gun, so a typo ran as one.
    """
    src = config.get("primary_source", "particles")
    if src not in allowed:
        raise ValueError(f"primary_source is {src!r}; expected one of {list(allowed)}")
    return src


def physics_config_path(detector: str) -> str:
    """``config/<detector>_physics_config.json`` of this checkout."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                        "config", f"{detector}_physics_config.json")


def resolve_readout_blocks(config: dict, physics_config: str):
    """The dataset's ``(digitizer, trigger)`` blocks.

    Each comes from the dataset config when it carries the key, else from the detector physics
    config ``physics_config``. A key in neither raises: a dataset that wants the ``basic``
    digitizer or no trigger says so (``"digitizer": "basic"``, ``"trigger": null``).

    Raises
    ------
    ValueError
        If a block is in neither config, or the digitizer is ``null``.
    OSError
        If the physics config is needed and cannot be read.
    """
    physics = None
    blocks = []
    for key, opt_out in (("digitizer", '"digitizer": "basic"'), ("trigger", '"trigger": null')):
        if key in config:
            blocks.append(config[key])
            continue
        if physics is None:
            with open(physics_config) as f:
                physics = json.load(f)
        if key not in physics:
            raise ValueError(
                f"no {key!r} block: the dataset config sets none and neither does "
                f"{os.path.normpath(physics_config)}. Set it in the dataset config "
                f"({opt_out} to opt out), or use a detector that carries it (*_WAND).")
        blocks.append(physics[key])
    digitizer, trigger = blocks
    if digitizer is None:
        raise ValueError('"digitizer": null names no model; use "digitizer": "basic" for the '
                         'basic digitizer')
    return digitizer, trigger


_SELECTION_MODES = ("trigger", "min_physics_hits")


def _read_min_physics_hits(config: dict, default=3):
    """Truth-level selection knob (low-energy 'fake trigger'). A dataset config's
    ``selection`` block picks the mode: ``{"mode": "min_physics_hits", "n": N}``
    keeps interactions with >= N real hits; ``{"mode": "trigger"}`` defers to the
    real readout trigger (returns None). Without a block, ``default``.

    Raises
    ------
    ValueError
        If a ``selection`` block is present but names an unknown mode, carries
        an unknown key or gives ``min_physics_hits`` no ``n``, or the config still
        sets the retired ``supernova.min_physics_hits``. Falling through to the default silently reversed the
        selection the config asked for: on the particles path the default is
        ``None`` (real trigger), on the supernova path it is 3 hits.
    """
    sel = config.get("selection")
    if sel is not None:
        if not isinstance(sel, dict):
            raise ValueError(
                f"config 'selection' must be a dict, got {type(sel).__name__}")
        unknown = [k for k in sel
                   if k not in ("mode", "n") and not str(k).startswith("_")]
        if unknown:
            raise ValueError(
                f"unknown key(s) in the config 'selection' block: {sorted(unknown)}; "
                "valid keys are ['mode', 'n']")
        mode = sel.get("mode")
        if mode not in _SELECTION_MODES:
            raise ValueError(
                f"config 'selection.mode' is {mode!r}; expected one of "
                f"{list(_SELECTION_MODES)}")
        if mode == "trigger":
            return None
        if "n" not in sel:
            raise ValueError('config \'selection\' mode "min_physics_hits" needs "n"')
        return int(sel["n"])
    if "min_physics_hits" in config.get("supernova", {}):
        raise ValueError("supernova.min_physics_hits is no longer read; state the selection as "
                         '{"selection": {"mode": "min_physics_hits", "n": N}}')
    return default


def _readout(config: dict, detector: str):
    """The dataset's :class:`~lucid.simulation.readout.Readout`: its digitizer and trigger
    (:func:`resolve_readout_blocks`) and its selection."""
    from lucid.simulation.readout import Readout
    digitizer, trigger = resolve_readout_blocks(config, physics_config_path(detector))
    readout = Readout.from_blocks(digitizer, trigger,
                                  _read_min_physics_hits(config, default=None))
    if config.get("primary_source") == "supernova" and readout.min_physics_hits is None:
        raise ValueError('a supernova config needs {"selection": {"mode": "min_physics_hits", '
                         '"n": N}}; the readout trigger cannot tag low-energy interactions')
    return readout


def _load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    primary_source(cfg)
    required = ["name", "material"]
    if cfg.get("pile_up"):
        required += ["vertices"]
        vertices = cfg.get("vertices")
        if not isinstance(vertices, list) or len(vertices) < 2:
            raise ValueError(
                f"Pile-up config {path!r} must declare at least 2 vertices; got {vertices!r}")
        for i, v in enumerate(vertices):
            try:
                src = primary_source(v, _VERTEX_SOURCES)
            except ValueError as e:
                raise ValueError(f"Pile-up vertex {i}: {e}") from None
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
    if cfg.get("primary_source") == "supernova":
        from lucid.production.dataprod.generators.supernova import sn_orderings
        sn_orderings(cfg["supernova"])
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
    sub["primary_source"] = primary_source(vertex, _VERTEX_SOURCES)
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


def _simulator(detector: str):
    """The given-photons simulator of ``config/<detector>_{geom,physics}_config.json``."""
    from lucid.simulation import setup_event_simulator
    from lucid.utils import base_dir_path

    detector_config_path = base_dir_path() + f"config/{detector}_geom_config.json"
    physics_path = base_dir_path() + f"config/{detector}_physics_config.json"
    simulator = setup_event_simulator(
        detector_config_path,
        0,
        source='photons',
        K=12,
        temperature=0.0,
        charge_resolution=None,  # the digitizer applies the charge resolution
        physics_config=physics_path,
        default_detector_params=True,
        # per-(segment, sensor) deposits: the step/sensor_hits ground truth the hits file is
        # aggregated from
        hits='segment_truth',
    )
    return simulator


def _lucid_options(config: dict) -> dict:
    opts = config.get("lucid_options", {})
    return {"apply_smearing": bool(opts.get("apply_smearing", True)),
            "apply_translation": bool(opts.get("apply_translation", True))}


def _run_lucid(
    *,
    root_file: Path,
    output_dir: Path,
    config: dict,
    file_index: int,
    n_events: int,
    master_seed: int,
    job_id: int,
    detector: str,
    readout,
    sn_ordering: Optional[str] = None,
) -> None:
    """Import LUCiD and write the four-file batch for this job.

    ``sn_ordering`` is ``--sn-ordering`` (supernova only; see
    :func:`~lucid.production.dataprod.generators.supernova.resolve_ordering`).

    Imports are deferred so that `--help` works without jax/numpy/uproot
    installed (e.g. on the login node).
    """
    from lucid.production.dataprod.drivers import run_particles, run_supernova

    print(f"=== Step 3: Running LUCiD writer ===", flush=True)
    print(f"    ROOT file:  {root_file}")
    print(f"    Output dir: {output_dir}")
    print(f"    Dataset:    {config['name']}")
    print(f"    file_index: {file_index}")
    print(f"    n_events:   {n_events}")

    simulator = _simulator(detector)
    det_geom = simulator.det_geom
    print(f"    geometry:   {det_geom.detector_type} / {det_geom.medium.material}, "
          f"{len(det_geom.sensor_points)} sensors")

    # Supernova: model the whole burst as ONE all-at-once event. Every sntools
    # interaction (one PhotonSim entry) is placed at its true time (from the
    # .times.json sidecar) via a shared global t0; the generator chunks the
    # burst at causal gaps and digitizes each chunk. Selection is trigger-free
    # (_readout requires it): keep interactions with >= min_physics_hits real
    # hits (dark kept + labelled), so the dataset isn't biased by a trigger choice.
    if config.get("primary_source") == "supernova":
        from lucid.production.dataprod.generators.supernova import resolve_ordering
        ordering, transformation = resolve_ordering(config["supernova"], sn_ordering)
        times_path = output_dir / f"gntp_job_{job_id:06d}.times.json"
        interaction_times_ms = json.loads(times_path.read_text())
        # Per-interaction sntools channel (ibd/es/o16...), stamped into
        # per_interaction/interaction_channel alongside the neutrino truth.
        chan_path = output_dir / f"gntp_job_{job_id:06d}.channels.json"
        interaction_channels = (json.loads(chan_path.read_text())
                                if chan_path.is_file() else None)
        # True SN direction (unit vector), recorded into the labl truth for
        # direction-pointing studies. Default +z if the sidecar is absent.
        dir_path = output_dir / f"gntp_job_{job_id:06d}.direction.json"
        sn_direction = (json.loads(dir_path.read_text())
                        if dir_path.is_file() else [0.0, 0.0, 1.0])
        saved_files = run_supernova(
            simulator, str(root_file), interaction_times_ms,
            output_dir=str(output_dir),
            file_index=file_index,
            master_seed=master_seed,
            job_id=job_id,
            readout=readout,
            dataset_name=config["name"],
            interaction_channels=interaction_channels,
            sn_direction=sn_direction,
            sn_ordering=ordering,
            sn_transformation=transformation,
            **_lucid_options(config),
        )
        print(f"LUCiD wrote {len(saved_files)} files under "
              f"{output_dir}/{{sensor,hits,step,labl}}/")
        return

    lucid_opts = config.get("lucid_options", {})
    saved_files = run_particles(
        simulator, str(root_file),
        output_dir=str(output_dir),
        file_index=file_index,
        n_events=n_events,
        master_seed=master_seed,
        job_id=job_id,
        readout=readout,
        dataset_name=config["name"],
        # Off by default: PhotonSim already throws isotropic directions. Enable to
        # re-emit the same Geant4 events at new orientations (augmentation).
        apply_rotation=bool(lucid_opts.get("apply_rotation", False)),
        primary_source=primary_source(config),
        pad_size_buckets=lucid_opts.get("pad_size_buckets", None),
        **_lucid_options(config),
    )
    print(f"LUCiD wrote {len(saved_files)} files under {output_dir}/{{sensor,hits,step,labl}}/")


def _prepare(args: argparse.Namespace, config: dict):
    """What every job checks before PhotonSim: the event count, the steps to run, the PhotonSim
    binary and the output directory. Returns ``(n_events, photonsim_bin, output_dir)`` or an
    exit code."""
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

    # The sensor/hits/step/labl subdirs are pre-created by the host-side fan-out
    # (dataprod_fanout.py) before any sub-job is submitted, so the worker does not race
    # other sub-jobs to create them; the writer creates them again before it opens the files.
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return n_events, photonsim_bin, output_dir


def _verify(output_dir: Path, file_index: int, config: dict) -> int:
    print("\n=== Step 4: Verifying output ===", flush=True)
    from lucid.dataset.layout import check_batch
    ok, messages = check_batch(output_dir, file_index, expected_dataset_name=config["name"])
    for m in messages:
        print(f"    {m}")
    return EXIT_OK if ok else EXIT_VERIFY


def _lucid_seed(args: argparse.Namespace) -> int:
    """The master seed of the LUCiD-side draws, resolved once for every path (single-vertex,
    pile-up, supernova) by :func:`~lucid.production.dataprod.seeds.resolve_master_seed`.
    PhotonSim / GENIE keep --master-seed, so a distinct --lucid-seed re-places and re-orients the
    same Geant4 sample."""
    from lucid.production.dataprod.seeds import resolve_master_seed
    return resolve_master_seed(args.lucid_seed if args.lucid_seed is not None else args.master_seed)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # Overlap-cache location. Read here and not in the forward path (B6 ratchet).
    # Documented resolution order: $LUCID_OVERLAP_CACHE_DIR -> install dir if
    # writable -> $XDG_CACHE_HOME/lucid/spatial_overlap_integrals.
    from lucid.overlap import set_cache_dir, set_user_cache_root
    if os.environ.get("LUCID_OVERLAP_CACHE_DIR"):
        set_cache_dir(os.environ["LUCID_OVERLAP_CACHE_DIR"])
    if os.environ.get("XDG_CACHE_HOME"):
        set_user_cache_root(os.environ["XDG_CACHE_HOME"])

    # 1. Load config
    try:
        config = _load_config(args.config)
    except Exception as e:
        print(f"Error loading config {args.config!r}: {e}", file=sys.stderr)
        return EXIT_CONFIG

    # A config that declares its detector (SN / Solar / Test) must agree with --detector; the
    # fan-out passes the declared one. The readout is resolved before PhotonSim runs.
    declared = config.get("detector")
    if declared is not None and args.detector is not None and declared != args.detector:
        print(f"Error: the dataset config declares detector {declared!r}, but --detector is "
              f"{args.detector!r}.", file=sys.stderr)
        return EXIT_CONFIG
    args.detector = declared or args.detector
    if args.detector is None and not args.skip_lucid:
        print(f"Error: {args.config!r} declares no 'detector'; pass --detector (e.g. SK_WAND).",
              file=sys.stderr)
        return EXIT_CONFIG
    readout = None
    if not args.skip_lucid:
        try:
            readout = _readout(config, args.detector)
        except (OSError, ValueError) as e:
            print(f"Error resolving the readout of {args.config!r}: {e}", file=sys.stderr)
            return EXIT_CONFIG

    # The mass ordering (supernova) is resolved before any step, like the readout.
    if config.get("primary_source") == "supernova":
        from lucid.production.dataprod.generators.supernova import resolve_ordering
        try:
            args.sn_ordering, _ = resolve_ordering(config["supernova"], args.sn_ordering)
        except ValueError as e:
            print(f"Error in {args.config!r}: {e}", file=sys.stderr)
            return EXIT_CONFIG

    prepared = _prepare(args, config)
    if isinstance(prepared, int):
        return prepared
    n_events, photonsim_bin, output_dir = prepared
    if config.get("pile_up"):
        return _main_pileup(args, config, n_events, photonsim_bin, output_dir, readout)

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
    # primary count so the downstream dataset has one entry per primary.
    photonsim_events = n_events

    # Derive per-subprocess seeds from the (master_seed, job_id, vertex_idx=0)
    # hierarchy. vertex_idx=0 is the only stream today; pile-up configs
    # (commit 3) will iterate this per vertex.
    from lucid.production.dataprod.seeds import derive_subprocess_seeds
    subproc_seeds = derive_subprocess_seeds(
        args.master_seed, args.job_id, vertex_idx=0
    )

    if uses_genie and not args.skip_genie:
        try:
            from lucid.production.dataprod.generators.genie import run_genie
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
            from lucid.production.dataprod.generators.supernova import run_sntools
            print("\n=== Step 0: supernova (sntools) event generation ===", flush=True)
            sn_cap = None
            if args.test:
                sn_cap = 2
            elif args.n_events is not None:
                sn_cap = args.n_events
            elif config["supernova"].get("cap_events") is not None:
                sn_cap = int(config["supernova"]["cap_events"])
            t_sn = time.time()
            produced, n_entries = run_sntools(
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
        from lucid.production.dataprod.generators.macro import generate_macro
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

    # For GENIE configs the LUCiD step sees one entry per injected primary, matching the
    # PhotonSim output; for particle-gun configs photonsim_events == n_events.
    try:
        _run_lucid(
            root_file=root_path,
            output_dir=output_dir,
            config=config,
            file_index=file_index,
            n_events=photonsim_events,
            master_seed=_lucid_seed(args),
            job_id=args.job_id,
            detector=args.detector,
            readout=readout,
            sn_ordering=args.sn_ordering,
        )
    except Exception as e:
        import traceback
        print(f"LUCiD failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return EXIT_LUCID

    rc = _verify(output_dir, file_index, config)
    if rc != EXIT_OK:
        return rc

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


def _main_pileup(args: argparse.Namespace, config: dict, n_events: int,
                 photonsim_bin: Optional[str], output_dir: Path, readout) -> int:
    """Run a pile-up job: N PhotonSim streams -> merged event."""
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

    from lucid.production.dataprod.seeds import derive_subprocess_seeds
    from lucid.production.dataprod.generators.macro import generate_macro

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
                from lucid.production.dataprod.generators.genie import run_genie
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
    from lucid.production.dataprod.drivers import run_pileup

    simulator = _simulator(args.detector)
    print(f"\n=== Step 3: LUCiD pile-up merger ({n_vertices} streams) ===", flush=True)
    saved_files = run_pileup(
        simulator, [str(p) for p in root_paths], vertex_sources,
        output_dir=str(output_dir),
        file_index=file_index,
        # Cap at the user-requested n_events. GENIE emits variable
        # primaries per interaction, so each vertex's PhotonSim ROOT
        # typically contains more events than requested; without this
        # cap the merger would use min(per_file_counts) and the job
        # would produce far more events than --n-events / --test asked for.
        n_events=n_events,
        master_seed=_lucid_seed(args),
        job_id=args.job_id,
        readout=readout,
        dataset_name=config["name"],
        **_lucid_options(config),
    )
    print(f"LUCiD wrote {len(saved_files)} files.")

    rc = _verify(output_dir, file_index, config)
    if rc != EXIT_OK:
        return rc

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
