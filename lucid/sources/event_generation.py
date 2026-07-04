"""Event generation drivers (single-vertex and pile-up).

Contains ``generate_events_from_photonsim_particles``,
``generate_events_from_photonsim_pileup``, and their internal helpers
(``_offset_track_ids_raw``, ``_merge_pileup_streams``).
"""
from __future__ import annotations

import os
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from lucid.sources.seed_utils import (
    T0_HALF_WINDOW_NS,
    _resolve_master_seed,
    derive_event_keys,
)
from lucid.sources.root_reader import _read_event_raw
from lucid.sources.event_builder import (
    _DEFAULT_PAD_SIZE_BUCKETS,
    _aggregate_from_photon_records,
    _derive_views_from_segments,
    _normalize_buckets,
    _trace_event_bucketed,
    _warmup_buckets,
    build_hits_sparse_per_process,
    build_seg_hits_merged_per_process,
    gather_photon_deposits,
    combine_t_per_sensor_across_processes,
    run_event_process_pipeline,
)
from lucid.sources.digitizer import digitize_and_decompose, resolve_model_config
from lucid.sources.scintillation_photons import expand_segments_to_photons
from lucid.sources.v3_writer import (
    EMISSION_PROCESS_CHERENKOV,
    EMISSION_PROCESS_SCINTILLATION,
    SOURCE_TYPE_GENIE,
    SOURCE_TYPE_PARTICLES,
    _compute_contained,
    _source_type_code,
    build_interaction_metadata,
    sample_translation_vector,
    save_hits_event_v3,
    save_labl_event_v3,
    save_step_event_v3,
    save_sensor_event_v3,
    write_hits_config_v3,
    write_labl_config_v3,
    write_step_config_v3,
    write_sensor_config_v3,
)
from lucid.sources.particle_physics import derive_particle_interaction_idx

__all__ = [
    "generate_events_from_photonsim_particles",
    "generate_events_from_photonsim_pileup",
]


def _detector_bounds_from_det_geom(det_geom):
    """Compute the ``detector_bounds`` dict from a ``DetectorGeometry``.

    Mirrors the per-shape branches that used to parse ``detector_config_path``
    JSON, but reads ``det_geom.detector_type`` and the per-shape attributes
    of ``det_geom.detector`` (e.g. ``Cylinder.r/H``, ``Sphere.r``,
    ``Box.L/W/H``) so the data-path wrapper doesn't have to re-load the
    geometry config. Returns ``None`` for shapes that don't enclose a
    canonical volume (e.g. 'string' arrays).
    """
    if det_geom is None or det_geom.detector is None:
        return None
    dt = str(det_geom.detector_type).lower()
    d = det_geom.detector
    if dt == 'cylinder':
        return {'type': 'cylinder', 'radius': float(d.r), 'height': float(d.H)}
    if dt == 'sphere':
        return {'type': 'sphere', 'radius': float(d.r)}
    if dt == 'box':
        return {'type': 'box',
                'length': float(d.L),
                'width':  float(d.W),
                'height': float(d.H)}
    return None


def generate_events_from_photonsim_particles(event_simulator, root_file_path,
                                             sensor_positions, output_dir=None,
                                             n_events=None, batch_size=100, master_seed=None,
                                             job_id=1,
                                             apply_smearing=False, apply_rotation=False, apply_translation=False,
                                             dataset_name='unnamed_dataset', run_id=None,
                                             file_index_start=0,
                                             primary_source='particles',
                                             pad_size_buckets=None,
                                             digitizer=None):
    """Generate events from a PhotonSim ROOT file, writing v3 four-file batches.

    For each batch of events, writes four HDF5 files under ``output_dir``:
    ``sensor/wc_sensor_NNNN.h5``, ``hits/wc_hits_NNNN.h5``,
    ``step/wc_step_NNNN.h5``, ``labl/wc_labl_NNNN.h5``. See
    ``docs/LUCID_DATASET.md`` for the full schema.

    Parameters
    ----------
    event_simulator : Callable
        Per-particle simulator with baked-in detector_params. Built via
        ``setup_event_simulator(..., default_detector_params=True)``; the
        call signature is ``(track_params, key, photonsim_data)``.
    root_file_path : str
        PhotonSim ROOT file path.
    sensor_positions : array-like (n_sensors, 3)
        PMT coordinates in meters.
    output_dir : str
        Dataset root directory; four subdirs are created under it.
    n_events : int, optional
        Number of events to generate (default: all entries in the ROOT file).
    batch_size : int
        Number of events per v3 batch file.
    master_seed : int, optional
        JAX PRNG seed; random if None.
    job_id : int
        1-based job id. Folded into the seed hierarchy so reusing
        ``master_seed`` across jobs yields independent RNG streams.
    apply_smearing, apply_rotation, apply_translation : bool
        Transform toggles; rotation is ignored (PhotonSim handles it).
    dataset_name : str
        Provenance: dataset identifier written to every ``config/`` group.
    run_id : str, optional
        Provenance: unique batch identifier; auto-UUID4 if None.
    file_index_start : int
        Index of the first batch file in this invocation (default 0).
    primary_source : str
        'particles' or 'genie'. Written into ``per_interaction/source_type``
        for every event of this batch.

    Notes
    -----
    ``detector_type`` / ``material`` / ``detector_bounds`` are derived
    internally from ``event_simulator.det_geom`` and ``event_simulator.medium``
    (attached by :func:`setup_event_simulator`). No separate
    ``detector_config_path`` arg is needed.
    """
    source_type_code = _source_type_code(primary_source)
    import uproot
    import time
    import uuid
    import subprocess
    from pathlib import Path
    from lucid.detector_params import ParticleParams
    import numpy as np
    from lucid.utils import smear_charges_SK_like

    # Generate random seed if not provided
    if master_seed is None:
        master_seed = int(time.time() * 1000000) % (2**32)
        print(f"Generated random master seed from time: {master_seed}")
    else:
        print(f"Using provided master seed: {master_seed}")

    # Resolve run_id
    if run_id is None:
        run_id = str(uuid.uuid4())
    print(f"Run id: {run_id}")

    # Resolve sensor positions / n_sensors
    sensor_positions_np = np.asarray(sensor_positions, dtype=np.float32)
    if sensor_positions_np.ndim != 2 or sensor_positions_np.shape[1] != 3:
        raise ValueError(
            f"sensor_positions must have shape (n_sensors, 3); got {sensor_positions_np.shape}")
    n_sensors = int(sensor_positions_np.shape[0])

    # Resolve git commit for provenance (fallback to env or 'unknown')
    try:
        lucid_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=lucid_repo_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = 'unknown'  # de-env (B6): git rev-parse above is primary; no env read in the forward/sources path

    # Create output directory tree
    out_root = Path(output_dir)
    for subdir in ('sensor', 'hits', 'step', 'labl'):
        (out_root / subdir).mkdir(parents=True, exist_ok=True)

    source_file_abs = os.path.abspath(root_file_path)

    # Open ROOT file and get number of entries
    print(f"Loading ROOT file: {root_file_path}")
    root_file = uproot.open(root_file_path)
    tree = root_file['OpticalPhotons']
    num_entries = tree.num_entries
    print(f"  Found {num_entries} entries")
    root_file.close()

    # Determine number of events to generate
    if n_events is None:
        n_events = num_entries
        print(f"No n_events specified, using all {n_events} entries")

    # Resolve bucket spec. Bucketing is mandatory now — the legacy
    # file-max single-PAD_SIZE path was removed when categorization
    # moved downstream of the kernel call (the per-particle vmap that
    # path relied on no longer exists). An empty/None ``pad_size_buckets``
    # falls back to ``_DEFAULT_PAD_SIZE_BUCKETS``.
    if pad_size_buckets is None or not pad_size_buckets:
        pad_size_buckets = _DEFAULT_PAD_SIZE_BUCKETS
    pad_size_buckets = _normalize_buckets(pad_size_buckets)
    use_bucketing = True
    print(f"  Using rays buckets: {list(pad_size_buckets)}")

    digitizer_model = resolve_model_config(digitizer)

    print(f"\nGenerating {n_events} events using VMAP-OPTIMIZED particle-based processing...")
    print(f"Using batch size of {batch_size} events for multithreaded I/O")
    print(f"Apply smearing: {apply_smearing}")
    print(f"Apply translation: {apply_translation}")
    print(f"Digitizer model: {digitizer_model['model']} "
          f"(dark_rate_khz={digitizer_model.get('dark_rate_khz', 0.0)})")
    # Note: Rotation is not applied in this workflow because PhotonSim already generates
    # primaries with random isotropic directions (/gun/randomDirection true). The photon
    # and track data are already in randomized coordinate frames, so rotation in LUCiD
    # would be redundant. Only translation is needed to place the vertex in the detector.
    if apply_rotation:
        print(f"WARNING: apply_rotation=True was passed but rotation is disabled in this workflow.")
        print(f"         PhotonSim already generates tracks with random directions, so rotation is unnecessary.")
    print(f"Saving events to directory: {output_dir}")

    # Provenance + detector_bounds derived from the simulator's attached
    # geometry. setup_event_simulator stamps `.det_geom` and `.medium` on
    # the returned callable (alongside `.default_detector_params`) so the
    # data-path wrapper doesn't have to re-load any config files.
    det_geom = getattr(event_simulator, 'det_geom', None)
    if det_geom is None:
        raise ValueError(
            "event_simulator has no .det_geom attribute — rebuild via "
            "setup_event_simulator(..., default_detector_params=True or "
            "DetectorParams) which attaches the geometry.")
    detector_type = str(det_geom.detector_type)
    material = str(det_geom.medium.material)
    detector_bounds = _detector_bounds_from_det_geom(det_geom)
    if apply_translation and detector_bounds is None:
        raise ValueError(
            f"apply_translation=True requires a detector with canonical "
            f"bounds; got detector_type={detector_type!r} (no bounds defined).")
    if detector_bounds is not None:
        print(f"Detector bounds: {detector_bounds}")

    saved_files = []
    event_times = []  # Track event processing times

    # Warm up the JIT cache once for every (rays_b, seg_b) bucket pair we
    # expect to hit. After this returns, every per-event kernel call lands
    # on a cached compile and runs at native cost. ~10-30s per pair on CPU,
    # one-time, amortized over thousands of events. Skipped in legacy mode
    # — only one (n_rays, n_segments) shape there, so first-event compile
    # suffices.
    if use_bucketing:
        _warmup_buckets(event_simulator, pad_size_buckets)

    # Vmap over a particle/segment axis — defined once and reused for both
    # the legacy per-event-vmap path and the optional segment-sensor-map
    # second pass below. The bucketed per-event path doesn't use this; it
    # calls ``event_simulator`` directly per (particle, chunk).
    def _simulate_single_particle(track_energy, track_pos, track_dir, photon_origins,
                                  photon_dirs, photon_times, photon_wavelengths, N, sim_key):
        track_params = ParticleParams.from_cartesian(
            energy=track_energy, position=track_pos, direction=track_dir, t0=0.0,
        )
        photonsim_data = {
            # Unify-on-cm: the simulator's data impl divides photon_origins by 100
            # (cm->m, matching the recon pad_photon_data convention). root_reader /
            # event_generation work in METERS, so convert m->cm here at the boundary.
            'photon_origins': photon_origins * 100.0,
            'photon_directions': photon_dirs,
            'photon_times': photon_times,
            'wavelengths': photon_wavelengths,
            'N': N,
            'apply_rotation': False,
            'rotation_axis': jnp.array([1.0, 0.0, 0.0]),
            'rotation_angle': 0.0,
            # Photons were already translated in NumPy upstream; do NOT ask
            # the JIT simulator to translate again.
            'apply_translation': False,
            'translation_vector': jnp.zeros(3),
        }
        return event_simulator(track_params, sim_key, photonsim_data)

    simulate_all_particles = jax.vmap(
        _simulate_single_particle,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0),
    )

    # Create batches
    num_batches = (n_events + batch_size - 1) // batch_size

    # Process each batch
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_events)
        batch_size_actual = end_idx - start_idx

        print(f"Processing batch {batch_idx+1}/{num_batches} (events {start_idx} to {end_idx-1})")

        # Lists to accumulate batch data
        batch_data = []
        batch_filenames = []
        batch_indices = []

        # Process each entry in the current batch
        for event_idx in range(start_idx, end_idx):
            event_start_time = time.time()
            print(f"\n  Event {event_idx+1}/{n_events} (index {event_idx}):", flush=True)

            # Deterministic RNG keys for this (job, event). All per-event
            # draws — vertex translation, t0, simulator, smearing — flow
            # from this hierarchy so reusing --master-seed across jobs
            # yields independent streams.
            event_keys = derive_event_keys(master_seed, job_id, event_idx,
                                           interaction_idx=0)
            master_key = event_keys['sim_key']
            t0 = float(np.random.default_rng(
                seed=event_keys['t0_seed']).uniform(
                    -T0_HALF_WINDOW_NS, T0_HALF_WINDOW_NS))

            # Draw the vertex once, up front — both dark-event and normal
            # branches write it into per_interaction/. When apply_translation
            # is False the vertex is the origin (nothing to apply).
            if apply_translation and detector_bounds is not None:
                vertex_rng = np.random.default_rng(
                    seed=event_keys['vertex_seed'])
                translation_vector = sample_translation_vector(
                    detector_bounds, vertex_rng)
            else:
                translation_vector = np.zeros(3, dtype=np.float32)

            # ----------------------------------------------------------------
            # Phase 1 — read raw event from ROOT (no categorization).
            # ----------------------------------------------------------------
            print(f"    Reading raw event from ROOT file...", flush=True)
            _t_root = time.perf_counter()
            raw = _read_event_raw(root_file_path, event_idx)
            print(f"    [timing] root_read {time.perf_counter() - _t_root:.3f}s", flush=True)

            n_segments_raw = raw['segments_raw']['n_segments']
            total_photons = int(raw['photon_origins'].shape[0])
            print(f"    Raw: photons={total_photons:,}  "
                  f"segments={n_segments_raw:,}", flush=True)

            # Apply vertex translation to the raw photon origins and the raw
            # segment positions BEFORE tracing, so the kernel sees vertex-
            # translated photons and the downstream segments dict carries the
            # translated positions too. The raw segment table stays in mm
            # (``_derive_views_from_segments`` converts to m), so the
            # translation_vector (m) is scaled by 1000 here.
            _t_pre = time.perf_counter()
            photon_origins_np      = raw['photon_origins'].astype(np.float32, copy=False)
            photon_directions_np   = raw['photon_directions'].astype(np.float32, copy=False)
            photon_times_np        = raw['photon_times'].astype(np.float32, copy=False)
            photon_wavelengths_np  = raw['photon_wavelengths'].astype(np.float32, copy=False)
            photon_segment_index_raw = np.asarray(raw['photon_segment_index_raw'], dtype=np.int64)

            if apply_translation:
                photon_origins_np = photon_origins_np + translation_vector[None, :]
                seg = raw['segments_raw']
                seg['start_x_mm'] = seg['start_x_mm'] + float(translation_vector[0]) * 1000.0
                seg['start_y_mm'] = seg['start_y_mm'] + float(translation_vector[1]) * 1000.0
                seg['start_z_mm'] = seg['start_z_mm'] + float(translation_vector[2]) * 1000.0
                seg['end_x_mm']   = seg['end_x_mm']   + float(translation_vector[0]) * 1000.0
                seg['end_y_mm']   = seg['end_y_mm']   + float(translation_vector[1]) * 1000.0
                seg['end_z_mm']   = seg['end_z_mm']   + float(translation_vector[2]) * 1000.0
            print(f"    [timing] preprocess {time.perf_counter() - _t_pre:.3f}s", flush=True)

            # ----------------------------------------------------------------
            # Build per-process photon-stream inputs. Always Cherenkov from
            # the PhotonSim arrays we just read; scintillation gets added
            # when the medium baked into the kernel has it in
            # ``emission_processes`` (Option C: one kernel call per process,
            # see .claude/plans/scintillation-data-mode.md).
            # ----------------------------------------------------------------
            medium = getattr(event_simulator, 'medium', None)
            has_scintillation = (
                medium is not None
                and "scintillation" in medium.emission_processes
            )

            process_inputs = [{
                'process_id':                EMISSION_PROCESS_CHERENKOV,
                'photon_origins':            photon_origins_np,
                'photon_directions':         photon_directions_np,
                'photon_times':              photon_times_np,
                'photon_wavelengths':        photon_wavelengths_np,
                'photon_segment_index_raw':  photon_segment_index_raw,
            }]

            if has_scintillation:
                # Scintillation scalars live on the NESTED ScintillationParams sub-tuple of
                # DetectorParams (no flat alias) — read via `.scintillation.*`.
                _sc = event_simulator.default_detector_params.scintillation
                _medium_params = {
                    'S':           float(_sc.S),
                    'kB':          float(_sc.kB),
                    'C':           float(_sc.C),
                    'tau_rise':    float(_sc.tau_rise),
                    'tau_fall':    float(_sc.tau_fall),
                    'moyal_loc':   float(_sc.moyal_loc),
                    'moyal_scale': float(_sc.moyal_scale),
                    'lambda_min':  float(medium.scintillation_lambda_min),
                    'lambda_max':  float(medium.scintillation_lambda_max),
                }
                _scint_rng = np.random.default_rng(event_keys['scint_seed'])
                # raw['segments_raw'] has already been translated above
                # (positions in vertex-shifted frame, mm). The expander
                # converts to m on the way out, so its photon_origins are
                # already in the absolute detector frame — no extra add.
                _scint_ph = expand_segments_to_photons(
                    raw['segments_raw'], _medium_params, _scint_rng)
                process_inputs.append({
                    'process_id':               EMISSION_PROCESS_SCINTILLATION,
                    'photon_origins':           _scint_ph['photon_origins'],
                    'photon_directions':        _scint_ph['photon_directions'],
                    'photon_times':             _scint_ph['photon_times'],
                    'photon_wavelengths':       _scint_ph['photon_wavelengths'],
                    'photon_segment_index_raw': _scint_ph['photon_segment_index_raw'],
                })

            # ----------------------------------------------------------------
            # Phase 2-4 per emission process: bucketed kernel + derive_views
            # + aggregator. Per-process sim_key folded off master_key by
            # process_id so each process gets a distinct kernel RNG stream.
            # ----------------------------------------------------------------
            print(f"    Running per-process pipeline "
                  f"({len(process_inputs)} process(es), "
                  f"rays_buckets={list(pad_size_buckets)})...", flush=True)
            sim_start_time = time.time()
            _t_sim = time.perf_counter()
            process_outputs = []
            for _p_in in process_inputs:
                _sim_key_p = jax.random.fold_in(
                    master_key, int(_p_in['process_id']))
                _out = run_event_process_pipeline(
                    event_simulator=event_simulator,
                    raw=raw,
                    photon_origins_np=_p_in['photon_origins'],
                    photon_directions_np=_p_in['photon_directions'],
                    photon_times_np=_p_in['photon_times'],
                    photon_wavelengths_np=_p_in['photon_wavelengths'],
                    photon_segment_index_raw=_p_in['photon_segment_index_raw'],
                    n_sensors=n_sensors,
                    rays_buckets=pad_size_buckets,
                    sim_key=_sim_key_p,
                )
                _out['process_id'] = _p_in['process_id']
                process_outputs.append(_out)
            sim_elapsed = time.time() - sim_start_time
            print(f"    Simulation completed in {sim_elapsed:.2f}s", flush=True)
            print(f"    [timing] simulate {time.perf_counter() - _t_sim:.3f}s", flush=True)

            _t_post = time.perf_counter()

            # Canonical event-level view comes from the Cherenkov pipeline
            # (process_outputs[0]) — the segment table and particle
            # categorization are identical across processes (same `raw`).
            particle_data = process_outputs[0]['particle_data']
            n_particles = particle_data['n_particles']
            particles = particle_data['particles']
            n_segments = int(particle_data['segments']['n_segments'])

            # ----------------------------------------------------------------
            # Hit-making (digitization): window the per-photon deposits into
            # digits and build the digit_idx-aware decomposition. The model
            # comes from the detector config; `basic` (default) reproduces the
            # legacy first-arrival + summed-charge one-hit-per-sensor.
            #   sensor.h5           : digit list (sensor_idx may repeat)
            #   hits.h5 rows        : per-(particle, sensor, digit, process)
            #   step/sensor_hits    : per-(segment, sensor, digit, process)
            # ----------------------------------------------------------------
            deposits = gather_photon_deposits(process_outputs)
            digi_rng = np.random.default_rng([int(master_seed or 0), int(event_idx)])
            sensor_digits, hits_sparse, seg_hits = digitize_and_decompose(
                sensor_idx=deposits['sensor_idx'], charge=deposits['charge'],
                t_true=deposits['t_true'], t_reco=deposits['t_reco'],
                particle_idx=deposits['particle_idx'],
                segment_idx=deposits['segment_idx'],
                emission_process=deposits['emission_process'],
                n_sensors=int(n_sensors), model=digitizer_model, rng=digi_rng,
                dark_rate_khz=float(digitizer_model.get('dark_rate_khz', 0.0)))

            # Shift G4/vertex-frame times into the absolute detector frame by
            # adding t0. Every digit / decomposition row is a real hit, so a
            # flat add suffices (no no-hit sentinels to preserve).
            t0_f32 = np.float32(t0)
            for _d, _keys in ((sensor_digits, ('T',)),
                              (hits_sparse, ('T', 'T_reco')),
                              (seg_hits, ('T', 'T_reco'))):
                for _k in _keys:
                    if _k in _d and _d[_k].size:
                        _d[_k] = _d[_k] + t0_f32
            # Segments always carry meaningful times — shift all of them.
            if 'segments' in particle_data and particle_data['segments'].get('n_segments', 0) > 0:
                particle_data['segments']['time'] = \
                    np.asarray(particle_data['segments']['time'], dtype=np.float32) + t0_f32

            # Create filename
            event_number = event_idx
            filename = os.path.join(output_dir, f'event_{event_number}.h5')

            # t0 already drawn at top of loop from the seed hierarchy.
            print(f"    [timing] post_jax {time.perf_counter() - _t_post:.3f}s", flush=True)
            _t_meta = time.perf_counter()

            # Extended info with particle structure
            interaction_meta = build_interaction_metadata(
                particle_data, t0=t0, vertex_xyz=translation_vector,
                source_type_code=source_type_code)
            primary_to_interaction = {tid: 0 for tid in interaction_meta['primary_track_ids']}
            extended_info = {
                'n_particles': n_particles,
                'particles': particles,
                'track_info_dict': particle_data['track_info_dict'],
                'primary_to_interaction': primary_to_interaction,
                'interaction_metadata': [interaction_meta],
                # hits payload: per-(particle, sensor, digit, process) rows
                # (with digit_idx). ``save_hits_event_v3`` consumes it directly.
                'hits_sparse': hits_sparse,
                # sensor.h5 digit list produced by the digitizer.
                'sensor_digits': sensor_digits,
                'source': 'PhotonSim_Particles_VMAP',
            }

            if 'meaningful_tracks' in particle_data:
                extended_info['meaningful_tracks'] = particle_data['meaningful_tracks']
                extended_info['segments'] = particle_data['segments']

            # step sparse triplets came directly from the host aggregator
            # (seg_hits dict already in writer-ready shape, with an
            # emission_process column folded in by
            # build_seg_hits_merged_per_process). Skip the writer wiring
            # when there are no hits at all.
            if seg_hits is not None and seg_hits['PE'].size > 0:
                extended_info['segment_sensor_hits'] = seg_hits

            # Geometric containment (per-segment / particle / interaction / event).
            # Requires meaningful_tracks + segments for the ownership walk.
            cont = _compute_contained(extended_info, detector_bounds)
            extended_info['contained_per_segment']     = cont['per_segment']
            extended_info['contained_per_particle']    = cont['per_particle']
            extended_info['contained_per_interaction'] = cont['per_interaction']
            extended_info['contained']                 = cont['overall']
            print(f"    [timing] meta_contain {time.perf_counter() - _t_meta:.3f}s", flush=True)

            # Store for batch processing
            extended_info['source_event_idx'] = int(event_number)
            batch_data.append(extended_info)
            batch_indices.append(event_number)

            event_total_time = time.time() - event_start_time
            event_times.append(event_total_time)
            print(f"    Event total time: {event_total_time:.2f}s", flush=True)

        # Write this batch as four v3 files (sensor/inst/seg/labl)
        print(f"Saving batch {batch_idx+1} as v3 four-file group...")
        t_save_start = time.time()

        file_idx = int(file_index_start + batch_idx)
        sensor_path = out_root / 'sensor' / f'wc_sensor_{file_idx:04d}.h5'
        hits_path = out_root / 'hits' / f'wc_hits_{file_idx:04d}.h5'
        step_path = out_root / 'step' / f'wc_step_{file_idx:04d}.h5'
        labl_path = out_root / 'labl' / f'wc_labl_{file_idx:04d}.h5'

        batch_src_idx = np.asarray(batch_indices, dtype=np.uint32)

        config_meta = {
            'n_events': len(batch_data),
            'git_commit': git_commit,
            'run_id': run_id,
            'dataset_name': dataset_name,
            'file_index': file_idx,
            'source_file': source_file_abs,
            'lucid_master_seed': int(master_seed),
            'photonsim_seed': -1,
            'n_sensors': n_sensors,
            'detector_type': detector_type,
            'material': material,
            'smearing_applied': bool(apply_smearing),
            'smearing_charge_function': 'SK_like' if apply_smearing else 'none',
            'smearing_time_function': 'SK_like' if apply_smearing else 'none',
            'label_names': ['category'],
        }

        # Optional geometry hints for step config
        if detector_bounds is not None:
            config_meta['detector_shape'] = detector_bounds['type']
            if detector_bounds['type'] == 'cylinder':
                config_meta['detector_radius'] = float(detector_bounds['radius'])
                config_meta['detector_half_height'] = float(detector_bounds['height']) / 2.0
                config_meta['detector_axis'] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            elif detector_bounds['type'] == 'sphere':
                config_meta['detector_radius'] = float(detector_bounds['radius'])
            elif detector_bounds['type'] == 'box':
                l, w, h = detector_bounds['length'], detector_bounds['width'], detector_bounds['height']
                config_meta['detector_bbox'] = np.array(
                    [-l/2, l/2, -w/2, w/2, -h/2, h/2], dtype=np.float32)

        with h5py.File(sensor_path, 'w') as fs, h5py.File(hits_path, 'w') as fi, \
                h5py.File(step_path, 'w') as fg, h5py.File(labl_path, 'w') as fl:
            write_sensor_config_v3(fs, config_meta, batch_src_idx, sensor_positions_np)
            write_hits_config_v3(fi, config_meta, batch_src_idx, sensor_positions_np)
            write_step_config_v3(fg, config_meta, batch_src_idx)
            write_labl_config_v3(fl, config_meta, batch_src_idx)

            for seq_idx, evdict in enumerate(batch_data):
                save_sensor_event_v3(fs, evdict, seq_idx)
                save_hits_event_v3(fi, evdict, seq_idx)
                save_step_event_v3(fg, evdict, seq_idx)
                save_labl_event_v3(fl, evdict, seq_idx)

        saved_files.extend([str(sensor_path), str(hits_path), str(step_path), str(labl_path)])

        t_save = time.time() - t_save_start
        print(f"Batch {batch_idx+1} save time: {t_save:.3f}s\n")

    print(f"\nSuccessfully wrote {num_batches} batches "
          f"({len(saved_files)} files total) to {output_dir}/"
          f"{{sensor,hits,step,labl}}/")

    # Print average event time
    if event_times:
        avg_time = sum(event_times) / len(event_times)
        print(f"Average event processing time: {avg_time:.3f}s")

    return saved_files


def _offset_track_ids_raw(raw, offset):
    """Shift all G4 track IDs in a ``_read_event_raw`` output by ``offset``.

    Shifts all G4 track IDs in the raw read dict (no ``meaningful_tracks``
    / ``particles`` yet — those come from
    :func:`_derive_views_from_segments` after the kernel call). Shifts:

      - ``raw['track_info_dict']``: rekey + shift each record's ``track_id``
        and ``parent_id``.
      - ``raw['segments_raw']['track_id']``: vectorized shift on the int64
        column.

    ``parent_id == 0`` (primary convention) is preserved so primaries stay
    recognizable after merging. ``photon_segment_index_raw`` is **not**
    shifted (it indexes the segment table, not a track id).

    Returns the max track_id seen post-shift so the caller can advance the
    running_offset for the next vertex stream.
    """
    if offset == 0:
        # Nothing to do; just compute the max for the caller.
        tid_d = raw.get('track_info_dict') or {}
        return max((int(t) for t in tid_d.keys()), default=0)

    def _shift(tid):
        return int(tid) + offset if int(tid) > 0 else 0

    tid_dict = raw.get('track_info_dict')
    if tid_dict:
        new_tid = {}
        for tid, t in tid_dict.items():
            t = dict(t)
            t['track_id']  = _shift(t.get('track_id', tid))
            t['parent_id'] = _shift(t.get('parent_id', 0))
            new_tid[_shift(tid)] = t
        raw['track_info_dict'] = new_tid

    seg = raw.get('segments_raw')
    if seg and seg.get('n_segments', 0) > 0:
        # Vectorized: track_id > 0 -> +offset, track_id <= 0 -> 0.
        tid_arr = np.asarray(seg['track_id'], dtype=np.int64)
        seg['track_id'] = np.where(tid_arr > 0, tid_arr + np.int64(offset),
                                    np.int64(0))

    tid_d = raw.get('track_info_dict') or {}
    return max((int(t) for t in tid_d.keys()), default=0)


def generate_events_from_photonsim_pileup(
    event_simulator,
    root_file_paths,
    vertex_primary_sources,
    sensor_positions,
    output_dir=None,
    n_events=None,
    batch_size=100,
    master_seed=None,
    job_id=1,
    apply_smearing=False,
    apply_translation=False,
    dataset_name='unnamed_pileup_dataset',
    run_id=None,
    file_index_start=0,
):
    """Generate pile-up events by merging N PhotonSim streams per event.

    Each entry in ``root_file_paths`` is a PhotonSim ROOT file from one
    vertex's interaction. For each event index, we draw an independent
    absolute t0 and fiducial vertex per vertex, simulate each vertex's
    photons, remap G4 track IDs to avoid collisions, and merge the
    per-vertex results into one event_dict. Sensor/hits/step/labl are
    written using the same v3 writers as the single-vertex path.

    Parameters
    ----------
    root_file_paths : list[str | Path]
        One PhotonSim ROOT file per vertex, matched by index to
        ``vertex_primary_sources``.
    vertex_primary_sources : list[str]
        'particles' or 'genie' per vertex, used to set
        per_interaction/source_type for each primary from that vertex.
    """
    import uproot
    import time as _time
    import uuid
    import subprocess
    from pathlib import Path

    if len(root_file_paths) != len(vertex_primary_sources):
        raise ValueError(
            f"root_file_paths and vertex_primary_sources length mismatch: "
            f"{len(root_file_paths)} vs {len(vertex_primary_sources)}")
    N_vertices = len(root_file_paths)
    if N_vertices < 2:
        raise ValueError("Pile-up requires at least 2 vertices.")

    master_seed = _resolve_master_seed(master_seed)
    print(f"Pile-up: master_seed={master_seed}, job_id={job_id}, "
          f"n_vertices={N_vertices}")

    if run_id is None:
        run_id = str(uuid.uuid4())
    print(f"Run id: {run_id}")

    sensor_positions_np = np.asarray(sensor_positions, dtype=np.float32)
    n_sensors = int(sensor_positions_np.shape[0])

    # Git commit (same as non-pile-up)
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo_root,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_commit = 'unknown'  # de-env (B6): git rev-parse above is primary; no env read in the forward/sources path

    out_root = Path(output_dir)
    for sub in ('sensor', 'hits', 'step', 'labl'):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    # Determine common number of events across all ROOT files.
    per_file_counts = []
    for p in root_file_paths:
        with uproot.open(p) as f:
            per_file_counts.append(int(f['OpticalPhotons'].num_entries))
    common = min(per_file_counts)
    if n_events is None:
        n_events = common
    elif n_events > common:
        print(f"WARNING: requested n_events={n_events} exceeds min per-vertex "
              f"entries {per_file_counts}; capping to {common}.")
        n_events = common
    print(f"Per-vertex ROOT entries: {per_file_counts}; "
          f"merging {n_events} events.")

    # Bucket spec — same as the single-vertex driver. Single axis (rays)
    # since the host aggregator runs on per-photon flat lists in numpy.
    rays_buckets = _normalize_buckets(_DEFAULT_PAD_SIZE_BUCKETS)
    print(f"Pile-up rays buckets: {list(rays_buckets)}")

    # Warm up the JIT cache once per rays bucket before the per-event /
    # per-vertex loop. Pile-up shares the same compiled kernel.
    _warmup_buckets(event_simulator, rays_buckets)

    # Provenance + detector_bounds derived from event_simulator.det_geom /
    # .medium, attached by setup_event_simulator. Same pattern as the
    # single-vertex path; see _detector_bounds_from_det_geom for details.
    det_geom = getattr(event_simulator, 'det_geom', None)
    if det_geom is None:
        raise ValueError(
            "event_simulator has no .det_geom attribute — rebuild via "
            "setup_event_simulator(..., default_detector_params=True or "
            "DetectorParams) which attaches the geometry.")
    detector_type = str(det_geom.detector_type)
    material = str(det_geom.medium.material)
    detector_bounds = _detector_bounds_from_det_geom(det_geom)
    if apply_translation and detector_bounds is None:
        raise ValueError(
            f"apply_translation=True requires a detector with canonical "
            f"bounds; got detector_type={detector_type!r} (no bounds defined).")

    saved_files = []
    event_times = []
    num_batches = (n_events + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_events)
        print(f"Pile-up batch {batch_idx+1}/{num_batches} "
              f"(events {start_idx}..{end_idx-1})")
        batch_data = []
        batch_indices = []

        for event_idx in range(start_idx, end_idx):
            t_start = _time.time()
            print(f"\n  Event {event_idx+1}/{n_events}:", flush=True)
            streams = []
            running_offset = 0

            for vidx in range(N_vertices):
                event_keys = derive_event_keys(
                    master_seed, job_id, event_idx, interaction_idx=vidx)
                t0_i = float(np.random.default_rng(
                    seed=event_keys['t0_seed']).uniform(
                        -T0_HALF_WINDOW_NS, T0_HALF_WINDOW_NS))
                if apply_translation:
                    vrng = np.random.default_rng(seed=event_keys['vertex_seed'])
                    vertex_i = sample_translation_vector(detector_bounds, vrng)
                else:
                    vertex_i = np.zeros(3, dtype=np.float32)
                print(f"    vertex {vidx}: t0={t0_i:+.2f} ns, "
                      f"xyz=({vertex_i[0]:.3f}, {vertex_i[1]:.3f}, {vertex_i[2]:.3f}) m",
                      flush=True)

                # Phase 1 — read raw event from this vertex's ROOT file.
                raw = _read_event_raw(str(root_file_paths[vidx]), event_idx)

                # Remap G4 track IDs on the raw dict so streams don't
                # collide. After this, both ``track_info_dict`` keys and
                # ``segments_raw['track_id']`` carry the offset; the
                # downstream categorization in
                # ``_derive_views_from_segments`` produces already-shifted
                # ``meaningful_tracks`` / ``particles`` naturally.
                stream_max = _offset_track_ids_raw(raw, running_offset)

                source_type_code_i = _source_type_code(vertex_primary_sources[vidx])
                total_photons_i = int(raw['photon_origins'].shape[0])
                n_segments_raw_i = int(raw['segments_raw']['n_segments'])
                print(f"      raw: photons={total_photons_i:,}  "
                      f"segments={n_segments_raw_i:,}", flush=True)

                # Apply this vertex's translation to the raw photon origins
                # + raw segment table (mm scale on segments).
                photon_origins_i      = raw['photon_origins'].astype(np.float32, copy=False)
                photon_directions_i   = raw['photon_directions'].astype(np.float32, copy=False)
                photon_times_i        = raw['photon_times'].astype(np.float32, copy=False)
                photon_wavelengths_i  = raw['photon_wavelengths'].astype(np.float32, copy=False)
                photon_segment_index_i = np.asarray(raw['photon_segment_index_raw'], dtype=np.int32)
                if apply_translation:
                    photon_origins_i = photon_origins_i + vertex_i[None, :]
                    seg_raw_i = raw['segments_raw']
                    if n_segments_raw_i > 0:
                        seg_raw_i['start_x_mm'] = seg_raw_i['start_x_mm'] + float(vertex_i[0]) * 1000.0
                        seg_raw_i['start_y_mm'] = seg_raw_i['start_y_mm'] + float(vertex_i[1]) * 1000.0
                        seg_raw_i['start_z_mm'] = seg_raw_i['start_z_mm'] + float(vertex_i[2]) * 1000.0
                        seg_raw_i['end_x_mm']   = seg_raw_i['end_x_mm']   + float(vertex_i[0]) * 1000.0
                        seg_raw_i['end_y_mm']   = seg_raw_i['end_y_mm']   + float(vertex_i[1]) * 1000.0
                        seg_raw_i['end_z_mm']   = seg_raw_i['end_z_mm']   + float(vertex_i[2]) * 1000.0

                # Build per-process inputs for this vertex (Cherenkov +
                # optional scintillation). Same pattern as the single-vertex
                # path — Option C, one kernel call per process.
                medium = getattr(event_simulator, 'medium', None)
                has_scintillation_i = (
                    medium is not None
                    and "scintillation" in medium.emission_processes
                )
                process_inputs_i = [{
                    'process_id':               EMISSION_PROCESS_CHERENKOV,
                    'photon_origins':           photon_origins_i,
                    'photon_directions':        photon_directions_i,
                    'photon_times':             photon_times_i,
                    'photon_wavelengths':       photon_wavelengths_i,
                    'photon_segment_index_raw': photon_segment_index_i.astype(np.int64),
                }]
                if has_scintillation_i:
                    # Nested ScintillationParams sub-tuple (no flat alias) — read via `.scintillation.*`.
                    _sc = event_simulator.default_detector_params.scintillation
                    _medium_params = {
                        'S':           float(_sc.S),
                        'kB':          float(_sc.kB),
                        'C':           float(_sc.C),
                        'tau_rise':    float(_sc.tau_rise),
                        'tau_fall':    float(_sc.tau_fall),
                        'moyal_loc':   float(_sc.moyal_loc),
                        'moyal_scale': float(_sc.moyal_scale),
                        'lambda_min':  float(medium.scintillation_lambda_min),
                        'lambda_max':  float(medium.scintillation_lambda_max),
                    }
                    _scint_rng = np.random.default_rng(event_keys['scint_seed'])
                    # segments_raw already vertex-translated above; expander
                    # output's photon_origins are already in absolute frame.
                    _scint_ph = expand_segments_to_photons(
                        raw['segments_raw'], _medium_params, _scint_rng)
                    process_inputs_i.append({
                        'process_id':               EMISSION_PROCESS_SCINTILLATION,
                        'photon_origins':           _scint_ph['photon_origins'],
                        'photon_directions':        _scint_ph['photon_directions'],
                        'photon_times':             _scint_ph['photon_times'],
                        'photon_wavelengths':       _scint_ph['photon_wavelengths'],
                        'photon_segment_index_raw': _scint_ph['photon_segment_index_raw'],
                    })

                # Phase 2-4 per emission process for this vertex.
                _t_sim = _time.time()
                process_outputs_i = []
                for _p_in in process_inputs_i:
                    _sim_key_p = jax.random.fold_in(
                        event_keys['sim_key'], int(_p_in['process_id']))
                    _out = run_event_process_pipeline(
                        event_simulator=event_simulator,
                        raw=raw,
                        photon_origins_np=_p_in['photon_origins'],
                        photon_directions_np=_p_in['photon_directions'],
                        photon_times_np=_p_in['photon_times'],
                        photon_wavelengths_np=_p_in['photon_wavelengths'],
                        photon_segment_index_raw=_p_in['photon_segment_index_raw'],
                        n_sensors=n_sensors,
                        rays_buckets=rays_buckets,
                        sim_key=_sim_key_p,
                    )
                    _out['process_id'] = _p_in['process_id']
                    process_outputs_i.append(_out)
                print(f"      [timing] simulate {_time.time() - _t_sim:.3f}s", flush=True)

                # Canonical particle_data from Cherenkov pass (segments table
                # identical across processes).
                particle_data_i = process_outputs_i[0]['particle_data']
                n_particles_i = particle_data_i['n_particles']

                # Per-vertex combined per-sensor PE/T (across processes), plus
                # the writer-ready sparse rows tagged with emission_process.
                pe_sensor_i = np.sum(
                    [p['pe_per_sensor'] for p in process_outputs_i], axis=0)
                t_sensor_i = combine_t_per_sensor_across_processes(
                    [p['t_per_sensor'] for p in process_outputs_i])
                t_reco_sensor_i = combine_t_per_sensor_across_processes(
                    [p['t_reco_per_sensor'] for p in process_outputs_i])
                hits_sparse_i = build_hits_sparse_per_process(
                    process_outputs_i, n_particles_i)
                seg_hits_i = build_seg_hits_merged_per_process(process_outputs_i)

                # Apply +t0_i to shift this vertex's outputs into the absolute
                # detector frame. All time arrays are sparse from here.
                t0_f32 = np.float32(t0_i)
                pe_sensor_i = pe_sensor_i.astype(np.float32, copy=False)
                t_sensor_i = t_sensor_i.copy()
                t_reco_sensor_i = t_reco_sensor_i.copy()
                np.add(t_sensor_i,      t0_f32, out=t_sensor_i,      where=t_sensor_i      > 0)
                np.add(t_reco_sensor_i, t0_f32, out=t_reco_sensor_i, where=t_reco_sensor_i > 0)
                if hits_sparse_i['T'].size > 0:
                    hits_sparse_i['T'] = hits_sparse_i['T'] + t0_f32
                if hits_sparse_i.get('T_reco') is not None and hits_sparse_i['T_reco'].size > 0:
                    hits_sparse_i['T_reco'] = hits_sparse_i['T_reco'] + t0_f32
                if seg_hits_i is not None and seg_hits_i['PE'].size > 0:
                    seg_hits_i['T'] = seg_hits_i['T'] + t0_f32
                    if 'T_reco' in seg_hits_i and seg_hits_i['T_reco'].size > 0:
                        seg_hits_i['T_reco'] = seg_hits_i['T_reco'] + t0_f32
                # Same shift for segment times.
                if particle_data_i['segments'].get('n_segments', 0) > 0:
                    particle_data_i['segments']['time'] = (
                        np.asarray(particle_data_i['segments']['time'], dtype=np.float32)
                        + t0_f32)

                streams.append({
                    'particles':              particle_data_i['particles'],
                    'meaningful_tracks':      particle_data_i['meaningful_tracks'],
                    'segments':               particle_data_i['segments'],
                    'pe_per_sensor':          pe_sensor_i,
                    't_per_sensor':           t_sensor_i,
                    't_reco_per_sensor':      t_reco_sensor_i,
                    'hits_sparse':            hits_sparse_i,
                    'seg_hits':               seg_hits_i,
                    'interaction_meta':       build_interaction_metadata(
                        particle_data_i, t0=t0_i, vertex_xyz=vertex_i,
                        source_type_code=source_type_code_i),
                })
                running_offset = stream_max + 1

            # ---- merge streams into one event_dict ----
            _t_merge = _time.time()
            merged = _merge_pileup_streams(
                streams, n_sensors=n_sensors,
                apply_smearing=apply_smearing,
                smear_key=derive_event_keys(
                    master_seed, job_id, event_idx,
                    interaction_idx=N_vertices)['smear_key'],
                detector_bounds=detector_bounds,
            )
            print(f"    [timing] merge {_time.time() - _t_merge:.3f}s", flush=True)
            merged['source_event_idx'] = int(event_idx)
            merged['source'] = 'PhotonSim_Pileup'

            batch_data.append(merged)
            batch_indices.append(int(event_idx))
            event_total_time = _time.time() - t_start
            event_times.append(event_total_time)
            print(f"    Event total time: {event_total_time:.2f}s", flush=True)

        # Write batch (same as non-pile-up)
        file_idx = int(file_index_start + batch_idx)
        sensor_path = out_root / 'sensor' / f'wc_sensor_{file_idx:04d}.h5'
        hits_path   = out_root / 'hits'   / f'wc_hits_{file_idx:04d}.h5'
        step_path   = out_root / 'step'   / f'wc_step_{file_idx:04d}.h5'
        labl_path   = out_root / 'labl'   / f'wc_labl_{file_idx:04d}.h5'

        batch_src_idx = np.asarray(batch_indices, dtype=np.uint32)
        config_meta = {
            'n_events': len(batch_data),
            'git_commit': git_commit,
            'run_id': run_id,
            'dataset_name': dataset_name,
            'file_index': file_idx,
            'source_file': ','.join(os.path.abspath(str(p)) for p in root_file_paths),
            'lucid_master_seed': int(master_seed),
            'photonsim_seed': -1,
            'n_sensors': n_sensors,
            'detector_type': detector_type,
            'material': material,
            'smearing_applied': bool(apply_smearing),
            'smearing_charge_function': 'SK_like' if apply_smearing else 'none',
            'smearing_time_function': 'SK_like' if apply_smearing else 'none',
            'label_names': ['category'],
        }
        if detector_bounds is not None:
            config_meta['detector_shape'] = detector_bounds['type']
            if detector_bounds['type'] == 'cylinder':
                config_meta['detector_radius']      = detector_bounds['radius']
                config_meta['detector_half_height'] = detector_bounds['height'] / 2.0

        _t_save = _time.time()
        with h5py.File(sensor_path, 'w') as fs, \
             h5py.File(hits_path,   'w') as fi, \
             h5py.File(step_path,   'w') as fg, \
             h5py.File(labl_path,   'w') as fl:
            write_sensor_config_v3(fs, config_meta, batch_src_idx, sensor_positions_np)
            write_hits_config_v3(fi, config_meta, batch_src_idx, sensor_positions_np)
            write_step_config_v3(fg, config_meta, batch_src_idx)
            write_labl_config_v3(fl, config_meta, batch_src_idx)
            for seq_idx, ev in enumerate(batch_data):
                save_sensor_event_v3(fs, ev, seq_idx)
                save_hits_event_v3(fi, ev, seq_idx)
                save_step_event_v3(fg, ev, seq_idx)
                save_labl_event_v3(fl, ev, seq_idx)

        saved_files.extend([str(sensor_path), str(hits_path), str(step_path), str(labl_path)])
        print(f"Batch {batch_idx+1} save time: {_time.time() - _t_save:.3f}s\n")

    print(f"\nSuccessfully wrote {num_batches} batches "
          f"({len(saved_files)} files total) to {output_dir}/"
          f"{{sensor,hits,step,labl}}/")

    if event_times:
        print(f"\nAverage pile-up event time: "
              f"{sum(event_times)/len(event_times):.3f}s")
    return saved_files


def _merge_pileup_streams(streams, *, n_sensors, apply_smearing,
                          smear_key, detector_bounds):
    """Merge per-vertex streams into a single event_dict.

    Per-interaction metadata (t0, vertex_xyz, source_type) is broadcast
    to one row per primary in the merged event. Primaries are identified
    after the merge by ``derive_track_ancestor_and_interaction`` (parent-
    chain walk to parent_id==0); each primary's vertex is looked up via
    the track_id range it falls into — streams are concatenated in
    declared order with monotonically increasing track IDs, so a
    primary's range uniquely identifies its source stream.

    ``smear_key`` is a jax key (not a concrete seed).
    """
    # Concatenate particles, meaningful_tracks, segments (all post-remap).
    all_particles = []
    all_tracks = {}
    all_segs = {
        'start_x': [], 'start_y': [], 'start_z': [],
        'end_x':   [], 'end_y':   [], 'end_z':   [],
        'dir_x':   [], 'dir_y':   [], 'dir_z':   [],
        'edep': [], 'time': [], 'beta_start': [], 'n_cherenkov': [],
    }

    # One interaction per stream. `interaction_metadata[i]` comes straight
    # from `build_interaction_metadata(...)` at stream-processing time —
    # including the primary_track_ids / pdgs / energies lists — so no
    # derivation is needed here. `primary_to_interaction[tid] = i` maps
    # each primary track_id (already remapped across streams so they're
    # globally unique) to the interaction row it belongs to.
    interaction_metadata = [s['interaction_meta'] for s in streams]
    primary_to_interaction = {}
    for i, meta in enumerate(interaction_metadata):
        for tid in meta['primary_track_ids']:
            primary_to_interaction[int(tid)] = i

    # Per-vertex sparse hits.h5 + step/sensor_hits. Each stream's index
    # columns are vertex-local; we shift them into the merged event's row
    # spaces (particle_idx by cumulative particle count, segment_idx by
    # cumulative segment count). Streams' particles and segments are
    # disjoint by construction (track ids were offset upstream).
    hits_sparse_shifted = []
    seg_hits_shifted = []
    pe_sensor_streams = []
    t_sensor_streams = []
    t_reco_sensor_streams = []
    particle_offset = 0
    seg_offset = 0
    for s in streams:
        all_particles.extend(s['particles'])
        all_tracks.update(s['meaningful_tracks'])
        segs = s['segments']
        n_seg_v = int(segs.get('n_segments', 0)) if segs else 0
        if n_seg_v > 0:
            for k in all_segs:
                all_segs[k].append(np.asarray(segs[k]))
        pe_sensor_streams.append(s['pe_per_sensor'])
        t_sensor_streams.append(s['t_per_sensor'])
        t_reco_sensor_streams.append(s['t_reco_per_sensor'])

        hs = s.get('hits_sparse')
        if hs is not None and hs['PE'].size > 0:
            entry = {
                'particle_idx':     hs['particle_idx'] + np.int32(particle_offset),
                'sensor_idx':       hs['sensor_idx'],
                'PE':               hs['PE'],
                'T':                hs['T'],
                'emission_process': hs['emission_process'],
            }
            if 'T_reco' in hs:
                entry['T_reco'] = hs['T_reco']
            hits_sparse_shifted.append(entry)

        sh = s.get('seg_hits')
        if sh is not None and sh['PE'].size > 0:
            shifted = {
                'segment_idx':      sh['segment_idx'] + np.int32(seg_offset),
                'sensor_idx':       sh['sensor_idx'],
                'PE':               sh['PE'],
                'T':                sh['T'],
                'emission_process': sh['emission_process'],
            }
            if 'T_reco' in sh:
                shifted['T_reco'] = sh['T_reco']
            seg_hits_shifted.append(shifted)

        particle_offset += len(s['particles'])
        seg_offset += n_seg_v

    n_particles_total = len(all_particles)

    # Per-sensor PE_true / T_true: sum / sentinel-aware min across the
    # per-vertex per-sensor arrays (each already process-combined upstream
    # via combine_t_per_sensor_across_processes).
    if pe_sensor_streams:
        PE_true = np.sum(pe_sensor_streams, axis=0).astype(np.float32)
        T_true = combine_t_per_sensor_across_processes(t_sensor_streams)
        T_reco = combine_t_per_sensor_across_processes(t_reco_sensor_streams)
    else:
        PE_true = np.zeros(n_sensors, dtype=np.float32)
        T_true  = np.zeros(n_sensors, dtype=np.float32)
        T_reco  = np.zeros(n_sensors, dtype=np.float32)

    # Charge smearing is still host-side (SK-like Poisson model);
    # time smearing is now kernel-side (TTS), so only PE gets smeared here.
    if apply_smearing and PE_true.sum() > 0:
        from lucid.utils import smear_charges_SK_like
        smear_pe_key, _unused_t_key = jax.random.split(smear_key)
        PE_reco = np.asarray(
            smear_charges_SK_like(jnp.asarray(PE_true), key=smear_pe_key),
            dtype=np.float32)
    else:
        PE_reco = PE_true.copy()

    # Merge segment arrays
    if all_segs['time']:
        seg_merged = {k: np.concatenate(v) for k, v in all_segs.items()}
        seg_merged['n_segments'] = int(len(seg_merged['time']))
    else:
        seg_merged = {'n_segments': 0}

    # Concat the per-vertex sparse hits rows (already particle_idx-shifted
    # into the merged per_particle row space).
    if hits_sparse_shifted:
        has_t_reco_any = all('T_reco' in d for d in hits_sparse_shifted)
        merged_hits_sparse = {
            'particle_idx':     np.concatenate([d['particle_idx']     for d in hits_sparse_shifted]).astype(np.int32),
            'sensor_idx':       np.concatenate([d['sensor_idx']       for d in hits_sparse_shifted]).astype(np.uint16),
            'PE':               np.concatenate([d['PE']               for d in hits_sparse_shifted]).astype(np.float32),
            'T':                np.concatenate([d['T']                for d in hits_sparse_shifted]).astype(np.float32),
            'emission_process': np.concatenate([d['emission_process'] for d in hits_sparse_shifted]).astype(np.int8),
        }
        if has_t_reco_any:
            merged_hits_sparse['T_reco'] = np.concatenate(
                [d['T_reco'] for d in hits_sparse_shifted]).astype(np.float32)
    else:
        merged_hits_sparse = {
            'particle_idx':     np.empty(0, dtype=np.int32),
            'sensor_idx':       np.empty(0, dtype=np.uint16),
            'PE':               np.empty(0, dtype=np.float32),
            'T':                np.empty(0, dtype=np.float32),
            'T_reco':           np.empty(0, dtype=np.float32),
            'emission_process': np.empty(0, dtype=np.int8),
        }

    merged = {
        'n_particles': int(n_particles_total),
        'particles': all_particles,
        'track_info_dict': {},  # unused by writers; merged into meaningful_tracks
        'meaningful_tracks': all_tracks,
        'segments': seg_merged,
        # v5 per-interaction routing (one entry per vertex stream; the
        # writer consumes these to populate the per_interaction/ subgroup).
        'interaction_metadata':   interaction_metadata,
        'primary_to_interaction': primary_to_interaction,
        'hits_sparse': merged_hits_sparse,
        'PE_reco': PE_reco,
        'T_reco':  T_reco,
    }

    # Concat the per-vertex sparse step triplets (already segment_idx-
    # shifted into the merged segment table's row space).
    if seg_hits_shifted:
        merged_seg_hits = {
            'segment_idx':      np.concatenate([d['segment_idx']      for d in seg_hits_shifted]).astype(np.int32),
            'sensor_idx':       np.concatenate([d['sensor_idx']       for d in seg_hits_shifted]).astype(np.uint16),
            'PE':               np.concatenate([d['PE']               for d in seg_hits_shifted]).astype(np.float32),
            'T':                np.concatenate([d['T']                for d in seg_hits_shifted]).astype(np.float32),
            'emission_process': np.concatenate([d['emission_process'] for d in seg_hits_shifted]).astype(np.int8),
        }
        if all('T_reco' in d for d in seg_hits_shifted):
            merged_seg_hits['T_reco'] = np.concatenate(
                [d['T_reco'] for d in seg_hits_shifted]).astype(np.float32)
        merged['segment_sensor_hits'] = merged_seg_hits

    # Geometric containment (same derivation as the single-vertex path;
    # works across merged streams because the helper uses cumulative
    # n_segments in track-insertion order rather than per-stream
    # segment_offset).
    cont = _compute_contained(merged, detector_bounds)
    merged['contained_per_segment']     = cont['per_segment']
    merged['contained_per_particle']    = cont['per_particle']
    merged['contained_per_interaction'] = cont['per_interaction']
    merged['contained']                 = cont['overall']
    return merged
