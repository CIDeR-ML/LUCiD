"""The three production loops: single-vertex particles, pile-up and supernova.

Each loop chooses which PhotonSim entries make an event, draws the per-event keys, time and
vertex, runs the library pieces (:mod:`lucid.sources.interaction`: place, simulate, digitize;
:class:`lucid.simulation.readout.Readout`: select) and writes one four-file batch
(:func:`lucid.dataset.writer.write_batch`). A job writes one batch; ``lucid-run-job`` calls these.

The ``Event total time``, ``Simulation completed in``, ``[timing] <stage>``, ``Batch N save time``
and ``Successfully wrote`` lines are read back by :mod:`lucid.production.qa.job_log`; keep their
format.
"""
from __future__ import annotations

import time
import uuid

import numpy as np

from lucid.dataset.schema import source_type_code
from lucid.dataset.writer import dataset_meta, write_batch
from lucid.sources.interaction import (
    digitize,
    place_raw_entry,
    recorded_vertex,
    resolve_rays_buckets,
    simulate_interaction,
    warmup_buckets,
)
from lucid.sources.placement import detector_bounds, random_rotation, sample_vertex
from lucid.sources.root_reader import read_event_raw
from lucid.sources.seed_utils import derive_event_keys
from lucid.utils import code_version

__all__ = ["run_particles", "run_pileup", "run_supernova"]

# t0 draw half-window (ns), per interaction: t0 ~ Uniform(-T0_HALF_WINDOW_NS, +T0_HALF_WINDOW_NS).
# Wide enough to randomize the absolute event time (models cannot assume t=0 is the start) and
# to cover a +-250 ns pile-up window.
T0_HALF_WINDOW_NS = 250.0


def _bounds(event_simulator, apply_translation):
    """The detector bounds; a detector without them cannot be translated into."""
    bounds = detector_bounds(event_simulator.det_geom)
    if apply_translation and bounds is None:
        raise ValueError(
            f"apply_translation=True requires a detector with canonical bounds; got "
            f"detector_type={event_simulator.det_geom.detector_type!r} (no bounds defined).")
    return bounds


def _t0(keys):
    return float(np.random.default_rng(seed=keys['t0_seed']).uniform(
        -T0_HALF_WINDOW_NS, T0_HALF_WINDOW_NS))


def _code_version():
    """``dataset_meta``'s ``git_commit`` and ``git_dirty``."""
    commit, dirty = code_version()
    return {'git_commit': commit, 'git_dirty': dirty}


def _sensor_positions(event_simulator):
    return np.asarray(event_simulator.det_geom.sensor_points, dtype=np.float32)


def _print_readout(readout):
    model = readout.digitizer
    print(f"Digitizer model: {model['model']} (dark_rate_khz={model.get('dark_rate_khz', 0.0)})")
    if readout.min_physics_hits is not None:
        print(f"Selection: min_physics_hits >= {readout.min_physics_hits} "
              f"(low-E truth cut; sub-threshold events dropped, dark kept + labelled)")
    elif readout.trigger is not None:
        t = readout.trigger
        print(f"Trigger: W={t.window_ns}ns N_thr={t.n_thr} "
              f"pad={t.pad_before_ns}/{t.pad_after_ns}ns (non-triggering events dropped)")


def _write(output_dir, file_index, events, meta, sensor_positions):
    t = time.time()
    paths = write_batch(output_dir, file_index, events, meta, sensor_positions)
    print(f"Batch 1 save time: {time.time() - t:.3f}s")
    print(f"Successfully wrote 1 batches ({len(paths)} files total) to {output_dir}/"
          f"{{sensor,hits,step,labl}}/")
    return paths


def run_particles(event_simulator, root_file, *, output_dir, file_index, n_events, master_seed,
                  job_id, readout, dataset_name, apply_smearing, apply_translation,
                  apply_rotation=False, primary_source='particles', pad_size_buckets=None):
    """One event per PhotonSim entry, entries ``0 .. n_events-1``, written as one batch.

    Parameters
    ----------
    event_simulator : Simulator
        ``setup_event_simulator(..., source='photons', default_detector_params=True,
        hits='segment_truth')``.
    root_file : path
        PhotonSim ROOT file.
    output_dir : path
        Dataset root; the batch is ``<output_dir>/<sub>/wc_<sub>_<file_index:04d>.h5``.
    file_index : int
        Batch index.
    n_events : int
        Entries to read.
    master_seed, job_id : int
        Seed hierarchy: event ``i`` draws from ``derive_event_keys(master_seed, job_id, i)``, its
        digitizer from ``default_rng([master_seed, job_id, i])``. ``master_seed`` is resolved
        (:func:`lucid.production.dataprod.seeds.resolve_master_seed`, below 2**31 - 1).
    readout : Readout
        Digitizer, trigger and selection.
    dataset_name : str
        Provenance.
    apply_smearing : bool
        Apply the digitizer's charge and time resolution.
    apply_translation : bool
        Place each event at a fiducial vertex (r, |z| <= 0.9 of the detector); otherwise it stays
        at the vertex PhotonSim recorded.
    apply_rotation : bool
        Rotate each event by a uniform random rotation about its vertex first. PhotonSim already
        throws isotropic directions, so this is for reusing one Geant4 sample at new orientations
        (``--skip-photonsim`` with another ``--lucid-seed``).
    primary_source : str
        'particles' or 'genie': ``per_interaction/source_type``.
    pad_size_buckets : sequence of int, optional
        Kernel ray buckets (default ``DEFAULT_RAYS_BUCKETS``).

    Returns
    -------
    list of str
        The four files.
    """
    source_type = source_type_code(primary_source)
    sensor_positions = _sensor_positions(event_simulator)
    n_sensors = int(sensor_positions.shape[0])
    rays_buckets = resolve_rays_buckets(pad_size_buckets)
    bounds = _bounds(event_simulator, apply_translation)
    print(f"Generating {n_events} events (seed {master_seed}, job {job_id}); smearing "
          f"{apply_smearing}, translation {apply_translation}, rotation {apply_rotation}; "
          f"rays buckets {list(rays_buckets)}")
    _print_readout(readout)
    warmup_buckets(event_simulator, rays_buckets)

    events = []
    for event_idx in range(n_events):
        t_event = time.time()
        print(f"\n  Event {event_idx+1}/{n_events} (index {event_idx}):", flush=True)
        keys = derive_event_keys(master_seed, job_id, event_idx, interaction_idx=0)
        t0 = _t0(keys)

        t = time.perf_counter()
        raw = read_event_raw(root_file, event_idx)
        print(f"    [timing] root_read {time.perf_counter() - t:.3f}s", flush=True)
        t = time.perf_counter()
        primary_vertex = recorded_vertex(raw)
        vertex = (sample_vertex(bounds, np.random.default_rng(seed=keys['vertex_seed']))
                  if apply_translation else primary_vertex)
        rotation = (random_rotation(np.random.default_rng(seed=keys['rot_seed']))
                    if apply_rotation else np.eye(3))
        place_raw_entry(raw, primary_vertex=primary_vertex, rotation=rotation, vertex=vertex)
        stream, _ = simulate_interaction(
            event_simulator, raw, keys, t0_ns=t0, vertex=vertex, source_type=source_type,
            n_sensors=n_sensors, rays_buckets=rays_buckets)
        print(f"    Simulation completed in {time.perf_counter() - t:.2f}s", flush=True)
        print(f"    [timing] simulate {time.perf_counter() - t:.3f}s", flush=True)

        t = time.perf_counter()
        # job_id folded in, like the kernel keys, so reusing a master_seed across jobs still
        # gives independent dark / SPE / jitter streams.
        event = digitize([stream], n_sensors=n_sensors, readout=readout,
                         rng=np.random.default_rng([int(master_seed), int(job_id), int(event_idx)]),
                         detector_bounds=bounds, apply_resolution=apply_smearing, local_frame=True)
        selected = readout.select(event['sensor_digits'], event['hits_sparse'],
                                  event['segment_sensor_hits'])
        if selected is None:
            print(f"    event {event_idx}: not selected — dropped", flush=True)
            continue
        sensor_digits, hits_sparse, seg_hits, per_window = selected
        event.update(sensor_digits=sensor_digits, hits_sparse=hits_sparse, per_window=per_window,
                     source_event_idx=int(event_idx))
        # an event with no physics hit left carries no step/sensor_hits group
        if seg_hits is not None and seg_hits['PE'].size > 0:
            event['segment_sensor_hits'] = seg_hits
        else:
            del event['segment_sensor_hits']
        print(f"    [timing] digitize {time.perf_counter() - t:.3f}s", flush=True)
        events.append(event)
        print(f"    Event total time: {time.time() - t_event:.2f}s", flush=True)

    meta = dataset_meta(
        event_simulator, readout, dataset_name=dataset_name, run_id=str(uuid.uuid4()),
        **_code_version(), source_files=[root_file], master_seed=master_seed,
        n_events_requested=n_events, apply_smearing=apply_smearing, detector_bounds=bounds)
    return _write(output_dir, file_index, events, meta, sensor_positions)


def run_pileup(event_simulator, root_files, primary_sources, *, output_dir, file_index, n_events,
               master_seed, job_id, readout, dataset_name, apply_smearing, apply_translation):
    """Pile-up: event ``i`` merges entry ``i`` of every vertex's PhotonSim file.

    Each vertex gets its own t0, fiducial vertex and keys (``interaction_idx`` = vertex index);
    track ids are offset so the streams do not collide; the streams are digitized together
    (:func:`digitize`, pooled), so overlapping light shares digits. ``n_events`` is capped at the
    shortest file. ``primary_sources`` ('particles' or 'genie') gives each vertex's
    ``per_interaction/source_type``. The other parameters are those of :func:`run_particles`.
    """
    import uproot

    if len(root_files) != len(primary_sources):
        raise ValueError(
            f"root_files and primary_sources length mismatch: "
            f"{len(root_files)} vs {len(primary_sources)}")
    n_vertices = len(root_files)
    if n_vertices < 2:
        raise ValueError("Pile-up requires at least 2 vertices.")
    sensor_positions = _sensor_positions(event_simulator)
    n_sensors = int(sensor_positions.shape[0])
    rays_buckets = resolve_rays_buckets()
    bounds = _bounds(event_simulator, apply_translation)

    per_file_counts = []
    for p in root_files:
        with uproot.open(p) as f:
            per_file_counts.append(int(f['OpticalPhotons'].num_entries))
    common = min(per_file_counts)
    if n_events > common:
        print(f"WARNING: requested n_events={n_events} exceeds min per-vertex "
              f"entries {per_file_counts}; capping to {common}.")
        n_events = common
    print(f"Pile-up: seed {master_seed}, job {job_id}, {n_vertices} vertices, "
          f"{n_events} events (per-vertex entries {per_file_counts})")
    _print_readout(readout)
    warmup_buckets(event_simulator, rays_buckets)

    events = []
    for event_idx in range(n_events):
        t_event = time.time()
        print(f"\n  Event {event_idx+1}/{n_events}:", flush=True)
        streams = []
        track_offset = 0
        for vidx in range(n_vertices):
            keys = derive_event_keys(master_seed, job_id, event_idx, interaction_idx=vidx)
            t0 = _t0(keys)
            t = time.time()
            raw = read_event_raw(str(root_files[vidx]), event_idx)
            primary_vertex = recorded_vertex(raw)
            vertex = (sample_vertex(bounds, np.random.default_rng(seed=keys['vertex_seed']))
                      if apply_translation else primary_vertex)
            print(f"    vertex {vidx}: t0={t0:+.2f} ns, "
                  f"xyz=({vertex[0]:.3f}, {vertex[1]:.3f}, {vertex[2]:.3f}) m", flush=True)
            place_raw_entry(raw, primary_vertex=primary_vertex, rotation=np.eye(3), vertex=vertex)
            stream, max_track_id = simulate_interaction(
                event_simulator, raw, keys, t0_ns=t0, vertex=vertex,
                source_type=source_type_code(primary_sources[vidx]), n_sensors=n_sensors,
                rays_buckets=rays_buckets, track_offset=track_offset)
            print(f"      [timing] simulate {time.time() - t:.3f}s", flush=True)
            streams.append(stream)
            track_offset = max_track_id + 1

        t = time.time()
        event = digitize(
            streams, n_sensors=n_sensors, readout=readout,
            rng=np.random.default_rng([int(master_seed), int(job_id), int(event_idx)]),
            detector_bounds=bounds, apply_resolution=apply_smearing)
        print(f"    [timing] merge {time.time() - t:.3f}s", flush=True)
        event['source_event_idx'] = int(event_idx)
        selected = readout.select(event['sensor_digits'], event['hits_sparse'],
                                  event.get('segment_sensor_hits'))
        if selected is None:
            print(f"    event {event_idx}: no trigger — event dropped", flush=True)
            continue
        event['sensor_digits'], event['hits_sparse'], seg_hits, per_window = selected
        if per_window is not None:
            event['per_window'] = per_window
        if seg_hits is not None:
            event['segment_sensor_hits'] = seg_hits
        events.append(event)
        print(f"    Event total time: {time.time() - t_event:.2f}s", flush=True)

    meta = dataset_meta(
        event_simulator, readout, dataset_name=dataset_name, run_id=str(uuid.uuid4()),
        **_code_version(), source_files=root_files, master_seed=master_seed,
        n_events_requested=n_events, apply_smearing=apply_smearing, detector_bounds=bounds)
    return _write(output_dir, file_index, events, meta, sensor_positions)


def _group_interactions_by_gap(t0_sorted, coupling_ns):
    """Cut time-sorted interactions into causally-independent chunks.

    A gap ``> coupling_ns`` between consecutive interactions means the two sides can share
    neither charge (integration window) nor a trigger window, so a boundary there introduces no
    discretization. Returns a list of index lists into ``t0_sorted``.
    """
    if len(t0_sorted) == 0:
        return []
    chunks = []
    cur = [0]
    for i in range(1, len(t0_sorted)):
        if t0_sorted[i] - t0_sorted[i - 1] > coupling_ns:
            chunks.append(cur)
            cur = [i]
        else:
            cur.append(i)
    chunks.append(cur)
    return chunks


def _concat_triggered_chunks(chunks):
    """Concatenate already-digitized and selected chunk events into one.

    Chunks are time-ordered and each one's digits are canonically ordered, so concatenation
    preserves the global ``(window, sensor, T)`` order. Digit-referencing indices are shifted by
    cumulative counts: ``digit_idx`` by digits, ``particle_idx`` by particles, ``segment_idx`` by
    segments, and per_window ``digit_offsets`` by digits.
    """
    if not chunks:
        return None
    if len(chunks) == 1:
        return chunks[0]

    HITCOLS = ('particle_idx', 'digit_idx', 'sensor_idx', 'PE', 'T', 'T_reco', 'emission_process')
    SEGCOLS = ('segment_idx', 'digit_idx', 'sensor_idx', 'PE', 'T', 'T_reco', 'emission_process')
    SEGKEYS = ('start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z',
               'dir_x', 'dir_y', 'dir_z', 'edep', 'time', 'beta_start', 'n_cherenkov')
    has_seg_hits = any('segment_sensor_hits' in c for c in chunks)
    has_windows = any(c.get('per_window') is not None for c in chunks)

    sd = {'sensor_idx': [], 'PE': [], 'T': []}
    hits = {k: [] for k in HITCOLS}
    segh = {k: [] for k in SEGCOLS}
    pw = {'window_start': [], 'window_end': [], 'digit_offsets': []}
    all_particles, all_tracks, all_meta = [], {}, []
    all_segs = {k: [] for k in SEGKEYS}
    cont_seg, cont_part, cont_int = [], [], []
    dig_off = part_off = seg_off = 0

    for c in chunks:
        n_dig = int(np.asarray(c['sensor_digits']['sensor_idx']).shape[0])
        n_part = len(c['particles'])
        n_seg = int(c['segments'].get('n_segments', 0))

        for k in sd:
            sd[k].append(np.asarray(c['sensor_digits'][k]))
        h = c['hits_sparse']
        for k in HITCOLS:
            v = np.asarray(h[k])
            if k == 'digit_idx':
                v = v + dig_off
            elif k == 'particle_idx':
                v = np.where(v >= 0, v + part_off, v)
            hits[k].append(v)
        if has_seg_hits:
            sh = c.get('segment_sensor_hits')
            if sh is not None:
                for k in SEGCOLS:
                    v = np.asarray(sh[k])
                    if k == 'digit_idx':
                        v = v + dig_off
                    elif k == 'segment_idx':
                        v = np.where(v >= 0, v + seg_off, v)
                    segh[k].append(v)
        pwc = c.get('per_window')
        if pwc is not None:
            pw['window_start'].append(np.asarray(pwc['window_start']))
            pw['window_end'].append(np.asarray(pwc['window_end']))
            pw['digit_offsets'].append(np.asarray(pwc['digit_offsets'])[1:] + dig_off)

        all_particles.extend(c['particles'])
        all_tracks.update(c['meaningful_tracks'])
        if n_seg > 0:
            for k in SEGKEYS:
                all_segs[k].append(np.asarray(c['segments'][k]))
        all_meta.extend(c['interaction_metadata'])
        cont_seg.append(np.asarray(c['contained_per_segment']))
        cont_part.append(np.asarray(c['contained_per_particle']))
        cont_int.append(np.asarray(c['contained_per_interaction']))

        dig_off += n_dig
        part_off += n_part
        seg_off += n_seg

    out = {
        'n_particles': len(all_particles),
        'particles': all_particles,
        'meaningful_tracks': all_tracks,
        'sensor_digits': {k: np.concatenate(v) for k, v in sd.items()},
        'hits_sparse': {k: np.concatenate(v) for k, v in hits.items()},
        'interaction_metadata': all_meta,
        'primary_to_interaction': {int(tid): i for i, m in enumerate(all_meta)
                                   for tid in m['primary_track_ids']},
        'contained_per_segment': np.concatenate(cont_seg) if cont_seg else np.array([], bool),
        'contained_per_particle': np.concatenate(cont_part),
        'contained_per_interaction': np.concatenate(cont_int),
    }
    out['contained'] = bool(np.all(out['contained_per_interaction']))
    if all_segs['time']:
        out['segments'] = {k: np.concatenate(v) for k, v in all_segs.items()}
        out['segments']['n_segments'] = int(out['segments']['time'].shape[0])
    else:
        out['segments'] = {'n_segments': 0}
    if has_seg_hits and segh['digit_idx']:
        out['segment_sensor_hits'] = {k: np.concatenate(v) for k, v in segh.items()}
    if has_windows and pw['window_start']:
        out['per_window'] = {
            'window_start': np.concatenate(pw['window_start']),
            'window_end': np.concatenate(pw['window_end']),
            'digit_offsets': np.concatenate([np.array([0], np.int64)] + pw['digit_offsets']),
        }
    return out


def run_supernova(event_simulator, burst_root_file, interaction_times_ms, *, output_dir,
                  file_index, master_seed, job_id, readout, dataset_name, apply_smearing,
                  apply_translation, sn_ordering, sn_transformation, interaction_channels=None,
                  sn_direction=None, source_event_idx=0):
    """ONE all-at-once supernova event from a burst PhotonSim file (one entry per interaction).

    Interactions are placed at their true times, ``t0_i = global_t0 + interaction_times_ms[i] *
    1e6`` (ns), with one +-250 ns ``global_t0`` for the burst. They are cut into
    causally-independent chunks at gaps wider than the digitizer's integration window; each chunk
    is pool-digitized (dark over just its span) and selected, so dark stays bounded and no
    boundary splits a charge-sharing coincidence. Surviving
    chunks are concatenated into one event. When none survives the batch is still written, with 0
    events and flagged complete, so the job reads as done and is not resubmitted forever (its seeds
    are deterministic, so a rerun keeps nothing either).

    Vertices fill nearly the whole inner volume (r, |z| <= 0.995) and are redrawn out of the PMTs
    (within the detector's ``S_radius`` of a sensor).
    ``readout`` must have a windowed digitizer (ski/hk) and ``min_physics_hits`` (the trigger
    cannot tag low-energy supernova interactions). ``interaction_channels`` (sntools codes) and
    ``sn_direction`` (unit vector, default +z) are recorded as truth. ``sn_ordering`` (the
    config's label, e.g. ``NMO``) and ``sn_transformation`` (the sntools ``--transformation`` it
    ran under) are recorded in ``config/``. The other parameters are those of
    :func:`run_particles`.
    """
    import uproot

    model = readout.digitizer
    if not model.get('integration_window_ns'):
        raise ValueError(
            "Supernova needs a windowed digitizer (ski/hk); got "
            f"{model.get('model')!r}. Set digitizer.model to ski or hk.")
    if readout.min_physics_hits is None:
        raise ValueError(
            "Supernova needs the truth-based selection (selection.mode = "
            "min_physics_hits), not the readout trigger.")
    sensor_positions = _sensor_positions(event_simulator)
    n_sensors = int(sensor_positions.shape[0])
    pmt_radius = float(event_simulator.det_geom.detector.S_radius)

    # Open the burst ROOT once and reuse the handle for every entry read: re-opening per
    # interaction re-parses the tree metadata each time (~10x read cost over a full burst).
    burst_file = uproot.open(burst_root_file)
    M = int(burst_file['OpticalPhotons'].num_entries)
    times_ms = np.asarray(interaction_times_ms, dtype=np.float64)
    if times_ms.shape[0] != M:
        raise ValueError(
            f"interaction_times_ms ({times_ms.shape[0]}) != burst entries ({M})")

    # Burst-level key for the single global t0, with a high sentinel index so it cannot collide
    # with the per-interaction keys (0..M-1); fold_in needs a non-negative uint32.
    burst_keys = derive_event_keys(master_seed, job_id, source_event_idx,
                                   interaction_idx=0x7FFFFFFF)
    global_t0 = _t0(burst_keys)
    t0_abs_ns = global_t0 + times_ms * 1.0e6           # ms -> ns, float64
    order = np.argsort(t0_abs_ns, kind='stable')
    t0_sorted = t0_abs_ns[order]

    # Interactions further apart than the integration window share no charge. No trigger window
    # couples them: the selection is min_physics_hits, which Readout refuses with a trigger.
    coupling_ns = float(model['integration_window_ns'])
    chunks = _group_interactions_by_gap(t0_sorted, coupling_ns)
    span_ms = (t0_sorted[-1] - t0_sorted[0]) / 1e6 if M else 0.0
    print(f"Supernova: {M} interactions, global_t0={global_t0:+.1f}ns, "
          f"span={span_ms:.2f}ms, coupling={coupling_ns:.0f}ns -> "
          f"{len(chunks)} causally-independent chunks", flush=True)

    rays_buckets = resolve_rays_buckets()
    warmup_buckets(event_simulator, rays_buckets)
    bounds = _bounds(event_simulator, apply_translation)

    track_offset = 0
    surviving = []
    for cidx, chunk in enumerate(chunks):
        streams = []
        for local_i in chunk:
            entry = int(order[local_i])
            t0 = float(t0_sorted[local_i])
            keys = derive_event_keys(master_seed, job_id, source_event_idx, interaction_idx=entry)
            raw = read_event_raw(str(burst_root_file), entry, opened_file=burst_file)
            primary_vertex = recorded_vertex(raw)
            vertex = (sample_vertex(bounds, np.random.default_rng(seed=keys['vertex_seed']),
                                    r_frac=0.995, z_frac=0.995, sensor_positions=sensor_positions,
                                    pmt_radius=pmt_radius)
                      if apply_translation else primary_vertex)
            channel = (int(interaction_channels[entry])
                       if interaction_channels is not None
                       and entry < len(interaction_channels) else None)
            place_raw_entry(raw, primary_vertex=primary_vertex, rotation=np.eye(3), vertex=vertex)
            stream, max_track_id = simulate_interaction(
                event_simulator, raw, keys, t0_ns=t0, vertex=vertex,
                source_type=source_type_code('supernova'), n_sensors=n_sensors,
                rays_buckets=rays_buckets, track_offset=track_offset,
                interaction_channel=channel)
            streams.append(stream)
            track_offset = max_track_id + 1

        chunk_event = digitize(
            streams, n_sensors=n_sensors, readout=readout,
            rng=np.random.default_rng(
                [int(master_seed), int(job_id), int(source_event_idx), int(cidx)]),
            detector_bounds=bounds, apply_resolution=apply_smearing)
        selected = readout.select(chunk_event['sensor_digits'], chunk_event['hits_sparse'],
                                  chunk_event.get('segment_sensor_hits'))
        if selected is None:
            continue
        chunk_event['sensor_digits'], chunk_event['hits_sparse'], seg_hits, \
            chunk_event['per_window'] = selected
        if seg_hits is not None:
            chunk_event['segment_sensor_hits'] = seg_hits
        elif 'segment_sensor_hits' in chunk_event:
            del chunk_event['segment_sensor_hits']
        surviving.append(chunk_event)

    burst_file.close()
    n_kept = len(surviving)
    print(f"Supernova: {n_kept}/{len(chunks)} interactions kept "
          f"(>= {readout.min_physics_hits} physics hits)", flush=True)
    meta = dataset_meta(
        event_simulator, readout, dataset_name=dataset_name, run_id=str(uuid.uuid4()),
        **_code_version(), source_files=[burst_root_file], master_seed=master_seed,
        n_events_requested=1, apply_smearing=apply_smearing, detector_bounds=bounds,
        sn_direction=sn_direction or [0.0, 0.0, 1.0], sn_ordering=sn_ordering,
        sn_transformation=sn_transformation)
    event = _concat_triggered_chunks(surviving)
    if event is None:
        paths = write_batch(output_dir, file_index, [], meta, sensor_positions)
        print("Supernova: no interaction kept — wrote a complete 0-event batch", flush=True)
        return paths
    event['source_event_idx'] = int(source_event_idx)
    paths = write_batch(output_dir, file_index, [event], meta, sensor_positions)
    n_dig = int(np.asarray(event['sensor_digits']['sensor_idx']).shape[0])
    n_win = (int(np.asarray(event['per_window']['window_start']).shape[0])
             if 'per_window' in event else 0)
    print(f"Supernova: wrote 1 event — {n_dig} digits in {n_win} windows "
          f"from {n_kept} kept interactions", flush=True)
    return paths
