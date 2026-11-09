"""The dataset writer: one batch of events into the four files.

:func:`write_batch` writes a batch; :func:`dataset_meta` builds its ``config/`` attributes. The
``write_*_config`` / ``save_*_event`` functions write one file's config group and one event
group into an open file.
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np

from lucid.dataset.labels import (
    derive_particle_idx_per_track,
    derive_particle_interaction_idx,
    derive_track_ancestor_and_interaction,
)
from lucid.dataset.layout import batch_paths
from lucid.dataset.schema import FORMAT_VERSION, GZIP_OPTS, SUBDIRS, TIME_DTYPE

__all__ = [
    "dataset_meta",
    "write_batch",
    "write_sensor_config",
    "write_hits_config",
    "write_step_config",
    "write_labl_config",
    "mark_config_complete",
    "save_sensor_event",
    "save_hits_event",
    "save_step_event",
    "save_labl_event",
]


def _smearing_functions(digitizer, apply_smearing):
    """``(charge, time)`` labels of the resolution the digitizer model applies: charge ``spe``
    (ski/hk) or the legacy ``charge_res`` (basic: ``SK_like``), time its ``time_model``."""
    if not apply_smearing:
        return 'none', 'none'
    if digitizer.get('charge_model', 'legacy') == 'spe':
        charge = 'spe'
    else:
        res = digitizer.get('charge_res', 'sk_like')
        charge = 'SK_like' if res == 'sk_like' else str(res)
    return charge, str(digitizer.get('time_model', 'none'))


def dataset_meta(event_simulator, readout, *, dataset_name, run_id, git_commit, source_files,
                 master_seed, n_events_requested, apply_smearing, detector_bounds,
                 git_dirty=None, sn_direction=None, sn_ordering=None, sn_transformation=None):
    """The ``config/`` attributes of one production batch, before :func:`write_batch` adds the
    event and sensor counts and the file index.

    Parameters
    ----------
    event_simulator : Simulator
        Provides the detector type, the medium and the reflection model.
    readout : Readout
        Digitizer, trigger and selection attributes (:meth:`Readout.meta`).
    dataset_name, run_id, git_commit : str
        Provenance.
    git_dirty : bool or None
        Whether the code had uncommitted changes (:func:`lucid.utils.code_version`); ``None``
        (not known) writes no ``git_dirty``.
    source_files : list of path
        The PhotonSim ROOT file(s) the batch was read from.
    master_seed : int
        The LUCiD-side seed.
    n_events_requested : int
        Events attempted before selection (``n_events`` counts the kept ones).
    apply_smearing : bool
        Whether the digitizer's charge and time resolution were applied.
    detector_bounds : dict or None
        ``lucid.sources.placement.detector_bounds``; adds the step file's geometry hints.
    sn_direction : (3,) sequence, optional
        Supernova true direction (unit vector), for pointing studies.
    sn_ordering, sn_transformation : str, optional
        Supernova mass ordering (the config's label) and the sntools transformation it ran under.
    """
    det_geom = event_simulator.det_geom
    charge_fn, time_fn = _smearing_functions(readout.digitizer, apply_smearing)
    meta = {
        'n_events_requested': int(n_events_requested),
        'git_commit': git_commit,
        'git_dirty': git_dirty,
        'run_id': run_id,
        'dataset_name': dataset_name,
        'source_file': ','.join(os.path.abspath(str(p)) for p in source_files),
        'lucid_master_seed': int(master_seed),
        'photonsim_seed': -1,
        'detector_type': str(det_geom.detector_type),
        'material': str(det_geom.medium.material),
        'reflection_model': str(event_simulator.reflection_model),
        'smearing_applied': bool(apply_smearing),
        'smearing_charge_function': charge_fn,
        'smearing_time_function': time_fn,
        **readout.meta(),
        'label_names': ['category'],
    }
    if sn_direction is not None:
        meta['sn_direction'] = [float(x) for x in sn_direction]
    if sn_ordering is not None:
        meta['sn_ordering'] = str(sn_ordering)
        meta['sn_transformation'] = str(sn_transformation)
    if detector_bounds is not None:
        meta['detector_shape'] = detector_bounds['type']
        if detector_bounds['type'] == 'cylinder':
            meta['detector_radius'] = float(detector_bounds['radius'])
            meta['detector_half_height'] = float(detector_bounds['height']) / 2.0
            meta['detector_axis'] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        elif detector_bounds['type'] == 'sphere':
            meta['detector_radius'] = float(detector_bounds['radius'])
        elif detector_bounds['type'] == 'box':
            l, w, h = detector_bounds['length'], detector_bounds['width'], detector_bounds['height']
            meta['detector_bbox'] = np.array([-l/2, l/2, -w/2, w/2, -h/2, h/2], dtype=np.float32)
    return meta


def write_batch(out_root, file_index, events, meta, sensor_positions):
    """Write one batch: ``<out_root>/<sub>/wc_<sub>_<file_index:04d>.h5`` for the four files.

    ``meta`` is :func:`dataset_meta`'s output; ``n_events``, ``n_sensors`` and ``file_index`` are
    added here. The files are flagged ``config/complete`` once every event is written. Returns
    the four paths, in :data:`SUBDIRS` order.
    """
    out_root = Path(out_root)
    paths = batch_paths(out_root, file_index)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta, n_events=len(events), n_sensors=int(len(sensor_positions)),
                file_index=int(file_index))
    src_idx = np.asarray([ev['source_event_idx'] for ev in events], dtype=np.uint32)
    with h5py.File(paths['sensor'], 'w') as fs, h5py.File(paths['hits'], 'w') as fi, \
            h5py.File(paths['step'], 'w') as fg, h5py.File(paths['labl'], 'w') as fl:
        write_sensor_config(fs, meta, src_idx, sensor_positions)
        write_hits_config(fi, meta, src_idx, sensor_positions)
        write_step_config(fg, meta, src_idx)
        write_labl_config(fl, meta, src_idx)
        for seq_idx, ev in enumerate(events):
            save_sensor_event(fs, ev, seq_idx)
            save_hits_event(fi, ev, seq_idx)
            save_step_event(fg, ev, seq_idx)
            save_labl_event(fl, ev, seq_idx)
        mark_config_complete(fs, fi, fg, fl)
    return [str(paths[sub]) for sub in SUBDIRS]


def _write_common_config_attrs(f, config_meta):
    """Create ``config/`` group with provenance attrs common to all files."""
    cfg = f.require_group('config')
    cfg.attrs['format_version'] = FORMAT_VERSION
    cfg.attrs['n_events'] = int(config_meta['n_events'])
    # Number of events *attempted* (before trigger/selection drops). n_events is
    # the number kept, so (requested - n_events) is the selection inefficiency.
    if 'n_events_requested' in config_meta:
        cfg.attrs['n_events_requested'] = int(config_meta['n_events_requested'])
    # Completion flag: set to True only after every event is written (see
    # mark_config_complete). Absent/False => the job was killed mid-write and the
    # file is incomplete. Health checks gate on this, not on n_events.
    cfg.attrs['complete'] = False
    cfg.attrs['git_commit'] = str(config_meta.get('git_commit', 'unknown'))
    # Absent when not known (an image whose version record predates it).
    if config_meta.get('git_dirty') is not None:
        cfg.attrs['git_dirty'] = bool(config_meta['git_dirty'])
    cfg.attrs['reflection_model'] = str(config_meta.get('reflection_model', 'scalar_mix'))
    cfg.attrs['run_id'] = str(config_meta['run_id'])
    cfg.attrs['dataset_name'] = str(config_meta['dataset_name'])
    cfg.attrs['file_index'] = int(config_meta.get('file_index', 0))
    cfg.attrs['source_file'] = str(config_meta['source_file'])
    cfg.attrs['lucid_master_seed'] = int(config_meta['lucid_master_seed'])
    cfg.attrs['photonsim_seed'] = int(config_meta.get('photonsim_seed', -1))
    # Trigger provenance ("none" when the readout trigger is off).
    cfg.attrs['trigger'] = str(config_meta.get('trigger', 'none'))
    if config_meta.get('trigger', 'none') != 'none':
        cfg.attrs['trigger_window_ns'] = float(config_meta['trigger_window_ns'])
        cfg.attrs['trigger_n_thr'] = int(config_meta['trigger_n_thr'])
        cfg.attrs['trigger_pad_before_ns'] = float(config_meta['trigger_pad_before_ns'])
        cfg.attrs['trigger_pad_after_ns'] = float(config_meta['trigger_pad_after_ns'])
    # Selection provenance: how interactions were kept. "trigger" = the readout
    # trigger above; "min_physics_hits" = the trigger-free truth cut (supernova).
    if 'selection' in config_meta:
        cfg.attrs['selection'] = str(config_meta['selection'])
        cfg.attrs['selection_min_physics_hits'] = int(
            config_meta.get('selection_min_physics_hits', 0))
    # Supernova true direction (unit vector) for pointing studies.
    if 'sn_direction' in config_meta:
        cfg.attrs['sn_direction'] = np.asarray(config_meta['sn_direction'], dtype=np.float64)
    # Supernova mass ordering: the config's label and the sntools transformation.
    if 'sn_ordering' in config_meta:
        cfg.attrs['sn_ordering'] = str(config_meta['sn_ordering'])
        cfg.attrs['sn_transformation'] = str(config_meta['sn_transformation'])
    return cfg


def mark_config_complete(*files):
    """Stamp ``config/complete = True`` on each open HDF5 file.

    Call this ONLY after every event has been written to all files, so a job that
    is killed mid-write leaves the flag ``False``. Health checks treat a missing
    or False flag as an incomplete (unhealthy) file — the number of events is
    allowed to vary (trigger/selection drops), so it is not the health signal.
    """
    for f in files:
        f['config'].attrs['complete'] = True


def write_sensor_config(f, config_meta, source_event_idx, sensor_positions):
    """Write the config/ group of a sensor file."""
    cfg = _write_common_config_attrs(f, config_meta)
    cfg.attrs['n_sensors'] = int(config_meta['n_sensors'])
    cfg.attrs['detector_type'] = str(config_meta['detector_type'])
    cfg.attrs['material'] = str(config_meta['material'])
    cfg.attrs['smearing_applied'] = bool(config_meta['smearing_applied'])
    cfg.attrs['smearing_charge_function'] = str(
        config_meta.get('smearing_charge_function', 'default'))
    cfg.attrs['smearing_time_function'] = str(
        config_meta.get('smearing_time_function', 'default'))
    # digitizer (hit-making) model; default 'basic' for callers that don't set it.
    cfg.attrs['digitizer_model'] = str(config_meta.get('digitizer_model', 'basic'))
    cfg.create_dataset('source_event_idx',
                       data=np.asarray(source_event_idx, dtype=np.uint32),
                       **GZIP_OPTS)
    cfg.create_dataset('sensor_positions',
                       data=np.asarray(sensor_positions, dtype=np.float32),
                       **GZIP_OPTS)


def write_hits_config(f, config_meta, source_event_idx, sensor_positions):
    """Write the config/ group of a hits file."""
    cfg = _write_common_config_attrs(f, config_meta)
    cfg.attrs['n_sensors'] = int(config_meta['n_sensors'])
    cfg.attrs['detector_type'] = str(config_meta['detector_type'])
    cfg.attrs['material'] = str(config_meta['material'])
    cfg.create_dataset('source_event_idx',
                       data=np.asarray(source_event_idx, dtype=np.uint32),
                       **GZIP_OPTS)
    cfg.create_dataset('sensor_positions',
                       data=np.asarray(sensor_positions, dtype=np.float32),
                       **GZIP_OPTS)


def write_step_config(f, config_meta, source_event_idx):
    """Write the config/ group of a step file."""
    cfg = _write_common_config_attrs(f, config_meta)
    cfg.attrs['detector_type'] = str(config_meta['detector_type'])
    cfg.attrs['material'] = str(config_meta['material'])
    if 'detector_shape' in config_meta:
        cfg.attrs['detector_shape'] = str(config_meta['detector_shape'])
    for key in ('detector_bbox', 'detector_axis'):
        if key in config_meta:
            cfg.create_dataset(key,
                               data=np.asarray(config_meta[key], dtype=np.float32))
    for key in ('detector_radius', 'detector_half_height'):
        if key in config_meta:
            cfg.attrs[key] = float(config_meta[key])
    cfg.create_dataset('source_event_idx',
                       data=np.asarray(source_event_idx, dtype=np.uint32),
                       **GZIP_OPTS)


def write_labl_config(f, config_meta, source_event_idx):
    """Write the config/ group of a labl file."""
    cfg = _write_common_config_attrs(f, config_meta)
    label_names = list(config_meta.get('label_names', ['category']))
    cfg.attrs['label_names'] = np.array(label_names, dtype=h5py.string_dtype())
    cfg.create_dataset('source_event_idx',
                       data=np.asarray(source_event_idx, dtype=np.uint32),
                       **GZIP_OPTS)


def _event_group_name(seq_idx):
    return f'event_{int(seq_idx):03d}'


def save_sensor_event(f, event_dict, seq_idx):
    """Write a single event_NNN/ group to an already-open sensor file.

    ``event_dict`` must contain ``source_event_idx`` and ``sensor_digits``, the digitizer's digit
    list ``{sensor_idx, PE, T}`` (a sensor index may repeat in multi-hit models). Times are
    expected in the absolute detector frame; the writer does not shift them.
    """
    grp = f.create_group(_event_group_name(seq_idx))
    grp.attrs['source_event_idx'] = int(event_dict['source_event_idx'])

    sd = event_dict['sensor_digits']
    indices = np.asarray(sd['sensor_idx'], dtype=np.uint16)
    pe_sparse = np.asarray(sd['PE'], dtype=np.float32)
    t_sparse = np.asarray(sd['T'], dtype=TIME_DTYPE)

    grp.attrs['n_hits'] = int(indices.size)
    grp.create_dataset('sensor_idx', data=indices, **GZIP_OPTS)
    grp.create_dataset('PE', data=pe_sparse, **GZIP_OPTS)
    grp.create_dataset('T', data=t_sparse, **GZIP_OPTS)


def save_hits_event(f, event_dict, seq_idx):
    """Write a single event_NNN/ group to an already-open hits file.

    ``event_dict['hits_sparse']`` holds the digitizer's decomposition, one row per (particle,
    sensor, digit, emission process): ``particle_idx``, ``digit_idx``, ``sensor_idx``, ``PE``,
    ``T``, ``emission_process`` and optionally ``T_reco``, written verbatim. Times are expected
    in the absolute detector frame.
    """
    grp = f.create_group(_event_group_name(seq_idx))
    grp.attrs['source_event_idx'] = int(event_dict['source_event_idx'])
    grp.attrs['n_particles'] = int(event_dict['n_particles'])

    sparse = event_dict['hits_sparse']
    particle_idx_arr = np.asarray(sparse['particle_idx'], dtype=np.int32)
    digit_idx_arr    = np.asarray(sparse['digit_idx'],    dtype=np.int32)
    sensor_idx_arr   = np.asarray(sparse['sensor_idx'],   dtype=np.uint16)
    pe_arr           = np.asarray(sparse['PE'],           dtype=np.float32)
    t_arr            = np.asarray(sparse['T'],            dtype=TIME_DTYPE)
    emp_arr          = np.asarray(sparse['emission_process'], dtype=np.int8)
    t_reco_arr = (np.asarray(sparse['T_reco'], dtype=TIME_DTYPE)
                  if 'T_reco' in sparse else None)

    grp.attrs['n_particle_hits'] = int(particle_idx_arr.size)
    grp.create_dataset('particle_idx', data=particle_idx_arr, **GZIP_OPTS)
    grp.create_dataset('digit_idx', data=digit_idx_arr, **GZIP_OPTS)
    grp.create_dataset('sensor_idx', data=sensor_idx_arr, **GZIP_OPTS)
    grp.create_dataset('PE', data=pe_arr, **GZIP_OPTS)
    grp.create_dataset('T', data=t_arr, **GZIP_OPTS)
    if t_reco_arr is not None:
        grp.create_dataset('T_reco', data=t_reco_arr, **GZIP_OPTS)
    grp.create_dataset('emission_process', data=emp_arr, **GZIP_OPTS)


def save_step_event(f, event_dict, seq_idx):
    """Write a single event_NNN/ group to an already-open step file.

    Each segment row gets a local ``track_idx`` FK (0..n_tracks-1). Times are
    shifted by ``t0`` so they live in the detector frame. ``beta_start`` and
    ``n_cherenkov`` are pass-through from PhotonSim.
    """
    grp = f.create_group(_event_group_name(seq_idx))
    grp.attrs['source_event_idx'] = int(event_dict['source_event_idx'])

    mt = event_dict.get('meaningful_tracks', {})
    seg = event_dict.get('segments', {'n_segments': 0})

    n_tracks = int(len(mt))
    n_segments = int(seg.get('n_segments', 0))
    grp.attrs['n_tracks'] = n_tracks
    grp.attrs['n_segments'] = n_segments

    track_idx_per_segment = []
    for track_local_idx, t_info in enumerate(mt.values()):
        track_idx_per_segment.extend(
            [track_local_idx] * int(t_info['n_segments']))
    track_idx_arr = np.asarray(track_idx_per_segment, dtype=np.int32)
    assert track_idx_arr.size == n_segments, (
        f"track_idx length {track_idx_arr.size} != n_segments {n_segments}")

    grp.create_dataset('track_idx', data=track_idx_arr, **GZIP_OPTS)

    # per-segment contained flag computed alongside the higher-level
    # AND-rollups by `compute_containment`. Saves any reader from having
    # to know detector geometry just to ask "did this step stay inside?".
    contained_per_segment = np.asarray(
        event_dict.get('contained_per_segment', np.zeros(n_segments, dtype=bool)),
        dtype=bool)
    if contained_per_segment.shape != (n_segments,):
        contained_per_segment = np.zeros(n_segments, dtype=bool)

    def _empty(dtype): return np.array([], dtype=dtype)
    if n_segments > 0:
        # group_id is supplied by read_particle_data_from_photonsim. Old
        # writes (no Python-side grouping) fall back to a contiguous range
        # so each raw step is its own group — keeps the column non-null.
        group_id_arr = np.asarray(
            seg.get('group_id', np.arange(n_segments, dtype=np.int32)),
            dtype=np.int32,
        )
        fields = {
            'start_x': (seg['start_x'], np.float32),
            'start_y': (seg['start_y'], np.float32),
            'start_z': (seg['start_z'], np.float32),
            'end_x': (seg['end_x'], np.float32),
            'end_y': (seg['end_y'], np.float32),
            'end_z': (seg['end_z'], np.float32),
            'dir_x': (seg['dir_x'], np.float16),
            'dir_y': (seg['dir_y'], np.float16),
            'dir_z': (seg['dir_z'], np.float16),
            'edep': (seg['edep'], np.float32),
            'time': (np.asarray(seg['time'], dtype=TIME_DTYPE), TIME_DTYPE),
            'beta_start': (seg['beta_start'], np.float32),
            'n_cherenkov': (seg['n_cherenkov'], np.int32),
            'group_id': (group_id_arr, np.int32),
            'contained': (contained_per_segment, bool),
        }
        for name, (arr, dtype) in fields.items():
            grp.create_dataset(name,
                               data=np.asarray(arr, dtype=dtype),
                               **GZIP_OPTS)
    else:
        for name, dtype in (('start_x', np.float32), ('start_y', np.float32),
                            ('start_z', np.float32), ('end_x', np.float32),
                            ('end_y', np.float32), ('end_z', np.float32),
                            ('dir_x', np.float16), ('dir_y', np.float16),
                            ('dir_z', np.float16), ('edep', np.float32),
                            ('time', TIME_DTYPE), ('beta_start', np.float32),
                            ('n_cherenkov', np.int32), ('group_id', np.int32),
                            ('contained', bool)):
            grp.create_dataset(name, data=_empty(dtype), **GZIP_OPTS)

    # Optional segment <-> sensor correspondence map. Mirrors hits file's flat
    # parallel-array shape: each row is one (segment, sensor) pair with PE+T.
    # Forward map (segment -> sensors): groupby segment_idx. Reverse map
    # (sensor -> segments): groupby sensor_idx. Both reconstructable in O(N).
    # Subgroup absence is the explicit "old run / flag off" signal — no format
    # version bump needed.
    seg_sen = event_dict.get('segment_sensor_hits')
    if seg_sen is not None:
        sh = grp.create_group('sensor_hits')
        sh.create_dataset('segment_idx',
                          data=np.asarray(seg_sen['segment_idx'], dtype=np.int32),
                          **GZIP_OPTS)
        sh.create_dataset('digit_idx',
                          data=np.asarray(seg_sen['digit_idx'], dtype=np.int32),
                          **GZIP_OPTS)
        sh.create_dataset('sensor_idx',
                          data=np.asarray(seg_sen['sensor_idx'], dtype=np.uint16),
                          **GZIP_OPTS)
        sh.create_dataset('PE',
                          data=np.asarray(seg_sen['PE'], dtype=np.float32),
                          **GZIP_OPTS)
        sh.create_dataset('T',
                          data=np.asarray(seg_sen['T'], dtype=TIME_DTYPE),
                          **GZIP_OPTS)
        if 'T_reco' in seg_sen:
            sh.create_dataset('T_reco',
                              data=np.asarray(seg_sen['T_reco'], dtype=TIME_DTYPE),
                              **GZIP_OPTS)
        sh.create_dataset('emission_process',
                          data=np.asarray(seg_sen['emission_process'], dtype=np.int8),
                          **GZIP_OPTS)
        sh.attrs['n_segment_hits'] = int(len(seg_sen['PE']))
        grp.attrs['has_segment_sensor_map'] = True


def _initial_direction_per_track(event_dict, mt):
    """Each track's initial direction, as an (n_tracks, 3) float32 unit vector.

    The direction belongs next to ``initial_energy``: both describe the track
    where it starts, and without it no consumer can regress direction from
    ``labl`` alone -- it would have to join ``step`` and repeat this selection,
    which is how the fiTQun truth reader had to do it.

    Taken from the track's EARLIEST-TIME segment rather than its first in array
    order, because segment order within a track is not guaranteed to be sorted
    by time. A track with no segments keeps (0, 0, 0); there is no direction to
    report and a zero vector cannot be mistaken for one.
    """
    n_tracks = len(mt)
    out = np.zeros((n_tracks, 3), dtype=np.float32)
    seg = event_dict.get('segments') or {}
    t_arr = np.asarray(seg.get('time', ()))
    if n_tracks == 0 or t_arr.size == 0:
        return out

    dx, dy, dz = (np.asarray(seg[k]) for k in ('dir_x', 'dir_y', 'dir_z'))
    n_seg = np.fromiter((int(t.get('n_segments', 0)) for t in mt.values()),
                        dtype=np.int64, count=n_tracks)
    off = np.concatenate(([0], np.cumsum(n_seg)))
    for i in range(n_tracks):
        a, b = int(off[i]), int(min(off[i + 1], t_arr.size))
        if b > a:
            j = a + int(np.argmin(t_arr[a:b]))
            out[i] = (dx[j], dy[j], dz[j])
    return out


def save_labl_event(f, event_dict, seq_idx):
    """Write a single event_NNN/ group to an already-open labl file.

    Subgroups:
    * ``per_event/`` — contained (bool) + t0 (= min per_interaction/t0).
    * ``per_interaction/`` — one row per source interaction (one per
      G4 event for non-pile-up; one per vertex for pile-up). Fields:
      source_type, t0, vertex_{x,y,z}, n_primaries, n_particles,
      neutrino_pdg, neutrino_energy_MeV, contained, and CSR-encoded
      primary_{track_ids,pdgs,energies}_{offsets,data}. Dark events
      (no tracks) still get a 1-row table with empty CSR lists.
    * ``per_particle/`` — category, contained, genealogy CSR, and
      ``interaction_idx`` FK into ``per_interaction/`` rows.
    * ``per_track/`` — track metadata + ``particle_idx`` and
      ``interaction`` FK (indexes per_interaction/ rows) columns.

    ``contained`` is a binary geometric-containment flag: True iff every
    meaningful segment in the entity has both endpoints inside
    ``detector_bounds``. AND-composes from segment -> particle ->
    interaction -> event. Empty subsets (interaction with zero
    particles, particle with no segments, event with no interactions)
    are False — see :func:`lucid.dataset.labels.compute_containment` for the full spec.
    """
    grp = f.create_group(_event_group_name(seq_idx))
    grp.attrs['source_event_idx'] = int(event_dict['source_event_idx'])
    grp.attrs['n_particles'] = int(event_dict['n_particles'])
    mt = event_dict.get('meaningful_tracks', {})
    grp.attrs['n_tracks'] = int(len(mt))

    # Track-level derivations (also used to size per_interaction/)
    if mt:
        particle_idx = derive_particle_idx_per_track(event_dict)
        ancestor, interaction = derive_track_ancestor_and_interaction(event_dict)
    else:
        particle_idx = np.array([], dtype=np.int32)
        ancestor = np.array([], dtype=np.int32)
        interaction = np.array([], dtype=np.int32)

    # --- per_interaction ---
    # Row `i` corresponds to tracks whose `interaction == i`. For
    # single-interaction events every row shares t0/vertex/source_type
    # (they come from the same PhotonSim stream); pile-up supplies
    # per-vertex arrays that get indexed here instead.
    pi_grp = grp.create_group('per_interaction')
    _write_per_interaction(pi_grp, event_dict, ancestor, interaction)

    # --- per_event (scalar summaries: contained + t0) ---
    # t0 is derived from per_interaction/t0 as the earliest interaction
    # time in the event — a single-scalar convenience for downstream
    # tools (e.g. the viewer) that want one reference time per event
    # without walking the per_interaction table. For single-interaction
    # events this equals the sole t0.
    pe_grp = grp.create_group('per_event')
    pe_grp.create_dataset('contained',
                          data=np.bool_(event_dict['contained']))
    pi_t0 = pi_grp['t0'][()]
    pe_grp.create_dataset('t0',
                          data=TIME_DTYPE(float(np.min(pi_t0))))

    # --- per_window (trigger gates) ---
    # Dimension table of readout windows: start/end (detector frame) and a CSR
    # digit_offsets into sensor.h5 (window w = digit rows [off[w], off[w+1])).
    # Present only for triggered datasets; absent when the trigger is off.
    pw = event_dict.get('per_window')
    if pw is not None:
        pw_grp = grp.create_group('per_window')
        pw_grp.attrs['n_windows'] = int(np.asarray(pw['window_start']).shape[0])
        pw_grp.create_dataset('window_start', data=np.asarray(pw['window_start'], TIME_DTYPE), **GZIP_OPTS)
        pw_grp.create_dataset('window_end', data=np.asarray(pw['window_end'], TIME_DTYPE), **GZIP_OPTS)
        pw_grp.create_dataset('digit_offsets', data=np.asarray(pw['digit_offsets'], np.int32), **GZIP_OPTS)

    # --- per_particle ---
    pp_grp = grp.create_group('per_particle')
    particles = event_dict['particles']

    cats = []
    for particle in particles:
        ti = particle.get('track_info')
        cat = ti['category'] if ti is not None else -1
        cats.append(cat if cat >= 0 else 255)
    pp_grp.create_dataset('category',
                          data=np.array(cats, dtype=np.uint8),
                          **GZIP_OPTS)

    cont = np.asarray(event_dict['contained_per_particle'], dtype=bool)
    pp_grp.create_dataset('contained', data=cont, **GZIP_OPTS)

    gen_offsets = [0]
    gen_data_list = []
    for particle in particles:
        gen = np.asarray(particle['genealogy'], dtype=np.int32).flatten()
        gen_data_list.append(gen)
        gen_offsets.append(gen_offsets[-1] + len(gen))
    pp_grp.create_dataset('genealogy_offsets',
                          data=np.array(gen_offsets, dtype=np.uint32),
                          **GZIP_OPTS)
    pp_grp.create_dataset('genealogy_data',
                          data=(np.concatenate(gen_data_list)
                                if gen_data_list else np.array([], dtype=np.int32)),
                          **GZIP_OPTS)

    ext_offsets = [0]
    ext_data_list = []
    for particle in particles:
        ext = particle.get('extended_genealogy')
        arr = (np.asarray(ext, dtype=np.int32).flatten()
               if ext is not None else np.array([], dtype=np.int32))
        ext_data_list.append(arr)
        ext_offsets.append(ext_offsets[-1] + len(arr))
    pp_grp.create_dataset('ext_genealogy_offsets',
                          data=np.array(ext_offsets, dtype=np.uint32),
                          **GZIP_OPTS)
    pp_grp.create_dataset('ext_genealogy_data',
                          data=(np.concatenate(ext_data_list)
                                if ext_data_list else np.array([], dtype=np.int32)),
                          **GZIP_OPTS)

    # interaction_idx per particle: derived by mapping each particle's
    # last-in-genealogy (primary) track_id to its interaction rank.
    pp_grp.create_dataset(
        'interaction_idx',
        data=derive_particle_interaction_idx(event_dict, interaction),
        **GZIP_OPTS)

    # --- per_track ---
    pt_grp = grp.create_group('per_track')
    if mt:
        track_id = np.array([t['track_id'] for t in mt.values()], dtype=np.int32)
        parent_id = np.array([t['parent_id'] for t in mt.values()], dtype=np.int32)
        pdg = np.array([t['pdg'] for t in mt.values()], dtype=np.int32)
        initial_energy = np.array([t['initial_energy'] for t in mt.values()],
                                   dtype=np.float32)
        n_ch = np.array([t['n_cherenkov'] for t in mt.values()], dtype=np.int32)
    else:
        track_id = np.array([], dtype=np.int32)
        parent_id = np.array([], dtype=np.int32)
        pdg = np.array([], dtype=np.int32)
        initial_energy = np.array([], dtype=np.float32)
        n_ch = np.array([], dtype=np.int32)

    pt_grp.create_dataset('track_id', data=track_id, **GZIP_OPTS)
    pt_grp.create_dataset('parent_id', data=parent_id, **GZIP_OPTS)
    pt_grp.create_dataset('pdg', data=pdg, **GZIP_OPTS)
    pt_grp.create_dataset('initial_energy', data=initial_energy, **GZIP_OPTS)
    for name, col in zip(('dir_x', 'dir_y', 'dir_z'),
                         _initial_direction_per_track(event_dict, mt).T):
        pt_grp.create_dataset(name, data=col, **GZIP_OPTS)
    pt_grp.create_dataset('n_cherenkov', data=n_ch, **GZIP_OPTS)
    pt_grp.create_dataset('particle_idx', data=particle_idx, **GZIP_OPTS)
    pt_grp.create_dataset('ancestor', data=ancestor, **GZIP_OPTS)
    pt_grp.create_dataset('interaction', data=interaction, **GZIP_OPTS)


def _write_per_interaction(pi_grp, event_dict, ancestor, interaction):
    """Populate the per_interaction/ subgroup.

    One row per interaction — a single G4 event in non-pile-up events
    (so always one row) and one vertex stream in pile-up events (N rows
    for N-way pile-up). Each interaction bundles every primary fired in
    that G4 event plus the full ancestry cascade of each, collapsing a
    multi-primary GENIE interaction or a multi-primary particle-gun
    shot into a single row.

    Fields:
      * ``source_type``           (uint8)   — 0=particles, 1=genie, 2=supernova.
      * ``t0`` / ``vertex_x/y/z`` (float32) — interaction time + vertex.
      * ``n_primaries``           (int32)   — count of ``parent_id==0``
                                              tracks in this interaction.
      * ``n_particles``           (int32)   — count of categorized particles
                                              (primaries + descendants).
      * ``neutrino_pdg``          (int16)   — probe PDG (0 for particle-gun).
      * ``neutrino_energy_MeV``   (float32) — probe KE (0.0 for particle-gun).
      * ``contained``             (bool)    — True iff every meaningful
                                              segment of every particle
                                              attributed to this
                                              interaction is contained;
                                              False for empty
                                              interactions.
      * CSR variable-length per interaction (offsets(n_interactions+1,) + data):
          - ``primary_track_ids_*``  (int32)
          - ``primary_pdgs_*``       (int16)
          - ``primary_energies_*``   (float32)

    Input contract: ``event_dict['interaction_metadata']`` is a list of
    per-interaction dicts produced by
    :func:`lucid.sources.interaction.build_interaction_metadata`.
    Per-interaction ``n_particles`` is derived by counting categorized
    particles whose ``interaction_idx`` matches the row, via the
    already-computed ``interaction`` column.
    """
    interactions = event_dict['interaction_metadata']
    n_interactions = len(interactions)
    assert n_interactions > 0, "interaction_metadata must contain at least one entry"

    # Per-interaction n_particles: count categorized particles assigned to each row.
    n_particles_per_interaction = np.zeros(n_interactions, dtype=np.int32)
    part_inter = derive_particle_interaction_idx(event_dict, interaction)
    for pi in part_inter:
        pi = int(pi)
        if 0 <= pi < n_interactions:
            n_particles_per_interaction[pi] += 1

    # Scalars per interaction
    source_type_arr = np.array(
        [int(s['source_type']) for s in interactions], dtype=np.uint8)
    t0_arr = np.array([float(s['t0']) for s in interactions], dtype=TIME_DTYPE)
    vx_arr = np.stack(
        [np.asarray(s['vertex_xyz'], dtype=np.float32) for s in interactions], axis=0)
    assert vx_arr.shape == (n_interactions, 3)
    neutrino_pdg_arr = np.array(
        [int(s['neutrino_pdg']) for s in interactions], dtype=np.int16)
    neutrino_ke_arr = np.array(
        [float(s['neutrino_energy_MeV']) for s in interactions], dtype=np.float32)
    # supernova interaction channel (sntools code; -1 for non-SN). Together with
    # neutrino_pdg / neutrino_energy_MeV / t0 this gives the SN truth per interaction.
    channel_arr = np.array(
        [int(s.get('interaction_channel', -1)) for s in interactions], dtype=np.int32)
    n_primaries_per_interaction = np.array(
        [len(s['primary_track_ids']) for s in interactions], dtype=np.int32)

    # CSR: per-interaction primary lists.
    tid_offsets = np.zeros(n_interactions + 1, dtype=np.uint32)
    pdg_offsets = np.zeros(n_interactions + 1, dtype=np.uint32)
    e_offsets   = np.zeros(n_interactions + 1, dtype=np.uint32)
    tid_chunks, pdg_chunks, e_chunks = [], [], []
    for i, s in enumerate(interactions):
        tid_chunks.append(np.asarray(s['primary_track_ids'], dtype=np.int32))
        pdg_chunks.append(np.asarray(s['primary_pdgs'],      dtype=np.int16))
        e_chunks.append(  np.asarray(s['primary_energies'],  dtype=np.float32))
        tid_offsets[i + 1] = tid_offsets[i] + len(tid_chunks[-1])
        pdg_offsets[i + 1] = pdg_offsets[i] + len(pdg_chunks[-1])
        e_offsets[i + 1]   = e_offsets[i]   + len(e_chunks[-1])
    tid_data = (np.concatenate(tid_chunks) if tid_chunks
                else np.array([], dtype=np.int32))
    pdg_data = (np.concatenate(pdg_chunks) if pdg_chunks
                else np.array([], dtype=np.int16))
    e_data = (np.concatenate(e_chunks) if e_chunks
              else np.array([], dtype=np.float32))

    contained_arr = np.asarray(
        event_dict['contained_per_interaction'], dtype=bool)
    assert contained_arr.shape == (n_interactions,), (
        f"contained_per_interaction shape {contained_arr.shape} "
        f"!= (n_interactions={n_interactions},)")

    pi_grp.create_dataset('source_type',         data=source_type_arr,    **GZIP_OPTS)
    pi_grp.create_dataset('t0',                  data=t0_arr,             **GZIP_OPTS)
    pi_grp.create_dataset('vertex_x',            data=vx_arr[:, 0].copy(),**GZIP_OPTS)
    pi_grp.create_dataset('vertex_y',            data=vx_arr[:, 1].copy(),**GZIP_OPTS)
    pi_grp.create_dataset('vertex_z',            data=vx_arr[:, 2].copy(),**GZIP_OPTS)
    pi_grp.create_dataset('n_primaries',         data=n_primaries_per_interaction,**GZIP_OPTS)
    pi_grp.create_dataset('n_particles',         data=n_particles_per_interaction,**GZIP_OPTS)
    pi_grp.create_dataset('neutrino_pdg',        data=neutrino_pdg_arr,   **GZIP_OPTS)
    pi_grp.create_dataset('neutrino_energy_MeV', data=neutrino_ke_arr,    **GZIP_OPTS)
    pi_grp.create_dataset('interaction_channel', data=channel_arr,        **GZIP_OPTS)
    pi_grp.create_dataset('contained',           data=contained_arr,      **GZIP_OPTS)
    pi_grp.create_dataset('primary_track_ids_offsets', data=tid_offsets,  **GZIP_OPTS)
    pi_grp.create_dataset('primary_track_ids_data',    data=tid_data,     **GZIP_OPTS)
    pi_grp.create_dataset('primary_pdgs_offsets',      data=pdg_offsets,  **GZIP_OPTS)
    pi_grp.create_dataset('primary_pdgs_data',         data=pdg_data,     **GZIP_OPTS)
    pi_grp.create_dataset('primary_energies_offsets',  data=e_offsets,    **GZIP_OPTS)
    pi_grp.create_dataset('primary_energies_data',     data=e_data,       **GZIP_OPTS)
