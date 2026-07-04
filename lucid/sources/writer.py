"""four-file HDF5 writers (sensor / hits / step / labl) and containment.

Contains all ``write_*_config``, ``save_*_event`` functions,
``_compute_contained``, ``_source_type_code``, ``build_interaction_metadata``,
``sample_translation_vector``, and the ``SOURCE_TYPE_*`` constants.
"""
from __future__ import annotations

import h5py
import numpy as np

from lucid.sources.event_builder import (
    derive_particle_idx_per_track,
    derive_track_ancestor_and_interaction,
)
from lucid.sources.particle_physics import derive_particle_interaction_idx

__all__ = [
    "SOURCE_TYPE_PARTICLES",
    "SOURCE_TYPE_GENIE",
    "SOURCE_TYPE_SUPERNOVA",
    "EMISSION_PROCESS_CHERENKOV",
    "EMISSION_PROCESS_SCINTILLATION",
    "EMISSION_PROCESS_DARK",
    "build_interaction_metadata",
    "sample_translation_vector",
    "write_sensor_config",
    "write_hits_config",
    "write_step_config",
    "write_labl_config",
    "save_sensor_event",
    "save_hits_event",
    "save_step_event",
    "save_labl_event",
]


# ---------------------------------------------------------------------------
# format: four-file per-event-group HDF5 (sensor / hits / step / labl).
# See docs/LUCID_DATASET.md for the full schema.
# ---------------------------------------------------------------------------

_GZIP_OPTS = dict(compression='gzip', compression_opts=4)
_FORMAT_VERSION = 5

# per_interaction/source_type encoding
SOURCE_TYPE_PARTICLES = 0
SOURCE_TYPE_GENIE     = 1
SOURCE_TYPE_SUPERNOVA = 2

# emission_process column encoding on hits.h5 + step.h5/sensor_hits/.
# Per-row tag identifying which physical process produced the contribution.
# Cherenkov-only datasets emit the column with all-zeros (Cherenkov), so
# downstream consumers can always group/filter by it without a presence check.
EMISSION_PROCESS_CHERENKOV     = 0
EMISSION_PROCESS_SCINTILLATION = 1
EMISSION_PROCESS_DARK          = 2   # electronic dark-noise contribution (particle_idx = -1)


def _source_type_code(primary_source):
    """Map config's ``primary_source`` string to the per_interaction/source_type int."""
    if primary_source == 'genie':
        return SOURCE_TYPE_GENIE
    if primary_source == 'supernova':
        return SOURCE_TYPE_SUPERNOVA
    return SOURCE_TYPE_PARTICLES


def _compute_contained(event_dict, detector_bounds):
    """Geometric containment flags at per-segment / particle / interaction / event level.

    A meaningful segment is *contained* iff both its start point and its
    end point lie inside ``detector_bounds``. A superset is contained iff
    every member is contained:

        * particle  — AND over all meaningful segments attributed to it
                       via the same parent-chain walk used by
                       ``derive_particle_idx_per_track``.
        * interaction — AND over its particles.
        * event       — AND over its interactions.

    Empty subsets evaluate to **False** (an interaction with zero
    categorized particles, a particle with no attributed segments, an
    event with no interactions, or any case with ``detector_bounds=None``).
    Identity-True would silently mark neutrinos / dark events as
    "contained," which downstream filters would misread.

    Segment positions are assumed to be in the detector frame (already
    shifted by the per-interaction translation_vector when applicable);
    the inside test uses the post-translation coordinates so the answer
    matches what the step file actually stores.

    Segment-to-track mapping uses cumulative ``n_segments`` over tracks
    in dict-insertion order — the same invariant ``save_step_event``
    relies on — so the helper works for both single-stream events and
    merged pile-up streams where per-track ``segment_offset`` fields are
    stream-local and no longer valid in the merged flat array.

    Returns
    -------
    dict:
        'per_segment'     : bool (n_segments,)
        'per_particle'    : bool (n_particles,)
        'per_interaction' : bool (n_interactions,)
        'overall'         : bool scalar
    """
    particles = event_dict.get('particles') or []
    meaningful_tracks = event_dict.get('meaningful_tracks') or {}
    segments = event_dict.get('segments') or {}
    interactions = event_dict.get('interaction_metadata') or []
    n_particles = len(particles)
    n_interactions = len(interactions)
    n_segments = int(segments.get('n_segments', 0))

    per_segment_arr = np.zeros(n_segments, dtype=bool)
    per_particle_arr = np.zeros(n_particles, dtype=bool)
    per_interaction_arr = np.zeros(n_interactions, dtype=bool)
    overall = np.bool_(False)

    if detector_bounds is None:
        return {'per_segment':     per_segment_arr,
                'per_particle':    per_particle_arr,
                'per_interaction': per_interaction_arr,
                'overall':         overall}

    # Inside test on both endpoints. AND over (start_inside, end_inside).
    if n_segments > 0:
        sx = np.asarray(segments['start_x'], dtype=np.float64)
        sy = np.asarray(segments['start_y'], dtype=np.float64)
        sz = np.asarray(segments['start_z'], dtype=np.float64)
        ex = np.asarray(segments['end_x'], dtype=np.float64)
        ey = np.asarray(segments['end_y'], dtype=np.float64)
        ez = np.asarray(segments['end_z'], dtype=np.float64)
        kind = detector_bounds['type']
        if kind == 'cylinder':
            R = float(detector_bounds['radius'])
            HZ = float(detector_bounds['height']) / 2.0
            start_in = (np.sqrt(sx * sx + sy * sy) <= R) & (np.abs(sz) <= HZ)
            end_in   = (np.sqrt(ex * ex + ey * ey) <= R) & (np.abs(ez) <= HZ)
        elif kind == 'sphere':
            R = float(detector_bounds['radius'])
            start_in = np.sqrt(sx * sx + sy * sy + sz * sz) <= R
            end_in   = np.sqrt(ex * ex + ey * ey + ez * ez) <= R
        elif kind == 'box':
            HL = float(detector_bounds['length']) / 2.0
            HW = float(detector_bounds['width'])  / 2.0
            HH = float(detector_bounds['height']) / 2.0
            start_in = (np.abs(sx) <= HL) & (np.abs(sy) <= HW) & (np.abs(sz) <= HH)
            end_in   = (np.abs(ex) <= HL) & (np.abs(ey) <= HW) & (np.abs(ez) <= HH)
        else:
            raise ValueError(f"Unknown detector_bounds type: {kind!r}")
        per_segment_arr = (start_in & end_in).astype(bool)

    # ------------------------------------------------------------------
    # Per-track contained, vectorized.
    # Group segments per meaningful track via cumulative n_segments (the
    # save_step_event invariant). A track is contained iff every owned
    # segment is contained. Zero-segment tracks -> True (no-evidence).
    # ------------------------------------------------------------------
    n_tracks = len(meaningful_tracks)
    if n_tracks == 0:
        per_particle_arr = np.zeros(n_particles, dtype=bool)
    else:
        track_ids_np   = np.fromiter(
            (int(tid) for tid in meaningful_tracks.keys()),
            dtype=np.int64, count=n_tracks)
        n_seg_per_track = np.fromiter(
            (int(t.get('n_segments', 0)) for t in meaningful_tracks.values()),
            dtype=np.int64, count=n_tracks)
        parent_per_track = np.fromiter(
            (int(t['parent_id']) for t in meaningful_tracks.values()),
            dtype=np.int64, count=n_tracks)

        # Reverse map track_id -> local idx in O(n).
        tid_to_local = {int(tid): i for i, tid in enumerate(track_ids_np.tolist())}

        # seg -> local track idx for in-range segments.
        # Out-of-range counts (cursor + ns > n_segments) get clamped so
        # we don't index past per_segment_arr; their entries fall into
        # the "empty" branch below and stay True.
        seg_idx_per_seg = np.repeat(np.arange(n_tracks, dtype=np.int64), n_seg_per_track)
        if seg_idx_per_seg.size > n_segments:
            # Defensive truncation matching the legacy "cursor+ns <= n_segments" guard.
            seg_idx_per_seg = seg_idx_per_seg[:n_segments]
        # Per-track AND of per_segment_arr: a track is False iff any of
        # its segments is False. Indexed assignment of False at False-
        # segment positions is order-independent (idempotent).
        seg_contained_per_track_np = np.ones(n_tracks, dtype=bool)
        if seg_idx_per_seg.size > 0:
            false_segs = ~per_segment_arr[:seg_idx_per_seg.size]
            seg_contained_per_track_np[seg_idx_per_seg[false_segs]] = False
        # Tracks with zero segments stay True (already initialized).

        # ------------------------------------------------------------------
        # Owner walk, vectorized. Each meaningful track climbs its
        # parent chain (within meaningful_tracks) until it hits either
        # a categorized root or falls off the chain. Iteration count is
        # bounded by max chain depth; for SK-class events this is shallow.
        # ------------------------------------------------------------------
        # Build per-track "categorized root" tag (-1 if track is not the
        # last entry of any particle's genealogy).
        owner_of_local = np.full(n_tracks, -1, dtype=np.int64)
        for i, particle in enumerate(particles):
            gen = particle.get('genealogy') or []
            if not gen:
                continue
            last_tid = int(gen[-1])
            li = tid_to_local.get(last_tid)
            if li is not None:
                owner_of_local[li] = i

        # Per-track parent local idx (-1 if parent not in meaningful_tracks).
        parent_local = np.full(n_tracks, -1, dtype=np.int64)
        for i, pid in enumerate(parent_per_track.tolist()):
            li = tid_to_local.get(pid)
            if li is not None:
                parent_local[i] = li

        cur = np.arange(n_tracks, dtype=np.int64)
        final_owner = np.full(n_tracks, -1, dtype=np.int64)
        alive = np.ones(n_tracks, dtype=bool)
        for _ in range(n_tracks):  # bounded by deepest chain
            new_owner = owner_of_local[cur]
            found = alive & (new_owner >= 0)
            if found.any():
                final_owner = np.where(found, new_owner, final_owner)
                alive &= ~found
            if not alive.any():
                break
            p = parent_local[cur]
            fell_off = alive & (p < 0)
            alive &= ~fell_off
            if not alive.any():
                break
            cur = np.where(alive, p, cur)

        valid_owners = final_owner >= 0
        particle_has_track = np.zeros(n_particles, dtype=bool)
        particle_and = np.ones(n_particles, dtype=bool)
        if valid_owners.any():
            owners = final_owner[valid_owners]
            contained = seg_contained_per_track_np[valid_owners]
            # Indexed assignment of True is order-independent.
            particle_has_track[owners] = True
            # AND aggregation: a particle becomes False if ANY owned
            # track is False. Idempotent assignment of False at the
            # False-track owners.
            not_contained = ~contained
            if not_contained.any():
                particle_and[owners[not_contained]] = False

        # Particle: True iff it owns >=1 track AND every owned segment is contained.
        per_particle_arr = particle_has_track & particle_and

    # ------------------------------------------------------------------
    # Interaction: AND over its particles. Empty interaction -> False.
    # ------------------------------------------------------------------
    inter_idx = derive_particle_interaction_idx(event_dict)
    interaction_has_particle = np.zeros(n_interactions, dtype=bool)
    interaction_and = np.ones(n_interactions, dtype=bool)
    if n_particles > 0 and inter_idx.size > 0:
        m = min(n_particles, inter_idx.size)
        ii = inter_idx[:m].astype(np.int64, copy=False)
        valid = (ii >= 0) & (ii < n_interactions)
        if valid.any():
            valid_inter = ii[valid]
            valid_part  = per_particle_arr[:m][valid]
            interaction_has_particle[valid_inter] = True
            not_contained = ~valid_part
            if not_contained.any():
                interaction_and[valid_inter[not_contained]] = False
    per_interaction_arr = interaction_has_particle & interaction_and

    # Event: AND over interactions. Empty event -> False.
    overall = np.bool_(n_interactions > 0 and bool(per_interaction_arr.all()))

    return {'per_segment':     per_segment_arr,
            'per_particle':    per_particle_arr,
            'per_interaction': per_interaction_arr,
            'overall':         overall}


def build_interaction_metadata(particle_data, *, t0, vertex_xyz, source_type_code):
    """Assemble the per-interaction metadata dict consumed by ``_write_per_interaction``.

    One interaction corresponds to one G4 event in the non-pile-up path
    and to one vertex stream in the pile-up path. The dict records the
    physics vertex (t0 + vertex_xyz), the source flag, the incoming-
    neutrino probe info from PhotonSim (GENIE-only — zero sentinels for
    particle-gun), and the full list of status==1 primaries fired in
    this interaction with their PDGs and initial kinetic energies.

    Primaries are identified as ``parent_id == 0`` entries in
    ``meaningful_tracks`` and are emitted sorted by track_id for a
    deterministic layout.
    """
    mt = particle_data.get('meaningful_tracks') or {}
    rows = [(int(tid), int(t['pdg']), float(t['initial_energy']))
            for tid, t in mt.items() if int(t['parent_id']) == 0]
    rows.sort(key=lambda r: r[0])
    primary_tids = [r[0] for r in rows]
    primary_pdgs = [r[1] for r in rows]
    primary_energies = [r[2] for r in rows]
    return {
        't0': float(t0),
        'vertex_xyz': np.asarray(vertex_xyz, dtype=np.float32).copy(),
        'source_type': int(source_type_code),
        'neutrino_pdg': int(particle_data.get('neutrino_pdg', 0)),
        'neutrino_energy_MeV': float(particle_data.get('neutrino_energy_MeV', 0.0)),
        'primary_track_ids': primary_tids,
        'primary_pdgs': primary_pdgs,
        'primary_energies': primary_energies,
    }


def sample_translation_vector(detector_bounds, rng):
    """Draw a random vertex inside the fiducial volume of the detector.

    Cylinder: uniform in (r, theta, z) with r <= 0.9*R, |z| <= 0.45*H.
    Sphere:   uniform in the 0.9*R ball.
    Box:      uniform in the 0.9-fraction-scaled box.

    Returns a length-3 float32 array in meters.
    """
    if detector_bounds is None:
        return np.zeros(3, dtype=np.float32)
    kind = detector_bounds['type']
    if kind == 'cylinder':
        r_max = detector_bounds['radius'] * 0.9
        h_max = detector_bounds['height'] * 0.9 / 2.0
        u = rng.uniform(0, 1, size=3).astype(np.float32)
        r = r_max * np.sqrt(u[0])
        theta = 2.0 * np.pi * u[1]
        z = (2.0 * u[2] - 1.0) * h_max
        return np.array([r * np.cos(theta), r * np.sin(theta), z], dtype=np.float32)
    if kind == 'sphere':
        r_max = detector_bounds['radius'] * 0.9
        u = rng.uniform(0, 1, size=3).astype(np.float32)
        r = r_max * (u[0] ** (1.0 / 3.0))
        cos_t = 2.0 * u[1] - 1.0
        phi = 2.0 * np.pi * u[2]
        sin_t = np.sqrt(1.0 - cos_t * cos_t)
        return r * np.array([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t],
                            dtype=np.float32)
    if kind == 'box':
        u = rng.uniform(0, 1, size=3).astype(np.float32)
        return np.array([
            (2.0 * u[0] - 1.0) * detector_bounds['length'] * 0.45,
            (2.0 * u[1] - 1.0) * detector_bounds['width']  * 0.45,
            (2.0 * u[2] - 1.0) * detector_bounds['height'] * 0.45,
        ], dtype=np.float32)
    raise ValueError(f"Unknown detector_bounds type: {kind!r}")


def _write_common_config_attrs(f, config_meta):
    """Create ``config/`` group with provenance attrs common to all files."""
    cfg = f.require_group('config')
    cfg.attrs['format_version'] = _FORMAT_VERSION
    cfg.attrs['n_events'] = int(config_meta['n_events'])
    cfg.attrs['git_commit'] = str(config_meta.get('git_commit', 'unknown'))
    cfg.attrs['run_id'] = str(config_meta['run_id'])
    cfg.attrs['dataset_name'] = str(config_meta['dataset_name'])
    cfg.attrs['file_index'] = int(config_meta.get('file_index', 0))
    cfg.attrs['source_file'] = str(config_meta['source_file'])
    cfg.attrs['lucid_master_seed'] = int(config_meta['lucid_master_seed'])
    cfg.attrs['photonsim_seed'] = int(config_meta.get('photonsim_seed', -1))
    return cfg


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
                       **_GZIP_OPTS)
    cfg.create_dataset('sensor_positions',
                       data=np.asarray(sensor_positions, dtype=np.float32),
                       **_GZIP_OPTS)


def write_hits_config(f, config_meta, source_event_idx, sensor_positions):
    """Write the config/ group of a hits file."""
    cfg = _write_common_config_attrs(f, config_meta)
    cfg.attrs['n_sensors'] = int(config_meta['n_sensors'])
    cfg.attrs['detector_type'] = str(config_meta['detector_type'])
    cfg.attrs['material'] = str(config_meta['material'])
    cfg.create_dataset('source_event_idx',
                       data=np.asarray(source_event_idx, dtype=np.uint32),
                       **_GZIP_OPTS)
    cfg.create_dataset('sensor_positions',
                       data=np.asarray(sensor_positions, dtype=np.float32),
                       **_GZIP_OPTS)


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
                       **_GZIP_OPTS)


def write_labl_config(f, config_meta, source_event_idx):
    """Write the config/ group of a labl file."""
    cfg = _write_common_config_attrs(f, config_meta)
    label_names = list(config_meta.get('label_names', ['category']))
    cfg.attrs['label_names'] = np.array(label_names, dtype=h5py.string_dtype())
    cfg.create_dataset('source_event_idx',
                       data=np.asarray(source_event_idx, dtype=np.uint32),
                       **_GZIP_OPTS)


def _event_group_name(seq_idx):
    return f'event_{int(seq_idx):03d}'


def save_sensor_event(f, event_dict, seq_idx):
    """Write a single event_NNN/ group to an already-open sensor file.

    ``event_dict`` must contain: ``source_event_idx``, ``PE_reco``,
    ``T_reco``. Times in ``T_reco`` are expected in absolute detector
    frame — the caller applies per-interaction t0 shifts before calling
    this writer; the writer does not shift times further.
    """
    grp = f.create_group(_event_group_name(seq_idx))
    grp.attrs['source_event_idx'] = int(event_dict['source_event_idx'])

    if 'sensor_digits' in event_dict:
        # Digitizer path: a pre-sparsified list of recorded digits. A sensor
        # index may repeat (multi-hit models); ``basic`` yields one row per
        # sensor. All rows are real hits — no masking needed.
        sd = event_dict['sensor_digits']
        indices = np.asarray(sd['sensor_idx'], dtype=np.uint16)
        pe_sparse = np.asarray(sd['PE'], dtype=np.float32)
        t_sparse = np.asarray(sd['T'], dtype=np.float32)
    else:
        # Legacy dense path (non-digitizer callers, e.g. calibration).
        pe = np.asarray(event_dict['PE_reco'], dtype=np.float32)
        t = np.asarray(event_dict['T_reco'], dtype=np.float32)
        # A "hit" is a sensor with real charge: pe > 0. SK-like charge smearing
        # preserves zero (sigma=0 when counts=0), so pe == 0 => no photon ever
        # reached this sensor. The isfinite & <1e5 checks catch smear_times'
        # 1e6 non-finite sentinel. Absolute time can legitimately be negative
        # (t0 in [-250, +250] ns), so no lower bound.
        mask = (pe > 0) & np.isfinite(t) & (t < 1e5)
        indices = np.where(mask)[0].astype(np.uint16)
        pe_sparse = pe[mask].astype(np.float32)
        t_sparse = np.where(np.isfinite(t[mask]), t[mask], np.float32(0.0)).astype(np.float32)

    grp.attrs['n_hits'] = int(indices.size)
    grp.create_dataset('sensor_idx', data=indices, **_GZIP_OPTS)
    grp.create_dataset('PE', data=pe_sparse, **_GZIP_OPTS)
    grp.create_dataset('T', data=t_sparse, **_GZIP_OPTS)


def save_hits_event(f, event_dict, seq_idx):
    """Write a single event_NNN/ group to an already-open hits file.

    Two input shapes accepted on ``event_dict``:

    * **Legacy / Cherenkov-only**: ``PE_per_particle``, ``T_per_particle``
      (and optional ``T_reco_per_particle``) as dense ``(n_particles,
      n_sensors)`` tensors. The writer sparsifies via the ``pe > 0`` mask
      and emits ``emission_process = EMISSION_PROCESS_CHERENKOV`` on every
      row.
    * **Per-process (Phase 2+)**: ``hits_sparse`` — a dict of pre-merged
      sparse columns ``particle_idx`` / ``sensor_idx`` / ``PE`` / ``T``
      (and optional ``T_reco``) plus an ``emission_process`` int8 column,
      one row per (particle, sensor, emission_process) triple. The
      sparsification + per-process tagging is done by the caller (e.g.
      ``generate_events_from_photonsim_particles`` after running the
      Cherenkov and scintillation aggregator passes); the writer writes
      the columns verbatim.

    Times are expected in absolute detector frame — no shift applied here.
    """
    grp = f.create_group(_event_group_name(seq_idx))
    grp.attrs['source_event_idx'] = int(event_dict['source_event_idx'])
    grp.attrs['n_particles'] = int(event_dict['n_particles'])

    if 'hits_sparse' in event_dict:
        sparse = event_dict['hits_sparse']
        particle_idx_arr = np.asarray(sparse['particle_idx'], dtype=np.int32)
        sensor_idx_arr   = np.asarray(sparse['sensor_idx'],   dtype=np.uint16)
        pe_arr           = np.asarray(sparse['PE'],           dtype=np.float32)
        t_arr            = np.asarray(sparse['T'],            dtype=np.float32)
        emp_arr          = np.asarray(sparse['emission_process'], dtype=np.int8)
        t_reco_arr = (np.asarray(sparse['T_reco'], dtype=np.float32)
                      if 'T_reco' in sparse else None)
        # digit_idx (FK -> sensor.h5 digit row) present in the digitizer path.
        digit_idx_arr = (np.asarray(sparse['digit_idx'], dtype=np.int32)
                         if 'digit_idx' in sparse else None)
    else:
        digit_idx_arr = None
        pe_pp = np.asarray(event_dict['PE_per_particle'], dtype=np.float32)
        t_pp = np.asarray(event_dict['T_per_particle'], dtype=np.float32)
        n_p = pe_pp.shape[0]

        t_reco_pp = event_dict.get('T_reco_per_particle')
        if t_reco_pp is not None:
            t_reco_pp = np.asarray(t_reco_pp, dtype=np.float32)

        particle_idx_parts, sensor_idx_parts, pe_parts, t_parts = [], [], [], []
        t_reco_parts = []
        for i in range(n_p):
            mask = pe_pp[i] > 0
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            particle_idx_parts.append(np.full(idx.shape[0], i, dtype=np.int32))
            sensor_idx_parts.append(idx.astype(np.uint16))
            pe_parts.append(pe_pp[i, mask].astype(np.float32))
            t_vals = t_pp[i, mask]
            t_vals = np.where(np.isfinite(t_vals), t_vals, np.float32(0.0))
            t_parts.append(t_vals.astype(np.float32))
            if t_reco_pp is not None:
                t_reco_vals = t_reco_pp[i, mask]
                t_reco_vals = np.where(np.isfinite(t_reco_vals), t_reco_vals, np.float32(0.0))
                t_reco_parts.append(t_reco_vals.astype(np.float32))

        def _cat(xs, dtype):
            return np.concatenate(xs).astype(dtype) if xs else np.array([], dtype=dtype)

        particle_idx_arr = _cat(particle_idx_parts, np.int32)
        sensor_idx_arr = _cat(sensor_idx_parts, np.uint16)
        pe_arr = _cat(pe_parts, np.float32)
        t_arr = _cat(t_parts, np.float32)
        t_reco_arr = (_cat(t_reco_parts, np.float32)
                      if t_reco_pp is not None else None)
        emp_arr = np.full(particle_idx_arr.size,
                          EMISSION_PROCESS_CHERENKOV, dtype=np.int8)

    grp.attrs['n_particle_hits'] = int(particle_idx_arr.size)
    # digit_idx links each decomposition row to its sensor.h5 digit. Legacy /
    # non-digitizer callers (one hit per sensor) get all-zeros.
    if digit_idx_arr is None:
        digit_idx_arr = np.zeros(particle_idx_arr.size, dtype=np.int32)
    grp.create_dataset('particle_idx', data=particle_idx_arr, **_GZIP_OPTS)
    grp.create_dataset('digit_idx', data=digit_idx_arr, **_GZIP_OPTS)
    grp.create_dataset('sensor_idx', data=sensor_idx_arr, **_GZIP_OPTS)
    grp.create_dataset('PE', data=pe_arr, **_GZIP_OPTS)
    grp.create_dataset('T', data=t_arr, **_GZIP_OPTS)
    if t_reco_arr is not None:
        grp.create_dataset('T_reco', data=t_reco_arr, **_GZIP_OPTS)
    grp.create_dataset('emission_process', data=emp_arr, **_GZIP_OPTS)


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

    grp.create_dataset('track_idx', data=track_idx_arr, **_GZIP_OPTS)

    # per-segment contained flag computed alongside the higher-level
    # AND-rollups by `_compute_contained`. Saves any reader from having
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
            'time': (np.asarray(seg['time'], dtype=np.float32), np.float32),
            'beta_start': (seg['beta_start'], np.float32),
            'n_cherenkov': (seg['n_cherenkov'], np.int32),
            'group_id': (group_id_arr, np.int32),
            'contained': (contained_per_segment, bool),
        }
        for name, (arr, dtype) in fields.items():
            grp.create_dataset(name,
                               data=np.asarray(arr, dtype=dtype),
                               **_GZIP_OPTS)
    else:
        for name, dtype in (('start_x', np.float32), ('start_y', np.float32),
                            ('start_z', np.float32), ('end_x', np.float32),
                            ('end_y', np.float32), ('end_z', np.float32),
                            ('dir_x', np.float16), ('dir_y', np.float16),
                            ('dir_z', np.float16), ('edep', np.float32),
                            ('time', np.float32), ('beta_start', np.float32),
                            ('n_cherenkov', np.int32), ('group_id', np.int32),
                            ('contained', bool)):
            grp.create_dataset(name, data=_empty(dtype), **_GZIP_OPTS)

    # Optional segment <-> sensor correspondence map. Mirrors hits file's flat
    # parallel-array shape: each row is one (segment, sensor) pair with PE+T.
    # Forward map (segment -> sensors): groupby segment_idx. Reverse map
    # (sensor -> segments): groupby sensor_idx. Both reconstructable in O(N).
    # Subgroup absence is the explicit "old run / flag off" signal — no format
    # version bump needed.
    seg_sen = event_dict.get('segment_sensor_hits')
    if seg_sen is not None:
        sh = grp.create_group('sensor_hits')
        n_seg_hits = int(np.asarray(seg_sen['segment_idx']).size)
        sh.create_dataset('segment_idx',
                          data=np.asarray(seg_sen['segment_idx'], dtype=np.int32),
                          **_GZIP_OPTS)
        # digit_idx (FK -> sensor.h5 digit); legacy callers get all-zeros so the
        # sensor_hits -> hits -> sensor aggregation stays digit-consistent.
        digit_idx_arr = seg_sen.get('digit_idx')
        if digit_idx_arr is None:
            digit_idx_arr = np.zeros(n_seg_hits, dtype=np.int32)
        sh.create_dataset('digit_idx',
                          data=np.asarray(digit_idx_arr, dtype=np.int32),
                          **_GZIP_OPTS)
        sh.create_dataset('sensor_idx',
                          data=np.asarray(seg_sen['sensor_idx'], dtype=np.uint16),
                          **_GZIP_OPTS)
        sh.create_dataset('PE',
                          data=np.asarray(seg_sen['PE'], dtype=np.float32),
                          **_GZIP_OPTS)
        sh.create_dataset('T',
                          data=np.asarray(seg_sen['T'], dtype=np.float32),
                          **_GZIP_OPTS)
        if 'T_reco' in seg_sen:
            sh.create_dataset('T_reco',
                              data=np.asarray(seg_sen['T_reco'], dtype=np.float32),
                              **_GZIP_OPTS)
        # emission_process: per-row physical-process tag. Caller-provided
        # column wins; falls back to all-zeros (Cherenkov) for legacy
        # callers that don't merge per-process sparse rows themselves.
        emp_arr = seg_sen.get('emission_process')
        if emp_arr is None:
            emp_arr = np.full(len(seg_sen['PE']),
                              EMISSION_PROCESS_CHERENKOV, dtype=np.int8)
        else:
            emp_arr = np.asarray(emp_arr, dtype=np.int8)
        sh.create_dataset('emission_process', data=emp_arr, **_GZIP_OPTS)
        sh.attrs['n_segment_hits'] = int(len(seg_sen['PE']))
        grp.attrs['has_segment_sensor_map'] = True


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
    are False — see ``_compute_contained`` for the full spec.
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
                          data=np.float32(float(np.min(pi_t0))))

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
                          **_GZIP_OPTS)

    cont = np.asarray(event_dict['contained_per_particle'], dtype=bool)
    pp_grp.create_dataset('contained', data=cont, **_GZIP_OPTS)

    gen_offsets = [0]
    gen_data_list = []
    for particle in particles:
        gen = np.asarray(particle['genealogy'], dtype=np.int32).flatten()
        gen_data_list.append(gen)
        gen_offsets.append(gen_offsets[-1] + len(gen))
    pp_grp.create_dataset('genealogy_offsets',
                          data=np.array(gen_offsets, dtype=np.uint32),
                          **_GZIP_OPTS)
    pp_grp.create_dataset('genealogy_data',
                          data=(np.concatenate(gen_data_list)
                                if gen_data_list else np.array([], dtype=np.int32)),
                          **_GZIP_OPTS)

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
                          **_GZIP_OPTS)
    pp_grp.create_dataset('ext_genealogy_data',
                          data=(np.concatenate(ext_data_list)
                                if ext_data_list else np.array([], dtype=np.int32)),
                          **_GZIP_OPTS)

    # interaction_idx per particle: derived by mapping each particle's
    # last-in-genealogy (primary) track_id to its interaction rank.
    pp_grp.create_dataset(
        'interaction_idx',
        data=derive_particle_interaction_idx(event_dict, interaction),
        **_GZIP_OPTS)

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

    pt_grp.create_dataset('track_id', data=track_id, **_GZIP_OPTS)
    pt_grp.create_dataset('parent_id', data=parent_id, **_GZIP_OPTS)
    pt_grp.create_dataset('pdg', data=pdg, **_GZIP_OPTS)
    pt_grp.create_dataset('initial_energy', data=initial_energy, **_GZIP_OPTS)
    pt_grp.create_dataset('n_cherenkov', data=n_ch, **_GZIP_OPTS)
    pt_grp.create_dataset('particle_idx', data=particle_idx, **_GZIP_OPTS)
    pt_grp.create_dataset('ancestor', data=ancestor, **_GZIP_OPTS)
    pt_grp.create_dataset('interaction', data=interaction, **_GZIP_OPTS)


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
    per-interaction dicts produced by :func:`build_interaction_metadata`.
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
    t0_arr = np.array([float(s['t0']) for s in interactions], dtype=np.float32)
    vx_arr = np.stack(
        [np.asarray(s['vertex_xyz'], dtype=np.float32) for s in interactions], axis=0)
    assert vx_arr.shape == (n_interactions, 3)
    neutrino_pdg_arr = np.array(
        [int(s['neutrino_pdg']) for s in interactions], dtype=np.int16)
    neutrino_ke_arr = np.array(
        [float(s['neutrino_energy_MeV']) for s in interactions], dtype=np.float32)
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

    pi_grp.create_dataset('source_type',         data=source_type_arr,    **_GZIP_OPTS)
    pi_grp.create_dataset('t0',                  data=t0_arr,             **_GZIP_OPTS)
    pi_grp.create_dataset('vertex_x',            data=vx_arr[:, 0].copy(),**_GZIP_OPTS)
    pi_grp.create_dataset('vertex_y',            data=vx_arr[:, 1].copy(),**_GZIP_OPTS)
    pi_grp.create_dataset('vertex_z',            data=vx_arr[:, 2].copy(),**_GZIP_OPTS)
    pi_grp.create_dataset('n_primaries',         data=n_primaries_per_interaction,**_GZIP_OPTS)
    pi_grp.create_dataset('n_particles',         data=n_particles_per_interaction,**_GZIP_OPTS)
    pi_grp.create_dataset('neutrino_pdg',        data=neutrino_pdg_arr,   **_GZIP_OPTS)
    pi_grp.create_dataset('neutrino_energy_MeV', data=neutrino_ke_arr,    **_GZIP_OPTS)
    pi_grp.create_dataset('contained',           data=contained_arr,      **_GZIP_OPTS)
    pi_grp.create_dataset('primary_track_ids_offsets', data=tid_offsets,  **_GZIP_OPTS)
    pi_grp.create_dataset('primary_track_ids_data',    data=tid_data,     **_GZIP_OPTS)
    pi_grp.create_dataset('primary_pdgs_offsets',      data=pdg_offsets,  **_GZIP_OPTS)
    pi_grp.create_dataset('primary_pdgs_data',         data=pdg_data,     **_GZIP_OPTS)
    pi_grp.create_dataset('primary_energies_offsets',  data=e_offsets,    **_GZIP_OPTS)
    pi_grp.create_dataset('primary_energies_data',     data=e_data,       **_GZIP_OPTS)
