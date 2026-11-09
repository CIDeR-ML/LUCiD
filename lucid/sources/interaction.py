"""One PhotonSim interaction through the data-mode simulator: placement, propagation, digits.

The library half of event generation; the loops that choose events, seeds and files live in
``lucid.production.dataprod``. For one interaction::

    raw = read_event_raw(root_file, entry)                     # lucid.sources.root_reader
    place_raw_entry(raw, primary_vertex=recorded_vertex(raw), rotation=R, vertex=v)
    stream, max_track_id = simulate_interaction(sim, raw, keys, t0_ns=..., vertex=v, ...)
    event = digitize([stream, ...], n_sensors=..., readout=..., rng=..., ...)
    selected = readout.select(...)                              # lucid.simulation.readout

A *stream* is one interaction's particles, tracks and segments (segment times in the detector
frame) plus its detected photo-electron deposits in the interaction's own time frame. ``digitize``
turns one or more streams into one event's digits and their decompositions.
"""
from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

from lucid.simulation.digitizer import digitize_and_decompose
from lucid.sources.data_event import pad_photons
from lucid.dataset.labels import (
    TrackEntry,
    assign_group_ids,
    bucket_photons_by_segment,
    categorize_event,
    compute_containment,
    derive_meaningful_tracks,
    filter_segments_to_meaningful,
)
from lucid.dataset.schema import EMISSION_PROCESS_CHERENKOV, EMISSION_PROCESS_SCINTILLATION
from lucid.sources.scintillation_photons import (
    expand_segments_to_photons,
    scintillation_medium_params,
)

__all__ = [
    "DEFAULT_RAYS_BUCKETS",
    "resolve_rays_buckets",
    "warmup_buckets",
    "recorded_vertex",
    "place_raw_entry",
    "simulate_interaction",
    "digitize",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Single-axis bucketing (n_rays)
# =============================================================================
#
# The data-mode kernel (`_common_propagation` in lucid/simulation/simulator.py)
# JIT-caches on ``n_rays`` (size of the per-chunk photon array). Quantizing
# this axis into a small finite set of bucket sizes is enough to keep the
# JIT cache bounded: a warmup pass compiles each entry once at startup; every
# subsequent event hits the cache. Per chunk, photons are padded to the
# smallest fitting bucket; the kernel returns per-photon flat lists, so no
# segment axis needs bucketing.
DEFAULT_RAYS_BUCKETS = (256, 2_048, 8_192, 32_768)


def resolve_rays_buckets(spec=None):
    """A bucket spec as a sorted tuple of unique positive ints; ``None`` or empty -> the default."""
    return tuple(sorted({int(b) for b in (spec or DEFAULT_RAYS_BUCKETS) if int(b) > 0}))


def _pick_bucket(n, buckets):
    """Smallest bucket >= n. Falls back to the largest bucket if n exceeds it."""
    for b in buckets:
        if b >= n:
            return b
    return buckets[-1]


def _split_into_chunks(n, buckets):
    """Split ``n`` photons into a list of bucket sizes covering all of them.

    For n <= max(buckets): a single chunk in the smallest fitting bucket.
    For n  > max(buckets): floor(n/max) chunks of `max(buckets)` plus one
    remainder chunk in the smallest fitting bucket. The kernel only consumes
    `Np` photons per call (mask in simulator.py:470), so unused slots are
    masked out — bucket sizes only affect the JIT cache key, never physics.
    """
    max_bucket = buckets[-1]
    chunks = []
    remaining = n
    while remaining > max_bucket:
        chunks.append(max_bucket)
        remaining -= max_bucket
    if remaining > 0:
        chunks.append(_pick_bucket(remaining, buckets))
    return chunks


def warmup_buckets(event_simulator, rays_buckets):
    """Trigger one JIT compile per ``n_rays_bucket`` (4 entries on default spec).

    Runs the simulator once per bucket with throwaway zero-filled photons
    (``N=0``, so no real propagation work) and blocks on the result. After
    this returns, every real-event kernel call whose ``n_rays`` matches
    one of the warmed buckets hits the JIT cache and runs at native
    kernel cost. Each compile is ~10-30 s on CPU; total warmup ~80-120 s
    on the default 4-bucket spec.
    """
    if not rays_buckets:
        return
    logger.info("Warming up the JIT cache for %d rays bucket(s)", len(rays_buckets))
    default_dir = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    for rays_b in rays_buckets:
        t0_warm = time.time()
        photonsim_data = {
            'photon_origins':    jnp.zeros((rays_b, 3), dtype=jnp.float32),
            'photon_directions': jnp.tile(default_dir, (rays_b, 1)),
            'photon_times':      jnp.zeros(rays_b, dtype=jnp.float32),
            'wavelengths':       jnp.zeros(rays_b, dtype=jnp.float32),
            'N': jnp.int32(0),
            'photon_segment_index': jnp.full(rays_b, -1, dtype=jnp.int32),
        }
        key = jax.random.PRNGKey(0)
        result = event_simulator(key, photonsim_data)
        # Block on every leaf so the JIT trace finishes.
        for arr in result:
            arr.block_until_ready()
        logger.info("  rays=%6d: %.2fs", rays_b, time.time() - t0_warm)


def _trace_event_bucketed(
        event_simulator,
        photon_origins_np, photon_directions_np,
        photon_times_np, photon_wavelengths_np,
        photon_segment_index_raw,
        n_sensors, rays_buckets,
        master_key):
    """Bucketed propagation for one event: one kernel call per bucket-sized chunk of photons.

    Per chunk, photons are padded to the smallest fitting ``rays_buckets`` entry and propagated in
    one kernel call; the kernel's per-photon flat records are kept, minus the bucket padding. The
    JIT cache key is the bucket size only. ``chunk_key = jax.random.fold_in(master_key, chunk_idx)``.

    Parameters
    ----------
    event_simulator : callable
        Built by ``setup_event_simulator(..., hits='segment_truth')``.
    photon_*_np : np.ndarray
        All-event flat photon arrays (already placed by the caller), origins in metres.
    photon_segment_index_raw : np.ndarray (int32 or int64)
        ``(N_photons,)`` indices into the **raw** segment table; -1 for orphans.
    n_sensors : int
        Detector sensor count.
    rays_buckets : tuple[int, ...]
        Sorted bucket spec (output of :func:`resolve_rays_buckets`).
    master_key : jax.Array
        Per-event master RNG key.

    Returns
    -------
    photon_qe_weight     : (factor·N_photons,) float32 — 0 for QE-failed entries
    photon_qe_time       : (factor·N_photons,) float32 — +inf for QE-failed / no hit
    photon_qe_time_reco  : (factor·N_photons,) float32 — TTS-smeared, +inf if failed
    photon_sensor_idx    : (factor·N_photons,) int32
    photon_global_idx    : (factor·N_photons,) int32 — source photon id (0..N-1) of each row

    ``factor = K · max_sensors_per_cell`` is the number of (propagation iteration, sensor cell)
    pairs each photon contributes to in the soft-hit kernel.
    """
    if not rays_buckets:
        raise ValueError("_trace_event_bucketed requires non-empty rays_buckets")
    rays_buckets = tuple(rays_buckets)

    N = int(photon_origins_np.shape[0])
    if N == 0:
        return (np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32))

    # Metres at the kernel boundary, as root_reader and place_raw_entry give them.
    po = np.ascontiguousarray(photon_origins_np, dtype=np.float32)
    pd = np.ascontiguousarray(photon_directions_np, dtype=np.float32)
    pt = np.ascontiguousarray(photon_times_np,      dtype=np.float32)
    pw = np.ascontiguousarray(photon_wavelengths_np, dtype=np.float32)
    psi = np.ascontiguousarray(photon_segment_index_raw, dtype=np.int32)

    qe_w_chunks      = []
    qe_t_chunks      = []
    qe_t_reco_chunks = []
    sen_i_chunks     = []
    gid_chunks       = []

    chunks = _split_into_chunks(N, rays_buckets)
    offset = 0
    for chunk_idx, bucket_size in enumerate(chunks):
        n_in_chunk = min(bucket_size, N - offset)
        sl = slice(offset, offset + n_in_chunk)

        chunk = pad_photons({'photon_origins': po[sl], 'photon_directions': pd[sl],
                             'photon_times': pt[sl], 'wavelengths': pw[sl],
                             'photon_segment_index': psi[sl]}, bucket_size)
        photonsim_data = {name: jax.device_put(a) for name, a in chunk.items()}
        photonsim_data['N'] = jnp.int32(n_in_chunk)

        chunk_key = jax.random.fold_in(master_key, chunk_idx)
        # The per-sensor totals (first three outputs) and the echoed segment index (last) are
        # not needed here: the deposits below carry everything the digitizer uses.
        (_, _, _, qe_w_chunk, qe_t_chunk, qe_t_reco_chunk, sen_i_chunk, _) = (
            event_simulator(chunk_key, photonsim_data))

        # The kernel returns flat arrays of length (K · max_sensors_per_cell · bucket_size),
        # one entry per (propagation iteration, sensor cell, photon): each photon's light is
        # spread over several (k, sensor_cell) pairs in the soft-hit model. Keep the FULL flat
        # arrays (a [:n_in_chunk] slice would keep only the k=0, cell=0 slab), drop only the
        # bucket padding (photon ids >= n_in_chunk), and emit a global photon id per kept row
        # so per-photon arrays (segment, particle) can be gathered to kernel-flat alignment.
        qe_w_arr      = np.asarray(qe_w_chunk,      dtype=np.float32)
        qe_t_arr      = np.asarray(qe_t_chunk,      dtype=np.float32)
        qe_t_reco_arr = np.asarray(qe_t_reco_chunk, dtype=np.float32)
        sen_i_arr     = np.asarray(sen_i_chunk,      dtype=np.int32)
        flat_len = qe_w_arr.shape[0]
        photon_axis = np.arange(flat_len, dtype=np.int64) % bucket_size
        keep = photon_axis < n_in_chunk
        qe_w_chunks     .append(qe_w_arr [keep])
        qe_t_chunks     .append(qe_t_arr [keep])
        qe_t_reco_chunks.append(qe_t_reco_arr[keep])
        sen_i_chunks    .append(sen_i_arr[keep])
        gid_chunks      .append((photon_axis[keep] + offset).astype(np.int32))

        offset += n_in_chunk

    return (np.concatenate(qe_w_chunks), np.concatenate(qe_t_chunks),
            np.concatenate(qe_t_reco_chunks), np.concatenate(sen_i_chunks),
            np.concatenate(gid_chunks))


def _derive_views_from_segments(raw, photon_records=None):
    """Categorize, filter, and assemble the downstream-view dict of one raw entry.

    Runs the categorizer and ``segment_grouping.assign_group_ids`` on the raw read output and, if
    the kernel's per-photon flat records are supplied, aligns them with the per-photon segment and
    particle indices.

    Parameters
    ----------
    raw : dict
        Output of :func:`lucid.sources.root_reader.read_event_raw`.
    photon_records : dict or None
        ``qe_weight``, ``qe_time``, ``qe_time_reco``, ``sensor_idx`` -- kernel-flat arrays of shape
        ``(factor·N_photons,)`` from :func:`_trace_event_bucketed` -- and ``photon_global_idx`` of
        the same shape, the source photon id of each row. ``None`` when there was no kernel call.

    Returns
    -------
    dict
        Particles, meaningful tracks, the filtered segment table (m), ``photon_particle_idx``
        (each photon's particle, -1 for none) and ``photon_records_filtered`` (the kernel records
        plus ``seg_idx_filtered`` and ``particle_idx`` at kernel-flat alignment; ``None`` without
        records).
    """
    track_info_dict = raw['track_info_dict']
    seg_track_id_full   = raw['segments_raw']['track_id']
    seg_n_cherenkov_full = raw['segments_raw']['n_cherenkov']

    # ---- Segment_* → meaningful_tracks (groupby on Segment_TrackID) ----
    meaningful_tracks = derive_meaningful_tracks(
        segment_track_id=seg_track_id_full,
        segment_n_cherenkov=seg_n_cherenkov_full,
        track_info=track_info_dict,
    )

    # Filter Segment_* arrays to meaningful only. Keeps step file size
    # the same as legacy and lines up with meaningful_tracks' segment
    # offsets (which index into the filtered array).
    keep_mask, photon_segment_index = filter_segments_to_meaningful(
        segment_track_id=seg_track_id_full,
        meaningful_tracks=meaningful_tracks,
        photon_segment_index=raw['photon_segment_index_raw'],
    )

    # ---- Run the Python categorizer ----
    # Build per-track entries; the decision tree only reads pdg, parent,
    # KE and creator_process.
    track_rows = [
        TrackEntry(
            track_id=tid,
            parent_id=ti['parent_id'],
            pdg=ti['pdg'],
            ke_mev=ti['energy'],
            creator_process=ti['creator_process'],
        )
        for tid, ti in track_info_dict.items()
    ]

    # The categorizer's secondary-pion category-parent walk needs the
    # meaningful track parent/pdg map. Build it from meaningful_tracks.
    meaningful_track_parent_pdg = {
        tid: (int(t['parent_id']), int(t['pdg']))
        for tid, t in meaningful_tracks.items()
    }
    cherenkov_count_by_mt_track = {
        tid: int(t['n_cherenkov']) for tid, t in meaningful_tracks.items()
    }
    cat_result = categorize_event(
        track_info_rows=track_rows,
        meaningful_track_parent_pdg=meaningful_track_parent_pdg,
        cherenkov_count_by_mt_track=cherenkov_count_by_mt_track,
    )

    # Plumb category / sub_id back into track_info_dict so save_labl_event
    # (and any other consumer) can read it. Iterate once over the category
    # dict (sub_id is a strict subset) and look up each track once.
    cat_dict = cat_result.category_by_track_id
    sub_dict = cat_result.sub_id_by_track_id
    for tid, cat in cat_dict.items():
        ti = track_info_dict.get(tid)
        if ti is None:
            continue
        ti['category'] = int(cat)
        sub = sub_dict.get(tid)
        if sub is not None:
            ti['sub_id'] = int(sub)

    # ---- Bucket photons → particle ----
    # photon_segment_index is now in *filtered* segment positions; pass
    # the matching filtered Segment_TrackID so seg→track lookup lines up.
    seg_track_id_filtered = seg_track_id_full[keep_mask]
    photon_to_particle = bucket_photons_by_segment(
        photon_segment_index=photon_segment_index,
        segment_track_id=seg_track_id_filtered,
        particle_idx_by_meaningful_track=cat_result.particle_idx_by_meaningful_track,
    )

    # ---- Build particles list ----
    n_particles = len(cat_result.genealogies)
    particles = []
    for p_idx in range(n_particles):
        genealogy = list(cat_result.genealogies[p_idx])
        ext_genealogy = list(cat_result.ext_genealogies[p_idx])
        last_track_id = genealogy[-1] if genealogy else None
        track_info = track_info_dict.get(last_track_id) if last_track_id is not None else None
        particles.append({
            'genealogy': genealogy,
            'extended_genealogy': ext_genealogy,
            'track_info': track_info,
        })

    # ---- Build the filtered segments dict ----
    # mm → m for endpoint coords. Filter every parallel array via keep_mask.
    seg_raw = raw['segments_raw']
    seg_start_x_mm = seg_raw['start_x_mm'][keep_mask]
    seg_start_y_mm = seg_raw['start_y_mm'][keep_mask]
    seg_start_z_mm = seg_raw['start_z_mm'][keep_mask]
    seg_end_x_mm   = seg_raw['end_x_mm'][keep_mask]
    seg_end_y_mm   = seg_raw['end_y_mm'][keep_mask]
    seg_end_z_mm   = seg_raw['end_z_mm'][keep_mask]
    seg_dir_x      = seg_raw['dir_x'][keep_mask]
    seg_dir_y      = seg_raw['dir_y'][keep_mask]
    seg_dir_z      = seg_raw['dir_z'][keep_mask]
    seg_edep       = seg_raw['edep'][keep_mask]
    seg_time       = seg_raw['time'][keep_mask]
    seg_beta_start = seg_raw['beta_start'][keep_mask]
    seg_n_cherenkov = seg_raw['n_cherenkov'][keep_mask]

    n_segments = int(seg_edep.shape[0])

    # Group raw G4 sub-steps into the (legacy) merged segments via
    # segment_grouping.assign_group_ids — the offsets in
    # meaningful_tracks index into this filtered array.
    group_id = assign_group_ids(
        start_x_mm=seg_start_x_mm,
        start_y_mm=seg_start_y_mm,
        start_z_mm=seg_start_z_mm,
        end_x_mm=seg_end_x_mm,
        end_y_mm=seg_end_y_mm,
        end_z_mm=seg_end_z_mm,
        dir_x=seg_dir_x,
        dir_y=seg_dir_y,
        dir_z=seg_dir_z,
        edep_mev=seg_edep,
        meaningful_tracks=meaningful_tracks,
    )

    segments = {
        'start_x': seg_start_x_mm / 1000.0,
        'start_y': seg_start_y_mm / 1000.0,
        'start_z': seg_start_z_mm / 1000.0,
        'end_x':   seg_end_x_mm / 1000.0,
        'end_y':   seg_end_y_mm / 1000.0,
        'end_z':   seg_end_z_mm / 1000.0,
        'dir_x': seg_dir_x,
        'dir_y': seg_dir_y,
        'dir_z': seg_dir_z,
        'edep': seg_edep,
        'time': seg_time,
        'beta_start': seg_beta_start,
        'n_cherenkov': seg_n_cherenkov,
        'group_id': group_id,
        'n_segments': n_segments,
    }

    edep_len = len(segments['edep'])
    assert len(segments['beta_start']) == edep_len, (
        f"Segment_BetaStart length {len(segments['beta_start'])} != Segment_Edep {edep_len}")
    assert len(segments['n_cherenkov']) == edep_len, (
        f"Segment_NCherenkov length {len(segments['n_cherenkov'])} != Segment_Edep {edep_len}")
    assert len(segments['group_id']) == edep_len, (
        f"group_id length {len(segments['group_id'])} != Segment_Edep {edep_len}")

    # ---- Build photon_records_filtered for the deposits ----
    # ``filter_segments_to_meaningful`` produced ``photon_segment_index`` in
    # the **filtered** segment space (length N); ``bucket_photons_by_segment``
    # gave us the per-photon particle index (length N). Both carry the -1
    # sentinel for orphans, which is what the deposits' QE-pass mask
    # needs. The kernel-flat arrays in ``photon_records`` have one row per
    # (propagation iteration, sensor cell, photon) tuple — length factor·N
    # — so we gather the per-photon arrays via ``photon_global_idx`` to bring
    # all five arrays to a common kernel-flat alignment for the groupbys.
    if photon_records is not None:
        gid = np.asarray(photon_records['photon_global_idx'], dtype=np.int64)
        photon_records_filtered = {
            'qe_weight':        photon_records['qe_weight'],
            'qe_time':          photon_records['qe_time'],
            'qe_time_reco':     photon_records['qe_time_reco'],
            'sensor_idx':       photon_records['sensor_idx'],
            'seg_idx_filtered': (photon_segment_index[gid]
                                 if photon_segment_index.size
                                 else photon_segment_index).astype(np.int32, copy=False),
            'particle_idx':     (photon_to_particle[gid]
                                 if photon_to_particle.size
                                 else photon_to_particle).astype(np.int32, copy=False),
        }
    else:
        photon_records_filtered = None

    return {
        'n_particles':         n_particles,
        'particles':           particles,
        'photon_origins':      raw['photon_origins'],
        'photon_directions':   raw['photon_directions'],
        'photon_times':        raw['photon_times'],
        'photon_wavelengths':  raw['photon_wavelengths'],
        'photon_segment_index': photon_segment_index,
        'photon_particle_idx': photon_to_particle,
        'primary_energy':      raw['primary_energy'],
        'track_info_dict':     track_info_dict,
        'meaningful_tracks':   meaningful_tracks,
        'segments':            segments,
        'photon_records_filtered': photon_records_filtered,
        # GENIE provenance — pass through.
        'rootracker_entry_id':  raw['rootracker_entry_id'],
        'neutrino_pdg':         raw['neutrino_pdg'],
        'neutrino_energy_MeV':  raw['neutrino_energy_MeV'],
    }


def _run_process(
        *,
        event_simulator,
        raw,
        photon_origins_np,
        photon_directions_np,
        photon_times_np,
        photon_wavelengths_np,
        photon_segment_index_raw,
        n_sensors,
        rays_buckets,
        sim_key):
    """Run the kernel and derive the views for **one** photon stream (one emission process).

    The drivers call this once per emission process (Cherenkov, then scintillation when the
    medium has it) and use the first process's view as the event-level structure (the segment
    table is the same for every process). Returns ``{'particle_data': view}``.
    """
    (photon_qe_w, photon_qe_t, photon_qe_t_reco,
     photon_sen_i, photon_gid) = _trace_event_bucketed(
        event_simulator,
        photon_origins_np, photon_directions_np,
        photon_times_np, photon_wavelengths_np,
        photon_segment_index_raw,
        n_sensors, rays_buckets,
        sim_key,
    )

    # _derive_views_from_segments reads raw['photon_segment_index_raw'] to
    # compute the filtered photon_segment_index. For the scintillation pass
    # we swap in the per-process indices via a shallow copy so the original
    # raw dict (used by other code paths) stays untouched.
    raw_for_derive = dict(raw)
    raw_for_derive['photon_segment_index_raw'] = photon_segment_index_raw
    particle_data = _derive_views_from_segments(raw_for_derive, photon_records={
        'qe_weight':         photon_qe_w,
        'qe_time':           photon_qe_t,
        'qe_time_reco':      photon_qe_t_reco,
        'sensor_idx':        photon_sen_i,
        'photon_global_idx': photon_gid,
    })
    return {'particle_data': particle_data}


def _gather_deposits(process_outputs):
    """Flatten per-deposit photon records across emission processes.

    Each process output carries per-(photon, sensor) detected deposits in
    ``particle_data['photon_records_filtered']`` (QE already applied upstream).
    This concatenates them into flat arrays for the digitizer, tagging each row
    with the process's ``process_id`` as the ``emission_process``. Returns a
    dict of equal-length arrays (empty when no deposits / no records).
    """
    S, W, TT, TR, PI, SEG, EMP = [], [], [], [], [], [], []
    for p_out in process_outputs:
        pr = (p_out.get('particle_data') or {}).get('photon_records_filtered')
        if not pr:
            continue
        w = np.asarray(pr['qe_weight'])
        if w.size == 0:
            continue
        # Keep only DETECTED photo-electrons (qe_weight > 0). The kernel-flat
        # records carry one row per (photon, candidate-sensor, step) — the vast
        # majority are QE-failed / zero-weight. Masking here collapses tens of millions of
        # rows to the ~detected PE before any concat/pool/sort downstream.
        keep = w > 0
        if not keep.any():
            continue
        proc = int(p_out.get('process_id', 0))
        S.append(np.asarray(pr['sensor_idx'])[keep])
        W.append(w[keep])
        TT.append(np.asarray(pr['qe_time'])[keep])
        TR.append(np.asarray(pr['qe_time_reco'])[keep])
        PI.append(np.asarray(pr['particle_idx'])[keep])
        SEG.append(np.asarray(pr['seg_idx_filtered'])[keep])
        EMP.append(np.full(int(keep.sum()), proc, dtype=np.int64))

    def _cat(xs, dt):
        return np.concatenate(xs).astype(dt) if xs else np.array([], dtype=dt)

    return {
        'sensor_idx':       _cat(S, np.int64),
        'charge':           _cat(W, np.float64),
        't_true':           _cat(TT, np.float64),
        't_reco':           _cat(TR, np.float64),
        'particle_idx':     _cat(PI, np.int64),
        'segment_idx':      _cat(SEG, np.int64),
        'emission_process': _cat(EMP, np.int64),
    }


# =============================================================================
# One interaction
# =============================================================================
def recorded_vertex(raw):
    """The vertex PhotonSim recorded for a raw entry: where its primaries (``parent_id == 0``)
    start, in metres. An entry whose primaries start at different points has no vertex; that
    raises."""
    starts = np.unique([t['position'] for t in raw['track_info_dict'].values()
                        if t['parent_id'] == 0], axis=0)
    if len(starts) != 1:
        raise ValueError(f'the entry\'s primaries start at {len(starts)} points; it has no single '
                         'vertex to place')
    return starts[0]


def place_raw_entry(raw, *, primary_vertex, rotation, vertex):
    """Place one raw PhotonSim entry, in place: every geometric field moves by the rigid motion
    ``x -> R (x - primary_vertex) + vertex`` (:func:`lucid.sources.placement.rigid_motion`).

    Distances from the vertex and all relative angles are preserved, so photon emission times stay
    valid and the photon <-> segment correspondence is untouched. Photons, segments and tracks move
    together: segment and track *directions* come from their own ROOT branches, not from the
    endpoints, so moving positions alone would desynchronise them. The photons move in float32
    (as the kernel takes them), the segments (mm) and tracks (m) in float64.

    Parameters
    ----------
    raw : dict
        Output of :func:`lucid.sources.root_reader.read_event_raw`; modified and returned.
    primary_vertex : (3,) array
        The entry's recorded vertex (m), :func:`recorded_vertex`.
    rotation : (3, 3) array
        Rotation matrix.
    vertex : (3,) array
        Where the vertex lands (m).
    """
    from lucid.sources.placement import rigid_motion
    R = np.asarray(rotation, np.float64)
    pivot = np.asarray(primary_vertex, np.float64)
    vertex = np.asarray(vertex, np.float64)
    raw['photon_origins'], raw['photon_directions'] = rigid_motion(
        raw['photon_origins'].astype(np.float32, copy=False),
        raw['photon_directions'].astype(np.float32, copy=False),
        pivot=pivot.astype(np.float32), rotation=R.astype(np.float32),
        vertex=vertex.astype(np.float32))
    seg = raw['segments_raw']
    in_mm = dict(pivot=pivot * 1000.0, rotation=R, vertex=vertex * 1000.0)
    dirs = np.stack([seg['dir_x'], seg['dir_y'], seg['dir_z']], axis=1)
    for pre in ('start', 'end'):
        v, d = rigid_motion(np.stack([seg[f'{pre}_{c}_mm'] for c in 'xyz'], axis=1), dirs, **in_mm)
        seg[f'{pre}_x_mm'], seg[f'{pre}_y_mm'], seg[f'{pre}_z_mm'] = v[:, 0], v[:, 1], v[:, 2]
    seg['dir_x'], seg['dir_y'], seg['dir_z'] = d[:, 0], d[:, 1], d[:, 2]
    for tinfo in raw['track_info_dict'].values():
        (tinfo['position'],), (tinfo['direction'],) = rigid_motion(
            np.asarray(tinfo['position'], np.float64)[None],
            np.asarray(tinfo['direction'], np.float64)[None],
            pivot=pivot, rotation=R, vertex=vertex)
    return raw


def _offset_track_ids_raw(raw, offset):
    """Shift every G4 track id of a raw entry by ``offset``, in place; return the max id after.

    Shifts ``raw['track_info_dict']`` (rekeyed, ``track_id`` and ``parent_id``) and
    ``raw['segments_raw']['track_id']``. ``parent_id == 0`` (the primary convention) is kept so
    primaries stay recognizable after merging; ``photon_segment_index_raw`` indexes the segment
    table, not a track id, and is not shifted.
    """
    if offset == 0:
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
        tid_arr = np.asarray(seg['track_id'], dtype=np.int64)
        seg['track_id'] = np.where(tid_arr > 0, tid_arr + np.int64(offset), np.int64(0))

    tid_d = raw.get('track_info_dict') or {}
    return max((int(t) for t in tid_d.keys()), default=0)


def build_interaction_metadata(particle_data, *, t0, vertex_xyz, source_type_code,
                               interaction_channel=None):
    """The ``per_interaction`` row of one interaction.

    Records the physics vertex (t0 + vertex_xyz), the source flag, the incoming-neutrino probe
    from PhotonSim (GENIE only; zero sentinels for a particle gun), the supernova channel (-1 when
    not a supernova interaction) and the primaries (``parent_id == 0`` meaningful tracks, sorted
    by track id) with their PDGs and initial kinetic energies.
    """
    mt = particle_data.get('meaningful_tracks') or {}
    rows = [(int(tid), int(t['pdg']), float(t['initial_energy']))
            for tid, t in mt.items() if int(t['parent_id']) == 0]
    rows.sort(key=lambda r: r[0])
    return {
        't0': float(t0),
        'vertex_xyz': np.asarray(vertex_xyz, dtype=np.float32).copy(),
        'source_type': int(source_type_code),
        'neutrino_pdg': int(particle_data.get('neutrino_pdg', 0)),
        'neutrino_energy_MeV': float(particle_data.get('neutrino_energy_MeV', 0.0)),
        'interaction_channel': int(interaction_channel) if interaction_channel is not None else -1,
        'primary_track_ids': [r[0] for r in rows],
        'primary_pdgs': [r[1] for r in rows],
        'primary_energies': [r[2] for r in rows],
    }


def simulate_interaction(event_simulator, raw, keys, *, t0_ns, vertex, source_type, n_sensors,
                         rays_buckets, track_offset=0, interaction_channel=None):
    """Propagate one placed PhotonSim interaction and build its stream.

    Track ids are first shifted by ``track_offset`` (so streams merged into one event do not
    collide). Then one kernel pass per emission process: Cherenkov from the PhotonSim photons and,
    when the simulator's medium scintillates, scintillation photons expanded from the segments with
    ``keys['scint_seed']``. Each process gets its own kernel key,
    ``fold_in(keys['sim_key'], process_id)``. The first process's view is the stream's particle
    structure (the segment table is the same for every process).

    Parameters
    ----------
    event_simulator : Simulator
        Built by ``setup_event_simulator(..., source='photons', default_detector_params=True,
        hits='segment_truth')``.
    raw : dict
        A raw entry, already placed (:func:`place_raw_entry`).
    keys : dict
        ``derive_event_keys(...)`` of this interaction.
    t0_ns : float
        The interaction time; segment times are shifted by it (float64).
    vertex : (3,) array
        The placed vertex (m), recorded in ``per_interaction``.
    source_type : int
        ``per_interaction/source_type`` code.
    n_sensors : int
        Detector sensor count.
    rays_buckets : tuple of int
        Output of :func:`resolve_rays_buckets`.
    track_offset : int
        Added to every track id.
    interaction_channel : int, optional
        Supernova interaction channel.

    Returns
    -------
    stream : dict
        ``particles``, ``meaningful_tracks``, ``segments``, ``deposits`` (per detected
        photo-electron, interaction time frame), ``t0`` and ``interaction_meta``.
    max_track_id : int
        The largest track id after the shift.
    """
    max_track_id = _offset_track_ids_raw(raw, track_offset)

    process_inputs = [{
        'process_id':               EMISSION_PROCESS_CHERENKOV,
        'photon_origins':           raw['photon_origins'].astype(np.float32, copy=False),
        'photon_directions':        raw['photon_directions'].astype(np.float32, copy=False),
        'photon_times':             raw['photon_times'].astype(np.float32, copy=False),
        'photon_wavelengths':       raw['photon_wavelengths'].astype(np.float32, copy=False),
        'photon_segment_index_raw': np.asarray(raw['photon_segment_index_raw'], dtype=np.int64),
    }]
    medium = event_simulator.det_geom.medium
    if "scintillation" in medium.emission_processes:
        scint = expand_segments_to_photons(
            raw['segments_raw'],
            scintillation_medium_params(event_simulator.default_detector_params, medium),
            np.random.default_rng(keys['scint_seed']))
        process_inputs.append({'process_id': EMISSION_PROCESS_SCINTILLATION, **scint})

    process_outputs = []
    for p_in in process_inputs:
        out = _run_process(
            event_simulator=event_simulator, raw=raw,
            photon_origins_np=p_in['photon_origins'],
            photon_directions_np=p_in['photon_directions'],
            photon_times_np=p_in['photon_times'],
            photon_wavelengths_np=p_in['photon_wavelengths'],
            photon_segment_index_raw=p_in['photon_segment_index_raw'],
            n_sensors=n_sensors, rays_buckets=rays_buckets,
            sim_key=jax.random.fold_in(keys['sim_key'], int(p_in['process_id'])),
        )
        out['process_id'] = p_in['process_id']
        process_outputs.append(out)

    particle_data = process_outputs[0]['particle_data']
    deposits = _gather_deposits(process_outputs)
    if particle_data['segments'].get('n_segments', 0) > 0:
        particle_data['segments']['time'] = (
            np.asarray(particle_data['segments']['time'], dtype=np.float64) + np.float64(t0_ns))

    stream = {
        'particles':         particle_data['particles'],
        'meaningful_tracks': particle_data['meaningful_tracks'],
        'segments':          particle_data['segments'],
        'deposits':          deposits,
        't0':                float(t0_ns),
        'interaction_meta':  build_interaction_metadata(
            particle_data, t0=t0_ns, vertex_xyz=vertex, source_type_code=source_type,
            interaction_channel=interaction_channel),
    }
    return stream, max_track_id


# =============================================================================
# Digitization of one or more streams
# =============================================================================
_SEGMENT_KEYS = ('start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z',
                 'dir_x', 'dir_y', 'dir_z', 'edep', 'time', 'beta_start', 'n_cherenkov')


def _digitize_local_frame(stream, *, n_sensors, readout, rng, apply_resolution):
    """One stream digitized in its own time frame, digit times then shifted by its t0 (float64).

    The late-light cap of the digitizer is then measured against the earliest deposit
    (``t_ref=None``); the pooled path measures it against each deposit's interaction t0.
    """
    sensor_digits, hits_sparse, seg_hits = digitize_and_decompose(
        **stream['deposits'],
        n_sensors=int(n_sensors), model=readout.digitizer, rng=rng,
        dark_rate_khz=float(readout.digitizer.get('dark_rate_khz', 0.0)),
        readout_pad_ns=readout.dark_pad_ns(),
        apply_resolution=apply_resolution)
    # Every digit / decomposition row is a real hit, so a flat add suffices. Done in float64:
    # t0 can reach second-scale (supernova bursts) where float32 loses the sub-ns light timing.
    t0 = np.float64(stream['t0'])
    for d, cols in ((sensor_digits, ('T',)), (hits_sparse, ('T', 'T_reco')),
                    (seg_hits, ('T', 'T_reco'))):
        for k in cols:
            if k in d and d[k].size:
                d[k] = d[k].astype(np.float64) + t0
    meta = stream['interaction_meta']
    return {
        'n_particles': len(stream['particles']),
        'particles': stream['particles'],
        'meaningful_tracks': stream['meaningful_tracks'],
        'segments': stream['segments'],
        'interaction_metadata': [meta],
        'primary_to_interaction': {tid: 0 for tid in meta['primary_track_ids']},
        'sensor_digits': sensor_digits,
        'hits_sparse': hits_sparse,
        'segment_sensor_hits': seg_hits,
    }


def _digitize_pooled(streams, *, n_sensors, readout, rng, apply_resolution):
    """Streams merged into one event and digitized together, in the detector frame.

    Particles, tracks and segments are concatenated in stream order (track ids were offset
    upstream, so they are disjoint). Every stream's deposits are pooled into one absolute-time
    list: times shifted by the stream's t0, ``particle_idx`` / ``segment_idx`` by the same
    cumulative counts used for the tables (the -1 dark/orphan sentinel kept), and a per-deposit
    ``t_ref`` = the stream's t0 so the late-light cap is measured per interaction. Overlapping
    light therefore merges into shared digits whose charge decomposes across the contributing
    streams' particles.
    """
    all_particles = []
    all_tracks = {}
    all_segs = {k: [] for k in _SEGMENT_KEYS}
    interaction_metadata = [s['interaction_meta'] for s in streams]
    primary_to_interaction = {}
    for i, meta in enumerate(interaction_metadata):
        for tid in meta['primary_track_ids']:
            primary_to_interaction[int(tid)] = i

    pool = {k: [] for k in ('sensor_idx', 'charge', 't_true', 't_reco',
                            'particle_idx', 'segment_idx', 'emission_process', 't_ref')}
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

        dep = s.get('deposits') or {}
        d_sensor = np.asarray(dep.get('sensor_idx', np.empty(0, np.int64)))
        if d_sensor.size:
            t0v = np.float64(s['t0'])
            pi = np.asarray(dep['particle_idx']).astype(np.int64, copy=True)
            pi[pi >= 0] += particle_offset
            si = np.asarray(dep['segment_idx']).astype(np.int64, copy=True)
            si[si >= 0] += seg_offset
            pool['sensor_idx'].append(d_sensor.astype(np.int64))
            pool['charge'].append(np.asarray(dep['charge'], np.float64))
            pool['t_true'].append(np.asarray(dep['t_true'], np.float64) + t0v)
            pool['t_reco'].append(np.asarray(dep['t_reco'], np.float64) + t0v)
            pool['t_ref'].append(np.full(d_sensor.size, t0v, np.float64))
            pool['particle_idx'].append(pi)
            pool['segment_idx'].append(si)
            pool['emission_process'].append(np.asarray(dep['emission_process'], np.int64))

        particle_offset += len(s['particles'])
        seg_offset += n_seg_v

    if all_segs['time']:
        seg_merged = {k: np.concatenate(v) for k, v in all_segs.items()}
        seg_merged['n_segments'] = int(len(seg_merged['time']))
    else:
        seg_merged = {'n_segments': 0}

    def _catp(key, dt):
        return np.concatenate(pool[key]).astype(dt) if pool[key] else np.array([], dtype=dt)

    sensor_digits, hits_sparse, seg_hits = digitize_and_decompose(
        sensor_idx=_catp('sensor_idx', np.int64),
        charge=_catp('charge', np.float64),
        t_true=_catp('t_true', np.float64),
        t_reco=_catp('t_reco', np.float64),
        particle_idx=_catp('particle_idx', np.int64),
        segment_idx=_catp('segment_idx', np.int64),
        emission_process=_catp('emission_process', np.int64),
        t_ref=_catp('t_ref', np.float64),
        n_sensors=n_sensors, model=readout.digitizer, rng=rng,
        dark_rate_khz=float(readout.digitizer.get('dark_rate_khz', 0.0)),
        readout_pad_ns=readout.dark_pad_ns(),
        apply_resolution=apply_resolution)

    event = {
        'n_particles': int(len(all_particles)),
        'particles': all_particles,
        'meaningful_tracks': all_tracks,
        'segments': seg_merged,
        'interaction_metadata': interaction_metadata,
        'primary_to_interaction': primary_to_interaction,
        'hits_sparse': hits_sparse,
        'sensor_digits': sensor_digits,
    }
    if seg_hits['PE'].size > 0:
        event['segment_sensor_hits'] = seg_hits
    return event


def digitize(streams, *, n_sensors, readout, rng, detector_bounds, apply_resolution,
             local_frame=False):
    """Digitize one event from its interaction streams and label its containment.

    Parameters
    ----------
    streams : list of dict
        Outputs of :func:`simulate_interaction`, in track-id order.
    n_sensors : int
        Detector sensor count.
    readout : Readout
        Digitizer model and the dark-noise span (:meth:`Readout.dark_pad_ns`).
    rng : np.random.Generator
        Dark noise, charge and time resolution.
    detector_bounds : dict or None
        For the containment flags (``lucid.sources.placement.detector_bounds``).
    apply_resolution : bool
        Apply the digitizer's charge and time resolution.
    local_frame : bool
        One stream only: digitize it in its own time frame and shift the digit times by its t0
        (the single-vertex production path). Otherwise the streams are pooled in the detector
        frame (pile-up, supernova). The two differ in the late-light cap reference and in float
        rounding, and the pooled event keeps only the written segment columns (no ``group_id``).

    Returns
    -------
    dict
        The event record the dataset writers take, before selection.
    """
    if local_frame:
        (stream,) = streams
        event = _digitize_local_frame(stream, n_sensors=n_sensors, readout=readout, rng=rng,
                                      apply_resolution=apply_resolution)
    else:
        event = _digitize_pooled(streams, n_sensors=n_sensors, readout=readout, rng=rng,
                                 apply_resolution=apply_resolution)
    cont = compute_containment(event, detector_bounds)
    event['contained_per_segment']     = cont['per_segment']
    event['contained_per_particle']    = cont['per_particle']
    event['contained_per_interaction'] = cont['per_interaction']
    event['contained']                 = cont['overall']
    return event
