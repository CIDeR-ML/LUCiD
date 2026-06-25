"""Event view derivation, bucketed propagation, and aggregation helpers.

Contains the bucketing/JIT-warmup chain, the bucketed photon
propagation kernel wrapper, segment view derivation
(``_derive_views_from_segments``), host-side photon-record aggregation,
and track/particle derivation helpers used by the v3 writers.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from lucid.sources.segment_grouping import assign_group_ids
from lucid.sources.particle_categorization import (
    TrackEntry,
    bucket_photons_by_segment,
    categorize_event,
    derive_meaningful_tracks,
    filter_segments_to_meaningful,
    pdg_to_g4name,
)

__all__ = [
    "derive_particle_idx_per_track",
    "derive_track_ancestor_and_interaction",
    "aggregate_hits_from_segments",
]


# =============================================================================
# Single-axis bucketing (n_rays)
# =============================================================================
#
# The data-mode kernel (`_common_propagation` in lucid/simulation/simulator.py)
# JIT-caches on ``n_rays`` (size of the per-chunk photon array). Quantizing
# this axis into a small finite set of bucket sizes is enough to keep the
# JIT cache bounded: a warmup pass compiles each entry once at startup; every
# subsequent event hits the cache. Per chunk, photons are padded to the
# smallest fitting bucket; the kernel returns per-photon flat lists that the
# host aggregates downstream — no segment-axis bucketing is needed because
# the kernel no longer emits a ``(n_segments, n_sensors)`` decomposition.
_DEFAULT_PAD_SIZE_BUCKETS = (256, 2_048, 8_192, 32_768)


def _normalize_buckets(buckets):
    """Coerce a user-provided bucket spec into a sorted tuple of unique ints.

    An empty / None / falsy input is the opt-out sentinel returned as ()."""
    if not buckets:
        return ()
    return tuple(sorted({int(b) for b in buckets if int(b) > 0}))


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


def _warmup_buckets(event_simulator, rays_buckets):
    """Trigger one JIT compile per ``n_rays_bucket`` (4 entries on default spec).

    Runs the simulator once per bucket with throwaway zero-filled photons
    (``N=0``, so no real propagation work) and blocks on the result. After
    this returns, every real-event kernel call whose ``n_rays`` matches
    one of the warmed buckets hits the JIT cache and runs at native
    kernel cost. Each compile is ~10-30 s on CPU; total warmup ~80-120 s
    on the default 4-bucket spec.

    The host-side aggregator (``_aggregate_from_photon_records``) is
    pure numpy and has no JIT step, so no separate aggregator warmup is
    needed.
    """
    if not rays_buckets:
        return
    from lucid.detector_params import ParticleParams
    print(f"Warming up JIT cache for {len(rays_buckets)} rays bucket(s)...")
    track_params = ParticleParams.from_cartesian(
        energy=jnp.float32(800.0),
        position=jnp.zeros(3, dtype=jnp.float32),
        direction=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
        t0=jnp.float32(0.0),
    )
    rotation_axis = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    translation_vector = jnp.zeros(3, dtype=jnp.float32)
    default_dir = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    for rays_b in rays_buckets:
        t0_warm = time.time()
        photonsim_data = {
            'photon_origins':    jnp.zeros((rays_b, 3), dtype=jnp.float32),
            'photon_directions': jnp.tile(default_dir, (rays_b, 1)),
            'photon_times':      jnp.zeros(rays_b, dtype=jnp.float32),
            'wavelengths':       jnp.zeros(rays_b, dtype=jnp.float32),
            'N': jnp.int32(0),
            'apply_rotation': False,
            'rotation_axis': rotation_axis,
            'rotation_angle': 0.0,
            'apply_translation': False,
            'translation_vector': translation_vector,
            'photon_segment_index': jnp.full(rays_b, -1, dtype=jnp.int32),
        }
        key = jax.random.PRNGKey(0)
        result = event_simulator(track_params, key, photonsim_data)
        # Block on every leaf so the JIT trace finishes.
        for arr in result:
            arr.block_until_ready()
        print(f"  rays={rays_b:>6,}: {time.time() - t0_warm:.2f}s", flush=True)


# Vestigial track params: the kernel reads these out of ParticleParams but
# never feeds them to ``_common_propagation`` (verified at simulator.py:454-456;
# the propagation only consumes per-photon arrays). One module-level dummy
# keeps the kernel's argument shape happy without per-event allocations.
_ZERO_TRACK_PARAMS = None  # lazily built (avoids importing ParticleParams at import time)


def _get_zero_track_params():
    global _ZERO_TRACK_PARAMS
    if _ZERO_TRACK_PARAMS is None:
        from lucid.detector_params import ParticleParams
        _ZERO_TRACK_PARAMS = ParticleParams.from_cartesian(
            energy=jnp.float32(0.0),
            position=jnp.zeros(3, dtype=jnp.float32),
            direction=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
            t0=jnp.float32(0.0),
        )
    return _ZERO_TRACK_PARAMS


def _trace_event_bucketed(
        event_simulator,
        photon_origins_np, photon_directions_np,
        photon_times_np, photon_wavelengths_np,
        photon_segment_index_raw,
        n_sensors, rays_buckets,
        master_key):
    """Bucketed propagation for one event — single trace per photon.

    Drops the per-particle loop and the per-(segment, sensor) dense
    output. Per chunk, photons are padded to the smallest fitting
    ``rays_buckets`` entry and propagated in one kernel call. The kernel
    returns:

      * per-sensor totals (PE, T) — accumulated across chunks via PE-sum
        and T-min as today;
      * per-photon flat arrays (qe_weight, qe_time, sensor_idx,
        seg_idx_raw) — sliced to the active photon count and concatenated
        across chunks. The host runs the per-(segment, sensor) and
        per-(particle, sensor) groupbys downstream
        (``_aggregate_from_photon_records``).

    The JIT cache key is ``n_rays_bucket`` only — 4 entries instead of 16.
    Memory per kernel call is bounded by
    ``n_rays_bucket × (4 + 4 + 4 + 4) bytes ≈ 0.5 MB``, independent of the
    event's segment total.

    RNG: ``chunk_key = jax.random.fold_in(master_key, chunk_idx)`` — same
    keying as today, so byte-equivalent (up to float reduction order)
    given the same master_seed.

    Parameters
    ----------
    event_simulator : callable
        Built by ``setup_event_simulator(..., hit_mode='per_segment')``.
    photon_*_np : np.ndarray
        All-event flat photon arrays (already vertex-translated by caller).
    photon_segment_index_raw : np.ndarray (int32 or int64)
        ``(N_photons,)`` indices into the **raw** segment table; -1
        sentinels for photons whose segment was dropped.
    n_sensors : int
        Detector sensor count.
    rays_buckets : tuple[int, ...]
        Sorted bucket spec (output of ``_normalize_buckets``).
    master_key : jax.Array
        Per-event master RNG key.

    Returns
    -------
    pe_per_sensor        : (n_sensors,) float32
    t_per_sensor         : (n_sensors,) float32  — 0 = no hit (unsmeared)
    t_reco_per_sensor    : (n_sensors,) float32  — 0 = no hit (TTS-smeared)
    photon_qe_weight     : (factor·N_photons,) float32 — 0 for QE-failed entries
    photon_qe_time       : (factor·N_photons,) float32 — +inf for QE-failed / no hit
    photon_qe_time_reco  : (factor·N_photons,) float32 — TTS-smeared, +inf if failed
    photon_sensor_idx    : (factor·N_photons,) int32
    photon_seg_idx_raw   : (factor·N_photons,) int32   — -1 for orphan photons
    photon_global_idx    : (factor·N_photons,) int32   — global photon id (0..N-1)
                           per kernel-flat row, for downstream gather of per-photon
                           arrays (seg_idx_filtered, particle_idx) into kernel-flat
                           alignment

    ``factor = K · max_sensors_per_cell`` is the number of (propagation
    iteration, sensor cell) pairs each photon contributes to in the soft-hit
    kernel. Per-sensor PE/T (``pe_per_sensor``, ``t_per_sensor``) already
    aggregate over all factor entries via segment_sum/segment_min inside
    the kernel, so they remain ``(n_sensors,)``-shaped.
    """
    if not rays_buckets:
        raise ValueError("_trace_event_bucketed requires non-empty rays_buckets")
    rays_buckets = tuple(rays_buckets)

    N = int(photon_origins_np.shape[0])
    pe_per_sensor    = np.zeros(n_sensors, dtype=np.float32)
    t_per_sensor_inf = np.full(n_sensors, np.inf, dtype=np.float32)
    t_reco_per_sensor_inf = np.full(n_sensors, np.inf, dtype=np.float32)

    if N == 0:
        return (
            pe_per_sensor,
            np.zeros(n_sensors, dtype=np.float32),
            np.zeros(n_sensors, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    rotation_axis    = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    zero_translation = jnp.zeros(3, dtype=jnp.float32)
    track_params = _get_zero_track_params()

    po = np.ascontiguousarray(photon_origins_np,    dtype=np.float32)
    pd = np.ascontiguousarray(photon_directions_np, dtype=np.float32)
    pt = np.ascontiguousarray(photon_times_np,      dtype=np.float32)
    pw = np.ascontiguousarray(photon_wavelengths_np, dtype=np.float32)
    psi = np.ascontiguousarray(photon_segment_index_raw, dtype=np.int32)

    qe_w_chunks      = []
    qe_t_chunks      = []
    qe_t_reco_chunks = []
    sen_i_chunks     = []
    seg_i_chunks     = []
    gid_chunks       = []

    chunks = _split_into_chunks(N, rays_buckets)
    offset = 0
    for chunk_idx, bucket_size in enumerate(chunks):
        n_in_chunk = min(bucket_size, N - offset)
        sl = slice(offset, offset + n_in_chunk)

        bo = np.zeros((bucket_size, 3), dtype=np.float32)
        bd = np.zeros((bucket_size, 3), dtype=np.float32)
        bd[:] = (0.0, 0.0, 1.0)
        bt = np.zeros(bucket_size, dtype=np.float32)
        bw = np.zeros(bucket_size, dtype=np.float32)
        bo[:n_in_chunk] = po[sl]
        bd[:n_in_chunk] = pd[sl]
        bt[:n_in_chunk] = pt[sl]
        bw[:n_in_chunk] = pw[sl]

        bs = np.full(bucket_size, -1, dtype=np.int32)
        bs[:n_in_chunk] = psi[sl]

        photonsim_data = {
            'photon_origins':    jax.device_put(bo),
            'photon_directions': jax.device_put(bd),
            'photon_times':      jax.device_put(bt),
            'wavelengths':       jax.device_put(bw),
            'N': jnp.int32(n_in_chunk),
            'apply_rotation': False,
            'rotation_axis': rotation_axis,
            'rotation_angle': 0.0,
            'apply_translation': False,
            'translation_vector': zero_translation,
            'photon_segment_index': jax.device_put(bs),
        }

        chunk_key = jax.random.fold_in(master_key, chunk_idx)
        (PE_chunk, T_chunk, T_reco_chunk,
         qe_w_chunk, qe_t_chunk, qe_t_reco_chunk,
         sen_i_chunk, seg_i_chunk) = (
            event_simulator(track_params, chunk_key, photonsim_data))

        PE_chunk_np      = np.asarray(PE_chunk, dtype=np.float32)
        T_chunk_np       = np.asarray(T_chunk,  dtype=np.float32)
        T_reco_chunk_np  = np.asarray(T_reco_chunk, dtype=np.float32)
        pe_per_sensor += PE_chunk_np
        t_per_sensor_inf = np.minimum(
            t_per_sensor_inf,
            np.where(T_chunk_np > 0, T_chunk_np, np.inf),
        )
        t_reco_per_sensor_inf = np.minimum(
            t_reco_per_sensor_inf,
            np.where(T_reco_chunk_np > 0, T_reco_chunk_np, np.inf),
        )

        # The kernel returns flat arrays of length (K · max_sensors_per_cell · bucket_size)
        # — one entry per (propagation iteration, sensor cell, photon) tuple — because
        # each photon's contribution is distributed across multiple (k, sensor_cell)
        # pairs in the soft-hit model. Per-sensor PE_chunk already sums these
        # contributions internally (segment_sum on flat_indices). To produce per-
        # (particle, sensor) and per-(segment, sensor) totals that match PE_chunk,
        # the host aggregator needs the FULL flat arrays paired with broadcast
        # particle_idx / seg_idx — NOT a [:n_in_chunk] slice (which would only
        # keep the (k=0, sensor_cell=0) slab, ~1/(K·max_sensors_per_cell) of data).
        # Drop only the bucket padding (photon ids >= n_in_chunk), and emit a
        # global photon id per kept row so the host can gather per-photon arrays
        # (seg_idx_filtered, particle_idx) up to kernel-flat alignment.
        qe_w_arr      = np.asarray(qe_w_chunk,      dtype=np.float32)
        qe_t_arr      = np.asarray(qe_t_chunk,      dtype=np.float32)
        qe_t_reco_arr = np.asarray(qe_t_reco_chunk, dtype=np.float32)
        sen_i_arr     = np.asarray(sen_i_chunk,      dtype=np.int32)
        seg_i_arr     = np.asarray(seg_i_chunk,      dtype=np.int32)
        flat_len = qe_w_arr.shape[0]
        photon_axis = np.arange(flat_len, dtype=np.int64) % bucket_size
        keep = photon_axis < n_in_chunk
        qe_w_chunks     .append(qe_w_arr [keep])
        qe_t_chunks     .append(qe_t_arr [keep])
        qe_t_reco_chunks.append(qe_t_reco_arr[keep])
        sen_i_chunks    .append(sen_i_arr[keep])
        seg_i_chunks    .append(seg_i_arr[keep])
        gid_chunks      .append((photon_axis[keep] + offset).astype(np.int32))

        offset += n_in_chunk

    t_per_sensor      = np.where(np.isfinite(t_per_sensor_inf), t_per_sensor_inf, 0.0)
    t_reco_per_sensor = np.where(np.isfinite(t_reco_per_sensor_inf), t_reco_per_sensor_inf, 0.0)
    photon_qe_weight    = np.concatenate(qe_w_chunks)
    photon_qe_time      = np.concatenate(qe_t_chunks)
    photon_qe_time_reco = np.concatenate(qe_t_reco_chunks)
    photon_sensor_idx   = np.concatenate(sen_i_chunks)
    photon_seg_idx_raw  = np.concatenate(seg_i_chunks)
    photon_global_idx   = np.concatenate(gid_chunks)

    return (pe_per_sensor, t_per_sensor, t_reco_per_sensor,
            photon_qe_weight, photon_qe_time, photon_qe_time_reco,
            photon_sensor_idx, photon_seg_idx_raw,
            photon_global_idx)


def _derive_views_from_segments(raw, photon_records=None):
    """Categorize, filter, and assemble the downstream-view dict.

    This is the post-kernel half of the legacy ``read_particle_data_from_photonsim``.
    Runs the four pure helpers from ``particle_categorization.py`` plus
    ``segment_grouping.assign_group_ids`` on the raw read output, and (if
    the kernel's per-photon flat lists were supplied) builds the
    ``photon_records_filtered`` dict that the downstream host aggregator
    consumes.

    Parameters
    ----------
    raw : dict
        Output of :func:`_read_event_raw`.
    photon_records : dict or None
        Optional dict with keys ``qe_weight``, ``qe_time``, ``sensor_idx``,
        ``seg_idx_raw`` — kernel-flat arrays of shape ``(factor·N_photons,)``
        emitted by :func:`_trace_event_bucketed` — and ``photon_global_idx``
        of the same shape, mapping each kernel-flat row to the source
        photon id (0..N-1). Optionally includes ``qe_time_reco`` (TTS-
        smeared per-photon times). When provided, the function attaches a
        ``photon_records_filtered`` entry whose ``qe_weight``, ``qe_time``,
        ``qe_time_reco``, ``sensor_idx`` are the kernel-flat arrays unchanged,
        plus ``seg_idx_filtered`` and ``particle_idx`` gathered via
        ``photon_global_idx`` so all arrays are kernel-flat-aligned.
        Pass ``None`` for dark events (no kernel call).

    Returns
    -------
    dict — same shape as the legacy ``read_particle_data_from_photonsim``
    output, plus ``photon_records_filtered`` (``None`` when no records
    were supplied).
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

    # Plumb category / sub_id back into track_info_dict so save_labl_event_v3
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
    if n_particles > 0 and photon_to_particle.size > 0:
        # Vectorized partition: photons whose particle_idx >= 0 are sorted
        # by particle_idx; np.searchsorted on the sorted column gives the
        # inclusive boundaries for each particle's photon-id slice.
        valid = photon_to_particle >= 0
        valid_ph_idx = np.flatnonzero(valid)
        valid_p_idx  = photon_to_particle[valid].astype(np.int64, copy=False)
        order = np.argsort(valid_p_idx, kind='stable')
        sorted_p  = valid_p_idx[order]
        sorted_ph = valid_ph_idx[order]
        boundaries = np.searchsorted(sorted_p, np.arange(n_particles + 1))
        photons_per_particle = [
            sorted_ph[boundaries[i]:boundaries[i + 1]].tolist()
            for i in range(n_particles)
        ]
    else:
        photons_per_particle = [[] for _ in range(n_particles)]

    particles = []
    for p_idx in range(n_particles):
        genealogy = list(cat_result.genealogies[p_idx])
        ext_genealogy = list(cat_result.ext_genealogies[p_idx])
        last_track_id = genealogy[-1] if genealogy else None
        track_info = track_info_dict.get(last_track_id) if last_track_id is not None else None
        particles.append({
            'genealogy': genealogy,
            'extended_genealogy': ext_genealogy,
            'photon_indices': photons_per_particle[p_idx],
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

    # ---- Build photon_records_filtered for the host aggregator ----
    # ``filter_segments_to_meaningful`` produced ``photon_segment_index`` in
    # the **filtered** segment space (length N); ``bucket_photons_by_segment``
    # gave us the per-photon particle index (length N). Both carry the -1
    # sentinel for orphans, which is what the host aggregator's QE-pass mask
    # needs. The kernel-flat arrays in ``photon_records`` have one row per
    # (propagation iteration, sensor cell, photon) tuple — length factor·N
    # — so we gather the per-photon arrays via ``photon_global_idx`` to bring
    # all five arrays to a common kernel-flat alignment for the groupbys.
    if photon_records is not None:
        gid = np.asarray(photon_records['photon_global_idx'], dtype=np.int64)
        photon_records_filtered = {
            'qe_weight':        photon_records['qe_weight'],
            'qe_time':          photon_records['qe_time'],
            'sensor_idx':       photon_records['sensor_idx'],
            'seg_idx_filtered': (photon_segment_index[gid]
                                 if photon_segment_index.size
                                 else photon_segment_index).astype(np.int32, copy=False),
            'particle_idx':     (photon_to_particle[gid]
                                 if photon_to_particle.size
                                 else photon_to_particle).astype(np.int32, copy=False),
        }
        if 'qe_time_reco' in photon_records:
            photon_records_filtered['qe_time_reco'] = photon_records['qe_time_reco']
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


def derive_particle_idx_per_track(event_dict):
    """Map each meaningful track to the local index of its owning particle.

    Walks the track's ``parent_id`` chain until reaching the ``track_id`` of
    a categorized particle (the last entry of that particle's
    ``genealogy``). Orphaned tracks (no categorized ancestor found) get -1.

    Returns
    -------
    np.ndarray (int32) shape (n_tracks,)
    """
    tracks = event_dict.get('meaningful_tracks', {})
    particles = event_dict.get('particles', [])

    id_to_idx = {}
    for i, particle in enumerate(particles):
        gen = particle.get('genealogy') or []
        if gen:
            id_to_idx[int(gen[-1])] = i

    out = np.full(len(tracks), -1, dtype=np.int32)
    for row, tinfo in enumerate(tracks.values()):
        cur = int(tinfo['track_id'])
        visited = set()
        while cur > 0 and cur not in visited:
            visited.add(cur)
            if cur in id_to_idx:
                out[row] = id_to_idx[cur]
                break
            parent = tracks.get(cur)
            if parent is None:
                break
            cur = int(parent['parent_id'])
    return out


def run_event_process_pipeline(
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
    """Run kernel + derive_views + aggregator for **one** photon stream.

    The single-vertex and pile-up data-mode drivers both call this once per
    emission process when scintillation is active (Option C: one kernel call
    per process, see ``.claude/plans/scintillation-data-mode.md``). For
    Cherenkov-only events it's called exactly once.

    Returns a dict with everything needed to combine across processes:
      * per-sensor kernel accumulators (PE / T / T_reco) — used to compute
        the process-combined ``PE_true`` / ``T_true`` at the writer boundary;
      * per-particle dense PE/T tensors from the aggregator — sparsified
        with an ``emission_process`` tag at writer time;
      * the sparse ``segment_sensor_hits`` triplet dict from the aggregator —
        the caller concatenates per-process dicts with a per-row
        ``emission_process`` column on the way to ``save_step_event_v3``;
      * the full ``particle_data`` view (categorization, segments, etc.) —
        the caller uses the first process's view as the canonical event-
        level structure (the segment table is identical across processes).
    """
    (pe_per_sensor_np, t_per_sensor_np, t_reco_per_sensor_np,
     photon_qe_w, photon_qe_t, photon_qe_t_reco,
     photon_sen_i, photon_seg_i_raw, photon_gid) = _trace_event_bucketed(
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
        'seg_idx_raw':       photon_seg_i_raw,
        'photon_global_idx': photon_gid,
    })

    pr = particle_data['photon_records_filtered']
    agg = _aggregate_from_photon_records(
        pr['qe_weight'], pr['qe_time'], pr['sensor_idx'],
        pr['seg_idx_filtered'], pr['particle_idx'],
        n_particles=particle_data['n_particles'], n_sensors=n_sensors,
        photon_qe_time_reco=pr.get('qe_time_reco'))

    return {
        'pe_per_sensor':       pe_per_sensor_np,
        't_per_sensor':        t_per_sensor_np,
        't_reco_per_sensor':   t_reco_per_sensor_np,
        'PE_per_particle':     agg['PE_per_particle'],
        'T_per_particle':      agg['T_per_particle'],
        'T_reco_per_particle': agg['T_reco_per_particle'],
        'segment_sensor_hits': agg['segment_sensor_hits'],
        'particle_data':       particle_data,
    }


def combine_t_per_sensor_across_processes(t_arrays):
    """Min over processes with the 0-as-no-hit sentinel convention.

    Each per-process ``t_per_sensor`` array (shape ``(n_sensors,)``) carries
    0 where no photon hit that sensor and a real arrival time elsewhere.
    To combine into a single per-sensor first-arrival across processes,
    map 0 → +inf, take elementwise min, map +inf → 0.
    """
    if not t_arrays:
        return np.zeros(0, dtype=np.float32)
    masked = [np.where(np.asarray(t) > 0, t, np.inf) for t in t_arrays]
    combined = np.minimum.reduce(masked)
    return np.where(np.isfinite(combined), combined, 0.0).astype(np.float32)


def build_hits_sparse_per_process(process_outputs, n_particles):
    """Sparse-merge per-process dense PE/T tensors into hits-file rows.

    For each emission process and each particle row of its dense
    ``(n_particles, n_sensors)`` ``PE_per_particle`` tensor, extract the
    sensors where PE > 0 and emit one row per ``(particle, sensor)``
    pair tagged with the process's ``process_id``. The same
    ``(particle_idx, sensor_idx)`` pair can therefore appear in multiple
    rows differing only in ``emission_process``.

    Returns a dict shaped for ``save_hits_event_v3``'s ``hits_sparse``
    input path: 1-D arrays for ``particle_idx`` / ``sensor_idx`` / ``PE``
    / ``T`` / ``T_reco`` / ``emission_process``, all the same length.
    """
    particle_idx_parts, sensor_idx_parts = [], []
    pe_parts, t_parts, t_reco_parts, emp_parts = [], [], [], []
    for p_out in process_outputs:
        pe_pp = np.asarray(p_out['PE_per_particle'], dtype=np.float32)
        t_pp = np.asarray(p_out['T_per_particle'], dtype=np.float32)
        t_reco_pp = np.asarray(p_out['T_reco_per_particle'], dtype=np.float32)
        proc_id = int(p_out['process_id'])
        for i in range(n_particles):
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
            t_reco_vals = t_reco_pp[i, mask]
            t_reco_vals = np.where(
                np.isfinite(t_reco_vals), t_reco_vals, np.float32(0.0))
            t_reco_parts.append(t_reco_vals.astype(np.float32))
            emp_parts.append(np.full(idx.shape[0], proc_id, dtype=np.int8))

    def _cat(xs, dt):
        return np.concatenate(xs).astype(dt) if xs else np.array([], dtype=dt)

    return {
        'particle_idx':     _cat(particle_idx_parts, np.int32),
        'sensor_idx':       _cat(sensor_idx_parts,   np.uint16),
        'PE':               _cat(pe_parts,           np.float32),
        'T':                _cat(t_parts,            np.float32),
        'T_reco':           _cat(t_reco_parts,       np.float32),
        'emission_process': _cat(emp_parts,          np.int8),
    }


def build_seg_hits_merged_per_process(process_outputs):
    """Concat per-process ``segment_sensor_hits`` triplets with a process tag.

    Each per-process aggregator output carries a sparse
    ``{segment_idx, sensor_idx, PE, T, T_reco?}`` dict. Concatenate them
    along axis=0 and add an ``emission_process`` column whose value is the
    source process_id of each row. Returned dict drops directly into
    ``event_dict['segment_sensor_hits']`` for ``save_step_event_v3``.
    """
    seg_idx_parts, sensor_idx_parts = [], []
    pe_parts, t_parts, t_reco_parts, emp_parts = [], [], [], []
    has_t_reco_any = False
    for p_out in process_outputs:
        sh = p_out['segment_sensor_hits']
        if sh['PE'].size == 0:
            continue
        seg_idx_parts.append(sh['segment_idx'])
        sensor_idx_parts.append(sh['sensor_idx'])
        pe_parts.append(sh['PE'])
        t_parts.append(sh['T'])
        if 'T_reco' in sh:
            t_reco_parts.append(sh['T_reco'])
            has_t_reco_any = True
        else:
            t_reco_parts.append(sh['T'])
        emp_parts.append(np.full(
            sh['PE'].shape[0], int(p_out['process_id']), dtype=np.int8))

    def _cat(xs, dt):
        return np.concatenate(xs).astype(dt) if xs else np.array([], dtype=dt)

    out = {
        'segment_idx':      _cat(seg_idx_parts,    np.int32),
        'sensor_idx':       _cat(sensor_idx_parts, np.uint16),
        'PE':               _cat(pe_parts,         np.float32),
        'T':                _cat(t_parts,          np.float32),
        'emission_process': _cat(emp_parts,        np.int8),
    }
    if has_t_reco_any:
        out['T_reco'] = _cat(t_reco_parts, np.float32)
    return out


def _aggregate_from_photon_records(
        photon_qe_weight,
        photon_qe_time,
        photon_sensor_idx,
        photon_seg_idx_filtered,
        photon_particle_idx,
        n_particles, n_sensors,
        photon_qe_time_reco=None):
    """One-pass host aggregation from per-photon flat lists.

    Replaces the dense ``(n_segments, n_sensors)`` PE/T tensors plus the
    JIT inst aggregator plus the ``np.nonzero(PE_seg)`` sparsifier with a
    single numpy lexsort+reduceat pass. Two groupbys, both keyed by
    ``(group, sensor_idx)``:

      * ``(particle_idx, sensor_idx)`` -> dense ``(n_particles, n_sensors)``
        ``PE_per_particle``, ``T_per_particle``, and ``T_reco_per_particle``
        for the hits file;
      * ``(seg_idx_filtered, sensor_idx)`` -> sparse triplets
        ``{segment_idx, sensor_idx, PE, T, T_reco}`` for the step file
        (``segment_sensor_hits``).

    PE per group is the sum of QE-passing photon weights; T per group is
    the min of unsmeared QE-filtered arrival times; T_reco per group is
    the min of TTS-smeared QE-filtered arrival times. ``qe_weight > 0`` is
    the QE-pass mask (failed photons have weight 0 and time +inf from
    ``_qe_roll``). Orphans (``-1``) drop out.

    Returns
    -------
    dict with keys 'PE_per_particle', 'T_per_particle',
    'T_reco_per_particle', 'segment_sensor_hits' (the latter is
    ``{'segment_idx', 'sensor_idx', 'PE', 'T', 'T_reco'}`` arrays).
    When no QE-passing photon points at a given group axis, returns
    zero-filled / empty outputs respectively.
    """
    has_reco = photon_qe_time_reco is not None
    PE_pp      = np.zeros((n_particles, n_sensors), dtype=np.float32)
    T_pp       = np.zeros((n_particles, n_sensors), dtype=np.float32)
    T_reco_pp  = np.zeros((n_particles, n_sensors), dtype=np.float32)
    seg_hits = {
        'segment_idx': np.empty(0, dtype=np.int32),
        'sensor_idx':  np.empty(0, dtype=np.uint16),
        'PE':          np.empty(0, dtype=np.float32),
        'T':           np.empty(0, dtype=np.float32),
        'T_reco':      np.empty(0, dtype=np.float32),
    }

    if photon_qe_weight.size == 0:
        return {'PE_per_particle': PE_pp, 'T_per_particle': T_pp,
                'T_reco_per_particle': T_reco_pp,
                'segment_sensor_hits': seg_hits}

    qe_pass = photon_qe_weight > 0

    # ---- hits: groupby (particle_idx, sensor_idx) ----
    p_mask = qe_pass & (photon_particle_idx >= 0)
    if n_particles > 0 and p_mask.any():
        pi = photon_particle_idx[p_mask].astype(np.int64)
        si = photon_sensor_idx[p_mask].astype(np.int64)
        w  = photon_qe_weight[p_mask]
        t  = photon_qe_time[p_mask]
        # Lexsort by (pi, si) — primary pi, secondary si.
        order = np.lexsort((si, pi))
        pi_s = pi[order]; si_s = si[order]
        w_s  = w[order];  t_s  = t[order]
        composite = pi_s * np.int64(n_sensors) + si_s
        change = np.empty(composite.size, dtype=bool)
        change[0] = True
        change[1:] = composite[1:] != composite[:-1]
        starts = np.flatnonzero(change)
        PE_groups = np.add.reduceat(w_s, starts)
        T_groups  = np.minimum.reduceat(t_s, starts)
        gp = pi_s[starts]; gs = si_s[starts]
        PE_pp[gp, gs] = PE_groups
        # Drop +inf cells back to 0 (no QE-passing photon).
        finite = np.isfinite(T_groups)
        if finite.any():
            T_pp[gp[finite], gs[finite]] = T_groups[finite]
        # T_reco parallel min-reduce.
        if has_reco:
            tr = photon_qe_time_reco[p_mask]
            tr_s = tr[order]
            T_reco_groups = np.minimum.reduceat(tr_s, starts)
            finite_r = np.isfinite(T_reco_groups)
            if finite_r.any():
                T_reco_pp[gp[finite_r], gs[finite_r]] = T_reco_groups[finite_r]

    # ---- step sparse triplets: groupby (seg_idx_filtered, sensor_idx) ----
    s_mask = qe_pass & (photon_seg_idx_filtered >= 0)
    if s_mask.any():
        seg = photon_seg_idx_filtered[s_mask].astype(np.int64)
        si  = photon_sensor_idx[s_mask].astype(np.int64)
        w   = photon_qe_weight[s_mask]
        t   = photon_qe_time[s_mask]
        order = np.lexsort((si, seg))
        seg_s = seg[order]; si_s = si[order]
        w_s   = w[order];   t_s  = t[order]
        composite = seg_s * np.int64(n_sensors) + si_s
        change = np.empty(composite.size, dtype=bool)
        change[0] = True
        change[1:] = composite[1:] != composite[:-1]
        starts = np.flatnonzero(change)
        PE_g = np.add.reduceat(w_s, starts).astype(np.float32)
        T_g  = np.minimum.reduceat(t_s, starts).astype(np.float32)
        seg_hits = {
            'segment_idx': seg_s[starts].astype(np.int32),
            'sensor_idx':  si_s[starts].astype(np.uint16),
            'PE':          PE_g,
            'T':           T_g,
        }
        if has_reco:
            tr = photon_qe_time_reco[s_mask]
            tr_s = tr[order]
            T_reco_g = np.minimum.reduceat(tr_s, starts).astype(np.float32)
            seg_hits['T_reco'] = T_reco_g
        else:
            seg_hits['T_reco'] = T_g.copy()

    return {'PE_per_particle': PE_pp, 'T_per_particle': T_pp,
            'T_reco_per_particle': T_reco_pp,
            'segment_sensor_hits': seg_hits}


def aggregate_hits_from_segments(pe_per_seg, t_per_seg,
                                  track_idx_per_segment,
                                  particle_idx_per_track,
                                  n_particles, n_sensors):
    """Aggregate per-(segment, sensor) PE/T into per-particle PE/T.

    The hits file's per-particle decomposition is a downstream view of
    ``step/event_NNN/sensor_hits/`` plus the segment->track->particle map.
    PE per particle is the sum over the particle's segments; T per
    particle is the min over the particle's segments' first-arrival
    times.

    Kept as the byte-identity oracle for ``test_aggregator_matches_oracle``
    — the production single-vertex / pile-up paths use
    ``_aggregate_from_photon_records`` instead, which works directly from
    the per-photon flat lists.

    The "no hit = 0" sentinel pattern is preserved exactly: 0 in the
    input means "no photons hit this (seg, sensor)"; we route
    0 -> +inf -> segment_min -> back to 0.

    Parameters
    ----------
    pe_per_seg : (n_segments, n_sensors) float32
        PE_total per (segment, sensor). 0 = no hit.
    t_per_seg : (n_segments, n_sensors) float32
        First-arrival time per (segment, sensor). 0 = no hit.
    track_idx_per_segment : (n_segments,) int32
        Local track index per segment (0..n_tracks-1).
    particle_idx_per_track : (n_tracks,) int32
        Particle index per track; -1 for tracks without a categorized
        ancestor (their segments are dropped from the hits file).
    n_particles, n_sensors : int

    Returns
    -------
    PE_per_particle : (n_particles, n_sensors) float32
    T_per_particle  : (n_particles, n_sensors) float32 — 0 = no hit.
    """
    PE_pp = np.zeros((n_particles, n_sensors), dtype=np.float32)
    T_pp = np.zeros((n_particles, n_sensors), dtype=np.float32)
    if n_particles == 0:
        return PE_pp, T_pp

    n_segments = int(pe_per_seg.shape[0])
    if n_segments == 0:
        return PE_pp, T_pp

    if track_idx_per_segment.size != n_segments:
        raise ValueError(
            f"track_idx_per_segment length {track_idx_per_segment.size} "
            f"!= n_segments {n_segments}")

    # Fast path. Two stages:
    #   1. Per-track aggregation via reduceat over contiguous track slices
    #      (segments are filled in track-insertion order, so
    #      track_idx_per_segment is monotonic non-decreasing — see the
    #      caller in generate_events_from_photonsim_particles).
    #   2. Per-particle routing on the much smaller per-track tensor via
    #      np.add.at / np.minimum.at. n_tracks << n_segments so the
    #      unbuffered scatter is now cheap.
    # ``np.unique`` on a sorted input returns the unique values and their
    # first-occurrence indices in one pass — exactly the boundaries
    # reduceat consumes.
    unique_tracks, track_starts = np.unique(track_idx_per_segment, return_index=True)

    PE_pt = np.add.reduceat(pe_per_seg, track_starts, axis=0)
    # Min ignoring zeros (the "no hit" sentinel): substitute +inf for
    # zeros, reduceat min, then any output cell still at +inf gets sent
    # back to 0 in the per-particle stage below.
    t_inf = np.where(t_per_seg > 0, t_per_seg, np.float32(np.inf))
    T_pt_inf = np.minimum.reduceat(t_inf, track_starts, axis=0)

    # Route per-track -> per-particle, dropping orphaned tracks (-1).
    particle_idx_per_unique_track = particle_idx_per_track[unique_tracks]
    valid_t = particle_idx_per_unique_track >= 0
    if not valid_t.any():
        return PE_pp, T_pp
    valid_pidx_t = particle_idx_per_unique_track[valid_t].astype(np.int64)

    np.add.at(PE_pp, valid_pidx_t, PE_pt[valid_t])
    T_pp_inf = np.full((n_particles, n_sensors), np.inf, dtype=np.float32)
    np.minimum.at(T_pp_inf, valid_pidx_t, T_pt_inf[valid_t])
    T_pp = np.where(np.isfinite(T_pp_inf), T_pp_inf, np.float32(0.0))
    return PE_pp, T_pp


def derive_track_ancestor_and_interaction(event_dict):
    """For each meaningful track, derive (ancestor_track_id, interaction_id).

    * ``ancestor_track_id`` is the root of the parent chain — the primary
      this track descends from. A track that is itself a primary
      (``parent_id == 0``) is its own ancestor. This column is the
      authoritative ancestry lookup and is **unchanged** across schema
      versions.
    * ``interaction_id`` is the **interaction index** the track's primary
      belongs to, read from ``event_dict['primary_to_interaction']``:
      always 0 for non-pile-up events (one interaction), 0..N-1 for
      N-way pile-up events (one interaction per vertex stream). All
      primaries that came from the same G4 event / vertex share the
      same interaction_id, so a multi-primary GENIE interaction or a
      multi-primary particle-gun shot collapses to a single
      ``per_interaction`` row.

    Returns
    -------
    (ancestor, interaction) : tuple of np.ndarray (int32, int32)
        Both shape ``(n_tracks,)``.
    """
    tracks = event_dict.get('meaningful_tracks', {})
    if not tracks:
        empty = np.array([], dtype=np.int32)
        return empty, empty.copy()

    parent_of = {int(tid): int(t['parent_id']) for tid, t in tracks.items()}

    def walk_to_root(tid):
        cur = tid
        visited = set()
        while cur > 0 and cur not in visited:
            visited.add(cur)
            parent = parent_of.get(cur, 0)
            if parent == 0 or parent not in parent_of:
                return cur
            cur = parent
        return cur

    ancestors = np.array(
        [walk_to_root(int(t['track_id'])) for t in tracks.values()],
        dtype=np.int32)

    primary_to_interaction = event_dict['primary_to_interaction']
    interaction = np.array([int(primary_to_interaction[int(a)]) for a in ancestors],
                           dtype=np.int32)
    return ancestors, interaction


