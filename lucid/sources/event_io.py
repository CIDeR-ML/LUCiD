"""ROOT I/O and event generation functions.

Moved from lucid/generate.py during Phase 2.2 refactor.
Additional I/O and event analysis functions moved from lucid/utils.py
during Phase 2.5 refactor.
"""

import jax
import jax.numpy as jnp
import numpy as np
import h5py
import os
import time
from functools import partial
from glob import glob as _glob
from tqdm import tqdm
from lucid.wavelength import DEFAULT_WAVELENGTH_NM
from lucid.sources.segment_grouping import assign_group_ids
from lucid.sources.particle_categorization import (
    TrackEntry,
    bucket_photons_by_segment,
    categorize_event,
    derive_meaningful_tracks,
    filter_segments_to_meaningful,
    pdg_to_g4name,
)


# Tag constants used with jax.random.fold_in so each subprocess stream in
# the seed hierarchy gets a distinct derivation. Value is arbitrary — it
# just has to be stable and distinct across tags.
_SUBPROC_PHOTONSIM_TAG = 0xB107
_SUBPROC_GENIE_TAG     = 0x6E1E


def _resolve_master_seed(master_seed):
    """Return a deterministic int seed, drawing from time if master_seed is None."""
    if master_seed is None:
        return int(time.time() * 1_000_000) % (2 ** 31 - 1)
    return int(master_seed) % (2 ** 31 - 1)


def derive_event_keys(master_seed, job_id, event_idx, interaction_idx=0):
    """Derive independent RNG keys for one (job, event, interaction) step.

    Combines ``master_seed``, ``job_id``, ``event_idx`` and
    ``interaction_idx`` via ``jax.random.fold_in`` so every dimension is
    independent — reusing a CLI seed across jobs no longer collides, and
    pile-up interactions within one event get distinct draws.

    Returns a dict with ``vertex_seed`` / ``t0_seed`` (concrete ints for
    ``np.random.default_rng``) and ``sim_key`` / ``smear_key`` (JAX keys
    to be consumed directly by ``jax.random.*``).
    """
    master_seed = _resolve_master_seed(master_seed)
    base = jax.random.PRNGKey(master_seed)
    job_key = jax.random.fold_in(base, int(job_id))
    event_key = jax.random.fold_in(job_key, int(event_idx))
    interaction_key = jax.random.fold_in(event_key, int(interaction_idx))
    vertex_key, t0_key, sim_key, smear_key = jax.random.split(interaction_key, 4)
    return {
        'vertex_seed': int(jax.random.randint(vertex_key, (), 1, 2**31 - 1)),
        't0_seed':     int(jax.random.randint(t0_key,     (), 1, 2**31 - 1)),
        'sim_key':     sim_key,
        'smear_key':   smear_key,
    }


def derive_subprocess_seeds(master_seed, job_id, vertex_idx=0):
    """Derive deterministic seeds for the per-job subprocesses (GENIE, PhotonSim).

    Subprocess seeds are folded at the (master_seed, job_id, vertex_idx)
    level — not per-event — because each subprocess produces all
    ``n_events`` internally and drives its own per-event RNG. The
    ``vertex_idx`` axis exists so pile-up configurations with N
    PhotonSim/GENIE streams per event get independent seeds per stream.

    PhotonSim's Geant4/CLHEP engine needs two seeds (`/random/setSeeds
    s1 s2`); GENIE's gevgen takes one.
    """
    master_seed = _resolve_master_seed(master_seed)
    base = jax.random.PRNGKey(master_seed)
    job_key = jax.random.fold_in(base, int(job_id))
    vertex_key = jax.random.fold_in(job_key, int(vertex_idx))
    genie_key = jax.random.fold_in(vertex_key, _SUBPROC_GENIE_TAG)
    ps_root = jax.random.fold_in(vertex_key, _SUBPROC_PHOTONSIM_TAG)
    ps_key1, ps_key2 = jax.random.split(ps_root, 2)
    return {
        'genie_seed':      int(jax.random.randint(genie_key, (), 1, 2**31 - 1)),
        'photonsim_seed1': int(jax.random.randint(ps_key1,   (), 1, 2**31 - 1)),
        'photonsim_seed2': int(jax.random.randint(ps_key2,   (), 1, 2**31 - 1)),
    }


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
    kernel cost. Each compile is ~10–30 s on CPU; total warmup ~80–120 s
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
    pe_per_sensor       : (n_sensors,) float32
    t_per_sensor        : (n_sensors,) float32  — 0 = no hit
    photon_qe_weight    : (N_photons,) float32  — 0 for QE-failed photons
    photon_qe_time      : (N_photons,) float32  — +inf for QE-failed / no hit
    photon_sensor_idx   : (N_photons,) int32
    photon_seg_idx_raw  : (N_photons,) int32    — -1 for orphan photons
    """
    if not rays_buckets:
        raise ValueError("_trace_event_bucketed requires non-empty rays_buckets")
    rays_buckets = tuple(rays_buckets)

    N = int(photon_origins_np.shape[0])
    pe_per_sensor    = np.zeros(n_sensors, dtype=np.float32)
    t_per_sensor_inf = np.full(n_sensors, np.inf, dtype=np.float32)

    if N == 0:
        return (
            pe_per_sensor,
            np.zeros(n_sensors, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
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

    qe_w_chunks  = []
    qe_t_chunks  = []
    sen_i_chunks = []
    seg_i_chunks = []

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
        PE_chunk, T_chunk, qe_w_chunk, qe_t_chunk, sen_i_chunk, seg_i_chunk = (
            event_simulator(track_params, chunk_key, photonsim_data))

        PE_chunk_np = np.asarray(PE_chunk, dtype=np.float32)
        T_chunk_np  = np.asarray(T_chunk,  dtype=np.float32)
        pe_per_sensor += PE_chunk_np
        t_per_sensor_inf = np.minimum(
            t_per_sensor_inf,
            np.where(T_chunk_np > 0, T_chunk_np, np.inf),
        )

        # Slice per-photon arrays to the chunk's active photon count
        # (drop bucket padding — those slots have qe_weight=0 from the
        # kernel's mask-driven QE roll, but truncating keeps the host's
        # photon-id alignment to the input.).
        qe_w_chunks.append(np.asarray(qe_w_chunk, dtype=np.float32)[:n_in_chunk])
        qe_t_chunks.append(np.asarray(qe_t_chunk, dtype=np.float32)[:n_in_chunk])
        sen_i_chunks.append(np.asarray(sen_i_chunk, dtype=np.int32)[:n_in_chunk])
        seg_i_chunks.append(np.asarray(seg_i_chunk, dtype=np.int32)[:n_in_chunk])

        offset += n_in_chunk

    t_per_sensor = np.where(np.isfinite(t_per_sensor_inf), t_per_sensor_inf, 0.0)
    photon_qe_weight   = np.concatenate(qe_w_chunks)
    photon_qe_time     = np.concatenate(qe_t_chunks)
    photon_sensor_idx  = np.concatenate(sen_i_chunks)
    photon_seg_idx_raw = np.concatenate(seg_i_chunks)

    return (pe_per_sensor, t_per_sensor,
            photon_qe_weight, photon_qe_time,
            photon_sensor_idx, photon_seg_idx_raw)


def _read_photons_for_event(raw_tree, event_idx):
    """Stitch OpticalPhotonsRaw chunks for one event into flat numpy arrays.

    OpticalPhotonsRaw stores per-photon scalars as fixed-K chunks (one
    TTree entry = one chunk of up to 100 k photons). Each entry has
    EventID + ChunkStartID stamped on it. This helper returns the
    per-event flat (NPhotons, 3)/(NPhotons,) arrays the rest of LUCiD
    consumes — units converted (PhotonSim mm → LUCiD m); chunk concat
    is in ascending ChunkStartID order so global photon IDs line up
    with Particle_PhotonIDsData.

    Memory: at 30 M photons / event this materializes ~1 GB of
    photon arrays in numpy. PhotonSim's streaming bound only applies
    to the simulator side; LUCiD still processes one event at a time.
    """
    import numpy as np

    eids = raw_tree['EventID'].array(library='np')
    chunk_start_ids = raw_tree['ChunkStartID'].array(library='np')
    mask = (eids == event_idx)
    matching = np.flatnonzero(mask)
    if matching.size == 0:
        # Empty event (no photons emitted). Return zero-length arrays.
        empty3 = np.zeros((0, 3), dtype=np.float32)
        empty1 = np.zeros((0,), dtype=np.float32)
        return empty3, empty3, empty1, empty1

    # Sort by ChunkStartID so chunks concatenate in global-id order.
    matching = matching[np.argsort(chunk_start_ids[matching])]
    entry_lo, entry_hi = int(matching.min()), int(matching.max())
    # Bulk-read the contiguous entry range (matching is contiguous in
    # practice — chunks for one event are written together — but the
    # sort+range approach is robust to ordering).
    chunk_data = raw_tree.arrays(
        ['PhotonPosX', 'PhotonPosY', 'PhotonPosZ',
         'PhotonDirX', 'PhotonDirY', 'PhotonDirZ',
         'PhotonTime', 'PhotonWavelength'],
        entry_start=entry_lo, entry_stop=entry_hi + 1, library='np',
    )
    # Re-index by `matching - entry_lo` to honor the sorted order.
    rel = matching - entry_lo

    posx = np.concatenate([chunk_data['PhotonPosX'][i] for i in rel]).astype(np.float32, copy=False) / 1000.0
    posy = np.concatenate([chunk_data['PhotonPosY'][i] for i in rel]).astype(np.float32, copy=False) / 1000.0
    posz = np.concatenate([chunk_data['PhotonPosZ'][i] for i in rel]).astype(np.float32, copy=False) / 1000.0
    dirx = np.concatenate([chunk_data['PhotonDirX'][i] for i in rel]).astype(np.float32, copy=False)
    diry = np.concatenate([chunk_data['PhotonDirY'][i] for i in rel]).astype(np.float32, copy=False)
    dirz = np.concatenate([chunk_data['PhotonDirZ'][i] for i in rel]).astype(np.float32, copy=False)
    times = np.concatenate([chunk_data['PhotonTime'][i] for i in rel]).astype(np.float32, copy=False)
    wls = np.concatenate([chunk_data['PhotonWavelength'][i] for i in rel]).astype(np.float32, copy=False)

    photon_positions = np.column_stack((posx, posy, posz))
    photon_directions = np.column_stack((dirx, diry, dirz))
    return photon_positions, photon_directions, times, wls


def get_max_photons_per_particle(root_file_path, n_events=None):
    """
    Return an upper bound on the number of photons in any single particle.

    Post-Stage-5a, PhotonSim no longer emits a per-particle photon-count
    branch (``Particle_PhotonIDsSize`` is gone). The per-event total
    ``NOpticalPhotons`` is a safe upper bound (one particle can't carry
    more photons than the entire event), and that's all PAD_SIZE needs
    — oversizing the JAX kernel padding is harmless, undersizing isn't.

    Parameters
    ----------
    root_file_path : str
        Path to the PhotonSim ROOT file
    n_events : int, optional
        Number of events to scan. If None, scans all events.

    Returns
    -------
    int
        Upper bound on photons per particle (= max NOpticalPhotons over
        the scanned events).
    """
    import uproot

    root_file = uproot.open(root_file_path)
    tree = root_file['OpticalPhotons']
    num_entries = tree.num_entries
    entry_stop = min(n_events, num_entries) if n_events is not None else num_entries

    n_photons_per_event = tree['NOpticalPhotons'].array(
        entry_start=0, entry_stop=entry_stop, library='np')
    max_photons = int(n_photons_per_event.max()) if n_photons_per_event.size > 0 else 0

    root_file.close()
    return max_photons


def generate_events_from_root(event_simulator, root_file_path, output_dir='events', n_events=None,
                            n_rings=1, pion_root_file_path=None,
                            sensor_params=None, max_sensors_per_cell=4, batch_size=100):
    """
    Generate and save events from a ROOT file, with support for N rings of particles.
    Ring 1 (N=1) is always a muon, and additional rings (N>1) are pions.
    Events are saved with sequential numbering: event_0.h5, event_1.h5, etc.

    Parameters
    ----------
    event_simulator : function
        The event simulation function to use
    root_file_path : str
        Path to the ROOT file for muons
    output_dir : str, optional
        Directory to save output files, by default 'events'
    n_events : int, optional
        Number of events to process (None for all), by default None
    n_rings : int, optional
        Number of rings (particles) to superimpose, by default 1
        First ring is always a muon, additional rings are pions
    pion_root_file_path : str, optional
        Path to ROOT file for pions, required if n_rings > 1, by default None
    sensor_params : tuple, optional
        Sensor parameters tuple passed to event_simulator, by default None
    max_sensors_per_cell : int, optional
        Maximum sensors per cell, by default 4
    batch_size : int, optional
        Number of events to accumulate before saving in parallel, by default 100

    Returns
    -------
    list
        List of saved file paths
    """
    import uproot
    import concurrent.futures
    from lucid.sources.calibration_sources import generate_random_direction, generate_random_vertex
    from lucid.utils import superimpose_multiple_events

    # Validate arguments
    if n_rings < 1:
        raise ValueError("n_rings must be at least 1")

    from lucid.detector_params import ParticleParams
    # If n_rings > 1, we need a pion ROOT file
    if n_rings > 1 and pion_root_file_path is None:
        raise ValueError("When n_rings > 1, pion_root_file_path must be provided")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open ROOT file to get number of entries
    root_file = uproot.open(root_file_path)
    tree = root_file['v_photon']
    total_entries = tree.num_entries

    if n_events is None:
        n_events = total_entries
    else:
        n_events = min(n_events, total_entries)

    # Prepare descriptor for printing
    ring_description = f"{n_rings} ring{'s' if n_rings > 1 else ''}"
    particle_description = "muon" if n_rings == 1 else f"muon + {n_rings-1} pion{'s' if n_rings > 1 else ''}"

    print(f"Processing {n_events} events with {ring_description} ({particle_description})...")
    print(f"Using batch size of {batch_size} events for multithreaded I/O")
    print(f"Saving events to directory: {output_dir}")

    saved_files = []

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
        batch_params = []
        batch_filenames = []
        batch_indices = []

        # Process each entry in the current batch
        for i in tqdm(range(start_idx, end_idx), desc=f"Generating batch {batch_idx+1}", unit="event"):
            # Initialize master random key for this event
            master_key = jax.random.PRNGKey(i * 1000)

            # Generate a random vertex for all events in this iteration
            vertex_key, master_key = jax.random.split(master_key)
            shared_vertex = generate_random_vertex(vertex_key)

            # Lists to store charges and times for all rings
            all_charges = []
            all_times = []
            all_energies = []
            all_directions = []
            all_indices = []

            # Process the first ring - always a muon
            muon_data = read_photon_data_from_root(root_file_path, i, 'muon')

            # Set up parameters
            muon_energy = muon_data['energy']

            # Generate random direction for muon
            dir_key, master_key = jax.random.split(master_key)
            muon_direction = generate_random_direction(dir_key)

            # Create parameters for muon
            track_params = ParticleParams.from_cartesian(
                energy=muon_energy,
                position=shared_vertex,
                direction=muon_direction,
                t0=0.0,
            )

            # Get a key for the muon simulation
            sim_key, master_key = jax.random.split(master_key)

            # Process muon data
            photon_origins = muon_data['photon_origins']
            photon_directions = muon_data['photon_directions']
            N = len(photon_origins)

            # the number 1_000_000 is hard coded also in _simulation_core
            padding_size = max(0, 1_000_000-N)

            # Pad the origins array (2D array with shape [N,3])
            muon_data['photon_origins'] = jnp.pad(photon_origins, ((0, padding_size), (0, 0)),
                                                mode='constant', constant_values=0)

            # Pad the directions array with a default unit vector [0,0,1]
            default_direction = jnp.array([0.0, 0.0, 1.0])
            padding_directions = jnp.tile(default_direction, (padding_size, 1))
            if padding_size > 0:
                muon_data['photon_directions'] = jnp.concatenate([photon_directions, padding_directions], axis=0)
            else:
                muon_data['photon_directions'] = photon_directions

            muon_data['N'] = N

            # Run simulation for muon
            muon_charges, muon_times = event_simulator(track_params, sensor_params, sim_key, muon_data)

            # Store muon data
            all_charges.append(muon_charges)
            all_times.append(muon_times)
            all_energies.append(muon_energy)
            all_directions.append(muon_direction)
            all_indices.append(i)

            # Process additional rings (pions) if n_rings > 1
            for ring_idx in range(1, n_rings):
                # Get a random entry index from the pion file
                random_idx = get_random_root_entry_index(pion_root_file_path)

                # Read photon data for pion
                pion_data = read_photon_data_from_root(pion_root_file_path, random_idx, 'pion')

                photon_origins = pion_data['photon_origins']
                photon_directions = pion_data['photon_directions']
                N = len(photon_origins)

                padding_size = max(0, 1_000_000-N)

                pion_data['photon_origins'] = jnp.pad(photon_origins, ((0, padding_size), (0, 0)),
                                                     mode='constant', constant_values=0)

                default_direction = jnp.array([0.0, 0.0, 1.0])
                padding_directions = jnp.tile(default_direction, (padding_size, 1))
                if padding_size > 0:
                    pion_data['photon_directions'] = jnp.concatenate([photon_directions, padding_directions], axis=0)
                else:
                    pion_data['photon_directions'] = photon_directions

                pion_data['N'] = N

                # Generate a new random direction for the pion
                pion_dir_key, master_key = jax.random.split(master_key)
                pion_direction = generate_random_direction(pion_dir_key)

                # Create parameters for pion
                pion_track_params = ParticleParams.from_cartesian(
                    energy=pion_data['energy'],
                    position=shared_vertex,
                    direction=pion_direction,
                    t0=0.0,
                )

                # Get a new key for the pion simulation
                pion_sim_key, master_key = jax.random.split(master_key)

                # Run simulation for pion
                pion_charges, pion_times = event_simulator(pion_track_params, sensor_params, pion_sim_key, pion_data)

                # Store pion data
                all_charges.append(pion_charges)
                all_times.append(pion_times)
                all_energies.append(pion_data['energy'])
                all_directions.append(pion_direction)
                all_indices.append(random_idx)

            # Combine all rings
            if n_rings > 1:
                combined_charges, combined_times = superimpose_multiple_events(all_charges, all_times)
            else:
                combined_charges, combined_times = all_charges[0], all_times[0]

            # Create filename with sequential numbering
            event_number = i - start_idx + batch_idx * batch_size
            filename = os.path.join(output_dir, f'event_{event_number}.h5')

            # Store original indices in extended_info
            particle_indices = [all_indices[ring_idx] for ring_idx in range(n_rings)]

            save_params = (all_energies[0], shared_vertex, all_directions[0])

            extended_info = {
                'n_rings': n_rings,
                'particle_types': ['muon'] + ['pion'] * (n_rings - 1),
                'energies': all_energies,
                'directions': [dir.tolist() for dir in all_directions],
                'indices': all_indices,
                'vertex': shared_vertex.tolist(),
                'original_indices': particle_indices
            }

            batch_data.append((all_charges, all_times, extended_info))
            batch_params.append(save_params)
            batch_filenames.append(filename)
            batch_indices.append(event_number)

        # Save all events in the batch using multithreading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    save_single_event_with_extended_info,
                    data[0], data[1],
                    params,
                    extended_info=data[2],
                    event_number=idx,
                    filename=filename
                )
                for data, params, filename, idx in zip(
                    batch_data, batch_params, batch_filenames, batch_indices
                )
            ]

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                desc=f"Saving batch {batch_idx+1}",
                total=len(futures),
                unit="file"
            ):
                try:
                    saved_file = future.result()
                    saved_files.append(saved_file)
                except Exception as e:
                    print(f"Error saving file: {e}")

    print(f"Successfully processed {len(saved_files)} events.")
    print(f"All events saved to {output_dir} with sequential naming (event_0.h5, event_1.h5, ...)")
    return saved_files


def read_photon_data_from_photonsim(root_file_path, entry_index):
    """
    Read photon data from a PhotonSim ROOT file for a specific entry.

    Parameters
    ----------
    root_file_path : str
        Path to the PhotonSim ROOT file
    entry_index : int
        Entry index to read from the file

    Returns
    -------
    dict
        Dictionary containing photon_origins, photon_directions, and energy
    """
    import uproot
    import numpy as np
    import jax.numpy as jnp

    # Open the ROOT file
    root_file = uproot.open(root_file_path)

    # Per-photon scalars are in OpticalPhotonsRaw (chunked); only
    # event-level metadata stays on OpticalPhotons.
    if 'OpticalPhotonsRaw' not in root_file:
        raise ValueError(
            f"PhotonSim ROOT file {root_file_path} is missing OpticalPhotonsRaw. "
            f"Re-simulate with the current PhotonSim build."
        )
    tree = root_file['OpticalPhotons']
    raw_tree = root_file['OpticalPhotonsRaw']

    tree_data = tree.arrays(['PrimaryEnergy'],
                            entry_start=entry_index, entry_stop=entry_index+1, library='np')

    # Extract primary energy (already in MeV)
    energy = float(tree_data['PrimaryEnergy'][0])

    # Stitch chunks for this event into flat per-photon arrays
    photon_positions, photon_directions, photon_times, photon_wavelengths = \
        _read_photons_for_event(raw_tree, entry_index)

    result = {
        'photon_origins': jnp.array(photon_positions),     # Combined position vectors in m
        'photon_directions': jnp.array(photon_directions), # Combined direction vectors
        'photon_times': jnp.array(photon_times),
        'energy': energy  # Energy in MeV
    }

    # Per-photon wavelengths (nm) — always present in OpticalPhotonsRaw.
    result['wavelengths'] = jnp.array(photon_wavelengths)

    return result

def _read_event_raw(root_file_path, entry_index):
    """Read one PhotonSim event from ROOT into a raw dict — no categorization.

    This is the I/O-only half of the legacy ``read_particle_data_from_photonsim``.
    The ray-tracing path consumes the raw output directly; ``meaningful_tracks``,
    ``segments`` (filtered), ``particles`` and any view of ``track_info_dict``
    enriched with category/sub_id are derived downstream by
    :func:`_derive_views_from_segments` after the kernel call.

    Parameters
    ----------
    root_file_path : str
    entry_index : int

    Returns
    -------
    dict with keys:
        - 'photon_origins', 'photon_directions', 'photon_times',
          'photon_wavelengths' — flat per-photon arrays for this event
        - 'photon_segment_index_raw' — (N_photons,) int64 indices into
          the **raw** segment table (no remap)
        - 'segments_raw' — dict of all-tracks segment arrays in **mm** plus
          ``track_id`` (int64) and ``n_segments`` (int). Endpoint
          conversion to metres is deferred to ``_derive_views_from_segments``.
        - 'track_info_dict' — raw per-track dict; ``category`` and
          ``sub_id`` are sentinel ``-1`` (filled downstream).
        - 'primary_energy', 'rootracker_entry_id', 'neutrino_pdg',
          'neutrino_energy_MeV'
    """
    import uproot
    import numpy as np

    root_file = uproot.open(root_file_path)
    tree = root_file['OpticalPhotons']

    # Per-photon scalar measurements live on a sister tree
    # (OpticalPhotonsRaw) as fixed-K chunks so PhotonSim's peak RAM is
    # bounded at any energy. We assemble per-event flat arrays here from
    # the chunks belonging to entry_index. Old ROOT files without
    # OpticalPhotonsRaw are not supported (re-simulate to migrate).
    if 'OpticalPhotonsRaw' not in root_file:
        raise ValueError(
            f"PhotonSim ROOT file {root_file_path} is missing OpticalPhotonsRaw. "
            f"This LUCiD release expects the chunked photon layout. "
            f"Re-simulate with the current PhotonSim build."
        )
    raw_tree = root_file['OpticalPhotonsRaw']

    # Verify the post-Stage-5a schema. ``Segment_TrackID`` is the inline
    # track-ownership branch added when the meaningful-tracks filter
    # was dropped in PhotonSim; ``Photon_SegmentIndex`` became
    # unconditional at the same time; ``TrackInfo_CreatorProcess`` is
    # the input the Python categorizer needs (Stage 1).
    available = set(tree.keys())
    required = {
        'Segment_TrackID', 'Photon_SegmentIndex',
        'TrackInfo_CreatorProcess',
    }
    missing = required - available
    if missing:
        raise ValueError(
            f"PhotonSim ROOT file is missing branches {sorted(missing)}. "
            f"Re-simulate with PhotonSim branch 'raw-segments-no-merge' "
            f"(commit 672066b or later)."
        )

    branches_to_read = [
        'PrimaryEnergy',
        'TrackInfo_TrackID',
        'TrackInfo_PosX', 'TrackInfo_PosY', 'TrackInfo_PosZ',
        'TrackInfo_DirX', 'TrackInfo_DirY', 'TrackInfo_DirZ',
        'TrackInfo_Energy', 'TrackInfo_Time',
        'TrackInfo_ParentTrackID', 'TrackInfo_PDG',
        'TrackInfo_CreatorProcess',
        'NSegments',
        'Segment_StartX', 'Segment_StartY', 'Segment_StartZ',
        'Segment_EndX', 'Segment_EndY', 'Segment_EndZ',
        'Segment_DirX', 'Segment_DirY', 'Segment_DirZ',
        'Segment_Edep', 'Segment_Time',
        'Segment_BetaStart', 'Segment_NCherenkov',
        'Segment_TrackID',
        'Photon_SegmentIndex',
        'RooTrackerEntryID', 'IncomingNuPdg', 'IncomingNuKE',
    ]
    tree_data = tree.arrays(
        branches_to_read,
        entry_start=entry_index, entry_stop=entry_index + 1,
        library='np',
    )

    primary_energy = float(tree_data['PrimaryEnergy'][0])

    # Pull this event's photon chunks from OpticalPhotonsRaw. Concatenate
    # in ascending ChunkStartID order so the resulting flat arrays line up
    # with global photon IDs.
    photon_positions, photon_directions, photon_times, photon_wavelengths = \
        _read_photons_for_event(raw_tree, entry_index)

    # ---- TrackInfo_* → track_info_dict (raw, pre-categorization) ----
    track_ids = np.asarray(tree_data['TrackInfo_TrackID'][0], dtype=np.int64)
    track_posx = np.asarray(tree_data['TrackInfo_PosX'][0], dtype=np.float64) / 1000.0  # mm → m
    track_posy = np.asarray(tree_data['TrackInfo_PosY'][0], dtype=np.float64) / 1000.0
    track_posz = np.asarray(tree_data['TrackInfo_PosZ'][0], dtype=np.float64) / 1000.0
    track_dirx = np.asarray(tree_data['TrackInfo_DirX'][0], dtype=np.float64)
    track_diry = np.asarray(tree_data['TrackInfo_DirY'][0], dtype=np.float64)
    track_dirz = np.asarray(tree_data['TrackInfo_DirZ'][0], dtype=np.float64)
    track_energies = np.asarray(tree_data['TrackInfo_Energy'][0], dtype=np.float64)
    track_times = np.asarray(tree_data['TrackInfo_Time'][0], dtype=np.float64)
    track_parent_ids = np.asarray(tree_data['TrackInfo_ParentTrackID'][0], dtype=np.int64)
    track_pdgs = np.asarray(tree_data['TrackInfo_PDG'][0], dtype=np.int64)
    track_processes = tree_data['TrackInfo_CreatorProcess'][0]  # vector<string>

    track_info_dict = {}
    for i in range(len(track_ids)):
        tid = int(track_ids[i])
        track_info_dict[tid] = {
            'track_id': tid,
            # category / sub_id filled in by _derive_views_from_segments.
            'category': -1,
            'sub_id': -1,
            'position': np.array([track_posx[i], track_posy[i], track_posz[i]]),
            'direction': np.array([track_dirx[i], track_diry[i], track_dirz[i]]),
            'energy': float(track_energies[i]),
            'time': float(track_times[i]),
            'parent_id': int(track_parent_ids[i]),
            'pdg': int(track_pdgs[i]),
            'creator_process': str(track_processes[i]),
        }

    # ---- Raw segment table (no filtering) ----
    # Endpoints stay in mm here so ``_derive_views_from_segments`` can hand
    # the same arrays into ``assign_group_ids`` without round-trip-converting.
    n_segments_raw = int(tree_data['NSegments'][0])
    segments_raw = {
        'start_x_mm': np.asarray(tree_data['Segment_StartX'][0], dtype=np.float64),
        'start_y_mm': np.asarray(tree_data['Segment_StartY'][0], dtype=np.float64),
        'start_z_mm': np.asarray(tree_data['Segment_StartZ'][0], dtype=np.float64),
        'end_x_mm':   np.asarray(tree_data['Segment_EndX'][0], dtype=np.float64),
        'end_y_mm':   np.asarray(tree_data['Segment_EndY'][0], dtype=np.float64),
        'end_z_mm':   np.asarray(tree_data['Segment_EndZ'][0], dtype=np.float64),
        'dir_x':      np.asarray(tree_data['Segment_DirX'][0], dtype=np.float64),
        'dir_y':      np.asarray(tree_data['Segment_DirY'][0], dtype=np.float64),
        'dir_z':      np.asarray(tree_data['Segment_DirZ'][0], dtype=np.float64),
        'edep':       np.asarray(tree_data['Segment_Edep'][0], dtype=np.float64),
        'time':       np.asarray(tree_data['Segment_Time'][0], dtype=np.float64),
        'beta_start': np.asarray(tree_data['Segment_BetaStart'][0], dtype=np.float64),
        'n_cherenkov':np.asarray(tree_data['Segment_NCherenkov'][0], dtype=np.int64),
        'track_id':   np.asarray(tree_data['Segment_TrackID'][0], dtype=np.int64),
        'n_segments': n_segments_raw,
    }
    photon_segment_index_raw = np.asarray(
        tree_data['Photon_SegmentIndex'][0], dtype=np.int64)

    return {
        'photon_origins':           photon_positions,
        'photon_directions':        photon_directions,
        'photon_times':              photon_times,
        'photon_wavelengths':        photon_wavelengths,
        'photon_segment_index_raw': photon_segment_index_raw,
        'segments_raw':              segments_raw,
        'track_info_dict':           track_info_dict,
        'primary_energy':            primary_energy,
        'rootracker_entry_id':       int(tree_data['RooTrackerEntryID'][0]),
        'neutrino_pdg':              int(tree_data['IncomingNuPdg'][0]),
        'neutrino_energy_MeV':       float(tree_data['IncomingNuKE'][0]),
    }


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
        Optional dict with keys ``qe_weight`` (N_photons,) float32,
        ``qe_time`` (N_photons,) float32, ``sensor_idx`` (N_photons,)
        int32, ``seg_idx_raw`` (N_photons,) int32 — the per-photon flat
        lists emitted by :func:`_trace_event_bucketed`. When provided,
        the function attaches a ``photon_records_filtered`` entry with
        the same fields plus ``seg_idx_filtered`` (raw→filtered remap)
        and ``particle_idx`` (filtered-track → categorized particle).
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

    # Filter Segment_* arrays to meaningful only. Keeps seg.h5 size
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
    # ``filter_segments_to_meaningful`` already produced ``photon_segment_index``
    # in the **filtered** segment space; ``bucket_photons_by_segment`` already
    # gave us the per-photon particle index. Both arrays carry the -1
    # sentinel for orphans, which is what the host aggregator's QE-pass
    # mask needs.
    if photon_records is not None:
        photon_records_filtered = {
            'qe_weight':        photon_records['qe_weight'],
            'qe_time':          photon_records['qe_time'],
            'sensor_idx':       photon_records['sensor_idx'],
            'seg_idx_filtered': photon_segment_index.astype(np.int32, copy=False),
            'particle_idx':     photon_to_particle.astype(np.int32, copy=False),
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


def read_particle_data_from_photonsim(root_file_path, entry_index):
    """Backward-compatible wrapper: read raw + derive views without per-segment data.

    Equivalent to the legacy implementation; preserves the exact return-dict
    shape for any external caller. The data-mode driver no longer routes
    through this wrapper — it calls :func:`_read_event_raw`,
    :func:`_trace_event_bucketed`, and :func:`_derive_views_from_segments`
    directly so the per-(segment, sensor) tensor that the kernel emits can
    flow downstream.
    """
    raw = _read_event_raw(root_file_path, entry_index)
    return _derive_views_from_segments(raw, photon_records=None)


def generate_events_from_photonsim_particles(event_simulator, root_file_path,
                                             sensor_positions, output_dir=None,
                                             n_events=None, batch_size=100, master_seed=None,
                                             job_id=1,
                                             apply_smearing=False, apply_rotation=False, apply_translation=False,
                                             detector_config_path=None,
                                             dataset_name='unnamed_dataset', run_id=None,
                                             file_index_start=0, detector_type='cylinder',
                                             material='water',
                                             primary_source='particles',
                                             pad_size_buckets=None):
    """Generate events from a PhotonSim ROOT file, writing v3 four-file batches.

    For each batch of events, writes four HDF5 files under ``output_dir``:
    ``sensor/wc_sensor_NNNN.h5``, ``inst/wc_inst_NNNN.h5``,
    ``seg/wc_seg_NNNN.h5``, ``labl/wc_labl_NNNN.h5``. See
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
    detector_config_path : str, optional
        Required when ``apply_translation=True``; also used for seg config
        geometry attrs.
    dataset_name : str
        Provenance: dataset identifier written to every ``config/`` group.
    run_id : str, optional
        Provenance: unique batch identifier; auto-UUID4 if None.
    file_index_start : int
        Index of the first batch file in this invocation (default 0).
    detector_type, material : str
        Provenance: detector geometry type and medium.
    primary_source : str
        'particles' or 'genie'. Written into ``per_interaction/source_type``
        for every event of this batch.
    """
    source_type_code = _source_type_code(primary_source)
    import uproot
    import time
    import uuid
    import subprocess
    from pathlib import Path
    from lucid.detector_params import ParticleParams
    import numpy as np
    import json
    from lucid.utils import smear_charges_SK_like, smear_times

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
        git_commit = os.environ.get('GIT_COMMIT', 'unknown')

    # Create output directory tree
    out_root = Path(output_dir)
    for subdir in ('sensor', 'inst', 'seg', 'labl'):
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

    print(f"\nGenerating {n_events} events using VMAP-OPTIMIZED particle-based processing...")
    print(f"Using batch size of {batch_size} events for multithreaded I/O")
    print(f"Apply smearing: {apply_smearing}")
    print(f"Apply translation: {apply_translation}")
    # Note: Rotation is not applied in this workflow because PhotonSim already generates
    # primaries with random isotropic directions (/gun/randomDirection true). The photon
    # and track data are already in randomized coordinate frames, so rotation in LUCiD
    # would be redundant. Only translation is needed to place the vertex in the detector.
    if apply_rotation:
        print(f"WARNING: apply_rotation=True was passed but rotation is disabled in this workflow.")
        print(f"         PhotonSim already generates tracks with random directions, so rotation is unnecessary.")
    print(f"Saving events to directory: {output_dir}")

    # Load detector bounds for containment calculation and (optionally) translation
    detector_bounds = None
    if apply_translation and detector_config_path is None:
        raise ValueError("detector_config_path must be provided when apply_translation=True")

    if detector_config_path is not None:
        with open(detector_config_path, 'r') as f:
            config = json.load(f)

        detector_type = config.get('detector_type', 'cylinder')
        geom_def = config['geometry_definitions']

        if detector_type == 'cylinder':
            detector_bounds = {
                'type': 'cylinder',
                'radius': geom_def['radius'],
                'height': geom_def['height']
            }
        elif detector_type == 'sphere':
            detector_bounds = {
                'type': 'sphere',
                'radius': geom_def['radius']
            }
        elif detector_type == 'box':
            detector_bounds = {
                'type': 'box',
                'length': geom_def['length'],
                'width': geom_def['width'],
                'height': geom_def['height']
            }

        print(f"Detector bounds loaded: {detector_bounds}")

    saved_files = []
    event_times = []  # Track event processing times

    # Warm up the JIT cache once for every (rays_b, seg_b) bucket pair we
    # expect to hit. After this returns, every per-event kernel call lands
    # on a cached compile and runs at native cost. ~10–30s per pair on CPU,
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
            'photon_origins': photon_origins,
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
            # Phase 2 — bucketed trace (per-photon flat lists out).
            # ----------------------------------------------------------------
            print(f"    Running bucketed trace "
                  f"(rays_buckets={list(pad_size_buckets)})...", flush=True)
            sim_start_time = time.time()
            _t_sim = time.perf_counter()
            pe_per_sensor_np, t_per_sensor_np, \
                photon_qe_w, photon_qe_t, photon_sen_i, photon_seg_i_raw = \
                _trace_event_bucketed(
                    event_simulator,
                    photon_origins_np, photon_directions_np,
                    photon_times_np, photon_wavelengths_np,
                    photon_segment_index_raw,
                    n_sensors, pad_size_buckets,
                    master_key,
                )
            sim_elapsed = time.time() - sim_start_time
            print(f"    Simulation completed in {sim_elapsed:.2f}s", flush=True)
            print(f"    [timing] simulate {time.perf_counter() - _t_sim:.3f}s", flush=True)

            # ----------------------------------------------------------------
            # Phase 3 — derive views (categorize, filter, build photon records).
            # ----------------------------------------------------------------
            _t_post = time.perf_counter()
            particle_data = _derive_views_from_segments(raw, photon_records={
                'qe_weight':   photon_qe_w,
                'qe_time':     photon_qe_t,
                'sensor_idx':  photon_sen_i,
                'seg_idx_raw': photon_seg_i_raw,
            })
            n_particles = particle_data['n_particles']
            particles = particle_data['particles']
            n_segments = int(particle_data['segments']['n_segments'])

            # ----------------------------------------------------------------
            # Phase 4 — host aggregation: inst.h5 PE/T + seg.h5 sparse triplets.
            # ----------------------------------------------------------------
            pr = particle_data['photon_records_filtered']
            agg = _aggregate_from_photon_records(
                pr['qe_weight'], pr['qe_time'], pr['sensor_idx'],
                pr['seg_idx_filtered'], pr['particle_idx'],
                n_particles=n_particles, n_sensors=n_sensors)
            PE_per_particle = agg['PE_per_particle']
            T_per_particle  = agg['T_per_particle']
            seg_hits        = agg['segment_sensor_hits']

            # PE_true / T_true (per-sensor pre-smearing) come from the
            # kernel's per-sensor accumulator: includes every photon's
            # contribution, even orphan-track photons that the
            # aggregator drops from inst.h5.
            PE_true = jnp.asarray(pe_per_sensor_np)
            T_true  = jnp.asarray(t_per_sensor_np)

            # Apply smearing if requested
            if apply_smearing:
                smear_pe_key, smear_t_key = jax.random.split(event_keys['smear_key'])
                PE_reco = smear_charges_SK_like(PE_true, key=smear_pe_key)
                T_reco = smear_times(T_true, key=smear_t_key)
            else:
                PE_reco = PE_true
                T_reco = T_true

            # Convert JAX arrays to numpy BEFORE storing in extended_info.
            # Critical for thread-safe saving with ThreadPoolExecutor, and
            # ``np.array`` (not ``asarray``) on the JAX-backed values
            # ensures we own a writable host buffer for the in-place
            # t0 shift below — JAX buffers come back read-only.
            PE_per_particle = np.asarray(PE_per_particle, dtype=np.float32)
            T_per_particle  = np.asarray(T_per_particle,  dtype=np.float32)
            PE_true = np.array(PE_true, dtype=np.float32, copy=True)
            T_true  = np.array(T_true,  dtype=np.float32, copy=True)
            PE_reco = np.array(PE_reco, dtype=np.float32, copy=True)
            T_reco  = np.array(T_reco,  dtype=np.float32, copy=True)

            # Shift simulator outputs from G4-frame (origin at vertex) into
            # absolute detector frame by adding the per-interaction t0.
            # Only the single-vertex path is in this function today; the
            # pile-up path applies per-vertex t0 in its merger. The
            # positivity mask preserves "no-hit" sentinels (0/inf) on the
            # dense per-sensor / per-particle tensors. seg_hits['T'] is
            # sparse — every entry is a real hit, so a flat += suffices.
            t0_f32 = np.float32(t0)
            np.add(T_per_particle, t0_f32, out=T_per_particle, where=T_per_particle > 0)
            np.add(T_true,         t0_f32, out=T_true,         where=T_true > 0)
            np.add(T_reco,         t0_f32, out=T_reco,         where=T_reco > 0)
            if seg_hits is not None and seg_hits['T'].size > 0:
                seg_hits['T'] = seg_hits['T'] + t0_f32
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
                'PE_per_particle': PE_per_particle,
                'T_per_particle': T_per_particle,
                'PE_reco': PE_reco,
                'T_reco': T_reco,
                'source': 'PhotonSim_Particles_VMAP',
            }

            if 'meaningful_tracks' in particle_data:
                extended_info['meaningful_tracks'] = particle_data['meaningful_tracks']
                extended_info['segments'] = particle_data['segments']

            # seg.h5 sparse triplets came directly from the host aggregator
            # (seg_hits dict already in writer-ready shape). Skip the
            # writer wiring when there are no hits at all (dark event /
            # all photons orphan-segmented).
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
        inst_path = out_root / 'inst' / f'wc_inst_{file_idx:04d}.h5'
        seg_path = out_root / 'seg' / f'wc_seg_{file_idx:04d}.h5'
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

        # Optional geometry hints for seg config
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

        with h5py.File(sensor_path, 'w') as fs, h5py.File(inst_path, 'w') as fi, \
                h5py.File(seg_path, 'w') as fg, h5py.File(labl_path, 'w') as fl:
            write_sensor_config_v3(fs, config_meta, batch_src_idx, sensor_positions_np)
            write_inst_config_v3(fi, config_meta, batch_src_idx, sensor_positions_np)
            write_seg_config_v3(fg, config_meta, batch_src_idx)
            write_labl_config_v3(fl, config_meta, batch_src_idx)

            for seq_idx, evdict in enumerate(batch_data):
                save_sensor_event_v3(fs, evdict, seq_idx)
                save_inst_event_v3(fi, evdict, seq_idx)
                save_seg_event_v3(fg, evdict, seq_idx)
                save_labl_event_v3(fl, evdict, seq_idx)

        saved_files.extend([str(sensor_path), str(inst_path), str(seg_path), str(labl_path)])

        t_save = time.time() - t_save_start
        print(f"Batch {batch_idx+1} save time: {t_save:.3f}s\n")

    print(f"\nSuccessfully wrote {num_batches} batches "
          f"({len(saved_files)} files total) to {output_dir}/"
          f"{{sensor,inst,seg,labl}}/")

    # Print average event time
    if event_times:
        avg_time = sum(event_times) / len(event_times)
        print(f"Average event processing time: {avg_time:.3f}s")

    return saved_files


def _simulate_vertex_stream(
    *,
    event_simulator,
    particle_data,
    translation_vector,
    apply_translation,
    n_sensors,
    pad_size,
    sim_key,
):
    """Run the vmap photon simulator for one PhotonSim stream.

    Returns (PE_per_particle, T_per_particle) as numpy float32 arrays of
    shape ``(n_particles, n_sensors)``. Inputs are mutated: track_info
    positions get shifted by ``translation_vector`` to keep the per-track
    info in the shifted frame. All times remain in G4 frame; the caller
    adds per-interaction t0 afterwards to move to absolute detector frame.
    """
    from lucid.detector_params import ParticleParams

    n_particles = particle_data['n_particles']
    particles = particle_data['particles']
    default_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    batched_origins_np = np.zeros((n_particles, pad_size, 3), dtype=np.float32)
    batched_directions_np = np.tile(default_direction, (n_particles, pad_size, 1))
    batched_times_np = np.zeros((n_particles, pad_size), dtype=np.float32)
    batched_wavelengths_np = np.zeros((n_particles, pad_size), dtype=np.float32)
    N_per_particle_np = np.zeros(n_particles, dtype=np.int32)
    track_energies_np = np.zeros(n_particles, dtype=np.float32)
    track_positions_np = np.zeros((n_particles, 3), dtype=np.float32)
    track_directions_np = np.zeros((n_particles, 3), dtype=np.float32)

    all_origins = particle_data['photon_origins']
    all_dirs    = particle_data['photon_directions']
    all_times   = particle_data['photon_times']
    all_wl      = particle_data['photon_wavelengths']

    for pi, particle in enumerate(particles):
        photon_indices = particle['photon_indices']
        N = len(photon_indices)
        N_per_particle_np[pi] = N
        ti = particle['track_info']
        if ti is not None:
            track_energies_np[pi]   = ti['energy']
            track_positions_np[pi]  = ti['position']
            track_directions_np[pi] = ti['direction']
        else:
            track_energies_np[pi]   = particle_data.get('primary_energy', 0.0)
            track_directions_np[pi] = default_direction
        if apply_translation:
            track_positions_np[pi] += translation_vector
            if ti is not None:
                ti['position'] = track_positions_np[pi].copy()
        if N > 0:
            batched_origins_np[pi, :N]      = all_origins[photon_indices]
            batched_directions_np[pi, :N]   = all_dirs[photon_indices]
            batched_times_np[pi, :N]        = all_times[photon_indices]
            batched_wavelengths_np[pi, :N]  = all_wl[photon_indices]

    batched_origins     = jax.device_put(batched_origins_np)
    batched_directions  = jax.device_put(batched_directions_np)
    batched_times       = jax.device_put(batched_times_np)
    batched_wavelengths = jax.device_put(batched_wavelengths_np)
    N_per_particle_array    = jax.device_put(N_per_particle_np)
    track_energies_array    = jax.device_put(track_energies_np)
    track_positions_array   = jax.device_put(track_positions_np)
    track_directions_array  = jax.device_put(track_directions_np)

    def _sim_one(energy, pos, dir_, po, pd, pt, pw, N, key):
        track_params = ParticleParams.from_cartesian(
            energy=energy, position=pos, direction=dir_, t0=0.0)
        photonsim_data = {
            'photon_origins': po, 'photon_directions': pd,
            'photon_times': pt, 'wavelengths': pw, 'N': N,
            'apply_rotation': False,
            'rotation_axis': jnp.array([1.0, 0.0, 0.0]),
            'rotation_angle': 0.0,
            # Photons were already translated in NumPy by the caller; do NOT
            # translate a second time inside the JIT simulator.
            'apply_translation': False,
            'translation_vector': jnp.zeros(3),
        }
        return event_simulator(track_params, key, photonsim_data)

    simulate_all = jax.vmap(_sim_one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))
    particle_keys = jax.random.split(sim_key, n_particles)
    PE_pp, T_pp = simulate_all(
        track_energies_array, track_positions_array, track_directions_array,
        batched_origins, batched_directions, batched_times, batched_wavelengths,
        N_per_particle_array, particle_keys)
    return np.asarray(PE_pp, dtype=np.float32), np.asarray(T_pp, dtype=np.float32)


def _offset_track_ids(particle_data, offset):
    """Shift all G4 track IDs in ``particle_data`` by ``offset``.

    parent_id == 0 (primary convention) is left alone so primaries
    remain recognizable after merging. Mutates in place and also returns
    the max track_id seen post-shift (so the caller can advance the
    running offset for the next vertex stream).
    """
    if offset == 0:
        return _max_track_id(particle_data)

    def _shift(tid):
        return int(tid) + offset if int(tid) > 0 else 0

    # meaningful_tracks: remap both the dict keys and each record's track_id / parent_id.
    mt = particle_data.get('meaningful_tracks')
    if mt:
        new_mt = {}
        for tid, t in mt.items():
            t = dict(t)
            t['track_id']  = _shift(t['track_id'])
            t['parent_id'] = _shift(t['parent_id'])
            new_mt[_shift(tid)] = t
        particle_data['meaningful_tracks'] = new_mt

    # track_info_dict: same treatment.
    tid_dict = particle_data.get('track_info_dict')
    if tid_dict:
        new_tid = {}
        for tid, t in tid_dict.items():
            t = dict(t)
            t['track_id']  = _shift(t.get('track_id', tid))
            t['parent_id'] = _shift(t.get('parent_id', 0))
            new_tid[_shift(tid)] = t
        particle_data['track_info_dict'] = new_tid

    # particles: genealogy and extended_genealogy lists of track IDs.
    for p in particle_data.get('particles', []):
        gen = p.get('genealogy') or []
        p['genealogy'] = [_shift(g) for g in gen]
        ext = p.get('extended_genealogy')
        if ext is not None:
            p['extended_genealogy'] = [_shift(g) for g in ext]
        # track_info inside particle (if any) — also remap.
        ti = p.get('track_info')
        if ti is not None and 'track_id' in ti:
            ti['track_id'] = _shift(ti['track_id'])
            if 'parent_id' in ti:
                ti['parent_id'] = _shift(ti['parent_id'])

    return _max_track_id(particle_data)


def _max_track_id(particle_data):
    """Largest track_id present in the stream (0 if the stream is empty)."""
    mt = particle_data.get('meaningful_tracks') or {}
    tid_d = particle_data.get('track_info_dict') or {}
    ids = [int(t) for t in mt.keys()] + [int(t) for t in tid_d.keys()]
    return max(ids) if ids else 0


def _offset_track_ids_raw(raw, offset):
    """Shift all G4 track IDs in a ``_read_event_raw`` output by ``offset``.

    Same role as :func:`_offset_track_ids` but operates on the raw read
    dict (no ``meaningful_tracks`` / ``particles`` yet — those come from
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
    detector_config_path=None,
    dataset_name='unnamed_pileup_dataset',
    run_id=None,
    file_index_start=0,
    detector_type='cylinder',
    material='water',
):
    """Generate pile-up events by merging N PhotonSim streams per event.

    Each entry in ``root_file_paths`` is a PhotonSim ROOT file from one
    vertex's interaction. For each event index, we draw an independent
    absolute t0 and fiducial vertex per vertex, simulate each vertex's
    photons, remap G4 track IDs to avoid collisions, and merge the
    per-vertex results into one event_dict. Sensor/inst/seg/labl are
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
    import json
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
        git_commit = os.environ.get('GIT_COMMIT', 'unknown')

    out_root = Path(output_dir)
    for sub in ('sensor', 'inst', 'seg', 'labl'):
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

    # Detector bounds for vertex sampling + containment (same as non-pile-up).
    detector_bounds = None
    if detector_config_path is not None:
        with open(detector_config_path) as fj:
            cfg = json.load(fj)
        detector_type_from_cfg = cfg.get('detector_type', 'cylinder')
        gd = cfg['geometry_definitions']
        if detector_type_from_cfg == 'cylinder':
            detector_bounds = {'type': 'cylinder', 'radius': gd['radius'], 'height': gd['height']}
        elif detector_type_from_cfg == 'sphere':
            detector_bounds = {'type': 'sphere', 'radius': gd['radius']}
        elif detector_type_from_cfg == 'box':
            detector_bounds = {'type': 'box',
                               'length': gd['length'], 'width': gd['width'],
                               'height': gd['height']}
    if apply_translation and detector_bounds is None:
        raise ValueError("detector_config_path required when apply_translation=True.")

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

                # Phase 2 — bucketed trace for this vertex (per-photon out).
                _t_sim = _time.time()
                pe_sensor_i, t_sensor_i, qe_w_i, qe_t_i, sen_i_i, seg_i_raw_i = \
                    _trace_event_bucketed(
                        event_simulator,
                        photon_origins_i, photon_directions_i,
                        photon_times_i, photon_wavelengths_i,
                        photon_segment_index_i,
                        n_sensors, rays_buckets,
                        event_keys['sim_key'],
                    )
                print(f"      [timing] simulate {_time.time() - _t_sim:.3f}s", flush=True)

                # Phase 3 — derive views (categorize, filter, build photon records).
                particle_data_i = _derive_views_from_segments(raw, photon_records={
                    'qe_weight':   qe_w_i,
                    'qe_time':     qe_t_i,
                    'sensor_idx':  sen_i_i,
                    'seg_idx_raw': seg_i_raw_i,
                })
                n_particles_i = particle_data_i['n_particles']

                # Phase 4 — host aggregation: per-vertex inst.h5 PE/T + sparse seg.h5 hits.
                pr_i = particle_data_i['photon_records_filtered']
                agg_i = _aggregate_from_photon_records(
                    pr_i['qe_weight'], pr_i['qe_time'], pr_i['sensor_idx'],
                    pr_i['seg_idx_filtered'], pr_i['particle_idx'],
                    n_particles=n_particles_i, n_sensors=n_sensors)
                PE_i      = agg_i['PE_per_particle']
                T_i       = agg_i['T_per_particle']
                seg_hits_i = agg_i['segment_sensor_hits']

                # Apply +t0_i to shift simulator output into absolute detector frame.
                t0_f32 = np.float32(t0_i)
                np.add(T_i, t0_f32, out=T_i, where=T_i > 0)
                # seg_hits['T'] is sparse — every entry is a real hit, flat += suffices.
                if seg_hits_i['T'].size > 0:
                    seg_hits_i['T'] = seg_hits_i['T'] + t0_f32
                # Same shift for segment times.
                if particle_data_i['segments'].get('n_segments', 0) > 0:
                    particle_data_i['segments']['time'] = (
                        np.asarray(particle_data_i['segments']['time'], dtype=np.float32)
                        + t0_f32)

                streams.append({
                    'particles':         particle_data_i['particles'],
                    'meaningful_tracks': particle_data_i['meaningful_tracks'],
                    'segments':          particle_data_i['segments'],
                    'PE_per_particle':   PE_i,
                    'T_per_particle':    T_i,
                    'seg_hits':          seg_hits_i,
                    'interaction_meta':  build_interaction_metadata(
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
        inst_path   = out_root / 'inst'   / f'wc_inst_{file_idx:04d}.h5'
        seg_path    = out_root / 'seg'    / f'wc_seg_{file_idx:04d}.h5'
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
             h5py.File(inst_path,   'w') as fi, \
             h5py.File(seg_path,    'w') as fg, \
             h5py.File(labl_path,   'w') as fl:
            write_sensor_config_v3(fs, config_meta, batch_src_idx, sensor_positions_np)
            write_inst_config_v3(fi, config_meta, batch_src_idx, sensor_positions_np)
            write_seg_config_v3(fg, config_meta, batch_src_idx)
            write_labl_config_v3(fl, config_meta, batch_src_idx)
            for seq_idx, ev in enumerate(batch_data):
                save_sensor_event_v3(fs, ev, seq_idx)
                save_inst_event_v3(fi, ev, seq_idx)
                save_seg_event_v3(fg, ev, seq_idx)
                save_labl_event_v3(fl, ev, seq_idx)

        saved_files.extend([str(sensor_path), str(inst_path), str(seg_path), str(labl_path)])
        print(f"Batch {batch_idx+1} save time: {_time.time() - _t_save:.3f}s\n")

    print(f"\nSuccessfully wrote {num_batches} batches "
          f"({len(saved_files)} files total) to {output_dir}/"
          f"{{sensor,inst,seg,labl}}/")

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
    PE_per_stream = []
    T_per_stream  = []

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

    # Per-vertex sparse seg.h5 hits. Each stream's ``seg_hits`` carries
    # segment indices that are local to that vertex's filtered segment
    # table (0..n_seg_v-1); the merged seg.h5 wants global indices into
    # the concatenated segment table, so each stream's segment_idx is
    # shifted by the cumulative segment count of preceding streams.
    # Streams' segments are disjoint by construction (track ids were
    # offset upstream so meaningful_tracks don't overlap across streams),
    # which mirrors today's `np.concatenate(axis=0)` row-offset semantics.
    seg_hits_shifted = []
    seg_offset = 0
    for s in streams:
        all_particles.extend(s['particles'])
        all_tracks.update(s['meaningful_tracks'])
        segs = s['segments']
        n_seg_v = int(segs.get('n_segments', 0)) if segs else 0
        if n_seg_v > 0:
            for k in all_segs:
                all_segs[k].append(np.asarray(segs[k]))
        PE_per_stream.append(s['PE_per_particle'])
        T_per_stream.append(s['T_per_particle'])
        sh = s.get('seg_hits')
        if sh is not None and sh['PE'].size > 0:
            seg_hits_shifted.append({
                'segment_idx': sh['segment_idx'] + np.int32(seg_offset),
                'sensor_idx':  sh['sensor_idx'],
                'PE':          sh['PE'],
                'T':           sh['T'],
            })
        seg_offset += n_seg_v

    n_particles_total = len(all_particles)
    PE_per_particle = (np.concatenate(PE_per_stream, axis=0)
                       if PE_per_stream and sum(x.shape[0] for x in PE_per_stream) > 0
                       else np.zeros((0, n_sensors), dtype=np.float32))
    T_per_particle  = (np.concatenate(T_per_stream, axis=0)
                       if T_per_stream and sum(x.shape[0] for x in T_per_stream) > 0
                       else np.zeros((0, n_sensors), dtype=np.float32))

    # Aggregate across particles for sensor/inst files
    if PE_per_particle.shape[0] > 0:
        PE_true = np.sum(PE_per_particle, axis=0).astype(np.float32)
        masked = np.where(T_per_particle > 0, T_per_particle, np.inf)
        T_true = np.min(masked, axis=0)
        T_true = np.where(np.isfinite(T_true), T_true, 0.0).astype(np.float32)
    else:
        PE_true = np.zeros(n_sensors, dtype=np.float32)
        T_true  = np.zeros(n_sensors, dtype=np.float32)

    if apply_smearing and PE_per_particle.shape[0] > 0:
        from lucid.utils import smear_charges_SK_like, smear_times
        smear_pe_key, smear_t_key = jax.random.split(smear_key)
        PE_reco = np.asarray(
            smear_charges_SK_like(jnp.asarray(PE_true), key=smear_pe_key),
            dtype=np.float32)
        T_reco = np.asarray(
            smear_times(jnp.asarray(T_true), key=smear_t_key),
            dtype=np.float32)
    else:
        PE_reco = PE_true.copy()
        T_reco  = T_true.copy()

    # Merge segment arrays
    if all_segs['time']:
        seg_merged = {k: np.concatenate(v) for k, v in all_segs.items()}
        seg_merged['n_segments'] = int(len(seg_merged['time']))
    else:
        seg_merged = {'n_segments': 0}

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
        'PE_per_particle': PE_per_particle,
        'T_per_particle':  T_per_particle,
        'PE_reco': PE_reco,
        'T_reco':  T_reco,
    }

    # Concat the per-vertex sparse seg.h5 triplets (already segment_idx-
    # shifted into the merged segment table's row space).
    if seg_hits_shifted:
        merged['segment_sensor_hits'] = {
            'segment_idx': np.concatenate([d['segment_idx'] for d in seg_hits_shifted]).astype(np.int32),
            'sensor_idx':  np.concatenate([d['sensor_idx']  for d in seg_hits_shifted]).astype(np.uint16),
            'PE':          np.concatenate([d['PE']          for d in seg_hits_shifted]).astype(np.float32),
            'T':           np.concatenate([d['T']           for d in seg_hits_shifted]).astype(np.float32),
        }

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


# ---------------------------------------------------------------------------
# Functions moved from lucid/utils.py during Phase 2.5 refactor
# ---------------------------------------------------------------------------

def full_to_sparse(charges, times):
    """Convert full arrays to sparse representation by removing zero elements.

    Parameters
    ----------
    charges : jnp.ndarray
        Array of charge values for all sensors
    times : jnp.ndarray
        Array of time values for all sensors

    Returns
    -------
    non_zero_indices : jnp.ndarray
        Indices where charges are non-zero
    non_zero_charges : jnp.ndarray
        Charge values at non-zero locations
    non_zero_times : jnp.ndarray
        Time values at non-zero locations
    """
    non_zero_indices = jnp.nonzero(charges)[0]
    non_zero_charges = charges[non_zero_indices]
    non_zero_times = times[non_zero_indices]
    return non_zero_indices, non_zero_charges, non_zero_times


def sparse_to_full(sparse_indices, sparse_values, full_size):
    """Convert sparse representation back to full array with zeros.

    Parameters
    ----------
    sparse_indices : jnp.ndarray
        Indices of non-zero elements
    sparse_values : jnp.ndarray
        Values at the non-zero indices
    full_size : int
        Size of the output array

    Returns
    -------
    jnp.ndarray
        Full array with sparse values inserted at specified indices
    """
    full_data = jnp.zeros(full_size)
    return full_data.at[sparse_indices].set(sparse_values)


def save_single_event(event_data, particle_params, sensor_params, event_number=0, filename=None, calibration_mode=False):
    """Save single event simulation data to an HDF5 file in sparse format.

    Parameters
    ----------
    event_data : tuple
        (charges, average_times) arrays for the event
    particle_params : ParticleParams or IsotropicSource
        if calibration_mode is True: IsotropicSource with position, intensity
        if calibration_mode is False: ParticleParams with energy, position, theta, phi, t0
    sensor_params : DetectorParams
        DetectorParams NamedTuple with all detector calibration fields
    event_number : int, optional
        Event identifier number, defaults to 0
    filename : str, optional
        Custom path to output HDF5 file. If None, auto-generates name
        in 'events' folder as 'event_X.h5' or 'event_X_TIMESTAMP.h5'

    Returns
    -------
    str
        Path to the saved file

    Notes
    -----
    Saves data in a hierarchical structure with two groups:
    - params: contains simulation parameters
    - event: contains sparse event data (indices, charges, times)
    """
    charges, average_times = event_data
    indices, sparse_charges, sparse_times = full_to_sparse(charges, average_times)

    # Generate filename if not provided
    if filename is None:
        from datetime import datetime

        # Create events directory if it doesn't exist
        os.makedirs('events', exist_ok=True)

        base_filename = os.path.join('events', f'event_{event_number}.h5')

        # If file exists, add timestamp
        if os.path.exists(base_filename):
            timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            filename = os.path.join('events', f'event_{event_number}_{timestamp}.h5')
        else:
            filename = base_filename

    with h5py.File(filename, 'w') as f:
        # Save simulation parameters
        if calibration_mode:
            params_group = f.create_group('calibration_params')
            params_group.create_dataset('source_position', data=np.array(particle_params.position))
            params_group.create_dataset('source_intensity', data=np.array(particle_params.intensity))

        else:
            params_group = f.create_group('particle_params')
            params_group.create_dataset('track_energy', data=np.array(particle_params.energy))
            params_group.create_dataset('track_origin', data=np.array(particle_params.position))
            params_group.create_dataset('track_direction', data=np.array(particle_params.direction))

        detector_group = f.create_group('sensor_params')
        detector_group.create_dataset('scatter_length', data=np.array(sensor_params.scatter_length))
        detector_group.create_dataset('wall_reflection_rate', data=np.array(sensor_params.wall_reflection_rate))
        detector_group.create_dataset('sensor_reflection_rate', data=np.array(sensor_params.sensor_reflection_rate))
        detector_group.create_dataset('absorption_length', data=np.array(sensor_params.absorption_length))
        detector_group.create_dataset('qe', data=np.array(sensor_params.qe))
        detector_group.create_dataset('qe_corrections', data=np.array(sensor_params.qe_corrections))

        # Save event data and number
        event_group = f.create_group('event')
        event_group.create_dataset('event_number', data=np.array(event_number))
        event_group.create_dataset('indices', data=np.array(indices))
        event_group.create_dataset('charges', data=np.array(sparse_charges))
        event_group.create_dataset('times', data=np.array(sparse_times))

    return filename


def load_single_event(filename, num_sensors, sparse=True, calibration_mode=False):
    """Load single event simulation data from an HDF5 file.

    Parameters
    ----------
    filename : str
        Path to HDF5 file
    num_sensors : int
        Total number of sensors (needed for dense format)
    sparse : bool, default=True
        If True, returns data in sparse format
        If False, converts to dense arrays
    calibration_mode : bool, default=False
        If True, loads calibration parameters instead of particle parameters

    Returns
    -------
    particle_params : ParticleParams or IsotropicSource
        if calibration_mode is True: IsotropicSource
        if calibration_mode is False: ParticleParams
    sensor_params : DetectorParams
        DetectorParams NamedTuple
    If sparse=True:
        indices, charges, times
    If sparse=False:
        charges, times (dense)
    """
    from lucid.detector_params import ParticleParams, DetectorParams, isotropic_source

    with h5py.File(filename, 'r') as f:
        if calibration_mode:
            params_group = f['calibration_params']
            source_position = jnp.array(params_group['source_position'][()])
            source_intensity = jnp.array(params_group['source_intensity'][()])
            particle_params = isotropic_source(position=source_position, intensity=source_intensity)
        else:
            params_group = f['particle_params']
            track_energy = jnp.array(params_group['track_energy'][()])
            track_origin = jnp.array(params_group['track_origin'][()])
            track_direction = jnp.array(params_group['track_direction'][()])
            particle_params = ParticleParams.from_cartesian(
                energy=track_energy, position=track_origin,
                direction=track_direction, t0=jnp.array(0.0))

        # Load detector parameters
        detector_group = f['sensor_params']
        sensor_params = DetectorParams(
            scatter_length=jnp.array(detector_group['scatter_length'][()]),
            wall_reflection_rate=jnp.array(detector_group['wall_reflection_rate'][()]),
            sensor_reflection_rate=jnp.array(detector_group['sensor_reflection_rate'][()]),
            absorption_length=jnp.array(detector_group['absorption_length'][()]),
            qe=jnp.array(detector_group['qe'][()]),
            qe_corrections=jnp.array(detector_group['qe_corrections'][()]),
        )

        # Load event data
        event_group = f['event']
        event_number = int(event_group['event_number'][()])
        indices = jnp.array(event_group['indices'][()])
        charges = jnp.array(event_group['charges'][()])
        times = jnp.array(event_group['times'][()])

    if sparse:
        return particle_params, sensor_params, indices, charges, times
    else:
        # Convert sparse arrays to full dense arrays
        dense_charges = sparse_to_full(indices, charges, num_sensors)
        dense_times = sparse_to_full(indices, times, num_sensors)

        return particle_params, sensor_params, dense_charges, dense_times


def get_random_root_entry_index(root_file_path):
    """
    Get a random valid entry index from a ROOT file.

    Parameters
    ----------
    root_file_path : str
        Path to the ROOT file

    Returns
    -------
    int
        Random valid entry index
    """
    import uproot

    root_file = uproot.open(root_file_path)
    tree = root_file['v_photon']
    total_entries = tree.num_entries

    return np.random.randint(0, total_entries - 1)

def read_photon_data_from_root(root_file_path, entry_index, particle_type='muon'):
    """
    Read photon data from a ROOT file for a specific entry, using the component vectors.

    Parameters
    ----------
    root_file_path : str
        Path to the ROOT file
    entry_index : int
        Entry index to read from the file
    particle_type : str, optional
        Type of particle ('muon' or 'pion'), by default 'muon'

    Returns
    -------
    dict
        Dictionary containing photon_origins, photon_directions, and energy
    """
    import uproot

    # Open the ROOT file
    root_file = uproot.open(root_file_path)

    # Access the tree
    tree = root_file['v_photon']

    # Read position components
    photon_posx = tree['photon_posx'].array(entry_start=entry_index, entry_stop=entry_index+1)[0]
    photon_posy = tree['photon_posy'].array(entry_start=entry_index, entry_stop=entry_index+1)[0]
    photon_posz = tree['photon_posz'].array(entry_start=entry_index, entry_stop=entry_index+1)[0]

    # Read direction components
    photon_dirx = tree['photon_dirx'].array(entry_start=entry_index, entry_stop=entry_index+1)[0]
    photon_diry = tree['photon_diry'].array(entry_start=entry_index, entry_stop=entry_index+1)[0]
    photon_dirz = tree['photon_dirz'].array(entry_start=entry_index, entry_stop=entry_index+1)[0]

    # Read momentum
    initmom = float(tree['initmom'].array(entry_start=entry_index, entry_stop=entry_index+1)[0])

    # Stack the components to form position and direction arrays
    photon_positions = np.column_stack((photon_posx, photon_posy, photon_posz))
    photon_directions = np.column_stack((photon_dirx, photon_diry, photon_dirz))

    # Convert initmom (momentum) to kinetic energy based on particle type
    if particle_type.lower() == 'muon':
        mass = 105.7  # MeV/c^2 (muon rest mass)
    elif particle_type.lower() == 'pion':
        mass = 139.6  # MeV/c^2 (charged pion rest mass)
    else:
        raise ValueError(f"Unsupported particle type: {particle_type}")

    # E_kinetic = sqrt(p^2 + m^2) - m
    energy = np.sqrt(initmom**2 + mass**2) - mass

    return {
        'photon_origins': jnp.array(photon_positions),     # Combined position vectors
        'photon_directions': jnp.array(photon_directions), # Combined direction vectors
        'energy': float(energy)
    }

def get_pdg_code(particle_type):
    """
    Convert particle type string to PDG code.

    Parameters
    ----------
    particle_type : str
        Particle type string (e.g., 'mu-', 'mu+', 'e-', 'e+', 'pi-', 'pi+', 'pi0')

    Returns
    -------
    int
        PDG code for the particle
    """
    pdg_map = {
        'mu-': 13,
        'mu+': -13,
        'muon': 13,  # backward compatibility
        'e-': 11,
        'e+': -11,
        'electron': 11,
        'positron': -11,
        'pi-': -211,
        'pi+': 211,
        'pi0': 111,
        'pion': 211,  # backward compatibility, assume pi+
        'gamma': 22,
        'photon': 22,
        'proton': 2212,
        'p': 2212,
        'neutron': 2112,
        'n': 2112
    }

    if particle_type in pdg_map:
        return pdg_map[particle_type]
    else:
        raise ValueError(f"Unknown particle type: {particle_type}")

def get_particle_mass(particle_type):
    """
    Get particle rest mass in MeV/c^2.

    Parameters
    ----------
    particle_type : str
        Particle type string (e.g., 'mu-', 'mu+', 'e-', 'e+', 'pi-', 'pi+', 'pi0')

    Returns
    -------
    float
        Rest mass in MeV/c^2
    """
    # Normalize particle type by removing charge for mass lookup
    particle_base = particle_type.replace('-', '').replace('+', '')

    mass_map = {
        'mu': 105.7,      # muon
        'muon': 105.7,
        'e': 0.511,       # electron
        'electron': 0.511,
        'positron': 0.511,
        'pi': 139.6,      # charged pion (pi+ and pi-)
        'pion': 139.6,
        'pi0': 135.0,     # neutral pion
        'gamma': 0.0,
        'photon': 0.0,
        'proton': 938.3,
        'p': 938.3,
        'neutron': 939.6,
        'n': 939.6
    }

    if particle_base in mass_map:
        return mass_map[particle_base]
    elif particle_type == 'pi0':  # special case for pi0
        return mass_map['pi0']
    else:
        raise ValueError(f"Unknown particle type: {particle_type}")

def extract_particle_properties(momentum, pdg_code):
    """
    Extract theta, phi angles and energy from particle momentum.

    Parameters
    ----------
    momentum : array_like
        3D momentum vector [px, py, pz] in MeV/c
    pdg_code : int
        PDG particle code (13 for muon, 211 for pion, etc.)

    Returns
    -------
    tuple
        (theta, phi, kinetic_energy) where:
        - theta: polar angle from z-axis in radians
        - phi: azimuthal angle in xy-plane in radians
        - kinetic_energy: kinetic energy in MeV
    """
    px, py, pz = momentum

    # Calculate momentum magnitude
    p_mag = np.sqrt(px**2 + py**2 + pz**2)

    # Calculate angles
    theta = np.arccos(pz / p_mag) if p_mag > 0 else 0.0  # polar angle from z-axis
    phi = np.arctan2(py, px)  # azimuthal angle in xy-plane

    # Get particle mass based on PDG code
    if pdg_code == 13 or pdg_code == -13:  # muon/antimuon
        mass = 105.7  # MeV/c^2
    elif pdg_code == 211 or pdg_code == -211:  # charged pion
        mass = 139.6  # MeV/c^2
    elif pdg_code == 11 or pdg_code == -11:  # electron/positron
        mass = 0.511  # MeV/c^2
    else:
        # Default to muon mass for unknown particles
        mass = 105.7
        print(f"Warning: Unknown PDG code {pdg_code}, using muon mass")

    # Calculate total energy: E^2 = p^2 + m^2
    total_energy = np.sqrt(p_mag**2 + mass**2)

    # Kinetic energy = Total energy - rest mass
    kinetic_energy = total_energy - mass

    return theta, phi, kinetic_energy

def analyze_loaded_particle(loaded_mom, loaded_vtx, pdg_code):
    """
    Analyze particle properties from loaded HDF5 data.

    Parameters
    ----------
    loaded_mom : array_like
        3D momentum vector [px, py, pz] in MeV/c
    loaded_vtx : array_like
        3D vertex position [x, y, z] in meters
    pdg_code : int
        PDG particle code

    Returns
    -------
    dict
        Dictionary containing particle properties
    """
    theta, phi, kinetic_energy = extract_particle_properties(loaded_mom, pdg_code)

    # Convert angles to degrees for easier interpretation
    theta_deg = np.degrees(theta)
    phi_deg = np.degrees(phi)

    # Calculate momentum magnitude
    p_mag = np.sqrt(np.sum(loaded_mom**2))

    # Particle type name
    particle_names = {13: 'muon', -13: 'antimuon', 211: 'pion+', -211: 'pion-',
                     11: 'electron', -11: 'positron'}
    particle_name = particle_names.get(pdg_code, f'unknown (PDG={pdg_code})')

    return {
        'particle_type': particle_name,
        'pdg_code': pdg_code,
        'momentum_magnitude': p_mag,
        'momentum_vector': loaded_mom,
        'theta_rad': theta,
        'phi_rad': phi,
        'theta_deg': theta_deg,
        'phi_deg': phi_deg,
        'kinetic_energy': kinetic_energy,
        'vertex': loaded_vtx,
        'direction': loaded_mom / p_mag if p_mag > 0 else np.array([0, 0, 1])
    }

def analyze_event_directory(directory, pattern="*.h5", max_files=None, summary_only=False):
    """
    Analyze multiple event files in a directory.

    Parameters
    ----------
    directory : str
        Directory containing HDF5 event files
    pattern : str, optional
        File pattern to match, by default "*.h5"
    max_files : int, optional
        Maximum number of files to analyze, by default None (all files)
    summary_only : bool, optional
        Whether to print only summary statistics and not individual files, by default False

    Returns
    -------
    list of dict
        List of data dictionaries for each event
    """
    # Find all files matching the pattern
    file_paths = _glob(os.path.join(directory, pattern))

    if max_files is not None:
        file_paths = file_paths[:max_files]

    print(f"Found {len(file_paths)} files to analyze")

    # Read all files
    all_data = []
    for file_path in file_paths:
        data = read_event_file(file_path, verbose=not summary_only)
        all_data.append(data)

    # Calculate summary statistics
    total_tracks = sum(data['PDG'].shape[0] for data in all_data)
    muon_count = sum(np.sum(data['PDG'] == 13) for data in all_data)
    pion_count = sum(np.sum(data['PDG'] == 211) for data in all_data)

    # Print summary
    print("\n" + "="*60)
    print(f"Summary Statistics for {len(file_paths)} Events")
    print("="*60)
    print(f"Total number of tracks: {total_tracks}")
    print(f"Total muons: {muon_count} ({muon_count/total_tracks*100:.1f}%)")
    print(f"Total pions: {pion_count} ({pion_count/total_tracks*100:.1f}%)")

    # Calculate charge statistics
    all_q_tot = np.concatenate([data['Q_tot'] for data in all_data])
    print(f"\nCharge Statistics:")
    print(f"Mean charge per track: {np.mean(all_q_tot):.2f}")
    print(f"Min charge: {np.min(all_q_tot):.2f}")
    print(f"Max charge: {np.max(all_q_tot):.2f}")

    # Calculate momentum statistics
    all_p_mag = np.concatenate([
        np.sqrt(np.sum(data['P']**2, axis=1)) for data in all_data
    ])
    print(f"\nMomentum Statistics:")
    print(f"Mean momentum magnitude: {np.mean(all_p_mag):.2f} MeV/c")
    print(f"Min momentum: {np.min(all_p_mag):.2f} MeV/c")
    print(f"Max momentum: {np.max(all_p_mag):.2f} MeV/c")

    # PMT statistics across all events
    if all_data:
        n_detectors = all_data[0]['Q'].shape[1]
        all_pmt_charges = np.zeros(n_detectors)

        for data in all_data:
            all_pmt_charges += np.sum(data['Q'], axis=0)

        active_pmts = np.where(all_pmt_charges > 0)[0]
        print(f"\nPMT Statistics Across All Events:")
        print(f"Number of active PMTs: {len(active_pmts)} / {n_detectors}")
        print(f"Mean charge per active PMT: {np.mean(all_pmt_charges[active_pmts]):.2f}")


    return all_data


# Particle physics constants (rest masses in MeV/c^2)
PARTICLE_MASSES = {
    13: 105.7,   # muon
    -13: 105.7,  # anti-muon
    211: 139.6,  # charged pion
    -211: 139.6, # negative pion
    111: 134.98, # neutral pion
    11: 0.511,   # electron
    -11: 0.511,  # positron
    22: 0.0,     # photon
    2212: 938.3, # proton
    2112: 939.6, # neutron
}

def momentum_to_angles_and_energy(momentum_vector, pdg_code):
    """
    Extract theta, phi angles and kinetic energy from particle momentum vector.

    Parameters
    ----------
    momentum_vector : jnp.ndarray
        3D momentum vector [px, py, pz] in MeV/c
    pdg_code : int
        PDG particle code (13 for muon, 211 for pion, etc.)

    Returns
    -------
    tuple
        (theta, phi, kinetic_energy) where:
        - theta: polar angle from z-axis in radians [0, pi]
        - phi: azimuthal angle in xy-plane in radians [0, 2*pi]
        - kinetic_energy: kinetic energy in MeV

    Notes
    -----
    - theta = 0 corresponds to positive z-direction
    - phi = 0 corresponds to positive x-direction
    - Uses relativistic energy-momentum relation: E^2 = p^2 + m^2
    - Kinetic energy = Total energy - Rest mass
    """
    # Get particle mass
    if pdg_code not in PARTICLE_MASSES:
        raise ValueError(f"Unknown PDG code: {pdg_code}. Supported codes: {list(PARTICLE_MASSES.keys())}")

    mass = PARTICLE_MASSES[pdg_code]

    # Extract momentum components
    px, py, pz = momentum_vector[0], momentum_vector[1], momentum_vector[2]

    # Calculate momentum magnitude
    p_magnitude = jnp.sqrt(px**2 + py**2 + pz**2)

    # Calculate polar angle theta (angle from z-axis)
    # theta = arccos(pz / |p|)
    theta = jnp.arccos(jnp.clip(pz / p_magnitude, -1.0, 1.0))

    # Calculate azimuthal angle phi (angle in xy-plane from x-axis)
    # phi = arctan2(py, px), adjusted to [0, 2*pi] range
    phi = jnp.arctan2(py, px)
    phi = jnp.where(phi < 0, phi + 2*jnp.pi, phi)  # Ensure phi is in [0, 2*pi]

    # Calculate total energy using relativistic energy-momentum relation
    # E^2 = p^2 + m^2
    total_energy = jnp.sqrt(p_magnitude**2 + mass**2)

    # Calculate kinetic energy
    kinetic_energy = total_energy - mass

    return theta, phi, kinetic_energy


def analyze_event_kinematics(event_data):
    """
    Wrapper function to analyze kinematics for all tracks in an event.

    Parameters
    ----------
    event_data : dict
        Event data dictionary containing 'P' (momentum) and 'PDG' arrays
        Expected format from read_event_file():
        - 'P': shape (N, 3) momentum vectors in MeV/c
        - 'PDG': shape (N,) PDG particle codes

    Returns
    -------
    dict
        Dictionary containing kinematic analysis results:
        - 'theta': polar angles in radians, shape (N,)
        - 'phi': azimuthal angles in radians, shape (N,)
        - 'kinetic_energy': kinetic energies in MeV, shape (N,)
        - 'momentum_magnitude': momentum magnitudes in MeV/c, shape (N,)
        - 'particle_types': list of particle type strings
        - 'n_tracks': number of tracks

    Example
    -------
    >>> # Load event data
    >>> event_data = read_event_file('event_0.h5')
    >>> # Analyze kinematics
    >>> kinematics = analyze_event_kinematics(event_data)
    >>> print(f"Track 0: theta={kinematics['theta'][0]:.3f} rad, "
    ...       f"phi={kinematics['phi'][0]:.3f} rad, "
    ...       f"KE={kinematics['kinetic_energy'][0]:.1f} MeV")
    """
    if 'P' not in event_data or 'PDG' not in event_data:
        raise ValueError("Event data must contain 'P' (momentum) and 'PDG' arrays")

    momentum_array = jnp.array(event_data['P'])  # Shape: (N, 3)
    pdg_array = jnp.array(event_data['PDG'])     # Shape: (N,)

    n_tracks = momentum_array.shape[0]

    # Initialize output arrays
    theta_array = jnp.zeros(n_tracks)
    phi_array = jnp.zeros(n_tracks)
    kinetic_energy_array = jnp.zeros(n_tracks)
    momentum_magnitude_array = jnp.zeros(n_tracks)

    # Process each track
    for i in range(n_tracks):
        theta, phi, kinetic_energy = momentum_to_angles_and_energy(
            momentum_array[i], int(pdg_array[i])
        )

        theta_array = theta_array.at[i].set(theta)
        phi_array = phi_array.at[i].set(phi)
        kinetic_energy_array = kinetic_energy_array.at[i].set(kinetic_energy)
        momentum_magnitude_array = momentum_magnitude_array.at[i].set(
            jnp.sqrt(jnp.sum(momentum_array[i]**2))
        )

    # Convert PDG codes to particle type strings
    particle_types = []
    for pdg in pdg_array:
        if pdg == 13:
            particle_types.append("muon")
        elif pdg == -13:
            particle_types.append("anti-muon")
        elif pdg == 211:
            particle_types.append("pi+")
        elif pdg == -211:
            particle_types.append("pi-")
        elif pdg == 111:
            particle_types.append("pi0")
        elif pdg == 11:
            particle_types.append("electron")
        elif pdg == -11:
            particle_types.append("positron")
        elif pdg == 22:
            particle_types.append("photon")
        elif pdg == 2212:
            particle_types.append("proton")
        elif pdg == 2112:
            particle_types.append("neutron")
        else:
            particle_types.append(f"unknown_{pdg}")

    return {
        'theta': theta_array,
        'phi': phi_array,
        'kinetic_energy': kinetic_energy_array,
        'momentum_magnitude': momentum_magnitude_array,
        'particle_types': particle_types,
        'n_tracks': n_tracks
    }


def print_event_kinematics(event_data, show_details=True):
    """
    Print kinematic analysis results for an event in a formatted way.

    Parameters
    ----------
    event_data : dict
        Event data dictionary containing 'P' and 'PDG' arrays
    show_details : bool, optional
        Whether to show detailed information for each track, by default True
    """
    kinematics = analyze_event_kinematics(event_data)

    print("\n" + "="*70)
    print("KINEMATIC ANALYSIS")
    print("="*70)
    print(f"Number of tracks: {kinematics['n_tracks']}")

    if show_details:
        print("\nTrack Details:")
        print("-" * 95)
        print(f"{'Track':<6}{'Particle':<12}{'P_mag':<12}{'KE':<12}{'Theta':<12}{'Phi':<12}{'Direction':<25}")
        print(f"{'#':<6}{'Type':<12}{'(MeV/c)':<12}{'(MeV)':<12}{'(rad)':<12}{'(rad)':<12}{'(unit vector)':<25}")
        print("-" * 95)

        for i in range(kinematics['n_tracks']):
            # Calculate unit direction vector
            theta = kinematics['theta'][i]
            phi = kinematics['phi'][i]
            direction = jnp.array([
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta)
            ])

            print(f"{i:<6}{kinematics['particle_types'][i]:<12}"
                  f"{kinematics['momentum_magnitude'][i]:<12.1f}"
                  f"{kinematics['kinetic_energy'][i]:<12.1f}"
                  f"{theta:<12.3f}"
                  f"{phi:<12.3f}"
                  f"[{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]")

    # Summary statistics
    print(f"\nSummary Statistics:")
    print(f"Mean kinetic energy: {jnp.mean(kinematics['kinetic_energy']):.1f} MeV")
    print(f"Mean momentum magnitude: {jnp.mean(kinematics['momentum_magnitude']):.1f} MeV/c")
    print(f"Theta range: {jnp.min(kinematics['theta']):.3f} - {jnp.max(kinematics['theta']):.3f} rad")
    print(f"Phi range: {jnp.min(kinematics['phi']):.3f} - {jnp.max(kinematics['phi']):.3f} rad")

    # Particle type distribution
    from collections import Counter
    particle_counts = Counter(kinematics['particle_types'])
    print(f"\nParticle Distribution:")
    for particle_type, count in particle_counts.items():
        print(f"  {particle_type}: {count}")

    print("="*70)


# ---------------------------------------------------------------------------
# V3 format: four-file per-event-group HDF5 (sensor / inst / seg / labl).
# See docs/LUCID_DATASET.md for the full schema.
# ---------------------------------------------------------------------------

_GZIP_OPTS = dict(compression='gzip', compression_opts=4)
_V3_FORMAT_VERSION = 5

# per_interaction/source_type encoding
SOURCE_TYPE_PARTICLES = 0
SOURCE_TYPE_GENIE     = 1

# t0 draw half-window (ns). Applied symmetrically per interaction:
# t0 ~ Uniform(-T0_HALF_WINDOW_NS, +T0_HALF_WINDOW_NS). Wide enough to
# (a) randomize absolute event time so downstream models can't assume
# t=0 is the true start, and (b) cover a ±250 ns pile-up window.
T0_HALF_WINDOW_NS = 250.0


def _source_type_code(primary_source):
    """Map config's ``primary_source`` string to the per_interaction/source_type int."""
    if primary_source == 'genie':
        return SOURCE_TYPE_GENIE
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
    matches what ``seg.h5`` actually stores.

    Segment-to-track mapping uses cumulative ``n_segments`` over tracks
    in dict-insertion order — the same invariant ``save_seg_event_v3``
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
    # save_seg_event_v3 invariant). A track is contained iff every owned
    # segment is contained. Zero-segment tracks → True (no-evidence).
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

        # seg → local track idx for in-range segments.
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

        # Particle: True iff it owns ≥1 track AND every owned segment is contained.
        per_particle_arr = particle_has_track & particle_and

    # ------------------------------------------------------------------
    # Interaction: AND over its particles. Empty interaction → False.
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

    # Event: AND over interactions. Empty event → False.
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




def _aggregate_from_photon_records(
        photon_qe_weight,
        photon_qe_time,
        photon_sensor_idx,
        photon_seg_idx_filtered,
        photon_particle_idx,
        n_particles, n_sensors):
    """One-pass host aggregation from per-photon flat lists.

    Replaces the dense ``(n_segments, n_sensors)`` PE/T tensors plus the
    JIT inst aggregator plus the ``np.nonzero(PE_seg)`` sparsifier with a
    single numpy lexsort+reduceat pass. Two groupbys, both keyed by
    ``(group, sensor_idx)``:

      * ``(particle_idx, sensor_idx)`` → dense ``(n_particles, n_sensors)``
        ``PE_per_particle`` and ``T_per_particle`` for inst.h5;
      * ``(seg_idx_filtered, sensor_idx)`` → sparse triplets
        ``{segment_idx, sensor_idx, PE, T}`` for seg.h5
        (``segment_sensor_hits``).

    PE per group is the sum of QE-passing photon weights; T per group is
    the min of unsmeared QE-filtered arrival times. ``qe_weight > 0`` is
    the QE-pass mask (failed photons have weight 0 and time +inf from
    ``_qe_roll``). Orphans (``-1``) drop out.

    Returns
    -------
    dict with keys 'PE_per_particle', 'T_per_particle',
    'segment_sensor_hits' (the latter is ``{'segment_idx', 'sensor_idx',
    'PE', 'T'}`` arrays). When no QE-passing photon points at a given
    group axis, returns zero-filled / empty outputs respectively.
    """
    PE_pp = np.zeros((n_particles, n_sensors), dtype=np.float32)
    T_pp  = np.zeros((n_particles, n_sensors), dtype=np.float32)
    seg_hits = {
        'segment_idx': np.empty(0, dtype=np.int32),
        'sensor_idx':  np.empty(0, dtype=np.uint16),
        'PE':          np.empty(0, dtype=np.float32),
        'T':           np.empty(0, dtype=np.float32),
    }

    if photon_qe_weight.size == 0:
        return {'PE_per_particle': PE_pp, 'T_per_particle': T_pp,
                'segment_sensor_hits': seg_hits}

    qe_pass = photon_qe_weight > 0

    # ---- inst.h5: groupby (particle_idx, sensor_idx) ----
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

    # ---- seg.h5 sparse triplets: groupby (seg_idx_filtered, sensor_idx) ----
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

    return {'PE_per_particle': PE_pp, 'T_per_particle': T_pp,
            'segment_sensor_hits': seg_hits}


def aggregate_inst_from_segments(pe_per_seg, t_per_seg,
                                  track_idx_per_segment,
                                  particle_idx_per_track,
                                  n_particles, n_sensors):
    """Aggregate per-(segment, sensor) PE/T into per-particle PE/T.

    inst.h5's per-particle decomposition is a downstream view of
    ``seg/event_NNN/sensor_hits/`` plus the segment→track→particle map.
    PE per particle is the sum over the particle's segments; T per
    particle is the min over the particle's segments' first-arrival
    times.

    Kept as the byte-identity oracle for ``test_aggregator_matches_oracle``
    — the production single-vertex / pile-up paths use
    ``_aggregate_from_photon_records`` instead, which works directly from
    the per-photon flat lists.

    The "no hit = 0" sentinel pattern is preserved exactly: 0 in the
    input means "no photons hit this (seg, sensor)"; we route
    0 → +inf → segment_min → back to 0.

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
        ancestor (their segments are dropped from inst.h5).
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
    #      np.add.at / np.minimum.at. n_tracks ≪ n_segments so the
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

    # Route per-track → per-particle, dropping orphaned tracks (-1).
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


def _write_common_config_attrs(f, config_meta):
    """Create ``config/`` group with provenance attrs common to all v3 files."""
    cfg = f.require_group('config')
    cfg.attrs['format_version'] = _V3_FORMAT_VERSION
    cfg.attrs['n_events'] = int(config_meta['n_events'])
    cfg.attrs['git_commit'] = str(config_meta.get('git_commit', 'unknown'))
    cfg.attrs['run_id'] = str(config_meta['run_id'])
    cfg.attrs['dataset_name'] = str(config_meta['dataset_name'])
    cfg.attrs['file_index'] = int(config_meta.get('file_index', 0))
    cfg.attrs['source_file'] = str(config_meta['source_file'])
    cfg.attrs['lucid_master_seed'] = int(config_meta['lucid_master_seed'])
    cfg.attrs['photonsim_seed'] = int(config_meta.get('photonsim_seed', -1))
    return cfg


def write_sensor_config_v3(f, config_meta, source_event_idx, sensor_positions):
    """Write the config/ group of a sensor v3 file."""
    cfg = _write_common_config_attrs(f, config_meta)
    cfg.attrs['n_sensors'] = int(config_meta['n_sensors'])
    cfg.attrs['detector_type'] = str(config_meta['detector_type'])
    cfg.attrs['material'] = str(config_meta['material'])
    cfg.attrs['smearing_applied'] = bool(config_meta['smearing_applied'])
    cfg.attrs['smearing_charge_function'] = str(
        config_meta.get('smearing_charge_function', 'default'))
    cfg.attrs['smearing_time_function'] = str(
        config_meta.get('smearing_time_function', 'default'))
    cfg.create_dataset('source_event_idx',
                       data=np.asarray(source_event_idx, dtype=np.uint32),
                       **_GZIP_OPTS)
    cfg.create_dataset('sensor_positions',
                       data=np.asarray(sensor_positions, dtype=np.float32),
                       **_GZIP_OPTS)


def write_inst_config_v3(f, config_meta, source_event_idx, sensor_positions):
    """Write the config/ group of an inst v3 file."""
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


def write_seg_config_v3(f, config_meta, source_event_idx):
    """Write the config/ group of a seg v3 file."""
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


def write_labl_config_v3(f, config_meta, source_event_idx):
    """Write the config/ group of a labl v3 file."""
    cfg = _write_common_config_attrs(f, config_meta)
    label_names = list(config_meta.get('label_names', ['category']))
    cfg.attrs['label_names'] = np.array(label_names, dtype=h5py.string_dtype())
    cfg.create_dataset('source_event_idx',
                       data=np.asarray(source_event_idx, dtype=np.uint32),
                       **_GZIP_OPTS)


def _event_group_name(seq_idx):
    return f'event_{int(seq_idx):03d}'


def save_sensor_event_v3(f, event_dict, seq_idx):
    """Write a single event_NNN/ group to an already-open sensor v3 file.

    ``event_dict`` must contain: ``source_event_idx``, ``PE_reco``,
    ``T_reco``. Times in ``T_reco`` are expected in absolute detector
    frame — the caller applies per-interaction t0 shifts before calling
    this writer; the writer does not shift times further.
    """
    grp = f.create_group(_event_group_name(seq_idx))
    grp.attrs['source_event_idx'] = int(event_dict['source_event_idx'])

    pe = np.asarray(event_dict['PE_reco'], dtype=np.float32)
    t = np.asarray(event_dict['T_reco'], dtype=np.float32)

    # A "hit" is a sensor with real charge: pe > 0. SK-like charge smearing
    # preserves zero (sigma=0 when counts=0), so pe == 0 ⇒ no photon ever
    # reached this sensor. Drop such sensors even if smear_times fabricated
    # a noisy time for them. The isfinite & <1e5 checks catch smear_times'
    # 1e6 non-finite sentinel. Absolute time can legitimately be negative
    # (t0 ∈ [-250, +250] ns), so no lower bound.
    mask = (pe > 0) & np.isfinite(t) & (t < 1e5)
    indices = np.where(mask)[0].astype(np.uint16)
    pe_sparse = pe[mask].astype(np.float32)
    t_sparse = np.where(np.isfinite(t[mask]), t[mask], np.float32(0.0)).astype(np.float32)

    grp.attrs['n_hits'] = int(indices.size)
    grp.create_dataset('sensor_idx', data=indices, **_GZIP_OPTS)
    grp.create_dataset('PE', data=pe_sparse, **_GZIP_OPTS)
    grp.create_dataset('T', data=t_sparse, **_GZIP_OPTS)


def save_inst_event_v3(f, event_dict, seq_idx):
    """Write a single event_NNN/ group to an already-open inst v3 file.

    Stores the per-particle PE/T decomposition as FK rows keyed by
    ``particle_idx`` (local to the event). Times in ``T_per_particle``
    are expected in absolute detector frame — no shift is applied here.
    """
    grp = f.create_group(_event_group_name(seq_idx))
    grp.attrs['source_event_idx'] = int(event_dict['source_event_idx'])
    grp.attrs['n_particles'] = int(event_dict['n_particles'])

    pe_pp = np.asarray(event_dict['PE_per_particle'], dtype=np.float32)
    t_pp = np.asarray(event_dict['T_per_particle'], dtype=np.float32)
    n_p = pe_pp.shape[0]

    particle_idx_parts, sensor_idx_parts, pe_parts, t_parts = [], [], [], []
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

    def _cat(xs, dtype):
        return np.concatenate(xs).astype(dtype) if xs else np.array([], dtype=dtype)

    particle_idx_arr = _cat(particle_idx_parts, np.int32)
    sensor_idx_arr = _cat(sensor_idx_parts, np.uint16)
    pe_arr = _cat(pe_parts, np.float32)
    t_arr = _cat(t_parts, np.float32)

    grp.attrs['n_particle_hits'] = int(particle_idx_arr.size)
    grp.create_dataset('particle_idx', data=particle_idx_arr, **_GZIP_OPTS)
    grp.create_dataset('sensor_idx', data=sensor_idx_arr, **_GZIP_OPTS)
    grp.create_dataset('PE', data=pe_arr, **_GZIP_OPTS)
    grp.create_dataset('T', data=t_arr, **_GZIP_OPTS)


def save_seg_event_v3(f, event_dict, seq_idx):
    """Write a single event_NNN/ group to an already-open seg v3 file.

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

    # Optional segment <-> sensor correspondence map. Mirrors inst.h5's flat
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
        sh.attrs['n_segment_hits'] = int(len(seg_sen['PE']))
        grp.attrs['has_segment_sensor_map'] = True


def save_labl_event_v3(f, event_dict, seq_idx):
    """Write a single event_NNN/ group to an already-open labl v5 file.

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
    ``detector_bounds``. AND-composes from segment → particle →
    interaction → event. Empty subsets (interaction with zero
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
    """Populate the per_interaction/ subgroup for v5.

    One row per interaction — a single G4 event in non-pile-up events
    (so always one row) and one vertex stream in pile-up events (N rows
    for N-way pile-up). Each interaction bundles every primary fired in
    that G4 event plus the full ancestry cascade of each, collapsing a
    multi-primary GENIE interaction or a multi-primary particle-gun
    shot into a single row.

    Fields:
      * ``source_type``           (uint8)   — 0=particles, 1=genie.
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


def derive_particle_interaction_idx(event_dict, track_interaction=None):
    """For each particle, return the interaction index of its primary ancestor.

    Uses each particle's last-in-genealogy track_id (== its primary
    track) and looks up that track's ``interaction`` rank. Particles
    with no genealogy or whose primary isn't in the tracks table get -1.

    Parameters
    ----------
    event_dict : dict
        Must carry ``meaningful_tracks`` (dict of track_id → info) and
        ``particles`` (list with a ``genealogy`` key).
    track_interaction : np.ndarray, optional
        Cached output of ``derive_track_ancestor_and_interaction``; if
        None, recomputed.
    """
    tracks = event_dict.get('meaningful_tracks', {})
    particles = event_dict.get('particles', [])
    if not particles:
        return np.array([], dtype=np.int32)
    if not tracks:
        return np.full(len(particles), -1, dtype=np.int32)
    if track_interaction is None:
        _, track_interaction = derive_track_ancestor_and_interaction(event_dict)
    tid_to_interaction = {
        int(tid): int(track_interaction[i])
        for i, tid in enumerate(tracks.keys())
    }
    out = np.full(len(particles), -1, dtype=np.int32)
    for i, particle in enumerate(particles):
        gen = particle.get('genealogy') or []
        if gen:
            out[i] = tid_to_interaction.get(int(gen[-1]), -1)
    return out


def list_events_v3(filename):
    """Return the ``config/source_event_idx`` array from a v3 file."""
    with h5py.File(filename, 'r') as f:
        return np.asarray(f['config/source_event_idx'][:])


def _v3_group_to_dict(grp):
    """Recursively copy attrs + datasets + subgroups into a plain dict."""
    out = {}
    for key, value in grp.attrs.items():
        out[key] = value
    for key in grp.keys():
        item = grp[key]
        if isinstance(item, h5py.Dataset):
            out[key] = item[()]
        else:  # subgroup
            out[key] = _v3_group_to_dict(item)
    return out


def _read_v3_event(filename, event_idx):
    """Return the event_NNN/ group contents as a dict keyed by dataset/attr name."""
    with h5py.File(filename, 'r') as f:
        name = f'event_{int(event_idx):03d}'
        if name not in f:
            raise KeyError(
                f"Event group {name!r} not found in {filename}. "
                f"Available: {sorted(k for k in f.keys() if k.startswith('event_'))[:5]} ...")
        return _v3_group_to_dict(f[name])


def read_sensor_event_v3(filename, event_idx):
    """Read event ``event_idx`` from a sensor v3 file."""
    return _read_v3_event(filename, event_idx)


def read_inst_event_v3(filename, event_idx):
    """Read event ``event_idx`` from an inst v3 file."""
    return _read_v3_event(filename, event_idx)


def read_seg_event_v3(filename, event_idx):
    """Read event ``event_idx`` from a seg v3 file."""
    return _read_v3_event(filename, event_idx)


def read_labl_event_v3(filename, event_idx):
    """Read event ``event_idx`` from a labl v5 file.

    The returned dict contains top-level attrs plus four subdicts:
    ``per_event`` (contained, t0 = min per_interaction/t0),
    ``per_interaction`` (source_type, t0, vertex_{x,y,z}, n_primaries,
    n_particles, neutrino_pdg, neutrino_energy_MeV, contained, and
    CSR-encoded primary_{track_ids,pdgs,energies}_{offsets,data}),
    ``per_particle`` (category, contained, genealogy CSR,
    interaction_idx), and ``per_track`` (track_id, parent_id, pdg,
    initial_energy, n_cherenkov, particle_idx, ancestor, interaction).
    """
    return _read_v3_event(filename, event_idx)


