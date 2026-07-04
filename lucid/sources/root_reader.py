"""PhotonSim ROOT file readers.

Functions that read photon, particle, and segment data from PhotonSim
ROOT files. The full wrapper ``read_particle_data_from_photonsim``
calls into ``event_builder._derive_views_from_segments`` for
categorization.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = [
    "get_max_photons_per_particle",
    "read_photon_data_from_photonsim",
    "read_event_data_from_photonsim",
    "read_particle_data_from_photonsim",
]


def read_event_data_from_photonsim(root_file_path, entry_index,
                                   emission_processes=("cherenkov",),
                                   medium_params=None, rng=None):
    """Read injectable per-photon arrays for one event, for the requested
    emission processes, ready to feed the ``is_data`` simulator.

    This is the data-mode analogue of the forward Cherenkov+scintillation path:
    Cherenkov photons are loaded from ``OpticalPhotonsRaw`` and scintillation
    photons are **expanded from the dE/dx segments** (``Segment_*``) via
    :func:`lucid.sources.scintillation_photons.expand_segments_to_photons` and
    **appended to the Cherenkov list**, so the simulator ray-traces both streams
    in a single call.

    Parameters
    ----------
    root_file_path : str
    entry_index : int
    emission_processes : tuple of str
        Any non-empty subset of ``("cherenkov", "scintillation")``. Pick a
        single process to get that process's photons alone (e.g. for a
        per-process fraction map), or both for the full event.
    medium_params : Mapping or None
        Required when ``"scintillation"`` is requested — the scintillation
        scalars (see :func:`expand_segments_to_photons`). Build it with
        :func:`lucid.sources.scintillation_photons.scintillation_medium_params`.
    rng : numpy.random.Generator or None
        Required when ``"scintillation"`` is requested (per-event RNG).

    Returns
    -------
    dict
        Combined per-photon arrays, in the SAME metre convention as
        :func:`read_photon_data_from_photonsim` (the caller applies the
        m->cm boundary and any rotation/translation exactly as for Cherenkov):

        - ``photon_origins``           : (N, 3) float32, meters
        - ``photon_directions``        : (N, 3) float32, unit vectors
        - ``photon_times``             : (N,)   float32, ns (G4 frame)
        - ``wavelengths``              : (N,)   float32, nm
        - ``photon_emission_process``  : (N,)   int8, per-photon process tag
          (``EMISSION_PROCESS_CHERENKOV`` / ``EMISSION_PROCESS_SCINTILLATION``)
        - ``N``                        : int, total photon count
        - ``energy``                   : float, primary energy (MeV)
    """
    from lucid.sources.scintillation_photons import expand_segments_to_photons
    from lucid.sources.writer import (
        EMISSION_PROCESS_CHERENKOV, EMISSION_PROCESS_SCINTILLATION)

    procs = tuple(emission_processes)
    valid = {"cherenkov", "scintillation"}
    bad = set(procs) - valid
    if bad:
        raise ValueError(f"unknown emission_processes {sorted(bad)}; "
                         f"valid: {sorted(valid)}")
    if not procs:
        raise ValueError("emission_processes must be non-empty")

    raw = _read_event_raw(root_file_path, entry_index)

    O, D, T, W, P = [], [], [], [], []
    if "cherenkov" in procs:
        nc = int(np.asarray(raw["photon_origins"]).shape[0])
        O.append(np.asarray(raw["photon_origins"], dtype=np.float32))
        D.append(np.asarray(raw["photon_directions"], dtype=np.float32))
        T.append(np.asarray(raw["photon_times"], dtype=np.float32))
        W.append(np.asarray(raw["photon_wavelengths"], dtype=np.float32))
        P.append(np.full(nc, EMISSION_PROCESS_CHERENKOV, dtype=np.int8))
    if "scintillation" in procs:
        if medium_params is None or rng is None:
            raise ValueError("emission_processes includes 'scintillation' but "
                             "medium_params and/or rng were not provided")
        sc = expand_segments_to_photons(raw["segments_raw"], medium_params, rng)
        O.append(np.asarray(sc["photon_origins"], dtype=np.float32))
        D.append(np.asarray(sc["photon_directions"], dtype=np.float32))
        T.append(np.asarray(sc["photon_times"], dtype=np.float32))
        W.append(np.asarray(sc["photon_wavelengths"], dtype=np.float32))
        P.append(np.asarray(sc["photon_emission_process"], dtype=np.int8))

    origins = (np.concatenate(O, axis=0) if O
               else np.zeros((0, 3), dtype=np.float32))
    directions = (np.concatenate(D, axis=0) if D
                  else np.zeros((0, 3), dtype=np.float32))
    times = np.concatenate(T) if T else np.zeros(0, dtype=np.float32)
    wls = np.concatenate(W) if W else np.zeros(0, dtype=np.float32)
    procarr = np.concatenate(P) if P else np.zeros(0, dtype=np.int8)

    return {
        "photon_origins": jnp.asarray(origins),
        "photon_directions": jnp.asarray(directions),
        "photon_times": jnp.asarray(times),
        "wavelengths": jnp.asarray(wls),
        "photon_emission_process": procarr,
        "N": int(origins.shape[0]),
        "energy": float(raw["primary_energy"]),
    }


def _read_photons_for_event(raw_tree, event_idx):
    """Stitch OpticalPhotonsRaw chunks for one event into flat numpy arrays.

    OpticalPhotonsRaw stores per-photon scalars as fixed-K chunks (one
    TTree entry = one chunk of up to 100 k photons). Each entry has
    EventID + ChunkStartID stamped on it. This helper returns the
    per-event flat (NPhotons, 3)/(NPhotons,) arrays the rest of LUCiD
    consumes — units converted (PhotonSim mm -> LUCiD m); chunk concat
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


def read_particle_data_from_photonsim(root_file_path, entry_index):
    """Backward-compatible wrapper: read raw + derive views without per-segment data.

    Equivalent to the legacy implementation; preserves the exact return-dict
    shape for any external caller. The data-mode driver no longer routes
    through this wrapper — it calls :func:`_read_event_raw`,
    :func:`_trace_event_bucketed`, and :func:`_derive_views_from_segments`
    directly so the per-(segment, sensor) tensor that the kernel emits can
    flow downstream.
    """
    from lucid.sources.event_builder import _derive_views_from_segments

    raw = _read_event_raw(root_file_path, entry_index)
    return _derive_views_from_segments(raw, photon_records=None)
