"""The labels of a dataset event: what its particles are, which segments belong together, who
owns what, and what is contained.

* **Categorization** (a port of PhotonSim's per-track decision tree): ``categorize_event`` with
  its ``CAT_*`` codes, ``derive_meaningful_tracks`` (Cherenkov producers and their ancestors),
  ``filter_segments_to_meaningful`` and ``bucket_photons_by_segment``. The output matches the
  legacy C++ ``TrackInfo_Category`` / ``Particle_*`` branches except for ``TrackInfo_SubID``
  within one (parent, decay process, time) tie: Geant4 processes multi-secondary decays in stack
  (LIFO) order, this port in track-id order. The sub-id is metadata no dataset file carries.
  The meaningful-track set mirrors the C++ filter in ``DataManager::EndEvent``.
* **Segment grouping** (a port of PhotonSim's segment merger, ``DataManager.cc:441-527``):
  ``assign_group_ids`` groups raw G4 steps the way the merger did, byte-for-byte; its inputs are
  in **mm**, as in the C++, so the length comparisons are bit-identical.
* **Ownership**: ``derive_particle_idx_per_track``, ``derive_track_ancestor_and_interaction``,
  ``derive_particle_interaction_idx``.
* **Containment**: ``compute_containment``.

The viewer and ``lucid.production.derived_views`` read the ``CAT_*`` codes back.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from lucid.pdg import PDG_MASS_MEV

__all__ = [
    "CAT_NONE", "CAT_PRIMARY", "CAT_DECAY_ELECTRON", "CAT_SECONDARY_PION", "CAT_GAMMA",
    "TrackEntry",
    "CategorizationResult",
    "categorize_event",
    "derive_meaningful_tracks",
    "filter_segments_to_meaningful",
    "bucket_photons_by_segment",
    "assign_group_ids",
    "derive_particle_idx_per_track",
    "derive_track_ancestor_and_interaction",
    "derive_particle_interaction_idx",
    "compute_containment",
]


# ---- Decision-tree constants (exact mirror of SteppingAction.cc) ----

CAT_NONE = -1            # not yet categorized / never categorized
CAT_PRIMARY = 0          # kPrimary
CAT_DECAY_ELECTRON = 1   # kDecayElectron
CAT_SECONDARY_PION = 2   # kSecondaryPion
CAT_GAMMA = 3            # kGamma

DECAY_PROCESSES = ("Decay", "muMinusCaptureAtRest")
INELASTIC_KEY = "Inelastic"
DEFLECTION_KEY = "Deflection"
DECAY_E_KE_MIN_MEV = 1.0
SECONDARY_PI_P_MIN_MEV = 195.0
PARENT_DECAY_E = ("mu-", "mu+", "pi-", "pi+")
PARENT_GAMMA_PI0 = ("pi0",)

# Charged-pion mass — used to convert TrackInfo_Energy (KE in MeV) into
# the 3-momentum magnitude that the C++ test checks
# (track->GetMomentum().mag()).
M_PI_CHARGED_MEV = PDG_MASS_MEV[211]

# PDG → Geant4 particle name mapping. PhotonSim's TrackInfo struct
# carries particleName as a G4String, but the ROOT export only ships
# pdg (TrackInfo_PDG) and the creator process. The decision tree
# branches on G4 particle names, so we recover them from PDG codes
# for the species that matter to the categorizer.
_PDG_TO_NAME = {
    11: "e-", -11: "e+",
    13: "mu-", -13: "mu+",
    22: "gamma",
    111: "pi0", 211: "pi+", -211: "pi-",
}


def pdg_to_g4name(pdg: int) -> str:
    """Map PDG code → Geant4 particle name. Returns 'pdg<N>' for codes
    we don't enumerate (these codes never trigger a category branch)."""
    return _PDG_TO_NAME.get(int(pdg), f"pdg{int(pdg)}")


@dataclass
class TrackEntry:
    """Per-track input bundle for the categorizer.

    Fields below are the minimum the C++ decision tree reads. Other
    columns from ``TrackInfo_*`` (position, direction, time) are
    preserved verbatim downstream by the existing reader and are not
    needed here.
    """
    track_id: int
    parent_id: int
    pdg: int
    ke_mev: float                 # TrackInfo_Energy
    creator_process: str          # TrackInfo_CreatorProcess (Stage 1)
    # filled by the categorizer
    category: int = CAT_NONE
    sub_id: int = -1


@dataclass
class CategorizationResult:
    """Output of ``categorize_event`` — covers everything the legacy
    ROOT branches expressed."""
    # category per track_id (in arbitrary order). The legacy
    # ``TrackInfo_Category`` array has the same values keyed by the
    # tracksToStore set (categorized + their direct parents), which is
    # rebuilt from this map by ``build_track_info_arrays``.
    category_by_track_id: Dict[int, int] = field(default_factory=dict)
    sub_id_by_track_id: Dict[int, int] = field(default_factory=dict)
    # Compressed genealogy chain (categorized tracks only) per
    # *unique* genealogy. Particle index is the position in this
    # ordered list — ordering matches the C++
    # std::map<vector<int>, ...> iteration (lexicographic by chain).
    genealogies: List[Tuple[int, ...]] = field(default_factory=list)
    # Extended genealogy (full chain through meaningful tracks),
    # parallel to ``genealogies``.
    ext_genealogies: List[Tuple[int, ...]] = field(default_factory=list)
    # Maps a meaningful track's id → which entry in ``genealogies``
    # it lives under (or -1 if not part of any categorized chain).
    particle_idx_by_meaningful_track: Dict[int, int] = field(default_factory=dict)


def categorize_event(
    track_info_rows: List[TrackEntry],
    meaningful_track_parent_pdg: Optional[Dict[int, Tuple[int, int]]] = None,
    cherenkov_count_by_mt_track: Optional[Dict[int, int]] = None,
) -> CategorizationResult:
    """Mirror ``SteppingAction.cc:167-265`` for one event's worth of tracks.

    Tracks are processed in track-id order — that matches Geant4's
    chronological track-creation order, which is what the C++ does
    inline at ``UserSteppingAction``. Categorizing parents before
    children keeps the inheritance check
    ("parent is a categorized pion?") well-defined.

    Parameters
    ----------
    track_info_rows
        One ``TrackEntry`` per track in the ``TrackInfo_*`` arrays.
        The C++ writes only categorized tracks + their direct parents
        into ``TrackInfo_*``, but every track the decision tree needs
        to look up *is* present (a kPrimary's daughters' parents are
        themselves the primary, hence stored).
    meaningful_track_parent_pdg
        Optional map ``track_id → (parent_id, pdg)`` covering tracks
        that may be missing from ``track_info_rows`` but are still
        relevant for the secondary-pion category-parent walk (e.g.,
        an intermediate hadron that produced Cherenkov segments but
        wasn't itself categorized). Built from ``MTrack_*``. ``None``
        means "no extra tracks", which is fine when the legacy ROOT
        output already covered the chain.
    cherenkov_count_by_mt_track
        Optional map ``track_id → n_cherenkov_photons``, built from
        ``MTrack_NCherenkov``. When provided, only meaningful tracks
        with ``n_cherenkov > 0`` contribute particles — matching the
        C++ ``fGenealogyToPhotonIDs`` semantics where a genealogy
        bucket only exists if at least one photon was emitted with
        that bucketed-genealogy. When omitted, every meaningful
        track produces a particle (legacy behaviour, may emit
        spurious zero-photon particles on GENIE-style events with
        primary tracks that have segments but below-threshold β).

    Returns
    -------
    CategorizationResult
    """
    # Index by track_id for O(1) lookups during the decision tree walk.
    by_id: Dict[int, TrackEntry] = {t.track_id: t for t in track_info_rows}

    # Sort track_ids ascending. Geant4 numbers tracks chronologically
    # so this gives parent-before-child order.
    ordered_ids = sorted(by_id.keys())

    next_primary_sub = 0
    next_decay_e_sub = 0
    next_pion_sub = 0
    next_gamma_sub = 0

    for tid in ordered_ids:
        t = by_id[tid]
        process = t.creator_process or ""
        name = pdg_to_g4name(t.pdg)

        # 1. Primary — parent_id == 0
        if t.parent_id == 0:
            t.category = CAT_PRIMARY
            t.sub_id = next_primary_sub
            next_primary_sub += 1
            continue

        # 2. Decay electron — e±, "Decay"|"muMinusCaptureAtRest" from a μ/π parent, KE > 1 MeV
        if name in ("e-", "e+"):
            if process in DECAY_PROCESSES:
                parent = by_id.get(t.parent_id)
                if parent is not None:
                    parent_name = pdg_to_g4name(parent.pdg)
                    if parent_name in PARENT_DECAY_E and t.ke_mev > DECAY_E_KE_MIN_MEV:
                        t.category = CAT_DECAY_ELECTRON
                        t.sub_id = next_decay_e_sub
                        next_decay_e_sub += 1
            continue

        # 3. Gamma from π0 decay
        if name == "gamma":
            if process == "Decay":
                parent = by_id.get(t.parent_id)
                if parent is not None and pdg_to_g4name(parent.pdg) in PARENT_GAMMA_PI0:
                    t.category = CAT_GAMMA
                    t.sub_id = next_gamma_sub
                    next_gamma_sub += 1
            continue

        # 4. Secondary pion — π± from inelastic / deflection / categorized-pion parent
        if name in ("pi+", "pi-"):
            parent = by_id.get(t.parent_id)
            is_from_inelastic = (INELASTIC_KEY in process) or ("inelastic" in process)
            is_from_deflection = DEFLECTION_KEY in process
            is_from_categorized_pion = (
                parent is not None
                and pdg_to_g4name(parent.pdg) in ("pi+", "pi-")
                and parent.category >= 0
            )
            if not (is_from_inelastic or is_from_deflection or is_from_categorized_pion):
                continue
            # Cherenkov-threshold gate: |p| >= 195 MeV/c, with KE in MeV.
            # |p| = sqrt((KE + m)^2 - m^2).
            ke = float(t.ke_mev)
            total_e = ke + M_PI_CHARGED_MEV
            p_mag = (total_e * total_e - M_PI_CHARGED_MEV * M_PI_CHARGED_MEV) ** 0.5
            if p_mag < SECONDARY_PI_P_MIN_MEV:
                continue
            t.category = CAT_SECONDARY_PION
            t.sub_id = next_pion_sub
            next_pion_sub += 1
            # Note: the C++ rewrites ``info.parentTrackID`` to the
            # category-parent on this branch. That re-write affects
            # subsequent calls to ``BuildGenealogy(trackID)`` (which
            # follows ``parentTrackID``), so the genealogy walker we
            # apply below MUST use this same overridden chain.
            cat_parent = t.parent_id
            cat_parent_info = by_id.get(cat_parent)
            while (cat_parent_info is not None
                   and cat_parent_info.category < 0
                   and cat_parent_info.parent_id > 0):
                cat_parent = cat_parent_info.parent_id
                cat_parent_info = by_id.get(cat_parent)
            t.parent_id = cat_parent  # overwrite (mirrors C++ UpdateTrackCategory)
            continue

        # All other tracks: stay CAT_NONE.

    # ---- Build per-track compressed genealogies ----
    # ``BuildGenealogy(track_id)`` walks parent chain via parent_id,
    # keeping only categorized tracks (root-to-leaf order).
    def _walk_compressed(tid: int) -> Tuple[int, ...]:
        chain: List[int] = []
        cur = tid
        # Defensive cycle guard — shouldn't ever fire on G4 output.
        seen = set()
        while cur > 0 and cur not in seen:
            seen.add(cur)
            info = by_id.get(cur)
            if info is None:
                break
            if info.category >= 0:
                chain.insert(0, cur)
            cur = info.parent_id
        return tuple(chain)

    # ---- Build per-track ext genealogies (meaningful-track chain) ----
    # Walks ``meaningful_track_parent_pdg`` only, since
    # ``BuildExtendedGenealogy`` follows ``fAllTrackSegments`` parent
    # chain and breaks when the next track isn't meaningful. Output is
    # sorted-set of meaningful track ids that were on the chain.
    mt_parent = meaningful_track_parent_pdg or {}

    def _walk_extended(leaf: int) -> Tuple[int, ...]:
        chain: List[int] = []
        cur = leaf
        seen = set()
        while cur in mt_parent and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            parent_id_of_cur, _pdg_of_cur = mt_parent[cur]
            cur = parent_id_of_cur
        return tuple(sorted(set(chain)))

    # Group meaningful tracks (= those with at least one Cherenkov
    # segment, equivalently those in ``mt_parent``) by their
    # categorized genealogy. Order across particles is the lexicographic
    # ordering of the genealogy tuple (matches C++
    # std::map<vector<int>, ...> traversal).
    genealogies_seen: Dict[Tuple[int, ...], int] = {}
    particle_idx_by_mtrack: Dict[int, int] = {}
    rows: List[Tuple[Tuple[int, ...], Tuple[int, ...], int]] = []
    # Build (genealogy, ext_genealogy, leaf_track_id) per particle.

    # Walk meaningful tracks (parents may not be categorized — that's
    # why we walk the compressed chain on the leaf and attribute the
    # whole meaningful track to whatever categorized chain it inherits).
    # Tracks with zero Cherenkov photons are skipped to mirror the C++
    # fGenealogyToPhotonIDs semantics — a genealogy bucket only exists
    # if some photon emitted bucketed there. A track with segments but
    # n_cherenkov==0 (e.g. a primary slow proton with sub-threshold β)
    # would otherwise create a spurious zero-photon particle.
    cher = cherenkov_count_by_mt_track or {}
    for mt_id in sorted(mt_parent.keys()):
        if cher and cher.get(mt_id, 0) <= 0:
            particle_idx_by_mtrack[mt_id] = -1
            continue
        gen = _walk_compressed(mt_id)
        if not gen:
            # No categorized ancestor — track contributes no particle.
            particle_idx_by_mtrack[mt_id] = -1
            continue
        if gen not in genealogies_seen:
            ext = _walk_extended(gen[-1])
            genealogies_seen[gen] = len(rows)
            rows.append((gen, ext, gen[-1]))
        particle_idx_by_mtrack[mt_id] = genealogies_seen[gen]

    # Lexicographic-sort rows so iteration order matches std::map.
    # (Actually they were added in mt_id order; the std::map<vector<int>>
    # order is lexicographic, which differs.) Re-key here.
    rows.sort(key=lambda r: r[0])
    sorted_lookup = {row[0]: new_idx for new_idx, row in enumerate(rows)}
    particle_idx_by_mtrack = {
        mt_id: (sorted_lookup[_walk_compressed(mt_id)] if particle_idx_by_mtrack[mt_id] >= 0 else -1)
        for mt_id in particle_idx_by_mtrack
    }

    return CategorizationResult(
        category_by_track_id={tid: t.category for tid, t in by_id.items()},
        sub_id_by_track_id={tid: t.sub_id for tid, t in by_id.items()},
        genealogies=[r[0] for r in rows],
        ext_genealogies=[r[1] for r in rows],
        particle_idx_by_meaningful_track=particle_idx_by_mtrack,
    )


def derive_meaningful_tracks(
    segment_track_id: np.ndarray,
    segment_n_cherenkov: np.ndarray,
    track_info: Dict[int, Dict[str, object]],
) -> Dict[int, Dict[str, object]]:
    """Reconstruct the legacy ``MTrack_*`` table from the all-tracks
    ``Segment_*`` table emitted by Stage 5a PhotonSim.

    "Meaningful" tracks are those that produced ≥1 Cherenkov photon
    plus their ancestors — same definition as the legacy C++ filter at
    ``DataManager::EndEvent``. Output dict shape matches the legacy
    ``meaningful_tracks`` consumed by
    :func:`assign_group_ids`:
    ``track_id → {parent_id, pdg, initial_energy,
    n_cherenkov, segment_offset, n_segments}``.

    The ``segment_offset`` returned here indexes into the **filtered**
    segment array produced by :func:`filter_segments_to_meaningful` —
    ``assign_group_ids`` must be called against the same filtered array.

    Parameters
    ----------
    segment_track_id
        ``Segment_TrackID`` from the PhotonSim ROOT — one entry per
        raw segment, naming the G4 track that emitted it.
    segment_n_cherenkov
        ``Segment_NCherenkov`` — number of Cherenkov photons emitted
        in each raw segment.
    track_info
        Per-track lookup keyed by track id, with at minimum
        ``parent_id``, ``pdg``, ``energy`` (KE in MeV) entries — same
        shape produced by the existing
        ``read_particle_data_from_photonsim`` reader.

    Returns
    -------
    dict
        ``{track_id: {parent_id, pdg, initial_energy,
        n_cherenkov, segment_offset, n_segments}}``. Iteration order
        is track-id ascending (matches the order tracks appear in the
        filtered segment array).
    """
    seg_tid = np.asarray(segment_track_id, dtype=np.int64)
    seg_nch = np.asarray(segment_n_cherenkov, dtype=np.int64)

    if seg_tid.size == 0:
        return {}

    # Per-track segment counts and Cherenkov totals via stable groupby.
    # Tracks appear in seg_tid in track-id-ascending order (PhotonSim's
    # std::map iteration), so ``np.unique`` returns sorted ``unique_tids``
    # with ``first_idx`` already in monotonic order — exactly the group
    # boundaries ``np.add.reduceat`` consumes. One vectorized pass over
    # ``seg_nch`` instead of one (n_seg,) bool-mask + sum per unique id.
    unique_tids, first_idx, counts = np.unique(
        seg_tid, return_index=True, return_counts=True)
    cher_sums = np.add.reduceat(seg_nch, first_idx)
    cher_sum_per_track: Dict[int, int] = {
        int(tid): int(s) for tid, s in zip(unique_tids.tolist(), cher_sums.tolist())
    }

    # Cherenkov producers — tracks with ≥1 Cherenkov segment.
    producers = {tid for tid, c in cher_sum_per_track.items() if c > 0}

    # Walk ancestors via track_info[parent_id] (mirrors the legacy
    # second pass in DataManager::EndEvent). Stop at parent_id <= 0
    # or when the parent is missing from the track_info / segment set.
    meaningful_ids = set(producers)
    track_set = set(int(tid) for tid in unique_tids)
    for leaf in producers:
        cur = leaf
        while cur > 0:
            ti = track_info.get(cur)
            if ti is None:
                break
            par = int(ti.get('parent_id', 0))
            if par <= 0:
                break
            if par in meaningful_ids:
                # Already covered upstream.
                break
            if par not in track_set:
                # Parent has no segment row (untracked) — stop.
                break
            meaningful_ids.add(par)
            cur = par

    # Build the output dict in track-id-ascending order so it matches
    # the order tracks appear in the filtered segment array.
    result: Dict[int, Dict[str, object]] = {}
    cumulative_offset = 0
    for tid_arr_idx, tid in enumerate(unique_tids):
        tid_int = int(tid)
        n_seg = int(counts[tid_arr_idx])
        if tid_int not in meaningful_ids:
            continue
        ti = track_info.get(tid_int, {})
        result[tid_int] = {
            'track_id': tid_int,
            'parent_id': int(ti.get('parent_id', 0)),
            'pdg': int(ti.get('pdg', 0)),
            'initial_energy': float(ti.get('energy', 0.0)),
            'n_cherenkov': cher_sum_per_track[tid_int],
            'segment_offset': cumulative_offset,
            'n_segments': n_seg,
        }
        cumulative_offset += n_seg
    return result


def filter_segments_to_meaningful(
    segment_track_id: np.ndarray,
    meaningful_tracks: Dict[int, Dict[str, object]],
    photon_segment_index: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Filter the all-tracks segment array to the meaningful subset.

    Returns ``(keep_mask, photon_segment_index_remapped)`` so the
    caller can reindex parallel ``Segment_*`` arrays in one shot.

    Parameters
    ----------
    segment_track_id
        ``Segment_TrackID`` from the PhotonSim ROOT.
    meaningful_tracks
        Output of :func:`derive_meaningful_tracks` — keys define the
        meaningful subset.
    photon_segment_index
        Optional ``Photon_SegmentIndex`` from the ROOT (positions in
        the all-tracks segment array). If provided, returned remapped
        to filtered-array positions; non-meaningful-segment photons
        get ``-1`` (which never happens for Cherenkov photons since
        every Cherenkov-producing track is meaningful, but defended).

    Returns
    -------
    keep_mask : np.ndarray (bool)
        Per-segment boolean mask; ``segment_array[keep_mask]`` is the
        filtered array.
    photon_segment_index_remapped : np.ndarray or None
        ``photon_segment_index`` remapped from all-tracks positions to
        filtered-array positions, or ``None`` if input was ``None``.
    """
    seg_tid = np.asarray(segment_track_id, dtype=np.int64)
    if seg_tid.size == 0:
        return (np.array([], dtype=bool),
                None if photon_segment_index is None else np.array([], dtype=np.int32))

    keep_mask = np.zeros(seg_tid.size, dtype=bool)
    if meaningful_tracks:
        meaningful_ids = np.fromiter(meaningful_tracks.keys(), dtype=np.int64)
        keep_mask = np.isin(seg_tid, meaningful_ids)

    if photon_segment_index is None:
        return keep_mask, None

    # cumsum-based remap: filtered_pos = cumsum(keep_mask) - 1 at each
    # all-tracks index. Non-meaningful segments get -1.
    full_to_filtered = np.cumsum(keep_mask, dtype=np.int64) - 1
    full_to_filtered = np.where(keep_mask, full_to_filtered, -1).astype(np.int64)

    psi = np.asarray(photon_segment_index, dtype=np.int64)
    out = np.where(psi >= 0, full_to_filtered[psi], -1).astype(np.int32)
    return keep_mask, out


def bucket_photons_by_segment(
    photon_segment_index: np.ndarray,
    segment_track_id: np.ndarray,
    particle_idx_by_meaningful_track: Dict[int, int],
) -> np.ndarray:
    """For each photon, return its particle index (or -1).

    ``Photon_SegmentIndex[p]`` points into the global segment table;
    ``segment_track_id[seg]`` is the meaningful track that emitted that
    segment; ``particle_idx_by_meaningful_track[track]`` is the
    categorized-particle bucket that track inherits.

    Photons whose segment id is ``-1`` (sentinel) or whose owning track
    has no categorized ancestor (``particle_idx == -1``) get ``-1`` —
    those photons exist in the per-sensor total but contribute to no
    particle row in the hits file.
    """
    if photon_segment_index.size == 0:
        return np.array([], dtype=np.int32)

    out = np.full(photon_segment_index.size, -1, dtype=np.int32)
    valid_seg = photon_segment_index >= 0
    if not valid_seg.any():
        return out

    seg_for_valid = photon_segment_index[valid_seg]
    track_for_valid = segment_track_id[seg_for_valid]

    # Vectorized lookup. Build a contiguous track→particle array for
    # the meaningful-track id range we'll query. Tracks not in the
    # dict get -1.
    if particle_idx_by_meaningful_track:
        max_tid = int(max(particle_idx_by_meaningful_track.keys()))
    else:
        max_tid = -1
    if max_tid >= 0:
        lut = np.full(max_tid + 2, -1, dtype=np.int32)
        for tid, pidx in particle_idx_by_meaningful_track.items():
            lut[int(tid)] = int(pidx)
        # Out-of-range track ids fall through to -1 by clipping.
        clipped = np.where(track_for_valid <= max_tid, track_for_valid, max_tid + 1)
        clipped = np.where(clipped >= 0, clipped, max_tid + 1)
        out[valid_seg] = lut[clipped]
    return out


# ---- Segment grouping ----

# Mirror of DataManager.cc:444-447. Names match the C++ locals.
MIN_SEGMENT_LENGTH_MM = 10.0           # save when running merged length ≥ this
MAX_ANGLE_FOR_MERGE_RAD = 2.0 * np.pi / 180.0   # 2 degrees, save when angle > this
LOW_ENERGY_THRESHOLD_MEV = 10.0        # tracks below this use edep-based merging
MIN_EDEP_FOR_LOW_ENERGY_MEV = 1.0      # save once accumulated edep ≥ this


def assign_group_ids(
    start_x_mm: np.ndarray,
    start_y_mm: np.ndarray,
    start_z_mm: np.ndarray,
    end_x_mm: np.ndarray,
    end_y_mm: np.ndarray,
    end_z_mm: np.ndarray,
    dir_x: np.ndarray,
    dir_y: np.ndarray,
    dir_z: np.ndarray,
    edep_mev: np.ndarray,
    meaningful_tracks: Mapping[int, dict],
) -> np.ndarray:
    """Assign a contiguous group id (0..G-1 within the event) to each raw segment.

    Walks ``meaningful_tracks`` in iteration order — the dict is
    derived from ``Segment_TrackID`` + ``Segment_NCherenkov`` (post
    Stage 5a), so iteration order matches the per-track block layout
    in the filtered ``Segment_*`` arrays.

    Parameters
    ----------
    start_x_mm, start_y_mm, start_z_mm
    end_x_mm, end_y_mm, end_z_mm
        Raw segment endpoints in **millimetres** (uproot reads
        ``Segment_StartX`` etc as float64 with no unit conversion).
    dir_x, dir_y, dir_z
        Per-segment unit direction (G4double, dimensionless).
    edep_mev
        Per-segment energy deposition in MeV (already divided by MeV
        on the C++ side at ``DataManager.cc:938``).
    meaningful_tracks
        Dict keyed by Geant4 track id; each value carries
        ``segment_offset``, ``n_segments``, ``pdg``, ``initial_energy``
        (the latter in MeV — see ``DataManager.cc:540``).

    Returns
    -------
    group_id : np.ndarray
        int32 array of length ``len(edep_mev)``. Group ids are
        contiguous within the event, never re-used across tracks.
    """
    # Cast to float64 so all running accumulators have G4double precision.
    sx = np.asarray(start_x_mm, dtype=np.float64)
    sy = np.asarray(start_y_mm, dtype=np.float64)
    sz = np.asarray(start_z_mm, dtype=np.float64)
    ex = np.asarray(end_x_mm, dtype=np.float64)
    ey = np.asarray(end_y_mm, dtype=np.float64)
    ez = np.asarray(end_z_mm, dtype=np.float64)
    dxs = np.asarray(dir_x, dtype=np.float64)
    dys = np.asarray(dir_y, dtype=np.float64)
    dzs = np.asarray(dir_z, dtype=np.float64)
    edep = np.asarray(edep_mev, dtype=np.float64)

    n = edep.shape[0]
    group_id = np.full(n, -1, dtype=np.int32)
    next_group = 0

    for t_info in meaningful_tracks.values():
        offset = int(t_info['segment_offset'])
        n_seg = int(t_info['n_segments'])
        if n_seg <= 0:
            continue

        pdg = int(t_info['pdg'])
        initial_energy_mev = float(t_info['initial_energy'])
        is_low_energy = initial_energy_mev < LOW_ENERGY_THRESHOLD_MEV
        is_electron = abs(pdg) == 11
        use_edep_merging = is_low_energy or is_electron

        # Mirrors DataManager.cc:476-526. The "current" merged accumulator
        # is initialised from the first sub-step; subsequent sub-steps
        # are absorbed into it until shouldSave fires, at which point a
        # new group is opened.
        cur_start_x = sx[offset]
        cur_start_y = sy[offset]
        cur_start_z = sz[offset]
        cur_end_x = ex[offset]
        cur_end_y = ey[offset]
        cur_end_z = ez[offset]
        cur_dir_x = dxs[offset]
        cur_dir_y = dys[offset]
        cur_dir_z = dzs[offset]
        cur_edep = edep[offset]

        gid = next_group
        group_id[offset] = gid

        for i in range(1, n_seg):
            j = offset + i
            nex, ney, nez = ex[j], ey[j], ez[j]
            ndx, ndy, ndz = dxs[j], dys[j], dzs[j]
            n_edep = edep[j]

            if use_edep_merging:
                # Edep path: save once accumulated edep crosses the
                # threshold (read pre-absorption — DataManager.cc:488).
                should_save = cur_edep >= MIN_EDEP_FOR_LOW_ENERGY_MEV
            else:
                # Geometric path: length OR angle threshold.
                # current.dir is preserved across merges (the merger
                # only updates endpoints + edep + nCherenkov), so the
                # angle test compares the merge group's first-sub-step
                # direction against the candidate.
                dx_len = cur_end_x - cur_start_x
                dy_len = cur_end_y - cur_start_y
                dz_len = cur_end_z - cur_start_z
                cur_length = np.sqrt(dx_len * dx_len + dy_len * dy_len + dz_len * dz_len)

                dot = cur_dir_x * ndx + cur_dir_y * ndy + cur_dir_z * ndz
                if dot < -1.0:
                    dot = -1.0
                elif dot > 1.0:
                    dot = 1.0
                angle = np.arccos(dot)

                significant_deflection = angle > MAX_ANGLE_FOR_MERGE_RAD
                reached_min_length = cur_length >= MIN_SEGMENT_LENGTH_MM
                should_save = significant_deflection or reached_min_length

            if should_save:
                # Close the current group, start a new one rooted at j.
                next_group = gid + 1
                gid = next_group
                cur_start_x = sx[j]
                cur_start_y = sy[j]
                cur_start_z = sz[j]
                cur_end_x = nex
                cur_end_y = ney
                cur_end_z = nez
                cur_dir_x = ndx
                cur_dir_y = ndy
                cur_dir_z = ndz
                cur_edep = n_edep
            else:
                # Merge: extend endpoint, accumulate edep. dir / start /
                # beta_start unchanged (DataManager.cc:514-521).
                cur_end_x = nex
                cur_end_y = ney
                cur_end_z = nez
                cur_edep += n_edep

            group_id[j] = gid

        # Track ends → close out the open group.
        next_group = gid + 1

    return group_id


# ---- Ownership ----

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


def derive_particle_interaction_idx(event_dict, track_interaction=None):
    """For each particle, return the interaction index of its primary ancestor.

    Uses each particle's last-in-genealogy track_id (== its primary
    track) and looks up that track's ``interaction`` rank. Particles
    with no genealogy or whose primary isn't in the tracks table get -1.

    Parameters
    ----------
    event_dict : dict
        Must carry ``meaningful_tracks`` (dict of track_id -> info) and
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


# ---- Containment ----

def compute_containment(event_dict, detector_bounds):
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
