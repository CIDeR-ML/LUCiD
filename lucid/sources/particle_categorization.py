"""Pure-Python port of PhotonSim's per-track categorization, genealogy
construction, photon→particle bucketing, and meaningful-track
derivation.

Until Stage 5a, PhotonSim categorized Geant4 tracks at creation time
and emitted ``Particle_*`` / ``MTrack_*`` / ``TrackInfo_Category``
ROOT branches. After Stage 5a, PhotonSim emits only the raw inputs
(``TrackInfo_*`` incl. ``TrackInfo_CreatorProcess``, every G4 track's
segments via ``Segment_*`` + ``Segment_TrackID``,
``Photon_SegmentIndex``, ``OpticalPhotonsRaw``); this module
reconstructs the categorization, the meaningful-track set, the
genealogies, and the photon→particle binding in Python.

The categorization output is bit-identical to the legacy C++ branches
on ``TrackInfo_Category`` and ``Particle_*`` except for
``TrackInfo_SubID`` within a single (parent, decay-process, time) tie
— Geant4 processes multi-secondary decays in stack (LIFO) order; this
implementation processes tracks in track-id order. The sub-id is
metadata only, never read downstream of
``read_particle_data_from_photonsim``, so this divergence is invisible
to inst.h5 / labl.h5 consumers.

The meaningful-track set ("Cherenkov producers + their ancestors")
mirrors the legacy C++ filter in ``DataManager::EndEvent`` — same
definition, same downstream HDF5 contents, just derived in Python
from groupby on ``Segment_TrackID`` + ``Segment_NCherenkov``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


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
M_PI_CHARGED_MEV = 139.57039

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
    :func:`lucid.sources.segment_grouping.assign_group_ids`:
    ``track_id → {parent_id, pdg, initial_energy, particle_name,
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
        ``{track_id: {parent_id, pdg, initial_energy, particle_name,
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
            'particle_name': pdg_to_g4name(int(ti.get('pdg', 0))),
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
    particle row in inst.h5.
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
