"""Python port of PhotonSim's segment merger.

PhotonSim used to apply a merging pass over G4 sub-steps in
``DataManager::EndEvent`` (``src/DataManager.cc:441-527``) before writing
``Segment_*`` ROOT branches. Branch ``raw-segments-no-merge`` makes
PhotonSim emit one ``Segment_*`` row per raw G4 step instead, and the
merger moves here. Aggregating raw rows by the ``group_id`` returned
from :func:`assign_group_ids` reproduces today's merged output
byte-for-byte.

The constants below mirror ``DataManager.cc:444-447`` exactly. Inputs
are in **millimetres** (the native unit used by the C++ merger) so
length comparisons are bit-identical; LUCiD's mm→m conversion happens
*after* this function runs.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


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
            nx, ny, nz = sx[j], sy[j], sz[j]  # next sub-step start (unused here)
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
