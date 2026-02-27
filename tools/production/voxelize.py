"""
Voxelization utilities for LUCiD photon data.

This module provides efficient voxelization of 3D photon coordinates,
storing only non-zero voxels for memory efficiency.

Grid specifications:
- Volume: 100x100x100 m³ centered at (0,0,0)
- Voxel size: 1cm³ (0.01m per side)
- Grid dimensions: 10,000 x 10,000 x 10,000 voxels

Storage format:
- Sparse representation: only non-zero voxels are stored
- Per-label voxel counts and indices
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass


# Grid configuration constants
GRID_SIZE_M = 100.0  # meters (total extent: -50 to +50 in each dimension)
VOXEL_SIZE_M = 0.01  # meters (1 cm)
N_VOXELS_PER_DIM = int(GRID_SIZE_M / VOXEL_SIZE_M)  # 10,000 voxels per dimension
GRID_ORIGIN_M = -GRID_SIZE_M / 2.0  # -50.0 meters (grid starts at -50m)


@dataclass
class VoxelGridConfig:
    """Configuration for the voxel grid."""
    size_m: float = GRID_SIZE_M
    voxel_size_m: float = VOXEL_SIZE_M
    n_voxels_per_dim: int = N_VOXELS_PER_DIM
    origin_m: float = GRID_ORIGIN_M

    def __post_init__(self):
        """Validate and recompute derived values."""
        self.n_voxels_per_dim = int(self.size_m / self.voxel_size_m)
        self.origin_m = -self.size_m / 2.0


def position_to_voxel_indices(positions: np.ndarray,
                               config: Optional[VoxelGridConfig] = None) -> np.ndarray:
    """
    Convert 3D positions (in meters) to voxel indices.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) containing 3D positions in meters
    config : VoxelGridConfig, optional
        Grid configuration. Uses default if not provided.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing voxel indices (ix, iy, iz)
        Indices range from 0 to n_voxels_per_dim - 1
        Out-of-bounds positions are clipped to valid range
    """
    if config is None:
        config = VoxelGridConfig()

    # Shift positions so that origin is at corner of grid
    shifted = positions - config.origin_m

    # Convert to voxel indices
    indices = np.floor(shifted / config.voxel_size_m).astype(np.int64)

    # Clip to valid range
    indices = np.clip(indices, 0, config.n_voxels_per_dim - 1)

    return indices


def voxel_indices_to_flat_index(indices: np.ndarray,
                                 config: Optional[VoxelGridConfig] = None) -> np.ndarray:
    """
    Convert 3D voxel indices to flat (1D) indices.

    Parameters
    ----------
    indices : np.ndarray
        Array of shape (N, 3) containing voxel indices (ix, iy, iz)
    config : VoxelGridConfig, optional
        Grid configuration. Uses default if not provided.

    Returns
    -------
    np.ndarray
        Array of shape (N,) containing flat indices
        flat_index = ix + iy * N + iz * N * N
    """
    if config is None:
        config = VoxelGridConfig()

    N = config.n_voxels_per_dim
    return indices[:, 0] + indices[:, 1] * N + indices[:, 2] * N * N


def flat_index_to_voxel_indices(flat_indices: np.ndarray,
                                 config: Optional[VoxelGridConfig] = None) -> np.ndarray:
    """
    Convert flat (1D) indices back to 3D voxel indices.

    Parameters
    ----------
    flat_indices : np.ndarray
        Array of shape (N,) containing flat indices
    config : VoxelGridConfig, optional
        Grid configuration. Uses default if not provided.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing voxel indices (ix, iy, iz)
    """
    if config is None:
        config = VoxelGridConfig()

    N = config.n_voxels_per_dim

    iz = flat_indices // (N * N)
    remainder = flat_indices % (N * N)
    iy = remainder // N
    ix = remainder % N

    return np.stack([ix, iy, iz], axis=1)


def voxel_indices_to_position(indices: np.ndarray,
                               config: Optional[VoxelGridConfig] = None,
                               center: bool = True) -> np.ndarray:
    """
    Convert voxel indices to 3D positions (in meters).

    Parameters
    ----------
    indices : np.ndarray
        Array of shape (N, 3) containing voxel indices (ix, iy, iz)
    config : VoxelGridConfig, optional
        Grid configuration. Uses default if not provided.
    center : bool, optional
        If True, return center of voxel. If False, return corner (min position).
        Default is True.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing 3D positions in meters
    """
    if config is None:
        config = VoxelGridConfig()

    positions = indices * config.voxel_size_m + config.origin_m

    if center:
        positions = positions + config.voxel_size_m / 2.0

    return positions


def flat_index_to_position(flat_indices: np.ndarray,
                           config: Optional[VoxelGridConfig] = None,
                           center: bool = True) -> np.ndarray:
    """
    Convert flat indices directly to 3D positions (in meters).

    Parameters
    ----------
    flat_indices : np.ndarray
        Array of shape (N,) containing flat indices
    config : VoxelGridConfig, optional
        Grid configuration. Uses default if not provided.
    center : bool, optional
        If True, return center of voxel. Default is True.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing 3D positions in meters
    """
    indices = flat_index_to_voxel_indices(flat_indices, config)
    return voxel_indices_to_position(indices, config, center)


def voxelize_photons(positions: np.ndarray,
                     config: Optional[VoxelGridConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Voxelize photon positions and count photons per voxel.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) containing photon positions in meters
    config : VoxelGridConfig, optional
        Grid configuration. Uses default if not provided.

    Returns
    -------
    unique_flat_indices : np.ndarray
        Array of unique flat voxel indices (sorted)
    counts : np.ndarray
        Array of photon counts for each unique voxel
    """
    if config is None:
        config = VoxelGridConfig()

    # Convert positions to voxel indices
    voxel_indices = position_to_voxel_indices(positions, config)

    # Convert to flat indices
    flat_indices = voxel_indices_to_flat_index(voxel_indices, config)

    # Count unique voxels
    unique_flat_indices, counts = np.unique(flat_indices, return_counts=True)

    return unique_flat_indices, counts


def voxelize_photons_by_label(positions: np.ndarray,
                               labels: np.ndarray,
                               config: Optional[VoxelGridConfig] = None) -> Dict:
    """
    Voxelize photon positions grouped by label.

    This is the main function for production use. It counts photons in each
    voxel for each label separately, storing only non-zero voxels.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) containing photon positions in meters
    labels : np.ndarray
        Array of shape (N,) containing label indices for each photon
    config : VoxelGridConfig, optional
        Grid configuration. Uses default if not provided.

    Returns
    -------
    dict
        Dictionary containing:
        - 'n_labels': int, number of unique labels
        - 'n_nonzero_voxels': np.ndarray of shape (n_labels,),
            number of non-zero voxels per label
        - 'voxel_indices': list of np.ndarray, flat voxel indices per label
        - 'voxel_counts': list of np.ndarray, photon counts per voxel per label
        - 'config': VoxelGridConfig used
    """
    if config is None:
        config = VoxelGridConfig()

    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    # Convert all positions to voxel indices at once
    voxel_indices = position_to_voxel_indices(positions, config)
    flat_indices = voxel_indices_to_flat_index(voxel_indices, config)

    # Process each label
    n_nonzero_voxels = np.zeros(n_labels, dtype=np.int64)
    voxel_indices_list = []
    voxel_counts_list = []

    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_flat_indices = flat_indices[mask]

        # Get unique voxels and counts for this label
        unique_indices, counts = np.unique(label_flat_indices, return_counts=True)

        n_nonzero_voxels[i] = len(unique_indices)
        voxel_indices_list.append(unique_indices)
        voxel_counts_list.append(counts)

    return {
        'n_labels': n_labels,
        'label_ids': unique_labels,
        'n_nonzero_voxels': n_nonzero_voxels,
        'voxel_indices': voxel_indices_list,
        'voxel_counts': voxel_counts_list,
        'config': config
    }


def voxelize_from_photon_indices(all_positions: np.ndarray,
                                  label_photon_indices: List[np.ndarray],
                                  config: Optional[VoxelGridConfig] = None) -> Dict:
    """
    Voxelize photons where labels are defined by lists of photon indices.

    This matches the PhotonSim data format where each label has a list of
    photon IDs that belong to it.

    Parameters
    ----------
    all_positions : np.ndarray
        Array of shape (N_total, 3) containing all photon positions in meters
    label_photon_indices : list of np.ndarray
        List of arrays, each containing photon indices for that label
    config : VoxelGridConfig, optional
        Grid configuration. Uses default if not provided.

    Returns
    -------
    dict
        Same format as voxelize_photons_by_label
    """
    if config is None:
        config = VoxelGridConfig()

    n_labels = len(label_photon_indices)

    # Convert all positions to flat indices at once (for efficiency)
    voxel_indices = position_to_voxel_indices(all_positions, config)
    all_flat_indices = voxel_indices_to_flat_index(voxel_indices, config)

    # Process each label
    n_nonzero_voxels = np.zeros(n_labels, dtype=np.int64)
    voxel_indices_list = []
    voxel_counts_list = []

    for i, photon_idx in enumerate(label_photon_indices):
        if len(photon_idx) == 0:
            n_nonzero_voxels[i] = 0
            voxel_indices_list.append(np.array([], dtype=np.int64))
            voxel_counts_list.append(np.array([], dtype=np.int64))
            continue

        # Get flat indices for photons in this label
        label_flat_indices = all_flat_indices[photon_idx]

        # Get unique voxels and counts
        unique_indices, counts = np.unique(label_flat_indices, return_counts=True)

        n_nonzero_voxels[i] = len(unique_indices)
        voxel_indices_list.append(unique_indices)
        voxel_counts_list.append(counts)

    return {
        'n_labels': n_labels,
        'n_nonzero_voxels': n_nonzero_voxels,
        'voxel_indices': voxel_indices_list,
        'voxel_counts': voxel_counts_list,
        'config': config
    }


def pack_voxel_data_for_hdf5(voxel_data: Dict) -> Dict:
    """
    Pack voxel data into a format suitable for HDF5 storage.

    Since HDF5 doesn't natively support ragged arrays, we flatten the
    per-label arrays and store offset information.

    Parameters
    ----------
    voxel_data : dict
        Output from voxelize_photons_by_label or voxelize_from_photon_indices

    Returns
    -------
    dict
        Dictionary with HDF5-compatible arrays:
        - 'voxel_n_nonzero': shape (n_labels,), count of non-zero voxels per label
        - 'voxel_offsets': shape (n_labels,), starting offset for each label
        - 'voxel_flat_indices': shape (total_nonzero,), concatenated flat indices
        - 'voxel_counts': shape (total_nonzero,), concatenated photon counts
        - 'voxel_grid_size_m': grid size in meters
        - 'voxel_size_m': voxel size in meters
        - 'voxel_n_per_dim': number of voxels per dimension
    """
    n_labels = voxel_data['n_labels']
    n_nonzero_voxels = voxel_data['n_nonzero_voxels']

    # Compute offsets
    offsets = np.zeros(n_labels, dtype=np.int64)
    offsets[1:] = np.cumsum(n_nonzero_voxels[:-1])

    # Concatenate indices and counts
    all_indices = np.concatenate(voxel_data['voxel_indices']) if sum(n_nonzero_voxels) > 0 else np.array([], dtype=np.int64)
    all_counts = np.concatenate(voxel_data['voxel_counts']) if sum(n_nonzero_voxels) > 0 else np.array([], dtype=np.int64)

    config = voxel_data['config']

    return {
        'voxel_n_nonzero': n_nonzero_voxels,
        'voxel_offsets': offsets,
        'voxel_flat_indices': all_indices,
        'voxel_counts': all_counts,
        'voxel_grid_size_m': config.size_m,
        'voxel_size_m': config.voxel_size_m,
        'voxel_n_per_dim': config.n_voxels_per_dim
    }


def unpack_voxel_data_from_hdf5(hdf5_data: Dict) -> Dict:
    """
    Unpack voxel data from HDF5 storage format.

    Parameters
    ----------
    hdf5_data : dict
        Dictionary read from HDF5 with keys from pack_voxel_data_for_hdf5

    Returns
    -------
    dict
        Dictionary matching the output format of voxelize_photons_by_label
    """
    n_nonzero = hdf5_data['voxel_n_nonzero']
    offsets = hdf5_data['voxel_offsets']
    all_indices = hdf5_data['voxel_flat_indices']
    all_counts = hdf5_data['voxel_counts']

    n_labels = len(n_nonzero)

    # Reconstruct per-label arrays
    voxel_indices_list = []
    voxel_counts_list = []

    for i in range(n_labels):
        start = offsets[i]
        end = start + n_nonzero[i]
        voxel_indices_list.append(all_indices[start:end])
        voxel_counts_list.append(all_counts[start:end])

    config = VoxelGridConfig(
        size_m=float(hdf5_data['voxel_grid_size_m']),
        voxel_size_m=float(hdf5_data['voxel_size_m'])
    )

    return {
        'n_labels': n_labels,
        'n_nonzero_voxels': n_nonzero,
        'voxel_indices': voxel_indices_list,
        'voxel_counts': voxel_counts_list,
        'config': config
    }


def get_voxel_statistics(voxel_data: Dict) -> Dict:
    """
    Compute statistics about voxelized data.

    Parameters
    ----------
    voxel_data : dict
        Output from voxelize_photons_by_label or voxelize_from_photon_indices

    Returns
    -------
    dict
        Dictionary with statistics:
        - 'total_photons': total photon count
        - 'total_nonzero_voxels': total non-zero voxels across all labels
        - 'photons_per_label': array of photon counts per label
        - 'max_photons_per_voxel': maximum photons in any single voxel
        - 'mean_photons_per_voxel': mean photons per non-zero voxel
    """
    n_labels = voxel_data['n_labels']
    n_nonzero = voxel_data['n_nonzero_voxels']
    counts_list = voxel_data['voxel_counts']

    photons_per_label = np.array([np.sum(c) for c in counts_list])
    total_photons = np.sum(photons_per_label)
    total_nonzero = np.sum(n_nonzero)

    all_counts = np.concatenate(counts_list) if total_nonzero > 0 else np.array([0])

    return {
        'total_photons': int(total_photons),
        'total_nonzero_voxels': int(total_nonzero),
        'photons_per_label': photons_per_label,
        'max_photons_per_voxel': int(np.max(all_counts)) if total_nonzero > 0 else 0,
        'mean_photons_per_voxel': float(np.mean(all_counts)) if total_nonzero > 0 else 0.0
    }


def check_positions_in_bounds(positions: np.ndarray,
                               config: Optional[VoxelGridConfig] = None) -> np.ndarray:
    """
    Check which positions are within the voxel grid bounds.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) containing positions in meters
    config : VoxelGridConfig, optional
        Grid configuration. Uses default if not provided.

    Returns
    -------
    np.ndarray
        Boolean array of shape (N,), True for positions within bounds
    """
    if config is None:
        config = VoxelGridConfig()

    min_bound = config.origin_m
    max_bound = config.origin_m + config.size_m

    in_bounds = np.all((positions >= min_bound) & (positions < max_bound), axis=1)
    return in_bounds


if __name__ == "__main__":
    # Simple test
    print("Voxelize module loaded successfully")
    print(f"Grid configuration:")
    print(f"  Size: {GRID_SIZE_M} m")
    print(f"  Voxel size: {VOXEL_SIZE_M} m = {VOXEL_SIZE_M * 100} cm")
    print(f"  Voxels per dimension: {N_VOXELS_PER_DIM}")
    print(f"  Total potential voxels: {N_VOXELS_PER_DIM ** 3:,}")
    print(f"  Grid origin: {GRID_ORIGIN_M} m")
    print(f"  Grid bounds: [{GRID_ORIGIN_M}, {-GRID_ORIGIN_M}] m")
