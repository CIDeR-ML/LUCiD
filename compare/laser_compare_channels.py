"""Deterministic representative-channel selection for laser comparisons."""

import numpy as np
import pyarrow.parquet as pq

from . import compare_sim as base


def _angular_distance(theta, phi, target_theta, target_phi):
    """Return great-circle distances in radians for angular coordinates."""
    theta_rad = np.deg2rad(np.asarray(theta, dtype=np.float64))
    phi_rad = np.deg2rad(np.asarray(phi, dtype=np.float64))
    target_theta_rad = np.deg2rad(float(target_theta))
    target_phi_rad = np.deg2rad(float(target_phi))
    cosine = (
        np.sin(theta_rad)
        * np.sin(target_theta_rad)
        * np.cos(phi_rad - target_phi_rad)
        + np.cos(theta_rad) * np.cos(target_theta_rad)
    )
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def representative_channels(
    pmt_map,
    theta_bin_width=10.0,
    phi_targets=(0.0, 180.0),
    pmt_types=(1, 2),
):
    """Select one PMT nearest each fixed geometry-stratum target.

    Input: PMT Parquet map, theta-bin width, target azimuths and PMT types.
    Output: sorted, unique channel identifiers for a reproducible plot panel.
    """
    if theta_bin_width <= 0.0 or not np.isclose(
        180.0 % theta_bin_width, 0.0
    ):
        raise ValueError("theta_bin_width must divide 180 degrees")
    table = pq.read_table(
        pmt_map,
        columns=["copy_number", "pmt_type", "theta", "phi"],
    )
    channel = table["copy_number"].to_numpy(zero_copy_only=False)
    pmt_type = table["pmt_type"].to_numpy(zero_copy_only=False)
    theta = table["theta"].to_numpy(zero_copy_only=False)
    phi = table["phi"].to_numpy(zero_copy_only=False)
    valid = (
        (channel >= 0)
        & (channel < base.FULL_CHANNEL_COUNT)
        & np.isfinite(theta)
        & np.isfinite(phi)
    )
    selected = []
    for low in np.arange(0.0, 180.0, theta_bin_width):
        high = low + theta_bin_width
        theta_mask = (theta >= low) & (theta < high)
        if np.isclose(high, 180.0):
            theta_mask = (theta >= low) & (theta <= high)
        for kind in pmt_types:
            candidates = np.flatnonzero(valid & theta_mask & (pmt_type == kind))
            for target_phi in phi_targets:
                available = candidates[~np.isin(channel[candidates], selected)]
                if not len(available):
                    raise ValueError(
                        "No unique PMT for theta/type/phi "
                        "representative stratum"
                    )
                distance = _angular_distance(
                    theta[available],
                    phi[available],
                    low + theta_bin_width / 2.0,
                    target_phi,
                )
                order = np.lexsort((channel[available], distance))
                selected.append(int(channel[available[order[0]]]))
    return np.sort(np.asarray(selected, dtype=np.int64))


def stratified_channels(
    pmt_map,
    stratum_size=28,
    theta_bin_width=10.0,
    pmt_types=(1, 2),
    seed=0,
):
    """Select ``stratum_size`` PMTs per theta/type stratum with a seed.

    Input: PMT Parquet map, per-stratum count, theta-bin width, PMT types
    and the random seed.
    Output: sorted unique channel identifiers; deterministic for a fixed
    seed, keeping every channel of strata with fewer members.
    """
    if stratum_size <= 0:
        raise ValueError("stratum_size must be positive")
    if theta_bin_width <= 0.0 or not np.isclose(
        180.0 % theta_bin_width, 0.0
    ):
        raise ValueError("theta_bin_width must divide 180 degrees")
    table = pq.read_table(
        pmt_map,
        columns=["copy_number", "pmt_type", "theta", "phi"],
    )
    channel = table["copy_number"].to_numpy(zero_copy_only=False)
    pmt_type = table["pmt_type"].to_numpy(zero_copy_only=False)
    theta = table["theta"].to_numpy(zero_copy_only=False)
    valid = (
        (channel >= 0)
        & (channel < base.FULL_CHANNEL_COUNT)
        & np.isfinite(theta)
    )
    rng = np.random.default_rng(seed)
    selected = []
    for low in np.arange(0.0, 180.0, theta_bin_width):
        high = low + theta_bin_width
        theta_mask = (theta >= low) & (theta < high)
        if np.isclose(high, 180.0):
            theta_mask = (theta >= low) & (theta <= high)
        for kind in pmt_types:
            candidates = np.flatnonzero(
                valid & theta_mask & (pmt_type == kind)
            )
            take = min(stratum_size, len(candidates))
            if not take:
                continue
            picked = rng.choice(len(candidates), size=take, replace=False)
            selected.extend(int(channel[candidates[i]]) for i in picked)
    return np.sort(np.asarray(selected, dtype=np.int64))
