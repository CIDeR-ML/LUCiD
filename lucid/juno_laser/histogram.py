"""Differentiable PMT/time-bin observation operator for laser calibration."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def weighted_time_histogram(flat_weights, flat_indices, flat_times, *,
                            n_sensors: int, time_min_ns: float,
                            time_max_ns: float, bin_width_ns: float,
                            qe: float = 1.0, qe_corrections=None):
    """Accumulate continuous hit weights into a PMT/time-bin histogram.

    A hit between two bin centers is linearly split between neighboring bins.
    Invalid candidates are masked before indexing, so padded sensor indices and
    out-of-window times cannot contaminate the histogram or its gradient.
    """
    if n_sensors <= 0 or time_max_ns <= time_min_ns or bin_width_ns <= 0:
        raise ValueError("invalid sensor count, time window or bin width")
    n_bins = int(math.ceil((time_max_ns - time_min_ns) / bin_width_ns))
    corrections = (jnp.ones(n_sensors, dtype=jnp.float32)
                   if qe_corrections is None else jnp.asarray(qe_corrections))
    if corrections.shape != (n_sensors,):
        raise ValueError(f"qe_corrections must have shape ({n_sensors},), got {corrections.shape}")

    weights = jnp.asarray(flat_weights, dtype=jnp.float32)
    indices = jnp.asarray(flat_indices, dtype=jnp.int32)
    times = jnp.asarray(flat_times, dtype=weights.dtype)
    bin_position = (times - float(time_min_ns)) / float(bin_width_ns)
    left = jnp.floor(bin_position).astype(jnp.int32)
    right = left + 1
    fraction = jnp.clip(bin_position - left, 0.0, 1.0)
    valid = ((weights > 1e-10) & jnp.isfinite(times) &
             (indices >= 0) & (indices < n_sensors) &
             (bin_position >= 0.0) & (bin_position < n_bins))
    safe_sensor = jnp.clip(indices, 0, n_sensors - 1)
    safe_left = jnp.clip(left, 0, n_bins - 1)
    safe_right = jnp.clip(right, 0, n_bins - 1)
    weighted = jnp.where(valid, weights * float(qe) * corrections[safe_sensor], 0.0)
    left_valid = valid & (left >= 0) & (left < n_bins)
    right_valid = valid & (right >= 0) & (right < n_bins)
    left_index = safe_sensor * n_bins + safe_left
    right_index = safe_sensor * n_bins + safe_right
    left_values = jnp.where(left_valid, weighted * (1.0 - fraction), 0.0)
    right_values = jnp.where(right_valid, weighted * fraction, 0.0)
    total = jax.ops.segment_sum(
        left_values, left_index, num_segments=n_sensors * n_bins)
    total += jax.ops.segment_sum(
        right_values, right_index, num_segments=n_sensors * n_bins)
    return total.reshape((n_sensors, n_bins))
