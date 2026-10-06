"""Differentiable PMT/time-bin observation operator for laser calibration."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .time_grid import TimeGrid


def weighted_time_histogram(flat_weights, flat_indices, flat_times, *,
                            n_sensors: int, time_grid: TimeGrid,
                            qe: float = 1.0, qe_corrections=None):
    """Accumulate continuous hit weights into a PMT/time-bin histogram.

    Entry ``b`` is the expected PE mass associated with the cell centered at
    ``time_min_ns + (b + 0.5) * bin_width_ns``.  A hit between two centers is
    linearly split between them.  At either finite-window edge, the otherwise
    out-of-range share is folded onto the nearest valid cell so every accepted
    hit retains its full weight.

    Invalid candidates are masked before accumulation, so padded sensor
    indices and out-of-window times cannot contaminate the histogram or its
    gradient.
    """
    if n_sensors <= 0:
        raise ValueError("n_sensors must be positive")
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid")
    n_bins = time_grid.n_bins
    corrections = (jnp.ones(n_sensors, dtype=jnp.float32)
                   if qe_corrections is None else jnp.asarray(qe_corrections))
    if corrections.shape != (n_sensors,):
        raise ValueError(
            "qe_corrections must have shape "
            f"({n_sensors},), got {corrections.shape}"
        )

    weights = jnp.asarray(flat_weights, dtype=jnp.float32)
    indices = jnp.asarray(flat_indices, dtype=jnp.int32)
    times = jnp.asarray(flat_times, dtype=weights.dtype)
    bin_position = (
        (times - time_grid.time_min_ns) / time_grid.bin_width_ns - 0.5
    )
    left = jnp.floor(bin_position).astype(jnp.int32)
    right = left + 1
    fraction = jnp.clip(bin_position - left, 0.0, 1.0)
    valid = ((weights > 1e-10) & jnp.isfinite(times) &
             (indices >= 0) & (indices < n_sensors) &
             (times >= time_grid.time_min_ns) &
             (times < time_grid.time_max_ns))
    safe_sensor = jnp.clip(indices, 0, n_sensors - 1)
    safe_left = jnp.clip(left, 0, n_bins - 1)
    safe_right = jnp.clip(right, 0, n_bins - 1)
    weighted = jnp.where(
        valid,
        weights * float(qe) * corrections[safe_sensor],
        0.0,
    )
    left_index = safe_sensor * n_bins + safe_left
    right_index = safe_sensor * n_bins + safe_right
    left_values = weighted * (1.0 - fraction)
    right_values = weighted * fraction
    total = jax.ops.segment_sum(
        left_values, left_index, num_segments=n_sensors * n_bins)
    total += jax.ops.segment_sum(
        right_values, right_index, num_segments=n_sensors * n_bins)
    return total.reshape((n_sensors, n_bins))
