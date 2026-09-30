"""Candidate-sensor adapter for SK 20-inch PMTs in shared propagation."""

import jax
import jax.numpy as jnp

from .sk_pmt_jax import intersect_sk20inch_pmt_jax
from .sk_pmt_lookup import sk20inch_lookup_coverage


def compute_sk20inch_sensor_intersections(
    sensor_indices,
    sensor_positions,
    sensor_directions,
    sensor_is_barrel,
    ray_origins,
    ray_directions,
    coverage_lookup=None,
    return_active=False,
):
    """Evaluate one candidate-sensor slot for a batch of rays.

    The exact central-ray intersection supplies geometry, timing, and normals.
    With a lookup, the deposited weight is the smooth Gaussian bundle
    coverage. Without a lookup, the weight is the hard active-photocathode
    decision. ``inside_sensor`` records intersection with the physical PMT,
    including its inactive barrel band, so transport reaches the PMT surface
    even when its collection weight is zero.
    """
    valid = sensor_indices != -1
    safe_indices = jnp.maximum(sensor_indices, 0)
    positions = sensor_positions[safe_indices]
    axes = sensor_directions[safe_indices]
    barrel = sensor_is_barrel[safe_indices]

    hits = jax.vmap(
        lambda origin, direction, pmt_position, pmt_axis, is_barrel:
        intersect_sk20inch_pmt_jax(
            origin,
            direction,
            pmt_position,
            pmt_axis,
            barrel=is_barrel,
        )
    )(ray_origins, ray_directions, positions, axes, barrel)

    normalized_directions = ray_directions / (
        jnp.linalg.norm(ray_directions, axis=1, keepdims=True) + 1e-12)
    closest_distance = jnp.sum(
        (positions - ray_origins) * normalized_directions, axis=1)
    closest_position = (
        ray_origins + closest_distance[:, None] * normalized_directions)

    geometric_hit = valid & hits.intersects
    sensor_times = jnp.where(
        geometric_hit, hits.distance, closest_distance)[:, None]
    hit_positions = jnp.where(
        geometric_hit[:, None], hits.position, closest_position)
    # The PMT oracle normal points from glass into the water. Photon transport
    # uses the opposite convention: out of the detector/water volume. The
    # fallback follows that same detector-outward convention.
    sensor_normals = jnp.where(
        geometric_hit[:, None], -hits.normal, -axes)

    if coverage_lookup is None:
        weights = hits.active.astype(ray_origins.dtype)
    else:
        weights = jax.vmap(
            lambda origin, direction, pmt_position, pmt_axis, is_barrel:
            sk20inch_lookup_coverage(
                origin,
                direction,
                pmt_position,
                pmt_axis,
                is_barrel,
                coverage_lookup,
            )
        )(ray_origins, ray_directions, positions, axes, barrel)
    weights = jnp.where(valid, weights, 0.0)

    result = (
        weights,
        sensor_times,
        sensor_indices,
        sensor_normals,
        geometric_hit,
        hit_positions,
    )
    if return_active:
        return result + (valid & hits.active,)
    return result


__all__ = ["compute_sk20inch_sensor_intersections"]
