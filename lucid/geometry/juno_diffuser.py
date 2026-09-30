"""Effective J26.5.1 laser-diffuser hardware geometry.

The source emission model and this collision geometry are deliberately
separate.  The former describes the measured effective light distribution;
this module only lets propagated photons hit the PTFE ball, PEEK upper tube,
or PE fibre.  All dimensions are in metres.
"""

from typing import NamedTuple

import jax.numpy as jnp

from .juno_structures import SurfaceHit


LARGE_DISTANCE = 1e10
INTERSECTION_EPSILON_M = 1e-6
DIFFUSER_PTFE = 4
DIFFUSER_PEEK = 5
DIFFUSER_FIBER = 6


class DiffuserHardware(NamedTuple):
    """Dimensions of the LS-phase JUNO laser-diffuser assembly."""

    ball_radius_m: float
    peek_radius_m: float
    peek_half_length_m: float
    peek_center_axial_m: float
    fiber_radius_m: float
    fiber_half_length_m: float
    fiber_center_axial_m: float

    @classmethod
    def junosw_j26_ls(cls):
        """Build the effective dimensions from J26.5.1 GDML constants."""
        return cls(
            ball_radius_m=0.010,
            peek_radius_m=0.002,
            peek_half_length_m=0.004,
            peek_center_axial_m=0.010,
            fiber_radius_m=0.0005,
            fiber_half_length_m=0.100,
            fiber_center_axial_m=0.110,
        )


def _normalise(vectors):
    """Return row-wise unit vectors without changing the input dtype."""
    return vectors / (
        jnp.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-12
    )


def _sphere_intersection(origins, directions, center, radius):
    """Return nearest strictly-forward sphere hit and outward normal."""
    offset = origins - center[None, :]
    qb = 2.0 * jnp.sum(offset * directions, axis=1)
    qc = jnp.sum(offset * offset, axis=1) - radius**2
    discriminant = qb**2 - 4.0 * qc
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    roots = jnp.stack([(-qb - root) / 2.0, (-qb + root) / 2.0], axis=1)
    valid = (
        (discriminant[:, None] >= 0.0)
        & (roots > INTERSECTION_EPSILON_M)
    )
    distance = jnp.min(
        jnp.where(valid, roots, LARGE_DISTANCE), axis=1
    )
    position = origins + distance[:, None] * directions
    outward = _normalise(position - center[None, :])
    return distance, outward


def _cylinder_intersections(origins, directions, centers, axis,
                            radii, half_lengths):
    """Return distances and outward normals for finite capped cylinders."""
    expanded_origins = origins[:, None, :]
    expanded_directions = directions[:, None, :]
    axes = jnp.broadcast_to(axis, centers.shape)
    offset = expanded_origins - centers
    d_parallel = jnp.sum(expanded_directions * axes, axis=2)
    o_parallel = jnp.sum(offset * axes, axis=2)
    d_perp = expanded_directions - d_parallel[:, :, None] * axes
    o_perp = offset - o_parallel[:, :, None] * axes
    qa = jnp.sum(d_perp * d_perp, axis=2)
    qb = 2.0 * jnp.sum(o_perp * d_perp, axis=2)
    qc = jnp.sum(o_perp * o_perp, axis=2) - radii**2
    discriminant = qb**2 - 4.0 * qa * qc
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    denominator = jnp.where(qa > 1e-12, 2.0 * qa, 1.0)
    side_roots = jnp.stack(
        [(-qb - root) / denominator, (-qb + root) / denominator],
        axis=2,
    )
    side_axial = (
        o_parallel[:, :, None]
        + side_roots * d_parallel[:, :, None]
    )
    side_valid = (
        (discriminant[:, :, None] >= 0.0)
        & (qa[:, :, None] > 1e-12)
        & (side_roots > INTERSECTION_EPSILON_M)
        & (jnp.abs(side_axial) <= half_lengths[:, :, None] + 1e-7)
    )
    side_distance = jnp.min(
        jnp.where(side_valid, side_roots, LARGE_DISTANCE), axis=2
    )

    signs = jnp.asarray([-1.0, 1.0], dtype=origins.dtype)
    safe_parallel = jnp.where(
        jnp.abs(d_parallel) > 1e-12, d_parallel, 1.0
    )
    cap_distance = (
        signs[None, None, :] * half_lengths[:, :, None]
        - o_parallel[:, :, None]
    ) / safe_parallel[:, :, None]
    cap_points = (
        offset[:, :, None, :]
        + cap_distance[:, :, :, None]
        * expanded_directions[:, :, None, :]
    )
    cap_axial = jnp.sum(cap_points * axes[:, :, None, :], axis=3)
    cap_radial = cap_points - cap_axial[:, :, :, None] * axes[:, :, None, :]
    cap_valid = (
        (jnp.abs(d_parallel[:, :, None]) > 1e-12)
        & (cap_distance > INTERSECTION_EPSILON_M)
        & (jnp.sum(cap_radial**2, axis=3) <= radii[:, :, None] ** 2)
    )
    cap_distance = jnp.min(
        jnp.where(cap_valid, cap_distance, LARGE_DISTANCE), axis=2
    )

    use_side = side_distance <= cap_distance
    distance = jnp.minimum(side_distance, cap_distance)
    position = expanded_origins + distance[:, :, None] * expanded_directions
    axial = jnp.sum((position - centers) * axes, axis=2)
    side_normal = _normalise(
        position - centers - axial[:, :, None] * axes
    )
    cap_normal = jnp.where(axial[:, :, None] >= 0.0, axes, -axes)
    outward = jnp.where(use_side[:, :, None], side_normal, cap_normal)
    return distance, outward


def intersect_diffuser_hardware(origins, directions, hardware, center,
                                axis):
    """Choose the nearest PTFE-ball, PEEK-tube, or PE-fibre intersection.

    ``center`` and ``axis`` are runtime inputs so the small assembly follows
    every deployed source position without rebuilding detector geometry.
    ``transport_normals`` follow the engineering-surface convention: they
    point into the surrounding LS, opposite the solid's outward normal.
    """
    origins = jnp.atleast_2d(origins)
    directions = _normalise(jnp.atleast_2d(directions))
    center = jnp.asarray(center, dtype=origins.dtype)
    axis = _normalise(jnp.asarray(axis, dtype=origins.dtype)[None, :])[0]

    sphere_distance, sphere_normal = _sphere_intersection(
        origins, directions, center, hardware.ball_radius_m
    )
    axial_offsets = jnp.asarray(
        [
            hardware.peek_center_axial_m,
            hardware.fiber_center_axial_m,
        ],
        dtype=origins.dtype,
    )
    centers = center[None, :] + axial_offsets[:, None] * axis[None, :]
    centers = jnp.broadcast_to(centers[None, :, :], (origins.shape[0], 2, 3))
    radii = jnp.broadcast_to(
        jnp.asarray(
            [hardware.peek_radius_m, hardware.fiber_radius_m],
            dtype=origins.dtype,
        )[None, :],
        (origins.shape[0], 2),
    )
    half_lengths = jnp.broadcast_to(
        jnp.asarray(
            [
                hardware.peek_half_length_m,
                hardware.fiber_half_length_m,
            ],
            dtype=origins.dtype,
        )[None, :],
        (origins.shape[0], 2),
    )
    cylinder_distance, cylinder_normal = _cylinder_intersections(
        origins,
        directions,
        centers,
        axis[None, :],
        radii,
        half_lengths,
    )
    distances = jnp.concatenate(
        [sphere_distance[:, None], cylinder_distance], axis=1
    )
    normals = jnp.concatenate(
        [sphere_normal[:, None, :], cylinder_normal], axis=1
    )
    selected = jnp.argmin(distances, axis=1)
    rows = jnp.arange(origins.shape[0])
    nearest = distances[rows, selected]
    valid = nearest < LARGE_DISTANCE
    positions = origins + nearest[:, None] * directions
    outward = normals[rows, selected]
    surface_codes = jnp.asarray(
        [DIFFUSER_PTFE, DIFFUSER_PEEK, DIFFUSER_FIBER],
        dtype=jnp.int32,
    )[selected]
    return SurfaceHit(
        distance_m=nearest,
        valid=valid,
        object_indices=jnp.where(valid, selected, -1),
        surface_codes=jnp.where(valid, surface_codes, -1),
        positions=positions,
        transport_normals=-outward,
        outward_normals=outward,
        inside_before=jnp.zeros_like(valid),
    )
