"""Piecewise-differentiable JAX intersections for the SK 20-inch PMT.

The hard NumPy oracle in :mod:`lucid.propagation.sk_pmt` defines the forward
geometry. This module evaluates the same sphere/torus surface with JAX. Root
positions and normals are differentiable within a selected surface branch;
the hit/miss and branch selections remain hard until the smooth-coverage stage.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .sk_pmt import (
    BARREL_INACTIVE_BAND_Z,
    GEANT_PMT_RADIUS,
    SPHERE_CENTER_Z,
    SPHERE_RADIUS,
    SPHERE_TO_TORUS_Z,
    TORUS_MAJOR_RADIUS,
    TORUS_MINOR_RADIUS,
)


SK_PMT_SURFACE_MISS = 0
SK_PMT_SURFACE_SPHERE = 1
SK_PMT_SURFACE_TORUS = 2

_NEWTON_ITERATIONS = 16
_DERIVATIVE_EPS = 1e-10
_ROOT_RESIDUAL_TOL = 2e-7
_CUT_TOL = 1e-9


class JAXSK20InchPMTHit(NamedTuple):
    """JAX-compatible result for one ray and one PMT."""

    intersects: jax.Array
    active: jax.Array
    distance: jax.Array
    position: jax.Array
    normal: jax.Array
    surface: jax.Array
    incidence_cosine: jax.Array
    axial_position: jax.Array
    radial_position: jax.Array


def _normalize(vector):
    return vector / jnp.linalg.norm(vector)


def _safe_normalize(vector):
    return vector / jnp.sqrt(jnp.dot(vector, vector) + _DERIVATIVE_EPS**2)


def _ray_sphere_roots(relative_origin, direction, radius):
    """Return ordered roots and the unclipped discriminant."""
    a = jnp.dot(direction, direction)
    b = 2.0 * jnp.dot(relative_origin, direction)
    c = jnp.dot(relative_origin, relative_origin) - radius**2
    discriminant = b * b - 4.0 * a * c
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    return ((-b - root) / (2.0 * a),
            (-b + root) / (2.0 * a),
            discriminant)


def _axial_and_transverse(relative_position, axis):
    axial = jnp.dot(relative_position, axis)
    transverse = relative_position - axial * axis
    # The epsilon leaves metre-scale forward values unchanged while keeping
    # reverse-mode derivatives finite for an exactly on-axis ray.
    radial = jnp.sqrt(
        jnp.dot(transverse, transverse) + _DERIVATIVE_EPS**2)
    return axial, transverse, radial


def _torus_implicit(relative_position, axis):
    axial, _, radial = _axial_and_transverse(relative_position, axis)
    value = ((radial - TORUS_MAJOR_RADIUS)**2 + axial**2
             - TORUS_MINOR_RADIUS**2)
    return value


def _torus_root_newton(relative_origin, direction, axis, lower, upper):
    """Find the first torus root after entry into its bounding sphere."""
    initial = lower

    def iteration(_, distance):
        relative_position = relative_origin + distance * direction
        axial, transverse, radial = _axial_and_transverse(
            relative_position, axis)
        safe_radial = jnp.maximum(radial, _DERIVATIVE_EPS)
        transverse_direction = direction - jnp.dot(direction, axis) * axis
        radial_derivative = jnp.dot(
            transverse, transverse_direction) / safe_radial
        axial_derivative = jnp.dot(direction, axis)
        value = ((radial - TORUS_MAJOR_RADIUS)**2 + axial**2
                 - TORUS_MINOR_RADIUS**2)
        derivative = (
            2.0 * (radial - TORUS_MAJOR_RADIUS) * radial_derivative
            + 2.0 * axial * axial_derivative
        )
        safe_derivative = jnp.where(
            jnp.abs(derivative) >= _DERIVATIVE_EPS,
            derivative,
            jnp.where(derivative >= 0.0,
                      _DERIVATIVE_EPS, -_DERIVATIVE_EPS),
        )
        proposal = distance - value / safe_derivative
        return jnp.clip(proposal, lower, upper)

    return jax.lax.fori_loop(0, _NEWTON_ITERATIONS, iteration, initial)


def intersect_sk20inch_pmt_jax(
    ray_origin,
    ray_direction,
    pmt_position,
    pmt_direction,
    *,
    barrel=False,
    min_distance=1e-9,
):
    """Intersect one ray with the SK 20-inch PMT using JAX operations.

    The returned root, position, normal, and incidence cosine carry gradients
    within the chosen sphere or torus branch. ``intersects``, ``active``, and
    ``surface`` are discrete selections and are therefore only piecewise
    differentiable. Inputs must be finite three-vectors with non-zero ray and
    PMT directions, as checked by the hard oracle before production use.
    """
    origin = jnp.asarray(ray_origin)
    direction = _normalize(jnp.asarray(ray_direction))
    pmt_origin = jnp.asarray(pmt_position)
    axis = _normalize(jnp.asarray(pmt_direction))
    relative_origin = origin - pmt_origin

    # Large front sphere, whose center lies 12.7 cm behind the PMT origin.
    sphere_center_relative = SPHERE_CENTER_Z * axis
    sphere_offset = relative_origin - sphere_center_relative
    sphere_near, sphere_far, sphere_discriminant = _ray_sphere_roots(
        sphere_offset, direction, SPHERE_RADIUS)
    sphere_near_position = relative_origin + sphere_near * direction
    sphere_far_position = relative_origin + sphere_far * direction
    sphere_near_axial = jnp.dot(sphere_near_position, axis)
    sphere_far_axial = jnp.dot(sphere_far_position, axis)
    sphere_near_valid = (
        (sphere_discriminant >= 0.0)
        & (sphere_near >= min_distance)
        & (sphere_near_axial >= SPHERE_TO_TORUS_Z - _CUT_TOL)
    )
    sphere_far_valid = (
        (sphere_discriminant >= 0.0)
        & (sphere_far >= min_distance)
        & (sphere_far_axial >= SPHERE_TO_TORUS_Z - _CUT_TOL)
    )
    sphere_distance = jnp.where(
        sphere_near_valid,
        sphere_near,
        jnp.where(sphere_far_valid, sphere_far, jnp.inf),
    )

    # The complete SK surface lies inside the 25.4 cm GEANT PMT sphere. Its
    # near/far roots provide a compact interval and a stable entry-side seed.
    bound_near, bound_far, bound_discriminant = _ray_sphere_roots(
        relative_origin, direction, GEANT_PMT_RADIUS)
    bound_lower = jnp.maximum(bound_near, min_distance)
    bound_valid = (
        (bound_discriminant >= 0.0)
        & (bound_far >= bound_lower)
    )
    torus_distance_raw = _torus_root_newton(
        relative_origin, direction, axis, bound_lower, bound_far)
    torus_position_raw = relative_origin + torus_distance_raw * direction
    torus_axial_raw, _, _ = _axial_and_transverse(
        torus_position_raw, axis)
    torus_residual = _torus_implicit(torus_position_raw, axis)
    torus_valid = (
        bound_valid
        & (torus_distance_raw >= min_distance)
        & (torus_axial_raw >= -_CUT_TOL)
        & (torus_axial_raw < SPHERE_TO_TORUS_Z + _CUT_TOL)
        & (jnp.abs(torus_residual) <= _ROOT_RESIDUAL_TOL)
    )
    torus_distance = jnp.where(torus_valid, torus_distance_raw, jnp.inf)

    use_sphere = sphere_distance <= torus_distance
    distance = jnp.minimum(sphere_distance, torus_distance)
    intersects = jnp.isfinite(distance)
    safe_distance = jnp.where(intersects, distance, 0.0)
    relative_position = relative_origin + safe_distance * direction
    axial, transverse, radial = _axial_and_transverse(
        relative_position, axis)

    sphere_normal = _safe_normalize(
        relative_position - sphere_center_relative)
    safe_radial = jnp.maximum(radial, _DERIVATIVE_EPS)
    torus_normal_raw = (
        (1.0 - TORUS_MAJOR_RADIUS / safe_radial) * transverse
        + axial * axis
    )
    torus_normal = _safe_normalize(torus_normal_raw)
    normal = jnp.where(use_sphere, sphere_normal, torus_normal)

    missing_vector = jnp.full_like(relative_position, jnp.nan)
    position = jnp.where(
        intersects, pmt_origin + relative_position, missing_vector)
    normal = jnp.where(intersects, normal, missing_vector)
    incidence_cosine = jnp.where(
        intersects,
        jnp.clip(-jnp.dot(direction, normal), 0.0, 1.0),
        0.0,
    )
    inactive_band = jnp.asarray(barrel) & (
        axial <= BARREL_INACTIVE_BAND_Z + _CUT_TOL)
    active = intersects & ~inactive_band
    surface = jnp.where(
        intersects,
        jnp.where(use_sphere,
                  SK_PMT_SURFACE_SPHERE, SK_PMT_SURFACE_TORUS),
        SK_PMT_SURFACE_MISS,
    )

    return JAXSK20InchPMTHit(
        intersects=intersects,
        active=active,
        distance=distance,
        position=position,
        normal=normal,
        surface=surface,
        incidence_cosine=incidence_cosine,
        axial_position=jnp.where(intersects, axial, jnp.nan),
        radial_position=jnp.where(intersects, radial, jnp.nan),
    )


__all__ = [
    "JAXSK20InchPMTHit",
    "SK_PMT_SURFACE_MISS",
    "SK_PMT_SURFACE_SPHERE",
    "SK_PMT_SURFACE_TORUS",
    "intersect_sk20inch_pmt_jax",
]
