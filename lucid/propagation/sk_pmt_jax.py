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

_TORUS_BRACKET_SAMPLES = 16
_TORUS_BISECTION_ITERATIONS = 8
_ROOT_REFINEMENT_ITERATIONS = 8
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
    """Return ordered roots using a stable closest-approach construction."""
    a = jnp.dot(direction, direction)
    center_distance = -jnp.dot(relative_origin, direction) / a
    closest = relative_origin + center_distance * direction
    perpendicular2 = jnp.dot(closest, closest)
    radial2 = radius**2 - perpendicular2
    half_chord = jnp.sqrt(jnp.maximum(radial2 / a, 0.0))
    # Preserve the old discriminant sign/API for downstream hit tests.
    discriminant = 4.0 * a * radial2
    return (center_distance - half_chord,
            center_distance + half_chord,
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


def _torus_root_sampled(relative_origin, direction, axis, upper):
    """Bracket the first physical-branch root and refine it with Newton.

    Return the local hit position as well as the distance.  Reconstructing it
    later from a detector-scale origin and distance loses the centimetre-scale
    torus coordinates in float32.
    """
    sample_fraction = jnp.linspace(
        1.0 / _TORUS_BRACKET_SAMPLES, 1.0,
        _TORUS_BRACKET_SAMPLES, dtype=relative_origin.dtype)
    sample_distance = upper * sample_fraction
    sample_positions = (
        relative_origin[None, :]
        + sample_distance[:, None] * direction)
    sample_values = jax.vmap(
        lambda position: _torus_implicit(position, axis))(
            sample_positions)
    inside = sample_values <= 0.0
    first_inside = jnp.argmax(inside)
    found = jnp.any(inside)
    bracket_inside = sample_distance[first_inside]
    bracket_outside = jnp.where(
        first_inside > 0,
        sample_distance[jnp.maximum(first_inside - 1, 0)],
        0.0,
    )

    def bisection_iteration(_, bounds):
        outside, inside_distance = bounds
        middle = 0.5 * (outside + inside_distance)
        value = _torus_implicit(
            relative_origin + middle * direction, axis)
        middle_inside = value <= 0.0
        return (jnp.where(middle_inside, outside, middle),
                jnp.where(middle_inside, middle, inside_distance))

    bracket_outside, bracket_inside = jax.lax.fori_loop(
        0, _TORUS_BISECTION_ITERATIONS, bisection_iteration,
        (bracket_outside, bracket_inside))
    traced = 0.5 * (bracket_outside + bracket_inside)

    def refine_iteration(_, distance):
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
        return jnp.clip(
            distance - value / safe_derivative, 0.0, upper)

    root = jax.lax.fori_loop(
        0, _ROOT_REFINEMENT_ITERATIONS, refine_iteration, traced)
    position = relative_origin + root * direction
    axial, _, _ = _axial_and_transverse(position, axis)
    residual = _torus_implicit(position, axis)
    valid = (
        found
        & (jnp.abs(residual) <= _ROOT_RESIDUAL_TOL)
        & (axial >= -_CUT_TOL)
        & (axial < SPHERE_TO_TORUS_Z + _CUT_TOL)
    )
    return root, position, valid


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
    # Rebase the torus solve at the nearby GEANT PMT boundary.  Direct laser
    # rays begin tens of metres away, and evaluating metre-scale origins while
    # solving a centimetre-scale torus loses enough float32 precision to miss
    # grazing roots.  The global distance remains differentiable because the
    # entry distance and local root are both carried through JAX.
    torus_bound_origin = relative_origin + bound_lower * direction
    torus_bound_interval = jnp.maximum(bound_far - bound_lower, 0.0)
    # The full spindle torus has mathematical roots outside the physical SK
    # branch. Restrict tracing to 0 <= axial < sphere/torus join so it cannot
    # stop on one of those discarded roots.
    bound_axial = jnp.dot(torus_bound_origin, axis)
    axial_direction = jnp.dot(direction, axis)
    safe_axial_direction = jnp.where(
        jnp.abs(axial_direction) >= _DERIVATIVE_EPS,
        axial_direction,
        jnp.where(axial_direction >= 0.0,
                  _DERIVATIVE_EPS, -_DERIVATIVE_EPS),
    )
    axial_zero_distance = -bound_axial / safe_axial_direction
    axial_join_distance = (
        SPHERE_TO_TORUS_Z - bound_axial) / safe_axial_direction
    axial_slab_lower = jnp.maximum(
        0.0, jnp.minimum(axial_zero_distance, axial_join_distance))
    axial_slab_upper = jnp.minimum(
        torus_bound_interval,
        jnp.maximum(axial_zero_distance, axial_join_distance))
    parallel_inside_slab = (
        (bound_axial >= -_CUT_TOL)
        & (bound_axial < SPHERE_TO_TORUS_Z + _CUT_TOL)
    )
    nonparallel = jnp.abs(axial_direction) >= _DERIVATIVE_EPS
    axial_slab_lower = jnp.where(nonparallel, axial_slab_lower, 0.0)
    axial_slab_upper = jnp.where(
        nonparallel, axial_slab_upper,
        jnp.where(parallel_inside_slab, torus_bound_interval, -1.0))
    axial_slab_valid = axial_slab_upper >= axial_slab_lower
    torus_origin = (
        torus_bound_origin + axial_slab_lower * direction)
    torus_interval = jnp.maximum(
        axial_slab_upper - axial_slab_lower, 0.0)
    torus_local_distance, torus_slab_position, torus_crosses = (
        _torus_root_sampled(
            torus_origin, direction, axis, torus_interval)
    )
    torus_distance_raw = (
        bound_lower + axial_slab_lower + torus_local_distance)
    torus_position_raw = torus_slab_position
    torus_axial_raw, _, _ = _axial_and_transverse(
        torus_position_raw, axis)
    torus_residual = _torus_implicit(torus_position_raw, axis)
    torus_valid = (
        bound_valid
        & axial_slab_valid
        & torus_crosses
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
    sphere_relative_position = relative_origin + safe_distance * direction
    relative_position = jnp.where(
        use_sphere, sphere_relative_position, torus_position_raw)
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
