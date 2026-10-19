"""Smooth Gaussian coverage of the projected SK 20-inch photocathode.

This is a differentiable surface-quadrature reference. It integrates a
Gaussian bundle of parallel rays over the projected curved photocathode. The
same continuous value is used in the forward and backward passes; no
straight-through estimator is involved.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .sk_pmt import (
    BARREL_INACTIVE_BAND_Z,
    SPHERE_CENTER_Z,
    SPHERE_RADIUS,
    SPHERE_TO_TORUS_Z,
    TORUS_MAJOR_RADIUS,
    TORUS_MINOR_RADIUS,
)


class SKPMTSurfaceQuadrature(NamedTuple):
    """Axisymmetric surface nodes and their physical area weights."""

    radial_position: jnp.ndarray
    axial_position: jnp.ndarray
    normal_radial: jnp.ndarray
    normal_axial: jnp.ndarray
    cos_azimuth: jnp.ndarray
    sin_azimuth: jnp.ndarray
    area_weight: jnp.ndarray


def _legendre_interval(order, lower, upper):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    scale = 0.5 * (upper - lower)
    return scale * nodes + 0.5 * (upper + lower), scale * weights


def create_sk20inch_surface_quadrature(
    n_polar=48,
    n_azimuth=96,
    *,
    barrel=False,
):
    """Build fixed quadrature nodes for the active SK photocathode.

    ``n_polar`` nodes are used independently on the sphere and torus patches.
    For barrel PMTs, the torus interval begins above SKDONUTS' inactive 2 cm
    base band. The returned arrays can be closed over by a jitted/vmapped JAX
    function.
    """
    if int(n_polar) != n_polar or n_polar <= 0:
        raise ValueError("n_polar must be a positive integer")
    if int(n_azimuth) != n_azimuth or n_azimuth <= 0:
        raise ValueError("n_azimuth must be a positive integer")
    n_polar = int(n_polar)
    n_azimuth = int(n_azimuth)

    sphere_cosine_at_join = (
        (SPHERE_TO_TORUS_Z - SPHERE_CENTER_Z) / SPHERE_RADIUS
    )
    sphere_angle_max = np.arccos(sphere_cosine_at_join)
    sphere_angle, sphere_weight = _legendre_interval(
        n_polar, 0.0, sphere_angle_max)
    sphere_radial = SPHERE_RADIUS * np.sin(sphere_angle)
    sphere_axial = (
        SPHERE_CENTER_Z + SPHERE_RADIUS * np.cos(sphere_angle)
    )
    sphere_normal_radial = np.sin(sphere_angle)
    sphere_normal_axial = np.cos(sphere_angle)
    sphere_area = (
        SPHERE_RADIUS**2 * np.sin(sphere_angle) * sphere_weight
    )

    # The visible envelope switches at the sphere's radial coordinate at the
    # SK z threshold. The torus lies about 1 mm behind it at this seam because
    # SKDONUTS' historical constants are rounded.
    radial_at_join = (
        SPHERE_RADIUS * np.sin(sphere_angle_max)
    )
    torus_angle_max = np.arccos(
        (radial_at_join - TORUS_MAJOR_RADIUS) / TORUS_MINOR_RADIUS)
    torus_angle_min = (
        np.arcsin(BARREL_INACTIVE_BAND_Z / TORUS_MINOR_RADIUS)
        if barrel else 0.0
    )
    torus_angle, torus_weight = _legendre_interval(
        n_polar, torus_angle_min, torus_angle_max)
    torus_radial = (
        TORUS_MAJOR_RADIUS + TORUS_MINOR_RADIUS * np.cos(torus_angle)
    )
    torus_axial = TORUS_MINOR_RADIUS * np.sin(torus_angle)
    torus_normal_radial = np.cos(torus_angle)
    torus_normal_axial = np.sin(torus_angle)
    torus_area = (
        TORUS_MINOR_RADIUS * torus_radial * torus_weight
    )

    # The rounded SKDONUTS constants leave a roughly 1 mm axial mismatch at
    # the sphere/torus switch. The outer boundary of the piecewise solid has a
    # narrow cylindrical seam joining those two curves. It projects to zero
    # area at normal incidence but closes the small oblique-incidence gap.
    seam_axial_lower = (
        TORUS_MINOR_RADIUS * np.sin(torus_angle_max)
    )
    seam_axial, seam_weight = _legendre_interval(
        n_polar, seam_axial_lower, SPHERE_TO_TORUS_Z)
    seam_radial = np.full_like(seam_axial, radial_at_join)
    seam_normal_radial = np.ones_like(seam_axial)
    seam_normal_axial = np.zeros_like(seam_axial)
    seam_area = radial_at_join * seam_weight

    radial = np.concatenate([sphere_radial, torus_radial, seam_radial])
    axial = np.concatenate([sphere_axial, torus_axial, seam_axial])
    normal_radial = np.concatenate([
        sphere_normal_radial, torus_normal_radial, seam_normal_radial])
    normal_axial = np.concatenate([
        sphere_normal_axial, torus_normal_axial, seam_normal_axial])
    polar_area = np.concatenate([sphere_area, torus_area, seam_area])

    # Midpoint azimuths preserve opposite-node symmetry for even orders.
    azimuth = (np.arange(n_azimuth) + 0.5) * (2.0 * np.pi / n_azimuth)
    azimuth_weight = 2.0 * np.pi / n_azimuth

    def expand(values):
        return np.repeat(values, n_azimuth)

    return SKPMTSurfaceQuadrature(
        radial_position=jnp.asarray(expand(radial)),
        axial_position=jnp.asarray(expand(axial)),
        normal_radial=jnp.asarray(expand(normal_radial)),
        normal_axial=jnp.asarray(expand(normal_axial)),
        cos_azimuth=jnp.asarray(np.tile(np.cos(azimuth), len(radial))),
        sin_azimuth=jnp.asarray(np.tile(np.sin(azimuth), len(radial))),
        area_weight=jnp.asarray(expand(polar_area) * azimuth_weight),
    )


def _normalize(vector):
    return vector / jnp.linalg.norm(vector)


def _transverse_basis(axis):
    """Duff et al. orthonormal basis, stable at both coordinate poles."""
    sign = jnp.copysign(1.0, axis[2])
    factor = -1.0 / (sign + axis[2])
    cross_term = axis[0] * axis[1] * factor
    first = jnp.array([
        1.0 + sign * axis[0] * axis[0] * factor,
        sign * cross_term,
        -sign * axis[0],
    ])
    second = jnp.array([
        cross_term,
        sign + axis[1] * axis[1] * factor,
        -axis[1],
    ])
    return first, second


def _world_surface(pmt_position, pmt_direction, quadrature):
    pmt_position = jnp.asarray(pmt_position)
    axis = _normalize(jnp.asarray(pmt_direction))
    first, second = _transverse_basis(axis)
    radial_direction = (
        quadrature.cos_azimuth[:, None] * first[None, :]
        + quadrature.sin_azimuth[:, None] * second[None, :]
    )
    relative_position = (
        quadrature.radial_position[:, None] * radial_direction
        + quadrature.axial_position[:, None] * axis[None, :]
    )
    normal = (
        quadrature.normal_radial[:, None] * radial_direction
        + quadrature.normal_axial[:, None] * axis[None, :]
    )
    return pmt_position[None, :] + relative_position, normal


def sk20inch_projected_area(ray_direction, pmt_direction, quadrature):
    """Integrate the active curved surface's orthographic projected area."""
    direction = _normalize(jnp.asarray(ray_direction))
    _, normal = _world_surface(jnp.zeros(3), pmt_direction, quadrature)
    projection = jnp.maximum(0.0, -normal @ direction)
    return jnp.sum(projection * quadrature.area_weight)


def sk20inch_gaussian_coverage(
    ray_origin,
    ray_direction,
    pmt_position,
    pmt_direction,
    sigma,
    quadrature,
    *,
    min_distance=0.0,
):
    """Probability that a Gaussian parallel-ray bundle hits the photocathode.

    The Gaussian is defined in the plane perpendicular to ``ray_direction``
    and centred on the supplied ray. ``sigma`` is its transverse standard
    deviation in metres. Surface elements behind the ray origin are excluded;
    detector propagation normally guarantees all candidate PMTs are forward.
    """
    origin = jnp.asarray(ray_origin)
    direction = _normalize(jnp.asarray(ray_direction))
    surface_position, normal = _world_surface(
        pmt_position, pmt_direction, quadrature)

    displacement = surface_position - origin[None, :]
    along_ray = displacement @ direction
    perpendicular = displacement - along_ray[:, None] * direction[None, :]
    perpendicular_squared = jnp.sum(perpendicular * perpendicular, axis=1)
    gaussian_density = (
        jnp.exp(-0.5 * perpendicular_squared / sigma**2)
        / (2.0 * jnp.pi * sigma**2)
    )
    projection = jnp.maximum(0.0, -(normal @ direction))
    forward = along_ray >= min_distance
    return jnp.sum(
        jnp.where(forward, gaussian_density * projection, 0.0)
        * quadrature.area_weight
    )


__all__ = [
    "SKPMTSurfaceQuadrature",
    "create_sk20inch_surface_quadrature",
    "sk20inch_gaussian_coverage",
    "sk20inch_projected_area",
]
