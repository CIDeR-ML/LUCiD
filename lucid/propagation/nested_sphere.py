"""Nested two-sphere photon propagation (JUNO-like: inner LS sphere + outer water shell).

The detector is two concentric spheres sharing the origin:

  * **outer sphere** at ``R_out`` — the PMT surface + reflecting wall (sensors live here,
    exactly as in the single :mod:`lucid.propagation.sphere` case).
  * **inner sphere** at ``R_in`` — the liquid-scintillator ↔ water optical interface
    (NO sensors; a photon crossing it refracts / Fresnel-reflects).

A propagation *step* casts each ray to the **nearest forward surface** among the two
spheres. When that surface is the inner one the ray has hit the interface (``hit_interface``
True): it carries no sensor detection this step (it must refract first), and the downstream
photon step does the Snell/Fresnel/TIR physics. When the nearest surface is the outer one
the behaviour is byte-identical to the single-sphere propagator.

Implementation note: this wraps the validated single-sphere differentiable propagator for
the outer-sphere sensor lookup, then overrides positions/normals and masks sensor weights
for the rays whose nearest surface is the inner interface. Outer-hit rays are therefore
untouched relative to the single-sphere path.
"""

from functools import partial
import logging
import time

import jax
import jax.numpy as jnp

from .sphere import (
    find_intersected_sphere_sensors_differentiable,
    assign_sensors_to_sphere_grid,
    create_sensor_sphere_grid_map,
    calculate_sphere_grid_centers,
    create_inverted_sphere_sensor_map,
)
from .base import find_closest_sensors
from .surfaces import sphere_forward_t, nearest_interface
from ..overlap import create_overlap_prob


LOG = logging.getLogger("lucid.propagation.nested_sphere")


def _sphere_forward_t(origin, direction, radius):
    """Smallest strictly-positive ray parameter ``t`` at which the ray crosses a sphere.

    Handles both sides transparently:

    * origin INSIDE the sphere (c < 0) → roots straddle 0 → forward crossing is the
      positive (exit) root.
    * origin OUTSIDE the sphere (c > 0) pointing toward it → both roots positive →
      forward crossing is the smaller (entry) root.
    * pointing away / missing → returns ``LARGE``.

    Returns ``(t, valid)`` where ``t`` is the forward crossing distance (``LARGE`` if none).
    """
    LARGE = 1e10
    eps = 1e-6
    a = jnp.sum(direction * direction)
    b = 2.0 * jnp.sum(origin * direction)
    c = jnp.sum(origin * origin) - radius * radius
    disc = b * b - 4.0 * a * c
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    t_small = (-b - sqrt_disc) / (2.0 * a)
    t_large = (-b + sqrt_disc) / (2.0 * a)
    # smallest strictly-positive root
    t = jnp.where(t_small > eps, t_small,
                  jnp.where(t_large > eps, t_large, LARGE))
    valid = (disc >= 0.0) & (t < LARGE)
    return jnp.where(valid, t, LARGE), valid


def intersect_two_spheres_forward(origin, direction, r_inner, r_outer):
    """Nearest forward surface among the inner and outer spheres (single ray).

    Returns
    -------
    t_hit : float          distance to the nearest forward surface
    hit_inner : bool       True if that surface is the inner interface
    point : (3,)           the hit position
    normal : (3,)          outward (radial) unit normal at the hit
    """
    t_out, _ = _sphere_forward_t(origin, direction, r_outer)
    t_in, valid_in = _sphere_forward_t(origin, direction, r_inner)
    hit_inner = valid_in & (t_in < t_out)
    t_hit = jnp.where(hit_inner, t_in, t_out)
    point = origin + t_hit * direction
    normal = point / (jnp.linalg.norm(point) + 1e-10)   # outward radial
    return t_hit, hit_inner, point, normal


def intersect_three_spheres_forward(origin, direction, r_ls, r_acrylic,
                                    r_outer):
    """Nearest forward surface for a three-layer concentric shell.

    Parameters
    ----------
    origin, direction : jnp.ndarray
        One ray origin and direction.  The direction need not be unit length.
    r_ls, r_acrylic, r_outer : float
        Increasing radii for the LS--Acrylic interface, Acrylic--Water
        interface, and the outer instrumented surface.

    Returns
    -------
    t_hit : float
        Distance to the nearest strictly-forward surface.
    surface_index : int
        ``0`` for LS--Acrylic, ``1`` for Acrylic--Water, ``2`` for the
        outer surface, or ``-1`` when no forward surface exists.
    point, normal : jnp.ndarray
        Hit point and outward radial normal.

    This helper is intentionally independent of sensor lookup and photon
    transport.  It provides the deterministic geometry contract used by the
    later three-medium propagator.
    """
    radii = jnp.asarray([r_ls, r_acrylic, r_outer])
    ts, valid = jax.vmap(
        lambda radius: _sphere_forward_t(origin, direction, radius)
    )(radii)
    index = jnp.argmin(ts)
    has_hit = jnp.any(valid)
    surface_index = jnp.where(has_hit, index, -jnp.ones_like(index))
    t_hit = ts[index]
    point = origin + t_hit * direction
    normal = point / (jnp.linalg.norm(point) + 1e-10)
    return t_hit, surface_index, point, normal


batch_intersect_two_spheres = jax.vmap(
    intersect_two_spheres_forward, in_axes=(0, 0, None, None))


def find_intersected_nested_sphere_sensors(ray_origins, ray_directions,
                                           sensor_positions, sensor_radius,
                                           r_inner, r_outer, n_divisions,
                                           inverted_sensor_map, temperature,
                                           overlap_prob,
                                           apply_angular_acceptance=False,
                                           r_acrylic=None,
                                           sensor_inward_axes=None,
                                           sensor_radial_radii=None,
                                           sensor_axial_radii=None):
    """Outer-sphere sensor lookup (reused) with inner-interface override.

    Computes the single-sphere result at ``r_outer`` (sensors), then for rays whose
    nearest surface is an internal interface overrides positions/normals and zeroes
    sensor weights.  With ``r_acrylic`` supplied, ``interface_index`` identifies
    the crossed boundary: 0 for LS--Acrylic, 1 for Acrylic--Water, and -1 for
    an outer-surface hit.  The historical two-layer call remains unchanged.
    """
    single_ray = ray_origins.ndim == 1
    if single_ray:
        ray_origins = ray_origins[None, :]
        ray_directions = ray_directions[None, :]

    # 1. Outer-sphere sensor lookup — identical to the single-sphere propagator. The PMT cosθ
    #    angular acceptance (radial normal — sensors are on the outer sphere) is OPT-IN via the
    #    detector config (apply_angular_acceptance); default OFF keeps it byte-identical and
    #    consistent with a single sphere that hasn't opted in.
    result = find_intersected_sphere_sensors_differentiable(
        ray_origins, ray_directions, sensor_positions, sensor_radius,
        r_outer, n_divisions, inverted_sensor_map, temperature, overlap_prob,
        apply_angular_acceptance=apply_angular_acceptance,
        sensor_inward_axes=sensor_inward_axes,
        sensor_radial_radii=sensor_radial_radii,
        sensor_axial_radii=sensor_axial_radii)

    # 2. Inner interface vs the outer sphere, via the generic surface list. The inner region
    #    is ONE interface surface (a sphere at the origin, radius r_inner); a ray "hits the
    #    interface" only if it crosses that surface before the outer (sensor) sphere. This
    #    generalizes intersect_two_spheres_forward (offset / non-spherical inner) and reduces
    #    to it exactly for two concentric spheres (validated: 0 mismatches).
    if r_acrylic is None:
        _inner_radii = jnp.asarray([r_inner])
    else:
        _inner_radii = jnp.asarray([r_inner, r_acrylic])
    _inner_centers = jnp.zeros((_inner_radii.shape[0], 3))

    def _interface_hit(o, d):
        t_out, _ = sphere_forward_t(o, d, jnp.zeros(3), r_outer)
        _t, _which, hit, pt, nrm = nearest_interface(
            o, d, _inner_centers, _inner_radii, t_outer=t_out)
        return hit, _which, pt, nrm

    hit_inner, interface_indices, inner_pts, inner_norms = jax.vmap(
        _interface_hit)(ray_origins, ray_directions)

    # 3. Override geometry hit for interface rays; mask their sensor weights.
    hi = hit_inner[:, None]                                   # (n_rays, 1)
    result['positions'] = jnp.where(hi, inner_pts, result['positions'])
    result['normals'] = jnp.where(hi, inner_norms, result['normals'])
    # sensor_weights / inside_sensor are (max_cand, n_rays) → mask along n_rays.
    keep = (~hit_inner)[None, :]
    result['sensor_weights'] = result['sensor_weights'] * keep
    result['inside_sensor'] = result['inside_sensor'] & keep
    result['hit_interface'] = hit_inner                      # (n_rays,) bool
    if r_acrylic is not None:
        result['interface_index'] = jnp.where(
            hit_inner, interface_indices, -1)

    if single_ray:
        result = jax.tree_map(lambda x: x[0], result)
    return result


def create_nested_sphere_propagator(sensor_positions, sensor_radius,
                                    r_inner, r_outer, n_divisions=50,
                                    temperature=0.2, max_candidates_per_ray=4,
                                    overlap_st_width_frac=0.35,
                                    overlap_renorm=1.0,
                                    overlap_mode='interp',
                                    apply_angular_acceptance=False,
                                    r_acrylic=None,
                                    sensor_inward_axes=None,
                                    sensor_radial_radii=None,
                                    sensor_axial_radii=None):
    """JIT-compiled photon propagator for nested sphere geometry.

    The outer sphere supplies the grid lookup. Optional per-sensor axes and
    radii replace the legacy spherical receiver with detector-facing PMT
    ellipsoid heads. The result includes a per-ray ``hit_interface`` flag;
    with ``r_acrylic`` it also includes the internal ``interface_index``.
    """
    started = time.perf_counter()
    sensor_positions = jnp.asarray(sensor_positions)
    pmt_geometry = (
        sensor_inward_axes,
        sensor_radial_radii,
        sensor_axial_radii,
    )
    if any(value is None for value in pmt_geometry) and not all(
        value is None for value in pmt_geometry
    ):
        raise ValueError(
            "PMT axes, radial radii, and axial radii must be supplied together"
        )
    if sensor_inward_axes is not None:
        sensor_inward_axes = jnp.asarray(sensor_inward_axes)
        sensor_radial_radii = jnp.asarray(sensor_radial_radii)
        sensor_axial_radii = jnp.asarray(sensor_axial_radii)
        count = sensor_positions.shape[0]
        if sensor_inward_axes.shape != (count, 3):
            raise ValueError(
                f"PMT axes must have shape ({count}, 3), got "
                f"{sensor_inward_axes.shape}"
            )
        if (
            sensor_radial_radii.shape != (count,)
            or sensor_axial_radii.shape != (count,)
        ):
            raise ValueError("PMT radii must contain one value per sensor")
    LOG.info("sensor map: assigning geometric PMT candidates")

    assignments_geometric = assign_sensors_to_sphere_grid(
        sensor_positions, sensor_radius, r_outer, n_divisions)
    jax.block_until_ready(assignments_geometric)
    LOG.info("sensor map: geometric candidates ready (%.2fs)",
             time.perf_counter() - started)

    LOG.info("sensor map: creating sphere-grid lookup")
    sensor_grid_map = create_sensor_sphere_grid_map(
        assignments_geometric, n_divisions)
    jax.block_until_ready(sensor_grid_map)
    LOG.info("sensor map: sphere-grid lookup ready (%.2fs)",
             time.perf_counter() - started)

    LOG.info("sensor map: finding nearest PMT candidates")
    assignments_distance = find_closest_sensors(
        calculate_sphere_grid_centers(r_outer, n_divisions),
        sensor_positions, max_candidates_per_ray)
    jax.block_until_ready(assignments_distance)
    LOG.info("sensor map: nearest candidates ready (%.2fs)",
             time.perf_counter() - started)

    LOG.info("sensor map: building inverted cell-to-PMT lookup")
    inverted_sensor_map = create_inverted_sphere_sensor_map(
        assignments_geometric, assignments_distance, n_divisions,
        max_candidates_per_ray, sensor_positions.shape[0])
    jax.block_until_ready(inverted_sensor_map)
    LOG.info("sensor map: inverted lookup ready (%.2fs total)",
             time.perf_counter() - started)

    if temperature is None:
        overlap_prob = create_overlap_prob(
            None, sensor_radius, st_width_frac=overlap_st_width_frac,
            renorm=overlap_renorm, mode=overlap_mode)
    else:
        overlap_prob = create_overlap_prob(
            temperature * sensor_radius, sensor_radius,
            st_width_frac=overlap_st_width_frac, renorm=overlap_renorm, mode=overlap_mode)

    LOG.info("sensor map: nested-sphere propagator ready (%.2fs total)",
             time.perf_counter() - started)

    @jax.jit
    def propagate_photons(photon_origins, photon_directions):
        return find_intersected_nested_sphere_sensors(
            photon_origins, photon_directions, sensor_positions, sensor_radius,
            r_inner, r_outer, n_divisions, inverted_sensor_map,
            temperature, overlap_prob,
            apply_angular_acceptance=apply_angular_acceptance,
            r_acrylic=r_acrylic,
            sensor_inward_axes=sensor_inward_axes,
            sensor_radial_radii=sensor_radial_radii,
            sensor_axial_radii=sensor_axial_radii)

    return propagate_photons
