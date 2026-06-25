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

import jax
import jax.numpy as jnp
from functools import partial

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


batch_intersect_two_spheres = jax.vmap(
    intersect_two_spheres_forward, in_axes=(0, 0, None, None))


def find_intersected_nested_sphere_sensors(ray_origins, ray_directions,
                                           sensor_positions, sensor_radius,
                                           r_inner, r_outer, n_divisions,
                                           inverted_sensor_map, temperature, overlap_prob):
    """Outer-sphere sensor lookup (reused) with inner-interface override.

    Computes the single-sphere result at ``r_outer`` (sensors), then for rays whose
    nearest surface is the inner interface overrides positions/normals and zeroes
    sensor weights, and tags ``hit_interface``.
    """
    single_ray = ray_origins.ndim == 1
    if single_ray:
        ray_origins = ray_origins[None, :]
        ray_directions = ray_directions[None, :]

    # 1. Outer-sphere sensor lookup — identical to the single-sphere propagator. cosθ angular
    #    acceptance is PR1 (a separate, all-geometry sensor-model change); PR2 is cosθ-free, so
    #    the nested and single-sphere detection use the SAME model (keeps the invisible-interface
    #    check self-consistent). The over-detection bias only skews the study contrast ratios,
    #    not the two-medium transport itself.
    result = find_intersected_sphere_sensors_differentiable(
        ray_origins, ray_directions, sensor_positions, sensor_radius,
        r_outer, n_divisions, inverted_sensor_map, temperature, overlap_prob)

    # 2. Inner interface vs the outer sphere, via the generic surface list. The inner region
    #    is ONE interface surface (a sphere at the origin, radius r_inner); a ray "hits the
    #    interface" only if it crosses that surface before the outer (sensor) sphere. This
    #    generalizes intersect_two_spheres_forward (offset / non-spherical inner) and reduces
    #    to it exactly for two concentric spheres (validated: 0 mismatches).
    _inner_centers = jnp.zeros((1, 3))
    _inner_radii = jnp.asarray([r_inner])

    def _interface_hit(o, d):
        t_out, _ = sphere_forward_t(o, d, jnp.zeros(3), r_outer)
        _t, _which, hit, pt, nrm = nearest_interface(o, d, _inner_centers, _inner_radii, t_outer=t_out)
        return hit, pt, nrm

    hit_inner, inner_pts, inner_norms = jax.vmap(_interface_hit)(ray_origins, ray_directions)

    # 3. Override geometry hit for interface rays; mask their sensor weights.
    hi = hit_inner[:, None]                                   # (n_rays, 1)
    result['positions'] = jnp.where(hi, inner_pts, result['positions'])
    result['normals'] = jnp.where(hi, inner_norms, result['normals'])
    # sensor_weights / inside_sensor are (max_cand, n_rays) → mask along n_rays.
    keep = (~hit_inner)[None, :]
    result['sensor_weights'] = result['sensor_weights'] * keep
    result['inside_sensor'] = result['inside_sensor'] & keep
    result['hit_interface'] = hit_inner                      # (n_rays,) bool

    if single_ray:
        result = jax.tree_map(lambda x: x[0], result)
    return result


def create_nested_sphere_propagator(sensor_positions, sensor_radius,
                                    r_inner, r_outer, n_divisions=50,
                                    temperature=0.2, max_candidates_per_ray=4,
                                    overlap_st_width_frac=0.35, overlap_renorm=1.0,
                                    overlap_mode='interp'):
    """JIT-compiled photon propagator for the nested two-sphere geometry.

    Sensors live on the outer sphere; the grid/sensor-map machinery is the
    single-sphere machinery at ``r_outer``. Returns the same dict as the single-sphere
    propagator plus a per-ray ``hit_interface`` flag.
    """
    sensor_positions = jnp.asarray(sensor_positions)

    assignments_geometric = assign_sensors_to_sphere_grid(
        sensor_positions, sensor_radius, r_outer, n_divisions)
    _ = create_sensor_sphere_grid_map(assignments_geometric, n_divisions)
    assignments_distance = find_closest_sensors(
        calculate_sphere_grid_centers(r_outer, n_divisions),
        sensor_positions, max_candidates_per_ray)
    inverted_sensor_map = create_inverted_sphere_sensor_map(
        assignments_geometric, assignments_distance, n_divisions,
        max_candidates_per_ray, sensor_positions.shape[0])

    if temperature is None:
        overlap_prob = create_overlap_prob(
            None, sensor_radius, st_width_frac=overlap_st_width_frac,
            renorm=overlap_renorm, mode=overlap_mode)
    else:
        overlap_prob = create_overlap_prob(
            temperature * sensor_radius, sensor_radius,
            st_width_frac=overlap_st_width_frac, renorm=overlap_renorm, mode=overlap_mode)

    @jax.jit
    def propagate_photons(photon_origins, photon_directions):
        return find_intersected_nested_sphere_sensors(
            photon_origins, photon_directions, sensor_positions, sensor_radius,
            r_inner, r_outer, n_divisions, inverted_sensor_map,
            temperature, overlap_prob)

    return propagate_photons
