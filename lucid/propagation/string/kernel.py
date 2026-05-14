"""
Core math kernels for string-telescope photon propagation (REFERENCE — superseded by fast.py).

Provides JIT-compatible functions for:
  1. Skew-line distance from a photon ray to a string axis
  2. Closest-approach parameters (t on ray, s on string)
  3. DOM snap: given s*, find the n_dom_snap nearest DOMs via searchsorted

All functions operate on single (ray, string) pairs and are designed
to be composed with jax.vmap for batching.
"""

import jax
import jax.numpy as jnp
from functools import partial

EPS = 1e-9


@jax.jit
def skew_line_distance(ray_origin, ray_direction, string_anchor, string_axis):
    """Distance and closest-approach parameters for a ray and a string axis.

    For ray R(t) = O + t*D and line L(s) = P + s*a (|a| = 1):

        denom = |D|² - (D·a)²  =  |D × a|²

        t* = ((D·a)(w·a) - (w·D)) / denom
        s* = ((w·a)|D|² - (w·D)(D·a)) / denom

        dist = |w · (D × a)| / |D × a|

    where w = O - P.

    When D is parallel to a (denom → 0), falls back to the perpendicular
    distance from the ray origin to the string axis, with s* set to the
    projection of O onto the axis from the anchor point.

    Parameters
    ----------
    ray_origin : (3,)
    ray_direction : (3,)      need not be unit length
    string_anchor : (3,)      a point on the string (e.g., bottom DOM)
    string_axis : (3,)        unit vector along string

    Returns
    -------
    distance : scalar    perpendicular distance between the two lines
    t_ray : scalar       parameter on ray at closest approach
    s_string : scalar    parameter on string axis at closest approach
    """
    w = ray_origin - string_anchor
    d = ray_direction
    a = string_axis

    dd = jnp.dot(d, d)       # |D|²
    da = jnp.dot(d, a)       # D·a
    wd = jnp.dot(w, d)       # w·D
    wa = jnp.dot(w, a)       # w·a

    denom = dd - da * da      # |D × a|² — zero when parallel

    # --- skew (non-parallel) case ---
    denom_safe = denom + EPS
    t_skew = (da * wa - wd) / denom_safe
    s_skew = (wa * dd - wd * da) / denom_safe

    cross = jnp.cross(d, a)
    dist_skew = jnp.abs(jnp.dot(w, cross)) / jnp.sqrt(denom + EPS * EPS)

    # --- parallel fallback ---
    # perpendicular distance from anchor to ray
    ww = jnp.dot(w, w)
    dist_parallel = jnp.sqrt(jnp.maximum(ww - wa * wa, 0.0) + EPS * EPS)

    # s* = projection of ray_origin onto string from anchor
    s_parallel = wa

    # t* = closest point on ray to (anchor + s*·axis)
    # minimize |O + t*D - (P + s*a)|² over t
    # → t* = -((w - s*a)·D) / |D|²
    residual = w - s_parallel * a
    t_parallel = -jnp.dot(residual, d) / (dd + EPS)

    # --- blend via smooth indicator ---
    is_parallel = denom < EPS
    distance = jnp.where(is_parallel, dist_parallel, dist_skew)
    t_ray = jnp.where(is_parallel, t_parallel, t_skew)
    s_string = jnp.where(is_parallel, s_parallel, s_skew)

    return distance, t_ray, s_string


@partial(jax.jit, static_argnums=(2,))
def snap_to_doms(s_string, dom_s_offsets, n_dom_snap):
    """Find the n_dom_snap nearest DOMs to s* along a string.

    dom_s_offsets is a sorted array of arc-length positions for each DOM
    on the string. Entries beyond the actual DOM count are set to +inf
    (so searchsorted naturally ignores them).

    Returns n_dom_snap contiguous indices centered on the bracket pair
    around s_string, clamped to [0, max_dom - 1].

    Parameters
    ----------
    s_string : scalar      arc-length parameter on string
    dom_s_offsets : (max_dom,)  sorted DOM positions (+inf padding)
    n_dom_snap : int       compile-time static; total DOMs to return (>= 2)

    Returns
    -------
    indices : (n_dom_snap,) int array of local DOM indices
    """
    max_dom = dom_s_offsets.shape[0]
    k_right = jnp.searchsorted(dom_s_offsets, s_string, side='right')
    k_start = k_right - n_dom_snap // 2
    k_start = jnp.clip(k_start, 0, max_dom - n_dom_snap)
    indices = k_start + jnp.arange(n_dom_snap)
    return indices


@jax.jit
def string_z_range_check(s_string, s_min, s_max, r_filter):
    """Check whether closest approach falls within the string's DOM range.

    A photon whose closest-approach s* is far outside [s_min, s_max]
    cannot hit any DOM on this string (even accounting for sensor_radius).
    This is a cheap second-stage filter after the distance test, avoiding
    wasted ray-sphere ops on strings where the DOMs are above/below.

    Parameters
    ----------
    s_string : scalar      closest-approach parameter on string axis
    s_min : scalar         arc-length of bottom DOM
    s_max : scalar         arc-length of top DOM
    r_filter : scalar      sensor_radius + padding (soft-kernel, curvature)

    Returns
    -------
    in_range : bool
    """
    return (s_string >= s_min - r_filter) & (s_string <= s_max + r_filter)


@partial(jax.jit, static_argnums=(9,))
def candidate_doms_for_ray_string(
    ray_origin,
    ray_direction,
    string_anchor,
    string_axis,
    dom_s_offsets,
    dom_global_ids,
    s_min,
    s_max,
    r_filter,
    n_dom_snap,
):
    """Full pipeline for one (ray, string) pair.

    1. Compute skew-line distance and closest-approach params.
    2. Check distance < r_filter.
    3. Check s* within string z-range (± r_filter).
    4. Snap to n_dom_snap DOMs.
    5. Return global DOM IDs (masked to -1 if checks fail).

    Parameters
    ----------
    ray_origin : (3,)
    ray_direction : (3,)
    string_anchor : (3,)
    string_axis : (3,)
    dom_s_offsets : (max_dom,)    sorted, +inf padded
    dom_global_ids : (max_dom,)   global sensor index, -1 padded
    s_min, s_max : scalars        arc-length range of actual DOMs
    r_filter : scalar             distance threshold for pre-filter
    n_dom_snap : int              static; DOMs to return per string

    Returns
    -------
    global_ids : (n_dom_snap,) int   DOM global IDs, -1 if filtered out
    distance : scalar                distance to string axis
    t_ray : scalar                   parameter on ray at closest approach
    """
    distance, t_ray, s_string = skew_line_distance(
        ray_origin, ray_direction, string_anchor, string_axis
    )

    passes_distance = distance < r_filter
    passes_z = string_z_range_check(s_string, s_min, s_max, r_filter)
    passes = passes_distance & passes_z

    local_indices = snap_to_doms(s_string, dom_s_offsets, n_dom_snap)
    global_ids = jnp.where(passes, dom_global_ids[local_indices], -1)

    return global_ids, distance, t_ray
