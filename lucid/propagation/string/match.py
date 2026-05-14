"""
Per-segment string matcher (REFERENCE IMPLEMENTATION — superseded by fast.py).

Chains the DDA traversal, spatial-hash gather, distance filter, and DOM
snap into a single JIT'd function that produces candidate DOM IDs for
a batch of photon rays.

Pipeline per ray:
  1. DDA traverse 2D grid along ray's xy projection → cell indices
  2. Gather candidate string IDs from hash cells
  3. For each candidate string: distance test + z-range test + DOM snap
  4. Flatten into (max_dom_per_segment,) candidate DOM global IDs

The output feeds directly into compute_sensor_intersections_base.
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

from .dda import dda_traverse_2d
from .kernel import candidate_doms_for_ray_string


@partial(jax.jit, static_argnums=(12, 13, 14))
def match_segment(
    ray_origin,
    ray_direction,
    # -- grid data --
    grid_origin,
    grid_shape,
    cell_size,
    cell_map,
    # -- string data --
    string_anchors,
    string_axes,
    dom_s_offsets,
    dom_global_ids,
    string_s_min,
    string_s_max,
    # -- static shape constants --
    max_cells,
    max_strings_per_cell,
    n_dom_snap,
    # -- filter radius --
    r_filter,
):
    """Find candidate DOM IDs for one photon ray segment.

    Parameters
    ----------
    ray_origin : (3,)
    ray_direction : (3,)
    grid_origin : (2,)             hash grid lower-left xy
    grid_shape : (2,) int          (n_cx, n_cy)
    cell_size : float
    cell_map : (n_cells, max_strings_per_cell) int32
    string_anchors : (N_str, 3)
    string_axes : (N_str, 3)
    dom_s_offsets : (N_str, max_dom) float — sorted, +inf padded
    dom_global_ids : (N_str, max_dom) int32 — -1 padded
    string_s_min : (N_str,)
    string_s_max : (N_str,)
    max_cells : int                static
    max_strings_per_cell : int     static
    n_dom_snap : int               static
    r_filter : float

    Returns
    -------
    candidate_dom_ids : (max_dom_per_segment,) int32
        Global DOM IDs, -1 for empty/filtered.
        max_dom_per_segment = max_cells * max_strings_per_cell * n_dom_snap
    """
    # Step 1: DDA to get visited cell indices
    ray_xy = ray_origin[:2]
    dir_xy = ray_direction[:2]
    cell_indices, cell_mask = dda_traverse_2d(
        ray_xy, dir_xy, grid_origin, grid_shape, cell_size, max_cells
    )

    # Step 2: Gather candidate string IDs from visited cells
    # Shape: (max_cells, max_strings_per_cell)
    def gather_cell_strings(cell_idx):
        return jnp.where(
            cell_idx >= 0,
            cell_map[cell_idx],
            jnp.full(max_strings_per_cell, -1, dtype=jnp.int32),
        )

    candidate_strings = jax.vmap(gather_cell_strings)(cell_indices)
    # Mask out strings from invalid cells
    cell_mask_2d = cell_mask[:, None] & (candidate_strings >= 0)
    candidate_strings = jnp.where(cell_mask_2d, candidate_strings, -1)

    # Flatten to 1D: (max_cells * max_strings_per_cell,)
    flat_strings = candidate_strings.reshape(-1)

    # Step 3: For each candidate string, compute distance + snap
    def process_one_string(str_id):
        valid = str_id >= 0

        anchor = jnp.where(valid, string_anchors[str_id], jnp.zeros(3))
        axis = jnp.where(valid, string_axes[str_id], jnp.array([0.0, 0.0, 1.0]))
        s_offs = jnp.where(valid, dom_s_offsets[str_id], jnp.full(dom_s_offsets.shape[1], jnp.inf))
        g_ids = jnp.where(valid, dom_global_ids[str_id], jnp.full(dom_global_ids.shape[1], -1))
        s_min = jnp.where(valid, string_s_min[str_id], 0.0)
        s_max = jnp.where(valid, string_s_max[str_id], 0.0)

        dom_ids, dist, t_ray = candidate_doms_for_ray_string(
            ray_origin, ray_direction,
            anchor, axis, s_offs, g_ids,
            s_min, s_max, r_filter, n_dom_snap,
        )
        return jnp.where(valid, dom_ids, -1)

    # vmap over all candidate strings
    all_dom_ids = jax.vmap(process_one_string)(flat_strings)
    # Shape: (max_cells * max_strings_per_cell, n_dom_snap)

    # Flatten to 1D output
    return all_dom_ids.reshape(-1)


def match_segment_batch(
    ray_origins,
    ray_directions,
    grid_origin,
    grid_shape,
    cell_size,
    cell_map,
    string_anchors,
    string_axes,
    dom_s_offsets,
    dom_global_ids,
    string_s_min,
    string_s_max,
    max_cells,
    max_strings_per_cell,
    n_dom_snap,
    r_filter,
):
    """Vectorized match_segment over a batch of rays.

    Parameters
    ----------
    ray_origins : (n_rays, 3)
    ray_directions : (n_rays, 3)
    (all other args as in match_segment)

    Returns
    -------
    candidate_dom_ids : (n_rays, max_dom_per_segment) int32
    """
    return jax.vmap(
        match_segment,
        in_axes=(0, 0, None, None, None, None, None, None, None, None,
                 None, None, None, None, None, None),
    )(
        ray_origins, ray_directions,
        grid_origin, grid_shape, cell_size, cell_map,
        string_anchors, string_axes, dom_s_offsets, dom_global_ids,
        string_s_min, string_s_max,
        max_cells, max_strings_per_cell, n_dom_snap,
        r_filter,
    )
