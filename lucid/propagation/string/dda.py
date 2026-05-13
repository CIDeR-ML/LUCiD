"""
2D Amanatides-Woo DDA (Digital Differential Analyzer) traversal kernel.

Given a 2D ray (origin_xy, direction_xy) and a regular grid, produces
a fixed-length list of cell indices traversed by the ray, up to a
maximum distance L_max. JIT-compatible via lax.fori_loop.

The DDA walks the ray from its 2D entry into the grid, stepping one
cell at a time along whichever axis has the nearest cell boundary,
until either max_cells steps are taken or the ray exits the grid.
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial


@partial(jax.jit, static_argnums=(5,))
def dda_traverse_2d(
    ray_origin_xy,
    ray_direction_xy,
    grid_origin,
    grid_shape,
    cell_size,
    max_cells,
):
    """Walk a 2D ray through a regular grid, returning visited cell indices.

    Parameters
    ----------
    ray_origin_xy : (2,)       ray origin in xy
    ray_direction_xy : (2,)    ray direction in xy (need not be unit)
    grid_origin : (2,)         (x_min, y_min) of grid lower-left
    grid_shape : (2,) int      (n_cells_x, n_cells_y)
    cell_size : float          cell edge length
    max_cells : int            compile-time static; max cells to visit

    Returns
    -------
    cell_indices : (max_cells,) int32     linear cell indices (-1 if unused)
    valid_mask : (max_cells,) bool        True for actually-visited cells
    """
    n_cx = grid_shape[0]
    n_cy = grid_shape[1]

    # Ray start in grid coordinates
    pos = (ray_origin_xy - grid_origin) / cell_size

    dx = ray_direction_xy[0]
    dy = ray_direction_xy[1]
    dir_len = jnp.sqrt(dx * dx + dy * dy + 1e-30)

    # Normalize direction for stepping
    inv_len = 1.0 / (dir_len + 1e-30)
    ndx = dx * inv_len
    ndy = dy * inv_len

    # Current cell
    ix = jnp.floor(pos[0]).astype(jnp.int32)
    iy = jnp.floor(pos[1]).astype(jnp.int32)

    # Step direction (+1 or -1)
    step_x = jnp.where(ndx >= 0, jnp.int32(1), jnp.int32(-1))
    step_y = jnp.where(ndy >= 0, jnp.int32(1), jnp.int32(-1))

    # Distance along ray to the next cell boundary in each axis
    # t_max_x: how far (in ray-parameter) to the next vertical cell boundary
    abs_ndx = jnp.abs(ndx) + 1e-30
    abs_ndy = jnp.abs(ndy) + 1e-30

    # Fractional position within current cell
    fx = pos[0] - jnp.floor(pos[0])
    fy = pos[1] - jnp.floor(pos[1])

    # Distance to next boundary
    t_max_x = jnp.where(ndx >= 0, (1.0 - fx) / abs_ndx, fx / abs_ndx)
    t_max_y = jnp.where(ndy >= 0, (1.0 - fy) / abs_ndy, fy / abs_ndy)

    # How far along the ray we move for one full cell in each axis
    t_delta_x = 1.0 / abs_ndx
    t_delta_y = 1.0 / abs_ndy

    # Handle near-zero direction components: set t_max and t_delta to huge
    # values so we never step in that direction
    is_dx_zero = jnp.abs(dx) < 1e-20
    is_dy_zero = jnp.abs(dy) < 1e-20
    t_max_x = jnp.where(is_dx_zero, 1e30, t_max_x)
    t_max_y = jnp.where(is_dy_zero, 1e30, t_max_y)
    t_delta_x = jnp.where(is_dx_zero, 1e30, t_delta_x)
    t_delta_y = jnp.where(is_dy_zero, 1e30, t_delta_y)

    def in_bounds(ix, iy):
        return (ix >= 0) & (ix < n_cx) & (iy >= 0) & (iy < n_cy)

    def linear_idx(ix, iy):
        return ix * n_cy + iy

    # Initial state
    init_valid = in_bounds(ix, iy)
    init_idx = jnp.where(init_valid, linear_idx(ix, iy), jnp.int32(-1))

    # State: (ix, iy, t_max_x, t_max_y, cell_indices, valid_mask, step_count)
    cell_indices_init = jnp.full(max_cells, -1, dtype=jnp.int32)
    valid_mask_init = jnp.zeros(max_cells, dtype=jnp.bool_)

    # Record first cell
    cell_indices_init = cell_indices_init.at[0].set(init_idx)
    valid_mask_init = valid_mask_init.at[0].set(init_valid)

    init_state = (ix, iy, t_max_x, t_max_y, cell_indices_init, valid_mask_init)

    def body(i, state):
        ix, iy, tmx, tmy, cells, mask = state
        step_idx = i + 1  # slot 0 is the starting cell

        # Step to next cell
        go_x = tmx <= tmy
        new_ix = jnp.where(go_x, ix + step_x, ix)
        new_iy = jnp.where(go_x, iy, iy + step_y)
        new_tmx = jnp.where(go_x, tmx + t_delta_x, tmx)
        new_tmy = jnp.where(go_x, tmy, tmy + t_delta_y)

        valid = in_bounds(new_ix, new_iy)
        cidx = jnp.where(valid, linear_idx(new_ix, new_iy), jnp.int32(-1))

        cells = cells.at[step_idx].set(cidx)
        mask = mask.at[step_idx].set(valid)

        return (new_ix, new_iy, new_tmx, new_tmy, cells, mask)

    final_state = lax.fori_loop(0, max_cells - 1, body, init_state)
    _, _, _, _, cell_indices, valid_mask = final_state

    return cell_indices, valid_mask
