"""The inverted sensor map: for each grid cell, the sensors a ray landing in it tests.

One builder for every surface geometry. A geometry supplies only its cell table -- the grid
coordinates of each linear cell index, in the convention of its forward assignments
(``assign_sensor_to_cells``) -- built from :func:`grid_face_cells`.
"""
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np


def grid_face_cells(n_i, n_j, *face):
    """The ``(i, j[, face])`` coordinates of an ``n_i`` x ``n_j`` face's cells, in linear order.

    Row ``i * n_j + j`` is cell ``(i, j)``, the order every geometry numbers a face's cells in;
    ``face`` (at most one value) appends the face id the forward assignments carry. The
    geometries concatenate their faces into the table :func:`build_inverted_sensor_map` reads.
    """
    ii, jj = np.meshgrid(np.arange(n_i), np.arange(n_j), indexing='ij')
    cols = [ii.reshape(-1), jj.reshape(-1)] + [np.full(n_i * n_j, k) for k in face]
    return np.stack(cols, axis=1).astype(np.int32)


@cache
def _build_inverted_sensor_map_on_cpu():
    """:func:`build_inverted_sensor_map` compiled onto the CPU, built once on first call.

    ``jax.devices()`` initialises a backend, which may not happen at import or in a decorator
    (G6, J3), so the compilation is deferred to the first call.

    Returns
    -------
    Callable
        The jitted ``_build_inverted_sensor_map``, pinned to ``jax.devices('cpu')[0]``.
    """
    return jax.jit(_build_inverted_sensor_map, static_argnums=(3,),
                   device=jax.devices('cpu')[0])


def build_inverted_sensor_map(assignments_geometric, assignments_distance, cell_coords,
                              max_candidates_per_ray):
    """Build the cell -> sensor table: geometric overlaps first, then the closest sensors.

    Runs on the CPU for every geometry; the table is integer indices, so where it runs does not
    change it.

    Parameters
    ----------
    assignments_geometric : (n_sensors, n_slots, n_coords) int
        Each sensor's overlapped cells as grid coordinates, rows of -1 unused (the
        geometry's ``assign_sensor_to_cells``).
    assignments_distance : (n_cells, n_closest) int
        Each cell's closest sensors (:func:`find_closest_sensors`).
    cell_coords : (n_cells, n_coords) int
        The grid coordinates of each linear cell index, in the forward assignments' convention
        (the geometry's cell table, built with :func:`grid_face_cells`).
    max_candidates_per_ray : int
        Slots per cell.

    Returns
    -------
    (n_cells, max_candidates_per_ray) int32, -1 in unfilled slots.
    """
    return _build_inverted_sensor_map_on_cpu()(
        assignments_geometric, assignments_distance, cell_coords, max_candidates_per_ray)


def _build_inverted_sensor_map(assignments_geometric, assignments_distance, cell_coords,
                               max_candidates_per_ray):
    """The traced body of :func:`build_inverted_sensor_map`; see it for the contract."""
    total_cells = cell_coords.shape[0]

    # Initialize map
    inverted_map = jnp.full((total_cells, max_candidates_per_ray), -1, dtype=jnp.int32)

    def update_cell(inv_map, i):
        cell = cell_coords[i]

        def add_geometric(carry, j):
            curr_map, curr_count = carry

            # Check if sensor j intersects with this cell: any of its cells has all its coordinates
            matches = jnp.all(assignments_geometric[j] == cell, axis=-1)

            cell_matches = jnp.any(matches)
            should_add = cell_matches & (curr_count < max_candidates_per_ray)

            new_map = jnp.where(
                should_add,
                curr_map.at[i, curr_count].set(j),
                curr_map
            )

            return (new_map, curr_count + should_add), None

        # Add geometric intersections
        (new_map, geom_count), _ = jax.lax.scan(
            add_geometric,
            (inv_map, 0),
            jnp.arange(len(assignments_geometric))
        )

        # Get closest sensors for this cell
        closest = assignments_distance[i]

        # Add closest sensors if there's room
        def add_closest(carry, j):
            curr_map, curr_count = carry
            sensor_idx = closest[j]

            # Check for duplicates
            def check_duplicate(k, is_dup):
                return is_dup | (curr_map[i, k] == sensor_idx)

            is_duplicate = jax.lax.fori_loop(
                0, curr_count,
                check_duplicate,
                False
            )

            # Add if not duplicate and have space
            should_add = (~is_duplicate) & (curr_count < max_candidates_per_ray)

            new_map = jnp.where(
                should_add,
                curr_map.at[i, curr_count].set(sensor_idx),
                curr_map
            )

            return (new_map, curr_count + should_add), None

        # Fill remaining slots with closest sensors
        (final_map, _), _ = jax.lax.scan(
            add_closest,
            (new_map, geom_count),
            jnp.arange(len(closest))
        )

        return final_map, None

    final_map, _ = jax.lax.scan(
        update_cell,
        inverted_map,
        jnp.arange(total_cells)
    )

    return final_map
