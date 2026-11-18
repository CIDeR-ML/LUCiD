"""Brute-force views of the surface propagator's sensor map, for the sensor-map tests.

The map has two halves. The FORWARD half, ``detector.assign_sensor_to_cells``, lists for each
sensor up to 9 grid cells its disk overlaps. The INVERSE half, built inside ``create_propagator``,
holds ``max_candidates_per_ray`` sensors per cell: the forward assignments first, then the
nearest sensors to the cell centre. When more than that many sensors are forward-assigned to one
cell, the excess cannot fit, whatever order they are taken in, so the number dropped is
``sum(max(0, count - K))`` over cells. These helpers count it from the forward half alone.
"""
import warnings

import jax.numpy as jnp
import numpy as np


def forward_cell_counts(det, grid):
    """Number of distinct sensors forward-assigned to each linear cell of ``grid`` (a
    ``det.make_grid`` record), and the number of sensors assigned to no cell at all."""
    fwd = np.asarray(det.assign_sensor_to_cells(
        jnp.asarray(det.all_points), float(det.S_radius), grid))
    total = det.total_grid_cells(grid)
    counts = np.zeros(total, dtype=int)
    orphans = 0
    for s in range(fwd.shape[0]):
        cells = {int(det.point_to_grid_cell_from_coords(c, grid))
                 for c in fwd[s] if not np.all(c == -1)}
        cells = {c for c in cells if 0 <= c < total}
        orphans += not cells
        for c in cells:
            counts[c] += 1
    return counts, orphans


def sensor_map_warnings(build):
    """Call ``build()`` and return the text of every ``Sensor map:`` warning it emitted."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = build()
    return [str(x.message) for x in w if 'Sensor map' in str(x.message)], result
