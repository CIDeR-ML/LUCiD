"""
2D spatial hash for string positions.

Bins string-telescope strings into a regular xy grid so that the DDA
traversal kernel can efficiently retrieve candidate strings per cell.

Each string is placed in every cell whose area overlaps the string's
xy footprint (the line segment from anchor.xy to top.xy) expanded by
a halo of (sensor_radius + soft_kernel_pad + string_curvature).

Output is a (n_cells_x * n_cells_y, max_strings_per_cell) int32 array
with -1 sentinel for empty slots — the same pattern as the existing
inverted_sensor_map in lucid/propagation/shared.py.
"""

import numpy as np


def build_string_hash(
    string_anchors,
    string_tops,
    cell_size,
    r_filter,
    max_strings_per_cell,
    *,
    pad_cells=1,
):
    """Build the 2D spatial hash for string positions.

    Parameters
    ----------
    string_anchors : (N_str, 3)    bottom DOM positions (or anchor points)
    string_tops : (N_str, 3)       top DOM positions
    cell_size : float              xy grid cell edge length (m)
    r_filter : float               halo radius: sensor_radius + soft pad + curvature
    max_strings_per_cell : int     upper bound on strings per cell
    pad_cells : int                extra cells around the footprint bounding box

    Returns
    -------
    cell_map : (n_cells_x * n_cells_y, max_strings_per_cell) int32
        Per-cell string indices, -1 for empty slots.
    grid_origin : (2,) float64
        (x_min, y_min) of the grid's lower-left corner.
    grid_shape : (2,) int
        (n_cells_x, n_cells_y).
    cell_size : float
        Cell edge length (echoed back for convenience).
    stats : dict
        'max_occupancy', 'mean_occupancy', 'overflow_count',
        'total_cells', 'string_coverage'.
    """
    n_str = len(string_anchors)
    if n_str == 0:
        raise ValueError("No strings provided")

    anchors_xy = np.asarray(string_anchors[:, :2], dtype=np.float64)
    tops_xy = np.asarray(string_tops[:, :2], dtype=np.float64)

    # Grid bounds: bounding box of all string xy positions, plus halo + padding
    all_xy = np.concatenate([anchors_xy, tops_xy], axis=0)
    xy_min = all_xy.min(axis=0) - r_filter - pad_cells * cell_size
    xy_max = all_xy.max(axis=0) + r_filter + pad_cells * cell_size

    n_cells = np.ceil((xy_max - xy_min) / cell_size).astype(int)
    n_cells = np.maximum(n_cells, 1)
    n_cells_x, n_cells_y = int(n_cells[0]), int(n_cells[1])
    total_cells = n_cells_x * n_cells_y

    grid_origin = xy_min.copy()

    cell_map = np.full((total_cells, max_strings_per_cell), -1, dtype=np.int32)
    occupancy = np.zeros(total_cells, dtype=np.int32)
    overflow_count = 0

    for i in range(n_str):
        ax, ay = anchors_xy[i]
        tx, ty = tops_xy[i]

        # Bounding box of string footprint + halo
        x_lo = min(ax, tx) - r_filter
        x_hi = max(ax, tx) + r_filter
        y_lo = min(ay, ty) - r_filter
        y_hi = max(ay, ty) + r_filter

        # Cell index range
        ci_lo = max(0, int(np.floor((x_lo - grid_origin[0]) / cell_size)))
        ci_hi = min(n_cells_x - 1, int(np.floor((x_hi - grid_origin[0]) / cell_size)))
        cj_lo = max(0, int(np.floor((y_lo - grid_origin[1]) / cell_size)))
        cj_hi = min(n_cells_y - 1, int(np.floor((y_hi - grid_origin[1]) / cell_size)))

        for ci in range(ci_lo, ci_hi + 1):
            for cj in range(cj_lo, cj_hi + 1):
                cell_idx = ci * n_cells_y + cj
                slot = occupancy[cell_idx]
                if slot < max_strings_per_cell:
                    cell_map[cell_idx, slot] = i
                    occupancy[cell_idx] = slot + 1
                else:
                    overflow_count += 1

    # Coverage: fraction of strings reachable from at least one cell
    reachable = set()
    for row in cell_map:
        for v in row:
            if v >= 0:
                reachable.add(int(v))
    coverage = len(reachable) / n_str if n_str > 0 else 1.0

    stats = {
        'max_occupancy': int(occupancy.max()) if total_cells > 0 else 0,
        'mean_occupancy': float(occupancy.mean()) if total_cells > 0 else 0.0,
        'overflow_count': overflow_count,
        'total_cells': total_cells,
        'string_coverage': coverage,
    }

    if overflow_count > 0:
        import warnings
        warnings.warn(
            f"Spatial hash overflow: {overflow_count} insertions dropped. "
            f"Increase max_strings_per_cell (current: {max_strings_per_cell}). "
            f"Max occupancy: {stats['max_occupancy']}."
        )

    if coverage < 1.0:
        missing = set(range(n_str)) - reachable
        import warnings
        warnings.warn(
            f"String coverage: {coverage:.2%}. Missing strings: {sorted(missing)}"
        )

    return cell_map, grid_origin, np.array([n_cells_x, n_cells_y]), cell_size, stats
