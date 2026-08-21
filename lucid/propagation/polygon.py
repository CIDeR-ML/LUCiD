"""Polygonal-prism photon propagation — the faceted barrel of a real water-Cherenkov ID.

A real SK-like inner detector is not a round cylinder: the barrel is a regular
N-gon prism (SK: N = 38, from WCSim's ``WCBarrelNumPMTHorizontal/WCPMTperCellHorizontal``
= 152/4). Panel ``k`` spans azimuth [k*dphi, (k+1)*dphi] with dphi = 2*pi/N, so panel
normals sit at (k+1/2)*dphi and polygon vertices at k*dphi.

Two design choices keep this as fast to compile as the cylinder:

* The intersection is the convex-polytope slab method — a single vectorised reduction
  over a face axis. No Python loop over faces, no ``lax.switch``/``cond`` per face, so
  the compiled program grows by a constant rather than by N. That matters because the
  propagation ``lax.scan`` body in ``lucid/simulation/simulator.py`` calls this once per
  step: the body compiles once regardless of K.
* The wall grid is the cylinder's wall grid relabelled. Wall columns are
  ``col = panel * n_u + u_idx`` with ``n_angular = N * n_u``, so cells never straddle a
  panel edge and every downstream index decoder (``create_inverted_sensor_map``, the
  linear-index packing, the modular neighbour wraparound) is shared with the cylinder
  unchanged. Caps keep the n_cap x n_cap square lattice, spanning the circumradius so
  the polygon corners are covered.

Unlike the cylinder there is no ``sqrt`` here: the grazing-ray discriminant floor
documented at the top of ``cylinder.py`` has no analogue, so the second derivative stays
finite. The trade is a C1 kink where the exit face switches (t itself is continuous —
both planes agree on the shared edge).
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from .cylinder import create_inverted_sensor_map          # index layout is shared

LARGE = 1e10
_EPS = 1e-12


def panel_geometry(n_sides, apothem):
    """Static per-panel quantities: outward normals, in-panel tangents, panel width.

    Returns
    -------
    nx, ny : ndarray, shape (n_sides,)
        Outward normal of each panel, at azimuth (k + 1/2) * dphi.
    width : float
        Panel width, 2 * apothem * tan(dphi/2).
    """
    dphi = 2.0 * np.pi / n_sides
    ang = (np.arange(n_sides) + 0.5) * dphi
    return np.cos(ang), np.sin(ang), 2.0 * apothem * np.tan(dphi / 2.0)


def circumradius(n_sides, apothem):
    """Vertex distance of the N-gon whose face-to-axis distance is ``apothem``."""
    return apothem / np.cos(np.pi / n_sides)


@partial(jax.jit, static_argnums=(4, 5, 6, 7))
def intersect_polygon_with_grid(ray_origin, ray_direction, apothem, h,
                                n_sides, n_u, n_height, n_cap):
    """Exit point of one ray from the prism, plus its grid-cell coordinates.

    The prism is the convex intersection of ``n_sides`` side half-spaces
    ``n_k . x <= apothem`` and the two caps ``|z| <= h/2``. For a ray starting inside,
    the exit is ``min`` over the faces it is approaching (``d . n > 0``) of the plane
    distance — and the face attaining that minimum *is* the exit face, so no separate
    in-face bounds test is needed.

    Returns
    -------
    tuple
        (intersects, t, is_wall, is_top_cap, wall_indices, cap_indices,
         intersection_point, panel) — matching the cylinder's tuple plus the exit
        panel index, which ``calculate_polygon_normals`` needs.
    """
    dphi = 2.0 * jnp.pi / n_sides
    ang = (jnp.arange(n_sides) + 0.5) * dphi
    nx, ny = jnp.cos(ang), jnp.sin(ang)                       # (n_sides,)

    # ── side planes: one vectorised reduction over the face axis ──
    den = ray_direction[0] * nx + ray_direction[1] * ny
    num = apothem - (ray_origin[0] * nx + ray_origin[1] * ny)
    approaching = den > _EPS
    # SAFE denominator: keep the unselected lanes finite so the reverse pass sees no
    # inf/nan (jnp.where alone would still differentiate through num/0).
    t_side_all = jnp.where(approaching, num / jnp.where(approaching, den, 1.0), LARGE)
    panel = jnp.argmin(t_side_all)
    t_side = t_side_all[panel]

    # ── caps ──
    dz = ray_direction[2]
    up, down = dz > _EPS, dz < -_EPS
    t_top = jnp.where(up, (h / 2.0 - ray_origin[2]) / jnp.where(up, dz, 1.0), LARGE)
    t_bot = jnp.where(down, (-h / 2.0 - ray_origin[2]) / jnp.where(down, dz, 1.0), LARGE)

    ts = jnp.stack([t_side, t_top, t_bot])
    part = jnp.argmin(ts)
    t = jnp.min(ts)
    intersects = t < LARGE
    p = ray_origin + t * ray_direction

    # ── wall cell: (panel, in-panel lateral, height) packed as (col, row) ──
    ang_p = (panel + 0.5) * dphi
    u = -p[0] * jnp.sin(ang_p) + p[1] * jnp.cos(ang_p)        # lateral coord on the panel
    width = 2.0 * apothem * jnp.tan(dphi / 2.0)
    u_idx = jnp.clip(jnp.floor((u + width / 2.0) / width * n_u).astype(jnp.int32),
                     0, n_u - 1)
    col = panel * n_u + u_idx
    row = jnp.clip(jnp.floor((p[2] + h / 2.0) / h * n_height).astype(jnp.int32),
                   0, n_height - 1)
    wall_indices = jnp.array([col, row])

    # ── cap cell: square lattice spanning the circumradius (corners included) ──
    rad = apothem / jnp.cos(dphi / 2.0)
    cap_x = jnp.clip(jnp.floor((p[0] + rad) / (2.0 * rad) * n_cap).astype(jnp.int32),
                     0, n_cap - 1)
    cap_y = jnp.clip(jnp.floor((p[1] + rad) / (2.0 * rad) * n_cap).astype(jnp.int32),
                     0, n_cap - 1)
    cap_indices = jnp.array([cap_x, cap_y])

    return (intersects, t, part == 0, part == 1,
            wall_indices, cap_indices, p, panel)


batch_intersect_polygon_with_grid = jax.vmap(
    intersect_polygon_with_grid, in_axes=(0, 0, None, None, None, None, None, None))


def calculate_polygon_normals(is_wall, is_top_cap, panel, n_sides):
    """Outward normals at the exit points. O(1) — a gather on the exit panel index."""
    dphi = 2.0 * jnp.pi / n_sides
    ang = (panel + 0.5) * dphi
    wall_normals = jnp.stack([jnp.cos(ang), jnp.sin(ang), jnp.zeros_like(ang)], axis=-1)
    return jnp.where(is_wall[:, None],
                     wall_normals,
                     jnp.where(is_top_cap[:, None],
                               jnp.array([0.0, 0.0, 1.0]),
                               jnp.array([0.0, 0.0, -1.0])))


def polygon_bounds_check(points, apothem, h, n_sides):
    """Inside test for the prism. O(1), no loop over faces.

    For a convex regular N-gon the binding half-space is the one whose angular sector
    contains the point, so a single dot product against that panel's normal suffices.
    """
    dphi = 2.0 * jnp.pi / n_sides
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    phi = jnp.arctan2(y, x) % (2.0 * jnp.pi)
    ang = (jnp.floor(phi / dphi) + 0.5) * dphi
    perp = x * jnp.cos(ang) + y * jnp.sin(ang)
    return (perp <= apothem) & (z >= -h / 2.0) & (z <= h / 2.0)


@partial(jax.jit, static_argnums=(4, 5, 6, 7))
def assign_sensors_to_polygon_grid(sensors, sensor_radius, apothem, h,
                                   n_sides, n_u, n_height, n_cap,
                                   pmt_offset=0.0, footprint=None):
    """Sensor -> up to 4 overlapping grid cells, in the cylinder's (i, j, k) encoding.

    ``k`` is 0 for wall, 1 for top cap, 2 for bottom cap; for the wall ``i`` is the
    packed column ``panel * n_u + u_idx``, so the modular wraparound in ``i`` moves
    correctly across a panel edge into the neighbouring panel.

    Two distinct roles that a single radius used to serve:

    ``footprint`` is how far the sensor spreads ALONG the surface, so it sets which
    neighbouring cells a sensor also occupies. That is the photocathode aperture, not the
    sphere's curvature radius — the latter over-covers and inflates candidates per cell.
    Defaults to ``sensor_radius`` so a centred sphere behaves as before.

    ``pmt_offset`` is how far the sensor centre sits OUTSIDE the wall, which the
    on-surface test must be centred on. A photocathode modelled as an offset spherical cap
    has its centre of curvature behind the wall by construction, so testing
    ``|perp - apothem| <= radius`` would be measuring against the wrong surface.
    """
    if footprint is None:
        footprint = sensor_radius
    dphi = 2.0 * jnp.pi / n_sides
    n_angular = n_sides * n_u
    width = 2.0 * apothem * jnp.tan(dphi / 2.0)
    rad = apothem / jnp.cos(dphi / 2.0)

    def assign_single_sensor(sensor):
        x, y, z = sensor
        phi = jnp.arctan2(y, x) % (2.0 * jnp.pi)
        panel = jnp.floor(phi / dphi).astype(jnp.int32)
        ang_p = (panel + 0.5) * dphi
        perp = x * jnp.cos(ang_p) + y * jnp.sin(ang_p)

        # Centred on wall+pmt_offset, which is where the sphere centre belongs: a real
        # photocathode is a spherical cap whose centre of curvature sits behind the
        # blacksheet. Tolerance is the footprint, so an npz a few cm off still assigns.
        on_wall = jnp.abs(perp - apothem - pmt_offset) <= footprint
        on_top = z > h / 2.0 + pmt_offset - footprint
        on_bottom = z < -h / 2.0 - pmt_offset + footprint

        def assign_wall():
            u = -x * jnp.sin(ang_p) + y * jnp.cos(ang_p)
            u_scaled = (u + width / 2.0) / width * n_u
            u_idx = jnp.clip(jnp.floor(u_scaled).astype(jnp.int32), 0, n_u - 1)
            col = panel * n_u + u_idx
            v_scaled = (jnp.clip(z, -h / 2.0, h / 2.0) + h / 2.0) / h * n_height
            row = jnp.clip(jnp.floor(v_scaled).astype(jnp.int32), 0, n_height - 1)

            cell_u = width / n_u
            cell_v = h / n_height
            u_frac = u_scaled % 1
            v_frac = v_scaled % 1
            include_right = u_frac >= 1 - footprint / cell_u
            include_left = u_frac <= footprint / cell_u
            include_top = v_frac >= 1 - footprint / cell_v
            include_bottom = v_frac <= footprint / cell_v

            row_up = jnp.clip(row + 1, 0, n_height - 1)
            row_down = jnp.clip(row - 1, 0, n_height - 1)

            indices = jnp.array([
                [col, row, 0],
                [(col + 1) % n_angular, row, 0],
                [col, row_up, 0],
                [(col + 1) % n_angular, row_up, 0],
                [col, row_down, 0],
                [(col + 1) % n_angular, row_down, 0],
                [(col - 1) % n_angular, row, 0],
                [(col - 1) % n_angular, row_up, 0],
                [(col - 1) % n_angular, row_down, 0],
            ])
            selection = jnp.array([
                1.0,
                include_right,
                include_top,
                include_right * include_top,
                include_bottom,
                include_right * include_bottom,
                include_left,
                include_left * include_top,
                include_left * include_bottom,
            ])
            sorted_indices = indices[jnp.argsort(-selection)]
            return jnp.where(jnp.arange(4)[:, None] < jnp.sum(selection),
                             sorted_indices[:4], -1)

        def assign_cap(is_top):
            x_scaled = (x + rad) / (2.0 * rad) * n_cap
            y_scaled = (y + rad) / (2.0 * rad) * n_cap
            x_idx = jnp.clip(jnp.floor(x_scaled).astype(jnp.int32), 0, n_cap - 1)
            y_idx = jnp.clip(jnp.floor(y_scaled).astype(jnp.int32), 0, n_cap - 1)

            cell = 2.0 * rad / n_cap
            x_frac = x_scaled % 1
            y_frac = y_scaled % 1
            include_right = x_frac >= 1 - footprint / cell
            include_left = x_frac <= footprint / cell
            include_top = y_frac >= 1 - footprint / cell
            include_bottom = y_frac <= footprint / cell

            x_right = jnp.clip(x_idx + 1, 0, n_cap - 1)
            x_left = jnp.clip(x_idx - 1, 0, n_cap - 1)
            y_up = jnp.clip(y_idx + 1, 0, n_cap - 1)
            y_down = jnp.clip(y_idx - 1, 0, n_cap - 1)
            face = 1 if is_top else 2

            indices = jnp.array([
                [x_idx, y_idx, face],
                [x_right, y_idx, face],
                [x_idx, y_up, face],
                [x_right, y_up, face],
                [x_idx, y_down, face],
                [x_right, y_down, face],
                [x_left, y_idx, face],
                [x_left, y_up, face],
                [x_left, y_down, face],
            ])
            selection = jnp.array([
                1.0,
                include_right,
                include_top,
                include_right * include_top,
                include_bottom,
                include_right * include_bottom,
                include_left,
                include_left * include_top,
                include_left * include_bottom,
            ])
            sorted_indices = indices[jnp.argsort(-selection)]
            return jnp.where(jnp.arange(4)[:, None] < jnp.sum(selection),
                             sorted_indices[:4], -1)

        return jax.lax.cond(
            on_wall,
            assign_wall,
            lambda: jax.lax.cond(
                on_top,
                lambda: assign_cap(True),
                lambda: jax.lax.cond(
                    on_bottom,
                    lambda: assign_cap(False),
                    lambda: jnp.full((4, 3), -1, dtype=jnp.int32),
                ),
            ),
        )

    return jax.vmap(assign_single_sensor)(sensors)


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def calculate_polygon_grid_centers(apothem, h, n_sides, n_u, n_height, n_cap):
    """Centres of every grid cell, in the shared linear order.

    Wall cells come first as ``col * n_height + row``, then the top cap, then the
    bottom cap — the same packing ``create_inverted_sensor_map`` decodes.
    """
    dphi = 2.0 * jnp.pi / n_sides
    ang = (jnp.arange(n_sides) + 0.5) * dphi                  # (n_sides,)
    width = 2.0 * apothem * jnp.tan(dphi / 2.0)
    u_c = -width / 2.0 + (jnp.arange(n_u) + 0.5) * (width / n_u)   # (n_u,)

    # (n_sides, n_u) -> flattened to the packed column index
    wx = (apothem * jnp.cos(ang)[:, None] - u_c[None, :] * jnp.sin(ang)[:, None]).reshape(-1)
    wy = (apothem * jnp.sin(ang)[:, None] + u_c[None, :] * jnp.cos(ang)[:, None]).reshape(-1)
    z_c = (jnp.arange(n_height) - n_height / 2.0 + 0.5) * (h / n_height)

    wall_centers = jnp.stack([
        jnp.repeat(wx, n_height),
        jnp.repeat(wy, n_height),
        jnp.tile(z_c, n_sides * n_u),
    ], axis=1)

    rad = apothem / jnp.cos(dphi / 2.0)
    cap_pos = (jnp.arange(n_cap) - n_cap / 2.0 + 0.5) * (2.0 * rad / n_cap)
    xg, yg = jnp.meshgrid(cap_pos, cap_pos, indexing='ij')
    flat_x, flat_y = xg.reshape(-1), yg.reshape(-1)
    cap_centers = jnp.concatenate([
        jnp.stack([flat_x, flat_y, jnp.full(n_cap * n_cap, h / 2.0)], axis=1),
        jnp.stack([flat_x, flat_y, jnp.full(n_cap * n_cap, -h / 2.0)], axis=1),
    ], axis=0)

    return jnp.concatenate([wall_centers, cap_centers], axis=0)
