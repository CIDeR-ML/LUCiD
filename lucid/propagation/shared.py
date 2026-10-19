"""The surface photon propagator: one factory for any Detector subclass.

Geometry enters only through the Detector methods called below (configure_grid,
assign_sensor_to_cells, grid_cell_centers, build_inverted_sensor_map, bounds_check,
intersect_ray, point_to_grid_cell, compute_normal); everything else here is shape-agnostic.

Not the only propagator: `lucid/propagation/string/` traverses a per-DOM volume rather than a
surface grid, and carries its own deposit implementation. The two have diverged: that one clamps
the closest-approach parameter where this one gates on it, and has no first-hit survival product.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from lucid.propagation.base import (
    compute_sensor_intersections_base,
    process_intersection_normals,
    find_closest_sensors,
)
from lucid.overlap import create_overlap_prob
from lucid.propagation.sk_pmt_sensor import (
    compute_sk20inch_sensor_intersections,
)
from lucid.propagation.sk_pmt import GEANT_PMT_RADIUS
from lucid.propagation.sk_acrylic import (
    SK_ACRYLIC_OUTER_RADIUS,
    SK_ACRYLIC_THICKNESS,
    SKAcrylicReflectionTransport,
)


def _build_sk_wall_cell_assignment(detector):
    """Build SK-IV's logical boundary-cell -> active PMT assignment.

    This mirrors ``SGHOLEBST``: 150 x 51 barrel cells and 48 x 48 cap
    grids with 70.7 cm pitch.  Empty cap cells remain -1 instead of being
    silently redirected to the nearest active tube.
    """
    required = ('pmt_id', 'surfaces', 'all_points', 'r', 'H')
    if any(not hasattr(detector, name) for name in required):
        raise ValueError(
            "SK wall-cell assignment requires measured SK PMT metadata")

    points = np.asarray(detector.all_points, dtype=float)
    surfaces = np.asarray(detector.surfaces)
    pmt_ids = np.asarray(detector.pmt_id, dtype=int)
    n_phi, n_z = 150, 51
    n_cap, cap_pitch = 48, 0.707
    barrel = np.full((n_phi, n_z), -1, dtype=np.int32)
    top = np.full((n_cap, n_cap), -1, dtype=np.int32)
    bottom = np.full((n_cap, n_cap), -1, dtype=np.int32)

    for sensor_index, (point, surface, cable) in enumerate(
            zip(points, surfaces, pmt_ids)):
        if surface == 'barrel':
            # SK barrel cable numbering is exactly iphi*51 + iz (1-based).
            iphi = (int(cable) - 1) // n_z
            iz = (int(cable) - 1) % n_z
            if 0 <= iphi < n_phi:
                barrel[iphi, iz] = sensor_index
        else:
            ix = int(np.floor(point[0] / cap_pitch)) + n_cap // 2
            iy = int(np.floor(point[1] / cap_pitch)) + n_cap // 2
            if 0 <= ix < n_cap and 0 <= iy < n_cap:
                (top if surface == 'top' else bottom)[ix, iy] = sensor_index

    if np.count_nonzero(barrel >= 0) != n_phi * n_z:
        raise ValueError("SK barrel logical map is incomplete")

    barrel = jnp.asarray(barrel)
    top = jnp.asarray(top)
    bottom = jnp.asarray(bottom)
    # SGHOLEBST divides the original ID height (the PMT-centre planes), not
    # the inward-shifted blacksheet height.  Measured cap positions retain
    # that original plane even when the cylinder boundary is moved inward.
    cap_z = np.abs(points[surfaces != 'barrel', 2])
    if cap_z.size == 0:
        raise ValueError("SK wall-cell assignment requires cap PMTs")
    height = 2.0 * float(np.median(cap_z))

    def assign(intersection_point, surface_info):
        is_wall, is_top = surface_info
        x, y, z = (intersection_point[:, i] for i in range(3))
        phi = jnp.mod(jnp.arctan2(y, x), 2.0 * jnp.pi)
        iphi = jnp.floor(phi / (2.0 * jnp.pi / n_phi) + 0.5).astype(jnp.int32)
        iphi = jnp.mod(iphi, n_phi)
        # Fortran IFIX is floor here because the offset keeps the expression
        # positive throughout the inner detector.
        iz_one_based = jnp.floor(z / (height / n_z) + 26.0 + 0.5).astype(jnp.int32)
        iz = jnp.clip(iz_one_based - 1, 0, n_z - 1)
        barrel_sensor = barrel[iphi, iz]

        ix = jnp.floor(x / cap_pitch).astype(jnp.int32) + n_cap // 2
        iy = jnp.floor(y / cap_pitch).astype(jnp.int32) + n_cap // 2
        cap_valid = (ix >= 0) & (ix < n_cap) & (iy >= 0) & (iy < n_cap)
        ix_safe = jnp.clip(ix, 0, n_cap - 1)
        iy_safe = jnp.clip(iy, 0, n_cap - 1)
        cap_sensor = jnp.where(is_top, top[ix_safe, iy_safe],
                               bottom[ix_safe, iy_safe])
        cap_sensor = jnp.where(cap_valid, cap_sensor, -1)
        return jnp.where(is_wall, barrel_sensor, cap_sensor)

    return assign


def _build_sk_seam_candidate_assignment(detector):
    """Return the cap PMT that can protrude across a barrel-first ray.

    Near either barrel/end-cap seam, the nominal cylinder is sometimes the
    first blacksheet plane even though an end-cap PMT bulb is reached before
    that plane. GEANT follows the physical PMT volumes and therefore calls
    ``SGPMT`` for the cap tube first. The ordinary SK wall-cell lookup cannot
    express that case because it commits to the cylinder cell up front.

    For each barrel-first ray, inspect the local 5 x 5 cap-cell neighbourhood
    and return the active cap-cell centre closest to the travelled ray
    segment. The caller still evaluates the exact sphere+torus surface and
    compares its distance with the wall PMT; this function only makes the
    discrete candidate choice. All indices are stop-gradient geometry data,
    while the selected PMT intersection remains differentiable with respect
    to the ray state.
    """
    required = ('pmt_id', 'surfaces', 'all_points')
    if any(not hasattr(detector, name) for name in required):
        raise ValueError(
            "SK seam assignment requires measured SK PMT metadata")

    points = np.asarray(detector.all_points, dtype=float)
    surfaces = np.asarray(detector.surfaces)
    n_cap, cap_pitch = 48, 0.707
    top = np.full((n_cap, n_cap), -1, dtype=np.int32)
    bottom = np.full((n_cap, n_cap), -1, dtype=np.int32)
    for sensor_index, (point, surface) in enumerate(zip(points, surfaces)):
        if surface == 'barrel':
            continue
        ix = int(np.floor(point[0] / cap_pitch)) + n_cap // 2
        iy = int(np.floor(point[1] / cap_pitch)) + n_cap // 2
        if 0 <= ix < n_cap and 0 <= iy < n_cap:
            (top if surface == 'top' else bottom)[ix, iy] = sensor_index

    cap_z = np.abs(points[surfaces != 'barrel', 2])
    if cap_z.size == 0:
        raise ValueError("SK seam assignment requires cap PMTs")
    cap_plane = float(np.median(cap_z))
    top = jnp.asarray(top)
    bottom = jnp.asarray(bottom)
    sensor_points = jnp.asarray(points)
    offsets = np.asarray([
        (dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)
    ], dtype=np.int32)
    offset_x = jnp.asarray(offsets[:, 0])
    offset_y = jnp.asarray(offsets[:, 1])

    def assign(origins, directions, intersection_point, t_geometry,
               surface_info):
        is_wall, _ = surface_info
        unit_direction = directions / (
            jnp.linalg.norm(directions, axis=1, keepdims=True) + 1e-12)
        choose_top = intersection_point[:, 2] >= 0.0
        target_z = jnp.where(choose_top, cap_plane, -cap_plane)
        dz = unit_direction[:, 2]
        safe_dz = jnp.where(jnp.abs(dz) > 1e-8, dz, 1.0)
        t_plane = (target_z - origins[:, 2]) / safe_dz
        t_projection = jnp.clip(t_plane, 0.0, t_geometry)
        projected = origins + t_projection[:, None] * unit_direction
        base_x = (jnp.floor(projected[:, 0] / cap_pitch).astype(jnp.int32)
                  + n_cap // 2)
        base_y = (jnp.floor(projected[:, 1] / cap_pitch).astype(jnp.int32)
                  + n_cap // 2)
        ix = base_x[:, None] + offset_x[None, :]
        iy = base_y[:, None] + offset_y[None, :]
        valid_cell = ((ix >= 0) & (ix < n_cap)
                      & (iy >= 0) & (iy < n_cap))
        ix_safe = jnp.clip(ix, 0, n_cap - 1)
        iy_safe = jnp.clip(iy, 0, n_cap - 1)
        cap_map = jnp.where(choose_top[:, None],
                            top[ix_safe, iy_safe],
                            bottom[ix_safe, iy_safe])
        cap_map = jnp.where(valid_cell, cap_map, -1)

        safe_sensor = jnp.maximum(cap_map, 0)
        centers = sensor_points[safe_sensor]
        center_delta = centers - origins[:, None, :]
        t_closest = jnp.sum(
            center_delta * unit_direction[:, None, :], axis=2)
        closest = (origins[:, None, :]
                   + jnp.clip(t_closest, 1e-5,
                              t_geometry[:, None])[:, :, None]
                   * unit_direction[:, None, :])
        distance2 = jnp.sum((closest - centers) ** 2, axis=2)
        # The complete sphere+torus photocathode lies inside SK's 25.4 cm
        # GEANT PMT sphere.  Rejecting ray segments that miss this bounding
        # sphere is therefore exact, and prevents the later expensive torus
        # solve from treating every barrel ray as a possible seam crossing.
        valid_segment = (
            (cap_map >= 0)
            & (t_closest > 1e-5)
            & (t_closest <= t_geometry[:, None])
            & (distance2 <= (GEANT_PMT_RADIUS + 1e-5) ** 2)
        )
        distance2 = jnp.where(valid_segment, distance2, jnp.inf)
        nearest_slot = jnp.argmin(distance2, axis=1)
        candidate = jnp.take_along_axis(
            cap_map, nearest_slot[:, None], axis=1)[:, 0]
        has_candidate = jnp.any(valid_segment, axis=1)
        return jnp.where(is_wall & has_candidate, candidate, -1)

    return assign


def _build_sk_seam_prefilter(detector):
    """Return a cheap conservative test for rays that can reach a cap seam.

    The detailed assignment searches nearby cap cells because a nearly
    tangential ray can reach a protruding end-cap PMT before its nominal
    barrel crossing.  Most rays never enter the bounding volume of any such
    tube.  This test encloses every relevant 25.4 cm GEANT PMT sphere in a
    cap slab, allowing the detailed lookup to run on a compact subset without
    rejecting a possible physical intersection.  The whole cap is retained:
    after a reflection, a ray can start beside a central cap PMT even when its
    next nominal cylinder crossing is on the barrel.
    """
    points = np.asarray(detector.all_points, dtype=float)
    surfaces = np.asarray(detector.surfaces)
    cap_points = points[surfaces != 'barrel']
    if cap_points.size == 0:
        raise ValueError("SK seam prefilter requires cap PMTs")
    cap_plane = float(np.median(np.abs(cap_points[:, 2])))
    slab_half_width = GEANT_PMT_RADIUS + 1e-5

    def possible(origins, directions, intersection_point, t_geometry,
                 surface_info):
        is_wall, _ = surface_info
        unit_direction = directions / (
            jnp.linalg.norm(directions, axis=1, keepdims=True) + 1e-12)
        target_z = jnp.where(
            intersection_point[:, 2] >= 0.0, cap_plane, -cap_plane)
        lower_z = target_z - slab_half_width
        upper_z = target_z + slab_half_width
        dz = unit_direction[:, 2]
        parallel = jnp.abs(dz) < 1e-8
        safe_dz = jnp.where(parallel, 1.0, dz)
        distance_a = (lower_z - origins[:, 2]) / safe_dz
        distance_b = (upper_z - origins[:, 2]) / safe_dz
        enter = jnp.maximum(0.0, jnp.minimum(distance_a, distance_b))
        leave = jnp.minimum(
            t_geometry, jnp.maximum(distance_a, distance_b))
        parallel_inside = (
            (origins[:, 2] >= lower_z) & (origins[:, 2] <= upper_z))
        enter = jnp.where(parallel, 0.0, enter)
        leave = jnp.where(
            parallel, jnp.where(parallel_inside, t_geometry, -1.0), leave)

        return is_wall & (leave >= enter)

    return possible


def first_hit_survival(weights, times):
    """Cap the deposit at one photon: P(hit s_i) = p_i * prod_{j before i} (1 - p_j).

    `overlap_prob` is applied INDEPENDENTLY per candidate, so on its own it leaves the total uncapped:
    at grazing incidence a ray skimming the wall passes within r of a whole row of sensors and each
    takes full weight. Physically the photon
    deposits on the FIRST sensor it reaches, so with candidates ordered by arrival time and p_i the
    conditional hit probability given the photon gets there, sum(P) = 1 - prod(1 - p_j) <= 1.
    It splits weight smoothly between adjacent sensors (p1, p2*(1-p1)) and reduces to p_1 when
    only one candidate is in range. Done in log space; gated-out candidates have p = 0 and
    contribute exactly 1 to the product.

    TIES ARE BROKEN BY SLOT. Two live candidates can arrive at EXACTLY the same time -- a ray
    equidistant from two sensors -- and with a strict `t_j < t_i` neither counts as earlier, both
    keep their full p, and the sum can exceed 1. The slot index orders them, so the cap holds by construction rather than almost always.

    THE ORDERING IS A HARD COMPARISON. The product is smooth in the p's, but which p's multiply
    which is decided by a threshold on arrival time, so when two candidates swap order the
    per-sensor split jumps while the total stays put. Aggregate checks cannot see it; a
    per-sensor gradient can.

    Parameters
    ----------
    weights : (C, N) per-candidate overlap probabilities
    times : (C, N) per-candidate arrival times

    Returns
    -------
    (C, N) capped weights, summing to at most 1 over C for every photon.
    """
    # Clipped below 1 so log1p(-p) stays finite -- STRAIGHT-THROUGH, so the clip shapes only the
    # forward value (bit-identical to a plain clip). Must not be a plain jnp.clip for the
    # gradient: in step mode (temperature=None) the forward overlap is EXACTLY 1 inside a
    # sphere and 0 outside, where a plain clip has zero derivative (above its maximum) or half
    # (at a tie with its minimum), which would remove the straight-through surrogate gradient
    # the hard step exists to keep.
    p = weights + jax.lax.stop_gradient(jnp.clip(weights, 0.0, 1.0 - 1e-6) - weights)
    slot = jnp.arange(p.shape[0])
    # before[i, j, n]: candidate j reaches photon n's path before candidate i does
    before = ((times[None, :, :] < times[:, None, :])
              | ((times[None, :, :] == times[:, None, :])
                 & (slot[None, :, None] < slot[:, None, None])))
    log_survive = jnp.sum(before * jnp.log1p(-p)[None, :, :], axis=1)
    return p * jnp.exp(log_survive)


def validate_sensor_map(assignments_geometric, inverted_sensor_map, num_sensors,
                        detector, max_candidates_per_ray):
    """Check consistency between forward (sensor→cells) and inverse (cell→sensors) maps.

    Runs at propagator build time (numpy, not JIT). Raises warnings for
    any issues that could silently degrade simulation quality.

    Checks:
    1. Index bounds — no out-of-range sensor IDs
    2. Cell coverage — fraction of cells with at least one sensor
    3. Sensor visibility — are all sensors reachable through the map?
    4. Overcrowding — cells where geometric assignments exceed max_candidates_per_ray
    5. Forward-inverse consistency — geometric assignments present in inverse map
    """
    inv = np.asarray(inverted_sensor_map)
    fwd = np.asarray(assignments_geometric)
    total_cells, slots = inv.shape
    valid_mask = inv != -1

    # --- 1. Index bounds ---
    if valid_mask.any():
        min_idx, max_idx = int(inv[valid_mask].min()), int(inv[valid_mask].max())
        if min_idx < 0 or max_idx >= num_sensors:
            warnings.warn(
                f"Sensor map: out-of-range indices [{min_idx}, {max_idx}] "
                f"for {num_sensors} sensors")

    # --- 2. Cell coverage ---
    cells_with_sensors = int(np.any(valid_mask, axis=1).sum())
    coverage_pct = 100.0 * cells_with_sensors / total_cells if total_cells > 0 else 0
    if coverage_pct < 90.0:
        warnings.warn(
            f"Sensor map: low cell coverage — {cells_with_sensors}/{total_cells} "
            f"({coverage_pct:.1f}%) cells have sensors. Photons hitting empty "
            f"cells will produce zero weights.")

    # --- 3. Sensor visibility ---
    sensors_in_map = set(int(x) for x in inv[valid_mask])
    missing_sensors = set(range(num_sensors)) - sensors_in_map
    if missing_sensors:
        warnings.warn(
            f"Sensor map: {len(missing_sensors)}/{num_sensors} sensors do not "
            f"appear in any cell's inverse map. These sensors can never be hit.")

    # --- 4. Overcrowding (geometric assignments exceed max_candidates_per_ray) ---
    # Count how many geometric assignments each cell receives
    cell_geo_count = np.zeros(total_cells, dtype=int)
    for sensor_id in range(fwd.shape[0]):
        for slot in range(fwd.shape[1]):
            coords = fwd[sensor_id, slot]
            if np.all(coords == -1):
                continue
            linear_idx = int(detector.point_to_grid_cell_from_coords(coords))
            if 0 <= linear_idx < total_cells:
                cell_geo_count[linear_idx] += 1

    max_geo = int(cell_geo_count.max()) if total_cells > 0 else 0
    if max_geo > max_candidates_per_ray:
        warnings.warn(
            f"Sensor map: max geometric sensor assignments per cell ({max_geo}) "
            f"exceeds max_candidates_per_ray={max_candidates_per_ray}. "
            f"Auto-adjusting to {max_geo}.")

    # --- 5. Forward-inverse consistency ---
    n_missing = 0
    n_checked = 0
    for sensor_id in range(fwd.shape[0]):
        for slot in range(fwd.shape[1]):
            coords = fwd[sensor_id, slot]
            if np.all(coords == -1):
                continue
            linear_idx = int(detector.point_to_grid_cell_from_coords(coords))
            if linear_idx < 0 or linear_idx >= total_cells:
                continue
            n_checked += 1
            if sensor_id not in inv[linear_idx]:
                n_missing += 1

    if n_missing > 0:
        warnings.warn(
            f"Sensor map: {n_missing}/{n_checked} geometric assignments are "
            f"missing from the inverse map — likely dropped due to "
            f"max_candidates_per_ray={max_candidates_per_ray} overflow.")


def create_propagator(detector, sensor_positions, sensor_radius,
                      temperature=0.2, max_candidates_per_ray=4,
                      overlap_st_width_frac=0.35, overlap_renorm=1.0,
                      overlap_mode='interp',
                      sensor_shape='sphere', sk_pmt_lookup=None,
                      sk_pmt_lookup_options=None,
                      sensor_candidate_selection='all',
                      sensor_wall_assignment='nearest',
                      pmt_gap_recovery=False,
                      pmt_gap_recovery_hole_radius=0.285,
                      sk_acrylic_optics=None,
                      deposit_leg_bound=False,
                      **grid_params):
    """Build a JIT-compiled photon propagator using detector methods.

    Parameters
    ----------
    detector : Detector
        Detector instance with Phase 9 methods implemented.
    sensor_positions : jnp.ndarray, shape (n_sensors, 3)
    sensor_radius : float
    temperature : float
        Soft-assignment temperature for overlap probability.
    overlap_st_width_frac : float
        Straight-through surrogate width (fraction of r) for the hard-step
        overlap. Backward-gradient only; default 0.35.
    overlap_renorm : float
        Soft-overlap renormalization constant C (default 1.0 = OFF).
    overlap_mode : str
        Soft-overlap lookup interpolation: 'interp' (default) or 'cubic'.
    sensor_shape : {'sphere', 'sk20inch'}
        Candidate-sensor geometry. ``'sphere'`` preserves the legacy model.
        ``'sk20inch'`` uses the measured PMT axes and SKDONUTS bulb shape.
    sk_pmt_lookup : SKPMTCoverageLookup or None
        Optional prebuilt smooth-coverage table, primarily for tests or reuse.
    sk_pmt_lookup_options : dict or None
        Options forwarded to ``build_sk20inch_coverage_lookup`` when a finite
        temperature requires a table.
    sensor_candidate_selection : {'all', 'wall'}
        ``'all'`` evaluates every PMT associated with the wall grid cell.
        ``'wall'`` assigns the ray to the candidate PMT whose center is closest
        to the nominal wall crossing, then applies the exact PMT intersection
        only to that assignment in the forward pass.  With finite temperature,
        gradients retain the smooth all-candidate coverage surrogate.
    sensor_wall_assignment : {'nearest', 'sk'}
        ``'sk'`` uses the logical SK-IV wall-cell tables instead of choosing
        the nearest active candidate.
    pmt_gap_recovery : bool
        Recover full-sphere PMT intersections behind the blacksheet plane.
    pmt_gap_recovery_hole_radius : float
        Radius of the inner water-gap region eligible for that recovery.
    deposit_leg_bound : bool
        Bound the deposit to the leg the photon actually travels, [0, t_geometry], instead of
        weighting by distance from the unbounded ray LINE. Default False; a Python bool
        resolved at trace time, so when off the leg-bound arithmetic is absent from the graph
        rather than present and unused, and the result is bit-identical to the behaviour without it.

        The line does not stop at the wall, so for a ray at incidence theta it passes within a
        sensor radius of sensors displaced along the wall from the landing point, over-counting
        hits by (1 - cos theta)/2 per ray. Normal incidence is unaffected. Switching it on
        changes the forward model, and with it any energy scale calibrated without it.
    max_candidates_per_ray : int
    **grid_params
        Geometry-specific grid parameters passed to ``detector.configure_grid()``.
        Cylinder: n_cap, n_angular, n_height.
        Sphere: n_divisions.
        Box: n_x, n_y, n_z.

    Returns
    -------
    callable
        JIT-compiled ``propagate_photons(origins, directions) -> dict``
    """
    sensor_positions = jnp.array(sensor_positions)
    num_sensors = len(sensor_positions)
    if sensor_shape not in ('sphere', 'sk20inch'):
        raise ValueError("sensor_shape must be 'sphere' or 'sk20inch'")
    if sensor_candidate_selection not in ('all', 'wall'):
        raise ValueError("sensor_candidate_selection must be 'all' or 'wall'")
    if sensor_candidate_selection == 'wall' and sensor_shape != 'sk20inch':
        raise ValueError(
            "sensor_candidate_selection='wall' requires "
            "sensor_shape='sk20inch'")
    if sensor_wall_assignment not in ('nearest', 'sk'):
        raise ValueError("sensor_wall_assignment must be 'nearest' or 'sk'")
    if sensor_wall_assignment == 'sk' and sensor_candidate_selection != 'wall':
        raise ValueError(
            "sensor_wall_assignment='sk' requires sensor_candidate_selection='wall'")
    if sensor_wall_assignment == 'sk' and temperature is not None:
        raise ValueError(
            "exact SK wall assignment currently requires temperature=None")
    if pmt_gap_recovery:
        if sensor_shape != 'sk20inch' or sensor_candidate_selection != 'wall':
            raise ValueError(
                "PMT gap recovery requires SK20-inch wall routing")
        if pmt_gap_recovery_hole_radius <= 0:
            raise ValueError("pmt_gap_recovery_hole_radius must be positive")
    if sk_acrylic_optics is not None:
        if sensor_shape != 'sk20inch' or sensor_candidate_selection != 'wall':
            raise ValueError(
                "SK acrylic transport requires SK20-inch wall routing")

    sensor_directions = None
    sensor_is_barrel = None
    coverage_lookup = None
    if sensor_shape == 'sk20inch':
        if not hasattr(detector, 'pmt_directions') or not hasattr(detector, 'surfaces'):
            raise ValueError(
                "sensor_shape='sk20inch' requires measured pmt_directions "
                "and per-PMT surfaces")
        directions_array = np.asarray(detector.pmt_directions)
        surfaces_array = np.asarray(detector.surfaces)
        sensor_directions = jnp.asarray(directions_array)
        sensor_is_barrel = jnp.asarray(surfaces_array == 'barrel')
        if sensor_directions.shape != sensor_positions.shape:
            raise ValueError(
                "pmt_directions shape must match sensor_positions")
        if surfaces_array.shape != (num_sensors,):
            raise ValueError("surfaces shape must match sensor_positions")
        unknown_surfaces = set(np.unique(surfaces_array)) - {
            'barrel', 'top', 'bottom'}
        if unknown_surfaces:
            raise ValueError(
                f"unknown PMT surface labels: {sorted(unknown_surfaces)}")
        if (not np.all(np.isfinite(directions_array))
                or np.any(np.linalg.norm(directions_array, axis=1) == 0.0)):
            raise ValueError("pmt_directions must be finite and non-zero")
        if temperature is not None:
            expected_sigma = float(temperature) * float(sensor_radius)
            if sk_pmt_lookup is None:
                from .sk_pmt_lookup import build_sk20inch_coverage_lookup
                lookup_options = dict(sk_pmt_lookup_options or {})
                if 'sigma' in lookup_options:
                    raise ValueError(
                        "sk_pmt_lookup_options must not set sigma; it is "
                        "temperature * sensor_radius")
                coverage_lookup = build_sk20inch_coverage_lookup(
                    expected_sigma, **lookup_options)
            else:
                if not np.isclose(
                        float(sk_pmt_lookup.sigma), expected_sigma,
                        rtol=1e-6, atol=1e-9):
                    raise ValueError(
                        "sk_pmt_lookup sigma does not match "
                        "temperature * sensor_radius")
                coverage_lookup = sk_pmt_lookup

    exact_wall_assignment = (
        _build_sk_wall_cell_assignment(detector)
        if sensor_wall_assignment == 'sk' else None)
    exact_seam_assignment = (
        _build_sk_seam_candidate_assignment(detector)
        if sensor_wall_assignment == 'sk' else None)
    exact_seam_prefilter = (
        _build_sk_seam_prefilter(detector)
        if sensor_wall_assignment == 'sk' else None)

    # Configure grid on detector — caller passes geometry-specific params.
    # max_candidates_per_ray is always forwarded so auto-derivation can
    # ensure no cell exceeds this limit.
    grid_params.setdefault('max_candidates_per_ray', max_candidates_per_ray)
    detector.configure_grid(**grid_params)

    # 1. Geometric sensor-to-cell assignments
    # Acrylic domes extend beyond the 25.4 cm PMT envelope.  Include that
    # larger footprint in the wall-cell candidate map so an oblique ray can
    # encounter a neighbouring cover before it reaches its logical
    # blacksheet cell, as it does in GEANT.
    assignment_radius = (
        max(float(sensor_radius), SK_ACRYLIC_OUTER_RADIUS)
        if sk_acrylic_optics is not None else sensor_radius)
    assignments_geometric = detector.assign_sensor_to_cells(
        sensor_positions, assignment_radius)

    # 2. Grid cell centers
    grid_centers = detector.grid_cell_centers()

    # 3. Distance-based fallback assignments (shared)
    assignments_distance = find_closest_sensors(
        grid_centers, sensor_positions, max_candidates_per_ray)

    # 4. Build inverted sensor map (geometry-specific decoder)
    inverted_sensor_map = detector.build_inverted_sensor_map(
        assignments_geometric, assignments_distance,
        max_candidates_per_ray, num_sensors)

    # 4b. Validate the sensor map
    validate_sensor_map(assignments_geometric, inverted_sensor_map,
                        num_sensors, detector, max_candidates_per_ray)

    # 5. Overlap probability (shared)
    # temperature=None → step function (hard assignment, non-differentiable)
    # temperature=float → Gaussian kernel with sigma = temperature * sensor_radius
    if sensor_shape == 'sphere':
        if temperature is None:
            overlap_prob = create_overlap_prob(
                None, sensor_radius,
                st_width_frac=overlap_st_width_frac, renorm=overlap_renorm, mode=overlap_mode)
        else:
            overlap_prob = create_overlap_prob(
                temperature * sensor_radius, sensor_radius,
                st_width_frac=overlap_st_width_frac, renorm=overlap_renorm, mode=overlap_mode)
    else:
        overlap_prob = None

    # 6. Bounds check closure
    def bounds_check(positions):
        return detector.bounds_check(positions)

    # 7. JIT-compiled propagation function
    @jax.jit
    def propagate_photons(photon_origins, photon_directions):
        """Trace photon rays through detector geometry.

        Parameters
        ----------
        photon_origins : jnp.ndarray, shape (n_rays, 3) or (3,)
        photon_directions : jnp.ndarray, shape (n_rays, 3) or (3,)

        Returns
        -------
        dict with keys: sensor_weights, sensor_indices, times, positions,
             normals, inside_sensor, per_sensor_positions, sensor_normals
        """
        single_ray = photon_origins.ndim == 1
        if single_ray:
            photon_origins = photon_origins[None, :]
            photon_directions = photon_directions[None, :]

        # a. Ray-geometry intersection
        intersection_point, t_geometry, grid_info, surface_info = \
            detector.intersect_ray(photon_origins, photon_directions)

        # b. Map to grid cell indices
        idx = detector.point_to_grid_cell(grid_info)

        # c. Look up candidate sensors (stop_gradient: geometry is static)
        potential_sensors = jax.lax.stop_gradient(inverted_sensor_map[idx])

        # d. Compute sensor intersections (shared, vmapped over sensor slots)
        def compute_for_slot(
                slot_sensors,
                origins=photon_origins,
                directions=photon_directions):
            if sensor_shape == 'sk20inch':
                return compute_sk20inch_sensor_intersections(
                    slot_sensors,
                    sensor_positions,
                    sensor_directions,
                    sensor_is_barrel,
                    origins,
                    directions,
                    coverage_lookup,
                    acrylic_optics=sk_acrylic_optics,
                    return_active=True,
                    return_incident_direction=True,
                    return_cover_intersection=True,
                    return_cover_transport=True,
                )
            result = compute_sensor_intersections_base(
                slot_sensors, sensor_positions, sensor_radius,
                origins, directions,
                bounds_check, overlap_prob,
                # None when off drops the leg-bound code from the trace; see
                # ``deposit_leg_bound``.
                t_geometry=t_geometry if deposit_leg_bound else None)
            normalized_directions = directions / (
                jnp.linalg.norm(directions, axis=1, keepdims=True) + 1e-12)
            return result + (
                result[4], normalized_directions,
                jnp.zeros_like(result[4], dtype=bool),
                origins,
                jnp.zeros_like(result[0]),
                jnp.ones_like(result[0]),
                SKAcrylicReflectionTransport(
                    outer_position=origins,
                    outer_direction=normalized_directions,
                    outer_time_distance=jnp.zeros_like(result[0]),
                    outer_probability=jnp.zeros_like(result[0]),
                    inner_exit_valid=jnp.zeros_like(result[4], dtype=bool),
                    inner_exit_position=origins,
                    inner_exit_direction=normalized_directions,
                    inner_exit_time_distance=jnp.zeros_like(result[0]),
                    inner_exit_probability=jnp.zeros_like(result[0]),
                ))

        if sensor_candidate_selection == 'wall':
            # SKDetSim first routes a boundary crossing to one wall PMT cell,
            # then asks whether that PMT's curved photocathode was intersected.
            # Approximate that cell routing with the candidate center nearest to
            # the nominal detector-wall crossing.  The stop-gradient selection
            # gives the exact discrete forward result while preserving the
            # smooth all-candidate coverage derivative used for calibration.
            if exact_wall_assignment is None:
                valid_candidates = potential_sensors >= 0
                safe_candidates = jnp.maximum(potential_sensors, 0)
                candidate_positions = sensor_positions[safe_candidates]
                distance2 = jnp.sum(
                    (candidate_positions - intersection_point[:, None, :]) ** 2,
                    axis=-1,
                )
                distance2 = jnp.where(valid_candidates, distance2, jnp.inf)
                wall_slot = jnp.argmin(distance2, axis=1)
                has_candidate = jnp.any(valid_candidates, axis=1)
            else:
                wall_sensor_exact = exact_wall_assignment(
                    intersection_point, surface_info)
                has_candidate = wall_sensor_exact >= 0

        if sensor_candidate_selection == 'wall' and coverage_lookup is None:
            # Hard forward simulation needs only the PMT selected at the wall.
            # Tracing every grid candidate and masking seven of eight results
            # afterwards made the exact SK surface unnecessarily expensive.
            # Keep a length-one sensor axis so the rest of propagation retains
            # its usual (candidate, ray, ...) array convention.
            wall_sensor = (wall_sensor_exact if exact_wall_assignment is not None
                           else jnp.take_along_axis(
                               potential_sensors, wall_slot[:, None], axis=1)[:, 0])
            wall_sensor = jnp.where(has_candidate, wall_sensor, -1)
            if exact_seam_assignment is None:
                selected = compute_for_slot(wall_sensor)
                (weights, sensor_times, sensor_indices,
                 sensor_normals_all, inside_sensor,
                 sensor_hit_positions, active_sensor,
                 sensor_incident_directions,
                 cover_intersections, cover_positions,
                 cover_time_distances, cover_transmissions,
                 cover_reflections) = jax.tree.map(
                    lambda value: value[None, ...], selected)
            else:
                # A cap bulb may protrude into a nominally barrel-first path.
                # Only a thin seam region can reach the cap candidate. Compact
                # those rays before both the 25-cell logical lookup and the
                # expensive sphere+torus solve. If an unusual event exceeds
                # the buffer, fall back to the dense exact calculation; the
                # optimization therefore cannot truncate or approximate hits.
                wall_traced = compute_for_slot(wall_sensor)
                n_rays = photon_origins.shape[0]
                assignment_capacity = max(1, (7 * n_rays + 15) // 16)
                trace_capacity = max(1, (n_rays + 15) // 16)
                seam_possible = exact_seam_prefilter(
                    photon_origins, photon_directions, intersection_point,
                    t_geometry, surface_info)
                possible_count = jnp.sum(seam_possible, dtype=jnp.int32)

                def sparse_seam_assignment(_):
                    scatter_index = jnp.nonzero(
                        seam_possible,
                        size=assignment_capacity,
                        fill_value=n_rays,
                    )[0]
                    gather_index = jnp.minimum(scatter_index, n_rays - 1)
                    compact_surface_info = jax.tree.map(
                        lambda value: value[gather_index], surface_info)
                    compact = exact_seam_assignment(
                        photon_origins[gather_index],
                        photon_directions[gather_index],
                        intersection_point[gather_index],
                        t_geometry[gather_index],
                        compact_surface_info,
                    )
                    return jnp.full(
                        (n_rays,), -1, dtype=compact.dtype).at[
                            scatter_index].set(compact, mode='drop')

                seam_sensor_exact = jax.lax.cond(
                    possible_count <= assignment_capacity,
                    sparse_seam_assignment,
                    lambda _: exact_seam_assignment(
                        photon_origins, photon_directions,
                        intersection_point, t_geometry, surface_info),
                    operand=None,
                )
                seam_mask = seam_sensor_exact >= 0
                seam_count = jnp.sum(seam_mask, dtype=jnp.int32)

                def sparse_seam_trace(_):
                    scatter_index = jnp.nonzero(
                        seam_mask,
                        size=trace_capacity,
                        fill_value=n_rays,
                    )[0]
                    gather_index = jnp.minimum(scatter_index, n_rays - 1)
                    compact = compute_for_slot(
                        seam_sensor_exact[gather_index],
                        photon_origins[gather_index],
                        photon_directions[gather_index],
                    )

                    defaults = (
                        jnp.zeros((n_rays,), dtype=compact[0].dtype),
                        jnp.full((n_rays, 1), jnp.inf,
                                 dtype=compact[1].dtype),
                        seam_sensor_exact,
                        jnp.zeros((n_rays, 3), dtype=compact[3].dtype),
                        jnp.zeros((n_rays,), dtype=bool),
                        jnp.zeros((n_rays, 3), dtype=compact[5].dtype),
                        jnp.zeros((n_rays,), dtype=bool),
                        photon_directions,
                        jnp.zeros((n_rays,), dtype=bool),
                        photon_origins,
                        jnp.zeros((n_rays,), dtype=compact[0].dtype),
                        jnp.ones((n_rays,), dtype=compact[0].dtype),
                        SKAcrylicReflectionTransport(
                            outer_position=photon_origins,
                            outer_direction=photon_directions,
                            outer_time_distance=jnp.zeros(
                                (n_rays,), dtype=compact[0].dtype),
                            outer_probability=jnp.zeros(
                                (n_rays,), dtype=compact[0].dtype),
                            inner_exit_valid=jnp.zeros(
                                (n_rays,), dtype=bool),
                            inner_exit_position=photon_origins,
                            inner_exit_direction=photon_directions,
                            inner_exit_time_distance=jnp.zeros(
                                (n_rays,), dtype=compact[0].dtype),
                            inner_exit_probability=jnp.zeros(
                                (n_rays,), dtype=compact[0].dtype),
                        ),
                    )

                    def scatter(default, values):
                        return default.at[scatter_index].set(
                            values, mode='drop')

                    return jax.tree.map(scatter, defaults, compact)

                seam_traced = jax.lax.cond(
                    seam_count <= trace_capacity,
                    sparse_seam_trace,
                    lambda _: compute_for_slot(seam_sensor_exact),
                    operand=None,
                )
                seam_before_wall = (
                    jnp.squeeze(seam_traced[1], axis=-1)
                    <= (t_geometry + 1e-5)
                )
                seam_physical = seam_traced[4] & seam_before_wall
                wall_physical = wall_traced[4]
                wall_time = jnp.squeeze(wall_traced[1], axis=-1)
                seam_time = jnp.squeeze(seam_traced[1], axis=-1)
                choose_seam = seam_physical & (
                    ~wall_physical | (seam_time < wall_time))

                # If both PMTs miss, retain the ordinary wall result so hole
                # and reflection handling remain exactly as before.
                def choose(wall_value, seam_value):
                    mask = choose_seam.reshape(
                        choose_seam.shape
                        + (1,) * (wall_value.ndim - 1))
                    return jnp.where(mask, seam_value, wall_value)[None, ...]

                (weights, sensor_times, sensor_indices,
                 sensor_normals_all, inside_sensor,
                 sensor_hit_positions, active_sensor,
                 sensor_incident_directions,
                 cover_intersections, cover_positions,
                 cover_time_distances, cover_transmissions,
                 cover_reflections) = jax.tree.map(
                    choose, wall_traced, seam_traced)

            if sk_acrylic_optics is not None:
                # GEANT transports through the first physical acrylic volume
                # crossed; it does not first choose a blacksheet hole.  Trace
                # the small wall-cell candidate set and replace the logical
                # wall result with the earliest cover that lies at the wall.
                # The discrete argmin is piecewise constant, while derivatives
                # of the selected Snell/Fresnel path remain fully visible to
                # JAX in both forward and reverse mode.
                cover_traced = jax.vmap(
                    compute_for_slot, in_axes=1, out_axes=0)(potential_sensors)
                candidate_cover = cover_traced[8]
                candidate_cover_distance = jnp.sqrt(jnp.sum(
                    (cover_traced[9] - photon_origins[None, :, :]) ** 2,
                    axis=-1) + 1e-12)
                # cover_positions are on the inner acrylic surface.  Allow
                # the 13 mm shell thickness beyond the ideal blacksheet plane
                # when deciding whether its outer surface was encountered
                # first.
                candidate_cover = candidate_cover & (
                    candidate_cover_distance
                    <= t_geometry[None, :] + SK_ACRYLIC_THICKNESS + 1e-5)
                ordered_distance = jnp.where(
                    candidate_cover, candidate_cover_distance, jnp.inf)
                first_cover_slot = jnp.argmin(ordered_distance, axis=0)
                has_first_cover = jnp.any(candidate_cover, axis=0)

                def take_first_cover(value):
                    index = first_cover_slot.reshape(
                        (1, first_cover_slot.shape[0])
                        + (1,) * (value.ndim - 2))
                    index = jnp.broadcast_to(
                        index, (1,) + value.shape[1:])
                    return jnp.take_along_axis(value, index, axis=0)[0]

                first_cover = jax.tree.map(take_first_cover, cover_traced)

                def choose_first_cover(wall_value, cover_value):
                    mask = has_first_cover.reshape(
                        has_first_cover.shape
                        + (1,) * (cover_value.ndim - 1))
                    return jnp.where(mask, cover_value, wall_value[0])[None, ...]

                (weights, sensor_times, sensor_indices,
                 sensor_normals_all, inside_sensor,
                 sensor_hit_positions, active_sensor,
                 sensor_incident_directions,
                 cover_intersections, cover_positions,
                 cover_time_distances, cover_transmissions,
                 cover_reflections) = jax.tree.map(
                    choose_first_cover,
                    (weights, sensor_times, sensor_indices,
                     sensor_normals_all, inside_sensor,
                     sensor_hit_positions, active_sensor,
                     sensor_incident_directions,
                     cover_intersections, cover_positions,
                     cover_time_distances, cover_transmissions,
                     cover_reflections),
                    first_cover)

            if pmt_gap_recovery:
                # SGGAPEDGE tests the continuation of a ray through the PMT
                # opening against the full 25.4 cm PMT sphere.  This recovers
                # rays whose sphere intersection lies behind the inward-shifted
                # blacksheet plane; all other hole misses terminate later via
                # the zero blacksheet-coverage factor.
                safe_wall_sensor = jnp.maximum(wall_sensor, 0)
                centers = sensor_positions[safe_wall_sensor]
                axes = sensor_directions[safe_wall_sensor]
                unit_dir = photon_directions / (
                    jnp.linalg.norm(photon_directions, axis=1, keepdims=True)
                    + 1e-12)
                wall_delta = intersection_point - centers
                tangent = wall_delta - (
                    jnp.sum(wall_delta * axes, axis=1, keepdims=True) * axes)
                hole_distance = jnp.sqrt(
                    jnp.sum(tangent * tangent, axis=1) + 1e-12)
                in_hole = hole_distance <= pmt_gap_recovery_hole_radius

                oc = intersection_point - centers
                b = jnp.sum(oc * unit_dir, axis=1)
                c = jnp.sum(oc * oc, axis=1) - sensor_radius ** 2
                disc = b * b - c
                sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0) + 1e-12)
                root_near = -b - sqrt_disc
                root_far = -b + sqrt_disc
                gap_distance = jnp.where(root_near > 1e-6, root_near, root_far)
                gap_hit = (
                    (wall_sensor >= 0) & (~inside_sensor[0]) & in_hole
                    & (disc >= 0.0) & (gap_distance > 1e-6)
                    & (gap_distance <= 0.5))
                if sk_acrylic_optics is not None:
                    # A ray that has just crossed the cover must first enter
                    # the water gap with its refracted direction.  Recovering
                    # the straight pre-cover ray here bypasses precisely the
                    # acrylic routing that SGGAPEDGE is meant to follow.
                    gap_hit = gap_hit & (~cover_intersections[0])
                gap_position = intersection_point + gap_distance[:, None] * unit_dir
                sphere_water_normal = (
                    gap_position - centers) / sensor_radius
                gap_transport_normal = -sphere_water_normal
                gap_time = (t_geometry + gap_distance)[:, None]

                weights = jnp.where(gap_hit[None, :], 1.0, weights)
                sensor_times = jnp.where(
                    gap_hit[None, :, None], gap_time[None, :, :], sensor_times)
                sensor_normals_all = jnp.where(
                    gap_hit[None, :, None],
                    gap_transport_normal[None, :, :], sensor_normals_all)
                sensor_hit_positions = jnp.where(
                    gap_hit[None, :, None],
                    gap_position[None, :, :], sensor_hit_positions)
                sensor_incident_directions = jnp.where(
                    gap_hit[None, :, None],
                    unit_dir[None, :, :], sensor_incident_directions)
                inside_sensor = inside_sensor | gap_hit[None, :]
                active_sensor = active_sensor | gap_hit[None, :]
        else:
            (weights, sensor_times, sensor_indices,
             sensor_normals_all, inside_sensor,
             sensor_hit_positions, active_sensor,
             sensor_incident_directions,
             cover_intersections, cover_positions,
             cover_time_distances, cover_transmissions,
             cover_reflections) = jax.vmap(
                compute_for_slot, in_axes=1, out_axes=0)(potential_sensors)

            if sensor_candidate_selection == 'wall':
                # Smooth calibration retains all-candidate lookup gradients;
                # only its exact forward value is replaced by wall assignment.
                wall_mask = (
                    jnp.arange(potential_sensors.shape[1])[:, None]
                    == wall_slot[None, :]
                ) & has_candidate[None, :]
                hard_weights = (
                    active_sensor & wall_mask).astype(weights.dtype)
                weights = weights + jax.lax.stop_gradient(
                    hard_weights - weights)
                inside_sensor = inside_sensor & wall_mask

        # First-hit semantics: the photon deposits on the first sensor it reaches. See
        # `first_hit_survival` for the cap, its tie-break, and what its hard ordering costs.
        # Wall routing is already one-hot in the forward pass: hard mode traces
        # one PMT, while smooth mode replaces all-candidate coverage with one
        # discrete wall assignment.  A second cap would change an exact unit hit
        # to 1 - 1e-6 and suppress its smooth backward geometry surrogate.
        if sensor_candidate_selection != 'wall':
            weights = first_hit_survival(
                weights,
                jnp.squeeze(sensor_times, -1)
                if sensor_times.ndim == 3 else sensor_times)

        # e. Compute geometry surface normals
        geometry_normals = detector.compute_normal(intersection_point, surface_info)

        # f. Process intersection normals (shared)
        final_results = process_intersection_normals(
            photon_origins, photon_directions, intersection_point,
            t_geometry, sensor_normals_all, sensor_hit_positions,
            inside_sensor, geometry_normals)

        hit_positions = final_results['positions']
        final_normals = final_results['normals']

        # g. Assemble result dict
        result = {
            'times': sensor_times,
            'sensor_weights': weights,
            'sensor_indices': sensor_indices,
            'per_sensor_positions': sensor_hit_positions,
            'positions': hit_positions,
            'normals': final_normals,
            'sensor_normals': sensor_normals_all,
            'sensor_directions': sensor_incident_directions,
            'acrylic_cover_intersections': cover_intersections,
            'acrylic_cover_positions': cover_positions,
            'acrylic_cover_time_distances': cover_time_distances,
            'acrylic_cover_transmissions': cover_transmissions,
            'acrylic_cover_reflections': cover_reflections,
            'inside_sensor': inside_sensor,
        }

        if single_ray:
            result = jax.tree.map(lambda x: x[0] if x.ndim > 0 else x, result)

        return result

    return propagate_photons
