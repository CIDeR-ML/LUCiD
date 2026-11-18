"""The surface photon propagator: one factory for any Detector subclass.

Geometry enters only through the Detector methods called below (make_grid,
assign_sensor_to_cells, grid_cell_centers, build_inverted_sensor_map, bounds_check,
intersect_ray, point_to_grid_cell, compute_normal), each given the frozen grid record make_grid
returns; everything else here is shape-agnostic.

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


def max_assignments_per_cell(assignments_geometric, detector, grid):
    """The most sensors the geometric map assigns to any one grid cell (numpy, build time).

    A cell's inverse-map row must hold at least this many slots, or sensors are dropped from it
    and rays crossing that cell can never reach them.
    """
    fwd = np.asarray(assignments_geometric)
    counts = {}
    # A sensor counts once per distinct cell: neighbour indices clipped at a grid edge repeat
    # its own cell.
    for sensor_cells in fwd:
        cells = {int(detector.point_to_grid_cell_from_coords(c, grid))
                 for c in sensor_cells if not np.all(c == -1)}
        for cell in cells:
            counts[cell] = counts.get(cell, 0) + 1
    return max(counts.values(), default=0)


def validate_sensor_map(assignments_geometric, inverted_sensor_map, num_sensors,
                        detector, grid, max_candidates_per_ray):
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
    # create_propagator sizes the rows to fit, so this fires only for a map built elsewhere.
    max_geo = max_assignments_per_cell(assignments_geometric, detector, grid)
    if max_geo > max_candidates_per_ray:
        warnings.warn(
            f"Sensor map: max geometric sensor assignments per cell ({max_geo}) "
            f"exceeds max_candidates_per_ray={max_candidates_per_ray}; sensors are dropped.")

    # --- 5. Forward-inverse consistency ---
    n_missing = 0
    n_checked = 0
    for sensor_id in range(fwd.shape[0]):
        for slot in range(fwd.shape[1]):
            coords = fwd[sensor_id, slot]
            if np.all(coords == -1):
                continue
            linear_idx = int(detector.point_to_grid_cell_from_coords(coords, grid))
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
    max_candidates_per_ray : int
        Sensor slots per grid cell (the candidates a ray tests). Raised, with a warning, to the
        most sensors geometry assigns to one cell, so no sensor is dropped from a cell it overlaps.
    **grid_params
        Geometry-specific grid parameters passed to ``detector.make_grid()``.
        Cylinder: n_cap, n_angular, n_height.
        Sphere: n_divisions.
        Box: n_x, n_y, n_z.

    Returns
    -------
    propagate_photons : callable
        JIT-compiled ``propagate_photons(origins, directions) -> dict``
    grid : NamedTuple
        The frozen grid record the propagator closes over (``CylinderGrid``, ``SphereGrid`` or
        ``BoxGrid``); the detector itself is left unchanged.
    """
    sensor_positions = jnp.array(sensor_positions)
    num_sensors = len(sensor_positions)

    # The grid, sized for max_candidates_per_ray unless the caller states the cell counts
    # (a caller's own max_candidates_per_ray in grid_params sizes it instead).
    grid_params.setdefault('max_candidates_per_ray', max_candidates_per_ray)
    grid = detector.make_grid(**grid_params)

    # 1. Geometric sensor-to-cell assignments
    assignments_geometric = detector.assign_sensor_to_cells(sensor_positions, sensor_radius, grid)

    # 1b. Every cell row must hold all the sensors geometry puts in that cell; a smaller
    # max_candidates_per_ray would drop sensors from it (they could never be hit through it).
    needed = max_assignments_per_cell(assignments_geometric, detector, grid)
    if needed > max_candidates_per_ray:
        warnings.warn(f"Sensor map: a grid cell holds {needed} sensors; raising "
                      f"max_candidates_per_ray from {max_candidates_per_ray} to {needed}.")
        max_candidates_per_ray = needed

    # 2. Grid cell centers
    grid_centers = detector.grid_cell_centers(grid)

    # 3. Distance-based fallback assignments (shared)
    assignments_distance = find_closest_sensors(
        grid_centers, sensor_positions, max_candidates_per_ray)

    # 4. Build inverted sensor map (lucid.propagation.sensor_map, from the geometry's cell table)
    inverted_sensor_map = detector.build_inverted_sensor_map(
        assignments_geometric, assignments_distance, max_candidates_per_ray, grid)

    # 4b. Validate the sensor map
    validate_sensor_map(assignments_geometric, inverted_sensor_map,
                        num_sensors, detector, grid, max_candidates_per_ray)

    # 5. Overlap probability (shared)
    # temperature=None → step function (hard assignment, non-differentiable)
    # temperature=float → Gaussian kernel with sigma = temperature * sensor_radius
    if temperature is None:
        overlap_prob = create_overlap_prob(None, sensor_radius)
    else:
        overlap_prob = create_overlap_prob(temperature * sensor_radius, sensor_radius)

    # 6. Bounds check closure
    def bounds_check(positions):
        return detector.bounds_check(positions)

    # 7. JIT-compiled propagation function
    @jax.jit
    def propagate_photons(photon_origins, photon_directions):
        """Trace photon rays through detector geometry.

        Parameters
        ----------
        photon_origins : jnp.ndarray, shape (n_rays, 3)
        photon_directions : jnp.ndarray, shape (n_rays, 3)

        Returns
        -------
        dict with keys: sensor_weights, sensor_indices, times, positions,
             normals, inside_sensor, per_sensor_positions, sensor_normals
        """
        if photon_origins.ndim != 2 or photon_directions.ndim != 2:
            raise ValueError('propagate_photons takes batches: origins and directions of shape '
                             f'(n_rays, 3), got {photon_origins.shape} and {photon_directions.shape}')

        # a. Ray-geometry intersection
        intersection_point, t_geometry, grid_info, surface_info = \
            detector.intersect_ray(photon_origins, photon_directions, grid)

        # b. Map to grid cell indices
        idx = detector.point_to_grid_cell(grid_info, grid)

        # c. Look up candidate sensors (stop_gradient: geometry is static)
        potential_sensors = jax.lax.stop_gradient(inverted_sensor_map[idx])

        # d. Compute sensor intersections (shared, vmapped over sensor slots)
        def compute_for_slot(slot_sensors):
            return compute_sensor_intersections_base(
                slot_sensors, sensor_positions, sensor_radius,
                photon_origins, photon_directions,
                bounds_check, overlap_prob, t_geometry=t_geometry)

        (weights, sensor_times, sensor_indices,
         sensor_normals_all, inside_sensor,
         sensor_hit_positions) = jax.vmap(
            compute_for_slot, in_axes=1, out_axes=0)(potential_sensors)

        # First-hit semantics: the photon deposits on the first sensor it reaches. See
        # `first_hit_survival` for the cap, its tie-break, and what its hard ordering costs.
        weights = first_hit_survival(
            weights, jnp.squeeze(sensor_times, -1) if sensor_times.ndim == 3 else sensor_times)

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
            'inside_sensor': inside_sensor,
        }

        return result

    return propagate_photons, grid
