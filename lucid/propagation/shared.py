"""Shared photon propagator factory using Detector abstract methods.

This replaces the 3 geometry-specific factories (create_photon_propagator,
create_sphere_photon_propagator, create_box_photon_propagator) with a
single function that works for any Detector subclass implementing the
Phase 9 abstract methods.
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
                      overlap_mode='interp', apply_angular_acceptance=False,
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

    # Configure grid on detector — caller passes geometry-specific params.
    # max_candidates_per_ray is always forwarded so auto-derivation can
    # ensure no cell exceeds this limit.
    grid_params.setdefault('max_candidates_per_ray', max_candidates_per_ray)
    detector.configure_grid(**grid_params)

    # 1. Geometric sensor-to-cell assignments
    assignments_geometric = detector.assign_sensor_to_cells(sensor_positions, sensor_radius)

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
    if temperature is None:
        overlap_prob = create_overlap_prob(
            None, sensor_radius,
            st_width_frac=overlap_st_width_frac, renorm=overlap_renorm, mode=overlap_mode)
    else:
        overlap_prob = create_overlap_prob(
            temperature * sensor_radius, sensor_radius,
            st_width_frac=overlap_st_width_frac, renorm=overlap_renorm, mode=overlap_mode)

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

        # d. Compute sensor intersections (shared, vmapped over sensor slots).
        #    The PMT cosθ angular acceptance is OPT-IN: applied only when the detector config
        #    sets apply_angular_acceptance AND the geometry exposes per-sensor normals. Default
        #    OFF → sensor_normals=None → byte-identical to the legacy capture (no charge rescale
        #    on existing detectors). The geometry always PROVIDES the normals (the capability);
        #    the config decides the POLICY.
        _sensor_normals = None
        if apply_angular_acceptance and getattr(detector, 'sensor_normals', None) is not None:
            _sensor_normals = jnp.asarray(detector.sensor_normals)   # numpy → device array for the gather

        def compute_for_slot(slot_sensors):
            return compute_sensor_intersections_base(
                slot_sensors, sensor_positions, sensor_radius,
                photon_origins, photon_directions,
                bounds_check, overlap_prob, sensor_normals=_sensor_normals)

        (weights, sensor_times, sensor_indices,
         sensor_normals_all, inside_sensor,
         sensor_hit_positions) = jax.vmap(
            compute_for_slot, in_axes=1, out_axes=0)(potential_sensors)

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

        if single_ray:
            result = jax.tree.map(lambda x: x[0] if x.ndim > 0 else x, result)

        return result

    return propagate_photons
