"""Shared photon propagator factory using Detector abstract methods.

This replaces the 3 geometry-specific factories (create_photon_propagator,
create_sphere_photon_propagator, create_box_photon_propagator) with a
single function that works for any Detector subclass implementing the
Phase 9 abstract methods.
"""
import jax
import jax.numpy as jnp

from lucid.propagation.base import (
    compute_sensor_intersections_base,
    process_intersection_normals,
    find_closest_sensors,
)
from lucid.overlap import create_overlap_prob


def create_propagator(detector, sensor_positions, sensor_radius,
                      temperature=0.2, max_sensors_per_cell=4,
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
    max_sensors_per_cell : int
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

    # Configure grid on detector — caller passes geometry-specific params
    detector.configure_grid(**grid_params)

    # 1. Geometric sensor-to-cell assignments
    assignments_geometric = detector.assign_sensor_to_cells(sensor_positions, sensor_radius)

    # 2. Grid cell centers
    grid_centers = detector.grid_cell_centers()

    # 3. Distance-based fallback assignments (shared)
    assignments_distance = find_closest_sensors(
        grid_centers, sensor_positions, max_sensors_per_cell)

    # 4. Build inverted sensor map (geometry-specific decoder)
    inverted_sensor_map = detector.build_inverted_sensor_map(
        assignments_geometric, assignments_distance,
        max_sensors_per_cell, num_sensors)

    # 5. Overlap probability (shared)
    sigma = temperature * sensor_radius
    overlap_prob = create_overlap_prob(sigma, sensor_radius)

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
        def compute_for_slot(slot_sensors):
            return compute_sensor_intersections_base(
                slot_sensors, sensor_positions, sensor_radius,
                photon_origins, photon_directions,
                bounds_check, overlap_prob)

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
