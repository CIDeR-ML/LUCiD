"""
Sphere-specific photon propagation functions with indexing fixes.
"""

import jax
import jax.numpy as jnp
from functools import partial
from jax import lax

from .base import (
    process_intersection_normals, compute_sensor_intersections_base,
    find_closest_sensors
)
from ..overlap import create_overlap_prob


@jax.jit
def intersect_sphere(ray_origin, ray_direction, center, radius):
    """Calculate intersection of a ray with a sphere.

    Parameters
    ----------
    ray_origin : jnp.ndarray
        Starting point of the ray [x, y, z]
    ray_direction : jnp.ndarray
        Direction vector of the ray [dx, dy, dz]
    center : jnp.ndarray
        Center of the sphere [x, y, z]
    radius : float
        Radius of the sphere

    Returns
    -------
    tuple
        (bool, float) - (whether intersection exists, distance to intersection)
    """
    LARGE = 1e10
    
    # Vector from ray origin to sphere center
    oc = ray_origin - center
    
    # Quadratic equation coefficients for sphere intersection
    a = jnp.sum(ray_direction * ray_direction)
    b = 2.0 * jnp.sum(oc * ray_direction)
    c = jnp.sum(oc * oc) - radius**2
    
    discriminant = b**2 - 4*a*c
    epsilon = 1e-6
    
    def intersection_branch(_):
        sqrt_disc = jnp.sqrt(jnp.maximum(0.0, discriminant))
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        
        # We want the intersection from inside the sphere going outward
        # So we take the positive t (exit point)
        t_candidate = jnp.maximum(t1, t2)
        
        valid_t = t_candidate > 0
        intersects_ = (discriminant >= -epsilon) & valid_t
        tval_ = jnp.where(intersects_, t_candidate, LARGE)
        
        return (intersects_, tval_)
    
    def no_intersection_branch(_):
        return (False, jnp.array(LARGE, dtype=jnp.float32))
    
    has_intersection = discriminant >= -epsilon
    intersects, tval = lax.cond(has_intersection,
                                intersection_branch,
                                no_intersection_branch,
                                operand=None)
    
    return intersects, tval


@partial(jax.jit, static_argnums=(2, 3))
def intersect_sphere_with_grid(ray_origin, ray_direction, radius, n_divisions):
    """Find intersection with sphere and compute grid cell indices.

    Parameters
    ----------
    ray_origin : jnp.ndarray
        Starting point of the ray [x, y, z]
    ray_direction : jnp.ndarray
        Direction vector of the ray [dx, dy, dz]
    radius : float
        Radius of the sphere
    n_divisions : int
        Number of divisions for grid resolution

    Returns
    -------
    tuple
        (intersects, t, theta_idx, phi_idx, intersection_point)
    """
    center = jnp.array([0.0, 0.0, 0.0])
    intersects, t = intersect_sphere(ray_origin, ray_direction, center, radius)
    intersection_point = ray_origin + t * ray_direction
    
    # Convert intersection point to spherical coordinates relative to sphere center
    relative_point = intersection_point - center
    
    # Calculate spherical coordinates
    r = jnp.linalg.norm(relative_point)
    theta = jnp.arccos(jnp.clip(relative_point[2] / (r + 1e-10), -1.0, 1.0))  # polar angle [0, π]
    phi = jnp.arctan2(relative_point[1], relative_point[0]) % (2 * jnp.pi)  # azimuthal angle [0, 2π]
    
    # Convert to grid indices
    n_theta = n_divisions
    n_phi = 2 * n_divisions  # More divisions in phi for roughly uniform cells
    
    theta_idx = jnp.floor(theta / jnp.pi * n_theta).astype(jnp.int32)
    phi_idx = jnp.floor(phi / (2 * jnp.pi) * n_phi).astype(jnp.int32)
    
    # Clamp indices to valid range (IMPORTANT: fixes for sphere)
    theta_idx = jnp.clip(theta_idx, 0, n_theta - 1)
    phi_idx = phi_idx % n_phi  # Handle wraparound for phi dimension
    
    return intersects, t, theta_idx, phi_idx, intersection_point


batch_intersect_sphere_with_grid = jax.vmap(intersect_sphere_with_grid,
                                           in_axes=(0, 0, None, None))


def calculate_sphere_normals(intersection_point):
    """
    Calculate normals for sphere surface.
    
    Parameters
    ----------
    intersection_point : ndarray
        Points of intersection on the sphere surface
        
    Returns
    -------
    ndarray
        Normal vectors for sphere intersections (pointing outward)
    """
    # For sphere, normal at any point is the vector from center to point
    center = jnp.array([0.0, 0.0, 0.0])
    normals = intersection_point - center
    # Normalize
    normals = normals / (jnp.linalg.norm(normals, axis=1, keepdims=True) + 1e-10)
    return normals


def sphere_bounds_check(points, radius):
    """
    Check if points are within sphere bounds.
    
    Parameters
    ----------
    points : ndarray
        Points to check [x, y, z]
    radius : float
        Sphere radius
        
    Returns
    -------
    ndarray
        Boolean array indicating which points are inside sphere
    """
    center = jnp.array([0.0, 0.0, 0.0])
    distances = jnp.linalg.norm(points - center, axis=1)
    return distances <= radius


@partial(jax.jit, static_argnums=(2, 3))
def assign_sensors_to_sphere_grid(sensors, sensor_radius, radius, n_divisions):
    """Assign sensors to spherical grid cells, handling overlap across cell boundaries.

    Parameters
    ----------
    sensors : jnp.ndarray
        Array of sensor positions, shape (n_sensors, 3)
    sensor_radius : float
        Radius of each sensor
    radius : float
        Sphere radius
    n_divisions : int
        Grid resolution parameter

    Returns
    -------
    jnp.ndarray
        Array of shape (n_sensors, 4, 2) containing up to 4 grid cell assignments
        per sensor. -1 indicates no assignment.
    """
    
    def assign_single_sensor(sensor):
        # Convert sensor position to spherical coordinates relative to sphere center
        center = jnp.array([0.0, 0.0, 0.0])
        relative_pos = sensor - center
        r = jnp.linalg.norm(relative_pos)
        
        # Check if sensor is approximately on sphere surface
        on_surface = jnp.abs(r - radius) <= sensor_radius
        
        def assign_surface():
            theta = jnp.arccos(jnp.clip(relative_pos[2] / (r + 1e-10), -1.0, 1.0))
            phi = jnp.arctan2(relative_pos[1], relative_pos[0]) % (2 * jnp.pi)
            
            n_theta = n_divisions
            n_phi = 2 * n_divisions
            
            theta_idx = jnp.floor(theta / jnp.pi * n_theta).astype(jnp.int32)
            phi_idx = jnp.floor(phi / (2 * jnp.pi) * n_phi).astype(jnp.int32)
            
            # Calculate overlap with neighboring cells
            theta_frac = (theta / jnp.pi * n_theta) % 1
            phi_frac = (phi / (2 * jnp.pi) * n_phi) % 1
            
            # Angular size of sensor relative to grid cell size
            theta_cell_size = jnp.pi / n_theta
            phi_cell_size = 2 * jnp.pi / n_phi
            
            # Approximate angular size of sensor on sphere surface
            angular_size = sensor_radius / radius
            
            include_theta_up = theta_frac >= 1 - angular_size / theta_cell_size
            include_theta_down = theta_frac <= angular_size / theta_cell_size
            include_phi_right = phi_frac >= 1 - angular_size / phi_cell_size
            include_phi_left = phi_frac <= angular_size / phi_cell_size
            
            # Fixed: Handle boundaries correctly for theta (can't wrap) and phi (wraps)
            theta_up = jnp.clip(theta_idx + 1, 0, n_theta - 1)
            theta_down = jnp.clip(theta_idx - 1, 0, n_theta - 1)
            phi_right = (phi_idx + 1) % n_phi  # Phi wraps around
            phi_left = (phi_idx - 1) % n_phi   # Phi wraps around
            
            indices = jnp.array([
                [theta_idx, phi_idx],  # Central cell
                [theta_up, phi_idx],   # Theta up
                [theta_down, phi_idx], # Theta down
                [theta_idx, phi_right], # Phi right
                [theta_idx, phi_left],  # Phi left
                [theta_up, phi_right],  # Diagonal
                [theta_up, phi_left],   # Diagonal
                [theta_down, phi_right], # Diagonal
                [theta_down, phi_left]   # Diagonal
            ])
            
            selection = jnp.array([
                1.0,  # Central cell always included
                include_theta_up,
                include_theta_down,
                include_phi_right,
                include_phi_left,
                include_theta_up * include_phi_right,
                include_theta_up * include_phi_left,
                include_theta_down * include_phi_right,
                include_theta_down * include_phi_left
            ])
            
            sorted_indices = indices[jnp.argsort(-selection)]
            
            return jnp.where(jnp.arange(4)[:, None] < jnp.sum(selection), sorted_indices[:4], -1)
        
        def assign_off_surface():
            return jnp.full((4, 2), -1, dtype=jnp.int32)
        
        return lax.cond(on_surface, assign_surface, assign_off_surface)
    
    return jax.vmap(assign_single_sensor)(sensors)


@partial(jax.jit, static_argnums=(1,))
def create_sensor_sphere_grid_map(assignments, n_divisions):
    """
    Creates a grid map counting the number of sensors in each cell of the spherical sensor grid.
    """
    n_theta = n_divisions
    n_phi = 2 * n_divisions
    total_cells = n_theta * n_phi
    grid = jnp.zeros(total_cells, dtype=jnp.int32)
    
    def update_grid(sensor_assignments):
        def update_cell(cell, g):
            theta_idx, phi_idx = cell
            is_valid = (theta_idx != -1) & (phi_idx != -1)
            
            # Calculate linear index
            idx = theta_idx * n_phi + phi_idx
            
            return g.at[idx].add(is_valid)
        
        return jax.lax.fori_loop(0, sensor_assignments.shape[0], 
                                lambda i, g: update_cell(sensor_assignments[i], g), grid)
    
    all_updates = jax.vmap(update_grid)(assignments)
    return all_updates.sum(axis=0)


@partial(jax.jit, static_argnums=(1,))
def calculate_sphere_grid_centers(radius, n_divisions):
    """Calculate center points of all spherical grid cells"""
    center = jnp.array([0.0, 0.0, 0.0])
    n_theta = n_divisions
    n_phi = 2 * n_divisions
    
    # Grid cell centers in spherical coordinates
    theta_step = jnp.pi / n_theta
    phi_step = 2 * jnp.pi / n_phi
    
    theta_centers = (jnp.arange(n_theta) + 0.5) * theta_step
    phi_centers = (jnp.arange(n_phi) + 0.5) * phi_step
    
    theta_grid, phi_grid = jnp.meshgrid(theta_centers, phi_centers, indexing='ij')
    
    # Convert to Cartesian coordinates
    x = radius * jnp.sin(theta_grid) * jnp.cos(phi_grid) + center[0]
    y = radius * jnp.sin(theta_grid) * jnp.sin(phi_grid) + center[1]
    z = radius * jnp.cos(theta_grid) + center[2]
    
    centers = jnp.stack([
        x.reshape(-1),
        y.reshape(-1),
        z.reshape(-1)
    ], axis=1)
    
    return centers


@partial(jax.jit, static_argnums=(2, 3, 4), device=jax.devices('cpu')[0])
def create_inverted_sphere_sensor_map(assignments_geometric, assignments_distance, n_divisions,
                                       max_sensors_per_cell, num_sensors):
    """Create inverted sensor map for sphere prioritizing geometric intersections then closest sensors"""
    n_theta = n_divisions
    n_phi = 2 * n_divisions
    total_cells = n_theta * n_phi
    
    # Initialize map
    inverted_map = jnp.full((total_cells, max_sensors_per_cell), -1, dtype=jnp.int32)
    
    def update_cell(carry, i):
        inv_map = carry
        
        def add_geometric(carry, j):
            curr_map, curr_count = carry
            
            # Convert linear index i back to theta, phi coordinates
            theta_idx = i // n_phi
            phi_idx = i % n_phi
            
            # Check if sensor j intersects with this cell
            sensor_assignments = assignments_geometric[j]
            matches = (sensor_assignments[:, 0] == theta_idx) & \
                     (sensor_assignments[:, 1] == phi_idx)
            
            cell_matches = jnp.any(matches)
            should_add = cell_matches & (curr_count < max_sensors_per_cell)
            
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
            should_add = (~is_duplicate) & (curr_count < max_sensors_per_cell)
            
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


def find_intersected_sphere_sensors_differentiable(ray_origins, ray_directions, sensor_positions, sensor_radius,
                                                    radius, n_divisions, inverted_sensor_map,
                                                    temperature, overlap_prob):
    """
    Finds sensors intersected by rays using a differentiable approximation with overlap-based weights.
    """
    single_ray = ray_origins.ndim == 1
    if single_ray:
        ray_origins = ray_origins[None, :]
        ray_directions = ray_directions[None, :]

    # Get sphere intersection points and grid indices
    center = jnp.array([0.0, 0.0, 0.0])
    intersects, t_sphere, theta_idx, phi_idx, intersection_point = (
        batch_intersect_sphere_with_grid(ray_origins, ray_directions, radius, n_divisions))

    def calculate_linear_index(theta_idx, phi_idx):
        n_theta = n_divisions
        n_phi = 2 * n_divisions
        idx = theta_idx * n_phi + phi_idx
        total_cells = n_theta * n_phi
        return jnp.clip(idx, 0, total_cells - 1)

    idx = calculate_linear_index(theta_idx, phi_idx)
    potential_sensors = jax.lax.stop_gradient(inverted_sensor_map[idx])

    # Create bounds check function
    bounds_check = lambda points: sphere_bounds_check(points, radius)

    # Process all potential sensors
    sensor_results = jax.vmap(
        lambda det_idx: compute_sensor_intersections_base(
            det_idx, sensor_positions, sensor_radius,
            ray_origins, ray_directions, bounds_check, overlap_prob
        )
    )(potential_sensors.T)
    
    weights = sensor_results[0]
    sensor_distances = sensor_results[1]
    sensor_indices = sensor_results[2]
    sensor_normals = sensor_results[3]
    inside_sensor = sensor_results[4]
    sensor_hit_positions = sensor_results[5]

    # Calculate sphere surface normals
    sphere_normals = calculate_sphere_normals(intersection_point)

    intersection_results = process_intersection_normals(
        ray_origins, ray_directions, intersection_point,
        t_sphere, sensor_normals, sensor_hit_positions,
        inside_sensor, sphere_normals
    )

    hit_positions = intersection_results['positions']
    final_normals = intersection_results['normals']

    result = {
        'sensor_distances': sensor_distances,
        'sensor_weights': weights,
        'sensor_indices': sensor_indices,
        'per_sensor_positions': sensor_hit_positions,
        'positions': hit_positions,
        'normals': final_normals,
        'sensor_normals': sensor_normals,
        'inside_sensor': inside_sensor
    }

    return result if not single_ray else jax.tree_map(lambda x: x[0], result)


def create_sphere_photon_propagator(sensor_positions, sensor_radius, sphere_radius=4.0, n_divisions=50,
                                   temperature=0.2, max_sensors_per_cell=4):
    """
    Creates a JIT-compiled function for efficient photon propagation simulation in sphere geometry.
    """

    assignments_geometric = assign_sensors_to_sphere_grid(
        sensor_positions, sensor_radius, sphere_radius, n_divisions)

    sensor_grid_map = create_sensor_sphere_grid_map(
        assignments_geometric, n_divisions)

    assignments_distance = find_closest_sensors(
        calculate_sphere_grid_centers(sphere_radius, n_divisions),
        sensor_positions,
        max_sensors_per_cell
    )

    inverted_sensor_map = create_inverted_sphere_sensor_map(
        assignments_geometric,
        assignments_distance,
        n_divisions,
        max_sensors_per_cell, sensor_positions.shape[0]
    )

    if temperature is None:
        overlap_prob = create_overlap_prob(temperature, sensor_radius)
    else:
        overlap_prob = create_overlap_prob(temperature * sensor_radius, sensor_radius)

    @jax.jit
    def propagate_photons(photon_origins, photon_directions):
        return find_intersected_sphere_sensors_differentiable(
            photon_origins, photon_directions, sensor_positions, sensor_radius,
            sphere_radius, n_divisions, inverted_sensor_map,
            temperature, overlap_prob)

    return propagate_photons