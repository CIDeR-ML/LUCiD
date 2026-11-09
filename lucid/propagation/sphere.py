"""
Sphere-specific photon propagation functions with indexing fixes.
"""

import jax
import jax.numpy as jnp
from functools import partial
from jax import lax

from lucid.propagation.sensor_map import grid_face_cells


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
        # dtype follows the ray, not a pinned float32 (as in cylinder.py's `parallel_side_branch`):
        # `lax.cond` requires both branches to agree, the other branch follows the input
        # precision, and a hard float32 would break the library under `jax_enable_x64`.
        return (False, jnp.array(LARGE, dtype=ray_origin.dtype))
    
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
        Array of shape (n_sensors, 9, 2) containing up to 9 grid cell assignments
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
            
            # Angular reach of the sensor disk: r_s / R in theta, and r_s / (R sin theta) in phi,
            # since a circle of latitude has radius R sin theta. Capped at 1 rad near the poles,
            # more than any cell.
            angular_size = sensor_radius / radius
            phi_reach = angular_size / jnp.maximum(jnp.sin(theta), angular_size)
            
            include_theta_up = theta_frac >= 1 - angular_size / theta_cell_size
            include_theta_down = theta_frac <= angular_size / theta_cell_size
            include_phi_right = phi_frac >= 1 - phi_reach / phi_cell_size
            include_phi_left = phi_frac <= phi_reach / phi_cell_size
            
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
            
            return jnp.where(jnp.arange(9)[:, None] < jnp.sum(selection), sorted_indices, -1)
        
        def assign_off_surface():
            return jnp.full((9, 2), -1, dtype=jnp.int32)
        
        return lax.cond(on_surface, assign_surface, assign_off_surface)
    
    return jax.vmap(assign_single_sensor)(sensors)


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


def sphere_grid_cells(n_divisions):
    """Grid coordinates (theta, phi) of every sphere cell, by linear index:
    (2 * n_divisions**2, 2) int32, the convention of :func:`assign_sensors_to_sphere_grid`."""
    return grid_face_cells(n_divisions, 2 * n_divisions)
