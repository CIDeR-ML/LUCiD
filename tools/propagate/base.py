"""
Base functionality shared across all detector geometries for photon propagation.
"""

import jax
import jax.numpy as jnp
from functools import partial
from jax import lax
from ..overlap import create_overlap_prob


def process_intersection_normals(ray_origins, ray_directions, intersection_point,
                                 t_geometry, sensor_normals, sensor_hit_positions,
                                 inside_sensor, geometry_normals):
    """
    Main function to process all normal and position calculations for intersections.
    
    This is a generic version that works for any geometry type.
    
    Parameters
    ----------
    ray_origins : ndarray
        Starting points of rays
    ray_directions : ndarray
        Direction vectors of rays
    intersection_point : ndarray
        Points of intersection with geometry
    t_geometry : ndarray
        Intersection times with geometry
    sensor_normals : ndarray
        Normal vectors for sensor intersections
    sensor_hit_positions : ndarray
        Hit positions for sensor intersections
    inside_sensor : ndarray
        Boolean array indicating which rays hit inside sensors
    geometry_normals : ndarray
        Normal vectors for the geometry surface
        
    Returns
    -------
    dict
        Contains hit positions and normals
    """
    # Calculate weighted sensor properties
    weighted_sensor_normals, weighted_sensor_positions = calculate_weighted_sensor_properties(
        sensor_normals, sensor_hit_positions, inside_sensor)

    # Calculate final hit properties
    hit_positions, final_normals = calculate_hit_properties(
        ray_origins, ray_directions, t_geometry, inside_sensor,
        weighted_sensor_normals, weighted_sensor_positions,
        geometry_normals)
    
    return {
        'positions': hit_positions,
        'normals': final_normals
    }


def calculate_weighted_sensor_properties(sensor_normals, sensor_hit_positions,
                                          inside_sensor):
    """
    Calculate weighted normals and positions for sensor hits.
    
    Parameters
    ----------
    sensor_normals : ndarray
        Normal vectors for all potential sensor intersections
    sensor_hit_positions : ndarray
        Hit positions for all potential sensor intersections
    inside_sensor : ndarray
        Boolean array indicating which rays hit inside sensors
        
    Returns
    -------
    tuple
        Weighted sensor normals and positions
    """
    sensor_weights = inside_sensor[..., None]
    
    # Calculate weighted normals
    weighted_normals = jnp.sum(sensor_normals * sensor_weights, axis=0)
    sensor_weights_sum = jnp.sum(inside_sensor, axis=0)[..., None]
    weighted_normals = weighted_normals / (sensor_weights_sum + 1e-10)
    
    # Calculate weighted positions
    weighted_positions = jnp.sum(sensor_hit_positions * sensor_weights, axis=0)
    weighted_positions = weighted_positions / (sensor_weights_sum + 1e-10)
    
    return weighted_normals, weighted_positions


def calculate_hit_properties(ray_origins, ray_directions, t_geometry, inside_sensor,
                             weighted_sensor_normals, weighted_sensor_positions,
                             geometry_normals):
    """
    Calculate final hit positions and normals based on intersection type.
    
    Parameters
    ----------
    ray_origins : ndarray
        Starting points of rays
    ray_directions : ndarray
        Direction vectors of rays
    t_geometry : ndarray
        Intersection times with geometry
    inside_sensor : ndarray
        Boolean array indicating which rays hit inside sensors
    weighted_sensor_normals : ndarray
        Weighted normal vectors for sensor hits
    weighted_sensor_positions : ndarray
        Weighted positions for sensor hits
    geometry_normals : ndarray
        Normal vectors for geometry hits
        
    Returns
    -------
    tuple
        Final hit positions and normals
    """
    # Calculate geometry hit positions
    geometry_hit_positions = ray_origins + t_geometry[:, None] * ray_directions
    
    # Determine if any sensor was hit
    hit_sensor = jnp.any(inside_sensor, axis=0)
    
    # Select appropriate hit position
    hit_positions = jnp.where(hit_sensor[:, None],
                              weighted_sensor_positions,
                              geometry_hit_positions)
    
    # Select appropriate normal
    final_normals = jnp.where(hit_sensor[:, None],
                              weighted_sensor_normals,
                              geometry_normals)
    
    return hit_positions, final_normals


def compute_sensor_intersections_base(sensor_idx, sensor_positions, sensor_radius,
                                        ray_origins, ray_directions, geometry_bounds_check,
                                        overlap_prob):
    """
    Base function to compute sensor intersections for any geometry.
    
    Parameters
    ----------
    sensor_idx : ndarray
        Indices of potential sensors
    sensor_positions : ndarray
        Positions of all sensors
    sensor_radius : float
        Radius of each sensor
    ray_origins : ndarray
        Starting points of rays
    ray_directions : ndarray
        Direction vectors of rays
    geometry_bounds_check : callable
        Function that checks if a point is within the geometry bounds
    overlap_prob : callable
        Function that calculates overlap probability
        
    Returns
    -------
    tuple
        weights, times, sensor_idx, normals, inside_sensor, points
    """
    valid = sensor_idx != -1
    sphere_centers = jnp.where(valid[:, None], sensor_positions[sensor_idx], jnp.zeros(3))
    
    # Find closest approach of ray to sensor center
    oc = ray_origins - sphere_centers
    ray_d = ray_directions / (jnp.linalg.norm(ray_directions, axis=1, keepdims=True) + 1e-10)
    
    # Calculate closest approach for all rays (stable for gradients)
    t_closest = -jnp.sum(oc * ray_d, axis=1, keepdims=True)
    closest = ray_origins + t_closest * ray_d
    to_sensor = closest - sphere_centers
    distance = jnp.linalg.norm(to_sensor, axis=1)
    
    # Calculate normal vectors for closest approach
    normals_closest = to_sensor / (jnp.linalg.norm(to_sensor, axis=1, keepdims=True) + 1e-10)
    
    # Ray-sphere intersection coefficients
    a = jnp.sum(ray_d * ray_d, axis=1)  # Should be 1 for normalized directions
    b = 2.0 * jnp.sum(oc * ray_d, axis=1)
    c = jnp.sum(oc * oc, axis=1) - sensor_radius ** 2
    
    # Discriminant determines if intersection exists
    discriminant = b ** 2 - 4 * a * c
    
    # Calculate actual intersection for rays that hit the sensor
    sqrt_term = jnp.sqrt(jnp.maximum(1e-10, discriminant))
    
    # Use numerically stable quadratic formula to prevent NaN gradients
    q = jnp.where(
        b > 0,
        -0.5 * (b + sqrt_term),
        -0.5 * (b - sqrt_term)
    )
    t1 = q / (a + 1e-10)
    t2 = c / (q + jnp.sign(q) * 1e-10)
    
    t_intersect = jnp.where((t1 > 0) & (t2 > 0), 
                    jnp.minimum(t1, t2),  # Both positive - take smaller
                    jnp.where(t1 > 0, t1,  # Only t1 positive
                           jnp.where(t2 > 0, t2, -1)))  # Only t2 positive or neither
    
    # Calculate intersection points
    intersection_points = ray_origins + t_intersect[:, None] * ray_d
    
    # Calculate normals at intersection points
    to_intersection = intersection_points - sphere_centers
    normals_intersect = to_intersection / (jnp.linalg.norm(to_intersection, axis=1, keepdims=True) + 1e-10)
    
    # Determine if ray intersects with sensor (add small epsilon for stability)
    intersects = (discriminant > 1e-6) & (t_intersect > 0)
   
    # Apply overlap function to get weights
    weights = jnp.where(valid, overlap_prob(distance), 0.0)    
    # Check if point is inside sensor (keep as boolean)
    inside_spherical_sensor = distance < sensor_radius
    
    # Check if intersection point is within geometry bounds
    inside_detector_volume = geometry_bounds_check(intersection_points)
    
    # # Use correct normals based on whether ray intersects or not
    # normals = jnp.where(intersects[:, None], normals_intersect, normals_closest)

    # Final sensor condition - point must be inside both sensor and geometry
    inside_sensor = inside_spherical_sensor & inside_detector_volume
    
    # Combine boolean conditions first, then add dimension
    intersects_and_inside = (intersects & inside_sensor)
    
    # Now use this combined condition
    times   = jnp.where(intersects_and_inside[:, None], t_intersect[:, None], t_closest)
    points  = jnp.where(intersects_and_inside[:, None], intersection_points, closest)
    normals = jnp.where(intersects_and_inside[:, None], normals_intersect, normals_closest)

    #return weights, times, sensor_idx, normals, inside_sensor, points

    # In principle we should not be using anything if intersects_and_inside is false, reflecting instead out of the detector surface.
    return weights, times, sensor_idx, normals, intersects_and_inside, points


@partial(jax.jit, static_argnums=(1,))
def calculate_linear_index_base(indices, grid_dims, index_map):
    """
    Calculate linear index for grid cells.
    
    Parameters
    ----------
    indices : tuple
        Grid indices for each dimension
    grid_dims : tuple
        Dimensions of the grid
    index_map : callable
        Function that maps grid indices to linear index
        
    Returns
    -------
    int
        Linear index
    """
    return index_map(indices, grid_dims)


@partial(jax.jit, static_argnums=(2,))
def find_closest_sensors(grid_centers, sensor_positions, max_sensors_per_cell):
    """Find closest sensors to each grid cell center"""
    squared_distances = jnp.sum(
        (grid_centers[:, None, :] - sensor_positions[None, :, :]) ** 2,
        axis=2
    )
    return jax.lax.top_k(-squared_distances, max_sensors_per_cell)[1]