"""
Unified geometry intersection functions for LUCiD propagation.
Implements vectorized and optimized ray-geometry intersection algorithms.
"""
from __future__ import annotations

from typing import Any, Final

import jax
import jax.numpy as jnp


# Geometry type constants for unified interface
SPHERE: Final[int] = 0
CYLINDER: Final[int] = 1
BOX: Final[int] = 2


def ray_sphere_intersection(
    ray_origin: jax.Array,       # (3,)
    ray_direction: jax.Array,    # (3,)
    sphere_center: jax.Array,    # (3,)
    sphere_radius: float,
) -> jax.Array:                  # scalar float (t, or -1)
    """
    Vectorized ray-sphere intersection using analytical solution.
    
    Parameters:
    -----------
    ray_origin : array (3,)
        Ray starting point
    ray_direction : array (3,)
        Ray direction (normalized)
    sphere_center : array (3,)
        Sphere center
    sphere_radius : float
        Sphere radius
        
    Returns:
    --------
    t : float
        Distance to intersection point, -1 if no intersection
    """
    oc = ray_origin - sphere_center
    a = jnp.dot(ray_direction, ray_direction)
    b = 2.0 * jnp.dot(oc, ray_direction) 
    c = jnp.dot(oc, oc) - sphere_radius**2
    
    discriminant = b**2 - 4*a*c
    
    sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 0))
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    
    t = jnp.where(t1 > 0, t1, t2)
    t = jnp.where(discriminant >= 0, t, -1)
    
    return t


def ray_cylinder_intersection(
    ray_origin: jax.Array,       # (3,)
    ray_direction: jax.Array,    # (3,)
    cylinder_radius: float,
    cylinder_height: float,
) -> jax.Array:                  # scalar float (t, or -1)
    """
    Vectorized ray-cylinder intersection (infinite cylinder + caps).
    
    Parameters:
    -----------
    ray_origin : array (3,)
        Ray starting point
    ray_direction : array (3,)
        Ray direction (normalized)
    cylinder_radius : float
        Cylinder radius
    cylinder_height : float
        Cylinder height (centered at origin)
        
    Returns:
    --------
    t : float
        Distance to intersection point, -1 if no intersection
    """
    # Infinite cylinder intersection (ignore z)
    a = ray_direction[0]**2 + ray_direction[1]**2
    b = 2 * (ray_origin[0] * ray_direction[0] + ray_origin[1] * ray_direction[1])
    c = ray_origin[0]**2 + ray_origin[1]**2 - cylinder_radius**2
    
    discriminant = b**2 - 4*a*c
    
    sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 0))
    t1_side = (-b - sqrt_disc) / (2*a)
    t2_side = (-b + sqrt_disc) / (2*a)

    z1 = ray_origin[2] + t1_side * ray_direction[2]
    z2 = ray_origin[2] + t2_side * ray_direction[2]
    
    valid1 = (discriminant >= 0) & (t1_side > 0) & (jnp.abs(z1) <= cylinder_height/2)
    valid2 = (discriminant >= 0) & (t2_side > 0) & (jnp.abs(z2) <= cylinder_height/2)
    
    t_side = jnp.where(valid1, t1_side, jnp.where(valid2, t2_side, jnp.inf))
    
    t_top = jnp.where(
        jnp.abs(ray_direction[2]) > 1e-8,
        (cylinder_height/2 - ray_origin[2]) / ray_direction[2],
        jnp.inf
    )
    t_bottom = jnp.where(
        jnp.abs(ray_direction[2]) > 1e-8, 
        (-cylinder_height/2 - ray_origin[2]) / ray_direction[2],
        jnp.inf
    )
    
    r_top = jnp.linalg.norm(ray_origin[:2] + t_top * ray_direction[:2])
    r_bottom = jnp.linalg.norm(ray_origin[:2] + t_bottom * ray_direction[:2])
    
    valid_top = (t_top > 0) & (r_top <= cylinder_radius)
    valid_bottom = (t_bottom > 0) & (r_bottom <= cylinder_radius)
    
    t_cap = jnp.minimum(
        jnp.where(valid_top, t_top, jnp.inf),
        jnp.where(valid_bottom, t_bottom, jnp.inf)
    )
    
    t = jnp.minimum(t_side, t_cap)
    return jnp.where(t == jnp.inf, -1, t)


def ray_box_intersection_vectorized(
    ray_origin: jax.Array,    # (3,)
    ray_direction: jax.Array, # (3,)
    box_min: jax.Array,       # (3,)
    box_max: jax.Array,       # (3,)
) -> jax.Array:               # scalar float (t, or -1)
    """
    Optimized vectorized ray-box intersection using slab method.
    Much faster JIT compilation than face-by-face approach.
    
    Parameters:
    -----------
    ray_origin : array (3,)
        Ray starting point
    ray_direction : array (3,)
        Ray direction (normalized)
    box_min : array (3,)
        Box minimum corner
    box_max : array (3,)
        Box maximum corner
        
    Returns:
    --------
    t : float
        Distance to intersection point, -1 if no intersection
    """
    # Vectorized slab method - compute all 6 t-values at once
    inv_dir = jnp.where(jnp.abs(ray_direction) > 1e-8, 1.0 / ray_direction, 1e8)
    
    t_min = (box_min - ray_origin) * inv_dir
    t_max = (box_max - ray_origin) * inv_dir
    
    # Ensure t_min <= t_max for each axis
    t1 = jnp.minimum(t_min, t_max)
    t2 = jnp.maximum(t_min, t_max)
    
    # Find intersection interval
    t_near = jnp.max(t1)
    t_far = jnp.min(t2)
    
    # Valid intersection if t_near <= t_far and t_far > 0
    valid = (t_near <= t_far) & (t_far > 0)
    t = jnp.where(t_near > 0, t_near, t_far)  # Prefer entry point
    
    return jnp.where(valid, t, -1)


def ray_box_intersection(
    ray_origin: jax.Array,    # (3,)
    ray_direction: jax.Array, # (3,)
    box_x: float,
    box_y: float,
    box_z: float,
) -> jax.Array:               # scalar float (t, or -1)
    """
    Convenience function for axis-aligned box intersection.
    
    Parameters:
    -----------
    ray_origin : array (3,)
        Ray starting point
    ray_direction : array (3,)
        Ray direction (normalized)
    box_x, box_y, box_z : float
        Box dimensions (full width, height, depth)
        
    Returns:
    --------
    t : float
        Distance to intersection point, -1 if no intersection
    """
    box_min = jnp.array([-box_x/2, -box_y/2, -box_z/2])
    box_max = jnp.array([box_x/2, box_y/2, box_z/2])
    
    return ray_box_intersection_vectorized(ray_origin, ray_direction, box_min, box_max)


def unified_ray_intersection(
    ray_origin: jax.Array,            # (3,)
    ray_direction: jax.Array,         # (3,)
    geometry_type: int,               # SPHERE=0, CYLINDER=1, BOX=2
    geometry_params: dict[str, Any],
) -> jax.Array:                       # scalar float (t, or -1)
    """
    Unified interface for all geometry intersections.
    Uses JAX switch for efficient branching.
    
    Parameters:
    -----------
    ray_origin : array (3,)
        Ray starting point
    ray_direction : array (3,)
        Ray direction (normalized)
    geometry_type : int
        Geometry type (SPHERE=0, CYLINDER=1, BOX=2)
    geometry_params : dict
        Geometry-specific parameters
        
    Returns:
    --------
    t : float
        Distance to intersection point, -1 if no intersection
    """
    def sphere_intersection(params):
        return ray_sphere_intersection(
            ray_origin, ray_direction, 
            params.get('center', jnp.zeros(3)),
            params['radius']
        )
    
    def cylinder_intersection(params):
        return ray_cylinder_intersection(
            ray_origin, ray_direction,
            params['radius'], 
            params['height']
        )
    
    def box_intersection(params):
        return ray_box_intersection(
            ray_origin, ray_direction,
            params['x'], params['y'], params['z']
        )
    
    return jax.lax.switch(
        geometry_type,
        [sphere_intersection, cylinder_intersection, box_intersection],
        geometry_params
    )


def compute_surface_normal(
    intersection_point: jax.Array,     # (3,)
    geometry_type: int,
    geometry_params: dict[str, Any],
) -> jax.Array:                        # (3,) unit normal
    """
    Compute surface normal at intersection point.
    
    Parameters:
    -----------
    intersection_point : array (3,)
        Point on surface
    geometry_type : int
        Geometry type (SPHERE=0, CYLINDER=1, BOX=2)
    geometry_params : dict
        Geometry-specific parameters
        
    Returns:
    --------
    normal : array (3,)
        Unit surface normal vector
    """
    def sphere_normal(params):
        center = params.get('center', jnp.zeros(3))
        normal = intersection_point - center
        return normal / jnp.linalg.norm(normal)
    
    def cylinder_normal(params):
        # For cylinder side: normal is radial
        # For caps: normal is ±z
        radius = params['radius']
        height = params['height']
        
        z = intersection_point[2]
        on_top_cap = jnp.abs(z - height/2) < 1e-6
        on_bottom_cap = jnp.abs(z + height/2) < 1e-6
        
        # Cap normals
        top_normal = jnp.array([0., 0., 1.])
        bottom_normal = jnp.array([0., 0., -1.])
        
        # Side normal (radial)
        side_normal = jnp.array([intersection_point[0], intersection_point[1], 0.])
        side_normal = side_normal / jnp.linalg.norm(side_normal)
        
        return jnp.where(
            on_top_cap, top_normal,
            jnp.where(on_bottom_cap, bottom_normal, side_normal)
        )
    
    def box_normal(params):
        # Determine which face by finding largest coordinate component
        abs_coords = jnp.abs(intersection_point)
        max_coord = jnp.argmax(abs_coords)
        
        normal = jnp.zeros(3)
        normal = normal.at[max_coord].set(jnp.sign(intersection_point[max_coord]))
        
        return normal
    
    return jax.lax.switch(
        geometry_type,
        [sphere_normal, cylinder_normal, box_normal],
        geometry_params
    )


def unified_bounds_check(
    points: jax.Array,                 # (N, 3)
    geometry_type: int,
    geometry_params: dict[str, Any],
) -> jax.Array:                        # (N,) bool
    """
    Unified bounds checking for all detector geometries.
    
    Parameters:
    -----------
    points : array (N, 3)
        Points to check
    geometry_type : int
        Geometry type constant (SPHERE=0, CYLINDER=1, BOX=2)
    geometry_params : dict
        Geometry-specific parameters
        
    Returns:
    --------
    valid : array (N,)
        Boolean array indicating which points are inside
    """
    def sphere_bounds(params):
        center = params.get('center', jnp.zeros(3))
        radius = params['radius']
        distances = jnp.linalg.norm(points - center, axis=1)
        return distances <= radius
    
    def cylinder_bounds(params):
        radius = params['radius']
        height = params['height']
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        inside_xy_circle = (x**2 + y**2) <= radius**2
        inside_z_bounds = (z >= -height/2) & (z <= height/2)
        return inside_xy_circle & inside_z_bounds
    
    def box_bounds(params):
        x_size = params.get('x', params.get('length', 1.0))
        y_size = params.get('y', params.get('width', 1.0))
        z_size = params.get('z', params.get('height', 1.0))
        
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        inside_x = (x >= -x_size/2) & (x <= x_size/2)
        inside_y = (y >= -y_size/2) & (y <= y_size/2)
        inside_z = (z >= -z_size/2) & (z <= z_size/2)
        return inside_x & inside_y & inside_z
    
    return jax.lax.switch(
        geometry_type,
        [sphere_bounds, cylinder_bounds, box_bounds],
        geometry_params
    )