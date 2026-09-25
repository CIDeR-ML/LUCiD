"""
LUCiD propagation module with unified geometry interface.
"""

# Import geometry functions for optimization
from .geometry import (
    ray_sphere_intersection, ray_cylinder_intersection, ray_box_intersection_vectorized,
    unified_ray_intersection, compute_surface_normal,
    SPHERE, CYLINDER, BOX
)

# Import detector-specific propagation functions
from . import cylinder, sphere, box

# Propagators for every geometry are built by `lucid.propagation.shared.create_propagator`;
# only the per-geometry bounds checks are re-exported here.
from .cylinder import cylinder_bounds_check
from .sphere import sphere_bounds_check
from .box import box_bounds_check

__all__ = [
    # Geometry functions
    'ray_sphere_intersection',
    'ray_cylinder_intersection', 
    'ray_box_intersection_vectorized',
    'unified_ray_intersection',
    'compute_surface_normal',
    'SPHERE', 'CYLINDER', 'BOX',
    
    # Detector modules
    'cylinder', 'sphere', 'box',
    
    # Bounds check functions
    'cylinder_bounds_check',
    'sphere_bounds_check',
    'box_bounds_check',
]