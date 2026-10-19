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
# only the per-geometry bounds checks and standalone SK PMT helpers are re-exported here.
from .cylinder import cylinder_bounds_check
from .sphere import sphere_bounds_check
from .box import box_bounds_check
from .sk_pmt_jax import JAXSK20InchPMTHit, intersect_sk20inch_pmt_jax
from .sk_pmt_coverage import (
    SKPMTSurfaceQuadrature,
    create_sk20inch_surface_quadrature,
    sk20inch_gaussian_coverage,
    sk20inch_projected_area,
)
from .sk_pmt_lookup import (
    SKPMTCoverageLookup,
    build_sk20inch_coverage_lookup,
    sk20inch_lookup_coordinates,
    sk20inch_lookup_coverage,
)
from .sk_pmt_sensor import compute_sk20inch_sensor_intersections

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
    # SK-like PMT propagation and smooth coverage
    'JAXSK20InchPMTHit',
    'intersect_sk20inch_pmt_jax',
    'SKPMTSurfaceQuadrature',
    'create_sk20inch_surface_quadrature',
    'sk20inch_gaussian_coverage',
    'sk20inch_projected_area',
    'SKPMTCoverageLookup',
    'build_sk20inch_coverage_lookup',
    'sk20inch_lookup_coordinates',
    'sk20inch_lookup_coverage',
    'compute_sk20inch_sensor_intersections',
]
