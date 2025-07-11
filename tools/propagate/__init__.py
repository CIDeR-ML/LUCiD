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

# Import specific functions for direct access
from .cylinder import create_photon_propagator as create_cylinder_propagator, cylinder_bounds_check
from .sphere import create_sphere_photon_propagator, sphere_bounds_check  
from .box import create_box_photon_propagator, box_bounds_check

# Main propagation function - unified interface
def create_photon_propagator(detector_type, sensor_positions, sensor_radius, **detector_params):
    """
    Unified interface for creating photon propagators for different detector geometries.
    
    Parameters:
    -----------
    detector_type : str
        Type of detector: 'cylinder', 'sphere', or 'box'
    sensor_positions : array
        Sensor positions
    sensor_radius : float
        Sensor radius
    **detector_params : dict
        Detector-specific parameters
        
    Returns:
    --------
    callable
        JIT-compiled photon propagation function
    """
    if detector_type.lower() == 'cylinder':
        return cylinder.create_photon_propagator(sensor_positions, sensor_radius, **detector_params)
    elif detector_type.lower() == 'sphere':
        return sphere.create_sphere_photon_propagator(sensor_positions, sensor_radius, **detector_params)
    elif detector_type.lower() == 'box':
        return box.create_box_photon_propagator(sensor_positions, sensor_radius, **detector_params)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


# Export key functions for backward compatibility
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
    
    # Specific propagator functions
    'create_cylinder_propagator',
    'create_sphere_photon_propagator', 
    'create_box_photon_propagator',
    
    # Bounds check functions
    'cylinder_bounds_check',
    'sphere_bounds_check',
    'box_bounds_check',
    
    # Unified interface
    'create_photon_propagator'
]