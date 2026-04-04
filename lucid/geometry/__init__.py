"""
LUCiD geometry module - detector geometries and utilities.

This module provides detector geometry classes (Cylinder, Sphere, Box) and utility functions
for sensor pattern generation and visualization.
"""

# Import utility functions
from .utils import (
    generate_concentric_hexagons,
    fibonacci_sphere_points_numpy,
    create_disc_mesh,
    calculate_surface_normals
)

# Import detector classes
from .base import Detector
from .cylinder import Cylinder
from .sphere import Sphere
from .box import Box

# Import detector functions
from .detector import (
    load_detector_config,
    load_detector_geom,
    generate_detector,
    get_material_from_config
)

# Registry
from .registry import get_detector_class, register_detector, list_detector_types

# Export all public functions and classes for backward compatibility
__all__ = [
    # Utility functions
    'generate_concentric_hexagons',
    'fibonacci_sphere_points_numpy', 
    'create_disc_mesh',
    'calculate_surface_normals',
    
    # Detector classes
    'Detector',
    'Cylinder',
    'Sphere', 
    'Box',
    
    # Detector functions
    'load_detector_config',
    'load_detector_geom',
    'generate_detector',
    'get_material_from_config'
]