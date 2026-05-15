"""
LUCiD geometry module - detector geometries and utilities.

This module provides detector geometry classes (Cylinder, Sphere, Box) and utility functions
for sensor pattern generation and visualization.
"""

from .utils import (
    generate_concentric_hexagons,
    fibonacci_sphere_points_numpy,
    create_disc_mesh,
    calculate_surface_normals
)

from .base import Detector
from .cylinder import Cylinder
from .sphere import Sphere
from .box import Box

from .detector import (
    load_detector_config,
    load_detector_geom,
    generate_detector,
    get_material_from_config
)

from .registry import get_detector_class, register_detector, list_detector_types

__all__ = [
    'generate_concentric_hexagons',
    'fibonacci_sphere_points_numpy',
    'create_disc_mesh',
    'calculate_surface_normals',

    'Detector',
    'Cylinder',
    'Sphere',
    'Box',

    'load_detector_config',
    'load_detector_geom',
    'generate_detector',
    'get_material_from_config',

    'get_detector_class',
    'register_detector',
    'list_detector_types',
]
