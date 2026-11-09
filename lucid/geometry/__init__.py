"""Detector geometries: the four registered types and the config loader.

``cylinder``, ``sphere``, ``box`` (``Cylinder``, ``Sphere``, ``Box``, exported here) and
``string`` (``lucid.geometry.string.StringTelescope``) register with
``@register_detector``; ``generate_detector(config)`` builds the type a geometry JSON's
``detector_type`` names. Also the sensor-pattern helpers.
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
    generate_detector,
)

# Registry
from .registry import get_detector_class, register_detector

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
    'generate_detector',

    # Registry
    'get_detector_class',
    'register_detector',
]