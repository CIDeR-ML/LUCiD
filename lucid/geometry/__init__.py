"""Detector geometries: the four registered types and the config loader.

``cylinder``, ``sphere``, ``box`` (``Cylinder``, ``Sphere``, ``Box``, exported here) and
``string`` (``lucid.geometry.string.StringTelescope``) register with
``@register_detector``; ``generate_detector(config)`` builds the type a geometry JSON's
``detector_type`` names. Also the sensor-pattern helpers.
"""

# Import utility functions
# Import detector classes
from .base import Detector
from .box import Box
from .cylinder import Cylinder

# Import detector functions
from .detector import (
    generate_detector,
    load_detector_config,
)

# Registry
from .registry import get_detector_class, register_detector
from .sphere import Sphere
from .utils import (
    calculate_surface_normals,
    create_disc_mesh,
    fibonacci_sphere_points_numpy,
    generate_concentric_hexagons,
)

__all__ = [
    # Utility functions
    "generate_concentric_hexagons",
    "fibonacci_sphere_points_numpy",
    "create_disc_mesh",
    "calculate_surface_normals",
    # Detector classes
    "Detector",
    "Cylinder",
    "Sphere",
    "Box",
    # Detector functions
    "load_detector_config",
    "generate_detector",
    # Registry
    "get_detector_class",
    "register_detector",
]
