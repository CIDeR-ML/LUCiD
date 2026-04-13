"""
Functions for creating detectors from configuration files.
"""

import json
import os
from .registry import get_detector_class
# Import subclasses so their @register_detector decorators run
from .cylinder import Cylinder  # noqa: F401
from .sphere import Sphere      # noqa: F401
from .box import Box            # noqa: F401
from .superk import SuperK      # noqa: F401


def load_detector_config(file_path):
    """
    Load detector configuration from JSON file.

    Parameters
    ----------
    file_path : str
        Path to the detector configuration JSON file

    Returns
    -------
    dict
        Detector configuration dictionary

    Raises
    ------
    ValueError
        If required fields are missing from the configuration
    """
    with open(file_path, 'r') as file:
        config = json.load(file)

    # Validate required fields
    if 'detector_type' not in config:
        raise ValueError(f"Detector config {file_path} missing required field 'detector_type'")

    if 'material' not in config:
        raise ValueError(
            f"Detector config {file_path} missing required field 'material'.\n"
            f"Please add '\"material\": \"water\"' to the configuration file."
        )

    if 'geometry_definitions' not in config:
        raise ValueError(f"Detector config {file_path} missing required field 'geometry_definitions'")

    return config


def get_material_from_config(file_path):
    """
    Get the material property from detector configuration.

    Parameters
    ----------
    file_path : str
        Path to the detector configuration JSON file

    Returns
    -------
    str
        Material type (e.g., 'water', 'ice')
    """
    config = load_detector_config(file_path)
    return config['material']


def load_detector_geom(file_path):
    """Load detector geometry from JSON config"""
    config = load_detector_config(file_path)
    
    detector_type = config['detector_type']
    geom_def = config['geometry_definitions']
    
    if detector_type == 'cylinder':
        return (detector_type, geom_def['radius'], geom_def['height'], 
                geom_def['n_sensors'], geom_def['sensor_radius'])
    elif detector_type == 'sphere':
        return (detector_type, geom_def['radius'], None, 
                geom_def['n_sensors'], geom_def['sensor_radius'])
    elif detector_type == 'box':
        return (detector_type, geom_def['length'], geom_def['width'], 
                geom_def['height'], geom_def['n_sensors'], geom_def['sensor_radius'])
    elif detector_type == 'superk':
        return (detector_type, geom_def['radius'], geom_def['height'],
                geom_def['n_sensors'], geom_def['sensor_radius'],
                geom_def['connection_table_path'])
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


def generate_detector(file_path):
    """Generate a detector from a JSON config file using the geometry registry.

    The detector type in the config (e.g. 'cylinder', 'sphere', 'box', 'superk')
    is looked up in the registry to find the appropriate class. Each class
    knows how to construct itself from the geometry_definitions dict.
    """
    config = load_detector_config(file_path)
    detector_type = config['detector_type']
    geom_def = config['geometry_definitions']

    cls = get_detector_class(detector_type)

    # Dispatch construction based on class — each geometry has different __init__ args
    if cls is Cylinder:
        return cls(geom_def['radius'], geom_def['height'],
                   geom_def['n_sensors'], geom_def['sensor_radius'])
    elif cls is Sphere:
        return cls(geom_def['radius'], geom_def['n_sensors'], geom_def['sensor_radius'])
    elif cls is Box:
        return cls(geom_def['length'], geom_def['width'], geom_def['height'],
                   geom_def['n_sensors'], geom_def['sensor_radius'])
    elif cls is SuperK:
        # Resolve connection_table_path relative to the config file directory
        config_dir = os.path.dirname(os.path.abspath(file_path))
        ct_path = os.path.join(config_dir, geom_def['connection_table_path'])
        return cls(ct_path, radius=geom_def['radius'],
                   height=geom_def['height'], n_sensors=geom_def['n_sensors'],
                   sensor_radius=geom_def['sensor_radius'])
    else:
        # Future-proof: class was registered but we don't know its constructor.
        # This shouldn't happen with the current set of geometries.
        raise NotImplementedError(
            f"No construction logic for registered detector class {cls.__name__}")