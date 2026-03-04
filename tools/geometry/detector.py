"""
Functions for creating detectors from configuration files.
"""

import json
from .cylinder import Cylinder
from .sphere import Sphere
from .box import Box
from .superk import SuperK


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
    """Function to generate detector from json config"""
    geom_data = load_detector_geom(file_path)
    detector_type = geom_data[0]
    
    if detector_type == 'cylinder':
        _, radius, height, n_sensors, sensor_radius = geom_data
        return Cylinder(radius, height, n_sensors, sensor_radius)
    elif detector_type == 'sphere':
        _, radius, _, n_sensors, sensor_radius = geom_data
        return Sphere(radius, n_sensors, sensor_radius)
    elif detector_type == 'box':
        _, length, width, height, n_sensors, sensor_radius = geom_data
        return Box(length, width, height, n_sensors, sensor_radius)
    elif detector_type == 'superk':
        _, radius, height, n_sensors, sensor_radius, connection_table_path = geom_data
        return SuperK(connection_table_path, radius=radius, height=height,
                      n_sensors=n_sensors, sensor_radius=sensor_radius)