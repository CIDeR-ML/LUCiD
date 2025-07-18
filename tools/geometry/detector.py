"""
Functions for creating detectors from configuration files.
"""

import json
from .cylinder import Cylinder
from .sphere import Sphere
from .box import Box


def load_detector_config(file_path):
    """Function to load detector configuration from JSON file"""
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config


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