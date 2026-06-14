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
from .string import StringTelescope  # noqa: F401  (telescope / volume detectors)


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
    """Load detector geometry from JSON config.

    Returns a tuple identifying the detector type and its geometry
    parameters. For cylinders the second element distinguishes the
    two construction paths:
        ('cylinder', 'algorithmic', radius, height, n_sensors, sensor_radius)
        ('cylinder', 'pmt_file',    npz_file_path)
    """
    config = load_detector_config(file_path)

    detector_type = config['detector_type']
    geom_def = config['geometry_definitions']

    if detector_type == 'cylinder':
        if 'npz_file_path' in geom_def:
            return (detector_type, 'pmt_file', geom_def['npz_file_path'])
        return (detector_type, 'algorithmic',
                geom_def['radius'], geom_def['height'],
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
    """Generate a detector from a JSON config file using the geometry registry.

    The detector type in the config (e.g. 'cylinder', 'sphere', 'box') is
    looked up in the registry to find the appropriate class.

    For ``detector_type: cylinder``, two construction paths are supported:

      * ``geometry_definitions`` carries ``npz_file_path`` →
        :meth:`Cylinder.from_pmt_file` loads measured PMT positions
        from that file (used for SK, HK, WCTE, SK_official, etc.).
      * Otherwise → algorithmic placement using
        ``radius / height / n_sensors / sensor_radius``.

    The npz path is resolved relative to the config-file directory.
    """
    config = load_detector_config(file_path)
    detector_type = config['detector_type']
    geom_def = config['geometry_definitions']

    cls = get_detector_class(detector_type)

    if cls is Cylinder:
        if 'npz_file_path' in geom_def:
            config_dir = os.path.dirname(os.path.abspath(file_path))
            npz_path = os.path.join(config_dir, geom_def['npz_file_path'])
            return cls.from_pmt_file(npz_path)
        return cls(geom_def['radius'], geom_def['height'],
                   geom_def['n_sensors'], geom_def['sensor_radius'])
    elif cls is Sphere:
        return cls(geom_def['radius'], geom_def['n_sensors'], geom_def['sensor_radius'])
    elif cls is Box:
        return cls(geom_def['length'], geom_def['width'], geom_def['height'],
                   geom_def['n_sensors'], geom_def['sensor_radius'])
    elif cls is StringTelescope:
        # Telescope geometry: per-DOM positions live in an NPZ next to the config.
        return cls.from_config(file_path)
    else:
        # Future-proof: class was registered but we don't know its constructor.
        raise NotImplementedError(
            f"No construction logic for registered detector class {cls.__name__}")
