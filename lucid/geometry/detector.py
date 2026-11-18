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


def generate_detector(file_path):
    """Generate a detector from a JSON config file using the geometry registry.

    The config's ``detector_type`` (e.g. 'cylinder', 'sphere', 'box', 'string') is looked up in
    the registry, and that class's ``from_config`` builds the detector from
    ``geometry_definitions``. Relative file paths in it (``npz_file_path``) resolve against the
    config file's directory.
    """
    config = load_detector_config(file_path)
    cls = get_detector_class(config['detector_type'])
    config_dir = os.path.dirname(os.path.abspath(file_path))
    return cls.from_config(config['geometry_definitions'], config_dir)
