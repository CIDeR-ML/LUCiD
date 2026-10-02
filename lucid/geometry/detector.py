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
from .nested_sphere import NestedSphere  # noqa: F401  (two-medium nested spheres)
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


def generate_detector(file_path, pmt_geometry_enabled=None,
                      support_geometry_enabled=None,
                      chimney_geometry_enabled=None,
                      bottom_chimney_geometry_enabled=None,
                      acrylic_bowl_geometry_enabled=None,
                      diffuser_hardware_enabled=None):
    """Generate a detector from a JSON config file using the geometry registry.

    The detector type in the config (e.g. 'cylinder', 'sphere', 'box') is
    looked up in the registry to find the appropriate class.

    For ``detector_type: cylinder``, two construction paths are supported:

      * ``geometry_definitions`` carries ``npz_file_path`` →
        :meth:`Cylinder.from_pmt_file` loads measured PMT positions
        from that file (used for SK, HK, WCTE, SK_official, etc.).
      * Otherwise → algorithmic placement using
        ``radius / height / n_sensors / sensor_radius``.

    For ``detector_type: nested_sphere``, ``pmt_file_path`` points to the
    JUNO PMT CSV and is resolved relative to the config-file directory.
    The NPZ path used by cylinder configurations is also resolved relative
    to the config-file directory. ``pmt_geometry_enabled`` optionally
    overrides only the configured explicit PMT intersection geometry, for a
    controlled geometry on/off ablation. ``support_geometry_enabled`` does
    the same independently for the effective JUNO support rods.  The
    chimney, Acrylic-bowl, and source-attached diffuser-hardware overrides are
    independent in the same way.
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
    elif cls is NestedSphere:
        if 'pmt_file_path' in geom_def:
            config_dir = os.path.dirname(os.path.abspath(file_path))
            csv_path = os.path.join(config_dir, geom_def['pmt_file_path'])
            configured_geometry = geom_def.get('pmt_geometry')
            if pmt_geometry_enabled is None:
                pmt_geometry = configured_geometry
            elif pmt_geometry_enabled:
                if configured_geometry is None:
                    raise ValueError(
                        "PMT geometry was enabled but none is configured"
                    )
                pmt_geometry = configured_geometry
            else:
                pmt_geometry = None
            type_path = (
                geom_def.get('pmt_type_file_path')
                if pmt_geometry is not None else None
            )
            if type_path is not None:
                type_path = os.path.join(config_dir, type_path)
            configured_support = geom_def.get('support_rod_geometry')
            if support_geometry_enabled is None:
                support_geometry = configured_support
            elif support_geometry_enabled:
                if configured_support is None:
                    raise ValueError(
                        "Support geometry was enabled but none is configured"
                    )
                support_geometry = configured_support
            else:
                support_geometry = None
            support_path = (
                geom_def.get('support_rod_file_path')
                if support_geometry is not None else None
            )
            if support_path is not None:
                support_path = os.path.join(config_dir, support_path)
            configured_chimney = geom_def.get('chimney_geometry')
            if chimney_geometry_enabled is None:
                chimney_geometry = configured_chimney
            elif chimney_geometry_enabled:
                if configured_chimney is None:
                    raise ValueError(
                        "Chimney geometry was enabled but none is configured"
                    )
                chimney_geometry = configured_chimney
            else:
                chimney_geometry = None
            configured_bottom_chimney = geom_def.get(
                'bottom_chimney_geometry'
            )
            if bottom_chimney_geometry_enabled is None:
                bottom_chimney_geometry = configured_bottom_chimney
            elif bottom_chimney_geometry_enabled:
                if configured_bottom_chimney is None:
                    raise ValueError(
                        "Bottom-chimney geometry was enabled but none is "
                        "configured"
                    )
                bottom_chimney_geometry = configured_bottom_chimney
            else:
                bottom_chimney_geometry = None
            configured_bowls = geom_def.get('acrylic_bowl_geometry')
            if acrylic_bowl_geometry_enabled is None:
                acrylic_bowl_geometry = configured_bowls
            elif acrylic_bowl_geometry_enabled:
                if configured_bowls is None:
                    raise ValueError(
                        "Acrylic-bowl geometry was enabled but none is "
                        "configured"
                    )
                acrylic_bowl_geometry = configured_bowls
            else:
                acrylic_bowl_geometry = None
            bowl_path = (
                geom_def.get('acrylic_bowl_file_path')
                if acrylic_bowl_geometry is not None else None
            )
            if bowl_path is not None:
                bowl_path = os.path.join(config_dir, bowl_path)
            configured_diffuser = geom_def.get(
                'diffuser_hardware_geometry'
            )
            if diffuser_hardware_enabled is None:
                diffuser_hardware_geometry = configured_diffuser
            elif diffuser_hardware_enabled:
                if configured_diffuser is None:
                    raise ValueError(
                        "Diffuser hardware was enabled but none is configured"
                    )
                diffuser_hardware_geometry = configured_diffuser
            else:
                diffuser_hardware_geometry = None
            return cls.from_pmt_file(
                csv_path,
                geom_def['inner_radius'],
                geom_def['outer_radius'],
                geom_def['sensor_radius'],
                acrylic_radius=geom_def.get('acrylic_radius'),
                pmt_type_file_path=type_path,
                pmt_geometry=pmt_geometry,
                support_rod_file_path=support_path,
                support_geometry=support_geometry,
                chimney_geometry=chimney_geometry,
                bottom_chimney_geometry=bottom_chimney_geometry,
                acrylic_bowl_file_path=bowl_path,
                acrylic_bowl_geometry=acrylic_bowl_geometry,
                diffuser_hardware_geometry=diffuser_hardware_geometry)
        return cls(geom_def['inner_radius'], geom_def['outer_radius'],
                   geom_def['n_sensors'], geom_def['sensor_radius'],
                   acrylic_radius=geom_def.get('acrylic_radius'))
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
