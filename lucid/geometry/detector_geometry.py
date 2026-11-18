"""DetectorGeometry container — aggregates geometry + propagator + medium."""
import os
from typing import NamedTuple, Optional, Callable

import jax.numpy as jnp

from lucid.geometry import get_detector_class, load_detector_config
from lucid.wavelength.medium import MediumProperties, load_medium
from lucid.propagation.shared import create_propagator as create_shared_propagator


class DetectorGeometry(NamedTuple):
    """Everything about the detector that is independent of the simulation mode.

    Built once via ``from_config()`` and reused across simulation configurations.
    """
    detector_type: str                          # as written in the geometry JSON, NOT normalised:
                                                # 'cylinder', 'sphere', 'box' or 'string'
    sensor_points: jnp.ndarray                  # (num_sensors, 3)
    sensor_radius: float
    num_sensors: int
    speed_of_light: float                       # m/ns in this medium
    medium: MediumProperties                    # material physics
    detector: object = None                     # the Detector instance
    propagator: Optional[Callable] = None       # JIT-compiled propagate_photons
    grid: Optional[NamedTuple] = None           # the propagator's frozen grid (surface detectors)

    @staticmethod
    def from_config(json_filename: str,
                    temperature: float = 0.2,
                    medium: Optional[MediumProperties] = None,
                    **grid_params) -> 'DetectorGeometry':
        """Build a DetectorGeometry from a config JSON file.

        Reads the JSON once: the detector class is the one registered under its
        ``detector_type``, the medium comes from its ``material`` (unless given), and the
        speed of light from the medium.

        Parameters
        ----------
        json_filename : str
            Path to detector geometry JSON.
        temperature : float or None
            Soft-assignment temperature for propagation. None uses step function.
        medium : MediumProperties, optional
            The loaded medium (a physics config's ``medium_model``); it must be the JSON's
            ``material``. None loads ``config/materials/<material>.json``.
        **grid_params
            Geometry-specific grid parameters forwarded to ``create_propagator()``.
            Cylinder: n_cap, n_angular, n_height.
            Sphere: n_divisions.
            Box: n_x, n_y, n_z.
            If not provided, auto-derived from detector geometry.
        """
        cfg = load_detector_config(json_filename)
        if medium is None:
            medium = load_medium(cfg['material'])
        elif medium.material != cfg['material']:
            # The geometry's material sets n, c and the emission processes; the physics config's
            # medium_model sets the optical curves and scintillation defaults: one medium.
            raise ValueError(f"{json_filename} is {cfg['material']!r} but the physics config's "
                             f"medium_model is {medium.material!r}")

        # Relative file paths in geometry_definitions resolve against the JSON's directory.
        detector = get_detector_class(cfg['detector_type']).from_config(
            cfg['geometry_definitions'], os.path.dirname(os.path.abspath(json_filename)))
        sensor_points = jnp.array(detector.all_points)
        sensor_radius = detector.S_radius
        num_sensors = len(sensor_points)

        # Propagator — string telescopes use the volume (per-DOM) propagator;
        # surface detectors (cylinder/sphere/box) use the shared grid propagator.
        from lucid.geometry.string import StringTelescope
        if isinstance(detector, StringTelescope):
            from lucid.propagation.string.string_propagator import create_string_propagator
            propagator = create_string_propagator(detector, sensor_radius,
                                                  temperature=temperature)
            grid = None
        else:
            propagator, grid = create_shared_propagator(
                detector, sensor_points, sensor_radius,
                temperature=temperature,
                **grid_params)

        return DetectorGeometry(
            detector_type=cfg['detector_type'],
            sensor_points=sensor_points,
            sensor_radius=sensor_radius,
            num_sensors=num_sensors,
            speed_of_light=medium.speed_of_light,
            medium=medium,
            detector=detector,
            propagator=propagator,
            grid=grid,
        )
