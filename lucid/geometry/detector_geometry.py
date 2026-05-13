"""DetectorGeometry container — aggregates geometry + propagator + medium."""
from typing import NamedTuple, Optional, Callable

import jax.numpy as jnp

from lucid.geometry import generate_detector, get_material_from_config
from lucid.wavelength.medium import MediumProperties, make_medium
from lucid.propagation.shared import create_propagator as create_shared_propagator
from lucid.geometry.string import StringTelescope


class DetectorGeometry(NamedTuple):
    """Everything about the detector that is independent of the simulation mode.

    Built once via ``from_config()`` and reused across different SimConfig /
    ParticleModel combinations.
    """
    detector_type: str                          # 'Cylinder', 'Sphere', 'Box'
    sensor_points: jnp.ndarray                  # (num_sensors, 3)
    sensor_radius: float
    num_sensors: int
    speed_of_light: float                       # m/ns in this medium
    medium: MediumProperties                    # material physics
    detector: object = None                     # the Detector instance
    propagator: Optional[Callable] = None       # JIT-compiled propagate_photons

    @staticmethod
    def from_config(json_filename: str,
                    temperature: float = 0.2,
                    max_sensors_per_cell: int = 4,
                    detector_type: str = 'Cylinder',
                    **grid_params) -> 'DetectorGeometry':
        """Build a DetectorGeometry from a config JSON file.

        This does everything the old inline code in ``setup_event_simulator``
        did: load config, create detector, build propagator, derive speed of
        light from the medium.

        Parameters
        ----------
        json_filename : str
            Path to detector geometry JSON.
        temperature : float or None
            Soft-assignment temperature for propagation. None uses step function.
        max_sensors_per_cell : int
            Grid cell sensor limit.
        detector_type : str
            'Cylinder', 'Sphere', or 'Box'.
        **grid_params
            Geometry-specific grid parameters forwarded to ``create_propagator()``.
            Cylinder: n_cap, n_angular, n_height.
            Sphere: n_divisions.
            Box: n_x, n_y, n_z.
            If not provided, auto-derived from detector geometry.
        """
        # Normalize casing
        dt_key = detector_type.lower()
        valid_types = ('cylinder', 'sphere', 'box', 'string')
        if dt_key not in valid_types:
            raise ValueError(f"detector_type must be one of {valid_types}, got {detector_type}")

        # Material
        material = get_material_from_config(json_filename)
        medium = make_medium(material)

        # Geometry
        detector = generate_detector(json_filename)
        sensor_points = jnp.array(detector.all_points)
        sensor_radius = detector.S_radius
        num_sensors = len(sensor_points)

        # Propagator — string uses its own factory, others use the shared one
        if isinstance(detector, StringTelescope):
            from lucid.propagation.string.propagator import create_string_propagator
            propagator = create_string_propagator(
                detector, sensor_points, sensor_radius,
                temperature=temperature,
                lambda_abs=grid_params.pop('lambda_abs', 100.0),
                lambda_scat=grid_params.pop('lambda_scat', 30.0),
                max_str_per_cell=grid_params.pop('max_str_per_cell', 6),
            )
        else:
            propagator = create_shared_propagator(
                detector, sensor_points, sensor_radius,
                temperature=temperature,
                max_sensors_per_cell=max_sensors_per_cell,
                **grid_params)

        return DetectorGeometry(
            detector_type=detector_type,
            sensor_points=sensor_points,
            sensor_radius=sensor_radius,
            num_sensors=num_sensors,
            speed_of_light=medium.speed_of_light,
            medium=medium,
            detector=detector,
            propagator=propagator,
        )
