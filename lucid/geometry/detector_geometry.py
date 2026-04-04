"""DetectorGeometry container — aggregates geometry + propagator + medium."""
from typing import NamedTuple, Optional, Callable

import jax.numpy as jnp

from lucid.geometry import generate_detector, get_material_from_config
from lucid.wavelength.medium import MediumProperties, make_medium, load_qe_curve
from lucid.propagation.cylinder import create_photon_propagator
from lucid.propagation.sphere import create_sphere_photon_propagator
from lucid.propagation.box import create_box_photon_propagator


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
    qe_curve: Optional[Callable] = None         # PMT spectral response fn
    detector: object = None                     # the Detector instance
    propagator: Optional[Callable] = None       # JIT-compiled propagate_photons

    @staticmethod
    def from_config(json_filename: str,
                    temperature: float = 0.2,
                    max_sensors_per_cell: int = 4,
                    detector_type: str = 'Cylinder') -> 'DetectorGeometry':
        """Build a DetectorGeometry from a config JSON file.

        This does everything the old inline code in ``setup_event_simulator``
        did: load config, create detector, build propagator, derive speed of
        light from the medium.

        Parameters
        ----------
        json_filename : str
            Path to detector geometry JSON.
        temperature : float
            Soft-assignment temperature for propagation.
        max_sensors_per_cell : int
            Grid cell sensor limit.
        detector_type : str
            'Cylinder', 'Sphere', or 'Box'.
        """
        if detector_type not in ('Cylinder', 'Sphere', 'Box'):
            raise ValueError(f"detector_type must be 'Cylinder', 'Sphere', or 'Box', got {detector_type}")

        # Material
        material = get_material_from_config(json_filename)
        medium = make_medium(material)

        # Geometry
        detector = generate_detector(json_filename)
        sensor_points = jnp.array(detector.all_points)
        sensor_radius = detector.S_radius
        num_sensors = len(sensor_points)

        # Propagator
        if detector_type == 'Cylinder':
            propagator = create_photon_propagator(
                sensor_points, sensor_radius,
                r=detector.r, h=detector.H,
                temperature=temperature,
                max_sensors_per_cell=max_sensors_per_cell)
        elif detector_type == 'Sphere':
            propagator = create_sphere_photon_propagator(
                sensor_points, sensor_radius,
                sphere_radius=detector.r,
                temperature=temperature,
                n_divisions=100,
                max_sensors_per_cell=max_sensors_per_cell)
        elif detector_type == 'Box':
            propagator = create_box_photon_propagator(
                sensor_points, sensor_radius,
                length=detector.L, width=detector.W, height=detector.H,
                temperature=temperature,
                max_sensors_per_cell=max_sensors_per_cell)

        # QE curve (try to load, None if not available)
        try:
            qe_fn = load_qe_curve()
        except FileNotFoundError:
            qe_fn = None

        return DetectorGeometry(
            detector_type=detector_type,
            sensor_points=sensor_points,
            sensor_radius=sensor_radius,
            num_sensors=num_sensors,
            speed_of_light=medium.speed_of_light,
            medium=medium,
            qe_curve=qe_fn,
            detector=detector,
            propagator=propagator,
        )
