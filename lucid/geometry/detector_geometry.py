"""DetectorGeometry container — aggregates geometry + propagator + medium."""
from typing import NamedTuple, Optional, Callable

import jax.numpy as jnp

from lucid.geometry import generate_detector, get_material_from_config
from lucid.wavelength.medium import MediumProperties, make_medium
from lucid.propagation.shared import create_propagator as create_shared_propagator


class DetectorGeometry(NamedTuple):
    """Everything about the detector that is independent of the simulation mode.

    Built once via ``from_config()`` and reused across different SimConfig /
    ParticleModel combinations.
    """
    detector_type: str                          # 'Cylinder', 'Sphere', 'Box'
    sensor_points: jnp.ndarray                  # (num_sensors, 3)
    sensor_radius: float
    num_sensors: int
    speed_of_light: float                       # m/ns in the (inner) medium
    medium: MediumProperties                    # material physics (inner medium for nested)
    detector: object = None                     # the Detector instance
    propagator: Optional[Callable] = None       # JIT-compiled propagate_photons
    # --- Two-medium (nested) extension. Appended LAST → single-medium construction
    #     is unchanged; these stay None / defaults for every single-medium detector. ---
    is_nested: bool = False                      # True for nested_sphere (two media + interface)
    medium_outer: Optional[MediumProperties] = None   # outer (buffer/water) medium
    speed_of_light_outer: Optional[float] = None      # m/ns in the outer medium
    r_inner: Optional[float] = None                   # interface radius (m)
    r_outer: Optional[float] = None                   # outer/PMT radius (m)

    @staticmethod
    def from_config(json_filename: str,
                    temperature: float = 0.2,
                    max_candidates_per_ray: int = 4,
                    detector_type: str = 'Cylinder',
                    overlap_st_width_frac: float = 0.35,
                    overlap_renorm: float = 1.0,
                    overlap_mode: str = 'interp',
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
        max_candidates_per_ray : int
            Grid cell sensor limit.
        detector_type : str
            'Cylinder', 'Sphere', or 'Box'.
        overlap_st_width_frac : float
            Straight-through overlap surrogate width (fraction of r); default 0.35.
        overlap_renorm : float
            Soft-overlap renormalization constant C; default 1.0 = OFF.
        overlap_mode : str
            Soft-overlap lookup interpolation: 'interp' (default) or 'cubic'.
        **grid_params
            Geometry-specific grid parameters forwarded to ``create_propagator()``.
            Cylinder: n_cap, n_angular, n_height.
            Sphere: n_divisions.
            Box: n_x, n_y, n_z.
            If not provided, auto-derived from detector geometry.
        """
        # Normalize casing. 'string' = telescope / volume detector (IceCube-style);
        # 'nested_sphere' = two concentric media (inner LS + outer water) with an interface.
        dt_key = detector_type.lower()
        if dt_key not in ('cylinder', 'sphere', 'box', 'string', 'nested_sphere'):
            raise ValueError(
                f"detector_type must be 'cylinder', 'sphere', 'box', 'string', or "
                f"'nested_sphere', got {detector_type}")

        # Material
        material = get_material_from_config(json_filename)
        medium = make_medium(material)

        # Geometry — the actual class is dispatched from the JSON's detector_type
        # (the detector_type arg may be the caller default 'Cylinder').
        detector = generate_detector(json_filename)
        import json as _json
        with open(json_filename) as _f:
            _cfg = _json.load(_f)
        actual_type = _cfg.get('detector_type', detector_type)
        sensor_points = jnp.array(detector.all_points)
        sensor_radius = detector.S_radius
        num_sensors = len(sensor_points)

        # ── Nested two-sphere detector: inner + outer media, interface propagator ──
        from lucid.geometry.nested_sphere import NestedSphere
        if isinstance(detector, NestedSphere):
            inner_material = _cfg.get('inner_material', material)
            outer_material = _cfg.get('outer_material', 'water')
            medium_inner = make_medium(inner_material)
            medium_outer = make_medium(outer_material)
            grid_params.setdefault('max_candidates_per_ray', max_candidates_per_ray)
            detector.configure_grid(**grid_params)
            from lucid.propagation.nested_sphere import create_nested_sphere_propagator
            propagator = create_nested_sphere_propagator(
                sensor_points, sensor_radius,
                detector.r_inner, detector.r_outer, detector._n_divisions,
                temperature=temperature, max_candidates_per_ray=max_candidates_per_ray,
                overlap_st_width_frac=overlap_st_width_frac,
                overlap_renorm=overlap_renorm, overlap_mode=overlap_mode)
            return DetectorGeometry(
                detector_type=actual_type, sensor_points=sensor_points,
                sensor_radius=sensor_radius, num_sensors=num_sensors,
                speed_of_light=medium_inner.speed_of_light, medium=medium_inner,
                detector=detector, propagator=propagator,
                is_nested=True, medium_outer=medium_outer,
                speed_of_light_outer=medium_outer.speed_of_light,
                r_inner=detector.r_inner, r_outer=detector.r_outer)

        # Propagator — string telescopes use the volume (per-DOM) propagator;
        # surface detectors (cylinder/sphere/box) use the shared grid propagator.
        from lucid.geometry.string import StringTelescope
        if isinstance(detector, StringTelescope):
            from lucid.propagation.string.string_propagator import create_string_propagator
            propagator = create_string_propagator(
                detector, sensor_radius, temperature=temperature,
                n_closest=max(1, max_candidates_per_ray // 2))
        else:
            propagator = create_shared_propagator(
                detector, sensor_points, sensor_radius,
                temperature=temperature,
                max_candidates_per_ray=max_candidates_per_ray,
                overlap_st_width_frac=overlap_st_width_frac,
                overlap_renorm=overlap_renorm,
                overlap_mode=overlap_mode,
                **grid_params)

        return DetectorGeometry(
            detector_type=actual_type,
            sensor_points=sensor_points,
            sensor_radius=sensor_radius,
            num_sensors=num_sensors,
            speed_of_light=medium.speed_of_light,
            medium=medium,
            detector=detector,
            propagator=propagator,
        )
