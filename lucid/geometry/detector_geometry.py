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
    detector_type: str                          # as written in the geometry JSON, NOT normalised:
                                                # 'cylinder', 'sphere', 'box' or 'string'
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
                    max_candidates_per_ray: int = 4,
                    detector_type: str = 'Cylinder',
                    overlap_st_width_frac: float = 0.35,
                    overlap_renorm: float = 1.0,
                    overlap_mode: str = 'interp',
                    sensor_shape: str = 'sphere',
                    sk_pmt_lookup=None,
                    sk_pmt_lookup_options=None,
                    sensor_candidate_selection: str = 'all',
                    sensor_wall_assignment: str = 'nearest',
                    pmt_gap_recovery: bool = False,
                    pmt_gap_recovery_hole_radius: float = 0.285,
                    sk_acrylic_optics=None,
                    deposit_leg_bound: bool = False,
                    **grid_params) -> 'DetectorGeometry':
        """Build a DetectorGeometry from a config JSON file.

        Load the detector layout and material, build the photon propagator,
        and derive the reference propagation speed from the medium. Lookup,
        wall-routing, gap-recovery, and acrylic options apply to surface
        detectors; string telescopes use their own volume propagator.

        Parameters
        ----------
        json_filename : str
            Path to detector geometry JSON.
        temperature : float or None
            Dimensionless overlap smoothing scale; default 0.2. For
            ``sensor_shape='sk20inch'``, a positive value gives a Gaussian
            ray-bundle width of ``temperature * sensor_radius`` in metres
            and uses a generated coverage lookup. ``None`` uses hard
            central-ray intersections without that coverage lookup.
        max_candidates_per_ray : int
            Maximum number of candidate sensors retained per surface grid
            cell; default 4. A string telescope instead uses
            ``max(1, max_candidates_per_ray // 2)`` nearest sensors.
        detector_type : str
            Accepted values are 'cylinder', 'superk', 'sphere', 'box', and
            'string', case-insensitive. Defaults to 'Cylinder'. This argument
            is validated, but the JSON's ``detector_type`` selects the
            detector class. The SuperK layout uses cylindrical propagation.
        overlap_st_width_frac : float
            Straight-through overlap surrogate width (fraction of r); default 0.35.
        overlap_renorm : float
            Soft-overlap renormalization constant C; default 1.0 = OFF.
        overlap_mode : str
            Soft-overlap lookup interpolation: 'interp' (default) or 'cubic'.
        sensor_shape : {'sphere', 'sk20inch'}
            PMT interception geometry; default 'sphere'. ``'sk20inch'`` uses
            a sphere-plus-torus bulb with an inactive barrel band and requires
            measured ``pmt_directions`` and per-PMT ``surfaces`` labels
            ('barrel', 'top', or 'bottom') on the loaded detector.
        sk_pmt_lookup : SKPMTCoverageLookup or None
            Optional prebuilt Gaussian-bundle coverage table from
            :func:`lucid.propagation.sk_pmt_lookup.build_sk20inch_coverage_lookup`.
            It stores active-photocathode coverage versus incidence cosine
            and transverse offsets, generated from the PMT geometry rather
            than an optical calibration table. Used only for 'sk20inch' PMTs
            when ``temperature`` is not None. Its ``sigma`` must match
            ``temperature * sensor_radius``. The default None builds or loads
            a cached table automatically; supplying one allows reuse across
            propagators. Ignored for spherical PMTs or hard intersections.
        sk_pmt_lookup_options : dict or None
            Options for automatic coverage-table construction when
            ``sk_pmt_lookup`` is None and smooth 'sk20inch' coverage is active.
            Accepted keys are ``n_incidence`` (default 31 angular samples),
            ``grid_spacing`` (metres; default max(sigma / 4, 0.002)),
            ``tail_sigma`` (default 5.0), ``raster_oversample`` (default 4),
            ``cache`` (default True), and ``cache_dir`` (default None, using
            the standard overlap-cache directory). Smaller grid spacing or
            more samples increase construction cost and storage. Do not set
            ``sigma`` here: it is determined by temperature and sensor radius.
            The default None uses builder defaults. Ignored when a prebuilt
            lookup is supplied or smooth coverage is inactive.
        sensor_candidate_selection : {'all', 'wall'}
            Candidate routing rule; default 'all', which evaluates the grid
            cell's candidate sensors. ``'wall'`` assigns a nominal wall PMT
            before applying the curved geometry and requires 'sk20inch' PMTs.
            With finite temperature, the hard forward assignment retains a
            smooth all-candidate coverage surrogate for gradients.
        sensor_wall_assignment : {'nearest', 'sk'}
            Rule used by 'wall' routing; default 'nearest', which chooses the
            candidate PMT closest to the nominal wall crossing. ``'sk'`` uses
            SK-like barrel/cap logical cell ownership, including empty cells
            and seam handling. It requires 'wall' routing and
            ``temperature=None``.
        pmt_gap_recovery : bool
            Default False. If True, also test the full PMT sphere for rays
            passing through the inner opening behind the blacksheet plane.
            Requires 'sk20inch' PMTs and 'wall' routing.
        pmt_gap_recovery_hole_radius : float
            Positive radius in metres of the inner water-gap region eligible
            for recovery; default 0.285. Used only when ``pmt_gap_recovery``
            is True. This geometric acceptance radius does not set the
            acrylic optical properties or a blacksheet reflection mask.
        sk_acrylic_optics : SKAcrylicOptics or None
            Precomputed optical properties for one wavelength, as returned
            by :func:`lucid.propagation.sk_acrylic.load_sk4_acrylic_optics`
            from a user-supplied table. Includes acrylic refractive index and
            absorption coefficient, plus water phase and group indices.
            Default None disables cover transport. Supplying optics enables
            refraction, transmission, absorption, and material timing through
            the spherical cover; requires 'sk20inch' PMTs and 'wall' routing.
            This argument takes optical values, not a table filename. The
            simulator configures how reflected cover paths are propagated.
        deposit_leg_bound : bool
            Bound the deposit to the travelled leg rather than the unbounded ray line. Default
            False (unbounded ray line). Surface detectors only: a string telescope uses its own
            propagator and deposit, and ignores it. See
            :func:`lucid.propagation.shared.create_propagator`.
        **grid_params
            Geometry-specific grid parameters forwarded to ``create_propagator()``.
            Cylinder: n_cap, n_angular, n_height.
            Sphere: n_divisions.
            Box: n_x, n_y, n_z.
            If not provided, auto-derived from detector geometry.

        Returns
        -------
        DetectorGeometry
            Detector layout, sensor arrays, material properties, reference
            speed in m/ns, and the constructed propagation callable. Optical
            and response parameters fitted per simulation are supplied
            separately through ``DetectorParams``.
        """
        if sensor_shape not in ('sphere', 'sk20inch'):
            raise ValueError("sensor_shape must be 'sphere' or 'sk20inch'")

        # Normalize casing. 'string' = telescope / volume detector (IceCube-style).
        dt_key = detector_type.lower()
        if dt_key == 'superk':
            dt_key = 'cylinder'
        if dt_key not in ('cylinder', 'sphere', 'box', 'string'):
            raise ValueError(
                f"detector_type must be 'cylinder', 'superk', 'sphere', "
                f"'box', or 'string', got {detector_type}")

        # Material
        material = get_material_from_config(json_filename)
        medium = make_medium(material)

        # Geometry — the actual class is dispatched from the JSON's detector_type
        # (the detector_type arg may be the caller default 'Cylinder').
        detector = generate_detector(json_filename)
        import json as _json
        with open(json_filename) as _f:
            configured_type = _json.load(_f).get('detector_type', detector_type)
        # Specialized layouts such as SuperK still use a standard geometric
        # family for containment and downstream event metadata.
        actual_type = str(
            getattr(detector, 'geometry_type', configured_type)).capitalize()
        sensor_points = jnp.array(detector.all_points)
        sensor_radius = detector.S_radius
        num_sensors = len(sensor_points)

        # Propagator — string telescopes use the volume (per-DOM) propagator;
        # surface detectors (cylinder/sphere/box) use the shared grid propagator.
        from lucid.geometry.string import StringTelescope
        if isinstance(detector, StringTelescope):
            if sensor_shape != 'sphere':
                raise ValueError(
                    "sensor_shape='sk20inch' is only available for measured "
                    "surface-detector PMTs")
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
                sensor_shape=sensor_shape,
                sk_pmt_lookup=sk_pmt_lookup,
                sk_pmt_lookup_options=sk_pmt_lookup_options,
                sensor_candidate_selection=sensor_candidate_selection,
                sensor_wall_assignment=sensor_wall_assignment,
                pmt_gap_recovery=pmt_gap_recovery,
                pmt_gap_recovery_hole_radius=pmt_gap_recovery_hole_radius,
                sk_acrylic_optics=sk_acrylic_optics,
                deposit_leg_bound=deposit_leg_bound,
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
