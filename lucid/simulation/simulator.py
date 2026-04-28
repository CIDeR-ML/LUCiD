"""Event simulator factory (setup_event_simulator)."""
from lucid.sources.siren_rays import (
    photonsim_differentiable_get_rays,
    predict_t0,
)
from lucid.propagation.cylinder import create_photon_propagator
from lucid.propagation.sphere import create_sphere_photon_propagator
from lucid.propagation.box import create_box_photon_propagator, box_bounds_check
from lucid.geometry import generate_detector, get_material_from_config
from lucid.utils import (
    unpack_t0_params, unpack_photonsim_params,
    get_speed_of_light_in_material,
    spherical_to_cartesian, base_dir_path,
    smear_times, smear_charges_SK_like,
)
from lucid.detector_params import DetectorParams, ParticleParams, load_detector_params, load_physics_config
from lucid.wavelength.medium import make_medium, load_qe_curve
from lucid.wavelength.spectrum import sample_cherenkov_wavelengths

import jax
import jax.numpy as jnp
from typing import Optional, Tuple
import os
from lucid.siren.core import create_photonsim_siren_grid
from functools import partial
from lucid.siren.training.inference import SIRENPredictor

from lucid.simulation.optics import (
    normalize, jax_normalize, jax_rotate_vector,
)
from lucid.simulation.photon_step import (
    photon_iteration_sample, photon_iteration_update_factors_safe,
)
from lucid.simulation.sensor_response import (
    make_hits_simulation, make_hits_data, make_hits_likelihood, make_hits_per_segment,
)

# ===================================================================
# Event simulator factory
# ===================================================================

def setup_event_simulator(
        json_filename,
        n_photons=1_000_000,
        temperature=0.2,
        K=7,
        is_data=False,
        is_calibration=False,
        max_sensors_per_cell=4,
        detector_type='Cylinder',
        use_expected_value=True,
        particle='muon',
        apply_smearing=True,
        physics_config=None,
        default_detector_params=False,
        wavelength_mode=True,
        hit_mode=None,
        n_grad_iters=None,
        pos_grad_threshold=None,  # None → use mode default (calib:K, track:0)
        **grid_params):
    """
    Set up and return an event simulator using DetectorParams / ParticleParams.

    Parameters
    ----------
    json_filename : str
        Path to detector geometry JSON.
    n_photons : int
        Number of photons per event.
    temperature : float
        Soft-assignment temperature for propagation.
    K : int
        Maximum scattering iterations.
    is_data : bool
        ROOT-file data mode.
    is_calibration : bool
        Calibration mode (source passed at call time).
    max_sensors_per_cell : int
        Grid cell sensor limit.
    detector_type : str
        'Cylinder', 'Sphere', or 'Box'.
    use_expected_value : bool
        True -> STE (differentiable), False -> MC sampling.
    particle : str
        Particle type (e.g., 'muon', 'electron'). Used to load SIREN model,
        t0 parameters, and photon normalization from config.
    apply_smearing : bool
        If True, apply SK-like charge and time smearing in data mode.
    physics_config : str or None
        Path to physics config JSON (e.g. ``SK_physics_config.json``).
        Required when ``default_detector_params=True``.
    default_detector_params : bool or DetectorParams
        Controls whether ``detector_params`` is baked into the returned function:

        - ``False`` (default) -- returned function **requires** ``detector_params``
          as an explicit argument.
        - ``True`` -- loads ``DetectorParams`` from *physics_config* at setup time
          and bakes it into the closure.
        - ``DetectorParams`` instance -- bakes that instance directly (no file load).
    hit_mode : str or None
        How sensor hits are aggregated. ``None`` (default) uses the source-appropriate
        choice; pass a value to override (useful for analysis):

        - ``'aggregated'`` -- differentiable per-sensor (charges, times). Default for calibration.
        - ``'per_photon'`` -- per-photon arrays (log_w, times, indices, charges). Default for track.
        - ``'realistic'`` -- Bernoulli QE sampling, hard-min timing, optional smearing. Default for data.

    Returns
    -------
    callable
        When ``default_detector_params`` is ``False``:

        - **Calibration** ``(source, detector_params, key) -> (charges, times)``
        - **Track**       ``(particle_params, detector_params, key) -> (charges, times)``
        - **Data**        ``(particle_params, detector_params, key, photon_data) -> (charges, times)``

        When ``default_detector_params`` is truthy (``True`` or a ``DetectorParams``):

        - **Calibration** ``(source, key) -> (charges, times)``
        - **Track**       ``(particle_params, key) -> (charges, times)``
        - **Data**        ``(particle_params, key, photon_data) -> (charges, times)``

        When detector params are baked in, the returned function also exposes
        a ``.default_detector_params`` attribute for inspection.
    """
    # ---- Resolve default_detector_params ------------------------------------
    _medium_model_path = None
    _qe_curve_path = None

    if default_detector_params is False:
        _default_dp = None
        # Still extract medium/QE paths from physics config if provided
        if physics_config is not None:
            _, _medium_model_path, _qe_curve_path = load_physics_config(physics_config)
    elif default_detector_params is True:
        if physics_config is None:
            raise ValueError("physics_config is required when default_detector_params=True")
        _default_dp, _medium_model_path, _qe_curve_path = load_physics_config(physics_config)
    elif isinstance(default_detector_params, DetectorParams):
        _default_dp = default_detector_params
        # Extract medium/QE paths from physics config if provided alongside
        if physics_config is not None:
            _, _medium_model_path, _qe_curve_path = load_physics_config(physics_config)
    else:
        raise TypeError(
            f"default_detector_params must be bool or DetectorParams, got {type(default_detector_params)}")

    # ---- Build containers from flat args -----------------------------------
    from lucid.geometry.detector_geometry import DetectorGeometry
    from lucid.simulation.config import SimConfig

    det_geom = DetectorGeometry.from_config(
        json_filename, temperature=temperature,
        max_sensors_per_cell=max_sensors_per_cell,
        detector_type=detector_type,
        **grid_params)

    mode = 'data' if is_data else ('calibration' if is_calibration else 'track')
    sim_config = SimConfig(
        n_photons=n_photons, K=K, mode=mode,
        use_expected_value=use_expected_value,
        apply_smearing=apply_smearing,
        n_grad_iters=n_grad_iters)

    # ---- Extract fields from containers ------------------------------------
    material = det_geom.medium.material
    SPEED_OF_LIGHT_MATERIAL = det_geom.speed_of_light
    detector = det_geom.detector
    sensor_points = det_geom.sensor_points
    NUM_SENSORS = det_geom.num_sensors
    Nphot = sim_config.n_photons
    propagate_photons = det_geom.propagator

    # ---- Handle qe_corrections for baked-in detector params -----------------
    if _default_dp is not None:
        qe_corr = _default_dp.qe_corrections
        # If scalar placeholder (from null in JSON), broadcast to NUM_SENSORS
        if qe_corr.ndim == 0:
            _default_dp = _default_dp._replace(qe_corrections=jnp.ones(NUM_SENSORS) * qe_corr)
        elif len(qe_corr) != NUM_SENSORS:
            raise ValueError(
                f"qe_corrections has {len(qe_corr)} elements "
                f"but detector has {NUM_SENSORS} sensors")

    # ---- Select photon update function --------------------------------------
    if sim_config.is_data:
        photon_update_fn = photon_iteration_sample
    elif sim_config.use_expected_value is False:
        photon_update_fn = photon_iteration_sample
    else:
        photon_update_fn = jax.remat(photon_iteration_update_factors_safe)

    # ---- Geometry bounds check (delegates to detector method) ----------------
    def get_inside_detector_flag(positions):
        return detector.bounds_check(positions)

    # ---- Resolve hit_mode ---------------------------------------------------
    _VALID_HIT_MODES = ('aggregated', 'per_photon', 'realistic', 'per_segment')
    if hit_mode is None:
        if sim_config.is_data:
            hit_mode = 'realistic'
        elif sim_config.is_calibration:
            hit_mode = 'aggregated'
        else:
            hit_mode = 'per_photon'
    elif hit_mode not in _VALID_HIT_MODES:
        raise ValueError(
            f"hit_mode must be one of {_VALID_HIT_MODES} or None, got {hit_mode!r}")

    # ---- make_hits wrapper selection ----------------------------------------
    # All wrappers accept the optional `flat_segment_idx` / `n_segments`
    # kwargs for signature uniformity; only `_make_hits_per_segment`
    # consumes them. The other modes ignore them silently.
    def _make_hits_aggregated(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections,
                              flat_segment_idx=None, n_segments=0):
        return make_hits_simulation(flat_weights, flat_indices, flat_times, num_sensors,
                                    qe=qe, qe_corrections=qe_corrections)

    def _make_hits_per_photon(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections,
                              flat_segment_idx=None, n_segments=0):
        return make_hits_likelihood(flat_weights, flat_indices, flat_times, num_sensors,
                                    qe=qe, qe_corrections=qe_corrections)

    def _make_hits_realistic(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections,
                             flat_segment_idx=None, n_segments=0):
        return make_hits_data(flat_weights, flat_indices, flat_times, num_sensors,
                              qe=qe, qe_corrections=qe_corrections,
                              rng_key=qe_key, apply_smearing=sim_config.apply_smearing)

    def _make_hits_per_segment(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections,
                           flat_segment_idx=None, n_segments=0):
        return make_hits_per_segment(flat_weights, flat_indices, flat_times, num_sensors,
                                 qe=qe, qe_corrections=qe_corrections,
                                 rng_key=qe_key, apply_smearing=sim_config.apply_smearing,
                                 flat_segment_idx=flat_segment_idx, n_segments=n_segments)

    _make_hits_fn = {
        'aggregated': _make_hits_aggregated,
        'per_photon': _make_hits_per_photon,
        'realistic': _make_hits_realistic,
        'per_segment': _make_hits_per_segment,
    }[hit_mode]

    # ---- Wavelength-dependent medium (when wavelength_mode=True) -----
    if wavelength_mode:
        _wl_grid = jnp.linspace(300.0, 700.0, 200)
        _medium_wl = make_medium(material, wavelength_grid=_wl_grid,
                                 medium_model_path=_medium_model_path)
        _qe_fn = load_qe_curve(_qe_curve_path) if _qe_curve_path else None
    else:
        _medium_wl = None
        _qe_fn = None

    def _get_optical_arrays(n, detector_params, key, wavelengths=None):
        """Compute per-photon (n,) scatter/absorption arrays and QE weights.

        When wavelength_mode=True: uses medium coefficients at given wavelengths.
        When wavelength_mode=False: broadcasts DetectorParams scalars.

        Returns (scatter_lengths, absorption_lengths, qe_weights, key).
        qe_weights is (n,) or None.
        """
        if wavelength_mode and _medium_wl is None:
            raise RuntimeError(
                "wavelength_mode=True but no medium model was loaded. "
                "Provide a physics_config with 'medium_model' or set wavelength_mode=False.")
        if not wavelength_mode:
            return (jnp.full(n, detector_params.scatter_length),
                    jnp.full(n, detector_params.absorption_length),
                    None, key)

        # Sample or use provided wavelengths
        if wavelengths is None:
            key, wl_key = jax.random.split(key)
            wavelengths = sample_cherenkov_wavelengths(wl_key, n)

        wavelengths = jnp.clip(wavelengths,
                               _medium_wl.wavelength_grid[0],
                               _medium_wl.wavelength_grid[-1])
        sc = jnp.interp(wavelengths, _medium_wl.wavelength_grid, _medium_wl.scatter_coeff)
        ac = jnp.interp(wavelengths, _medium_wl.wavelength_grid, _medium_wl.absorption_coeff)
        scatter_lengths = 1.0 / (sc + 1e-30)
        absorption_lengths = 1.0 / (ac + 1e-30)

        qe_weights = _qe_fn(wavelengths) if _qe_fn is not None else None
        return scatter_lengths, absorption_lengths, qe_weights, key

    # ================================================================
    # Core propagation (shared by all modes)
    # ================================================================

    @partial(jax.jit, static_argnames=(
        'n_rays', 'K', 'n_grad_iters', 'max_sensors_per_cell', 'num_sensors',
        'propagate_fn', 'photon_update_fn', 'pos_grad_threshold', 'make_hits_fn',
        'n_segments'))
    def _common_propagation(
            positions, directions, intensities, times,
            scatter_lengths, absorption_lengths,
            qe_per_photon,
            n_rays, detector_params, key,
            num_sensors, K, n_grad_iters, max_sensors_per_cell,
            propagate_fn, photon_update_fn,
            pos_grad_threshold, make_hits_fn,
            segment_idx=None, n_segments=0):
        """Core photon propagation loop.

        Parameters
        ----------
        scatter_lengths : jnp.ndarray
            Per-photon scattering lengths, shape (n_rays,).
        absorption_lengths : jnp.ndarray
            Per-photon absorption lengths, shape (n_rays,).
        qe_per_photon : jnp.ndarray
            Per-photon quantum efficiency, shape (n_rays,).
        pos_grad_threshold : int
            Iteration threshold for position stop_gradient.
        make_hits_fn : callable
            Sensor response aggregation function.
        """

        wall_reflection_rate = detector_params.wall_reflection_rate
        sensor_reflection_rate = detector_params.sensor_reflection_rate
        qe_corrections = detector_params.qe_corrections
        # Scalar placeholder (e.g. from JSON `qe_corrections: 1.0`) — broadcast
        # to per-sensor so indexing in make_hits_* works when detector_params
        # comes in directly (not baked-in via setup-time guard).
        if qe_corrections.ndim == 0:
            qe_corrections = jnp.ones(num_sensors) * qe_corrections

        from lucid.simulation.types import PhotonState

        initial_survival = jnp.ones(n_rays)

        def propagation_step(carry, i):
            state = carry
            key, prop_key = jax.random.split(state.key)

            prop_results = propagate_fn(state.positions, state.directions)
            depositions = prop_results['sensor_weights']
            sensor_indices = prop_results['sensor_indices']
            hit_times_meters = prop_results['times']
            hit_positions = prop_results['positions']
            normals = prop_results['normals']
            inside_sensor = prop_results['inside_sensor']

            hit_sensor = jnp.max(inside_sensor, axis=0)
            surface_distances = jnp.linalg.norm(hit_positions - state.positions, axis=1) - 1e-6

            key, subkey = jax.random.split(key)
            rng_keys = jax.random.split(subkey, n_rays)

            # vmap: 12 args — per-photon scatter/absorption, scalar reflections
            (new_positions, new_directions, new_times,
             detect_probs, reflection_attenuations,
             continuing_factors) = jax.vmap(
                photon_update_fn,
                in_axes=(0, 0, 0, 0, 0,
                         0, None, None, 0,
                         0, 0, None)
            )(state.positions, state.directions, state.times,
              surface_distances, normals,
              scatter_lengths, wall_reflection_rate, sensor_reflection_rate,
              absorption_lengths,
              hit_sensor, rng_keys, SPEED_OF_LIGHT_MATERIAL)

            inside_detector = get_inside_detector_flag(new_positions)
            safe_continuing = jnp.where(inside_detector, continuing_factors, 0.0)

            new_survival = state.survival * safe_continuing

            physical_intensities = intensities * state.survival
            detected_factors = detect_probs * reflection_attenuations
            updated_weights = depositions * physical_intensities[None, :] * detected_factors[None, :]
            times_ns = hit_times_meters / SPEED_OF_LIGHT_MATERIAL
            total_times = times_ns + state.times[:, None]

            iter_weights = updated_weights
            iter_indices = sensor_indices
            iter_times = total_times.squeeze(-1)

            # Stop gradient on positions and directions independently:
            # - Position: threshold=K → gradient flows all iterations (standard)
            #             threshold=0 → always stop (likelihood)
            # - Direction: n_grad_iters=0 (reconstruction) → always stop
            #              n_grad_iters=2 (calibration) → gradient flows for first 2
            next_pos = jnp.where(i < pos_grad_threshold, new_positions, jax.lax.stop_gradient(new_positions))
            next_dir = jnp.where(i < n_grad_iters, new_directions, jax.lax.stop_gradient(new_directions))

            new_state = PhotonState(
                positions=next_pos,
                directions=next_dir,
                times=new_times,
                survival=new_survival,
                key=key,
            )
            outputs = (iter_weights, iter_indices, iter_times)
            return new_state, outputs

        init_state = PhotonState(
            positions=positions,
            directions=directions,
            times=times,
            survival=initial_survival,
            key=key,
        )
        propagation_step_remat = jax.remat(propagation_step)

        _, (all_weights, all_indices, all_times) = jax.lax.scan(
            propagation_step_remat, init_state, jnp.arange(K))

        flat_weights = all_weights.reshape(-1)
        flat_indices = all_indices.reshape(-1)
        flat_times = all_times.reshape(-1)

        # Tile per-photon QE to match flat shape.
        # all_weights shape: (K, max_sensors_per_cell, n_rays), C-order reshape
        # → photon index is i % n_rays
        photon_idx = jnp.arange(flat_weights.shape[0]) % n_rays
        flat_qe = qe_per_photon[photon_idx]

        # Per-photon segment id, broadcast to flat shape via the same trick.
        # None when no segment decomposition is requested (every mode except
        # 'per_segment'); make_hits_fn dispatches accordingly.
        flat_segment_idx = (segment_idx[photon_idx]
                             if segment_idx is not None else None)

        key, qe_key = jax.random.split(key)
        return make_hits_fn(
            flat_weights, flat_indices, flat_times, num_sensors,
            qe_key, flat_qe, qe_corrections,
            flat_segment_idx=flat_segment_idx, n_segments=n_segments)

    # ================================================================
    # Mode-specific simulation functions
    # ================================================================

    @partial(jax.jit, static_argnames=('n_segments',))
    def _simulation_with_data_impl(particle_params, detector_params, key, photon_data,
                                   n_segments=0):
        """Data mode: photons from ROOT/PhotonSim files, particle_params is ParticleParams.

        When ``photon_data`` carries ``'photon_segment_index'`` (per-photon
        int32 array, padded with -1 sentinels) and ``n_segments > 0`` is
        passed, the underlying ``make_hits_fn`` may emit per-(segment,
        sensor) decomposition outputs (used by ``hit_mode='per_segment'``).
        Existing data-mode callers do not pass either and see no change.
        """
        energy = particle_params.energy
        track_origin = particle_params.position
        track_direction = particle_params.direction  # property

        photon_origins = photon_data['photon_origins']  # already m
        photon_directions = photon_data['photon_directions']
        photon_times = photon_data['photon_times']

        # Apply rotation if specified (using jax.lax.cond for JIT compatibility)
        def apply_rotation_fn(args):
            origins, directions, rot_axis, rot_angle = args
            rotated_directions = jax.vmap(
                lambda v: jax_rotate_vector(v, rot_axis, rot_angle)
            )(directions)
            rotated_origins = jax.vmap(
                lambda v: jax_rotate_vector(v, rot_axis, rot_angle)
            )(origins)
            return rotated_origins, rotated_directions

        def no_rotation_fn(args):
            origins, directions, _, _ = args
            return origins, directions

        rotation_axis = photon_data['rotation_axis']
        rotation_angle = photon_data['rotation_angle']
        apply_rotation = photon_data['apply_rotation']

        final_origins, final_directions = jax.lax.cond(
            apply_rotation,
            apply_rotation_fn,
            no_rotation_fn,
            (photon_origins, photon_directions, rotation_axis, rotation_angle)
        )

        # Apply translation if specified (AFTER rotation)
        def apply_translation_fn(args):
            origins, translation_vec = args
            return origins + translation_vec[None, :]

        def no_translation_fn(args):
            origins, _ = args
            return origins

        apply_translation = photon_data.get('apply_translation', False)
        translation_vector = photon_data.get('translation_vector', jnp.array([0.0, 0.0, 0.0]))

        final_origins = jax.lax.cond(
            apply_translation,
            apply_translation_fn,
            no_translation_fn,
            (final_origins, translation_vector)
        )

        n_rays = photon_origins.shape[0]
        mask = jnp.arange(n_rays) < photon_data['N']
        photon_intensities = 1.0 * mask.astype(jnp.float32)

        # Per-photon optical properties — use PhotonSim wavelengths if available
        data_wavelengths = photon_data.get('wavelengths', None)
        scatter_lengths, absorption_lengths, qe_weights, key = _get_optical_arrays(
            n_rays, detector_params, key, wavelengths=data_wavelengths)

        # Per-photon QE: wavelength curve * scalar qe (passed to make_hits, not baked into weights)
        if qe_weights is not None:
            qe_per_photon = qe_weights * detector_params.qe
        else:
            qe_per_photon = jnp.full(n_rays, detector_params.qe)

        # Optional per-photon segment id (per_segment mode). Realistic / track /
        # calibration paths don't pass this — segment_idx stays None and the
        # decomposition branch in make_hits_per_segment is never taken.
        segment_idx = photon_data.get('photon_segment_index', None)

        _pgt = sim_config.K if pos_grad_threshold is None else pos_grad_threshold
        return _common_propagation(
            final_origins, final_directions, photon_intensities, photon_times,
            scatter_lengths, absorption_lengths,
            qe_per_photon,
            n_rays, detector_params, key, NUM_SENSORS, sim_config.K, sim_config.effective_n_grad_iters, max_sensors_per_cell,
            propagate_photons, photon_update_fn,
            pos_grad_threshold=_pgt, make_hits_fn=_make_hits_fn,
            segment_idx=segment_idx, n_segments=n_segments)

    # Load photonsim parameters from configuration (power-law normalization, SIREN path)
    photonsim_params = unpack_photonsim_params(particle, material)
    tot_n_photons_a, tot_n_photons_b, tot_n_photons_c = photonsim_params['tot_n_photons_normalization']
    num_seeds_a, num_seeds_b, num_seeds_c = photonsim_params['num_seeds']

    @jax.jit
    def tot_n_photons_normalization(x):
        """Power law: a * energy^b + c. Parameters loaded from config."""
        return tot_n_photons_a * jnp.power(x, tot_n_photons_b) + tot_n_photons_c

    @jax.jit
    def _simulation_without_data_impl(particle_params, detector_params, key, grid_data, model_params):
        """SIREN mode: particle_params is ParticleParams."""
        energy = particle_params.energy
        track_origin = particle_params.position
        track_direction = particle_params.direction  # property

        key, ray_key, opt_key = jax.random.split(key, 3)
        photon_directions, photon_origins, photon_weights = photonsim_differentiable_get_rays(
            track_origin, track_direction, energy, Nphot, grid_data, model_params, ray_key,
            num_seeds_a, num_seeds_b, num_seeds_c
        )

        total_photons_norm = tot_n_photons_normalization(energy)
        photon_intensities = (total_photons_norm * photon_weights) / Nphot
        photon_times = jnp.zeros((Nphot,))

        distances_to_vertex = jnp.linalg.norm(photon_origins - track_origin, axis=1) * 1000
        predict_t0_vec = jax.vmap(predict_t0, in_axes=(0, None, None, None, None, None, None, None, None))
        baseline_slope, baseline_intercept, A_slope, A_intercept, B_slope, B_intercept, offset = t0_params
        t0 = jax.lax.stop_gradient(
            predict_t0_vec(distances_to_vertex, energy,
                           baseline_slope, baseline_intercept,
                           A_slope, A_intercept,
                           B_slope, B_intercept, offset))

        # Per-photon optical properties (Cherenkov spectrum when wavelength_mode)
        scatter_lengths, absorption_lengths, qe_weights, key = _get_optical_arrays(
            Nphot, detector_params, opt_key)

        # Per-photon QE: wavelength curve * scalar qe (passed to make_hits, not baked into weights)
        if qe_weights is not None:
            qe_per_photon = qe_weights * detector_params.qe
        else:
            qe_per_photon = jnp.full(Nphot, detector_params.qe)

        _pgt = 0 if pos_grad_threshold is None else pos_grad_threshold
        return _common_propagation(
            photon_origins, photon_directions, photon_intensities, photon_times + t0,
            scatter_lengths, absorption_lengths,
            qe_per_photon,
            Nphot, detector_params, key, NUM_SENSORS, sim_config.K, sim_config.effective_n_grad_iters, max_sensors_per_cell,
            propagate_photons, photon_update_fn,
            pos_grad_threshold=_pgt, make_hits_fn=_make_hits_fn)

    @jax.jit
    def _simulation_sensor_calibration_impl(source, detector_params, key):
        """Calibration mode: source is a callable (IsotropicSource or LaserSource)."""
        key, source_key, opt_key = jax.random.split(key, 3)
        photon_directions, photon_origins, photon_intensities = source(Nphot, source_key)
        photon_times = jnp.zeros((Nphot,))

        # Per-photon optical properties
        source_wl = getattr(source, 'wavelength', None)
        if source_wl is not None:
            wavelengths = jnp.full(Nphot, source_wl)
        else:
            wavelengths = None
        scatter_lengths, absorption_lengths, qe_weights, key = _get_optical_arrays(
            Nphot, detector_params, opt_key, wavelengths=wavelengths)

        # Per-photon QE: wavelength curve * scalar qe (passed to make_hits, not baked into weights)
        if qe_weights is not None:
            qe_per_photon = qe_weights * detector_params.qe
        else:
            qe_per_photon = jnp.full(Nphot, detector_params.qe)

        _pgt = sim_config.K if pos_grad_threshold is None else pos_grad_threshold
        return _common_propagation(
            photon_origins, photon_directions, photon_intensities, photon_times,
            scatter_lengths, absorption_lengths,
            qe_per_photon,
            Nphot, detector_params, key, NUM_SENSORS, sim_config.K, sim_config.effective_n_grad_iters, max_sensors_per_cell,
            propagate_photons, photon_update_fn,
            pos_grad_threshold=_pgt, make_hits_fn=_make_hits_fn)

    # ---- Return the right function ------------------------------------------
    if sim_config.is_data:
        if _default_dp is not None:
            @partial(jax.jit, static_argnames=('n_segments',))
            def _sim_data_default(particle_params, key, photon_data, n_segments=0):
                return _simulation_with_data_impl(
                    particle_params, _default_dp, key, photon_data,
                    n_segments=n_segments)
            _sim_data_default.default_detector_params = _default_dp
            return _sim_data_default
        else:
            return _simulation_with_data_impl
    elif sim_config.is_calibration:
        if _default_dp is not None:
            @jax.jit
            def _sim_calibration_default(source, key):
                return _simulation_sensor_calibration_impl(source, _default_dp, key)
            _sim_calibration_default.default_detector_params = _default_dp
            return _sim_calibration_default
        else:
            return _simulation_sensor_calibration_impl
    else:
        model_base_path = photonsim_params['siren_model_path']
        photonsim_predictor = SIRENPredictor(model_base_path)
        grid_data = create_photonsim_siren_grid(photonsim_predictor)
        model_params = photonsim_predictor.params
        t0_params = unpack_t0_params(particle, material)
        if _default_dp is not None:
            @jax.jit
            def _sim_track_default(particle_params, key):
                return _simulation_without_data_impl(particle_params, _default_dp, key,
                                                     grid_data=grid_data, model_params=model_params)
            _sim_track_default.default_detector_params = _default_dp
            return _sim_track_default
        else:
            return partial(_simulation_without_data_impl,
                           grid_data=grid_data,
                           model_params=model_params)
