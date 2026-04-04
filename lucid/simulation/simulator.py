"""Event simulator factory (setup_event_simulator)."""
from lucid.generate import (
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
from lucid.detector_params import DetectorParams, ParticleParams, load_detector_params

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
    make_hits_simulation, make_hits_data, make_hits_likelihood,
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
        default_detector_params=False):
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
    if default_detector_params is False:
        _default_dp = None
    elif default_detector_params is True:
        if physics_config is None:
            raise ValueError("physics_config is required when default_detector_params=True")
        _default_dp = load_detector_params(physics_config)
    elif isinstance(default_detector_params, DetectorParams):
        _default_dp = default_detector_params
    else:
        raise TypeError(
            f"default_detector_params must be bool or DetectorParams, got {type(default_detector_params)}")

    if detector_type not in ('Cylinder', 'Sphere', 'Box'):
        raise ValueError(f"detector_type must be 'Cylinder', 'Sphere', or 'Box', got {detector_type}")

    # ---- Read material and compute speed of light ---------------------------
    material = get_material_from_config(json_filename)
    SPEED_OF_LIGHT_MATERIAL = get_speed_of_light_in_material(material)

    # ---- Detector geometry --------------------------------------------------
    detector = generate_detector(json_filename)
    sensor_points = jnp.array(detector.all_points)
    photosensor_radius = detector.S_radius
    NUM_SENSORS = len(sensor_points)
    Nphot = n_photons

    if detector_type == 'Cylinder':
        propagate_photons = create_photon_propagator(
            sensor_points, photosensor_radius,
            r=detector.r, h=detector.H,
            temperature=temperature,
            max_sensors_per_cell=max_sensors_per_cell)
    elif detector_type == 'Sphere':
        propagate_photons = create_sphere_photon_propagator(
            sensor_points, photosensor_radius,
            sphere_radius=detector.r,
            temperature=temperature,
            n_divisions=100,
            max_sensors_per_cell=max_sensors_per_cell)
    elif detector_type == 'Box':
        propagate_photons = create_box_photon_propagator(
            sensor_points, photosensor_radius,
            length=detector.L, width=detector.W, height=detector.H,
            temperature=temperature,
            max_sensors_per_cell=max_sensors_per_cell)

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
    if is_data:
        photon_update_fn = photon_iteration_sample
    elif use_expected_value is False:
        photon_update_fn = photon_iteration_sample
    else:
        photon_update_fn = jax.remat(photon_iteration_update_factors_safe)

    # ---- Geometry bounds check ----------------------------------------------
    if detector_type == 'Cylinder':
        def get_inside_detector_flag(positions):
            x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
            inside_xy = (x ** 2 + y ** 2) <= detector.r ** 2
            inside_z = (z >= -detector.H / 2) & (z <= detector.H / 2)
            return inside_xy & inside_z
    elif detector_type == 'Sphere':
        def get_inside_detector_flag(positions):
            return jnp.linalg.norm(positions, axis=1) <= detector.r
    elif detector_type == 'Box':
        def get_inside_detector_flag(positions):
            return box_bounds_check(positions, detector.L, detector.W, detector.H)

    # ---- make_hits wrapper selection ----------------------------------------
    if is_data:
        def _make_hits_fn(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections):
            return make_hits_data(flat_weights, flat_indices, flat_times, num_sensors,
                                  qe=qe, rng_key=qe_key, apply_smearing=apply_smearing)
    else:
        def _make_hits_fn(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections):
            return make_hits_simulation(flat_weights, flat_indices, flat_times, num_sensors,
                                        qe=qe, qe_corrections=qe_corrections)

        def _make_hits_likelihood_fn(flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections):
            return make_hits_likelihood(flat_weights, flat_indices, flat_times, num_sensors,
                                        qe=qe, qe_corrections=qe_corrections)

    # ================================================================
    # Core propagation (shared by all modes)
    # ================================================================

    @partial(jax.jit, static_argnames=(
        'n_rays', 'K', 'n_grad_iters', 'max_sensors_per_cell', 'num_sensors',
        'propagate_fn', 'photon_update_fn'))
    def _common_propagation(
            positions, directions, intensities, times,
            n_rays, detector_params, key,
            num_sensors, K, n_grad_iters, max_sensors_per_cell,
            propagate_fn, photon_update_fn):
        """Core photon propagation loop using DetectorParams."""

        # Named field access (no tuple unpacking)
        scatter_length = detector_params.scatter_length
        wall_reflection_rate = detector_params.wall_reflection_rate
        sensor_reflection_rate = detector_params.sensor_reflection_rate
        absorption_length = detector_params.absorption_length
        qe = detector_params.qe
        qe_corrections = detector_params.qe_corrections

        initial_survival = jnp.ones(n_rays)

        def propagation_step(carry, i):
            current_pos, current_dir, current_times, survival, key = carry
            key, prop_key = jax.random.split(key)

            prop_results = propagate_fn(current_pos, current_dir)
            depositions = prop_results['sensor_weights']
            sensor_indices = prop_results['sensor_indices']
            hit_times_meters = prop_results['times']  # ray parameter in meters
            hit_positions = prop_results['positions']
            normals = prop_results['normals']
            inside_sensor = prop_results['inside_sensor']

            hit_sensor = jnp.max(inside_sensor, axis=0)
            surface_distances = jnp.linalg.norm(hit_positions - current_pos, axis=1) - 1e-6

            key, subkey = jax.random.split(key)
            rng_keys = jax.random.split(subkey, n_rays)

            # vmap: 12 args — dual reflection, no tau_gs
            (new_positions, new_directions, new_times,
             detect_probs, reflection_attenuations,
             continuing_factors) = jax.vmap(
                photon_update_fn,
                in_axes=(0, 0, 0, 0, 0,
                         None, None, None, None,
                         0, 0, None)
            )(current_pos, current_dir, current_times,
              surface_distances, normals,
              scatter_length, wall_reflection_rate, sensor_reflection_rate,
              absorption_length,
              hit_sensor, rng_keys, SPEED_OF_LIGHT_MATERIAL)

            inside_detector = get_inside_detector_flag(new_positions)
            safe_continuing = jnp.where(inside_detector, continuing_factors, 0.0)

            new_survival = survival * safe_continuing

            physical_intensities = intensities * survival
            detected_factors = detect_probs * reflection_attenuations
            updated_weights = depositions * physical_intensities[None, :] * detected_factors[None, :]
            times_ns = hit_times_meters / SPEED_OF_LIGHT_MATERIAL  # m / (m/ns) = ns
            total_times = times_ns + current_times[:, None]

            iter_weights = updated_weights
            iter_indices = sensor_indices
            iter_times = total_times.squeeze(-1)

            # Stop gradient on position/direction after n_grad_iters iterations
            # n_grad_iters=0 (reconstruction): always stop_gradient
            # n_grad_iters=2 (calibration): gradient flows for first 2 iterations
            next_pos = jnp.where(i < K, new_positions, jax.lax.stop_gradient(new_positions))
            next_dir = jnp.where(i < n_grad_iters, new_directions, jax.lax.stop_gradient(new_directions))
            next_times = new_times
            next_survival = new_survival

            new_carry = (next_pos, next_dir, next_times, next_survival, key)
            outputs = (iter_weights, iter_indices, iter_times)
            return new_carry, outputs

        init_carry = (positions, directions, times, initial_survival, key)
        propagation_step_remat = jax.remat(propagation_step)

        _, (all_weights, all_indices, all_times) = jax.lax.scan(
            propagation_step_remat, init_carry, jnp.arange(K))

        flat_weights = all_weights.reshape(-1)
        flat_indices = all_indices.reshape(-1)
        flat_times = all_times.reshape(-1)

        key, qe_key = jax.random.split(key)
        corrected_q, aligned_times = _make_hits_fn(
            flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections)

        return corrected_q, aligned_times

    @partial(jax.jit, static_argnames=(
        'n_rays', 'K', 'n_grad_iters', 'max_sensors_per_cell', 'num_sensors',
        'propagate_fn', 'photon_update_fn'))
    def _common_propagation_likelihood(
            positions, directions, intensities, times,
            n_rays, detector_params, key,
            num_sensors, K, n_grad_iters, max_sensors_per_cell,
            propagate_fn, photon_update_fn):
        """Core photon propagation returning per-photon data for likelihood losses."""

        scatter_length = detector_params.scatter_length
        wall_reflection_rate = detector_params.wall_reflection_rate
        sensor_reflection_rate = detector_params.sensor_reflection_rate
        absorption_length = detector_params.absorption_length
        qe = detector_params.qe
        qe_corrections = detector_params.qe_corrections

        initial_survival = jnp.ones(n_rays)

        def propagation_step(carry, i):
            current_pos, current_dir, current_times, survival, key = carry
            key, prop_key = jax.random.split(key)

            prop_results = propagate_fn(current_pos, current_dir)
            depositions = prop_results['sensor_weights']
            sensor_indices = prop_results['sensor_indices']
            hit_times_meters = prop_results['times']
            hit_positions = prop_results['positions']
            normals = prop_results['normals']
            inside_sensor = prop_results['inside_sensor']

            hit_sensor = jnp.max(inside_sensor, axis=0)
            surface_distances = jnp.linalg.norm(hit_positions - current_pos, axis=1) - 1e-6

            key, subkey = jax.random.split(key)
            rng_keys = jax.random.split(subkey, n_rays)

            (new_positions, new_directions, new_times,
             detect_probs, reflection_attenuations,
             continuing_factors) = jax.vmap(
                photon_update_fn,
                in_axes=(0, 0, 0, 0, 0,
                         None, None, None, None,
                         0, 0, None)
            )(current_pos, current_dir, current_times,
              surface_distances, normals,
              scatter_length, wall_reflection_rate, sensor_reflection_rate,
              absorption_length,
              hit_sensor, rng_keys, SPEED_OF_LIGHT_MATERIAL)

            inside_detector = get_inside_detector_flag(new_positions)
            safe_continuing = jnp.where(inside_detector, continuing_factors, 0.0)

            new_survival = survival * safe_continuing

            physical_intensities = intensities * survival
            detected_factors = detect_probs * reflection_attenuations
            updated_weights = depositions * physical_intensities[None, :] * detected_factors[None, :]
            times_ns = hit_times_meters / SPEED_OF_LIGHT_MATERIAL
            total_times = times_ns + current_times[:, None]

            iter_weights = updated_weights
            iter_indices = sensor_indices
            iter_times = total_times.squeeze(-1)

            next_pos = jnp.where(i < 0, new_positions, jax.lax.stop_gradient(new_positions))
            next_dir = jnp.where(i < n_grad_iters, new_directions, jax.lax.stop_gradient(new_directions))
            next_times = new_times
            next_survival = new_survival

            new_carry = (next_pos, next_dir, next_times, next_survival, key)
            outputs = (iter_weights, iter_indices, iter_times)
            return new_carry, outputs

        init_carry = (positions, directions, times, initial_survival, key)
        propagation_step_remat = jax.remat(propagation_step)

        _, (all_weights, all_indices, all_times) = jax.lax.scan(
            propagation_step_remat, init_carry, jnp.arange(K))

        flat_weights = all_weights.reshape(-1)
        flat_indices = all_indices.reshape(-1)
        flat_times = all_times.reshape(-1)

        key, qe_key = jax.random.split(key)
        return _make_hits_likelihood_fn(
            flat_weights, flat_indices, flat_times, num_sensors, qe_key, qe, qe_corrections)

    # ================================================================
    # Mode-specific simulation functions
    # ================================================================

    @jax.jit
    def _simulation_with_data_impl(particle_params, detector_params, key, photon_data):
        """Data mode: photons from ROOT/PhotonSim files, particle_params is ParticleParams."""
        energy = particle_params.energy
        track_origin = particle_params.position
        track_direction = particle_params.direction  # property

        photon_origins = photon_data['photon_origins'] / 100.0  # cm to m
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

        return _common_propagation(
            final_origins, final_directions, photon_intensities, photon_times,
            n_rays, detector_params, key, NUM_SENSORS, K, 0, max_sensors_per_cell,
            propagate_photons, photon_update_fn)

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

        photon_directions, photon_origins, photon_weights = photonsim_differentiable_get_rays(
            track_origin, track_direction, energy, Nphot, grid_data, model_params, key,
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

        return _common_propagation_likelihood(
            photon_origins, photon_directions, photon_intensities, photon_times + t0,
            Nphot, detector_params, key, NUM_SENSORS, K, 0, max_sensors_per_cell,
            propagate_photons, photon_update_fn)

    @jax.jit
    def _simulation_sensor_calibration_impl(source, detector_params, key):
        """Calibration mode: source is a callable (IsotropicSource or LaserSource)."""
        photon_directions, photon_origins, photon_intensities = source(Nphot, key)
        photon_times = jnp.zeros((Nphot,))

        return _common_propagation(
            photon_origins, photon_directions, photon_intensities, photon_times,
            Nphot, detector_params, key, NUM_SENSORS, K, 2, max_sensors_per_cell,
            propagate_photons, photon_update_fn)

    # ---- Return the right function ------------------------------------------
    if is_data:
        if _default_dp is not None:
            @jax.jit
            def _sim_data_default(particle_params, key, photon_data):
                return _simulation_with_data_impl(particle_params, _default_dp, key, photon_data)
            _sim_data_default.default_detector_params = _default_dp
            return _sim_data_default
        else:
            return _simulation_with_data_impl
    elif is_calibration:
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
        grid_data = create_photonsim_siren_grid(photonsim_predictor, 250)
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
