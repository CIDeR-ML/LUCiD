from tools.generate import (
    get_isotropic_rays,
    photonsim_differentiable_get_rays,
    predict_t0,
    generate_laser_photons,
    setup_calibration_generator,
)
from tools.propagation.cylinder import create_photon_propagator
from tools.propagation.sphere import create_sphere_photon_propagator
from tools.propagation.box import create_box_photon_propagator, box_bounds_check
from tools.geometry import generate_detector, get_material_from_config
from tools.utils import (
    unpack_t0_params, unpack_photonsim_params,
    get_speed_of_light_in_material,
    spherical_to_cartesian, base_dir_path,
    smear_times, smear_charges_SK_like,
)
from tools.detector_params import DetectorParams, ParticleParams, load_detector_params

import jax
import jax.numpy as jnp
from typing import Optional, Tuple
import os
from tools.siren.core import *
from functools import partial
from tools.siren.training.inference import SIRENPredictor


# ===================================================================
# Helper functions
# ===================================================================

def normalize(v, epsilon=1e-6):
    """Normalize a vector (or batch of vectors)."""
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.maximum(norm, epsilon)


def compute_reflection_direction(incident_dir, normal):
    """Compute specular reflection direction."""
    normal = normalize(normal)
    dot_product = jnp.sum(incident_dir * normal, axis=-1, keepdims=True)
    return normalize(incident_dir - 2 * dot_product * normal)


def sample_cosine_hemisphere(normal, rng_key):
    """Cosine-weighted hemisphere sampling (reparameterisation trick)."""
    k1, k2 = jax.random.split(rng_key)
    u1 = jax.random.uniform(k1)
    u2 = jax.random.uniform(k2)

    cos_theta = jnp.sqrt(1 - u1)
    sin_theta = jnp.sqrt(u1)
    phi = 2 * jnp.pi * u2

    local_dir = jnp.array([
        sin_theta * jnp.cos(phi),
        sin_theta * jnp.sin(phi),
        cos_theta,
    ])

    frame = create_local_frame(normal)
    return normalize(frame.T @ local_dir)


def create_local_frame(z):
    """Create an orthonormal frame with *z* as the z-axis."""
    z = normalize(z)
    t = jnp.where(
        jnp.abs(z[0]) < 0.9,
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0]),
    )
    x = normalize(jnp.cross(t, z))
    y = jnp.cross(z, x)
    return jnp.stack([x, y, z])


def sample_scatter_distance(D, S, rng_key):
    """Sample from a truncated exponential (scatter before surface)."""
    u = jax.random.uniform(rng_key)
    prob_term = -jnp.expm1(-D / S)
    return -S * jnp.log1p(-u * prob_term)


def solve_rayleigh_inverse_cdf(u):
    """
    Solve the inverse CDF for Rayleigh scattering: P(μ) ∝ (1 + μ²)
    Uses Cardano's formula to solve: μ³ + 3μ - (8u - 4) = 0
    """
    # Transform to standard form: t³ + pt + q = 0 where μ = t
    p = 3.0
    q = -(8.0 * u - 4.0)

    # Cardano's formula
    discriminant = -(4 * p ** 3 + 27 * q ** 2)

    # Three real roots case
    sqrt_disc_pos = jnp.sqrt(jnp.abs(discriminant))
    rho = jnp.sqrt(-p ** 3 / 27)
    theta = jnp.arccos(jnp.clip(-q / (2 * rho), -1, 1))
    mu_three_roots = 2 * jnp.cbrt(rho) * jnp.cos(theta / 3)

    # One real root case
    sqrt_disc_neg = jnp.sqrt(-discriminant)
    A = jnp.cbrt((-q + sqrt_disc_neg / (3 * jnp.sqrt(3))) / 2)
    B = jnp.cbrt((-q - sqrt_disc_neg / (3 * jnp.sqrt(3))) / 2)
    mu_one_root = A + B

    # Select based on discriminant sign
    mu = jnp.where(discriminant >= 0, mu_three_roots, mu_one_root)
    return jnp.clip(mu, -1.0, 1.0)


def compute_scatter_direction(incident_dir, rng_key):
    """Rayleigh-phase-function scattering direction."""
    k1, k2 = jax.random.split(rng_key)
    u1 = jax.random.uniform(k1)
    u2 = jax.random.uniform(k2)

    cos_theta = solve_rayleigh_inverse_cdf(u1)
    sin_theta = jnp.sqrt(1 - cos_theta ** 2)
    phi = 2 * jnp.pi * u2
    local_dir = normalize(jnp.array([
        sin_theta * jnp.cos(phi),
        sin_theta * jnp.sin(phi),
        cos_theta,
    ]))
    frame = create_local_frame(incident_dir)
    return normalize(frame @ local_dir)


# JAX-compatible rotation helpers (for data mode)
def jax_normalize(v, epsilon=1e-8):
    """Normalize a vector with numerical stability using JAX."""
    norm = jnp.linalg.norm(v)
    return jnp.where(norm > epsilon, v / norm, v)


def jax_rotate_vector(vector, axis, angle):
    """Rotate a vector around an axis by a given angle in radians using JAX."""
    axis = jax_normalize(axis)
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    cross_product = jnp.cross(axis, vector)
    dot_product = jnp.dot(axis, vector) * (1 - cos_angle)
    return cos_angle * vector + sin_angle * cross_product + dot_product * axis


# ===================================================================
# Photon iteration functions (12-arg signatures: dual reflection, no tau_gs)
# ===================================================================

def photon_iteration_sample(
        position, direction, time, surface_distance,
        normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
        absorption_length,
        hit_sensor, rng_key, speed_of_light):
    """
    Sampling version of photon iteration that makes binary decisions.

    Performs Monte Carlo sampling where photons make discrete choices
    (detect/reflect/scatter) rather than computing expected values.
    Walls use diffuse reflection, sensors use specular reflection.

    Parameters
    ----------
    position : jnp.ndarray
        Current 3D position of the photon
    direction : jnp.ndarray
        Current normalized direction vector of the photon
    time : float
        Current time of the photon
    surface_distance : float
        Distance to the nearest surface intersection
    normal : jnp.ndarray
        Surface normal at the intersection point
    scatter_length : float
        Mean free path for scattering in the medium
    wall_reflection_rate : float
        Probability of reflection when hitting a wall
    sensor_reflection_rate : float
        Probability of reflection when hitting a sensor
    absorption_length : float
        Mean free path for absorption in the medium
    hit_sensor : bool
        Whether the photon hit a sensor (True) or wall (False)
    rng_key : jax.random.PRNGKey
        Random key for sampling
    speed_of_light : float
        Speed of light in the medium (m/ns)

    Returns
    -------
    new_pos : jnp.ndarray
        Updated photon position
    new_dir : jnp.ndarray
        Updated photon direction
    new_time : float
        Updated photon time
    detect_prob : float
        1.0 if photon is detected, 0.0 otherwise
    reflection_attenuation : float
        Attenuation factor due to absorption
    continuing_factor : float
        Factor for continuing photons (0.0 if detected, attenuation if continues)
    """

    k1, k2, k3, k4 = jax.random.split(rng_key, 4)

    reflection_rate = jnp.where(hit_sensor, sensor_reflection_rate, wall_reflection_rate)
    scatter_distance = sample_scatter_distance(surface_distance, scatter_length, k2)

    reach_surface_prob = jnp.exp(-surface_distance / scatter_length)

    u1 = jax.random.uniform(k1)
    reaches_surface = u1 < reach_surface_prob

    u2 = jax.random.uniform(k2)
    reflects = reaches_surface & (u2 < reflection_rate)
    detects = reaches_surface & (u2 >= reflection_rate)
    scatters = ~reaches_surface

    # The fraction of 'errors' for a given epsilon change with the detector size.
    # This is because the numerical error in direction over a large distance translates
    # into a relatively larger deviation. For HK-size epsilon 1e-4 translates into a few
    # tens of rays going out of the detector per each million after several steps.
    # The rule of thumb is that epsilon needs to go down/up proportionally to the detector size.
    epsilon = 1e-4
    new_pos = jnp.where(
        scatters,
        position + scatter_distance * normalize(direction),
        position + surface_distance * normalize(direction) + epsilon * normalize(normal),
    )

    specular_dir = compute_reflection_direction(direction, normal)
    diffuse_dir = sample_cosine_hemisphere(normal, k4)
    reflection_dir = jnp.where(hit_sensor, specular_dir, diffuse_dir)
    scatter_dir = compute_scatter_direction(direction, k3)

    new_dir = jnp.where(
        reflects,
        reflection_dir,
        jnp.where(scatters, scatter_dir, direction),
    )

    distance_traveled = jnp.where(scatters, scatter_distance, surface_distance)
    new_time = time + distance_traveled / speed_of_light

    # Binary absorption sampling (Bernoulli)
    survival_prob = jnp.exp(-distance_traveled / absorption_length)
    u_absorption = jax.random.uniform(k3)
    survives_absorption = u_absorption < survival_prob
    attenuation = survives_absorption.astype(jnp.float32)

    detect_prob = detects.astype(jnp.float32)
    reflection_attenuation = attenuation
    continuing_factor = jnp.where(detects, 0.0, attenuation)

    return new_pos, new_dir, new_time, detect_prob, reflection_attenuation, continuing_factor


def photon_iteration_update_factors(
        position, direction, time, surface_distance,
        normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
        absorption_length,
        hit_sensor, rng_key, speed_of_light):
    """
    Expected-value photon update with Straight-Through Estimator (STE).

    Computes expected values for detect/reflect/scatter outcomes for full
    differentiability. Walls use diffuse reflection, sensors use specular.

    Parameters
    ----------
    position, direction : jnp.ndarray
        (3,) arrays for the photon's current state.
    time : float
        Current photon time.
    surface_distance : float
        Distance to nearest surface (norm of hit_position - position).
    normal : jnp.ndarray
        (3,) surface normal at the hit position.
    scatter_length : float
        Mean free path for scattering.
    wall_reflection_rate : float
        Probability of reflection at walls.
    sensor_reflection_rate : float
        Probability of reflection at sensors.
    absorption_length : float
        Mean free path for absorption.
    hit_sensor : bool
        Whether the photon hit a sensor (True) or wall (False).
    rng_key : jax.random.PRNGKey
        JAX PRNG key.
    speed_of_light : float
        Speed of light in the medium (m/ns).

    Returns
    -------
    new_pos : jnp.ndarray
        (3,) updated photon position.
    new_dir : jnp.ndarray
        (3,) updated photon direction.
    new_time : float
        Updated photon time.
    detect_prob : float
        Probability of detection.
    reflection_attenuation : float
        Attenuation factor from absorption over surface distance.
    continuing_factor : float
        Factor for continuation (reflection or scatter).
    """

    k1, k2, k3 = jax.random.split(rng_key, 3)

    reflection_rate = jnp.where(hit_sensor, sensor_reflection_rate, wall_reflection_rate)
    scatter_distance = sample_scatter_distance(surface_distance, scatter_length, k2)

    ratio = surface_distance / scatter_length
    reach_surface_prob = jnp.exp(-ratio)
    scatter_prob = -jnp.expm1(-ratio)

    reflect_prob = reach_surface_prob * reflection_rate
    detect_prob = reach_surface_prob * (1 - reflection_rate)

    reflection_attenuation = jnp.exp(-surface_distance / absorption_length)
    scatter_attenuation = jnp.exp(-scatter_distance / absorption_length)

    # Straight-Through Estimator for action selection:
    # Sample discrete action, but let soft probabilities flow in backward pass
    probs = jnp.array([reach_surface_prob, scatter_prob])
    probs_normalized = probs / (jnp.sum(probs) + 1e-10)

    u = jax.random.uniform(k1)
    hard_choice = (u < probs_normalized[0]).astype(jnp.float32)
    hard_weights = jnp.array([hard_choice, 1.0 - hard_choice])
    action_weights = hard_weights - jax.lax.stop_gradient(probs_normalized) + probs_normalized

    surface_weight = action_weights[0]
    scatter_weight = action_weights[1]

    # The fraction of 'errors' for a given epsilon change with the detector size.
    # This is because the numerical error in direction over a large distance translates
    # into a relatively larger deviation. For HK-size epsilon 1e-4 translates into a few
    # tens of rays going out of the detector per each million after several steps.
    # The rule of thumb is that epsilon needs to go down/up proportionally to the detector size.
    epsilon = 1e-4
    surface_pos = position + surface_distance * normalize(direction) + epsilon * normalize(normal)
    scatter_pos = position + scatter_distance * normalize(direction)

    specular_dir = compute_reflection_direction(direction, normal)
    diffuse_dir = sample_cosine_hemisphere(normal, k3)
    reflection_dir = jnp.where(hit_sensor, specular_dir, diffuse_dir)
    scatter_dir = compute_scatter_direction(direction, k3)

    new_pos = surface_weight * surface_pos + scatter_weight * scatter_pos
    new_dir = normalize(surface_weight * reflection_dir + scatter_weight * scatter_dir)

    continuing_factor = reflect_prob * reflection_attenuation + scatter_prob * scatter_attenuation

    distance_traveled = surface_weight * surface_distance + scatter_weight * scatter_distance
    new_time = time + distance_traveled / speed_of_light

    return new_pos, new_dir, new_time, detect_prob, reflection_attenuation, continuing_factor


# ===================================================================
# Custom VJP wrapper — NaN gradient sanitisation
# ===================================================================

@jax.custom_vjp
def photon_iteration_update_factors_safe(
        position, direction, time, surface_distance,
        normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
        absorption_length,
        hit_sensor, rng_key, speed_of_light):
    return photon_iteration_update_factors(
        position, direction, time, surface_distance,
        normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
        absorption_length,
        hit_sensor, rng_key, speed_of_light)


def _fwd(position, direction, time, surface_distance,
         normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
         absorption_length,
         hit_sensor, rng_key, speed_of_light):
    outputs = photon_iteration_update_factors(
        position, direction, time, surface_distance,
        normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
        absorption_length,
        hit_sensor, rng_key, speed_of_light)
    residuals = (position, direction, time, surface_distance,
                 normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
                 absorption_length,
                 hit_sensor, rng_key, speed_of_light)
    return outputs, residuals


def _bwd(residuals, g):
    g_pos, g_dir, g_time, g_detect, g_refl, g_cont = g

    g_pos = jnp.nan_to_num(g_pos, nan=0.0, posinf=0.0, neginf=0.0)
    g_dir = jnp.nan_to_num(g_dir, nan=0.0, posinf=0.0, neginf=0.0)
    g_time = jnp.nan_to_num(g_time, nan=0.0, posinf=0.0, neginf=0.0)
    g_detect = jnp.nan_to_num(g_detect, nan=0.0, posinf=0.0, neginf=0.0)
    g_refl = jnp.nan_to_num(g_refl, nan=0.0, posinf=0.0, neginf=0.0)
    g_cont = jnp.nan_to_num(g_cont, nan=0.0, posinf=0.0, neginf=0.0)

    _, vjp_fn = jax.vjp(photon_iteration_update_factors, *residuals)
    return vjp_fn((g_pos, g_dir, g_time, g_detect, g_refl, g_cont))


photon_iteration_update_factors_safe.defvjp(_fwd, _bwd)


# ===================================================================
# make_hits functions
# ===================================================================

def make_hits_simulation(
        flat_weights, flat_indices, flat_times, num_detectors,
        qe=0.2, qe_corrections=None, threshold=1e-10, temperature=0.01):
    """Differentiable soft-min first-arrival timing with per-sensor QE corrections."""
    per_photon_qe = qe * qe_corrections[flat_indices]
    qe_weights = flat_weights * per_photon_qe

    valid_mask = (qe_weights > threshold) & (flat_times > 0) & jnp.isfinite(flat_times)
    filtered_times = jnp.where(valid_mask, flat_times, jnp.inf)

    detector_mins = jax.ops.segment_min(filtered_times, flat_indices, num_segments=num_detectors)
    photon_offsets = detector_mins[flat_indices]

    shifted_times = jnp.where(valid_mask, flat_times - photon_offsets, jnp.inf)
    exp_terms = jnp.where(valid_mask, jnp.exp(-shifted_times / temperature), 0.0)
    exp_sums = jax.ops.segment_sum(exp_terms, flat_indices, num_segments=num_detectors)

    segment_min_time = detector_mins - temperature * jnp.log(exp_sums + 1e-20)
    has_photons = jnp.isfinite(detector_mins)
    segment_min_time = jnp.where(has_photons, segment_min_time, jnp.inf)

    total_charge = jax.ops.segment_sum(qe_weights, flat_indices, num_segments=num_detectors)

    nonzero_mask = (total_charge > threshold) & jnp.isfinite(segment_min_time)
    measured_charge = jnp.where(nonzero_mask, total_charge, 0.0)
    measured_time = jnp.where(nonzero_mask, segment_min_time, 0.0)

    return measured_charge, measured_time


def make_hits_data(
        flat_weights, flat_indices, flat_times, num_detectors,
        qe=0.2, rng_key=None, threshold=1e-5, apply_smearing=False):
    """Data-mode hits with Bernoulli QE, segment_min timing, and optional SK-like smearing."""
    timing_mask = (flat_weights > threshold) & (flat_times > 0)
    filtered_times = jnp.where(timing_mask, flat_times, jnp.inf)

    rng_key, smear_time_key = jax.random.split(rng_key)
    qe_key, smear_counts_key = jax.random.split(rng_key)

    # Always apply Bernoulli QE sampling — when qe >= 1.0, uniform(0,1) < qe
    # is always true so all photons pass.  Avoids Python `if` on traced values.
    detection_probs = jax.random.uniform(qe_key, shape=flat_weights.shape)
    detected_mask = detection_probs < qe
    qe_weights = flat_weights * detected_mask.astype(jnp.float32)
    qe_filtered_times = jnp.where(detected_mask & timing_mask, flat_times, jnp.inf)

    total_charge = jax.ops.segment_sum(qe_weights, flat_indices, num_segments=num_detectors)
    detector_mins = jax.ops.segment_min(qe_filtered_times, flat_indices, num_segments=num_detectors)

    nonzero_mask = (total_charge > 1e-10) & (detector_mins > 0)

    if apply_smearing:
        measured_time = jnp.where(
            jnp.any(nonzero_mask),
            smear_times(detector_mins, key=smear_time_key),
            0.0
        )
        measured_charge = jnp.where(
            nonzero_mask,
            smear_charges_SK_like(total_charge, key=smear_counts_key),
            0
        )
    else:
        measured_time = jnp.where(jnp.any(nonzero_mask), detector_mins, 0.0)
        measured_charge = jnp.where(nonzero_mask, total_charge, 0)

    return measured_charge, measured_time


def make_hits_likelihood(
        flat_weights, flat_indices, flat_times, num_detectors,
        qe=0.2, qe_corrections=None, threshold=1e-10):
    """Likelihood mode: return per-photon log-weights and per-sensor total charge.

    Instead of aggregating times to per-sensor first-arrival values, this
    returns the raw per-photon arrays so that ``first_arrival_nll`` (or
    similar likelihood-based losses) can operate on them directly.

    Parameters
    ----------
    flat_weights : jnp.ndarray
        Per-photon detection weights (K * max_sensors * n_rays,).
    flat_indices : jnp.ndarray
        Per-photon sensor indices (same shape).
    flat_times : jnp.ndarray
        Per-photon arrival times in ns (same shape).
    num_detectors : int
        Total number of sensors.
    qe : float
        Quantum efficiency.
    qe_corrections : jnp.ndarray
        Per-sensor QE correction factors (num_detectors,).
    threshold : float
        Minimum weight to consider a photon valid.

    Returns
    -------
    log_w : jnp.ndarray
        Log of QE-corrected weights (per photon). Invalid photons get -1e10.
    safe_times : jnp.ndarray
        Arrival times with invalid entries zeroed out (per photon).
    flat_indices : jnp.ndarray
        Sensor indices (per photon, unchanged).
    total_charge : jnp.ndarray
        Predicted total charge per sensor (num_detectors,).
    """
    per_photon_qe = qe * qe_corrections[flat_indices]
    qe_weights = flat_weights * per_photon_qe

    valid_mask = (qe_weights > threshold) & (flat_times > 0) & jnp.isfinite(flat_times)
    safe_weights = jnp.where(valid_mask, qe_weights, 0.0)
    safe_times = jnp.where(valid_mask, flat_times, 0.0)
    log_w = jnp.where(valid_mask, jnp.log(safe_weights + 1e-30), -1e10)

    total_charge = jax.ops.segment_sum(safe_weights, flat_indices, num_segments=num_detectors)

    return log_w, safe_times, flat_indices, total_charge


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
