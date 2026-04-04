"""SIREN-based photon ray generation functions.

Moved from lucid/generate.py during Phase 2.2 refactor.
"""

import jax
import jax.numpy as jnp
from jax import random, jit
from functools import partial
from lucid.siren.core import SIREN
from lucid.utils import normalize, generate_orthonormal_basis, jax_rotate_vector_local


@partial(jax.jit, static_argnums=(2,))
def generate_random_cone_vectors(R, theta, num_vectors, key):
    """Generate random vectors uniformly distributed on a cone surface.

    Parameters
    ----------
    R : jnp.ndarray
        Direction vector of the cone axis
    theta : float
        Opening angle of the cone in radians
    num_vectors : int
        Number of random vectors to generate
    key : jax.random.PRNGKey
        Random number generator key

    Returns
    -------
    jnp.ndarray
        Array of shape (num_vectors, 3) containing random unit vectors on cone surface
    """
    R = normalize(R)
    theta = jnp.clip(theta, 1e-6, jnp.pi - 1e-6)

    key1, key2 = random.split(key)
    phi = random.uniform(key1, (num_vectors,), minval=0, maxval=2 * jnp.pi)

    # Convert from polar to cartesian coordinates on cone surface
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    x = jnp.cos(phi) * sin_theta
    y = jnp.sin(phi) * sin_theta
    z = cos_theta * jnp.ones_like(x)

    basis = generate_orthonormal_basis(R)
    vectors = jnp.column_stack((x, y, z))
    rotated_vectors = jnp.einsum('ij,kj->ki', basis, vectors)

    return rotated_vectors

@jax.jit
def denormalize_log_predictions(predictions, log_max, log_min):
    log_predictions = predictions * (log_max - log_min) + log_min
    return 10 ** log_predictions - 1e-10

@jax.jit
def normalize_inputs_jit(inputs, energy_min, energy_max, angle_min, angle_max, distance_min, distance_max):
    """
    Normalize inputs to the range [-1, 1] in all dimensions.

    Args:
        inputs: Array of shape (..., 3) containing [energy, angle, distance] values.
        energy_min, energy_max: The minimum and maximum energy values.
        angle_min, angle_max: The minimum and maximum angle values.
        distance_min, distance_max: The minimum and maximum distance values.

    Returns:
        Array of shape (..., 3) with normalized values in the range [-1, 1].
    """
    # Extract the individual components
    energy = inputs[:, 0]
    angle = inputs[:, 1]
    distance = inputs[:, 2]

    # Normalize each component to [-1, 1]
    normalized_energy = 2.0 * (energy - energy_min) / (energy_max - energy_min) - 1.0
    normalized_angle = 2.0 * (angle - angle_min) / (angle_max - angle_min) - 1.0
    normalized_distance = 2.0 * (distance - distance_min) / (distance_max - distance_min) - 1.0

    # Stack the normalized components back together
    normalized_inputs = jnp.stack([normalized_energy, normalized_angle, normalized_distance], axis=1)

    return normalized_inputs

@partial(jax.jit, static_argnums=(3,))
def photonsim_differentiable_get_rays(track_origin, track_direction, energy, Nphot,
                                         table_data, model_params, key, num_seeds_a=0.035882, num_seeds_b=1.417106, num_seeds_c=2.75):
    """
    Generate photon rays using SIREN model for photon generation.

    Parameters
    ----------
    track_origin : jnp.ndarray
        3D position of the track origin
    track_direction : jnp.ndarray
        3D direction vector of the track
    energy : float
        Energy of the particle in MeV
    Nphot : int
        Number of photons to generate
    table_data : tuple
        Grid data for SIREN model evaluation
    model_params : dict
        SIREN model parameters
    key : jax.random.PRNGKey
        Random key for sampling
    num_seeds_a : float, optional
        Parameter 'a' for power law num_seeds calculation, default 0.035882
    num_seeds_b : float, optional
        Parameter 'b' for power law num_seeds calculation, default 1.417106
    num_seeds_c : float, optional
        Parameter 'c' for power law num_seeds calculation, default 2.75

    Returns
    -------
    ray_vectors : jnp.ndarray
        Array of photon direction vectors
    ray_origins : jnp.ndarray
        Array of photon origin positions
    photon_weights : jnp.ndarray
        Array of photon weights
    """
    key, subkey = random.split(key)

    n_bins, energy_min, energy_max, angle_min, angle_max, distance_min, distance_max, angle_bins, distance_bins, angle_dist_grid, angle_mesh, distance_mesh, log_min, log_max = table_data

    # ============================================================================
    # FIRST EVALUATION: Full grid to get photon weights for sampling
    # ============================================================================
    evaluation_grid = jnp.stack([
        jnp.full_like(angle_mesh, energy).ravel(),  # Energy (MeV)
        angle_mesh.ravel(),                         # Angle (radians)
        distance_mesh.ravel(),                      # Distance (mm)
    ], axis=1)

    normalized_grid = normalize_inputs_jit(evaluation_grid, energy_min, energy_max, angle_min, angle_max, distance_min, distance_max)

    # Initialize SIREN model
    model = SIREN(
        hidden_features=256,
        hidden_layers=3,
        out_features=1,
    )

    # FIRST MODEL CALL
    photon_weights, _ = model.apply(model_params, normalized_grid)

    # Sample points based on weights
    key, sampling_key = random.split(key)
    key, noise_key_angle = random.split(key)
    key, noise_key_dist = random.split(key)

    # Calculate number of seeds using power law: num_seeds = int32(a * energy^b + c)
    num_seeds = jnp.int32(num_seeds_a * jnp.power(energy, num_seeds_b) + num_seeds_c)

    seed_indices = random.randint(sampling_key, (Nphot,), 0, num_seeds)
    indices_by_weight = jnp.argsort(-photon_weights.squeeze())[seed_indices]

    angle_dist_mesh = jnp.array(angle_dist_grid)
    selected_angle_dist = angle_dist_mesh[indices_by_weight]

    # Split into separate angle and distance arrays
    sampled_angle = selected_angle_dist[:, 0]
    sampled_dist  = selected_angle_dist[:, 1]

    # ============================================================================
    # STRATIFIED SAMPLING: Better coverage than pure MC
    # ============================================================================
    bin_width_angle = (angle_max - angle_min) / n_bins
    bin_width_dist = (distance_max - distance_min) / n_bins

    # Create stratified samples: divide [0, 1] into Nphot strata
    # Then shuffle to avoid systematic bias
    key, subkey_angle = random.split(key)
    key, subkey_dist = random.split(key)
    key, subkey_jitter_angle = random.split(key)
    key, subkey_jitter_dist = random.split(key)

    # Permute stratum indices for random assignment
    strata_indices_angle = random.permutation(subkey_angle, Nphot)
    strata_indices_dist = random.permutation(subkey_dist, Nphot)

    # Sample within each stratum: (stratum_index + uniform[0,1]) / Nphot
    jitter_angle = random.uniform(subkey_jitter_angle, (Nphot,))
    jitter_dist = random.uniform(subkey_jitter_dist, (Nphot,))

    strata_angle = (strata_indices_angle + jitter_angle) / Nphot
    strata_dist = (strata_indices_dist + jitter_dist) / Nphot

    # Map from [0, 1] to [-bin_width/2, bin_width/2]
    stratified_angle = (strata_angle - 0.5) * bin_width_angle
    stratified_dist = (strata_dist - 0.5) * bin_width_dist

    smeared_angle = sampled_angle + stratified_angle
    smeared_dist = sampled_dist + stratified_dist

    # ============================================================================
    # SECOND EVALUATION: Smeared points to get final photon weights
    # ============================================================================
    new_evaluation_grid = jnp.stack([
        jnp.full_like(smeared_angle, energy),
        smeared_angle,
        smeared_dist,
    ], axis=1)

    new_normalized_grid = normalize_inputs_jit(new_evaluation_grid, energy_min, energy_max, angle_min, angle_max, distance_min, distance_max)

    # SECOND MODEL CALL
    new_photon_weights, _ = model.apply(model_params, new_normalized_grid)

    photon_thetas = smeared_angle

    # Generate ray vectors and origins
    subkey, subkey2 = random.split(subkey)
    ray_vectors = generate_random_cone_vectors(track_direction, photon_thetas, Nphot, subkey)

    # Convert ranges to meters and compute ray origins
    ranges = smeared_dist / 1000
    ray_origins = jnp.ones((Nphot, 3)) * track_origin[None, :] + ranges[:, None] * normalize(track_direction[None, :])

    # Apply boundary conditions
    new_photon_weights = jnp.squeeze(new_photon_weights)
    new_photon_weights = jnp.where(smeared_angle < angle_min, 0, new_photon_weights)
    new_photon_weights = jnp.where(smeared_angle > angle_max, 0, new_photon_weights)
    new_photon_weights = jnp.where(smeared_dist < distance_min, 0, new_photon_weights)
    new_photon_weights = jnp.where(smeared_dist > distance_max, 0, new_photon_weights)

    return ray_vectors, ray_origins, denormalize_log_predictions(new_photon_weights, log_max, log_min)


@jit
def predict_t0(distance, energy, baseline_slope, baseline_intercept,
                   A_slope, A_intercept, B_slope, B_intercept, offset):
    """
    JAX JIT-compatible version of predict_t0.
    Parameters are passed as individual arrays/scalars instead of nested dict.
    """
    # Baseline from 1000 MeV linear fit
    baseline = baseline_slope * distance + baseline_intercept

    # Calculate delta timing
    log10_A = A_slope * energy + A_intercept
    B = B_slope * energy + B_intercept
    delta = 10**log10_A * jnp.power(distance, B) + offset

    return baseline + delta

# Helper function to unpack your existing params dict
def predict_t0_wrapper(distance, energy, params):
    """Wrapper to use your existing params dict structure"""
    return predict_t0(
        distance, energy,
        params['baseline_1000MeV']['slope'],
        params['baseline_1000MeV']['intercept'],
        params['delta_parameterization']['A_slope'],
        params['delta_parameterization']['A_intercept'],
        params['delta_parameterization']['B_slope'],
        params['delta_parameterization']['B_intercept'],
        params['delta_parameterization']['offset']
    )
