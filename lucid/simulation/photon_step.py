"""Photon iteration functions (sample, update_factors, custom VJP)."""
import jax
import jax.numpy as jnp
from lucid.simulation.optics import (
    normalize, compute_reflection_direction, sample_cosine_hemisphere,
    sample_scatter_distance, compute_scatter_direction,
)

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
    inward_normal = -normal  # geometry normals point outward; negate to get into-medium direction
    new_pos = jnp.where(
        scatters,
        position + scatter_distance * normalize(direction),
        position + surface_distance * normalize(direction) + epsilon * normalize(inward_normal),
    )

    specular_dir = compute_reflection_direction(direction, normal)
    diffuse_dir = sample_cosine_hemisphere(inward_normal, k4)
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
    inward_normal = -normal  # geometry normals point outward; negate to get into-medium direction
    surface_pos = position + surface_distance * normalize(direction) + epsilon * normalize(inward_normal)
    scatter_pos = position + scatter_distance * normalize(direction)

    specular_dir = compute_reflection_direction(direction, normal)
    diffuse_dir = sample_cosine_hemisphere(inward_normal, k3)
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
