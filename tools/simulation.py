from tools.generate import differentiable_get_rays
from tools.propagate import create_photon_propagator
from tools.geometry import generate_detector

import jax
import jax.numpy as jnp


def setup_event_simulator(json_filename, n_photons=1_000_000, temperature=0.2, K=2, is_data=False, max_detectors_per_cell=4):
    """
    Sets up and returns an event simulator with the specified configuration.

    Parameters
    ----------
    json_filename : str
        Path to the JSON file containing detector configuration.
    n_photons : int, optional
        Number of photons to simulate per event, default 1,000,000.
    temperature : float, optional
        Temperature parameter for photon propagation, default 0.2.
    K : int, optional
        Number of scattering iterations to simulate per event.
    is_data : bool, optional
        If true, subtract minimum times such that t0 = 0.
    max_detectors_per_cell : int, optional
        Maximum number of detectors per cell in the detector configuration. Default is 4.

    Returns
    -------
    callable
        simulate_event: a JIT-compiled function that takes (params, key) and returns event data.
        The parameter tuple for the simulation function is expected to be:
           (cone_opening, track_origin, track_direction, initial_intensity,
            scatter_length, reflection_rate, absorption_length, sim_temperature)
        and K is precompiled as part of the simulator.
    """
    # Initialize detector configuration.
    detector = generate_detector(json_filename)
    detector_points = jnp.array(detector.all_points)
    detector_radius = detector.S_radius

    # Setup photon propagator with specified parameters.
    propagate_photons = create_photon_propagator(
        detector_points,
        detector_radius,
        temperature=temperature,
        max_detectors_per_cell=max_detectors_per_cell
    )

    # Get number of detectors from the points array.
    NUM_DETECTORS = len(detector_points)

    # Create the event simulator with a fixed number of scattering iterations K.
    simulate_event = create_event_simulator(
        propagate_photons,
        n_photons,
        NUM_DETECTORS,
        detector_points,
        K,
        is_data,
        max_detectors_per_cell
    )

    return simulate_event

def normalize(v, epsilon=1e-6):
    """Normalize a vector (or batch of vectors)."""
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + epsilon)

def compute_reflection_direction(incident_dir, normal):
    """Compute reflection direction given an incident direction and surface normal."""
    return normalize(incident_dir - 2 * jnp.dot(incident_dir, normal) * normal)


def create_local_frame(z):
    """Create a local coordinate frame given a z-axis vector."""
    z = normalize(z)
    t = jnp.where(jnp.abs(z[0]) < 0.9,
                  jnp.array([1.0, 0.0, 0.0]),
                  jnp.array([0.0, 1.0, 0.0]))
    x = normalize(jnp.cross(t, z))
    y = jnp.cross(z, x)
    return jnp.stack([x, y, z])


def gumbel_softmax(probs, temperature, rng_key):
    """Differentiable sampling from a discrete distribution using Gumbel-softmax."""
    uniform = jax.random.uniform(rng_key, shape=probs.shape)
    gumbel = -jnp.log(-jnp.log(uniform))
    logits = jnp.log(probs) + gumbel
    return jax.nn.softmax(logits / temperature)


def sample_scatter_distance(D, S, rng_key):
    """Sample a scatter distance from a truncated exponential distribution."""
    u = jax.random.uniform(rng_key)
    return -S * jnp.log(1 - u * (1 - jnp.exp(-D / S)))


def compute_scatter_direction(incident_dir, rng_key):
    """Compute a new scattering direction based on a Rayleigh phase function."""
    k1, k2 = jax.random.split(rng_key)
    u1 = jax.random.uniform(k1)
    u2 = jax.random.uniform(k2)
    cos_theta = jnp.cbrt(2 * u1 - 1)
    sin_theta = jnp.sqrt(1 - cos_theta ** 2)
    phi = 2 * jnp.pi * u2
    local_dir = normalize(jnp.array([sin_theta * jnp.cos(phi),
                                     sin_theta * jnp.sin(phi),
                                     cos_theta]))
    frame = create_local_frame(incident_dir)
    return normalize(frame @ local_dir)


# def photon_iteration_update_factors(position, direction, time, surface_distance,
#                                     normal, scatter_length, reflection_rate,
#                                     absorption_length, temperature, rng_key):
#     """
#     A vectorized version of the scattering update that returns scaling factors instead of applying intensity.
#
#     Parameters:
#       - position, direction: (3,) arrays for the photon's current state.
#       - time: scalar, current photon time.
#       - surface_distance: scalar, computed from norm(hit_position - position).
#       - normal: (3,) array, the surface normal at the hit position.
#       - scatter_length: scalar, mean free path for scattering.
#       - reflection_rate: scalar, probability of reflection at the surface.
#       - absorption_length: scalar, mean free path for absorption.
#       - temperature: scalar, temperature for gumbel-softmax sampling.
#       - rng_key: JAX PRNG key.
#
#     Returns:
#       - new_pos: (3,) array, updated photon position.
#       - new_dir: (3,) array, updated photon direction.
#       - new_time: scalar, updated photon time.
#       - detect_prob: scalar, probability of detection.
#       - reflection_attenuation: scalar, attenuation factor.
#       - continuing_factor: scalar, factor for continuation (reflection or scatter).
#     """
#     k1, k2, k3 = jax.random.split(rng_key, 3)
#
#     # Sample the distance along the ray where scattering might occur.
#     scatter_distance = sample_scatter_distance(surface_distance, scatter_length, k2)
#
#     # Core probabilities
#     # Probability of reaching the surface without scattering
#     reach_surface_prob = jnp.exp(-surface_distance / scatter_length)
#     # Probability of scattering before reaching the surface
#     scatter_prob = 1 - reach_surface_prob
#
#     # Total probabilities for each possible outcome
#     # 1. Reflection: reach surface AND reflect
#     reflect_prob = reach_surface_prob * reflection_rate
#     # 2. Detection: reach surface AND NOT reflect
#     detect_prob = reach_surface_prob * (1 - reflection_rate)
#
#     # Attenuation factors (only affect intensity, not probabilities)
#     reflection_attenuation = jnp.exp(-surface_distance / absorption_length)
#     scatter_attenuation = jnp.exp(-scatter_distance / absorption_length)
#
#     # Use gumbel-softmax to get soft weights for reflection and scatter.
#     # We only consider continuing paths (not detector absorption)
#     probs = jnp.array([reflect_prob, scatter_prob])
#     action_weights = gumbel_softmax(probs, temperature, k1)
#     reflection_weight = action_weights[0]
#     scatter_weight = action_weights[1]
#
#     # Compute the candidate positions and directions.
#     reflection_pos = position + surface_distance * direction
#     scatter_pos = position + scatter_distance * direction
#     reflection_dir = compute_reflection_direction(direction, normal)
#     scatter_dir = compute_scatter_direction(direction, k3)
#
#     # Blend the two possibilities for continuing paths.
#     new_pos = reflection_weight * reflection_pos + scatter_weight * scatter_pos
#     new_dir = normalize(reflection_weight * reflection_dir + scatter_weight * scatter_dir)
#
#     # Calculate the continuing factor (reflection + scatter)
#     continuing_factor = reflect_prob * reflection_attenuation + scatter_prob * scatter_attenuation
#
#     # Time increment based on distance traveled along the weighted path
#     time_increment = reflection_weight * surface_distance + scatter_weight * scatter_distance
#     new_time = time + time_increment
#
#     return new_pos, new_dir, new_time, detect_prob, reflection_attenuation, continuing_factor

def photon_iteration_update_factors(position, direction, time, surface_distance,
                                    normal, scatter_length, reflection_rate,
                                    absorption_length, temperature, rng_key):
    """
    Reflection-only variant that returns scaling factors instead of applying intensity.
    This variant only handles reflection (no scattering).

    Parameters:
      - position, direction: (3,) arrays for the photon's current state.
      - time: scalar, current photon time.
      - surface_distance: scalar, computed from norm(hit_position - position).
      - normal: (3,) array, the surface normal at the hit position.
      - scatter_length: scalar, ignored in this variant.
      - reflection_rate: scalar, probability of reflection at the surface.
      - absorption_length: scalar, mean free path for absorption.
      - temperature: scalar, ignored in this variant.
      - rng_key: JAX PRNG key.

    Returns:
      - new_pos: (3,) array, updated photon position.
      - new_dir: (3,) array, updated photon direction.
      - new_time: scalar, updated photon time.
      - detect_prob: scalar, probability of detection.
      - reflection_attenuation: scalar, attenuation factor.
      - continuing_factor: scalar, factor for continuation (always reflection).
    """
    # In reflection-only mode, we always reach the surface
    # No need to split the RNG key for scatter distance or direction

    # In this variant, photons always reach the surface
    reach_surface_prob = 1.0

    # No scattering in this variant
    scatter_prob = 0.0

    # Total probabilities for each possible outcome
    # 1. Reflection: reach surface AND reflect (based on reflection_rate)
    reflect_prob = reach_surface_prob * reflection_rate
    # 2. Detection: reach surface AND NOT reflect
    detect_prob = reach_surface_prob * (1.0 - reflection_rate)

    # Attenuation factor (only affects intensity, not probabilities)
    reflection_attenuation = jnp.exp(-surface_distance / absorption_length)

    # Since we're only reflecting, we don't need Gumbel-softmax
    # Compute the reflection position and direction directly
    new_pos = position + surface_distance * direction
    new_dir = compute_reflection_direction(direction, normal)

    # Calculate the continuing factor (only reflection)
    continuing_factor = reflect_prob * reflection_attenuation

    # Time increment based on distance traveled to surface
    new_time = time + surface_distance

    return new_pos, new_dir, new_time, detect_prob, reflection_attenuation, continuing_factor


def create_event_simulator(propagate_photons, Nphot, NUM_DETECTORS, detector_points, K, is_data,
                           max_detectors_per_cell):
    """
    Creates a simulator that accumulates all depositions in structured arrays and performs segment_sum at the end.
    Keeps the same structure as the working implementation but with different array organization.
    """

    @jax.jit
    def _simulation_core(params, key):
        # Unpack simulation parameters.
        (cone_opening, track_origin, track_direction, initial_intensity,
         scatter_length, reflection_rate, absorption_length, sim_temperature) = params

        # Generate initial rays.
        photon_directions, photon_origins = differentiable_get_rays(
            track_origin, track_direction, cone_opening, Nphot, key
        )
        n_rays = photon_origins.shape[0]
        # Set the initial intensity per photon.
        photon_intensities = jnp.full((n_rays,), initial_intensity / Nphot)
        photon_times = jnp.zeros((n_rays,))

        # Initialize photon state.
        current_positions = photon_origins
        current_directions = photon_directions
        current_intensities = photon_intensities
        current_times = photon_times

        # Initialize arrays with shape (K, max_detectors_per_cell, n_rays)
        all_weights = jnp.zeros((K, max_detectors_per_cell, n_rays))
        all_indices = jnp.zeros((K, max_detectors_per_cell, n_rays), dtype=jnp.int32)
        all_times = jnp.zeros((K, max_detectors_per_cell, n_rays))

        # Loop over K scattering iterations.
        for i in range(K):
            key, subkey = jax.random.split(key)
            # Propagate photons from the current state.
            prop_results = propagate_photons(current_positions, current_directions)
            depositions = prop_results['detector_weights']
            detector_indices = prop_results['detector_indices']
            times = prop_results['times']
            hit_positions = prop_results['positions']
            normals = prop_results['normals']

            # Compute distance each photon traveled (used for scattering).
            surface_distances = jnp.linalg.norm(hit_positions - current_positions, axis=1)

            # Generate independent RNG keys per photon.
            key, subkey = jax.random.split(key)
            rng_keys = jax.random.split(subkey, n_rays)

            # Run the scattering update for each photon.
            new_positions, new_directions, new_times, detect_probs, reflection_attenuations, continuing_factors = jax.vmap(
                photon_iteration_update_factors,
                in_axes=(0, 0, 0, 0, 0, None, None, None, None, 0)
            )(current_positions, current_directions, current_times,
              surface_distances, normals, scatter_length, reflection_rate,
              absorption_length, sim_temperature, rng_keys)

            # Calculate the detected intensity for each photon
            detected_intensity_factors = detect_probs * reflection_attenuations

            # Scale the propagation weights by current intensities and detection factors
            updated_weights = depositions * current_intensities[None, :] * detected_intensity_factors[None, :]
            total_times = times + current_times[:, None]

            # Store data in the i-th slice of our 3D arrays
            all_weights = all_weights.at[i].set(updated_weights)
            all_indices = all_indices.at[i].set(detector_indices)
            all_times = all_times.at[i].set(total_times.squeeze(-1))

            # Scale intensities for continuing paths
            new_intensities = current_intensities * continuing_factors

            # Update photon state for the next iteration.
            current_positions = new_positions
            current_directions = new_directions
            current_intensities = new_intensities
            current_times = new_times

        # Flatten arrays for segment sum
        flat_weights = all_weights.reshape(-1)
        flat_indices = all_indices.reshape(-1)
        flat_times = all_times.reshape(-1)

        # Perform segment sum at the end
        total_detector_weights = jax.ops.segment_sum(flat_weights, flat_indices, num_segments=NUM_DETECTORS)
        total_time_weighted = jax.ops.segment_sum(flat_weights * flat_times, flat_indices, num_segments=NUM_DETECTORS)

        # Compute average hit times per detector (with protection against division by zero).
        eps = 1e-10
        average_times = jnp.where(total_detector_weights > eps,
                                  total_time_weighted / (total_detector_weights + eps),
                                  jnp.zeros_like(total_detector_weights))
        return total_detector_weights, average_times, total_detector_weights

    @jax.jit
    def _simulate_event_core(params, key):
        charges, average_times, _ = _simulation_core(params, key)
        return charges, average_times

    @jax.jit
    def _simulate_event_core_data(params, key):
        charges, average_times, detector_weight_totals = _simulation_core(params, key)

        # Align times such that the smallest nonzero time becomes zero.
        eps = 1e-10
        nonzero_mask = (average_times > eps) & (detector_weight_totals > eps)
        min_nonzero_time = jnp.where(
            jnp.any(nonzero_mask),
            jnp.min(jnp.where(nonzero_mask, average_times, jnp.inf)),
            0.0
        )
        aligned_times = jnp.where(nonzero_mask, average_times - min_nonzero_time, average_times)
        return charges, aligned_times

    if is_data:
        return jax.jit(lambda p, k: _simulate_event_core_data(p, k))
    else:
        return jax.jit(lambda p, k: _simulate_event_core(p, k))