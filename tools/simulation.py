from tools.generate import differentiable_get_rays, new_differentiable_get_rays
from tools.propagate import create_photon_propagator
from tools.geometry import generate_detector

import jax
import jax.numpy as jnp

from tools.siren import *
from tools.table import *

base_dir_path = os.path.dirname(os.path.abspath(__file__))+'/'

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


def create_siren_grid(table):
    ene_bins = table.normalize(0, table.binning[0])
    cos_bins = table.normalize(1, np.linspace(0.3, max(table.binning[1]), 500))
    trk_bins = table.normalize(2, np.linspace(min(table.binning[2]), 400, 500))
    cos_trk_mesh = np.array([[x,y] for x in cos_bins for y in trk_bins])
    x_data = table.binning[0]
    y_data = ene_bins
    grid_shape = (np.shape(cos_bins)[0]*np.shape(trk_bins)[0],3)
    return cos_bins, trk_bins, cos_trk_mesh, (x_data, y_data), grid_shape


def photon_iteration_update_factors(position, direction, time, surface_distance,
                                    normal, scatter_length, reflection_rate,
                                    absorption_length, temperature, rng_key):
    """
    A vectorized version of the scattering update that returns scaling factors instead of applying intensity.

    Parameters:
      - position, direction: (3,) arrays for the photon's current state.
      - time: scalar, current photon time.
      - surface_distance: scalar, computed from norm(hit_position - position).
      - normal: (3,) array, the surface normal at the hit position.
      - scatter_length: scalar, mean free path for scattering.
      - reflection_rate: scalar, probability of reflection at the surface.
      - absorption_length: scalar, mean free path for absorption.
      - temperature: scalar, temperature for gumbel-softmax sampling.
      - rng_key: JAX PRNG key.

    Returns:
      - new_pos: (3,) array, updated photon position.
      - new_dir: (3,) array, updated photon direction.
      - new_time: scalar, updated photon time.
      - detect_prob: scalar, probability of detection.
      - reflection_attenuation: scalar, attenuation factor.
      - continuing_factor: scalar, factor for continuation (reflection or scatter).
    """
    k1, k2, k3 = jax.random.split(rng_key, 3)

    # Sample the distance along the ray where scattering might occur.
    scatter_distance = sample_scatter_distance(surface_distance, scatter_length, k2)

    # Core probabilities
    # Probability of reaching the surface without scattering
    reach_surface_prob = jnp.exp(-surface_distance / scatter_length)
    # Probability of scattering before reaching the surface
    scatter_prob = 1 - reach_surface_prob

    # Total probabilities for each possible outcome
    # 1. Reflection: reach surface AND reflect
    reflect_prob = reach_surface_prob * reflection_rate
    # 2. Detection: reach surface AND NOT reflect
    detect_prob = reach_surface_prob * (1 - reflection_rate)

    # Attenuation factors (only affect intensity, not probabilities)
    reflection_attenuation = jnp.exp(-surface_distance / absorption_length)
    scatter_attenuation = jnp.exp(-scatter_distance / absorption_length)

    # Use gumbel-softmax to get soft weights for reflection and scatter.
    # We only consider continuing paths (not detector absorption)
    probs = jnp.array([reflect_prob, scatter_prob])
    action_weights = gumbel_softmax(probs, temperature, k1)
    reflection_weight = action_weights[0]
    scatter_weight = action_weights[1]

    # Compute the candidate positions and directions.
    reflection_pos = position + surface_distance * direction
    scatter_pos = position + scatter_distance * direction
    reflection_dir = compute_reflection_direction(direction, normal)
    scatter_dir = compute_scatter_direction(direction, k3)

    # Blend the two possibilities for continuing paths.
    new_pos = reflection_weight * reflection_pos + scatter_weight * scatter_pos
    new_dir = normalize(reflection_weight * reflection_dir + scatter_weight * scatter_dir)

    # Calculate the continuing factor (reflection + scatter)
    continuing_factor = reflect_prob * reflection_attenuation + scatter_prob * scatter_attenuation

    # Time increment based on distance traveled along the weighted path
    time_increment = reflection_weight * surface_distance + scatter_weight * scatter_distance
    new_time = time + time_increment

    return new_pos, new_dir, new_time, detect_prob, reflection_attenuation, continuing_factor


def create_event_simulator(propagate_photons, Nphot, NUM_DETECTORS, detector_points, K, is_data,
                           max_detectors_per_cell):
    """
    Creates a simulator that accumulates all depositions in structured arrays and performs segment_sum at the end.
    Keeps the same structure as the working implementation but with different array organization.
    """

    table = Table(base_dir_path+'../siren/cprof_mu_train_10000ev.h5')
    grid_data = create_siren_grid(table)
    siren_model, model_params = load_siren_jax(base_dir_path+'../siren/siren_cprof_mu.pkl')

    @jax.jit
    def _simulation_core(params, key):
        # Unpack simulation parameters
        (energy, track_origin, track_direction, initial_intensity,
         scatter_length, reflection_rate, absorption_length, sim_temperature) = params

        photon_directions, photon_origins, photon_weights = new_differentiable_get_rays(track_origin, track_direction, energy, Nphot, grid_data, model_params, key)
        tot_real_photons_norm = (energy*852.97855369-148646.90865158)
        expanded_photon_weights = tot_real_photons_norm * photon_weights

        n_rays = photon_origins.shape[0]
        # Set the initial intensity per photon.
        photon_intensities = jnp.full((n_rays,), initial_intensity * expanded_photon_weights / Nphot)
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
            if i == K-1:
                scatter_length = 10e20
                reflection_rate = 0
                absorption_length = 10e20

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
            current_positions = jax.lax.stop_gradient(new_positions)
            current_directions = jax.lax.stop_gradient(new_directions)
            current_intensities = new_intensities
            current_times = jax.lax.stop_gradient(new_times)



        # Compute average hit times for each detector using weighted averages
        flat_weights = all_weights.reshape(-1)
        flat_indices = all_indices.reshape(-1)
        flat_times = all_times.reshape(-1)

        # Calculate numerator and denominator for weighted average time
        total_time_weighted = jax.ops.segment_sum(
            flat_weights * flat_times,
            flat_indices,
            num_segments=NUM_DETECTORS
        )

        total_charge_weighted = jax.ops.segment_sum(
            flat_weights,
            flat_indices,
            num_segments=NUM_DETECTORS
        )

        # Compute average times with protection against division by zero
        eps = 1e-10
        average_times = jnp.where(
            total_charge_weighted > eps,
            total_time_weighted / (total_charge_weighted + eps),
            jnp.zeros_like(total_charge_weighted)
        )

        # Find minimum non-zero time
        # Create a mask for non-zero times and weights
        nonzero_mask = total_charge_weighted>1e-5
        # Get minimum of non-zero times, defaulting to 0 if no valid times exist
        min_nonzero_time = jnp.where(
            jnp.any(nonzero_mask),
            jnp.min(jnp.where(nonzero_mask, average_times, jnp.inf)),
            0.0
        )

        # Subtract minimum time from non-zero times only
        aligned_times = jnp.where(
            nonzero_mask,
            average_times - min_nonzero_time,
            0
        )

        corrected_q = jnp.where(
            nonzero_mask,
            total_charge_weighted,
            0
        )

        return corrected_q, aligned_times

    @jax.jit
    def _simulate_event_core(params, key):
        total_charge_weighted, average_times = _simulation_core(params, key)
        return total_charge_weighted, average_times

    @jax.jit
    def _simulate_event_core_data(params, key):
        total_charge_weighted, average_times = _simulation_core(params, key)
        # this allows to implement additional conditions for data that are not applied to the simulation
        return total_charge_weighted, average_times

    if is_data:
        return jax.jit(lambda p, k: _simulate_event_core_data(p, k))
    else:
        return jax.jit(lambda p, k: _simulate_event_core(p, k))