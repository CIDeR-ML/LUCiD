from tools.generate import differentiable_get_rays, new_differentiable_get_rays, get_isotropic_rays

from tools.propagate import create_photon_propagator
from tools.geometry import generate_detector

import jax
import jax.numpy as jnp

from tools.siren import *
from tools.table import *

from functools import partial

from tools.utils import spherical_to_cartesian

base_dir_path = os.path.dirname(os.path.abspath(__file__))+'/'

def setup_event_simulator(json_filename, n_photons=1_000_000, temperature=0.2, K=2, is_data=False, is_calibration=False,
                          max_detectors_per_cell=4, use_expected_value=True):
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
        Maximum scattering iterations before forced detection (default: 2)
    is_data : bool, optional
        True for ROOT file data mode (default: False)
    is_calibration : bool, optional
        True for isotropic calibration mode (default: False)
    max_detectors_per_cell : int, optional
        Grid cell detector limit (default: 4)
    use_expected_value : bool, optional
        Propagation mode selection:
        - True: Expected value (differentiable)
        - False: Monte Carlo sampling
        - None: Auto-select based on mode

    Returns
    -------
    callable
        simulate_event: a JIT-compiled function that takes (params, key) and returns event data.
        The parameter tuple for the simulation function is expected to be:
           (cone_opening, track_origin, track_direction, initial_intensity,
            scatter_length, reflection_rate, absorption_length, tau_gs)
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
        is_calibration,
        max_detectors_per_cell,
        use_expected_value
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


def photon_iteration_sample(position, direction, time, surface_distance,
                            normal, scatter_length, reflection_rate,
                            absorption_length, tau_gs, rng_key):
    """
    Sampling version of photon iteration that makes binary decisions.

    This function performs Monte Carlo sampling where photons make discrete
    choices (detect/reflect/scatter) rather than computing expected values.

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
    reflection_rate : float
        Probability of reflection when reaching a surface
    absorption_length : float
        Mean free path for absorption in the medium
    tau_gs : float
        Temperature parameter for Gumbel-softmax (included for signature compatibility, not used)
    rng_key : jax.random.PRNGKey
        Random key for sampling

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

    Notes
    -----
    The tau_gs parameter is included to match the signature of photon_iteration_update_factors
    but is not used in the sampling mode.
    """
    k1, k2, k3 = jax.random.split(rng_key, 3)

    # Sample scatter distance from truncated exponential
    scatter_distance = sample_scatter_distance(surface_distance, scatter_length, k2)

    # Probability of reaching surface without scattering
    reach_surface_prob = jnp.exp(-surface_distance / scatter_length)

    # Sample whether photon reaches surface
    u1 = jax.random.uniform(k1)
    reaches_surface = u1 < reach_surface_prob

    # If reaches surface, sample whether it reflects or is detected
    u2 = jax.random.uniform(k2)
    reflects = reaches_surface & (u2 < reflection_rate)
    detects = reaches_surface & (u2 >= reflection_rate)
    scatters = ~reaches_surface

    # Calculate new position based on outcome
    new_pos = jnp.where(
        scatters,
        position + scatter_distance * direction,
        position + surface_distance * direction
    )

    # Calculate new direction
    reflection_dir = compute_reflection_direction(direction, normal)
    scatter_dir = compute_scatter_direction(direction, k3)

    new_dir = jnp.where(
        reflects,
        reflection_dir,
        jnp.where(scatters, scatter_dir, direction)
    )

    # Time increment based on distance traveled
    time_increment = jnp.where(scatters, scatter_distance, surface_distance)
    new_time = time + time_increment

    # Calculate attenuation based on distance traveled
    distance_traveled = jnp.where(scatters, scatter_distance, surface_distance)
    attenuation = jnp.exp(-distance_traveled / absorption_length)

    # Binary detection probability
    detect_prob = detects.astype(jnp.float32)

    # Attenuation is always applied
    reflection_attenuation = attenuation

    # Continuing factor: 0 if detected (photon stops), attenuation if continues
    continuing_factor = jnp.where(detects, 0.0, attenuation)

    return new_pos, new_dir, new_time, detect_prob, reflection_attenuation, continuing_factor

def photon_iteration_update_factors(position, direction, time, surface_distance,
                                    normal, scatter_length, reflection_rate,
                                    absorption_length, tau_gs, rng_key):
    """
    An update function for photon propagation that employs a variance reduction technique through computation of expected values.
    It employs Gumbel-softmax sampling to blend reflection and scattering outcomes for full differentiability.

    Parameters:
      - position, direction: (3,) arrays for the photon's current state.
      - time: scalar, current photon time.
      - surface_distance: scalar, computed from norm(hit_position - position).
      - normal: (3,) array, the surface normal at the hit position.
      - scatter_length: scalar, mean free path for scattering.
      - reflection_rate: scalar, probability of reflection at the surface.
      - absorption_length: scalar, mean free path for absorption.
      - tau_gs: scalar, temperature for Gumbel-softmax sampling.
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
    action_weights = gumbel_softmax(probs, tau_gs, k1)
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

# JAX-compatible versions of the rotation functions
def jax_normalize(v, epsilon=1e-8):
    """Normalize a vector with numerical stability using JAX."""
    norm = jnp.linalg.norm(v)
    return jnp.where(norm > epsilon, v / norm, v)

def jax_rotate_vector(vector, axis, angle):
    """ Rotate a vector around an axis by a given angle in radians using JAX. """
    axis = jax_normalize(axis)
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    cross_product = jnp.cross(axis, vector)
    dot_product = jnp.dot(axis, vector) * (1 - cos_angle)
    return cos_angle * vector + sin_angle * cross_product + dot_product * axis


def create_event_simulator(propagate_photons, Nphot, NUM_DETECTORS, detector_points, K,
                           is_data, is_calibration, max_detectors_per_cell, use_expected_value=None):
    """
    Create an event simulator with the appropriate configuration.

    This function sets up the simulator by choosing the correct photon update
    function and simulation mode based on the provided parameters. All simulation
    functions are defined within this scope to avoid redundant parameter passing.

    Parameters
    ----------
    propagate_photons : callable
        Function to propagate photons through the detector geometry
    Nphot : int
        Number of photons to simulate per event
    NUM_DETECTORS : int
        Total number of detectors in the system
    detector_points : jnp.ndarray
        Array of detector positions, shape (NUM_DETECTORS, 3)
    K : int
        Number of scattering iterations before forcing detection
    is_data : bool
        If True, simulator reads photon data from ROOT files
    is_calibration : bool
        If True, simulator uses isotropic point source
    max_detectors_per_cell : int
        Maximum number of detectors per grid cell
    use_expected_value : bool, optional
        Propagation mode selection:
        - True: Expected value mode (differentiable)
        - False: Monte Carlo sampling mode
        - None: Auto-select (sampling for data, expected value otherwise)

    Returns
    -------
    callable
        JIT-compiled simulation function
    """
    # Select photon update function based on mode
    if is_data:
        photon_update_fn = photon_iteration_sample  # Data always uses sampling
    elif use_expected_value is False:
        photon_update_fn = photon_iteration_sample
    else:
        photon_update_fn = photon_iteration_update_factors

    # Define simulation function for ROOT data
    @jax.jit
    def _simulation_with_data(particle_params, detector_params, key, photon_data):
        """Simulate events using photon data from ROOT files."""
        energy, track_origin, track_direction, initial_intensity = particle_params

        # Transform photons from ROOT coordinate system
        original_track_dir = jnp.array([0.0, 0.0, 1.0])
        photon_origins = photon_data['photon_origins'] / 100.0  # cm to m
        photon_directions = photon_data['photon_directions']

        # Calculate rotation to align with track direction
        track_direction_norm = jax_normalize(track_direction)
        rotation_axis = jnp.cross(original_track_dir, track_direction_norm)
        axis_norm = jnp.linalg.norm(rotation_axis)

        rotation_axis = jnp.where(
            axis_norm < 1e-6,
            jnp.array([1.0, 0.0, 0.0]),
            rotation_axis / (axis_norm + 1e-8)
        )

        rotation_angle = jnp.arccos(jnp.clip(
            jnp.dot(original_track_dir, track_direction_norm), -1.0, 1.0
        ))

        # Apply rotation and translation
        rotated_directions = jax.vmap(
            lambda v: jax_rotate_vector(v, rotation_axis, rotation_angle)
        )(photon_directions)

        rotated_origins = jax.vmap(
            lambda v: jax_rotate_vector(v, rotation_axis, rotation_angle)
        )(photon_origins)

        final_origins = rotated_origins + track_origin[None, :]

        # Create mask for valid photons
        n_rays = photon_origins.shape[0]
        mask = jnp.arange(n_rays) < photon_data['N']
        photon_intensities = initial_intensity * mask.astype(jnp.float32)
        photon_times = jnp.zeros((n_rays,))

        return _common_propagation(
            final_origins, rotated_directions, photon_intensities, photon_times,
            n_rays, detector_params, key, NUM_DETECTORS, K, max_detectors_per_cell,
            propagate_photons, photon_update_fn
        )

    # Define simulation function for SIREN model
    @jax.jit
    def _simulation_without_data(particle_params, detector_params, key, grid_data, model_params):
        """Simulate events using SIREN model for photon generation."""
        energy, track_origin, direction_angles = particle_params
        
        # Convert theta and phi angles to direction vector
        theta, phi = direction_angles
        track_direction = spherical_to_cartesian(theta, phi)

        # Generate photons using SIREN
        photon_directions, photon_origins, photon_weights = new_differentiable_get_rays(
            track_origin, track_direction, energy, Nphot, grid_data, model_params, key
        )

        # Scale weights to physical photon count
        total_photons = energy * 852.97855369 - 148646.90865158
        photon_intensities = (total_photons * photon_weights) / Nphot
        photon_times = jnp.zeros((Nphot,))

        return _common_propagation(
            photon_origins, photon_directions, photon_intensities, photon_times,
            Nphot, detector_params, key, NUM_DETECTORS, K, max_detectors_per_cell,
            propagate_photons, photon_update_fn
        )

    # Define calibration simulation function
    @jax.jit
    def _simulation_detector_calibration(source_params, detector_params, key):
        """Simulate isotropic point source for detector calibration."""
        source_origin, source_intensity = source_params

        # Generate isotropic photons
        photon_directions, photon_origins, photon_intensities = get_isotropic_rays(
            source_origin, source_intensity, Nphot, key
        )
        photon_times = jnp.zeros((Nphot,))

        return _common_propagation(
            photon_origins, photon_directions, photon_intensities, photon_times,
            Nphot, detector_params, key, NUM_DETECTORS, K, max_detectors_per_cell,
            propagate_photons, photon_update_fn
        )

    @partial(jax.jit, static_argnames=(
    'n_rays', 'K', 'max_detectors_per_cell', 'num_detectors', 'propagate_fn', 'photon_update_fn'))
    def _common_propagation(positions, directions, intensities, times, n_rays, detector_params, key,
                            num_detectors, K, max_detectors_per_cell, propagate_fn, photon_update_fn):
        """
        Common photon propagation logic for all simulation modes.

        This function handles the core propagation loop, tracking photons through
        K iterations of scattering/reflection/detection. It can work with different
        photon update functions (expected value or sampling).

        Parameters
        ----------
        positions : jnp.ndarray
            Initial positions of photons, shape (n_rays, 3)
        directions : jnp.ndarray
            Initial normalized direction vectors, shape (n_rays, 3)
        intensities : jnp.ndarray
            Initial intensities of photons, shape (n_rays,)
        times : jnp.ndarray
            Initial times of photons, shape (n_rays,)
        n_rays : int
            Number of photons to propagate
        detector_params : tuple
            (scatter_length, reflection_rate, absorption_length, tau_gs)
        key : jax.random.PRNGKey
            Random key for stochastic operations
        num_detectors : int
            Total number of detectors in the system
        K : int
            Number of scattering iterations
        max_detectors_per_cell : int
            Maximum detectors per grid cell
        propagate_fn : callable
            Function to propagate photons and find detector intersections
        photon_update_fn : callable
            Either photon_iteration_update_factors or photon_iteration_sample

        Returns
        -------
        corrected_q : jnp.ndarray
            Total charge deposited at each detector, shape (num_detectors,)
        aligned_times : jnp.ndarray
            Average detection times aligned to earliest detection, shape (num_detectors,)
        """
        original_scatter_length, original_reflection_rate, original_absorption_length, tau_gs = detector_params

        # Define the step function for lax.scan
        def propagation_step(carry, i):
            current_positions, current_directions, current_intensities, current_times, key = carry

            # Handle special case for last iteration
            # Force all remaining photons to be detected while maintaining absorption
            scatter_length = jnp.where(i == K - 1, 1e20, original_scatter_length)
            reflection_rate = jnp.where(i == K - 1, 0.0, original_reflection_rate)
            absorption_length = original_absorption_length  # Keep absorption even in last iteration

            # Split keys
            key, prop_key = jax.random.split(key)

            # Propagate photons to find intersections
            prop_results = propagate_fn(current_positions, current_directions)
            depositions = prop_results['detector_weights']
            detector_indices = prop_results['detector_indices']
            times = prop_results['times']
            hit_positions = prop_results['positions']
            normals = prop_results['normals']

            # Compute distances to intersection points
            surface_distances = jnp.linalg.norm(hit_positions - current_positions, axis=1)

            # Generate RNG keys for each photon
            key, subkey = jax.random.split(key)
            rng_keys = jax.random.split(subkey, n_rays)

            # Apply photon update function (same signature for both modes)
            new_positions, new_directions, new_times, detect_probs, reflection_attenuations, continuing_factors = jax.vmap(
                photon_update_fn,
                in_axes=(0, 0, 0, 0, 0, None, None, None, None, 0)
            )(current_positions, current_directions, current_times,
              surface_distances, normals, scatter_length, reflection_rate,
              absorption_length, tau_gs, rng_keys)

            # Calculate detected intensities
            detected_intensity_factors = detect_probs * reflection_attenuations
            updated_weights = depositions * current_intensities[None, :] * detected_intensity_factors[None, :]
            total_times = times + current_times[:, None]

            # Create outputs for this iteration
            iteration_weights = updated_weights
            iteration_indices = detector_indices
            iteration_times = total_times.squeeze(-1)

            # Update intensities for next iteration
            new_intensities = current_intensities * continuing_factors

            # Apply stop_gradient to prevent backpropagation through iterations
            next_positions = jax.lax.stop_gradient(new_positions)
            next_directions = jax.lax.stop_gradient(new_directions)
            next_intensities = new_intensities
            next_times = jax.lax.stop_gradient(new_times)

            # Return updated state and outputs
            new_carry = (next_positions, next_directions, next_intensities, next_times, key)
            outputs = (iteration_weights, iteration_indices, iteration_times)

            return new_carry, outputs

        # Initial state
        init_carry = (positions, directions, intensities, times, key)

        # Run the propagation loop
        _, (all_iter_weights, all_iter_indices, all_iter_times) = jax.lax.scan(
            propagation_step,
            init_carry,
            jnp.arange(K)
        )

        # Flatten arrays across iterations and photons
        flat_weights = all_iter_weights.reshape(-1)
        flat_indices = all_iter_indices.reshape(-1)
        flat_times = all_iter_times.reshape(-1)

        # Accumulate charges and times at each detector
        total_time_weighted = jax.ops.segment_sum(
            flat_weights * flat_times,
            flat_indices,
            num_segments=num_detectors
        )

        total_charge_weighted = jax.ops.segment_sum(
            flat_weights,
            flat_indices,
            num_segments=num_detectors
        )

        # Compute average times
        eps = 1e-10
        average_times = jnp.where(
            total_charge_weighted > eps,
            total_time_weighted / (total_charge_weighted + eps),
            jnp.zeros_like(total_charge_weighted)
        )

        # Find minimum non-zero time for alignment
        nonzero_mask = total_charge_weighted > 1e-5
        min_nonzero_time = jnp.where(
            jnp.any(nonzero_mask),
            jnp.min(jnp.where(nonzero_mask, average_times, jnp.inf)),
            0.0
        )

        # Align times to earliest detection
        aligned_times = jnp.where(
            nonzero_mask,
            average_times - min_nonzero_time,
            0
        )

        # Zero out charges below threshold
        corrected_q = jnp.where(
            nonzero_mask,
            total_charge_weighted,
            0
        )

        return corrected_q, aligned_times

    # Return appropriate simulation function
    if is_data:
        return _simulation_with_data
    elif is_calibration:
        return _simulation_detector_calibration
    else:
        # Load SIREN model data
        table = Table('../siren/cprof_mu_train_10000ev.h5')
        grid_data = create_siren_grid(table)
        siren_model, model_params = load_siren_jax('../siren/siren_cprof_mu.pkl')

        # Return partially applied function with model data
        return partial(_simulation_without_data,
                       grid_data=grid_data,
                       model_params=model_params)