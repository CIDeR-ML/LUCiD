from tools.generate import differentiable_get_rays, new_differentiable_get_rays
from tools.propagate import create_photon_propagator
from tools.geometry import generate_detector

import jax
import jax.numpy as jnp

from siren.siren import *
from siren.table import *

def setup_event_simulator(json_filename, n_photons=1_000_000, temperature=100):
    """
    Sets up and returns an event simulator with the specified configuration.

    Parameters
    ----------
    json_filename : str
        Path to the JSON file containing detector configuration
    n_photons : int, optional
        Number of photons to simulate per event, default 1,000,000
    temperature : float, optional
        Temperature parameter for photon propagation, default 100

    Returns
    -------
    tuple
        simulate_event: callable that takes (params, key) and returns event data

    Example
    -------
     simulate_event = setup_event_simulator("config.json")
     true_params = (
    ...     jnp.array(90.0),  # opening angle
    ...     jnp.array([0.0, 0.0, 0.0]),  # position
    ...     jnp.array([0.0, 1.0, 0.0]),  # direction
    ...     jnp.array(5000.0)  # intensity
    ... )
     key = jax.random.PRNGKey(42)
     event_data = simulate_event(true_params, key)
    """
    # Initialize detector configuration
    detector = generate_detector(json_filename)
    detector_points = jnp.array(detector.all_points)
    detector_radius = detector.S_radius

    # Setup photon propagator with specified parameters
    propagate_photons = create_photon_propagator(
        detector_points,
        detector_radius,
        temperature=temperature
    )

    # Get number of detectors from points array
    NUM_DETECTORS = len(detector_points)

    # Create the event simulator
    simulate_event = create_event_simulator(
        propagate_photons,
        n_photons,
        NUM_DETECTORS,
        detector_points
    )

    return simulate_event


def create_siren_grid(table):
    ene_bins = table.normalize(0, table.binning[0])
    cos_bins = table.normalize(1, np.linspace(0.3, max(table.binning[1]), 1500))
    trk_bins = table.normalize(2, np.linspace(min(table.binning[2]), max(table.binning[2]), 1500))
    cos_trk_mesh = np.array([[x,y] for x in cos_bins for y in trk_bins])
    x_data = table.binning[0]
    y_data = ene_bins
    grid_shape = (np.shape(cos_bins)[0]*np.shape(trk_bins)[0],3)
    return cos_bins, trk_bins, cos_trk_mesh, (x_data, y_data), grid_shape

def create_event_simulator(propagate_photons, Nphot, NUM_DETECTORS, detector_points):
    """
    Creates a memory-efficient differentiable event simulator with fixed time calculation.

    Parameters
    ----------
    propagate_photons : callable
        Function that handles photon propagation through the medium
    Nphot : int
        Number of photons to simulate per event
    NUM_DETECTORS : int
        Total number of detectors in the system
    detector_points : array_like
        Array of detector positions, shape (NUM_DETECTORS, 3)

    Returns
    -------
    callable
        JIT-compiled function that takes parameters and random key as input
        and returns (charges, average_times) for each detector
    """

    table = Table('siren/cprof_mu_train_10000ev.h5')
    grid_data = create_siren_grid(table)
    grid_shape = grid_data[4]
    siren_model, model_params = load_siren_jax('siren/siren_cprof_mu.pkl', grid_shape)

    @jax.jit
    def _simulate_event_core(params, key):
        # Unpack simulation parameters
        cone_opening, track_origin, track_direction, initial_intensity = params

        # # Generate and propagate photons
        # photon_directions, photon_origins = differentiable_get_rays(
        #     track_origin, track_direction, cone_opening, Nphot, key)
        # prop_results = propagate_photons(photon_origins, photon_directions)

        energy=500
        reduced_Nphot = 50000
        photon_directions, photon_origins, photon_weights = new_differentiable_get_rays(track_origin, track_direction, energy, reduced_Nphot, grid_data, model_params, key)
        prop_results = propagate_photons(photon_origins, photon_directions)

        # Extract results - each array includes data for all possible detector-ray combinations
        weights = prop_results['detector_weights']  # [max_detectors, n_rays]
        detector_indices = prop_results['detector_indices']  # [max_detectors]
        hit_times = prop_results['times']  # [max_detectors, n_rays]
        hit_positions = prop_results['positions']  # [max_detectors, n_rays, 3]

        # # Calculate charge deposits using competitive normalized weights
        # photon_intensity = initial_intensity / Nphot
        # flat_weights = weights.reshape(-1)
        # flat_indices = detector_indices.reshape(-1)
        # charges = jax.ops.segment_sum(
        #     flat_weights * photon_intensity,
        #     flat_indices,
        #     num_segments=NUM_DETECTORS
        # )

        # Expand photon_weights to match weights dimension
        expanded_photon_weights = jnp.broadcast_to(photon_weights, weights.shape)  # [max_detectors, n_rays]

        # Element-wise multiplication of all components
        flat_weights = (weights * expanded_photon_weights).reshape(-1)  # Flatten after multiplication
        flat_indices = detector_indices.reshape(-1)

        # Calculate charges using segment_sum
        charges = jax.ops.segment_sum(
            flat_weights,
            flat_indices,
            num_segments=NUM_DETECTORS
        )

        # Compute average hit times for each detector using weighted averages
        flat_times = hit_times.reshape(-1)  # Flatten times

        # Calculate numerator and denominator for weighted average time
        detector_time_totals = jax.ops.segment_sum(
            flat_weights * flat_times,
            flat_indices,
            num_segments=NUM_DETECTORS
        )

        detector_weight_totals = jax.ops.segment_sum(
            flat_weights,
            flat_indices,
            num_segments=NUM_DETECTORS
        )

        # Compute average times with protection against division by zero
        eps = 1e-10
        average_times = jnp.where(
            detector_weight_totals > eps,
            detector_time_totals / (detector_weight_totals + eps),
            jnp.zeros_like(detector_weight_totals)
        )

        # time alignment - currently not needed but can be used for additional analysis
        # Calculate and subtract global mean time for time alignment
        # weighted_sum = jnp.sum(average_times * detector_weight_totals)
        # total_weights = jnp.sum(detector_weight_totals) + eps
        # mean_time = weighted_sum / total_weights

        # # Return mean-subtracted times
        # average_times = average_times

        # # Position calculation - currently not needed but can be used for additional analysis
        # flat_positions = hit_positions.reshape(-1, 3)  # [max_detectors * n_rays, 3]
        #
        # detector_position_totals = jax.ops.segment_sum(
        #     flat_weights[:, None] * flat_positions,
        #     flat_indices,
        #     num_segments=NUM_DETECTORS
        # )
        #
        # average_positions = jnp.where(
        #     detector_weight_totals[:, None] > eps,
        #     detector_position_totals / (detector_weight_totals[:, None] + eps),
        #     detector_points
        # )

        return charges, average_times#, weights

    return jax.jit(lambda p, k: _simulate_event_core(p, k))