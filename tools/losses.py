import jax.numpy as jnp
from jax import jit

@jit
def compute_loss(
        true_charge: jnp.ndarray,
        true_time: jnp.ndarray,
        simulated_charge: jnp.ndarray,
        simulated_time: jnp.ndarray,
        sigma_time: float = 100.0,
        eps: float = 1e-8,
) -> float:
    """Compute loss between true and simulated detector measurements.

    Calculates loss between true and simulated detectors comparing PMT charge and time measurements. (same PMTs)

    Parameters
    ----------
    true_charge : ndarray
        Array of shape (N_detectors,) containing true charge measurements
    true_time : ndarray
        Array of shape (N_detectors,) containing true timing measurements. Not used in this loss
    simulated_charge : ndarray
        Array of shape (N_detectors,) containing simulated charge predictions
    simulated_time : ndarray
        Array of shape (N_detectors,) containing simulated timing predictions. Not used in this loss
    sigma_time : float, optional
        Scale factor for temporal differences, by default 0.1
    eps : float, optional
        Small constant to prevent division by zero, by default 1e-8

    Returns
    -------
    float
        Combined loss value from charge distribution and total intensity
    """
    # Loss component based on charge spatial distribution
    delta_charge = jnp.abs(simulated_charge - true_charge)
    charge_loss = jnp.sum(delta_charge)

    simulated_time = simulated_time - jnp.min(simulated_time) / (jnp.std(simulated_time) + eps)
    true_time = true_time - jnp.min(true_time) / (jnp.std(true_time) + eps)

    # Loss component based on time distribution
    delta_time = jnp.abs(simulated_time - true_time) / sigma_time
    time_loss = jnp.sum(delta_time)

    intensity_loss = jnp.abs(jnp.log(jnp.sum(simulated_charge) / (jnp.sum(true_charge) + eps)))

    return charge_loss + time_loss + intensity_loss



@jit
def compute_loss_gaussian(
    detector_points: jnp.ndarray,
    true_charge: jnp.ndarray,
    true_time: jnp.ndarray,
    simulated_charge: jnp.ndarray,
    simulated_time: jnp.ndarray,
    sigma_position: float = 0.1,
    eps: float = 1e-8,
) -> float:
    """Compute loss between true and simulated detector measurements.

    Calculates loss based on spatial distribution and total charge, using
    an inverse quadratic distance metric for spatial similarity.

    Parameters
    ----------
    detector_points : ndarray
        Array of shape (N_detectors, 3) containing detector point coordinates
    true_charge : ndarray
        Array of shape (N_detectors,) containing true charge measurements
    true_time : ndarray
        Array of shape (N_detectors,) containing true timing measurements. Not used in this loss
    simulated_charge : ndarray
        Array of shape (N_detectors,) containing simulated charge predictions
    simulated_time : ndarray
        Array of shape (N_detectors,) containing simulated timing predictions. Not used in this loss
    sigma_position : float, optional
        Scale factor for spatial distances, by default 0.1
        This is used to control how far apart detector points are weighted in the loss
    eps : float, optional
        Small constant to prevent division by zero, by default 1e-8

    Returns
    -------
    float
        Combined loss value from charge distribution and total intensity
    """
    # Calculate pairwise distances between detector points, normalized by sigma
    spatial_dist = jnp.linalg.norm(
        detector_points[:, jnp.newaxis, :] - detector_points[jnp.newaxis, :, :],
        axis=2
    ) / sigma_position

    # Convert distances to similarities using inverse quadratic function
    similarity_matrix = 1/(1+spatial_dist**2)
    sums = jnp.sum(similarity_matrix, axis=1, keepdims=True)

    # Normalize similarities to sum to 1
    normalized_similarity = similarity_matrix / (sums + eps)

    # Loss component based on total charge conservation
    total_true_charge = jnp.sum(true_charge)
    total_sim_charge = jnp.sum(simulated_charge)
    intensity_loss = jnp.abs(jnp.log(total_sim_charge / (total_true_charge + eps)))

    # Loss component based on charge spatial distribution
    delta_charge = jnp.abs(simulated_charge[jnp.newaxis, :] - true_charge[:, jnp.newaxis])
    distribution_loss = jnp.sum(normalized_similarity * delta_charge)

    return distribution_loss + intensity_loss


# This loss is not used as the times and final positions are embedded in the simulated_charge calculation.
# Something like this can be used when stochasticity is added to the simulation.
@jit
def compute_loss_with_time(
        detector_points: jnp.ndarray,
        true_charge: jnp.ndarray,
        true_time: jnp.ndarray,
        simulated_charge: jnp.ndarray,
        simulated_time: jnp.ndarray,
        simulated_positions: jnp.ndarray,
        sigma_position: float = 0.1,
        sigma_time: float = 1.0,
        eps: float = 1e-8,
) -> float:
    """Compute loss between true and simulated detector measurements with timing.

    Extended version that includes temporal information and uses simulated positions
    instead of detector points for spatial comparisons.

    Parameters
    ----------
    detector_points : ndarray
        Array of shape (N_detectors, 3) containing detector point coordinates
    true_charge : ndarray
        Array of shape (N_detectors,) containing true charge measurements
    true_time : ndarray
        Array of shape (N_detectors,) containing true timing measurements
    simulated_charge : ndarray
        Array of shape (N_detectors,) containing simulated charge predictions
    simulated_time : ndarray
        Array of shape (N_detectors,) containing simulated timing predictions
    simulated_positions : ndarray
        Array of shape (N_detectors, 3)
        Contains simulated final positions of each photon aggregated over each detector
    sigma_position : float, optional
        Scale factor for spatial distances, by default 0.1
        This is used to control how far apart detector points are weighted in the loss
    sigma_time : float, optional
        Scale factor for temporal differences, by default 1.0
        This is used to control how far apart timing values are weighted in the loss
    eps : float, optional
        Small constant to prevent division by zero, by default 1e-8

    Returns
    -------
    float
        Combined loss value from charge distribution, total intensity, and timing
    """
    # Calculate pairwise distances between detector and simulated points
    spatial_dist = jnp.linalg.norm(
        detector_points[:, jnp.newaxis, :] - simulated_positions[jnp.newaxis, :, :],
        axis=2
    ) / sigma_position

    # Convert distances to similarities using inverse quadratic function
    similarity_matrix = 1 / (1 + spatial_dist ** 2)
    sums = jnp.sum(similarity_matrix, axis=1, keepdims=True)

    # Normalize similarities to sum to 1
    normalized_similarity = similarity_matrix / (sums + eps)

    # Loss component based on total charge conservation
    total_true_charge = jnp.sum(true_charge)
    total_sim_charge = jnp.sum(simulated_charge)
    intensity_loss = jnp.abs(jnp.log(total_sim_charge / (total_true_charge + eps)))

    # Loss component based on charge spatial distribution
    delta_charge = jnp.abs(simulated_charge[jnp.newaxis, :] - true_charge[:, jnp.newaxis])
    distribution_loss = jnp.sum(normalized_similarity * delta_charge)

    # Normalize timing values to account for different time scales
    true_time_std = jnp.std(true_time) + eps
    sim_time_std = jnp.std(simulated_time) + eps

    true_time_normalized = (true_time - jnp.mean(true_time)) / true_time_std
    sim_time_normalized = (simulated_time - jnp.mean(simulated_time)) / sim_time_std

    # Calculate normalized temporal differences
    delta_time = jnp.abs(
        true_time_normalized[:, jnp.newaxis] - sim_time_normalized[jnp.newaxis, :]
    ) / sigma_time

    # Combine spatial similarity with charge information for temporal weighting
    weights = normalized_similarity * jnp.sqrt(
        (true_charge[:, jnp.newaxis] * simulated_charge[jnp.newaxis, :]) + eps
    )
    temporal_loss = jnp.sum(weights * delta_time) / (jnp.sum(weights) + eps)

    return distribution_loss + intensity_loss + temporal_loss