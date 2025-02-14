import jax.numpy as jnp
from jax import jit
import jax

@jit
def compute_simple_loss(
        true_charge: jnp.ndarray,
        true_time: jnp.ndarray,
        simulated_charge: jnp.ndarray,
        simulated_time: jnp.ndarray,
        sigma_time: float = 100.0,
        eps: float = 1e-8,
) -> float:
    """Compute point-wise loss between true and simulated detector measurements.

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


import jax.numpy as jnp


def compute_loss_with_time(
        detector_points: jnp.ndarray,
        true_charge: jnp.ndarray,
        true_time: jnp.ndarray,
        simulated_charge: jnp.ndarray,
        simulated_time: jnp.ndarray,
        tau_position: float = 0.08,
        tau_time: float = 0.08,
        lambda_time: float = 1.0,
        eps: float = 1e-8,
        threshold: float = 1e-8,
) -> float:
    """
    Compute loss using a Euclidean combination of spatial and temporal distances.

    The joint squared distance is:
        d_total^2 = (d_space)^2 + lambda_time * (d_time)^2,
    with:
        d_space = ||x_i - x_j|| / tau_position,
        d_time  = |t_i' - T_j'| / tau_time,  (after subtracting the mean over active detectors)
    and similarity:
        S = 1 / (1 + d_total^2).

    The overall loss is the sum of:
      - intensity loss: |log(total_sim_charge / total_true_charge)|
      - distribution loss: sum(normalized similarity * |simulated_charge - true_charge|).

    Only pairs where both true and simulated charges are above threshold contribute.

    Parameters
    ----------
    detector_points : jnp.ndarray
        Array of shape (N_detectors, 3) with detector coordinates.
    true_charge : jnp.ndarray
        Array of shape (N_detectors,) with true charges.
    true_time : jnp.ndarray
        Array of shape (N_detectors,) with true times.
    simulated_charge : jnp.ndarray
        Array of shape (N_detectors,) with simulated charges.
    simulated_time : jnp.ndarray
        Array of shape (N_detectors,) with simulated times.
    tau_position : float, optional
        Scale factor for spatial distances, by default 0.05.
    tau_time : float, optional
        Scale factor for temporal distances, by default 0.05.
    lambda_time : float, optional
        Weight for mixing temporal and spatial distances, by default 1.0.
    eps : float, optional
        Small constant to prevent division by zero, by default 1e-8.
    threshold : float, optional
        Threshold for considering a detector active, by default 1e-8.

    Returns
    -------
    float
        Loss value.
    """
    # --- Spatial distances ---
    spatial_dist = jnp.linalg.norm(
        detector_points[:, jnp.newaxis, :] - detector_points[jnp.newaxis, :, :],
        axis=2
    ) / tau_position
    spatial_dist_sq = spatial_dist ** 2

    # --- Active masks ---
    true_active = true_charge > threshold
    sim_active = simulated_charge > threshold

    # --- Time normalization (only for active detectors) ---
    mean_true_time = jnp.sum(jnp.where(true_active, true_time, 0.0)) / (jnp.sum(true_active) + eps)
    mean_sim_time = jnp.sum(jnp.where(sim_active, simulated_time, 0.0)) / (jnp.sum(sim_active) + eps)
    true_time_normalized = jnp.where(true_active, true_time - mean_true_time, 0.0)
    sim_time_normalized = jnp.where(sim_active, simulated_time - mean_sim_time, 0.0)

    # --- Temporal distances ---
    temporal_dist = jnp.abs(
        true_time_normalized[:, jnp.newaxis] - sim_time_normalized[jnp.newaxis, :]
    ) / tau_time
    temporal_dist_sq = temporal_dist ** 2

    # --- Combined distance and similarity ---
    combined_dist_sq = spatial_dist_sq + lambda_time * temporal_dist_sq
    similarity_matrix = 1 / (1 + combined_dist_sq)

    # --- Mask inactive pairs ---
    active_pairs = true_active[:, jnp.newaxis] & sim_active[jnp.newaxis, :]
    similarity_matrix = jnp.where(active_pairs, similarity_matrix, 0.0)

    # --- Normalize similarity ---
    sums_per_row = jnp.sum(similarity_matrix, axis=1)
    sums_per_col = jnp.sum(similarity_matrix, axis=0)
    normalized_similarity = similarity_matrix / (jnp.sqrt((sums_per_row[:, jnp.newaxis] *
                                                           sums_per_col[jnp.newaxis, :]) + eps))

    # # --- Intensity loss --- include for more sharp total charge difference
    # total_true_charge = jnp.sum(true_charge)
    # total_sim_charge = jnp.sum(simulated_charge)
    # intensity_loss = jnp.abs(jnp.log(total_sim_charge / (total_true_charge + eps)))

    # --- Distribution loss ---
    delta_charge = jnp.abs(simulated_charge[jnp.newaxis, :] - true_charge[:, jnp.newaxis])
    distribution_loss = jnp.sum(normalized_similarity * delta_charge)

    return distribution_loss


@jit
def compute_softmin_loss(
        detector_points: jnp.ndarray,
        true_charge: jnp.ndarray,
        true_time: jnp.ndarray,
        simulated_charge: jnp.ndarray,
        simulated_time: jnp.ndarray,
        tau: float = 0.01,
        eps: float = 1e-8,
        lambda_time: float = 1.0
) -> float:
    """
    Compute a differentiable loss using soft assignments between simulated and true detectors.
    Times are mean-subtracted considering only active (non-zero charge) locations.
    This version uses absolute distance scales where the distance between detector differences is controlled by tau

    Parameters
    ----------
    detector_points : jnp.ndarray
        Array of shape (N, 3) with detector coordinates.
    true_charge : jnp.ndarray
        Array of shape (N,) of true charges.
    true_time : jnp.ndarray
        Array of shape (N,) of true times.
    simulated_charge : jnp.ndarray
        Array of shape (N,) of simulated charges.
    simulated_time : jnp.ndarray
        Array of shape (N,) of simulated times.
    tau : float, optional
        Temperature parameter for the softmin. Smaller tau => sharper assignments.
    eps : float, optional
        Small constant to prevent division by zero, by default 1e-8
    lambda_time : float, optional
        Scaling factor for time loss, by default 1.0.

    Returns
    -------
    float
        Total loss.
    """
    # Compute mean times for active locations
    true_active_mask = true_charge > eps
    sim_active_mask = simulated_charge > eps

    # Compute mean times only for active locations
    true_mean_time = jnp.sum(true_time * true_active_mask) / (
                jnp.sum(true_active_mask) + eps)
    sim_mean_time = jnp.sum(simulated_time * sim_active_mask) / (
                jnp.sum(sim_active_mask) + eps)

    # Subtract means from times
    true_time_centered = jnp.where(true_active_mask, true_time - true_mean_time, 0.0)
    sim_time_centered = jnp.where(sim_active_mask, simulated_time - sim_mean_time, 0.0)

    total_true_charge = jnp.sum(true_charge)
    total_sim_charge = jnp.sum(simulated_charge)
    intensity_loss = jnp.abs(jnp.log(total_sim_charge / (total_true_charge + eps)))

    # Compute distance matrix d[i,j] = ||x_i - x_j||
    N = detector_points.shape[0]
    dist = jnp.linalg.norm(
        detector_points[:, None, :] - detector_points[None, :, :],
        axis=-1
    )  # Shape (N, N)

    # Soft assignments Sim -> True
    logits_s2t = -dist / tau
    w_s2t = jax.nn.softmax(logits_s2t, axis=1)   # shape (N, N)

    # Aggregated charges (Sim->True)
    Q_sim_per_true = w_s2t.T @ simulated_charge  # shape (N,)
    # Charge-weighted centered times:
    qt_sim_per_true = w_s2t.T @ (simulated_charge * sim_time_centered)  # shape (N,)
    avg_sim_time_per_true = qt_sim_per_true / (Q_sim_per_true + eps)

    # Loss terms S->T
    L_charge_s2t = jnp.sum(jnp.abs(Q_sim_per_true - true_charge))
    L_time_s2t = jnp.sum(jnp.abs(avg_sim_time_per_true - true_time_centered) * Q_sim_per_true)

    # Soft assignments True -> Sim
    logits_t2s = -dist.T / tau
    w_t2s = jax.nn.softmax(logits_t2s, axis=1) # shape (N, N)

    # Aggregated charges (True->Sim)
    Q_true_per_sim = w_t2s.T @ true_charge  # shape (N,)
    qt_true_per_sim = w_t2s.T @ (true_charge * true_time_centered)
    avg_true_time_per_sim = qt_true_per_sim / (Q_true_per_sim + eps)

    # Loss terms T->S
    L_charge_t2s = jnp.sum(jnp.abs(Q_true_per_sim - simulated_charge))
    L_time_t2s = jnp.sum(jnp.abs(avg_true_time_per_sim - sim_time_centered) * Q_true_per_sim)

    # Combine losses
    L_charge = L_charge_s2t + L_charge_t2s
    L_time = (L_time_s2t + L_time_t2s) * lambda_time

    return L_charge + L_time + intensity_loss