import jax.numpy as jnp
from jax import jit
import jax
from jax.scipy.special import gammaln
from optax.losses import huber_loss


@jit
def compute_simple_loss(
        true_charge: jnp.ndarray,
        true_time: jnp.ndarray,
        simulated_charge: jnp.ndarray,
        simulated_time: jnp.ndarray,
        sigma_time: float = 100.0,
        eps: float = 1e-8,
) -> float:
    """Compute point-wise loss between true and simulated sensor measurements.

    Calculates loss between true and simulated sensors comparing PMT charge and time measurements. (same PMTs)

    Parameters
    ----------
    true_charge : ndarray
        Array of shape (N_sensors,) containing true charge measurements
    true_time : ndarray
        Array of shape (N_sensors,) containing true timing measurements. Not used in this loss
    simulated_charge : ndarray
        Array of shape (N_sensors,) containing simulated charge predictions
    simulated_time : ndarray
        Array of shape (N_sensors,) containing simulated timing predictions. Not used in this loss
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
        sensor_points: jnp.ndarray,
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
        d_time  = |t_i' - T_j'| / tau_time,  (after subtracting the mean over active sensors)
    and similarity:
        S = 1 / (1 + d_total^2).

    The overall loss is the sum of:
      - intensity loss: |log(total_sim_charge / total_true_charge)|
      - distribution loss: sum(normalized similarity * |simulated_charge - true_charge|).

    Only pairs where both true and simulated charges are above threshold contribute.

    Parameters
    ----------
    sensor_points : jnp.ndarray
        Array of shape (N_sensors, 3) with sensor coordinates.
    true_charge : jnp.ndarray
        Array of shape (N_sensors,) with true charges.
    true_time : jnp.ndarray
        Array of shape (N_sensors,) with true times.
    simulated_charge : jnp.ndarray
        Array of shape (N_sensors,) with simulated charges.
    simulated_time : jnp.ndarray
        Array of shape (N_sensors,) with simulated times.
    tau_position : float, optional
        Scale factor for spatial distances, by default 0.05.
    tau_time : float, optional
        Scale factor for temporal distances, by default 0.05.
    lambda_time : float, optional
        Weight for mixing temporal and spatial distances, by default 1.0.
    eps : float, optional
        Small constant to prevent division by zero, by default 1e-8.
    threshold : float, optional
        Threshold for considering a sensor active, by default 1e-8.

    Returns
    -------
    float
        Loss value.
    """
    # --- Spatial distances ---
    spatial_dist = jnp.linalg.norm(
        sensor_points[:, jnp.newaxis, :] - sensor_points[jnp.newaxis, :, :],
        axis=2
    ) / tau_position
    spatial_dist_sq = spatial_dist ** 2

    # --- Active masks ---
    true_active = true_charge > threshold
    sim_active = simulated_charge > threshold

    # --- Time normalization (only for active sensors) ---
    mean_true_time = jax.lax.stop_gradient(jnp.sum(jnp.where(true_active, true_time, 0.0)) / (jnp.sum(true_active) + eps))
    mean_sim_time = jax.lax.stop_gradient(jnp.sum(jnp.where(sim_active, simulated_time, 0.0)) / (jnp.sum(sim_active) + eps))
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
    total_true_charge = jnp.sum(true_charge)
    total_sim_charge = jnp.sum(simulated_charge)
    intensity_loss = jnp.abs(jnp.log(total_sim_charge / (total_true_charge + eps)))

    # --- Distribution loss ---
    delta_charge = jnp.abs(simulated_charge[jnp.newaxis, :] - true_charge[:, jnp.newaxis])
    distribution_loss = jnp.sum(normalized_similarity * delta_charge)

    return jnp.log(distribution_loss) + intensity_loss # let's do log such that both losses have comparable size

@jit
def compute_softmin_loss(
        sensor_points: jnp.ndarray,
        true_charge: jnp.ndarray,
        true_time: jnp.ndarray,
        simulated_charge: jnp.ndarray,
        simulated_time: jnp.ndarray,
        tau: float = 0.01,
        eps: float = 1e-8,
        threshold_mean: float = 1e-8,  # Low threshold for mean computation
        threshold_loss: float = 40.00001,  # High threshold for loss masking
        lambda_time: float = 50.0,
        lambda_intensity: float = 0.1,
        lambda_charge: float = 1.0,
) -> float:
    """
    Compute a differentiable loss using soft assignments between simulated and true sensors.
    Times are mean-subtracted considering only active (non-zero charge) locations.
    This version uses absolute distance scales where the distance between sensor differences is controlled by tau

    Parameters
    ----------
    sensor_points : jnp.ndarray
        Array of shape (N, 3) with sensor coordinates.
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
    threshold_mean : float, optional
        Threshold for considering a sensor active when computing means for time centering, by default 1e-8.
    threshold_loss : float, optional
        Threshold for considering a sensor active when computing what terms contribute to the time loss, by default 40.00001.
    lambda_time : float, optional
        Scaling factor for time loss, by default 1.0.
    lambda_intensity : float, optional
        Scaling factor for intensity loss, by default 1.0.

    Returns
    -------
    float
        Total loss.
    """
    # Mean centering with LOW threshold (include all active sensors)
    true_active_mean = true_charge > threshold_mean
    sim_active_mean = simulated_charge > threshold_mean

    true_mean_time = jax.lax.stop_gradient(jnp.sum(true_time * true_active_mean) / (
            jnp.sum(true_active_mean) + eps))
    sim_mean_time = jax.lax.stop_gradient(jnp.sum(simulated_time * sim_active_mean) / (
            jnp.sum(sim_active_mean) + eps))

    # Subtract means from times
    true_time_centered = jnp.where(true_active_mean, true_time - true_mean_time, true_time)
    sim_time_centered = jnp.where(sim_active_mean, simulated_time - sim_mean_time, simulated_time)

    # Loss masking with HIGH threshold (focus on significant sensors)
    true_active_loss = true_charge > threshold_loss
    sim_active_loss = simulated_charge > threshold_loss

    # ============= Energy/Intensity Loss =============
    total_true_charge = jnp.sum(true_charge)
    total_sim_charge = jnp.sum(simulated_charge)
    intensity_loss = jnp.abs(jnp.log(total_sim_charge / (total_true_charge + eps)))

    # For Scaling the loss terms
    charge_norm = (jnp.sum(true_charge) + jnp.sum(simulated_charge)) / 2.0 + eps

    total_true_time_scale = jax.lax.stop_gradient(
        jnp.sum(jnp.abs(true_time_centered) * true_active_loss)  + eps
    )

    total_sim_time_scale = jax.lax.stop_gradient(
        jnp.sum(jnp.abs(sim_time_centered) * sim_active_loss) + eps
    )

    time_norm = (total_true_time_scale + total_sim_time_scale) / 2.0 + eps

    # ============= Spatial Loss Components =============
    # Compute distance matrix
    N = sensor_points.shape[0]
    dist = jnp.linalg.norm(
        sensor_points[:, None, :] - sensor_points[None, :, :],
        axis=-1
    )

    logits = -dist / tau
    dist_weights = jax.nn.softmax(logits, axis=1)

    # ------------- Sim -> True Assignment -------------
    # Aggregated charges
    Q_sim_per_true = dist_weights.T @ simulated_charge

    # Charge-weighted time aggregation for better gradient flow
    T_sim_per_true = dist_weights.T @ sim_time_centered

    # Loss terms S->T
    L_charge_s2t = jnp.sum(jnp.abs(Q_sim_per_true - true_charge)) / charge_norm
    # Use charge-weighted difference
    L_time_s2t = jnp.sum(jnp.abs(T_sim_per_true - true_time_centered) * true_active_loss) / time_norm

    # ------------- True -> Sim Assignment -------------
    # Aggregated charges
    Q_true_per_sim = dist_weights.T @ true_charge

    # Charge-weighted time aggregation
    T_true_per_sim = dist_weights.T @ true_time_centered

    # Loss terms T->S
    L_charge_t2s = jnp.sum(jnp.abs(Q_true_per_sim - simulated_charge)) / charge_norm
    # Use charge-weighted difference
    L_time_t2s = jnp.sum(jnp.abs(T_true_per_sim - sim_time_centered) * sim_active_loss)/ time_norm

    # ============= Combine Losses =============
    L_charge = (L_charge_s2t + L_charge_t2s) * lambda_charge
    L_time = (L_time_s2t + L_time_t2s) * lambda_time
    L_intensity = intensity_loss * lambda_intensity

    return L_charge + L_time + L_intensity


def poisson_nll(true: jnp.ndarray, pred: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Compute the Poisson negative log-likelihood loss.

    Parameters
    ----------
    true : jnp.ndarray
        Ground truth values (non-negative).
    pred : jnp.ndarray
        Predicted values (non-negative).
    eps : float, optional
        Small constant to prevent log(0), by default 1e-8.

    Returns
    -------
    jnp.ndarray
        Poisson negative log-likelihood loss.
    """
    nll = pred - true * jnp.log(pred + eps) + gammaln(true + 1.0)
    normalized_nll = jnp.sum(nll) / (jnp.sum(true) + eps)
    return normalized_nll

def WC_loss(
        sensor_points: jnp.ndarray,
        true_charge: jnp.ndarray,
        true_time: jnp.ndarray,
        simulated_charge: jnp.ndarray,
        simulated_time: jnp.ndarray,
        eps: float = 1e-8,
        threshold: float = 1e-8,
        lambda_poisson: float = 1.0,
        lambda_time: float = 1.0,
) -> float:
    """
    Optimized Loss
    Compute a loss function with two components:
    1. Poisson loss: Poisson negative log-likelihood for charge distributions
    2. Time loss: Huber loss on time differences

    Parameters
    ----------
    sensor_points : jnp.ndarray
        Array of shape (N, 3) with sensor coordinates.
    true_charge : jnp.ndarray
        Array of shape (N,) of true charges.
    true_time : jnp.ndarray
        Array of shape (N,) of true times.
    simulated_charge : jnp.ndarray
        Array of shape (N,) of simulated charges.
    simulated_time : jnp.ndarray
        Array of shape (N,) of simulated times.
    eps : float, optional
        Small constant to prevent division by zero, by default 1e-8
    threshold : float, optional
        Threshold for considering a sensor active, by default 1e-8
    lambda_poisson : float, optional
        Scaling factor for Poisson loss, by default 1.0.
    lambda_time : float, optional
        Scaling factor for time loss, by default 1.0.

    Returns
    -------
    float
        Total loss.
    """
    # ============= Poisson Negative Log Likelihood =============
    poisson_loss = poisson_nll(true_charge, simulated_charge, eps)

    # ============= Time Loss with Charge Weighted Mean Subtraction =============
    # Mean centering with LOW threshold (include all active sensors)
    true_active = true_charge > threshold
    sim_active = simulated_charge > threshold

    true_weights = jnp.where(true_active, true_charge, 0.0)
    sim_weights = jnp.where(sim_active, simulated_charge, 0.0)

    true_t0 = jax.lax.stop_gradient(jnp.sum(true_time * true_weights) / (jnp.sum(true_weights) + 1e-8))
    sim_t0 = jnp.sum(simulated_time * sim_weights) / (jnp.sum(sim_weights) + 1e-8)

    # Compute aligned times and loss
    both_active = jax.lax.stop_gradient(true_active & sim_active)
    diff = jnp.where(both_active,
                     jnp.abs((simulated_time - sim_t0) - (true_time - true_t0)),
                     0.0)

    time_norm = jnp.where(both_active, jnp.abs(true_time - true_t0), 0.0)

    time_loss = jnp.sum(diff) / (jnp.sum(time_norm) + eps)

    L_charge = poisson_loss * lambda_poisson
    L_time = time_loss * lambda_time

    return L_charge + L_time


@jit
def compute_simplified_loss(
        sensor_points: jnp.ndarray,
        true_charge: jnp.ndarray,
        true_time: jnp.ndarray,
        simulated_charge: jnp.ndarray,
        simulated_time: jnp.ndarray,
        eps: float = 1e-8,
        threshold: float = 1e-8,
        lambda_centroid: float = 1.0,
        lambda_time: float = 1.0,
        lambda_intensity: float = 0.5
) -> float:
    """
    Compute a simplified loss function with three components:
    1. Centroid loss: Distance between charge-weighted centroids
    2. Intensity loss: Difference in total charge
    3. Time loss: Simple time spread (standard deviation) comparison

    Parameters
    ----------
    sensor_points : jnp.ndarray
        Array of shape (N, 3) with sensor coordinates.
    true_charge : jnp.ndarray
        Array of shape (N,) of true charges.
    true_time : jnp.ndarray
        Array of shape (N,) of true times.
    simulated_charge : jnp.ndarray
        Array of shape (N,) of simulated charges.
    simulated_time : jnp.ndarray
        Array of shape (N,) of simulated times.
    eps : float, optional
        Small constant to prevent division by zero, by default 1e-8
    threshold : float, optional
        Threshold for considering a sensor active, by default 1e-8
    lambda_centroid : float, optional
        Scaling factor for centroid loss, by default 1.0.
    lambda_time : float, optional
        Scaling factor for time loss, by default 1.0.
    lambda_intensity : float, optional
        Scaling factor for intensity loss, by default 1.0.

    Returns
    -------
    float
        Total loss.
    """
    # Compute active masks for non-zero charges
    true_active_mask = true_charge > threshold
    sim_active_mask = simulated_charge > threshold

    # Calculate total charges
    total_true_charge = jnp.sum(true_charge)
    total_sim_charge = jnp.sum(simulated_charge)

    # --------------------------------------
    # 1. Intensity Loss
    # --------------------------------------
    # Log ratio of total charges (same as original)
    L_intensity = jnp.abs(jnp.log(total_sim_charge / (total_true_charge + eps))) * lambda_intensity

    # --------------------------------------
    # 2. Centroid Loss
    # --------------------------------------
    # Calculate charge-weighted centroids
    true_centroid = jnp.sum(true_charge[:, None] * sensor_points, axis=0) / (total_true_charge + eps)
    sim_centroid = jnp.sum(simulated_charge[:, None] * sensor_points, axis=0) / (total_sim_charge + eps)

    # Euclidean distance between centroids
    L_centroid = jnp.linalg.norm(true_centroid - sim_centroid) * lambda_centroid

    # --------------------------------------
    # 3. Simple Time Spread Loss
    # --------------------------------------
    # Calculate mean times only for active locations
    true_mean_time = jax.lax.stop_gradient(jnp.sum(true_time * true_charge) / (total_true_charge + eps))
    sim_mean_time = jax.lax.stop_gradient(jnp.sum(simulated_time * simulated_charge) / (total_sim_charge + eps))

    # Subtract means from times to account for arbitrary time offsets
    true_time_centered = true_time - true_mean_time
    sim_time_centered = simulated_time - sim_mean_time

    # Simple time spread comparison (standard deviation)
    true_time_var = jnp.sum(true_charge * jnp.square(true_time_centered)) / (total_true_charge + eps)
    sim_time_var = jnp.sum(simulated_charge * jnp.square(sim_time_centered)) / (total_sim_charge + eps)
    L_time_spread = jnp.abs(jnp.sqrt(true_time_var) - jnp.sqrt(sim_time_var))

    # Time loss is just the spread comparison
    L_time = L_time_spread * lambda_time

    # Total loss
    return L_centroid + L_intensity + L_time