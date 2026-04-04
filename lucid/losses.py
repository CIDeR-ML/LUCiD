import jax.numpy as jnp
from jax import jit
import jax
from jax.scipy.special import gammaln
from jax.scipy import special
from functools import partial
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


def hellinger_loss(true_charge_smooth, simulated_charge_smooth, eps=1e-8):
    sqrt_true = jnp.sqrt(true_charge_smooth + eps)
    sqrt_sim = jnp.sqrt(simulated_charge_smooth + eps)
    return jnp.sum((sqrt_true - sqrt_sim) ** 2)


@jit
def WC_smooth_loss(
        sensor_points: jnp.ndarray,
        true_charge: jnp.ndarray,
        true_time: jnp.ndarray,
        simulated_charge: jnp.ndarray,
        simulated_time: jnp.ndarray,
        tau: float = 0.05,
        eps: float = 1e-8,
        threshold: float = 1e-8,
        lambda_poisson: float = 1.0,
        lambda_time: float = 1.0,
) -> float:
    """
    Smoothed version of WC_loss using Gaussian distance weights.

    Applies spatial Gaussian smoothing to both true and simulated charges
    before computing Poisson loss, which helps with gradient flow.

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
        Gaussian width parameter for spatial smoothing, by default 0.05
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
    # ============= Compute Distance-based Gaussian Weights =============
    dist = jnp.linalg.norm(
        sensor_points[:, None, :] - sensor_points[None, :, :],
        axis=-1
    )
    dist_weights = jnp.exp(-dist**2 / (2 * tau**2))
    col_sums = jnp.sum(dist_weights, axis=0, keepdims=True)
    dist_weights = dist_weights / (col_sums + eps)

    # ============= Apply Smoothing to Charges =============
    true_charge_smooth = dist_weights @ true_charge
    simulated_charge_smooth = dist_weights @ simulated_charge

    # ============= Poisson Loss on Smoothed Charges =============
    # poisson_loss = hellinger_loss(true_charge_smooth, simulated_charge_smooth, eps)
    poisson_loss = poisson_nll(true_charge_smooth, simulated_charge_smooth, eps)

    # ============= Time Loss (same as original WC_loss) =============
    true_active = true_charge > threshold
    sim_active = simulated_charge > threshold

    true_weights = jnp.where(true_active, true_charge, 0.0)
    sim_weights = jnp.where(sim_active, simulated_charge, 0.0)

    true_t0 = jax.lax.stop_gradient(jnp.sum(true_time * true_weights) / (jnp.sum(true_weights) + eps))
    sim_t0 = jnp.sum(simulated_time * sim_weights) / (jnp.sum(sim_weights) + eps)

    both_active = jax.lax.stop_gradient(true_active & sim_active)
    diff = jnp.where(both_active,
                     jnp.abs((simulated_time - sim_t0) - (true_time - true_t0)),
                     0.0)

    time_norm = jnp.where(both_active, jnp.abs(true_time - true_t0), 0.0)

    time_loss = jnp.sum(diff) / (jnp.sum(time_norm) + eps)

    # ============= Combine Losses =============
    L_charge = poisson_loss * lambda_poisson
    L_time = time_loss * lambda_time

    return L_charge + L_time


@jit
def WC_smooth_loss_qe_decoupled(
        sensor_points: jnp.ndarray,
        true_charge: jnp.ndarray,
        true_time: jnp.ndarray,
        simulated_charge: jnp.ndarray,
        simulated_time: jnp.ndarray,
        qe_corrections: jnp.ndarray,
        tau: float = 0.05,
        eps: float = 1e-8,
        threshold: float = 1e-8,
        lambda_poisson: float = 1.0,
        lambda_time: float = 1.0,
) -> float:
    """
    Smoothed loss with QE-decoupled gradients.

    Divides out per-sensor QE corrections before smoothing,
    then multiplies back after. This prevents QE corrections
    from mixing across sensors during the spatial convolution,
    resulting in approximately decoupled gradients for each
    sensor's QE correction.

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
    qe_corrections : jnp.ndarray
        Array of shape (N,) of per-sensor QE correction factors.
    tau : float, optional
        Gaussian width parameter for spatial smoothing, by default 0.05
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
    # ============= Compute Distance-based Gaussian Weights =============
    dist = jnp.linalg.norm(
        sensor_points[:, None, :] - sensor_points[None, :, :],
        axis=-1
    )
    dist_weights = jnp.exp(-dist**2 / (2 * tau**2))
    col_sums = jnp.sum(dist_weights, axis=0, keepdims=True)
    dist_weights = dist_weights / (col_sums + eps)

    # ============= Divide both by QE corrections =============
    raw_sim = simulated_charge / (qe_corrections + eps)
    raw_true = true_charge / (qe_corrections + eps)

    # ============= Smooth both =============
    raw_sim_smooth = dist_weights @ raw_sim
    raw_true_smooth = dist_weights @ raw_true

    # ============= Poisson Loss =============
    poisson_loss = poisson_nll(raw_true_smooth, raw_sim_smooth, eps)

    # ============= Time Loss (same as WC_smooth_loss) =============
    true_active = true_charge > threshold
    sim_active = simulated_charge > threshold

    true_weights = jnp.where(true_active, true_charge, 0.0)
    sim_weights = jnp.where(sim_active, simulated_charge, 0.0)

    true_t0 = jax.lax.stop_gradient(jnp.sum(true_time * true_weights) / (jnp.sum(true_weights) + eps))
    sim_t0 = jnp.sum(simulated_time * sim_weights) / (jnp.sum(sim_weights) + eps)

    both_active = jax.lax.stop_gradient(true_active & sim_active)
    diff = jnp.where(both_active,
                     jnp.abs((simulated_time - sim_t0) - (true_time - true_t0)),
                     0.0)
    time_norm = jnp.where(both_active, jnp.abs(true_time - true_t0), 0.0)
    time_loss = jnp.sum(diff) / (jnp.sum(time_norm) + eps)

    # ============= Combine Losses =============
    L_charge = poisson_loss * lambda_poisson
    L_time = time_loss * lambda_time

    return L_charge + L_time


@jit
def WC_smooth_loss_hybrid(
        sensor_points: jnp.ndarray,
        true_charge: jnp.ndarray,
        true_time: jnp.ndarray,
        simulated_charge: jnp.ndarray,
        simulated_time: jnp.ndarray,
        qe_corrections: jnp.ndarray,
        tau: float = 0.05,
        eps: float = 1e-8,
        threshold: float = 1e-8,
        lambda_poisson: float = 1.0,
        lambda_time: float = 1.0,
        lambda_mse: float = 1.0,
) -> float:
    """
    Hybrid loss with cleanly separated gradients:
    - Poisson loss: gradients flow ONLY to physics parameters
    - MSE loss: gradients flow ONLY to QE corrections

    The separation is achieved using stop_gradient:
    1. For Poisson: sim_for_poisson = sim_charge / c * stop_gradient(c)
       - This blocks gradient path to qe_corrections
    2. For MSE: raw_sim_sg = stop_gradient(sim_charge / c)
       - This blocks gradient path to physics params
       - Gradients only flow through raw_true = true_charge / c

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
    qe_corrections : jnp.ndarray
        Array of shape (N,) of per-sensor QE correction factors.
    tau : float, optional
        Gaussian width parameter for spatial smoothing, by default 0.05
    eps : float, optional
        Small constant to prevent division by zero, by default 1e-8
    threshold : float, optional
        Threshold for considering a sensor active, by default 1e-8
    lambda_poisson : float, optional
        Scaling factor for Poisson loss (physics params), by default 1.0.
    lambda_time : float, optional
        Scaling factor for time loss, by default 1.0.
    lambda_mse : float, optional
        Scaling factor for MSE loss (QE corrections), by default 1.0.

    Returns
    -------
    float
        Total loss.
    """
    # ============= Compute Distance-based Gaussian Weights =============
    dist = jnp.linalg.norm(
        sensor_points[:, None, :] - sensor_points[None, :, :],
        axis=-1
    )
    dist_weights = jnp.exp(-dist**2 / (2 * tau**2))
    col_sums = jnp.sum(dist_weights, axis=0, keepdims=True)
    dist_weights = dist_weights / (col_sums + eps)

    # ============= Poisson Loss (physics params ONLY) =============
    qe_corrections_sg = jax.lax.stop_gradient(qe_corrections)
    sim_for_poisson = simulated_charge / (qe_corrections + eps) * qe_corrections_sg

    true_charge_smooth = dist_weights @ true_charge
    sim_charge_smooth = dist_weights @ sim_for_poisson
    poisson_loss = poisson_nll(true_charge_smooth, sim_charge_smooth, eps)

    # ============= MSE Loss (QE corrections ONLY) =============
    raw_sim = simulated_charge / (qe_corrections + eps)
    raw_sim_sg = jax.lax.stop_gradient(raw_sim)

    raw_true = true_charge / (qe_corrections + eps)

    mse_loss = jnp.mean((raw_true - raw_sim_sg) ** 2) / jnp.sum(raw_true**2)

    # ============= Time Loss =============
    true_active = true_charge > threshold
    sim_active = simulated_charge > threshold

    true_weights = jnp.where(true_active, true_charge, 0.0)
    sim_weights = jnp.where(sim_active, simulated_charge, 0.0)

    true_t0 = jax.lax.stop_gradient(jnp.sum(true_time * true_weights) / (jnp.sum(true_weights) + eps))
    sim_t0 = jnp.sum(simulated_time * sim_weights) / (jnp.sum(sim_weights) + eps)

    both_active = jax.lax.stop_gradient(true_active & sim_active)
    diff = jnp.where(both_active,
                     jnp.abs((simulated_time - sim_t0) - (true_time - true_t0)),
                     0.0)
    time_norm = jnp.where(both_active, jnp.abs(true_time - true_t0), 0.0)
    time_loss = jnp.sum(diff) / (jnp.sum(time_norm) + eps)

    # ============= Combine Losses =============
    L_poisson = poisson_loss * lambda_poisson
    L_time = time_loss * lambda_time
    L_mse = mse_loss * lambda_mse

    return L_poisson + L_time + L_mse


# ===================================================================
# Optimization loss functions (formerly in lucid/optimization/losses.py)
# ===================================================================

@jit
def energy_loss(simulated_counts, true_counts):
    """
    Core energy loss computation using intensity matching.

    Parameters:
    -----------
    simulated_counts : jnp.ndarray
        Simulated counts values
    true_counts : jnp.ndarray
        True counts values

    Returns:
    --------
    float
        Energy loss value
    """
    total_true_counts = jnp.sum(true_counts)
    total_sim_counts = jnp.sum(simulated_counts)
    eps = 1e-8

    return jnp.abs(jnp.log(total_sim_counts / (total_true_counts + eps)))

@jit
def counts_loss(true: jnp.ndarray, pred: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
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

    # cap=1000.0
    # nll = pred - true * jnp.log(pred + eps) + gammaln(true + 1.0)

    # # Smooth saturating cap
    # nll_capped = cap * (1.0 - jnp.exp(-nll / cap))

    # return jnp.sum(nll_capped) / (jnp.sum(true) + eps)

@jit
def grid_origin_time_loss(
    origin,
    detector_positions,
    true_times,
    true_q,
    t0,
    photosensor_radius=0.25,
    c_medium=(0.299792 / 1.33),
    w_neg=100.0,
):
    eps = 1e-9

    # --- Residuals
    distances = jnp.linalg.norm(detector_positions - origin[None, :], axis=1)
    expected_times = (distances - photosensor_radius) / c_medium
    r = true_times - expected_times - t0

    # --- Active sensors
    mask = (true_q > 0.).astype(jnp.float32)
    n = jnp.sum(mask) + eps

    # --- Split residuals
    r_neg = jnp.clip(-r, 0.0, jnp.inf)
    r_pos = jnp.clip(r, 0.0, jnp.inf)

    # --- 1) Penalty for early photons
    neg_pen = jnp.sum(r_neg * mask) / n

    # --- 2) Positive part analysis
    pos_mask = mask * (r > 0.0).astype(jnp.float32)
    n_pos = jnp.sum(pos_mask)
    has_pos = n_pos > 0

    # Prevent divisions by zero
    safe_n_pos = jnp.maximum(n_pos, 1.0)

    sum_pos = jnp.sum(r_pos * pos_mask)
    mean_pos = sum_pos / safe_n_pos
    mean_pos = jnp.maximum(mean_pos, eps)  # avoid log(0)

    # NLL term (only meaningful if we have positives)
    nll_pos = (n_pos * jnp.log(mean_pos)) / (n + eps)

    # Mask all positive-related terms if no positives
    total_loss = w_neg * neg_pen + has_pos * (nll_pos)

    # Replace NaNs by large number to avoid contamination
    total_loss = jnp.nan_to_num(total_loss, nan=1e6, posinf=1e6, neginf=1e6)
    return total_loss

def softplus(x):
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.)

def smooth_pinball(r, tau=0.1, sigma=0.5):
    """
    Smooth version of pinball/quantile loss.
    Minimizer sets approx quantile_tau(r) ~= 0.
    """
    # Approx max(r,0) and max(-r,0)
    pos = softplus(r / sigma) * sigma
    neg = softplus(-r / sigma) * sigma
    return tau * pos + (1.0 - tau) * neg

@jit
def origin_time_loss(origin, detector_positions, true_times, true_q, t0,
                     photosensor_radius=0.25, c_medium=(0.299792/1.33), tau=0.23):
    d = jnp.linalg.norm(detector_positions - origin[None, :], axis=1)
    expected = (d - photosensor_radius) / c_medium

    r = true_times - expected - t0

    w = jnp.where(true_q > 0., 1., 0.)
    wsum = jnp.sum(w) + 1e-8

    main = jnp.sum(w * smooth_pinball(r, tau=tau, sigma=0.25)) / wsum

    return main

@jit
def cone_time_loss(observed_counts, simulated_time, observed_times, t0, tau=0.12):
    r = observed_times - simulated_time - t0
    w = jnp.where(observed_counts > 0., observed_counts, 0.)
    wsum = jnp.sum(w) + 1e-8
    main = jnp.sum(w * smooth_pinball(r, tau=tau, sigma=0.25)) / wsum
    return main


# ===================================================================
# Likelihood-based loss functions
# ===================================================================

def segment_logsumexp(data, indices, num_segments):
    """Numerically stable log-sum-exp aggregated per segment."""
    max_vals = jax.ops.segment_max(
        data, indices, num_segments=num_segments,
        indices_are_sorted=False
    )
    shifted = data - max_vals[indices]
    exp_summed = jax.ops.segment_sum(
        jnp.exp(shifted), indices, num_segments=num_segments
    )
    return max_vals + jnp.log(exp_summed)


def first_arrival_nll(log_w, flat_times, flat_indices,
                      t_obs_per_sensor, tau, num_detectors):
    t_obs_per_photon = t_obs_per_sensor[flat_indices]
    x = (t_obs_per_photon - flat_times) / tau

    # Filter invalid photons
    valid = log_w > -20.0
    safe_log_w = jnp.where(valid, log_w, -1e6)

    # Normalize weights per sensor in log-space
    log_w_total = segment_logsumexp(safe_log_w, flat_indices, num_detectors)
    log_w_norm = safe_log_w - log_w_total[flat_indices]

    # N_s = expected photon count per sensor
    N_s = jnp.exp(log_w_total)

    # log f(t_obs) — mixture density
    log_kernel = (jax.nn.log_sigmoid(x)
                  + jax.nn.log_sigmoid(-x)
                  - jnp.log(tau))
    log_f = segment_logsumexp(
        log_w_norm + log_kernel, flat_indices, num_detectors
    )

    # log(1 - F) via the identity: 1-F = Σ p_i σ(-x_i)
    log_one_minus_F = segment_logsumexp(
        log_w_norm + jax.nn.log_sigmoid(-x), flat_indices, num_detectors
    )

    # Order statistic NLL: -log[N_s · f · (1-F)^{N_s-1}]
    loss = -log_w_total - log_f - (N_s - 1) * log_one_minus_F

    return loss


# =============================================================================
# TAU_VTX PARAMETRIZATION
# =============================================================================
# Coefficients from weighted least-squares fit on tau hyperparameter scan.
# To recalculate these parameters:
#   1. Run: python s3df_jobs/submit_tau_hyperparameter_tuning_job.py --output output/tau_scan --submit
#   2. Wait for job completion, results in output/tau_scan/result.csv
#   3. Run analysis notebook: good_notebooks/analyze_tau_scan.ipynb
#   4. Update coefficients below with new fit results

TAU_VTX_PARAM_A = 1.092557e-06  # coefficient for Nrays
TAU_VTX_PARAM_B = 2.578522e-04  # coefficient for Energy (MeV)
TAU_VTX_PARAM_C = -0.0442       # intercept


def get_optimal_tau_vtx(nrays, energy_mev):
    """
    Get optimal tau_vtx based on learned parametrization.

    tau_vtx = a * Nrays + b * Energy + c

    This parametrization was derived from scanning tau_vtx across
    different (Nrays, Energy) combinations and fitting to minimize
    position reconstruction error.

    Args:
        nrays: Number of photon rays (can be JAX array or scalar)
        energy_mev: Energy in MeV (can be JAX array or scalar)

    Returns:
        Optimal tau_vtx value (typically in range 0.1-0.8)
    """
    return TAU_VTX_PARAM_A * nrays + TAU_VTX_PARAM_B * energy_mev + TAU_VTX_PARAM_C


# =============================================================================
# ALIASES FOR CONSISTENCY ACROSS CODEBASE
# =============================================================================
# poisson_nll is an alias for counts_loss (same formula)
poisson_nll = counts_loss

# origin_time_loss already accepts tau parameter, so it can be used as configurable
# This alias makes the API clearer when using dynamic tau_vtx
origin_time_loss_configurable = origin_time_loss