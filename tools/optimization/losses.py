"""
Loss functions for optimization in LUCiD.
"""

import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy import special
from jax.scipy.special import gammaln
from functools import partial

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

# @jit
# def grid_origin_time_loss(
#     origin,
#     detector_positions,
#     true_times,
#     true_q,
#     t0,
#     photosensor_radius=0.25,
#     c_medium=(0.299792 / 1.33),
#     w_neg=200.0,
#     w_nll=5.0,
#     w_shape_cv=0.01,
#     w_shape_med=0.01,
# ):
#     eps = 1e-9

#     # --- Residuals
#     distances = jnp.linalg.norm(detector_positions - origin[None, :], axis=1)
#     expected_times = (distances - photosensor_radius) / c_medium
#     r = true_times - expected_times - t0

#     # --- Active sensors
#     mask = (true_q > 0.).astype(jnp.float32)
#     n = jnp.sum(mask) + eps

#     # --- Split residuals
#     r_neg = jnp.clip(-r, 0.0, jnp.inf)
#     r_pos = jnp.clip(r, 0.0, jnp.inf)

#     # --- 1) Penalty for early photons
#     neg_pen = w_neg * jnp.sum(r_neg * mask) / n

#     # --- 2) Positive part analysis
#     pos_mask = mask * (r > 0.0).astype(jnp.float32)
#     n_pos = jnp.sum(pos_mask)
#     has_pos = n_pos > 0

#     # Prevent divisions by zero
#     safe_n_pos = jnp.maximum(n_pos, 1.0)

#     sum_pos = jnp.sum(r_pos * pos_mask)
#     mean_pos = sum_pos / safe_n_pos
#     mean_pos = jnp.maximum(mean_pos, eps)  # avoid log(0)

#     # NLL term (only meaningful if we have positives)
#     nll_pos = w_nll * (n_pos * jnp.log(mean_pos)) / (n + eps)

#     # 3) Shape penalties (compute safely)
#     var_pos = jnp.sum(((r_pos - mean_pos) ** 2) * pos_mask) / safe_n_pos
#     std_pos = jnp.sqrt(var_pos + eps)
#     cv = std_pos / (mean_pos + eps)
#     cv_pen = w_shape_cv * (cv - 1.0) ** 2

#     med_pen = 0.0  # Skip median for robustness (optional)

#     # Mask all positive-related terms if no positives
#     total_loss = neg_pen + has_pos * (nll_pos + cv_pen + med_pen)

#     # Replace NaNs by large number to avoid contamination
#     total_loss = jnp.nan_to_num(total_loss, nan=1e6, posinf=1e6, neginf=1e6)
#     return total_loss


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

# @jit
# def origin_time_loss(
#     origin,
#     detector_positions,
#     true_times,
#     true_q,
#     t0,
#     photosensor_radius=0.25,
#     c_medium=(0.299792 / 1.33),
#     w_neg=40.0,
# ):
#     eps = 1e-9

#     # --- Residuals
#     distances = jnp.linalg.norm(detector_positions - origin[None, :], axis=1)
#     expected_times = (distances - photosensor_radius) / c_medium
#     r = true_times - expected_times - t0

#     # --- Active sensors
#     #mask = (true_q > 0.).astype(jnp.float32)
#     n = jnp.sum(true_q) + eps

#     # --- Split residuals
#     r_neg = jnp.clip(-r, 0.0, jnp.inf)
#     r_pos = jnp.clip(r, 0.0, jnp.inf)

#     # --- 1) Penalty for early photons
#     neg_pen = jnp.sum(r_neg * true_q) / n

#     # --- 2) Positive part analysis
#     pos_mask = true_q * (r > 0.0).astype(jnp.float32)
#     n_pos = jnp.sum(pos_mask)
#     has_pos = n_pos > 0

#     # Prevent divisions by zero
#     safe_n_pos = jnp.maximum(n_pos, 1.0)

#     sum_pos = jnp.sum(r_pos * pos_mask)
#     mean_pos = sum_pos / safe_n_pos
#     mean_pos = jnp.maximum(mean_pos, eps)  # avoid log(0)

#     # NLL term (only meaningful if we have positives)
#     nll_pos = (n_pos * jnp.log(mean_pos)) / (n + eps)

#     # Mask all positive-related terms if no positives
#     total_loss = w_neg * neg_pen + has_pos * (nll_pos)

#     # Replace NaNs by large number to avoid contamination
#     total_loss = jnp.nan_to_num(total_loss, nan=1e6, posinf=1e6, neginf=1e6)
#     return total_loss


# @jit
# def origin_time_loss(
#     origin,
#     detector_positions,
#     true_times,
#     true_q,
#     t0,
#     photosensor_radius=0.25,
#     c_medium=(0.299792 / 1.33),
#     w_neg=0.05,
# ):
#     eps = 1e-9


#     # --- Residuals
#     distances = jnp.linalg.norm(detector_positions - origin[None, :], axis=1)
#     expected_times = (distances - photosensor_radius) / c_medium
#     r = true_times - expected_times - t0

#     # --- Active sensors
#     #mask = (true_q > 0.).astype(jnp.float32)
#     mask = true_q
#     n = jnp.sum(mask) + eps

#     # --- Split residuals
#     r_neg = jnp.clip(-r, 0.0, jnp.inf)
#     r_pos = jnp.clip(r, 0.0, jnp.inf)

#     # --- 1) Penalty for early photons
#     #neg_pen = jnp.sum(r_neg * mask) / n
#     neg_pen = jnp.sum(r_neg * (true_q > 0.).astype(jnp.float32)/0.4)

#     # --- 2) Positive part analysis
#     pos_mask = mask * (r > 0.0).astype(jnp.float32)
#     n_pos = jnp.sum(pos_mask)
#     has_pos = n_pos > 0

#     # Prevent divisions by zero
#     safe_n_pos = jnp.maximum(n_pos, 1.0)

#     sum_pos = jnp.sum(r_pos * pos_mask)
#     mean_pos = sum_pos / safe_n_pos
#     mean_pos = jnp.maximum(mean_pos, eps)  # avoid log(0)

#     # NLL term (only meaningful if we have positives)
#     nll_pos = (n_pos * jnp.log(mean_pos)) / (n + eps)

#     # Mask all positive-related terms if no positives
#     total_loss = w_neg * neg_pen + has_pos * (nll_pos)
#     #jax.debug.print("terms: {}     {}", w_neg * neg_pen, has_pos * (nll_pos))

#     # Replace NaNs by large number to avoid contamination
#     total_loss = jnp.nan_to_num(total_loss, nan=1e6, posinf=1e6, neginf=1e6)
#     return total_loss


@jit
def origin_time_loss(origin, detector_positions, true_times, true_q, t0, photosensor_radius=0.25, c_medium=(0.299792/1.33)):
    """Vertex time loss component"""
    distances = jnp.linalg.norm(detector_positions - origin[None, :], axis=1)
    expected_times = (distances - photosensor_radius)/ c_medium
    time_residuals = true_times - expected_times - (t0 + 0.15)
    
    shift = 0.075

    w1 = jnp.where(true_q>0., true_q, 0.)

    neg_time_res = jnp.where(time_residuals < 0, jnp.abs(time_residuals), 0.0)

    ultra_rel_term = jnp.sum(jnp.abs(neg_time_res*w1)) / (jnp.sum(w1) + 1e-8)+shift

    pos_time_res = jnp.where(time_residuals > 0, jnp.abs(time_residuals), 0.0)

    w2 = jnp.where(true_q>0., true_q, 0.)
    
    proximity_term =  jnp.sum(jnp.abs(time_residuals*w2**2)) / (jnp.sum(w2) + 1e-8)+shift

    total_loss = (ultra_rel_term*proximity_term)
 
    return total_loss


@jit
def direction_time_loss(origin, direction, detector_positions, true_times, true_q, t0, photosensor_radius=0.25, c_medium=(0.299792/1.33), \
                        inv_scale_w=0.5, angle_scale_deg=1.5):
    """Direction-aware time loss component using Cherenkov angle weighting"""
    # Stop gradients for origin
    origin_no_grad = jax.lax.stop_gradient(origin)

    # Calculate distances and time residuals (same as origin_time_loss)
    distances = jnp.linalg.norm(detector_positions - origin_no_grad[None, :], axis=1)
    expected_times = (distances - photosensor_radius) / c_medium
    time_residuals = true_times - expected_times - jax.lax.stop_gradient(t0)
    
    w = jnp.where(true_q>0., true_q, 0.)

    # Calculate vectors from origin to each detector
    vectors_to_detectors = detector_positions - origin_no_grad[None, :]
    vectors_normalized = vectors_to_detectors / (jnp.linalg.norm(vectors_to_detectors, axis=1, keepdims=True) + 1e-8)

    # Normalize direction
    direction_normalized = direction / (jnp.linalg.norm(direction) + 1e-8)

    # Calculate opening angles
    cos_angles = jnp.dot(vectors_normalized, direction_normalized)
    cos_angles = jnp.clip(cos_angles, -1.0, 1.0)
    opening_angles = jnp.arccos(cos_angles)

    # Cherenkov angle: cos(θ_c) = 1/n
    n_refraction = 1.0 / (c_medium / 0.299792)
    cherenkov_angle = jnp.arccos(1.0 / n_refraction)

    # Angular weight peaked at Cherenkov angle
    angle_scale_rad = angle_scale_deg * jnp.pi / 180.0
    w_opening = jnp.exp(-0.5 * ((opening_angles - cherenkov_angle) / angle_scale_rad) ** 2)

    # Combined weight
    w_combined = w * w_opening

    # Time residual loss components
    neg_time_res = jnp.where(time_residuals < 0, jnp.abs(time_residuals), 0.0)
    pos_time_res = jnp.where(time_residuals > 0, jnp.abs(time_residuals), 0.0)

    neg_loss_per_detector = neg_time_res * w_combined
    pos_loss_per_detector = pos_time_res * w_combined

    total_loss = 1.+inv_scale_w/(inv_scale_w+jnp.sum(neg_loss_per_detector) / (jnp.sum(w_combined) + 1e-8))

    return total_loss

