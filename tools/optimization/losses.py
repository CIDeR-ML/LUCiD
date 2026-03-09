"""
Loss functions for optimization in LUCiD.
"""

import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy import special
from jax.scipy.special import gammaln
from functools import partial
from tools.detector_params import ParticleParams

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

# @jit
# def origin_time_loss(origin, detector_positions, true_times, true_q, t0, photosensor_radius=0.25, c_medium=(0.299792/1.33)):
#     """Vertex time loss component"""
#     distances = jnp.linalg.norm(detector_positions - origin[None, :], axis=1)
#     expected_times = (distances - photosensor_radius)/ c_medium
#     time_residuals = true_times - expected_times - (t0 + 0.15)
    
#     shift = 0.075

#     w1 = jnp.where(true_q>0., true_q, 0.)

#     neg_time_res = jnp.where(time_residuals < 0, jnp.abs(time_residuals), 0.0)

#     ultra_rel_term = jnp.sum(jnp.abs(neg_time_res*w1)) / (jnp.sum(w1) + 1e-8)+shift

#     pos_time_res = jnp.where(time_residuals > 0, jnp.abs(time_residuals), 0.0)

#     w2 = jnp.where(true_q>0., true_q, 0.)
    
#     proximity_term =  jnp.sum(jnp.abs(time_residuals*w2**2)) / (jnp.sum(w2) + 1e-8)+shift

#     total_loss = jnp.sqrt((ultra_rel_term*proximity_term))
 
#     return total_loss

# def softplus(x):
#     return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.)

# def smooth_pinball(r, tau=0.1, sigma=0.5):
#     """
#     Smooth version of pinball/quantile loss.
#     Minimizer sets approx quantile_tau(r) ~= 0.
#     """
#     # Approx max(r,0) and max(-r,0)
#     pos = softplus(r / sigma) * sigma
#     neg = softplus(-r / sigma) * sigma
#     return tau * pos + (1.0 - tau) * neg

# @jit
# def origin_time_loss(origin, detector_positions, true_times, true_q, t0,
#                               photosensor_radius=0.25, c_medium=(0.299792/1.33)):
    
#     d = jnp.linalg.norm(detector_positions - origin[None, :], axis=1)
#     expected = (d - photosensor_radius) / c_medium

#     b = true_times - expected          # "implied t0" per sensor
#     r = b - t0

#     w = jnp.where(true_q > 0., true_q, 0.)
#     wsum = jnp.sum(w) + 1e-8

#     # Main term: quantile alignment => correct t0 minimum (not gradient peak)
#     main = jnp.sum(w * smooth_pinball(r, tau=0.70, sigma=0.05)) / wsum

#     return main

# @jit
# def cone_time_loss(observed_counts, simulated_time, observed_times, t0):
#     b = observed_times - simulated_time
#     r = b - t0
#     w = jnp.where(observed_counts > 0., observed_counts, 0.)
#     wsum = jnp.sum(w) + 1e-8
#     main = jnp.sum(w * smooth_pinball(r, tau=0.12, sigma=0.05)) / wsum
#     return main

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

def create_combined_loss_function(prediction_simulator):
    """Create combined loss function with specified parameters and return its gradient function"""

    @jit
    def combined_product_loss(track, hit_detector_positions, observed_times, observed_counts,
                                        true_data, key):
        """
        Combined loss function: product of vertex loss, counts loss, and energy loss

        Args:
            track: ParticleParams with energy, position, theta, phi, t0
            vertex_weight: scaling factor for vertex loss contribution
            counts_weight: scaling factor for counts loss contribution
            energy_weight: scaling factor for energy loss contribution
        """
        position = track.position
        t0 = track.t0
        theta = track.theta
        phi = track.phi
        energy = track.energy

        simulated_data = prediction_simulator(track, key)
        simulated_counts = simulated_data[0]
        simulated_time = simulated_data[1]

        # Calculate individual loss components
        vertex_loss_val = origin_time_loss(jax.lax.stop_gradient(position), hit_detector_positions, observed_times,
                                          observed_counts, t0)

        counts_loss_val = counts_loss(observed_counts, simulated_counts)
        time_loss_val = cone_time_loss(observed_counts, simulated_time, observed_times, t0)

        combined = jnp.sqrt(
            (vertex_loss_val + 1e-6) * (counts_loss_val + 1e-6) * (time_loss_val + 1e-6)
        ) + counts_loss_val
        return combined, (vertex_loss_val, counts_loss_val, vertex_loss_val)


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


# def first_arrival_nll(log_w, flat_times, flat_indices,
#                       t_obs_per_sensor, tau, num_detectors):
#     """First-arrival negative log-likelihood using a logistic kernel.
#
#     Parameters
#     ----------
#     log_w : jnp.ndarray
#         Log of per-photon QE-corrected weights.
#     flat_times : jnp.ndarray
#         Per-photon simulated arrival times.
#     flat_indices : jnp.ndarray
#         Per-photon sensor indices.
#     t_obs_per_sensor : jnp.ndarray
#         Observed first-arrival time per sensor (num_detectors,).
#     tau : float
#         Kernel width parameter.
#     num_detectors : int
#         Number of sensors.
#
#     Returns
#     -------
#     loss : jnp.ndarray
#         Per-sensor NLL (num_detectors,).
#     """
#     t_obs_per_photon = t_obs_per_sensor[flat_indices]
#     x = (t_obs_per_photon - flat_times) / tau
#
#     # log lambda: log(w_i * K(x_i)) where K is logistic density
#     log_kernel = (jax.nn.log_sigmoid(x)
#                   + jax.nn.log_sigmoid(-x)
#                   - jnp.log(tau))
#     log_lambda = segment_logsumexp(
#         log_w + log_kernel, flat_indices, num_detectors
#     )
#
#     # Lambda: cumulative intensity up to t_obs
#     log_Lambda = segment_logsumexp(
#         log_w + jax.nn.log_sigmoid(x), flat_indices, num_detectors
#     )
#
#     Lambda = jnp.exp(log_Lambda)
#
#     # softplus(log_Lambda) = log(1 + Lambda): recovers standard NLL
#     # near the optimum but grows logarithmically for large Lambda,
#     # preventing the t0 plateau from survival-term saturation
#     loss = -log_lambda + Lambda
#
#     return loss

# def first_arrival_nll(log_w, flat_times, flat_indices,
#                       t_obs_per_sensor, tau, num_detectors):
#     t_obs_per_photon = t_obs_per_sensor[flat_indices]
#     x = (t_obs_per_photon - flat_times) / tau
#
#     # log rate: log λ(t_obs)
#     log_kernel = (jax.nn.log_sigmoid(x)
#                   + jax.nn.log_sigmoid(-x)
#                   - jnp.log(tau))
#     log_lambda = segment_logsumexp(
#         log_w + log_kernel, flat_indices, num_detectors
#     )
#
#     # Cumulative intensity: Λ(t_obs) — must be exp, NOT softplus
#     log_Lambda = segment_logsumexp(
#         log_w + jax.nn.log_sigmoid(x), flat_indices, num_detectors
#     )
#     Lambda = jnp.exp(jnp.clip(log_Lambda, -20.0, 10.0))  # stability clamp
#
#     return -log_lambda + Lambda

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