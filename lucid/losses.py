from __future__ import annotations

import jax.numpy as jnp
from jax import jit
import jax
from jax.scipy.special import gammaln


@jit
def WC_loss(
        sensor_points: jax.Array,
        true_charge: jax.Array,
        true_time: jax.Array,
        simulated_charge: jax.Array,
        simulated_time: jax.Array,
        eps: float = 1e-8,
        threshold: float = 1e-8,
        lambda_poisson: float = 1.0,
        lambda_time: float = 1.0,
) -> jax.Array:
    """
    Optimized Loss
    Compute a loss function with two components:
    1. Poisson loss: Poisson negative log-likelihood for charge distributions
    2. Time loss: L1 (mean absolute error) on time differences

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
    poisson_loss = poisson_nll(true_charge, simulated_charge, eps)

    # Mean centering with LOW threshold (include all active sensors)
    true_active = true_charge > threshold
    sim_active = simulated_charge > threshold

    true_weights = jnp.where(true_active, true_charge, 0.0)
    sim_weights = jnp.where(sim_active, simulated_charge, 0.0)

    true_t0 = jax.lax.stop_gradient(jnp.sum(true_time * true_weights) / (jnp.sum(true_weights) + 1e-8))
    sim_t0 = jnp.sum(simulated_time * sim_weights) / (jnp.sum(sim_weights) + 1e-8)

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
        sensor_points: jax.Array,
        true_charge: jax.Array,
        true_time: jax.Array,
        simulated_charge: jax.Array,
        simulated_time: jax.Array,
        eps: float = 1e-8,
        threshold: float = 1e-8,
        lambda_centroid: float = 1.0,
        lambda_time: float = 1.0,
        lambda_intensity: float = 0.5
) -> jax.Array:
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
        Scaling factor for intensity loss, by default 0.5.

    Returns
    -------
    float
        Total loss.
    """
    true_active_mask = true_charge > threshold
    sim_active_mask = simulated_charge > threshold

    total_true_charge = jnp.sum(true_charge)
    total_sim_charge = jnp.sum(simulated_charge)

    # --------------------------------------
    # 1. Intensity Loss
    # --------------------------------------
    L_intensity = jnp.abs(jnp.log(total_sim_charge / (total_true_charge + eps))) * lambda_intensity

    # --------------------------------------
    # 2. Centroid Loss
    # --------------------------------------
    true_centroid = jnp.sum(true_charge[:, None] * sensor_points, axis=0) / (total_true_charge + eps)
    sim_centroid = jnp.sum(simulated_charge[:, None] * sensor_points, axis=0) / (total_sim_charge + eps)

    L_centroid = jnp.linalg.norm(true_centroid - sim_centroid) * lambda_centroid

    # --------------------------------------
    # 3. Simple Time Spread Loss
    # --------------------------------------
    true_mean_time = jax.lax.stop_gradient(jnp.sum(true_time * true_charge) / (total_true_charge + eps))
    sim_mean_time = jax.lax.stop_gradient(jnp.sum(simulated_time * simulated_charge) / (total_sim_charge + eps))

    # Subtract means from times to account for arbitrary time offsets
    true_time_centered = true_time - true_mean_time
    sim_time_centered = simulated_time - sim_mean_time

    true_time_var = jnp.sum(true_charge * jnp.square(true_time_centered)) / (total_true_charge + eps)
    sim_time_var = jnp.sum(simulated_charge * jnp.square(sim_time_centered)) / (total_sim_charge + eps)
    L_time_spread = jnp.abs(jnp.sqrt(true_time_var) - jnp.sqrt(sim_time_var))

    L_time = L_time_spread * lambda_time

    return L_centroid + L_intensity + L_time


@jit
def WC_smooth_loss(
        sensor_points: jax.Array,
        true_charge: jax.Array,
        true_time: jax.Array,
        simulated_charge: jax.Array,
        simulated_time: jax.Array,
        tau: float = 0.05,
        eps: float = 1e-8,
        threshold: float = 1e-8,
        lambda_poisson: float = 1.0,
        lambda_time: float = 1.0,
) -> jax.Array:
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
    dist = jnp.linalg.norm(
        sensor_points[:, None, :] - sensor_points[None, :, :],
        axis=-1
    )
    dist_weights = jnp.exp(-dist**2 / (2 * tau**2))
    col_sums = jnp.sum(dist_weights, axis=0, keepdims=True)
    dist_weights = dist_weights / (col_sums + eps)

    true_charge_smooth = dist_weights @ true_charge
    simulated_charge_smooth = dist_weights @ simulated_charge

    poisson_loss = poisson_nll(true_charge_smooth, simulated_charge_smooth, eps)

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

    L_charge = poisson_loss * lambda_poisson
    L_time = time_loss * lambda_time

    return L_charge + L_time



@jit
def energy_loss(simulated_counts: jax.Array, true_counts: jax.Array) -> jax.Array:
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
def counts_loss(true: jax.Array, pred: jax.Array, eps: float = 1e-8) -> jax.Array:
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

@jit
def grid_origin_time_loss(
    origin: jax.Array,
    detector_positions: jax.Array,
    true_times: jax.Array,
    true_q: jax.Array,
    t0: float,
    photosensor_radius: float = 0.25,
    c_medium: float = (0.299792 / 1.33),
    w_neg: float = 100.0,
) -> jax.Array:
    """Grid-based origin time loss for vertex position estimation."""
    eps = 1e-9

    distances = jnp.linalg.norm(detector_positions - origin[None, :], axis=1)
    expected_times = (distances - photosensor_radius) / c_medium
    r = true_times - expected_times - t0

    mask = (true_q > 0.).astype(jnp.float32)
    n = jnp.sum(mask) + eps

    r_neg = jnp.clip(-r, 0.0, jnp.inf)
    r_pos = jnp.clip(r, 0.0, jnp.inf)

    # --- 1) Penalty for early photons
    neg_pen = jnp.sum(r_neg * mask) / n

    # --- 2) Positive part analysis
    pos_mask = mask * (r > 0.0).astype(jnp.float32)
    n_pos = jnp.sum(pos_mask)
    has_pos = n_pos > 0

    safe_n_pos = jnp.maximum(n_pos, 1.0)

    sum_pos = jnp.sum(r_pos * pos_mask)
    mean_pos = sum_pos / safe_n_pos
    mean_pos = jnp.maximum(mean_pos, eps)  # avoid log(0)

    # NLL term (only meaningful if we have positives)
    nll_pos = (n_pos * jnp.log(mean_pos)) / (n + eps)

    total_loss = w_neg * neg_pen + has_pos * (nll_pos)

    # Replace NaNs by large number to avoid contamination
    total_loss = jnp.nan_to_num(total_loss, nan=1e6, posinf=1e6, neginf=1e6)
    return total_loss

def softplus(x: jax.Array) -> jax.Array:
    """Soft approximation to ReLU."""
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.)

def smooth_pinball(r: jax.Array, tau: float = 0.1, sigma: float = 0.5) -> jax.Array:
    """
    Smooth version of pinball/quantile loss.
    Minimizer sets approx quantile_tau(r) ~= 0.
    """
    # Approx max(r,0) and max(-r,0)
    pos = softplus(r / sigma) * sigma
    neg = softplus(-r / sigma) * sigma
    return tau * pos + (1.0 - tau) * neg

@jit
def origin_time_loss(
    origin: jax.Array,
    detector_positions: jax.Array,
    true_times: jax.Array,
    true_q: jax.Array,
    t0: float,
    photosensor_radius: float = 0.25,
    c_medium: float = (0.299792 / 1.33),
    tau: float = 0.23,
) -> jax.Array:
    """Time-of-origin loss for track reconstruction."""
    d = jnp.linalg.norm(detector_positions - origin[None, :], axis=1)
    expected = (d - photosensor_radius) / c_medium

    r = true_times - expected - t0

    w = jnp.where(true_q > 0., 1., 0.)
    wsum = jnp.sum(w) + 1e-8

    main = jnp.sum(w * smooth_pinball(r, tau=tau, sigma=0.25)) / wsum

    return main

@jit
def cone_time_loss(
    observed_counts: jax.Array,
    simulated_time: jax.Array,
    observed_times: jax.Array,
    t0: float,
    tau: float = 0.12,
) -> jax.Array:
    """Cherenkov cone timing loss."""
    r = observed_times - simulated_time - t0
    w = jnp.where(observed_counts > 0., observed_counts, 0.)
    wsum = jnp.sum(w) + 1e-8
    main = jnp.sum(w * smooth_pinball(r, tau=tau, sigma=0.25)) / wsum
    return main


def segment_logsumexp(data: jax.Array, indices: jax.Array, num_segments: int) -> jax.Array:
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


def first_arrival_nll(
    log_w: jax.Array,
    flat_times: jax.Array,
    flat_indices: jax.Array,
    t_obs_per_sensor: jax.Array,
    tau: float,
    num_detectors: int,
) -> jax.Array:
    """Negative log-likelihood based on first photon arrival."""
    t_obs_per_photon = t_obs_per_sensor[flat_indices]
    x = (t_obs_per_photon - flat_times) / tau

    valid = log_w > -20.0
    safe_log_w = jnp.where(valid, log_w, -1e6)

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


def get_optimal_tau_vtx(nrays: float, energy_mev: float) -> float:
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