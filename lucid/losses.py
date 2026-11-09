import jax.numpy as jnp
from jax import jit
import jax
from jax.scipy.special import gammaln, erf


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
    Charge and time loss with Gaussian distance weights.

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
    poisson_loss = poisson_nll(true_charge_smooth, simulated_charge_smooth, eps)

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
    L_charge = poisson_loss * lambda_poisson
    L_time = time_loss * lambda_time

    return L_charge + L_time


# ===================================================================
# Optimization loss functions
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
def counts_loss(true: jnp.ndarray, pred: jnp.ndarray, eps: float = 1e-8,
                normalize: bool = True) -> jnp.ndarray:
    """Compute the Poisson negative log-likelihood loss.

    Parameters
    ----------
    true : jnp.ndarray
        Ground truth values (non-negative).
    pred : jnp.ndarray
        Predicted values (non-negative).
    eps : float, optional
        Small constant to prevent log(0), by default 1e-8.
    normalize : bool, optional
        If True (default), divide by ``sum(true)`` (the historical behavior — preserves
        every existing caller). If False, return the raw ``sum(nll)`` — the form recon
        reconstruction uses (normalizing by the total light kills the energy constraint).

    Returns
    -------
    jnp.ndarray
        Poisson negative log-likelihood loss.
    """
    nll = pred - true * jnp.log(pred + eps) + gammaln(true + 1.0)
    total = jnp.sum(nll)
    # jnp.where (not a Python ``if``) so ``normalize`` stays jit-traceable as a flag.
    return jnp.where(normalize, total / (jnp.sum(true) + eps), total)

@jit
def grid_origin_time_loss(origin, detector_positions, true_times, true_q, t0, *,
                          sensor_radius, light_speed, early_penalty):
    """First-arrival residual loss of a vertex hypothesis, for the first-light seed grid.

    The expected arrival at each hit sensor is the straight-line flight from ``origin`` to the
    sensor face (``sensor_radius`` short of its centre) at ``light_speed`` (m/ns), after ``t0``.
    Arrivals earlier than that are impossible and cost ``early_penalty`` per ns; the late
    residuals enter as the log of their mean.
    """
    eps = 1e-9

    # --- Residuals
    distances = jnp.linalg.norm(detector_positions - origin[None, :], axis=1)
    expected_times = (distances - sensor_radius) / light_speed
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
    return early_penalty * neg_pen + has_pos * (nll_pos)


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
    """Per-sensor first-arrival NLL of a logistic-smeared photon mixture.

    Each predicted photon arrives at ``flat_times`` smeared by a logistic kernel of scale ``tau``;
    the observed time is the first of ``N_s`` arrivals, with ``N_s`` the sensor's expected count,
    so the loss is ``-log[N_s f(t) (1 - F(t))^(N_s - 1)]``. An alternative to
    :func:`first_arrival_window_nll` (the reconstruction's time term), kept for comparing time
    likelihoods. Empty sensors contribute 0.
    """
    t_obs_per_photon = t_obs_per_sensor[flat_indices]
    x = (t_obs_per_photon - flat_times) / tau

    # Filter invalid photons
    valid = log_w > -20.0
    safe_log_w = jnp.where(valid, log_w, -1e6)

    # Normalize weights per sensor in log-space
    log_w_total = segment_logsumexp(safe_log_w, flat_indices, num_detectors)
    log_w_norm = safe_log_w - log_w_total[flat_indices]

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

    # --- Empty/degenerate-segment safety -------------------------------------
    # A sensor with no predicted photons yields segment_logsumexp = -inf for all
    # three terms, so the raw order-statistic loss below is (inf - inf) = NaN.
    # Worse, masking the output afterwards does NOT remove the NaN gradient
    # (0 * NaN = NaN in the VJP). Floor the per-sensor log terms to a finite
    # value so the loss is finite *everywhere*; the floor's max() gate also
    # zeros the gradient on empty segments. Real sensors (log_w_total ~ O(1))
    # are nowhere near the floor, so well-posed numerics are unchanged.
    LOG_FLOOR = -60.0
    empty = log_w_total <= (LOG_FLOOR + 1.0)
    log_w_total_s = jnp.maximum(log_w_total, LOG_FLOOR)
    log_f_s = jnp.maximum(log_f, LOG_FLOOR)
    log_one_minus_F_s = jnp.maximum(log_one_minus_F, LOG_FLOOR)

    # N_s = expected photon count per sensor (clamped to keep exp finite)
    N_s = jnp.exp(jnp.clip(log_w_total_s, LOG_FLOOR, 50.0))

    # Order statistic NLL: -log[N_s · f · (1-F)^{N_s-1}]
    loss = -log_w_total_s - log_f_s - (N_s - 1) * log_one_minus_F_s

    # Empty segments contribute nothing (caller decides whether to penalise the
    # missing prediction via the charge term); finite loss -> clean gradient.
    loss = jnp.where(empty, 0.0, loss)

    return loss


def _log1mexp(a):
    """Numerically-stable ``log(1 - exp(a))`` for ``a < 0`` (Mächler 2012)."""
    a = jnp.minimum(a, -1e-7)
    return jnp.where(a > -0.6931, jnp.log(-jnp.expm1(a)), jnp.log1p(-jnp.exp(a)))


def first_arrival_window_nll(log_w, flat_times, flat_indices, t_obs_per_sensor,
                             mu_total, obs_counts, num_detectors, *, sigma, delta):
    """Windowed first-arrival ORDER-STATISTIC time NLL — the reconstruction time term.

    Mean-field-correct first arrival (the recon recipe). The
    expected-value engine cannot represent a model ``min`` (an order statistic), so instead
    build ``R(t) = Σ wᵢ·½(1+erf((t−tᵢ)/(σ√2)))`` = the expected cumulative photon count before
    ``t`` (a MEAN, hence engine-exact), the analytic survival ``S(t) = (μ − R(t))/μ``, and the
    count-conditioned NLL over the window ``[t_obs ± δ/2]``::

        tnll = −log( S(t_lo)ⁿ − S(t_hi)ⁿ )   (the probability that the first of n arrivals lies in the window)

    computed as ``−n·log S(t_lo) − log1mexp(n·(log S(t_hi) − log S(t_lo)))`` for stability.

    AMP_DETACH is baked in: the photon weights ``ww`` and the per-PMT total ``μ`` are
    ``stop_gradient``'d, so this term's gradient flows ONLY through the predicted times
    ``flat_times`` — charge owns counts/energy, time owns geometry (without the detach the two
    terms fight and energy blows up). Per-PMT; unlit PMTs (``obs_counts<=0``) contribute 0.

    Parameters
    ----------
    log_w, flat_times, flat_indices : per-photon log-weight, predicted arrival time, sensor idx
        (the ``per_photon`` hit-mode outputs).
    t_obs_per_sensor : (num_detectors,) observed first-arrival time, on the clock of
        ``flat_times`` (the track forward starts the predicted ones at the track's t0).
    mu_total : (num_detectors,) predicted per-PMT total ``μ`` — the survival denominator
        ``S=(μ−R)/μ``. Pass the UNSCALED engine total (not an energy-rescaled charge):
        scaling the survival denominator corrupts ``S`` and collapses far-capture (the recon
        recipe keeps it unscaled). It is ``stop_gradient``'d here anyway.
    obs_counts : (num_detectors,) observed per-PMT count ``n``.
    num_detectors : int.
    sigma : per-photon time resolution (= TTS), ns. delta : window width, ns.

    Returns
    -------
    (num_detectors,) per-PMT time NLL (sum it for the scalar term).
    """
    sq = jnp.sqrt(2.0)
    ww = jax.lax.stop_gradient(jnp.exp(jnp.clip(log_w, -60, 20)))      # AMP_DETACH (weights)
    muS = jax.lax.stop_gradient(jnp.maximum(mu_total, 1e-8))           # AMP_DETACH (total)
    tobs = t_obs_per_sensor[flat_indices]
    Rlo = jax.ops.segment_sum(ww * 0.5 * (1 + erf((tobs - delta / 2 - flat_times) / (sigma * sq))),
                              flat_indices, num_segments=num_detectors)
    Rhi = jax.ops.segment_sum(ww * 0.5 * (1 + erf((tobs + delta / 2 - flat_times) / (sigma * sq))),
                              flat_indices, num_segments=num_detectors)
    Slo = jnp.clip((muS - Rlo) / muS, 1e-12, 1.)
    Shi = jnp.clip((muS - Rhi) / muS, 1e-12, 1.)
    n = jnp.maximum(obs_counts, 0.)
    # No clamp here: the only saturation cap on `a` is the -1e-7 inside `_log1mexp`. That cap is
    # active on many PMTs, so a reimplementation must use -1e-7 (a -1e-9 cap differs by 4.6 nats
    # per saturated PMT). `tests/fitting/test_first_arrival_a_cap.py` pins where the cap lives.
    a = n * (jnp.log(Shi) - jnp.log(Slo))
    return jnp.where(obs_counts > 0, -n * jnp.log(Slo) - _log1mexp(a), 0.)


# =============================================================================
# ALIASES FOR CONSISTENCY ACROSS CODEBASE
# =============================================================================
# poisson_nll is an alias for counts_loss (same formula)
poisson_nll = counts_loss