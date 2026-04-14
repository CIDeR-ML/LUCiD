"""Sensor response: make_hits_* functions."""
import jax
import jax.numpy as jnp
from lucid.utils import smear_times, smear_charges_SK_like

# ===================================================================
# make_hits functions
# ===================================================================

def make_hits_simulation(
        flat_weights, flat_indices, flat_times, num_detectors,
        qe=0.2, qe_corrections=None, threshold=1e-10, temperature=0.01):
    """Differentiable soft-min first-arrival timing with per-sensor QE corrections."""
    per_photon_qe = qe * qe_corrections[flat_indices]
    qe_weights = flat_weights * per_photon_qe

    valid_mask = (qe_weights > threshold) & (flat_times > 0) & jnp.isfinite(flat_times)
    filtered_times = jnp.where(valid_mask, flat_times, jnp.inf)

    detector_mins = jax.ops.segment_min(filtered_times, flat_indices, num_segments=num_detectors)
    photon_offsets = detector_mins[flat_indices]

    shifted_times = jnp.where(valid_mask, flat_times - photon_offsets, jnp.inf)
    exp_terms = jnp.where(valid_mask, jnp.exp(-shifted_times / temperature), 0.0)
    exp_sums = jax.ops.segment_sum(exp_terms, flat_indices, num_segments=num_detectors)

    segment_min_time = detector_mins - temperature * jnp.log(exp_sums + 1e-20)
    has_photons = jnp.isfinite(detector_mins)
    segment_min_time = jnp.where(has_photons, segment_min_time, jnp.inf)

    total_charge = jax.ops.segment_sum(qe_weights, flat_indices, num_segments=num_detectors)

    nonzero_mask = (total_charge > threshold) & jnp.isfinite(segment_min_time)
    measured_charge = jnp.where(nonzero_mask, total_charge, 0.0)
    measured_time = jnp.where(nonzero_mask, segment_min_time, 0.0)

    return measured_charge, measured_time


def make_hits_data(
        flat_weights, flat_indices, flat_times, num_detectors,
        qe=0.2, rng_key=None, threshold=1e-5, apply_smearing=False):
    """Data-mode hits with Bernoulli QE, segment_min timing, and optional SK-like smearing."""
    timing_mask = (flat_weights > threshold) & (flat_times > 0)
    filtered_times = jnp.where(timing_mask, flat_times, jnp.inf)

    rng_key, smear_time_key = jax.random.split(rng_key)
    qe_key, smear_counts_key = jax.random.split(rng_key)

    # Always apply Bernoulli QE sampling — when qe >= 1.0, uniform(0,1) < qe
    # is always true so all photons pass.  Avoids Python `if` on traced values.
    detection_probs = jax.random.uniform(qe_key, shape=flat_weights.shape)
    detected_mask = detection_probs < qe
    qe_weights = flat_weights * detected_mask.astype(jnp.float32)
    qe_filtered_times = jnp.where(detected_mask & timing_mask, flat_times, jnp.inf)

    total_charge = jax.ops.segment_sum(qe_weights, flat_indices, num_segments=num_detectors)
    detector_mins = jax.ops.segment_min(qe_filtered_times, flat_indices, num_segments=num_detectors)

    nonzero_mask = (total_charge > 1e-10) & (detector_mins > 0) & jnp.isfinite(detector_mins)

    if apply_smearing:
        measured_time = jnp.where(
            jnp.any(nonzero_mask),
            smear_times(detector_mins, key=smear_time_key),
            0.0
        )
        measured_charge = jnp.where(
            nonzero_mask,
            smear_charges_SK_like(total_charge, key=smear_counts_key),
            0
        )
    else:
        measured_time = jnp.where(jnp.any(nonzero_mask), detector_mins, 0.0)
        measured_charge = jnp.where(nonzero_mask, total_charge, 0)

    return measured_charge, measured_time


def make_hits_likelihood(
        flat_weights, flat_indices, flat_times, num_detectors,
        qe=0.2, qe_corrections=None, threshold=1e-10):
    """Likelihood mode: return per-photon log-weights and per-sensor total charge.

    Instead of aggregating times to per-sensor first-arrival values, this
    returns the raw per-photon arrays so that ``first_arrival_nll`` (or
    similar likelihood-based losses) can operate on them directly.

    Parameters
    ----------
    flat_weights : jnp.ndarray
        Per-photon detection weights (K * max_sensors * n_rays,).
    flat_indices : jnp.ndarray
        Per-photon sensor indices (same shape).
    flat_times : jnp.ndarray
        Per-photon arrival times in ns (same shape).
    num_detectors : int
        Total number of sensors.
    qe : float
        Quantum efficiency.
    qe_corrections : jnp.ndarray
        Per-sensor QE correction factors (num_detectors,).
    threshold : float
        Minimum weight to consider a photon valid.

    Returns
    -------
    log_w : jnp.ndarray
        Log of QE-corrected weights (per photon). Invalid photons get -1e10.
    safe_times : jnp.ndarray
        Arrival times with invalid entries zeroed out (per photon).
    flat_indices : jnp.ndarray
        Sensor indices (per photon, unchanged).
    total_charge : jnp.ndarray
        Predicted total charge per sensor (num_detectors,).
    """
    per_photon_qe = qe * qe_corrections[flat_indices]
    qe_weights = flat_weights * per_photon_qe

    valid_mask = (qe_weights > threshold) & (flat_times > 0) & jnp.isfinite(flat_times)
    safe_weights = jnp.where(valid_mask, qe_weights, 0.0)
    safe_times = jnp.where(valid_mask, flat_times, 0.0)
    log_w = jnp.where(valid_mask, jnp.log(safe_weights + 1e-30), -1e10)

    total_charge = jax.ops.segment_sum(safe_weights, flat_indices, num_segments=num_detectors)

    return log_w, safe_times, flat_indices, total_charge

