"""Sensor response: make_hits_* functions."""
import os
import jax
import jax.numpy as jnp
from functools import partial
from lucid.utils import smear_times, smear_charges_SK_like

# Per-PHOTON transit-time-spread (TTS) sigma in ns for data-mode truth, env-gated (0 = off, default).
# Applied to each photon's time BEFORE the first-arrival segment_min, so the min carries the correct
# TTS early-bias (smearing the per-sensor min instead would be symmetric and physically wrong).
_TTS_PERPHOTON_NS = float(os.environ.get('TTS_NS', '0'))

# ===================================================================
# make_hits functions
# ===================================================================

def make_hits_simulation(
        flat_weights, flat_indices, flat_times, num_detectors,
        qe=0.2, qe_corrections=None, threshold=1e-10, temperature=0.1):
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


def _expected_min_normal(occ):
    """Smooth approx of E[min of ``occ`` iid N(0,1)] as a function of the continuous
    expected occupancy ``occ`` (≤0; →0 as occ→1, → −√(2 ln occ) for large occ).

    Blom's order-statistic approximation E[max_n] = Φ⁻¹((n−α)/(n−2α+1)), α=0.375,
    negated for the minimum, clamped so occ<1 gives no early-bias (a single photon's
    arrival has E[N(0,1)]=0). This is the occupancy-dependent first-arrival bias that
    a per-photon transit-time spread (TTS) imprints on the segment-min time.
    """
    n = jnp.clip(occ, 1.0, None)
    alpha = 0.375
    p = (n - alpha) / (n - 2.0 * alpha + 1.0)
    return -jax.scipy.special.ndtri(jnp.clip(p, 1e-6, 1.0 - 1e-6))


def make_hits_moments(
        flat_weights, flat_indices, flat_times, num_detectors,
        qe=0.2, qe_corrections=None, gain=None, spe_width=0.0, t0=None, tts=0.0,
        threshold=1e-10, temperature=0.1):
    """Compound-Poisson charge MOMENTS + first-arrival time, per sensor.

    The per-PMT charge is compound-Poisson: rate ``μ[s] = Σ flat_weights·qe·qe_corr``
    (expected detected photo-electrons), single-PE charge ~ (mean ``gain[s]``,
    relative width ``w = spe_width``). The validated moments (mie_hunter/chargedata.py):

        E[Q][s]   = gain[s] · μ[s]
        Var[Q][s] = gain[s]² · (1 + w²) · μ[s]

    The mean alone measures the degenerate product QE·gain; the variance measures the
    RATE μ (``v/m² = (1+w²)/μ``) and so BREAKS the per-PMT QE↔gain degeneracy and yields
    the SPE width. Time is the same soft-min first-arrival as ``make_hits_simulation``
    plus a per-PMT offset ``t0[s]`` (SK TQ-map).

    Neutral defaults (gain=1, w=0, t0=0) give ``mean = var = μ`` and the unshifted
    first-arrival — i.e. the charge/time of ``make_hits_simulation`` with an extra
    variance output equal to the charge.

    Returns
    -------
    mean_charge, var_charge, measured_time : each (num_detectors,)
    """
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

    # Rate μ (expected PE count), then the compound-Poisson moments.
    mu = jax.ops.segment_sum(qe_weights, flat_indices, num_segments=num_detectors)
    g = jnp.ones(num_detectors) if gain is None else gain
    mean_charge_raw = g * mu
    var_charge_raw = (g ** 2) * (1.0 + spe_width ** 2) * mu

    # Per-PMT time offset (TQ-map t0) + TTS occupancy-bias: the per-photon transit-time
    # spread (sigma=tts) pulls the first-arrival earlier by tts·E[min over μ photons].
    t0_arr = jnp.zeros(num_detectors) if t0 is None else t0
    tts_bias = tts * _expected_min_normal(mu)            # ≤0; →0 at tts=0 (byte-identical)
    measured_time_raw = segment_min_time + t0_arr + tts_bias

    nonzero_mask = (mu > threshold) & jnp.isfinite(segment_min_time)
    mean_charge = jnp.where(nonzero_mask, mean_charge_raw, 0.0)
    var_charge = jnp.where(nonzero_mask, var_charge_raw, 0.0)
    measured_time = jnp.where(nonzero_mask, measured_time_raw, 0.0)

    return mean_charge, var_charge, measured_time


def make_hits_data(
        flat_weights, flat_indices, flat_times, num_detectors,
        qe=0.2, qe_corrections=None, rng_key=None, threshold=1e-5, apply_smearing=False,
        tts=0.0):
    """Data-mode hits with Bernoulli QE, segment_min timing, and optional SK-like smearing.

    ``tts`` (ns) is the per-photon transit-time-spread sigma applied to each photon's
    time BEFORE the first-arrival segment_min (so the min carries the correct early
    bias). The module env ``TTS_NS`` still overrides when larger (legacy). ``tts=0``
    (and env unset) ⇒ no smear (byte-identical).
    """
    timing_mask = (flat_weights > threshold) & (flat_times > 0)
    filtered_times = jnp.where(timing_mask, flat_times, jnp.inf)

    rng_key, smear_time_key = jax.random.split(rng_key)
    qe_key, smear_counts_key = jax.random.split(rng_key)

    # Per-photon QE including per-sensor corrections (consistent with simulation/likelihood)
    per_photon_qe = qe * qe_corrections[flat_indices] if qe_corrections is not None else qe

    # Bernoulli QE sampling — when qe >= 1.0, uniform(0,1) < qe
    # is always true so all photons pass.  Avoids Python `if` on traced values.
    detection_probs = jax.random.uniform(qe_key, shape=flat_weights.shape)
    detected_mask = detection_probs < per_photon_qe
    qe_weights = flat_weights * detected_mask.astype(jnp.float32)
    # PER-PHOTON TTS: smear each detected photon's time BEFORE the first-arrival min.
    # Driven by the dp.response.tts field (env TTS_NS overrides when larger, legacy).
    # Applied unconditionally scaled by eff_tts (0 ⇒ no shift, byte-identical) so the
    # key stream is stable and tts stays differentiable.
    eff_tts = jnp.maximum(jnp.asarray(tts), _TTS_PERPHOTON_NS)
    photon_times = flat_times + jax.random.normal(smear_time_key, shape=flat_times.shape) * eff_tts
    qe_filtered_times = jnp.where(detected_mask & timing_mask, photon_times, jnp.inf)

    total_charge = jax.ops.segment_sum(qe_weights, flat_indices, num_segments=num_detectors)
    detector_mins = jax.ops.segment_min(qe_filtered_times, flat_indices, num_segments=num_detectors)

    # CHARGE is gated on a valid first-arrival time (detector_mins>0 & finite).
    nonzero_mask = (total_charge > 1e-10) & (detector_mins > 0) & jnp.isfinite(detector_mins)

    if apply_smearing:
        # PER-SENSOR gate (was scalar jnp.any → empty sensors got smeared inf/1e6 garbage,
        # PORT_PLAN §4.3): only lit sensors carry a time; empty sensors are exactly 0.
        measured_time = jnp.where(
            nonzero_mask,
            smear_times(detector_mins, key=smear_time_key),
            0.0
        )
        measured_charge = jnp.where(
            nonzero_mask,
            smear_charges_SK_like(total_charge, key=smear_counts_key),
            0
        )
    else:
        measured_time = jnp.where(nonzero_mask, detector_mins, 0.0)
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


# ===================================================================
# Shotgun mode: dense waveform & per-photon hit list
# ===================================================================

def _resolve_first_detection(
        flat_weights, flat_indices, flat_times, n_photons,
        per_photon_qe, qe_key, threshold):
    """Compact flat propagation arrays to per-photon first-detection records.

    Returns arrays of length ``n_photons``:
    - detected : bool    — did this photon pass QE Bernoulli at any iteration?
    - sensor_id : int32  — sensor hit (first-detection), or -1 if not detected
    - hit_time : float32 — propagation time at first detection (0 if not detected)
    """
    base_valid = (flat_weights > threshold) & (flat_times > 0) & jnp.isfinite(flat_times)
    detection_probs = jax.random.uniform(qe_key, shape=flat_weights.shape)
    detected_flat = base_valid & (detection_probs < per_photon_qe)

    photon_idx = jnp.arange(flat_weights.shape[0]) % n_photons

    safe_time = jnp.where(detected_flat, flat_times, jnp.inf)
    first_time = jax.ops.segment_min(safe_time, photon_idx, num_segments=n_photons)

    matches_first = detected_flat & (flat_times == first_time[photon_idx])
    safe_flat_idx = jnp.where(matches_first, jnp.arange(flat_weights.shape[0]),
                              jnp.iinfo(jnp.int32).max)
    first_flat_idx = jax.ops.segment_min(safe_flat_idx, photon_idx, num_segments=n_photons)

    detected = jnp.isfinite(first_time)
    sensor_id = jnp.where(detected, flat_indices[first_flat_idx], -1)
    hit_time = jnp.where(detected, first_time, 0.0)
    return detected, sensor_id, hit_time


def build_make_hits_waveform(
    n_photons,
    window_ns=500.0,
    bin_width_ns=1.0,
    tts_sigma_ns=1.0,
    t_min_ns=0.0,
    smear_time=True,
    smear_charge=True,
    threshold=1e-10,
):
    """Factory: returns a ``make_hits_waveform`` closure with baked-in bin grid.

    Pipeline: propagation flat arrays → per-photon first-detection (n_photons) →
    TTS + gain smearing on n_photons entries → bin to (num_detectors, n_time_bins).
    This keeps the segment_sum input small even when K×max_sensors×n_photons is large.

    Parameters
    ----------
    n_photons : int
        Number of photons per case (must match n_rays).
    window_ns : float
        Readout window in ns; default 500.
    bin_width_ns : float
        Waveform bin width in ns; default 1 ns (1 GHz FADC convention).
    tts_sigma_ns : float
        Gaussian σ of per-photon TTS; default 1.0 ns.
    t_min_ns : float
        Start of the window; default 0.
    smear_time, smear_charge : bool
        Toggle Gaussian TTS and SK-like gain smearing.
    threshold : float
        Minimum per-slot weight to treat as a physical candidate.

    Returns
    -------
    callable
        ``make_hits_waveform(flat_weights, flat_indices, flat_times,
        num_detectors, rng_key, qe, qe_corrections)`` returning
        ``(waveform, n_dropped, n_detected)``.
    """
    n_time_bins = int(round(window_ns / bin_width_ns))

    @partial(jax.jit, static_argnames=('num_detectors',))
    def make_hits_waveform(
            flat_weights, flat_indices, flat_times, num_detectors,
            rng_key, qe, qe_corrections):
        per_photon_qe = qe * qe_corrections[flat_indices]
        qe_key, tts_key, gain_key = jax.random.split(rng_key, 3)

        detected, sensor_id, hit_time = _resolve_first_detection(
            flat_weights, flat_indices, flat_times, n_photons,
            per_photon_qe, qe_key, threshold)

        if smear_time:
            noise = jax.random.normal(tts_key, shape=hit_time.shape) * tts_sigma_ns
            hit_time_smeared = hit_time + noise
        else:
            hit_time_smeared = hit_time

        bin_idx = jnp.floor((hit_time_smeared - t_min_ns) / bin_width_ns).astype(jnp.int32)
        in_window = (bin_idx >= 0) & (bin_idx < n_time_bins)

        charges = jnp.ones((n_photons,), dtype=jnp.float32)
        if smear_charge:
            charges = smear_charges_SK_like(charges, key=gain_key)

        keep = detected & in_window
        dropped = detected & ~in_window

        safe_sensor = jnp.where(keep, sensor_id, 0)
        safe_bin = jnp.where(keep, bin_idx, 0)
        flat_bin_idx = safe_sensor * n_time_bins + safe_bin
        safe_charge = jnp.where(keep, charges, 0.0)

        waveform_flat = jax.ops.segment_sum(
            safe_charge, flat_bin_idx, num_segments=num_detectors * n_time_bins)
        waveform = waveform_flat.reshape(num_detectors, n_time_bins)

        n_dropped = jnp.sum(dropped.astype(jnp.int32))
        n_detected = jnp.sum(detected.astype(jnp.int32))

        return waveform, n_dropped, n_detected

    make_hits_waveform.n_time_bins = n_time_bins
    make_hits_waveform.window_ns = float(window_ns)
    make_hits_waveform.bin_width_ns = float(bin_width_ns)
    make_hits_waveform.tts_sigma_ns = float(tts_sigma_ns)
    make_hits_waveform.t_min_ns = float(t_min_ns)
    return make_hits_waveform


def build_make_hits_waveform_expected(
    n_photons,
    window_ns=500.0,
    bin_width_ns=1.0,
    tts_sigma_ns=1.0,
    t_min_ns=0.0,
    smear_time=True,
    threshold=1e-10,
):
    """Factory: continuous QE-weighted waveform (no Bernoulli, no gain smearing).

    Companion to ``build_make_hits_waveform`` but for expected-value mode.
    Every propagation slot contributes its continuous ``flat_weight · QE``
    deposit to the ``(sensor, time_bin)`` cell it lands in. No Bernoulli coin
    is flipped; the output is the expected waveform given the sampled photon
    trajectories.

    Same ``(num_detectors, n_time_bins)`` output shape as
    ``build_make_hits_waveform`` — IO, merge, and analysis code are unchanged.
    ``n_dropped`` counts valid slots that landed outside the readout window;
    ``n_detected`` is the total (continuous) integrated charge across all
    sensors, which replaces the integer photon count from Bernoulli mode.

    Parameters
    ----------
    n_photons : int
        Photons per case. Kept for API symmetry with the Bernoulli factory —
        expected mode doesn't need it for first-detection compaction.
    window_ns, bin_width_ns, t_min_ns : float
        Readout window / bin grid (matches Bernoulli factory).
    tts_sigma_ns : float
        Per-slot Gaussian TTS σ. Each slot (photon × scattering iteration ×
        cell-sensor) gets its own draw — slightly over-smooths relative to a
        per-detection draw but the effect is tiny at σ ≲ bin_width.
    smear_time : bool
        Toggle TTS smearing.
    threshold : float
        Minimum per-slot weight to be counted as physical.
    """
    n_time_bins = int(round(window_ns / bin_width_ns))
    del n_photons  # unused; present for API symmetry

    @partial(jax.jit, static_argnames=('num_detectors',))
    def make_hits_waveform_expected(
            flat_weights, flat_indices, flat_times, num_detectors,
            rng_key, qe, qe_corrections):
        per_slot_qe = qe * qe_corrections[flat_indices]
        slot_charge = flat_weights * per_slot_qe

        if smear_time:
            noise = jax.random.normal(rng_key, shape=flat_times.shape) * tts_sigma_ns
            smeared_times = flat_times + noise
        else:
            smeared_times = flat_times

        bin_idx = jnp.floor((smeared_times - t_min_ns) / bin_width_ns).astype(jnp.int32)
        in_window = (bin_idx >= 0) & (bin_idx < n_time_bins)
        base_valid = (flat_weights > threshold) & (flat_times > 0) & jnp.isfinite(flat_times)
        keep = base_valid & in_window
        dropped = base_valid & ~in_window

        safe_sensor = jnp.where(keep, flat_indices, 0)
        safe_bin = jnp.where(keep, bin_idx, 0)
        flat_bin_idx = safe_sensor * n_time_bins + safe_bin
        safe_charge = jnp.where(keep, slot_charge, 0.0)

        waveform_flat = jax.ops.segment_sum(
            safe_charge, flat_bin_idx, num_segments=num_detectors * n_time_bins)
        waveform = waveform_flat.reshape(num_detectors, n_time_bins)

        n_dropped = jnp.sum(dropped.astype(jnp.int32))
        # In expected mode, "detected" is a continuous charge total, not a
        # photon count. Exposed via n_detected for reporting symmetry.
        n_detected = jnp.sum(waveform)

        return waveform, n_dropped, n_detected

    make_hits_waveform_expected.n_time_bins = n_time_bins
    make_hits_waveform_expected.window_ns = float(window_ns)
    make_hits_waveform_expected.bin_width_ns = float(bin_width_ns)
    make_hits_waveform_expected.tts_sigma_ns = float(tts_sigma_ns)
    make_hits_waveform_expected.t_min_ns = float(t_min_ns)
    return make_hits_waveform_expected


def build_make_hits_per_photon_shotgun(
    n_photons,
    tts_sigma_ns=1.0,
    smear_time=True,
    threshold=1e-10,
):
    """Factory: returns a ``make_hits_per_photon`` closure for shotgun mode.

    For each input photon, resolves the first-iteration detected slot (if any)
    and returns (detected_flag, sensor_id, hit_time) arrays of length ``n_photons``.

    Must be used with the MC-sampling propagator so weights are binary.

    Parameters
    ----------
    n_photons : int
        Number of photons per case (must match n_rays).
    tts_sigma_ns : float
        Gaussian σ of per-photon TTS; default 1.0 ns.
    smear_time : bool
        If True, apply Gaussian TTS smearing to hit times.
    threshold : float
        Minimum per-slot weight to consider a candidate.
    """

    @partial(jax.jit, static_argnames=('num_detectors',))
    def make_hits_per_photon(
            flat_weights, flat_indices, flat_times, num_detectors,
            rng_key, qe, qe_corrections):
        per_photon_qe = qe * qe_corrections[flat_indices]
        qe_key, tts_key = jax.random.split(rng_key)

        detected, sensor_id, hit_time = _resolve_first_detection(
            flat_weights, flat_indices, flat_times, n_photons,
            per_photon_qe, qe_key, threshold)

        if smear_time:
            noise = jax.random.normal(tts_key, shape=hit_time.shape) * tts_sigma_ns
            hit_time = jnp.where(detected, hit_time + noise, hit_time)

        return detected, sensor_id, hit_time

    return make_hits_per_photon
