"""Sensor response: make_hits_* functions."""
import jax
import jax.numpy as jnp
from functools import partial
from lucid.utils import smear_times, smear_charges_SK_like

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


def _qe_roll(flat_weights, flat_indices, flat_times,
             qe, qe_corrections, qe_key, threshold):
    """Per-photon Bernoulli QE survival.

    Shared between make_hits_data and make_hits_per_photon so both modes
    use identical RNG and threshold semantics.
    """
    timing_mask = (flat_weights > threshold) & (flat_times > 0)
    per_photon_qe = qe * qe_corrections[flat_indices] if qe_corrections is not None else qe
    detection_probs = jax.random.uniform(qe_key, shape=flat_weights.shape)
    detected_mask = detection_probs < per_photon_qe
    qe_weights = flat_weights * detected_mask.astype(jnp.float32)
    qe_filtered_times = jnp.where(detected_mask & timing_mask, flat_times, jnp.inf)
    return qe_weights, qe_filtered_times, detected_mask, timing_mask


def make_hits_data(
        flat_weights, flat_indices, flat_times, num_detectors,
        qe=0.2, qe_corrections=None, rng_key=None, threshold=1e-5,
        apply_smearing=False, tts_sigma_ns=2.5):
    """Data-mode hits with Bernoulli QE, per-photon TTS, then segment_min timing.

    Returns ``(measured_charge, measured_time_true, measured_time_reco)``.

    When ``apply_smearing`` is true, each detected photon receives its own
    Gaussian TTS draw *before* the per-sensor segment_min is taken — the
    physical PMT picture (each PE has independent transit-time jitter and the
    discriminator fires on the earliest detected pulse). Two segment_mins
    are computed: one on true (unsmeared) times and one on TTS-smeared times,
    so callers get both T_true and T_reco from a single kernel pass.

    ``tts_sigma_ns`` is the single-PE transit-time spread of the PMT (Gaussian
    σ in ns). Default 2.5 ns matches SK 20-inch R3600 (Fukuda et al. 2003,
    NIM A 501, 418). For HK 20-inch HQE R12860 use ~1.15 ns.
    """
    rng_key, smear_time_key = jax.random.split(rng_key)
    qe_key, smear_counts_key = jax.random.split(rng_key)

    qe_weights, qe_filtered_times, detected_mask, timing_mask = _qe_roll(
        flat_weights, flat_indices, flat_times,
        qe, qe_corrections, qe_key, threshold)

    total_charge = jax.ops.segment_sum(qe_weights, flat_indices, num_segments=num_detectors)

    # True (unsmeared) first-arrival per sensor.
    detector_mins_true = jax.ops.segment_min(
        qe_filtered_times, flat_indices, num_segments=num_detectors)

    # TTS-smeared first-arrival: per-photon Gaussian jitter before segment_min
    # so per-sensor time is min_i(t_i + ε_i) with correct N-dependent narrowing.
    if apply_smearing:
        smeared_flat_times = smear_times(
            flat_times, time_resolution=tts_sigma_ns, key=smear_time_key)
        qe_filtered_smeared = jnp.where(
            detected_mask & timing_mask, smeared_flat_times, jnp.inf)
        detector_mins_reco = jax.ops.segment_min(
            qe_filtered_smeared, flat_indices, num_segments=num_detectors)
    else:
        detector_mins_reco = detector_mins_true

    nonzero_mask = (
        (total_charge > 1e-10)
        & (detector_mins_true > 0)
        & jnp.isfinite(detector_mins_true))

    if apply_smearing:
        measured_charge = jnp.where(
            nonzero_mask,
            smear_charges_SK_like(total_charge, key=smear_counts_key),
            0)
    else:
        measured_charge = jnp.where(nonzero_mask, total_charge, 0)

    measured_time_true = jnp.where(nonzero_mask, detector_mins_true, 0.0)
    measured_time_reco = jnp.where(nonzero_mask, detector_mins_reco, 0.0)

    return measured_charge, measured_time_true, measured_time_reco


def make_hits_per_photon(
        flat_weights, flat_indices, flat_times, num_detectors,
        qe=0.2, qe_corrections=None, rng_key=None, threshold=1e-5,
        apply_smearing=False, tts_sigma_ns=2.5,
        flat_segment_idx=None):
    """Per-sensor totals PLUS pass-through per-photon arrays for host aggregation.

    Per-sensor outputs (measured_charge, measured_time_true, measured_time_reco)
    use identical QE and TTS draws as make_hits_data. The per-photon arrays are
    returned to the host so it can do per-(segment, sensor) and per-(particle,
    sensor) groupbys in numpy.

    Returns
    -------
    measured_charge        : (num_detectors,) float32
    measured_time_true     : (num_detectors,) float32 — first-arrival, no TTS
    measured_time_reco     : (num_detectors,) float32 — first-arrival, TTS-smeared
    qe_weights             : (n_rays_bucket,) float32 — 0 for QE-failed photons
    qe_filtered_times      : (n_rays_bucket,) float32 — unsmeared per-photon times
    qe_filtered_smeared    : (n_rays_bucket,) float32 — TTS-smeared per-photon times
    flat_indices           : (n_rays_bucket,) int32   — per-photon sensor index
    flat_segment_idx       : (n_rays_bucket,) int32   — per-photon segment index
    """
    rng_key, smear_time_key = jax.random.split(rng_key)
    qe_key, smear_counts_key = jax.random.split(rng_key)

    qe_weights, qe_filtered_times, detected_mask, timing_mask = _qe_roll(
        flat_weights, flat_indices, flat_times,
        qe, qe_corrections, qe_key, threshold)

    total_charge = jax.ops.segment_sum(qe_weights, flat_indices, num_segments=num_detectors)
    detector_mins_true = jax.ops.segment_min(
        qe_filtered_times, flat_indices, num_segments=num_detectors)

    if apply_smearing:
        smeared_flat_times = smear_times(
            flat_times, time_resolution=tts_sigma_ns, key=smear_time_key)
        qe_filtered_smeared = jnp.where(
            detected_mask & timing_mask, smeared_flat_times, jnp.inf)
        detector_mins_reco = jax.ops.segment_min(
            qe_filtered_smeared, flat_indices, num_segments=num_detectors)
    else:
        qe_filtered_smeared = qe_filtered_times
        detector_mins_reco = detector_mins_true

    nonzero_mask = (
        (total_charge > 1e-10)
        & (detector_mins_true > 0)
        & jnp.isfinite(detector_mins_true))

    if apply_smearing:
        measured_charge = jnp.where(
            nonzero_mask,
            smear_charges_SK_like(total_charge, key=smear_counts_key),
            0)
    else:
        measured_charge = jnp.where(nonzero_mask, total_charge, 0)

    measured_time_true = jnp.where(nonzero_mask, detector_mins_true, 0.0)
    measured_time_reco = jnp.where(nonzero_mask, detector_mins_reco, 0.0)

    return (measured_charge, measured_time_true, measured_time_reco,
            qe_weights, qe_filtered_times, qe_filtered_smeared,
            flat_indices, flat_segment_idx)


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
