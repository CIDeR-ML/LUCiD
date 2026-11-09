"""What the sensors report: one hits function per (hits, detection) pair, in the registry ``HITS``.

The photon loop ends in a flat list of deposits: deposit ``i`` puts weight ``flat_weights[i]`` on
sensor ``flat_indices[i]`` at time ``flat_times[i]`` (ns). A hits function turns that list into
what the sensors report. Every one has the same signature::

    fn(flat_weights, flat_indices, flat_times, num_sensors, key, qe, qe_corrections, response,
       flat_segment_idx, flat_indirect, *, <setup options>)

``key`` is the PRNG key for its own draws, ``qe`` the per-deposit QE, ``qe_corrections`` the
per-sensor QE corrections, ``response`` the call's ``DetectorParams.response`` (its ``tts``),
``flat_segment_idx`` the per-deposit segment index (None unless the photons carry one) and
``flat_indirect`` whether the photon had scattered or reflected before the deposit. Each reads
what it needs and returns one of the NamedTuples below. ``detection`` says how QE, the TTS and the
charge resolution enter: ``'expected'`` as expectations (smooth, differentiable), ``'sampled'`` as
draws. The keyword-only options are set at setup from ``hits_config``; the simulator's own
``n_photons`` and ``charge_resolution`` reach the functions that take them (:func:`bind_hits`).

A new kind of hits is one function with this signature and one ``HITS`` entry.
"""
import inspect
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from lucid.utils import smear_charges_SK_like, smear_charges_spe


class FirstArrival(NamedTuple):
    """Per sensor: the charge (PE) and the first-arrival time (ns); 0 on unlit sensors."""
    charge: jnp.ndarray
    time: jnp.ndarray


class PhotonList(NamedTuple):
    """Per deposit: the log QE-weight (-1e10 when invalid), the time (ns, 0 when invalid) and the
    sensor; then the per-sensor charge."""
    log_w: jnp.ndarray
    times: jnp.ndarray
    indices: jnp.ndarray
    charge: jnp.ndarray


class SegmentTruth(NamedTuple):
    """Per sensor: the charge and the first arrival without and with the TTS. Per deposit, for the
    host-side per-(segment, sensor) groupby: the weight (0 when QE failed), the time without and
    with the TTS (inf when QE failed), the sensor and the segment index."""
    charge: jnp.ndarray
    time_true: jnp.ndarray
    time_reco: jnp.ndarray
    deposit_weight: jnp.ndarray
    deposit_time_true: jnp.ndarray
    deposit_time_reco: jnp.ndarray
    deposit_sensor: jnp.ndarray
    deposit_segment: jnp.ndarray


class Waveform(NamedTuple):
    """The ``(num_sensors, n_time_bins)`` waveform, the detections outside the window, and the
    detections (a count when sampled, the total charge when expected)."""
    waveform: jnp.ndarray
    n_dropped: jnp.ndarray
    n_detected: jnp.ndarray


class PhotonHits(NamedTuple):
    """Per emitted photon: whether it was detected, its sensor (-1 if not), its hit time (ns, 0 if
    not) and whether it had scattered or reflected before."""
    detected: jnp.ndarray
    sensor_id: jnp.ndarray
    hit_time: jnp.ndarray
    indirect: jnp.ndarray



def _smear_charge(total_charge, charge_resolution, key):
    """Apply the configured per-sensor charge-resolution model.

    ``charge_resolution`` (config-driven; no boolean flags):
      * ``None``          — no smearing (raw Poisson photoelectron counts).
      * ``"Abe_2013"``    — SK charge-dependent fractional Gaussian (Abe 2013, Table 2).
      * ``"Bellamy_94"``  — per-photoelectron SPE spectrum (WCSim SK PMT; Bellamy 1994).
    """
    if charge_resolution is None:
        return total_charge
    if charge_resolution == "Abe_2013":
        return smear_charges_SK_like(total_charge, key=key)
    if charge_resolution == "Bellamy_94":
        return smear_charges_spe(total_charge, key=key)
    raise ValueError(
        "charge_resolution must be None|'Abe_2013'|'Bellamy_94', "
        f"got {charge_resolution!r}")

# ===================================================================
# First arrival, photon list, segment truth
# ===================================================================

def _exclusive_cumsum_in_runs(x, run_id):
    """Cumulative sum of ``x`` over the earlier entries of the same run, for ``run_id`` sorted so
    that each run is contiguous. A segmented scan, so no run's sum depends on another's."""
    starts = jnp.concatenate([jnp.array([True]), run_id[1:] != run_id[:-1]])

    def add(a, b):
        return jnp.where(b[1], b[0], a[0] + b[0]), a[1] | b[1]

    inclusive, _ = jax.lax.associative_scan(add, (x, starts))
    return inclusive - x


def _mean_first_arrival_hard(w, t, idx, num_sensors, norm):
    """E[first arrival | at least one] when deposit i is Poisson(w_i) photo-electrons at t_i.

    Sorted by time on each sensor, the first photo-electron comes from deposit k with probability
    exp(-R_k) (1 - exp(-w_k)), R_k the weight arriving before it; these sum to 1 - exp(-mu)."""
    order = jnp.lexsort((t, idx))
    w_s, t_s, i_s = w[order], t[order], idx[order]
    p_first = jnp.exp(-_exclusive_cumsum_in_runs(w_s, i_s)) * -jnp.expm1(-w_s)
    return jax.ops.segment_sum(p_first * t_s, i_s, num_segments=num_sensors) / norm


def _mean_first_arrival_smeared(w, t, idx, num_sensors, mu, norm, t_lo, t_hi, sigma,
                                n_nodes):
    """The same mean when each photo-electron's time is smeared by N(0, sigma^2).

    The expected count before t is R(t) = sum_i w_i Phi((t - t_i) / sigma), as in
    :func:`lucid.losses.first_arrival_window_nll`, and P(first > t | at least one) =
    (exp(-R(t)) - exp(-mu)) / (1 - exp(-mu)). Its integral over [t_lo, t_hi] (6 sigma beyond the
    sensor's earliest and latest deposit, where it is 1 and 0) plus t_lo is the mean. The nodes
    are t_lo + (t_hi - t_lo) x^2 on a uniform x, dense where the first arrival falls."""
    x = jnp.linspace(0.0, 1.0, n_nodes)
    span = t_hi - t_lo

    def survival(xg):
        tg = t_lo + span * xg * xg
        R = jax.ops.segment_sum(w * 0.5 * (1.0 + jax.scipy.special.erf(
            (tg[idx] - t) / (sigma * jnp.sqrt(2.0)))), idx, num_segments=num_sensors)
        R = jnp.minimum(R, mu)
        return jnp.exp(-R) * -jnp.expm1(-(mu - R)) / norm

    h = jax.lax.map(survival, x)                                 # (n_nodes, num_sensors)
    dt_dx = 2.0 * span[None, :] * x[:, None]
    f = h * dt_dx
    integral = jnp.sum(0.5 * (f[1:] + f[:-1]), axis=0) * (x[1] - x[0])
    return t_lo + integral


def first_arrival_expected(
        flat_weights, flat_indices, flat_times, num_sensors, key, qe, qe_corrections, response,
        flat_segment_idx, flat_indirect, *,
        threshold=1e-10, n_nodes=128):
    """Expected per-sensor charge and the mean of the predicted first-arrival distribution.

    Each deposit is Poisson-distributed photo-electrons (mean ``w_i`` = weight x QE) at its
    arrival time, each smeared by the transit-time spread ``tts`` (ns): the photon model of the
    reconstruction's first-arrival likelihood (:func:`lucid.losses.first_arrival_window_nll`).
    The time is E[first arrival | at least one photo-electron], a smooth function of the weights
    and times; at low occupancy it tends to the weight-averaged arrival, at high occupancy to the
    earliest arrival less the spread's order-statistic shift. ``tts = 0`` evaluates it exactly
    over the sorted deposits; ``tts > 0`` integrates the survival function on ``n_nodes`` nodes.
    """
    tts = response.tts
    per_photon_qe = qe * qe_corrections[flat_indices]
    qe_weights = flat_weights * per_photon_qe

    valid_mask = (qe_weights > threshold) & (flat_times > 0) & jnp.isfinite(flat_times)
    w = jnp.where(valid_mask, qe_weights, 0.0)
    t = jnp.where(valid_mask, flat_times, 0.0)
    t_min = jax.ops.segment_min(jnp.where(valid_mask, flat_times, jnp.inf), flat_indices,
                                num_segments=num_sensors)
    has_photons = jnp.isfinite(t_min)

    total_charge = jax.ops.segment_sum(qe_weights, flat_indices, num_segments=num_sensors)
    mu = jax.ops.segment_sum(w, flat_indices, num_segments=num_sensors)
    norm = jnp.where(has_photons, -jnp.expm1(-mu), 1.0)

    def hard(_):
        return _mean_first_arrival_hard(w, t, flat_indices, num_sensors, norm)

    def smeared(sigma):
        t_max = jax.ops.segment_max(jnp.where(valid_mask, flat_times, -jnp.inf), flat_indices,
                                    num_segments=num_sensors)
        t_lo = jax.lax.stop_gradient(jnp.where(has_photons, t_min - 6.0 * sigma, 0.0))
        t_hi = jax.lax.stop_gradient(jnp.where(has_photons, t_max + 6.0 * sigma, 1.0))
        return _mean_first_arrival_smeared(w, t, flat_indices, num_sensors, mu, norm,
                                           t_lo, t_hi, sigma, n_nodes)

    tts = jnp.asarray(tts, flat_times.dtype)
    t_first = jax.lax.cond(tts > 0, smeared, hard, jnp.where(tts > 0, tts, 1.0))

    nonzero_mask = (total_charge > threshold) & has_photons
    measured_charge = jnp.where(nonzero_mask, total_charge, 0.0)
    measured_time = jnp.where(nonzero_mask, t_first, 0.0)

    return FirstArrival(measured_charge, measured_time)


def first_arrival_sampled(
        flat_weights, flat_indices, flat_times, num_sensors, key, qe, qe_corrections, response,
        flat_segment_idx, flat_indirect, *,
        charge_resolution=None, threshold=1e-5):
    """Sampled per-sensor hits: Bernoulli QE, segment_min timing, configurable charge resolution.

    ``response.tts`` (ns) is the per-photon transit-time-spread sigma applied to each photon's
    time BEFORE the first-arrival segment_min (so the min carries the correct early
    bias). ``tts=0`` ⇒ no smear (byte-identical). Timing resolution comes solely from this
    per-photon TTS.

    ``charge_resolution`` selects the per-sensor charge-resolution model:
      * ``None``          — raw Poisson photoelectron counts (no charge smearing).
      * ``"Abe_2013"``    — SK charge-dependent fractional Gaussian (Abe 2013, Table 2).
      * ``"Bellamy_94"``  — per-photoelectron SPE spectrum (WCSim SK PMT; Bellamy 1994).
    """
    timing_mask = (flat_weights > threshold) & (flat_times > 0)

    rng_key, smear_time_key = jax.random.split(key)
    qe_key, smear_counts_key = jax.random.split(rng_key)

    # Per-photon QE including per-sensor corrections (consistent with the expected hits)
    per_photon_qe = qe * qe_corrections[flat_indices]

    # Bernoulli QE sampling — when qe >= 1.0, uniform(0,1) < qe
    # is always true so all photons pass.  Avoids Python `if` on traced values.
    detection_probs = jax.random.uniform(qe_key, shape=flat_weights.shape)
    detected_mask = detection_probs < per_photon_qe
    qe_weights = flat_weights * detected_mask.astype(jnp.float32)
    # PER-PHOTON TTS: smear each detected photon's time BEFORE the first-arrival min.
    # Driven by the dp.response.tts field. Applied unconditionally scaled by tts (0 ⇒ no
    # shift, byte-identical) so the key stream is stable and tts stays differentiable.
    eff_tts = jnp.asarray(response.tts)
    photon_times = flat_times + jax.random.normal(smear_time_key, shape=flat_times.shape) * eff_tts
    qe_filtered_times = jnp.where(detected_mask & timing_mask, photon_times, jnp.inf)

    total_charge = jax.ops.segment_sum(qe_weights, flat_indices, num_segments=num_sensors)
    detector_mins = jax.ops.segment_min(qe_filtered_times, flat_indices, num_segments=num_sensors)

    # CHARGE is gated on a valid first-arrival time (detector_mins>0 & finite).
    nonzero_mask = (total_charge > 1e-10) & (detector_mins > 0) & jnp.isfinite(detector_mins)

    # TIME: first-arrival of the per-photon-TTS-smeared photons (no post-hoc smear).
    # PER-SENSOR gate: only lit sensors carry a time; empty sensors are exactly 0
    # (their segment_min is inf, which must not reach the output).
    measured_time = jnp.where(nonzero_mask, detector_mins, 0.0)

    # CHARGE: config-driven resolution model (None ⇒ raw Poisson counts).
    smeared_charge = _smear_charge(total_charge, charge_resolution, smear_counts_key)
    measured_charge = jnp.where(nonzero_mask, smeared_charge, 0)

    return FirstArrival(measured_charge, measured_time)


def segment_truth_sampled(
        flat_weights, flat_indices, flat_times, num_sensors, key, qe, qe_corrections, response,
        flat_segment_idx, flat_indirect, *,
        charge_resolution=None, threshold=1e-5):
    """Per-sensor totals PLUS pass-through per-deposit arrays for host aggregation.

    The production 'hits' file needs a per-(segment, sensor) PE decomposition,
    which is done on the host in NumPy. This returns the per-sensor measured
    charge/time (identical QE + TTS draws to :func:`first_arrival_sampled`) AND the
    surviving per-deposit arrays (QE-weight, true + TTS-smeared times, sensor index,
    segment index) so the caller can group them however it likes.

    ``response.tts`` (ns) is the per-photon transit-time-spread sigma; ``tts=0`` ⇒ no smear.
    Mirrors first_arrival_sampled's RNG/threshold semantics so the per-sensor outputs match it.
    """
    rng_key, smear_time_key = jax.random.split(key)
    qe_key, smear_counts_key = jax.random.split(rng_key)

    timing_mask = (flat_weights > threshold) & (flat_times > 0)
    per_photon_qe = qe * qe_corrections[flat_indices]
    detection_probs = jax.random.uniform(qe_key, shape=flat_weights.shape)
    detected_mask = detection_probs < per_photon_qe
    qe_weights = flat_weights * detected_mask.astype(jnp.float32)

    # First-arrival WITHOUT TTS (true) and WITH per-photon TTS (reco).
    qe_filtered_times = jnp.where(detected_mask & timing_mask, flat_times, jnp.inf)
    eff_tts = jnp.asarray(response.tts)
    smeared_times = flat_times + jax.random.normal(smear_time_key, shape=flat_times.shape) * eff_tts
    qe_filtered_smeared = jnp.where(detected_mask & timing_mask, smeared_times, jnp.inf)

    total_charge = jax.ops.segment_sum(qe_weights, flat_indices, num_segments=num_sensors)
    detector_mins_true = jax.ops.segment_min(qe_filtered_times, flat_indices, num_segments=num_sensors)
    detector_mins_reco = jax.ops.segment_min(qe_filtered_smeared, flat_indices, num_segments=num_sensors)

    nonzero_mask = ((total_charge > 1e-10) & (detector_mins_true > 0)
                    & jnp.isfinite(detector_mins_true))
    smeared_charge = _smear_charge(total_charge, charge_resolution, smear_counts_key)
    measured_charge = jnp.where(nonzero_mask, smeared_charge, 0.0)
    measured_time_true = jnp.where(nonzero_mask, detector_mins_true, 0.0)
    measured_time_reco = jnp.where(nonzero_mask, detector_mins_reco, 0.0)

    return SegmentTruth(measured_charge, measured_time_true, measured_time_reco,
                        qe_weights, qe_filtered_times, qe_filtered_smeared,
                        flat_indices, flat_segment_idx)


def photon_list_expected(
        flat_weights, flat_indices, flat_times, num_sensors, key, qe, qe_corrections, response,
        flat_segment_idx, flat_indirect, *,
        threshold=1e-10):
    """Per-deposit log-weights and times, and the per-sensor total charge, for the likelihood.

    Instead of aggregating times to per-sensor first-arrival values, this
    returns the raw per-photon arrays so that ``first_arrival_window_nll`` (or
    similar likelihood-based losses) can operate on them directly.

    ``threshold`` is the minimum QE-weight of a valid deposit.
    """
    per_photon_qe = qe * qe_corrections[flat_indices]
    qe_weights = flat_weights * per_photon_qe

    valid_mask = (qe_weights > threshold) & (flat_times > 0) & jnp.isfinite(flat_times)
    safe_weights = jnp.where(valid_mask, qe_weights, 0.0)
    safe_times = jnp.where(valid_mask, flat_times, 0.0)
    log_w = jnp.where(valid_mask, jnp.log(safe_weights + 1e-30), -1e10)

    total_charge = jax.ops.segment_sum(safe_weights, flat_indices, num_segments=num_sensors)

    return PhotonList(log_w, safe_times, flat_indices, total_charge)


# ===================================================================
# Waveform and photon hits
# ===================================================================

def _resolve_first_detection(
        flat_weights, flat_indices, flat_times, n_photons,
        per_photon_qe, qe_key, threshold, flat_indirect=None):
    """Compact flat propagation arrays to per-photon first-detection records.

    Returns arrays of length ``n_photons``:
    - detected : bool    — did this photon pass QE Bernoulli at any iteration?
    - sensor_id : int32  — sensor hit (first-detection), or -1 if not detected
    - hit_time : float32 — propagation time at first detection (0 if not detected)
    - indirect : bool    — had it scattered or reflected before that detection?
      All-False when ``flat_indirect`` is not supplied.
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
    if flat_indirect is None:
        indirect = jnp.zeros_like(detected)
    else:
        indirect = detected & flat_indirect[first_flat_idx]
    return detected, sensor_id, hit_time, indirect


def waveform_sampled(
        flat_weights, flat_indices, flat_times, num_sensors, key, qe, qe_corrections, response,
        flat_segment_idx, flat_indirect, *,
        n_photons, window_ns=500.0, bin_width_ns=1.0, tts_sigma_ns=1.0, t_min_ns=0.0,
        smear_time=True, smear_charge=True, threshold=1e-10):
    """The per-sensor waveform of the detected photons: Bernoulli QE, TTS and gain smearing.

    Pipeline: flat deposits → per-photon first-detection (n_photons) → TTS + gain smearing on
    n_photons entries → bin to (num_sensors, n_time_bins). This keeps the segment_sum input small
    even when K×max_sensors×n_photons is large. Must be used with the sampled transport so the
    weights are binary.

    Parameters
    ----------
    n_photons : int
        Number of photons per case (the simulator's; must match n_rays).
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
    """
    n_time_bins = int(round(window_ns / bin_width_ns))
    per_photon_qe = qe * qe_corrections[flat_indices]
    qe_key, tts_key, gain_key = jax.random.split(key, 3)

    detected, sensor_id, hit_time, _ = _resolve_first_detection(
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
        safe_charge, flat_bin_idx, num_segments=num_sensors * n_time_bins)
    waveform = waveform_flat.reshape(num_sensors, n_time_bins)

    n_dropped = jnp.sum(dropped.astype(jnp.int32))
    n_detected = jnp.sum(detected.astype(jnp.int32))

    return Waveform(waveform, n_dropped, n_detected)


def waveform_expected(
        flat_weights, flat_indices, flat_times, num_sensors, key, qe, qe_corrections, response,
        flat_segment_idx, flat_indirect, *,
        window_ns=500.0, bin_width_ns=1.0, tts_sigma_ns=1.0, t_min_ns=0.0, smear_time=True,
        threshold=1e-10):
    """The continuous QE-weighted waveform (no Bernoulli, no gain smearing).

    Every deposit contributes its continuous ``flat_weight · QE`` to the ``(sensor, time_bin)``
    cell it lands in. No Bernoulli coin is flipped; the output is the expected waveform given the
    photon trajectories, in :func:`waveform_sampled`'s ``(num_sensors, n_time_bins)`` layout.
    ``n_dropped`` counts valid deposits that landed outside the readout window; ``n_detected`` is
    the total (continuous) integrated charge across all sensors.

    Parameters
    ----------
    window_ns, bin_width_ns, t_min_ns : float
        Readout window / bin grid (as :func:`waveform_sampled`).
    tts_sigma_ns : float
        Per-deposit Gaussian TTS σ. Each deposit (photon × scattering iteration × cell-sensor)
        gets its own draw — slightly over-smooths relative to a per-detection draw but the
        effect is tiny at σ ≲ bin_width.
    smear_time : bool
        Toggle TTS smearing.
    threshold : float
        Minimum per-slot weight to be counted as physical.
    """
    n_time_bins = int(round(window_ns / bin_width_ns))
    per_slot_qe = qe * qe_corrections[flat_indices]
    slot_charge = flat_weights * per_slot_qe

    if smear_time:
        noise = jax.random.normal(key, shape=flat_times.shape) * tts_sigma_ns
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
        safe_charge, flat_bin_idx, num_segments=num_sensors * n_time_bins)
    waveform = waveform_flat.reshape(num_sensors, n_time_bins)

    n_dropped = jnp.sum(dropped.astype(jnp.int32))
    n_detected = jnp.sum(waveform)

    return Waveform(waveform, n_dropped, n_detected)


def photon_hits_sampled(
        flat_weights, flat_indices, flat_times, num_sensors, key, qe, qe_corrections, response,
        flat_segment_idx, flat_indirect, *,
        n_photons, tts_sigma_ns=1.0, smear_time=True, threshold=1e-10):
    """Per emitted photon: its first detection, if any, under Bernoulli QE.

    ``indirect`` marks light that scattered or reflected before detection -- the direct/indirect
    split fiTQun's scattering table is built on. Must be used with the sampled transport so the
    weights are binary.

    Parameters
    ----------
    n_photons : int
        Number of photons per case (the simulator's; must match n_rays).
    tts_sigma_ns : float
        Gaussian σ of per-photon TTS; default 1.0 ns.
    smear_time : bool
        If True, apply Gaussian TTS smearing to hit times.
    threshold : float
        Minimum per-slot weight to consider a candidate.
    """
    per_photon_qe = qe * qe_corrections[flat_indices]
    qe_key, tts_key = jax.random.split(key)

    detected, sensor_id, hit_time, indirect = _resolve_first_detection(
        flat_weights, flat_indices, flat_times, n_photons,
        per_photon_qe, qe_key, threshold, flat_indirect)

    if smear_time:
        noise = jax.random.normal(tts_key, shape=hit_time.shape) * tts_sigma_ns
        hit_time = jnp.where(detected, hit_time + noise, hit_time)

    return PhotonHits(detected, sensor_id, hit_time, indirect)


# ===================================================================
# The registry
# ===================================================================

HITS = {
    ('first_arrival', 'expected'): first_arrival_expected,
    ('first_arrival', 'sampled'): first_arrival_sampled,
    ('photon_list', 'expected'): photon_list_expected,
    ('segment_truth', 'sampled'): segment_truth_sampled,
    ('waveform', 'expected'): waveform_expected,
    ('waveform', 'sampled'): waveform_sampled,
    ('photon_hits', 'sampled'): photon_hits_sampled,
}

# The options the simulator supplies itself, not hits_config.
_SIMULATOR_OPTIONS = ('n_photons', 'charge_resolution')


def bind_hits(hits, detection, hits_config, n_photons, charge_resolution):
    """The ``HITS`` entry for ``(hits, detection)`` with its setup-time options bound.

    ``hits_config`` sets the entry's keyword options; ``n_photons`` and ``charge_resolution`` reach
    the entries that take them.
    """
    if (hits, detection) not in HITS:
        raise ValueError(
            f"hits={hits!r} with detection={detection!r} is not supported; the supported "
            f"(hits, detection) pairs are {sorted(HITS)}")
    fn = HITS[(hits, detection)]
    options = [name for name, p in inspect.signature(fn).parameters.items()
               if p.kind is p.KEYWORD_ONLY]
    accepted = [name for name in options if name not in _SIMULATOR_OPTIONS]
    hits_config = hits_config or {}
    unknown = sorted(set(hits_config) - set(accepted))
    if unknown:
        raise ValueError(
            f"hits_config keys {unknown} are not options of hits={hits!r} with "
            f"detection={detection!r}; it accepts {accepted}")
    setup = dict(n_photons=n_photons, charge_resolution=charge_resolution)
    return partial(fn, **{k: v for k, v in setup.items() if k in options}, **hits_config)
