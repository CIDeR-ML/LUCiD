"""Sensor hit-making (digitization) for the LUCiD data pipeline.

The optical simulation produces, per event, a flat list of detected photo-
electrons — each with a sensor index, an arrival time (PMT-TTS smeared), and a
charge. This module turns that per-photon list into **digits** (recorded hits)
by integrating charge in per-sensor time windows, exactly as real front-end
electronics do. A sensor may yield more than one digit when light arrives in
well-separated time clusters (delayed coincidence, pile-up, dark noise).

All digitizer models flow through the SAME code path and produce the SAME
output shape — a digit list. ``basic`` is not special-cased: it is simply the
preset with an infinite integration window, so every sensor collapses to one
digit (first-arrival time, summed charge), reproducing the historical
one-hit-per-sensor behaviour.

Models. Every electronics/PMT number is taken from the WCSim source (the SK/HK
reference simulation); citations are ``WCSim/...`` paths, verifiable in the
checkout. ``basic`` is the idealized passthrough (kept for backward parity);
``ski`` and ``hk`` are physical models differing in their PMT response:

    basic  window=inf,  deadtime=0, thr=0     — legacy first-arrival + summed charge
    ski    window=200ns, deadtime=0, thr=0.25pe, dark=4.2kHz — SK 20" (WCSim PMT20inch)
    hk     window=200ns, deadtime=0, thr=0.25pe, dark=4.2kHz — HK 20" Box&Line (R12860)

Provenance (all in ``WCSim/``, the checkout alongside LUCiD):
  * window 200 ns / deadtime 0 ns — ``include/WCSimWCDigitizer.hh:97-98`` (the
    only digitizer WCSim ships is SKI; there is no separate QBEE model).
  * charge = per-photoelectron SPE spectrum, sampled and summed. We fit each
    PMT's tabulated SPE CDF (``src/WCSimPMTObject.cc`` ``Getqpe``) to a
    Gaussian+exponential (Bellamy et al. 1994, NIM A 339 468); see ``_SPE_*``.
  * time = charge-dependent PMT jitter — ``src/WCSimPMTObject.cc`` HitTimeSmearing
    (SK ``:88-99`` Gaussian; HK ``:2124-2138`` Gaussian+exp tail) + 0.4 ns TDC
    truncation (``WCSimWCDigitizer.hh:99``).
  * dark rate 4.2 kHz — ``src/WCSimPMTObject.cc:262`` (SK-I; WCSim uses the same
    value for the HK B&L PMT, ``:2295``).

The model is chosen from the detector physics config (a ``"digitizer"`` block);
absent ⇒ ``basic``. Dark noise is off by default; when enabled it is generated
here and tagged by the caller so it lands in the ``hits.h5`` decomposition as
``emission_process = DARK`` (``particle_idx = -1``), keeping ``sensor.h5``
a clean ``(sensor_idx, PE, T)`` list.

This module is pure NumPy (no JAX) — digitization is inherently variable-length
per sensor and runs host-side after the simulation kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


# --- Single-photoelectron (SPE) charge spectra -------------------------------
# Gaussian+exponential (Bellamy 1994) fits to WCSim's tabulated SPE CDF
# (``WCSimPMTObject.cc`` ``Getqpe``): mixture weight ``w`` of the exponential
# low-charge tail (scale ``tau``) vs the Gaussian main peak (``mu``, ``sig``).
# ``fit_mean`` is the raw mean of one SPE draw in p.e.; the sampler divides it
# out so reco charge is unbiased (mean = photoelectron count) while the *shape*
# (resolution) is preserved. KS vs the tabulated LUT quoted per PMT.
_SPE_SK = dict(w=0.2970, mu=1.1288, sig=0.5587, tau=0.1978, fit_mean=0.852)  # KS 5.1%
_SPE_HK = dict(w=0.2026, mu=1.1019, sig=0.3495, tau=0.5857, fit_mean=0.997)  # KS 1.0%

# --- Model presets -----------------------------------------------------------
# window/deadtime in ns; threshold in p.e.; dark_rate_khz per PMT.
#   charge_model: "legacy" (basic → the old sk_like fractional smear, for parity)
#                 or "spe"  (sample+sum the per-pe SPE spectrum in ``spe``).
#   time_model:   "none" | "sk_gauss" | "hk_emg"  (charge-dependent PMT jitter);
#                 tdc_ns is the electronics timing truncation applied after.
MODEL_PRESETS: dict[str, dict] = {
    "basic": dict(integration_window_ns=None, deadtime_ns=0.0, threshold_pe=0.0,
                  charge_model="legacy", charge_res="sk_like",
                  time_model="none", tdc_ns=0.0, dark_rate_khz=0.0, provisional=False),
    "ski":  dict(integration_window_ns=200.0, deadtime_ns=0.0, threshold_pe=0.25,
                 charge_model="spe", spe=_SPE_SK,
                 time_model="sk_gauss", tdc_ns=0.4, dark_rate_khz=4.2, provisional=False),
    "hk":   dict(integration_window_ns=200.0, deadtime_ns=0.0, threshold_pe=0.25,
                 charge_model="spe", spe=_SPE_HK,
                 time_model="hk_emg", tdc_ns=0.4, dark_rate_khz=4.2, provisional=False),
}

# emission_process encoding for the hits.h5 decomposition. 0/1 mirror
# the writer's EMISSION_PROCESS_{CHERENKOV,SCINTILLATION}; DARK tags noise digits.
EMISSION_PROCESS_DARK = 2

# Readout time cap (ns): deposits later than this are dropped before windowing.
# Late nuclear-channel light (n-capture, de-excitation) reaches ms–s; no real
# detector records it. Matches the legacy sensor-writer ``t < 1e5`` convention.
_MAX_DIGIT_TIME_NS = 1e5
# Times are float64 throughout hit-making: a supernova interaction sits at an
# absolute t0 of ~1e9-1e10 ns, where float32 (ULP ~64-1024 ns) collapses distinct
# digit/dark times to identical values — spuriously piling dark hits into a
# single window and firing false triggers. Charge stays float32.
_T_DTYPE = np.float64


@dataclass
class DigitizeResult:
    """Output of :func:`digitize_event` — pure windowing, no readout smearing.

    ``digit_*`` are the recorded hits written to ``sensor.h5`` (a sensor index
    may repeat). ``digit_pe_true`` is the summed detected charge and
    ``digit_time`` the first-arrival time in the window; the caller applies the
    model's **charge resolution** (once) and any **electronics time jitter** —
    kept out of here so ``basic``/``ski`` can reuse the exact existing
    ``smear_charges_SK_like`` (byte-parity), never double-smeared.

    ``photon_digit_idx`` maps each *input* photon to its digit's global index
    (``0..n_digits-1``), or ``-1`` if dropped (sub-threshold digit, or vetoed
    during deadtime) — the caller uses it to build the per-(source, sensor,
    digit) decomposition for ``hits.h5``.
    """
    digit_sensor_idx: np.ndarray  # (D,) int32
    digit_pe_true: np.ndarray     # (D,) float32 — summed detected charge
    digit_time: np.ndarray        # (D,) float32 — first-arrival time in window
    photon_digit_idx: np.ndarray  # (P,) int32 — input photon -> digit, -1 if dropped

    @property
    def n_digits(self) -> int:
        return int(self.digit_sensor_idx.shape[0])


def resolve_model_config(cfg: Union[None, str, dict]) -> dict:
    """Resolve a physics-config ``digitizer`` block to a full parameter dict.

    ``cfg`` may be ``None`` (⇒ ``basic``), a model-name string, or a dict with a
    ``"model"`` key plus any parameter overrides. Unknown keys are kept (so a
    config can override e.g. ``dark_rate_khz``); an unknown model raises.
    """
    if cfg is None:
        name, overrides = "basic", {}
    elif isinstance(cfg, str):
        name, overrides = cfg, {}
    elif isinstance(cfg, dict):
        name = cfg.get("model", "basic")
        overrides = {k: v for k, v in cfg.items() if k != "model"}
    else:
        raise TypeError(f"digitizer config must be None/str/dict, got {type(cfg)}")
    if name not in MODEL_PRESETS:
        raise ValueError(f"unknown digitizer model {name!r}; choices: {sorted(MODEL_PRESETS)}")
    model = dict(MODEL_PRESETS[name])
    model.update(overrides)
    model["model"] = name
    return model


def charge_resolution_sigma(pe_true: np.ndarray, charge_res: Union[str, float]) -> np.ndarray:
    """Per-digit charge-resolution sigma (p.e.) for the legacy ``basic`` path only.

    ``ski``/``hk`` now use the physical per-photoelectron SPE spectrum
    (:func:`_sample_spe_charge`); this remains for ``basic``'s ``"sk_like"``
    smear, matching ``lucid.utils.smear_charges_SK_like`` for byte-parity with
    the historical output. (A float ``f`` gives ``f·sqrt(max(Q,1))``, retained
    for config overrides.)
    """
    q = np.asarray(pe_true, dtype=np.float64)
    if charge_res == "sk_like":
        return np.where(q < 20, q * 0.012, np.where(q < 130, q * 0.0075, q * 0.005))
    return float(charge_res) * np.sqrt(np.maximum(q, 1.0))


def digitize_event(
    sensor_idx: np.ndarray,
    times: np.ndarray,
    charges: np.ndarray,
    n_sensors: int,
    model: dict,
) -> DigitizeResult:
    """Digitize one event's per-photon hits into a list of recorded digits.

    Pure, deterministic windowing — no readout smearing (see
    :class:`DigitizeResult`).

    Parameters
    ----------
    sensor_idx, times, charges : (P,) arrays
        Per detected photon: sensor index, arrival time (ns, detector frame,
        already TTS-smeared), and charge (p.e.). Dark-noise photons, if any, are
        just extra entries the caller appended (and tracks separately for
        labelling) — this function is source-agnostic.
    n_sensors : int
        Kept for API symmetry / validation; not otherwise required.
    model : dict
        A resolved model config (see :func:`resolve_model_config`).

    Notes
    -----
    Algorithm per sensor (mirrors WCSim's SKI integrator, generalized):
    sort hits by time; open an integration window at the first hit and sum
    charge over ``[t0, t0+window]``; emit a digit at ``t0`` if the integrated
    charge clears ``threshold``; then veto hits within the following
    ``deadtime``; the next surviving hit opens the next window. ``window=None``
    ⇒ infinite (one digit per sensor = ``basic``).
    """
    sensor_idx = np.asarray(sensor_idx, dtype=np.int64)
    times = np.asarray(times, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    n_ph = times.shape[0]

    win_raw = model.get("integration_window_ns")
    window = np.inf if win_raw is None else float(win_raw)
    deadtime = float(model.get("deadtime_ns", 0.0))
    threshold = float(model.get("threshold_pe", 0.0))

    photon_digit_idx = np.full(n_ph, -1, dtype=np.int32)
    if n_ph == 0:
        return DigitizeResult(
            digit_sensor_idx=np.empty(0, np.int32), digit_pe_true=np.empty(0, np.float32),
            digit_time=np.empty(0, _T_DTYPE), photon_digit_idx=photon_digit_idx)

    # Sort by (sensor, time); find per-sensor group boundaries. Vectorized.
    order = np.lexsort((times, sensor_idx))
    s_sorted = sensor_idx[order]
    t_sorted = times[order]
    q_sorted = charges[order]
    change = np.empty(n_ph, dtype=bool)
    change[0] = True
    change[1:] = s_sorted[1:] != s_sorted[:-1]
    gstart = np.flatnonzero(change)
    gend = np.empty(gstart.size, dtype=np.int64)
    gend[:-1] = gstart[1:]
    gend[-1] = n_ph
    gsize = gend - gstart
    first_t = t_sorted[gstart]                        # min time (sorted) per group
    span = t_sorted[gend - 1] - first_t
    gcharge = np.add.reduceat(q_sorted, gstart)
    # A group is "simple" if all its light fits in one integration window —
    # then it is exactly one digit (sum charge, first-arrival time). The
    # sequential window loop is only needed for groups that span more than the
    # window (delayed light). `basic` (window=inf) => every group is simple.
    simple = span <= window

    pdi_sorted = np.full(n_ph, -1, dtype=np.int32)     # in sorted order
    # --- simple groups: fully vectorized ---
    simple_pass = simple & (gcharge >= threshold)
    n_simple = int(simple_pass.sum())
    digit_id_per_group = np.full(gstart.size, -1, dtype=np.int64)
    digit_id_per_group[simple_pass] = np.arange(n_simple)
    pdi_sorted = np.repeat(digit_id_per_group, gsize).astype(np.int32)
    d_sensor = [s_sorted[gstart[simple_pass]].astype(np.int32)]
    d_pe = [gcharge[simple_pass].astype(np.float32)]
    d_t = [first_t[simple_pass].astype(_T_DTYPE)]
    digit_counter = n_simple

    # --- complex groups: the sliding-window loop, only where needed ---
    for g in np.flatnonzero(~simple):
        i0, i1 = int(gstart[g]), int(gend[g])
        gt = t_sorted[i0:i1]
        gq = q_sorted[i0:i1]
        m = i1 - i0
        i = 0
        s_val = int(s_sorted[i0])
        while i < m:
            t0 = gt[i]
            upper = t0 + window
            j = i
            pe = 0.0
            while j < m and gt[j] <= upper:
                pe += gq[j]
                j += 1
            if pe >= threshold:
                pdi_sorted[i0 + i:i0 + j] = digit_counter
                d_sensor.append(np.int32(s_val))
                d_pe.append(np.float32(pe))
                d_t.append(_T_DTYPE(t0))
                digit_counter += 1
            dead_end = upper + deadtime
            while j < m and gt[j] <= dead_end:
                j += 1
            i = j

    photon_digit_idx[order] = pdi_sorted
    return DigitizeResult(
        digit_sensor_idx=np.concatenate([np.atleast_1d(a) for a in d_sensor]).astype(np.int32)
            if digit_counter else np.empty(0, np.int32),
        digit_pe_true=np.concatenate([np.atleast_1d(a) for a in d_pe]).astype(np.float32)
            if digit_counter else np.empty(0, np.float32),
        digit_time=np.concatenate([np.atleast_1d(a) for a in d_t]).astype(_T_DTYPE)
            if digit_counter else np.empty(0, _T_DTYPE),
        photon_digit_idx=photon_digit_idx,
    )


def _sample_spe_charge(pe_true: np.ndarray, spe: dict, rng: np.random.Generator) -> np.ndarray:
    """Reco charge = sum of N per-photoelectron SPE draws, N = round(pe_true).

    Each photoelectron draws from the Gaussian+exponential SPE mixture (``spe``).
    The N-fold sum is sampled **exactly** (not a CLT approximation) and in
    O(n_digits): split the N pes into ``k ~ Binomial(N, 1-w)`` Gaussian-core pes
    and ``N-k`` exponential-tail pes, then

        Σ k Gaussians = Normal(k·mu, √k·sig)      Σ (N-k) Exp(tau) = Gamma(N-k, tau).

    The exact per-pe shape is what makes the low-pe response (dark noise = 1 pe,
    single-photon hits) faithful. Normalized by the SPE ``fit_mean`` so the mean
    reco charge equals the photoelectron count.
    """
    q = np.asarray(pe_true, dtype=np.float64)
    n = np.maximum(1, np.rint(q)).astype(np.int64)          # photoelectron count
    w, mu, sig, tau = spe["w"], spe["mu"], spe["sig"], spe["tau"]
    k = rng.binomial(n, 1.0 - w)                            # Gaussian-core pes
    nmk = n - k                                             # exponential-tail pes
    charge = rng.normal(k * mu, np.sqrt(k) * sig)           # Σ Gaussians (0 if k=0)
    tail = nmk > 0
    if tail.any():
        charge[tail] += rng.gamma(nmk[tail], tau)          # Σ Exp(tau) = Gamma
    return np.clip(charge / spe["fit_mean"], 0.0, None)


def _sample_time_jitter(digit_time: np.ndarray, digit_pe: np.ndarray,
                        model: dict, rng: np.random.Generator) -> np.ndarray:
    """Charge-dependent PMT time jitter + electronics TDC truncation.

    ``sk_gauss`` — WCSim ``PMT20inch::HitTimeSmearing`` (``WCSimPMTObject.cc:88-99``):
        Gaussian, sigma = max(0.58, 0.33 + sqrt(10/Q)) ns.
    ``hk_emg``   — WCSim ``BoxandLine20inchHQE::HitTimeSmearing`` (``:2124-2138``):
        exponentially-modified Gaussian, Normal(-0.2, sigma(Q)) plus a positive
        exponential late tail of mean 1/lambda(Q).
    Then truncate to the TDC step ``tdc_ns`` (WCSim SKI timing precision 0.4 ns).
    """
    t = np.asarray(digit_time, dtype=np.float64).copy()
    if t.size == 0:
        return t
    Q = np.maximum(np.asarray(digit_pe, dtype=np.float64), 1e-6)
    tm = model.get("time_model", "none")
    if tm == "sk_gauss":
        sig = np.maximum(0.58, 0.33 + np.sqrt(10.0 / Q))
        t = t + rng.normal(0.0, sig)
    elif tm == "hk_emg":
        s = (0.6314, 0.06260, 0.5711, 23.96)               # WCSim sig_param
        sig_low = s[0] * (np.exp(-s[1] * Q) + s[2])
        hc0 = 2 * s[0] * s[1] * s[3] * np.sqrt(s[3]) * np.exp(-s[1] * s[3])
        hc1 = s[0] * ((1 - 2 * s[1] * s[3]) * np.exp(-s[1] * s[3]) + s[2])
        sig = np.where(Q < s[3], sig_low, hc0 / np.sqrt(Q) + hc1)
        lam = 0.4113 + 0.07827 * Q                         # WCSim lambda_param
        t = t + rng.normal(-0.2, sig) - np.log(1.0 - rng.random(t.shape)) / lam
    tdc = float(model.get("tdc_ns", 0.0))
    if tdc > 0.0:
        t = np.round(t / tdc) * tdc
    return t


def apply_readout_resolution(
    digit_pe_true: np.ndarray,
    digit_time: np.ndarray,
    model: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the model's charge response and charge-dependent time jitter.

    ``charge_model="spe"`` (``ski``/``hk``) samples+sums the per-photoelectron
    SPE spectrum; ``"legacy"`` (``basic``) keeps the historical ``sk_like``
    fractional smear for byte-parity. Time gets the PMT's charge-dependent jitter
    (:func:`_sample_time_jitter`), on top of the TTS already in ``digit_time``.
    Returns ``(pe_reco, t_reco)`` as float32.
    """
    pe_true = np.asarray(digit_pe_true, dtype=np.float64)
    if pe_true.size == 0:
        pe_reco = pe_true
    elif model.get("charge_model", "legacy") == "spe":
        pe_reco = _sample_spe_charge(pe_true, model["spe"], rng)
    else:
        sigma = charge_resolution_sigma(pe_true, model.get("charge_res", "sk_like"))
        pe_reco = np.clip(pe_true + rng.normal(size=pe_true.shape) * sigma, 0.0, None)
    t = _sample_time_jitter(digit_time, pe_true, model, rng)
    return pe_reco.astype(np.float32), t.astype(_T_DTYPE)


def _encode_keys(key_cols):
    """Pack integer key columns into one int64 composite (mixed radix).

    A single ``argsort`` on the composite is much faster than an N-array
    ``lexsort`` (one pass vs one per key). Returns ``(composite, ok)``; ``ok``
    is False if the radix product would overflow int64 (caller falls back to
    lexsort). ``key_cols[0]`` is the most-significant key.
    """
    shifted, radices = [], []
    for c in key_cols:
        c = np.asarray(c, dtype=np.int64)
        mn = int(c.min()) if c.size else 0
        s = c - mn                                   # -> non-negative (handles -1)
        shifted.append(s)
        radices.append((int(s.max()) + 1) if s.size else 1)
    prod = 1
    for r in radices:
        prod *= r
        if prod >= (1 << 62):
            return None, False
    key = shifted[0].copy()
    for s, r in zip(shifted[1:], radices[1:]):
        key = key * r + s
    return key, True


def _group_reduce(key_cols: tuple, pe: np.ndarray, t: np.ndarray, t_reco: np.ndarray):
    """Group rows by the tuple of integer key columns; PE=sum, T/T_reco=min.

    Returns ``(key_cols_out, PE, T, T_reco)`` where ``key_cols_out`` are the
    per-group key values (same tuple arity as ``key_cols``). ``key_cols[0]`` is
    the primary sort key. Sorts on a single int64 composite key (falls back to
    lexsort only if the composite would overflow).
    """
    n = pe.shape[0]
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return (tuple(empty for _ in key_cols),
                np.empty(0, np.float64), np.empty(0, np.float64), np.empty(0, np.float64))
    composite, ok = _encode_keys(key_cols)
    if ok:
        order = np.argsort(composite, kind="stable")
        comp_s = composite[order]
        change = np.empty(n, dtype=bool)
        change[0] = True
        change[1:] = comp_s[1:] != comp_s[:-1]
    else:
        order = np.lexsort(tuple(reversed(key_cols)))
        cols_tmp = [np.asarray(c)[order] for c in key_cols]
        change = np.zeros(n, dtype=bool)
        change[0] = True
        for c in cols_tmp:
            change[1:] |= c[1:] != c[:-1]
    cols_s = [np.asarray(c)[order] for c in key_cols]
    pe_s, t_s, tr_s = pe[order], t[order], t_reco[order]
    starts = np.flatnonzero(change)
    return ([c[starts] for c in cols_s],
            np.add.reduceat(pe_s, starts),
            np.minimum.reduceat(t_s, starts),
            np.minimum.reduceat(tr_s, starts))


def digitize_and_decompose(
    *,
    sensor_idx, charge, t_true, t_reco, particle_idx, segment_idx, emission_process,
    n_sensors: int, model: dict, rng: np.random.Generator,
    dark_rate_khz: float = 0.0, readout_pad_ns: float = 100.0,
    apply_resolution: bool = True,
):
    """Full host-side hit-making: per-photon deposits -> digits + decomposition.

    Windows the per-deposit ``(sensor, t_reco, charge)`` list into digits, adds
    optional dark noise (tagged ``emission_process=DARK``, ``particle_idx=-1``,
    ``segment_idx=-1``), applies the model's readout resolution, and aggregates
    the per-deposit charge into the ``digit_idx``-aware decompositions.

    All input arrays are per detected deposit (QE already applied upstream),
    in the G4/vertex time frame — the caller applies the per-interaction t0
    shift to the returned times.

    Returns ``(sensor_digits, hits_sparse, seg_hits)``:
      * ``sensor_digits``: ``{sensor_idx, PE, T}`` — the recorded digits (a
        sensor index may repeat). ``sensor.h5``.
      * ``hits_sparse``: ``{particle_idx, sensor_idx, digit_idx, PE, T, T_reco,
        emission_process}`` — per-(source, sensor, digit) rows; dark rows carry
        ``particle_idx=-1`` / ``emission_process=DARK``. ``hits.h5``.
      * ``seg_hits``: ``{segment_idx, sensor_idx, digit_idx, PE, T, T_reco,
        emission_process}`` — real deposits only (dark has no segment).
    """
    sensor_idx = np.asarray(sensor_idx, dtype=np.int64)
    charge = np.asarray(charge, dtype=np.float64)
    t_true = np.asarray(t_true, dtype=np.float64)
    t_reco = np.asarray(t_reco, dtype=np.float64)
    particle_idx = np.asarray(particle_idx, dtype=np.int64)
    segment_idx = np.asarray(segment_idx, dtype=np.int64)
    emission_process = np.asarray(emission_process, dtype=np.int64)

    # Drop non-finite/undetected deposits and cap the readout time at
    # _MAX_DIGIT_TIME_NS *relative to the earliest deposit*. Late nuclear-channel
    # photons (neutron capture, nuclear de-excitation) legitimately arrive at
    # ms–s scales after their interaction, far outside any real readout window;
    # capping the late tail keeps digits, the decomposition, and the dark window
    # bounded. The cap must be relative: an absolute cap would drop an entire
    # supernova interaction sitting at t0 ~ ms–s. No lower bound.
    valid = np.isfinite(t_reco) & np.isfinite(t_true) & (charge > 0)
    t_ref = np.min(t_reco[valid]) if valid.any() else 0.0
    finite = valid & ((t_reco - t_ref) < _MAX_DIGIT_TIME_NS)
    if not finite.all():
        sensor_idx, charge, t_true, t_reco, particle_idx, segment_idx, emission_process = (
            a[finite] for a in (sensor_idx, charge, t_true, t_reco,
                                particle_idx, segment_idx, emission_process))

    # Dark noise, over the event's readout span (± pad), tagged as DARK.
    if dark_rate_khz > 0.0 and t_reco.size:
        t_lo = float(t_reco.min()) - readout_pad_ns
        t_hi = float(t_reco.max()) + readout_pad_ns
        d_s, d_t, d_q = generate_dark_noise(n_sensors, dark_rate_khz, t_lo, t_hi, rng)
        if d_s.size:
            sensor_idx = np.concatenate([sensor_idx, d_s])
            charge = np.concatenate([charge, d_q])
            t_true = np.concatenate([t_true, d_t])
            t_reco = np.concatenate([t_reco, d_t])
            particle_idx = np.concatenate([particle_idx, np.full(d_s.size, -1, np.int64)])
            segment_idx = np.concatenate([segment_idx, np.full(d_s.size, -1, np.int64)])
            emission_process = np.concatenate(
                [emission_process, np.full(d_s.size, EMISSION_PROCESS_DARK, np.int64)])

    res = digitize_event(sensor_idx, t_reco, charge, n_sensors, model)
    if apply_resolution:
        pe_reco, t_reco_digit = apply_readout_resolution(
            res.digit_pe_true, res.digit_time, model, rng)
    else:
        # No readout smearing (mirrors the legacy apply_smearing=False path):
        # digits carry true integrated charge and first-arrival time.
        pe_reco = res.digit_pe_true.astype(np.float32)
        t_reco_digit = res.digit_time.astype(_T_DTYPE)
    sensor_digits = {
        "sensor_idx": res.digit_sensor_idx.astype(np.uint16),
        "PE": pe_reco,
        "T": t_reco_digit,
    }

    didx = res.photon_digit_idx
    is_dark = emission_process == EMISSION_PROCESS_DARK

    # hits.h5: keep digit-assigned deposits with a real particle OR dark.
    hmask = (didx >= 0) & ((particle_idx >= 0) | is_dark)
    hk, hPE, hT, hTR = _group_reduce(
        (particle_idx[hmask], sensor_idx[hmask], didx[hmask], emission_process[hmask]),
        charge[hmask], t_true[hmask], t_reco[hmask])
    hits_sparse = {
        "particle_idx": hk[0].astype(np.int32), "sensor_idx": hk[1].astype(np.uint16),
        "digit_idx": hk[2].astype(np.int32), "emission_process": hk[3].astype(np.int8),
        "PE": hPE.astype(np.float32), "T": hT.astype(_T_DTYPE),
        "T_reco": hTR.astype(_T_DTYPE),
    }

    # step/sensor_hits: real deposits only (dark has no segment).
    smask = (didx >= 0) & (segment_idx >= 0)
    sk, sPE, sT, sTR = _group_reduce(
        (segment_idx[smask], sensor_idx[smask], didx[smask], emission_process[smask]),
        charge[smask], t_true[smask], t_reco[smask])
    seg_hits = {
        "segment_idx": sk[0].astype(np.int32), "sensor_idx": sk[1].astype(np.uint16),
        "digit_idx": sk[2].astype(np.int32), "emission_process": sk[3].astype(np.int8),
        "PE": sPE.astype(np.float32), "T": sT.astype(_T_DTYPE),
        "T_reco": sTR.astype(_T_DTYPE),
    }
    return sensor_digits, hits_sparse, seg_hits


def generate_dark_noise(
    n_sensors: int,
    rate_khz: float,
    t_start_ns: float,
    t_end_ns: float,
    rng: np.random.Generator,
    charge_pe: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate dark-noise photo-electrons over a readout window.

    Poisson(``rate·Δt``) hits per sensor, uniform in ``[t_start, t_end]``, each
    ``charge_pe`` p.e. Returns ``(sensor_idx, times, charges)`` for the caller to
    concatenate with the real per-photon list before :func:`digitize_event`,
    tracking which rows are dark so they can be tagged ``emission_process=DARK``
    in the decomposition. Returns empty arrays when ``rate_khz <= 0``.
    """
    if rate_khz <= 0.0 or t_end_ns <= t_start_ns:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return empty_i, empty_f, empty_f
    mu = rate_khz * (t_end_ns - t_start_ns) * 1e-6  # kHz·ns → count
    counts = rng.poisson(mu, size=n_sensors)
    total = int(counts.sum())
    sensor_idx = np.repeat(np.arange(n_sensors, dtype=np.int64), counts)
    times = rng.uniform(t_start_ns, t_end_ns, size=total)
    charges = np.full(total, float(charge_pe), dtype=np.float64)
    return sensor_idx, times, charges
