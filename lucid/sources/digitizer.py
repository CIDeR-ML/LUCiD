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

Models (parameters justified in the design notes / commit message):

    basic   window=inf,   deadtime=0,    thr=0      — legacy first-arrival+sum
    ski     window=200ns, deadtime=0,    thr~0.2pe  — SK-I (WCSim SKI model)
    qbee    window=400ns, deadtime~900ns,thr=0.25pe — SK-IV/V QBEE/QTC
    hk      window=200ns*,deadtime~0,    thr=0.25pe — Hyper-K (dead-time free)
    (* hk window/threshold are provisional; dead-time-free is the firm trait.)

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


# --- Model presets -----------------------------------------------------------
# window/deadtime/time in ns; threshold in p.e.; charge_res is either the string
# "sk_like" (the SK charge-dependent fractional resolution) or a float f giving
# a single-p.e. sigma that scales as f·sqrt(Q). dark_rate_khz default 0 (off);
# set it (e.g. 4.2) to enable dark noise. `provisional` flags not-yet-final
# parameter sets (HK).
MODEL_PRESETS: dict[str, dict] = {
    "basic": dict(integration_window_ns=None, deadtime_ns=0.0, threshold_pe=0.0,
                  charge_res="sk_like", time_res_ns=0.0, dark_rate_khz=0.0,
                  provisional=False),
    "ski":  dict(integration_window_ns=200.0, deadtime_ns=0.0, threshold_pe=0.2,
                 charge_res="sk_like", time_res_ns=0.0, dark_rate_khz=0.0,
                 provisional=False),
    "qbee": dict(integration_window_ns=400.0, deadtime_ns=900.0, threshold_pe=0.25,
                 charge_res=0.1, time_res_ns=0.3, dark_rate_khz=0.0,
                 provisional=False),
    "hk":   dict(integration_window_ns=200.0, deadtime_ns=0.0, threshold_pe=0.25,
                 charge_res=0.05, time_res_ns=0.3, dark_rate_khz=0.0,
                 provisional=True),
}

# emission_process encoding for the hits.h5 decomposition. 0/1 mirror
# v3_writer's EMISSION_PROCESS_{CHERENKOV,SCINTILLATION}; DARK tags noise digits.
EMISSION_PROCESS_DARK = 2


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
    """Per-digit charge-resolution sigma (p.e.).

    ``"sk_like"``: the SK charge-dependent fractional resolution (arXiv:1307.0162
    Table 2), matching ``lucid.utils.smear_charges_SK_like``. A float ``f``:
    single-p.e. sigma scaling as ``f·sqrt(max(Q,1))`` (SPE spread added in
    quadrature over Q p.e.), so ``f`` is the 1-p.e. resolution (qbee 0.1, hk 0.05).

    The caller applies the smear (so ``basic``/``ski`` = ``"sk_like"`` reuse the
    existing keyed ``smear_charges_SK_like`` for parity; float models use this).
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
    d_sensor: list[int] = []
    d_pe: list[float] = []
    d_t: list[float] = []

    if n_ph:
        # Group photons by sensor, time-ordered within each group.
        order = np.lexsort((times, sensor_idx))
        s_sorted = sensor_idx[order]
        t_sorted = times[order]
        q_sorted = charges[order]

        digit_counter = 0
        start = 0
        while start < n_ph:
            s = s_sorted[start]
            end = start
            while end < n_ph and s_sorted[end] == s:
                end += 1
            gt = t_sorted[start:end]
            gq = q_sorted[start:end]
            gpos = order[start:end]
            m = end - start

            i = 0
            while i < m:
                t0 = gt[i]
                upper = t0 + window
                j = i
                pe = 0.0
                while j < m and gt[j] <= upper:
                    pe += gq[j]
                    j += 1
                if pe >= threshold:
                    photon_digit_idx[gpos[i:j]] = digit_counter
                    d_sensor.append(int(s))
                    d_pe.append(float(pe))
                    d_t.append(float(t0))
                    digit_counter += 1
                # Deadtime veto: hits in (upper, upper+deadtime] are dropped.
                dead_end = upper + deadtime
                while j < m and gt[j] <= dead_end:
                    j += 1
                i = j
            start = end

    return DigitizeResult(
        digit_sensor_idx=np.asarray(d_sensor, dtype=np.int32),
        digit_pe_true=np.asarray(d_pe, dtype=np.float32),
        digit_time=np.asarray(d_t, dtype=np.float32),
        photon_digit_idx=photon_digit_idx,
    )


def apply_readout_resolution(
    digit_pe_true: np.ndarray,
    digit_time: np.ndarray,
    model: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the model's charge resolution and optional electronics time jitter.

    Uniform across all models (including ``basic``): charge is smeared **once**
    by the model's ``charge_res`` (never stacked on anything else), and time
    gets an additive electronics jitter ``time_res_ns`` that is distinct from —
    and on top of — the PMT TTS already carried in ``digit_time``. Returns
    ``(pe_reco, t_reco)`` as float32.
    """
    pe_true = np.asarray(digit_pe_true, dtype=np.float64)
    if pe_true.size:
        sigma = charge_resolution_sigma(pe_true, model.get("charge_res", "sk_like"))
        pe_reco = np.clip(pe_true + rng.normal(size=pe_true.shape) * sigma, 0.0, None)
    else:
        pe_reco = pe_true
    t = np.asarray(digit_time, dtype=np.float64).copy()
    t_res = float(model.get("time_res_ns", 0.0))
    if t_res > 0.0 and t.size:
        t = t + rng.normal(size=t.shape) * t_res
    return pe_reco.astype(np.float32), t.astype(np.float32)


def _group_reduce(key_cols: tuple, pe: np.ndarray, t: np.ndarray, t_reco: np.ndarray):
    """Group rows by the tuple of integer key columns; PE=sum, T/T_reco=min.

    Returns ``(key_cols_out, PE, T, T_reco)`` where ``key_cols_out`` are the
    per-group key values (same tuple arity as ``key_cols``). ``key_cols[0]`` is
    the primary sort key. Mirrors the lexsort+reduceat groupby used elsewhere.
    """
    n = pe.shape[0]
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return (tuple(empty for _ in key_cols),
                np.empty(0, np.float64), np.empty(0, np.float64), np.empty(0, np.float64))
    order = np.lexsort(tuple(reversed(key_cols)))  # key_cols[0] primary
    cols_s = [np.asarray(c)[order] for c in key_cols]
    pe_s, t_s, tr_s = pe[order], t[order], t_reco[order]
    change = np.zeros(n, dtype=bool)
    change[0] = True
    for c in cols_s:
        change[1:] |= c[1:] != c[:-1]
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

    # Drop non-finite times (undetected/invalid deposits carry +inf).
    finite = np.isfinite(t_reco) & np.isfinite(t_true) & (charge > 0)
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
    pe_reco, t_reco_digit = apply_readout_resolution(
        res.digit_pe_true, res.digit_time, model, rng)
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
        "PE": hPE.astype(np.float32), "T": hT.astype(np.float32),
        "T_reco": hTR.astype(np.float32),
    }

    # step/sensor_hits: real deposits only (dark has no segment).
    smask = (didx >= 0) & (segment_idx >= 0)
    sk, sPE, sT, sTR = _group_reduce(
        (segment_idx[smask], sensor_idx[smask], didx[smask], emission_process[smask]),
        charge[smask], t_true[smask], t_reco[smask])
    seg_hits = {
        "segment_idx": sk[0].astype(np.int32), "sensor_idx": sk[1].astype(np.uint16),
        "digit_idx": sk[2].astype(np.int32), "emission_process": sk[3].astype(np.int8),
        "PE": sPE.astype(np.float32), "T": sT.astype(np.float32),
        "T_reco": sTR.astype(np.float32),
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
