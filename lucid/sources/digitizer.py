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
