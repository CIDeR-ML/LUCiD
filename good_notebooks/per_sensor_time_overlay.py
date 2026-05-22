"""Event-level and per-sensor visualization for the likelihood loss.

For a single ground-truth event generated from a PhotonSim ROOT file,
this script produces:

  1. Four 2D unwrapped detector displays (true / sim, charge / time)
     using ``lucid.visualization.create_detector_display`` and saving
     to ``figures/event_displays/``.

  2. One per-sensor diagnostic plot per selected sensor (stratified into
     high / medium / low predicted PE), saving to
     ``figures/per_sensor_time/``. Each plot has three stacked panels:

       (a) Predicted photon-arrival rate λ(t) at this sensor: the
           Gaussian-convolved sum of per-photon weights, plus the bare
           weighted swarm. This is the "physical rate" of photon
           arrivals at the channel.

       (b) Cumulative expected PE vs time, with a 1-PE reference line.

       (c) **First-arrival density p_first(t) = λ(t) · exp(-Λ(t))** —
           this is what the time NLL evaluates. Each photon contributes
           a Gaussian K_σ at its predicted t_i; the survival factor
           exp(-Λ) suppresses the body and tail of the rate, leaving a
           sharp distribution concentrated at the leading edge. The
           observed first-arrival t_obs is overlaid; the per-sensor
           NLL_s = -log p_first(t_obs) is annotated.

     The data simulator is run with ``apply_smearing=True`` so the
     observed first-arrival times include per-photon TTS through the
     simulator's segment_min path. σ_TTS used for the visualization
     matches sim_config.tts_sigma_ns (default 2.5 ns, SK R3600).

Mirrors the simulator setup of ``parameter_scans_1D_likelihood.ipynb``
and the display style of ``cylinder_2D_displays.ipynb``.

Run:

    cd good_notebooks && python per_sensor_time_overlay.py
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.append('..')

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.generate import read_photon_data_from_photonsim
from lucid.detector_params import ParticleParams
from lucid.visualization import create_detector_display


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['savefig.dpi'] = 200


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_JSON_FILENAME = '../config/SK_geom_config.json'
PHYSICS_CONFIG = '../config/SK_physics_config.json'
DATA_FILE = '../data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root'

TEMPERATURE = 0.10
K_PRED = 7
K_DATA = 20
NPHOT_PRED = 1_000_000   # photons used by the prediction simulators
NPHOT_DATA = 150_000     # photon budget for the data simulator (data pads to 1M anyway)
ENTRY_IDX = 2
SEED = 44

# Color-scale percentiles for the 2D event displays (shared across true / sim)
CHARGE_PERC = (1.0, 99.0)   # log-scale, computed from union of true + pred positive values
TIME_PERC = (1.0, 99.0)     # linear-scale, computed from union of true + pred positive values

N_PER_STRATUM = 3        # 3 high + 3 medium + 3 low = 9 per-sensor plots
TIME_BINS = 80

# Single-PE TTS used for the analytic likelihood overlay. Should match the
# sim_config.tts_sigma_ns used by the data simulator (default 2.5 ns SK).
SIGMA_TTS_NS = 2.5

EVENT_DISPLAY_DIR = Path('figures/event_displays')
PER_SENSOR_DIR = Path('figures/per_sensor_time')


# ---------------------------------------------------------------------------
# Simulator setup
# ---------------------------------------------------------------------------
def build_simulators():
    """Build the data simulator and a single per-photon prediction simulator.

    Per-sensor charges and arrival times are derived later by aggregating the
    per-photon output, so no separate ``hit_mode='aggregated'`` simulator is
    needed.
    """
    detector = generate_detector(DEFAULT_JSON_FILENAME)
    num_detectors = len(detector.all_points)

    # temperature=None → step-function sensor assignment (hard, single sensor
    # per photon hit). With Bernoulli QE and binary survival/detect, this
    # gives integer per-sensor PE counts in data mode.
    data_simulator = setup_event_simulator(
        DEFAULT_JSON_FILENAME, NPHOT_DATA, temperature=None, K=K_DATA,
        is_data=True, is_calibration=False, apply_smearing=True,
        physics_config=PHYSICS_CONFIG, default_detector_params=True)

    prediction_simulator = setup_event_simulator(
        DEFAULT_JSON_FILENAME, NPHOT_PRED, TEMPERATURE, max_candidates_per_ray=4, K=K_PRED,
        is_data=False, hit_mode='per_photon',
        physics_config=PHYSICS_CONFIG, default_detector_params=True)

    return detector, num_detectors, data_simulator, prediction_simulator


# ---------------------------------------------------------------------------
# Event generation (matches parameter_scans_1D_likelihood.ipynb cell 7)
# ---------------------------------------------------------------------------
def build_event(detector, data_simulator, prediction_simulator, key):
    """Generate a ground-truth event and run the per-photon prediction."""
    photon_data = read_photon_data_from_photonsim(DATA_FILE, ENTRY_IDX)

    photon_origins = photon_data['photon_origins']
    photon_directions = photon_data['photon_directions']
    photon_times = photon_data['photon_times']
    n = len(photon_origins)
    pad = max(0, 1_000_000 - n)

    photon_data['photon_origins'] = jnp.pad(
        photon_origins, ((0, pad), (0, 0)), mode='constant', constant_values=0)
    if pad > 0:
        photon_data['photon_directions'] = jnp.concatenate(
            [photon_directions,
             jnp.tile(jnp.array([0.0, 0.0, 1.0]), (pad, 1))], axis=0)
    else:
        photon_data['photon_directions'] = photon_directions
    photon_data['photon_times'] = jnp.pad(
        photon_times, (0, pad), mode='constant', constant_values=0)
    # Per-photon wavelengths must be padded to match the other photon arrays:
    # the data simulator sizes its optical arrays by photon_origins.shape[0].
    # `mode='edge'` keeps the pad values inside the medium's wavelength grid
    # (padded photons carry intensity 0, so the value itself is inert).
    if 'wavelengths' in photon_data and pad > 0:
        photon_data['wavelengths'] = jnp.pad(
            photon_data['wavelengths'], (0, pad), mode='edge')
    photon_data['N'] = n

    # Random vertex inside 60% of the detector volume
    fraction = 0.6
    r_vert = jax.random.uniform(key, shape=(), minval=0,
                                maxval=detector.r * fraction)
    key, _ = jax.random.split(key)
    theta_pos = jax.random.uniform(key, shape=(), minval=0,
                                   maxval=2 * jnp.pi)
    key, _ = jax.random.split(key)
    z_vert = jax.random.uniform(key, shape=(),
                                minval=-detector.H / 2 * fraction,
                                maxval=detector.H / 2 * fraction)
    true_position = jnp.array([r_vert * jnp.cos(theta_pos),
                               r_vert * jnp.sin(theta_pos),
                               z_vert])

    # Random direction (uniform on sphere)
    key, _ = jax.random.split(key)
    phi = jax.random.uniform(key, shape=(), minval=0, maxval=2 * jnp.pi)
    key, _ = jax.random.split(key)
    cos_theta = jax.random.uniform(key, shape=(), minval=-1, maxval=1)
    sin_theta = jnp.sqrt(1 - cos_theta**2)
    true_direction = jnp.array([sin_theta * jnp.cos(phi),
                                sin_theta * jnp.sin(phi),
                                cos_theta])

    # Rotation aligning ROOT photons (along z) with the true direction
    original_direction = jnp.array([0.0, 0.0, 1.0])
    true_dir_n = true_direction / (jnp.linalg.norm(true_direction) + 1e-8)
    rotation_axis = jnp.cross(original_direction, true_dir_n)
    axis_norm = jnp.linalg.norm(rotation_axis)
    rotation_axis = jnp.where(
        axis_norm < 1e-6,
        jnp.array([1.0, 0.0, 0.0]),
        rotation_axis / (axis_norm + 1e-8))
    rotation_angle = jnp.arccos(jnp.clip(
        jnp.dot(original_direction, true_dir_n), -1.0, 1.0))

    photon_data['rotation_axis'] = rotation_axis
    photon_data['rotation_angle'] = rotation_angle
    photon_data['apply_rotation'] = jnp.array(True)
    photon_data['apply_translation'] = jnp.array(True)
    photon_data['translation_vector'] = true_position

    true_energy = photon_data['energy']
    true_track = ParticleParams.from_cartesian(
        energy=true_energy, position=true_position,
        direction=true_direction, t0=0.0)

    # Run both simulators on the same track. ``photon_data`` is also
    # returned so callers (e.g. the validation script) can re-run the
    # data simulator many times with fresh keys on the same true event.
    key, sub_data = jax.random.split(key)
    true_data = jax.lax.stop_gradient(
        data_simulator(true_track, sub_data, photon_data))

    key, sub_pred = jax.random.split(key)
    pred_data_pp = jax.lax.stop_gradient(
        prediction_simulator(true_track, sub_pred))

    return (true_track, true_data, pred_data_pp,
            true_position, true_direction, true_energy, photon_data)


# ---------------------------------------------------------------------------
# 2D event displays
# ---------------------------------------------------------------------------
def aggregate_per_sensor(flat_times, weights, flat_indices, num_sensors):
    """Aggregate per-photon arrays into per-sensor (total_pe, t_first_pe, t_median).

    For each sensor s collects all photons routed to it, sorts by arrival
    time, and computes the cumulative weighted PE. Returns:

      total_pe[s]    - sum of weights routed to sensor s (expected PE)
      t_first_pe[s]  - smallest time at which cumsum >= 1 PE (NaN if total < 1)
      t_median[s]    - 50% cumulative-weight crossing time (NaN if no photons)
    """
    flat_indices = np.asarray(flat_indices)
    flat_times = np.asarray(flat_times)
    weights = np.asarray(weights)

    valid = (flat_indices >= 0) & (flat_indices < num_sensors)
    fi = flat_indices[valid]
    ft = flat_times[valid]
    fw = weights[valid]

    order = np.lexsort((ft, fi))
    sorted_i = fi[order]
    sorted_t = ft[order]
    sorted_w = fw[order]

    boundaries = np.searchsorted(sorted_i, np.arange(num_sensors + 1))

    total_pe = np.zeros(num_sensors)
    t_first_pe = np.full(num_sensors, np.nan)
    t_median = np.full(num_sensors, np.nan)

    for s in range(num_sensors):
        start, end = boundaries[s], boundaries[s + 1]
        if start >= end:
            continue
        t_s = sorted_t[start:end]
        w_s = sorted_w[start:end]
        cum = np.cumsum(w_s)
        total = float(cum[-1])
        total_pe[s] = total
        if total > 0:
            mid_idx = min(int(np.searchsorted(cum, 0.5 * total)), len(t_s) - 1)
            t_median[s] = float(t_s[mid_idx])
        if total >= 1.0:
            idx = int(np.searchsorted(cum, 1.0))
            if idx < len(t_s):
                t_first_pe[s] = float(t_s[idx])

    return total_pe, t_first_pe, t_median


def _to_sparse(charges, times):
    """Convert dense (charges, times) to (indices, charges, times) over active sensors only."""
    charges = np.asarray(charges)
    times = np.asarray(times)
    active = np.where(charges > 0)[0]
    return active, charges[active], times[active]


def render_event_displays(true_data, pred_charges, pred_times):
    """Save four 2D unwrapped detector displays (true/sim x charge/time).

    Sparse mode is used so inactive sensors are excluded from the percentile
    range (matches cylinder_2D_displays.ipynb). True and predicted share a
    single color scale per quantity so the panels are directly comparable.
    Charge is log-scale (multi-decade dynamic range), time is linear.
    """
    EVENT_DISPLAY_DIR.mkdir(parents=True, exist_ok=True)
    display = create_detector_display(DEFAULT_JSON_FILENAME, sparse=True)

    true_idx, true_charges_a, true_times_a = _to_sparse(true_data[0], true_data[2])
    pred_idx, pred_charges_a, pred_times_a = _to_sparse(pred_charges, pred_times)

    # Shared color ranges over active sensors only.
    all_charges = np.concatenate([true_charges_a, pred_charges_a])
    all_times = np.concatenate([true_times_a, pred_times_a])
    charge_vmin = float(np.percentile(all_charges, CHARGE_PERC[0]))
    charge_vmax = float(np.percentile(all_charges, CHARGE_PERC[1]))
    time_vmin = float(np.percentile(all_times, TIME_PERC[0]))
    time_vmax = float(np.percentile(all_times, TIME_PERC[1]))

    print(f'Active sensors: true={true_idx.size}, pred={pred_idx.size}')
    print(f'Charge color range (log):  [{charge_vmin:.3g}, {charge_vmax:.3g}] PE')
    print(f'Time color range (linear): [{time_vmin:.2f}, {time_vmax:.2f}] ns')

    # True charge (log)
    display(true_idx, true_charges_a, true_times_a,
            file_name=str(EVENT_DISPLAY_DIR / 'true_event_charge.png'),
            plot_time=False, log_scale=True,
            vmin=charge_vmin, vmax=charge_vmax)
    # Predicted charge (log)
    display(pred_idx, pred_charges_a, pred_times_a,
            file_name=str(EVENT_DISPLAY_DIR / 'pred_event_charge.png'),
            plot_time=False, log_scale=True,
            vmin=charge_vmin, vmax=charge_vmax)
    # True time (linear)
    display(true_idx, true_charges_a, true_times_a,
            file_name=str(EVENT_DISPLAY_DIR / 'true_event_time.png'),
            plot_time=True, log_scale=False,
            vmin=time_vmin, vmax=time_vmax)
    # Predicted time (linear)
    display(pred_idx, pred_charges_a, pred_times_a,
            file_name=str(EVENT_DISPLAY_DIR / 'pred_event_time.png'),
            plot_time=True, log_scale=False,
            vmin=time_vmin, vmax=time_vmax)

    print(f'Saved 4 event displays to {EVENT_DISPLAY_DIR}/')


# ---------------------------------------------------------------------------
# Per-sensor time plots
# ---------------------------------------------------------------------------
def stratified_sensor_selection(total_charge, hit_counts,
                                n_per_stratum=N_PER_STRATUM,
                                hit_threshold=1.0):
    """Pick (high, medium, low) PE sensors among those with at least
    ``hit_threshold`` PE observed. ``total_charge`` from ``make_hits_data``
    is fractional (continuous propagation weights × Bernoulli QE), so
    using ``hit_counts > 0`` admits sub-PE channels that aren't really
    hits — must threshold at ≥ 1 PE."""
    hit_counts = np.asarray(hit_counts)
    total_charge = np.asarray(total_charge)
    candidates = np.where(hit_counts >= hit_threshold)[0]
    if candidates.size < 3 * n_per_stratum:
        raise RuntimeError(
            f'Only {candidates.size} sensors with obs PE >= {hit_threshold}; '
            f'need at least {3 * n_per_stratum} for stratified selection.')

    order = np.argsort(total_charge[candidates])
    sorted_idx = candidates[order]
    n = len(sorted_idx)

    high = sorted_idx[-n_per_stratum:][::-1]
    mid_start = max(n // 2 - n_per_stratum // 2, 0)
    medium = sorted_idx[mid_start:mid_start + n_per_stratum]
    low = sorted_idx[:n_per_stratum]
    return list(map(int, high)), list(map(int, medium)), list(map(int, low))


def per_sensor_arrivals(flat_times, weights, flat_indices, sensor_id):
    """Slice predicted per-photon arrivals routed to a single sensor."""
    mask = flat_indices == sensor_id
    return flat_times[mask], weights[mask]


def make_diagnostics_evaluator(num_sensors, n_grid=200, sigma_tts=SIGMA_TTS_NS,
                                edge_pad_sigmas=5.0, support_frac=0.01):
    """Factory for the unified per-sensor likelihood-diagnostics JIT call.

    Each sensor's grid spans [t_lead - 5σ, t_max + 5σ] where t_lead and
    t_max are the earliest and latest detected photon times *for that
    sensor*. No arbitrary global window — the grid follows the data. This
    means per-sensor resolution scales with how spread the photon swarm
    is at that sensor (sharp at high-PE, coarser at low-PE with long
    scattered tails) but the curves are never truncated.

    Returns a dict of per-sensor numpy arrays:

        t_lead, t_max       (num_sensors,)             grid endpoints
        t_grid              (num_sensors, n_grid)      absolute time per cell
        lam_grid            (num_sensors, n_grid)
        Lam_grid            (num_sensors, n_grid)
        p_first_grid        (num_sensors, n_grid)
        mode                (num_sensors,)             argmax of p_first
        support_lo, hi      (num_sensors,)
        log_lambda_obs, Lam_obs, nll_obs   (num_sensors,)

    All quantities computed in one fused JIT call.
    """
    K_norm = 1.0 / (sigma_tts * jnp.sqrt(2.0 * jnp.pi))
    log_K_const = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_tts)
    fractions = jnp.linspace(0.0, 1.0, n_grid)

    @jax.jit
    def evaluate_jit(flat_times, flat_weights, flat_indices,
                     t_obs_per_sensor):
        valid_mask = flat_weights > 0
        valid_t_lo = jnp.where(valid_mask, flat_times, jnp.inf)
        valid_t_hi = jnp.where(valid_mask, flat_times, -jnp.inf)

        # Per-sensor earliest and latest detected photon times.
        t_lead = jax.ops.segment_min(
            valid_t_lo, flat_indices, num_segments=num_sensors)
        t_max_raw = jax.ops.segment_max(
            valid_t_hi, flat_indices, num_segments=num_sensors)
        t_lead = jnp.where(jnp.isfinite(t_lead), t_lead, 0.0)
        t_max_raw = jnp.where(jnp.isfinite(t_max_raw), t_max_raw, t_lead)

        # Cap the grid right edge at t_lead + 100σ. A single outlier
        # photon at e.g. 900 ns doesn't affect λ near the bulk (Gaussian
        # tail decays in <10σ), but would otherwise destroy the
        # per-sensor grid resolution. The cap keeps grid spacing fine
        # enough to resolve the leading edge of high-PE sensors.
        t_max_cap = t_lead + 100.0 * sigma_tts
        t_max = jnp.minimum(t_max_raw, t_max_cap)

        # Per-sensor grid endpoints with 5σ Gaussian-smearing buffer.
        t_grid_lo = t_lead - edge_pad_sigmas * sigma_tts
        t_grid_hi = t_max + edge_pad_sigmas * sigma_tts
        t_grid_hi = jnp.maximum(t_grid_hi, t_grid_lo + 2.0 * sigma_tts)

        # t_grid_per_sensor[s, j] = t_grid_lo[s] + fractions[j]·(hi[s]-lo[s])
        t_grid_per_sensor = (t_grid_lo[:, None]
                             + fractions[None, :]
                             * (t_grid_hi - t_grid_lo)[:, None])

        # ----------------------------------------------------------------
        # λ and Λ on per-sensor grids. Scan over grid index j; at each j,
        # broadcast the j-th column of the per-sensor grid to per-photon
        # via flat_indices, then segment_sum.
        # ----------------------------------------------------------------
        def step(_, j_idx):
            t_eval_pp = t_grid_per_sensor[flat_indices, j_idx]
            x = (t_eval_pp - flat_times) / sigma_tts
            K = jnp.exp(-0.5 * x * x) * K_norm
            F = jax.scipy.special.ndtr(x)
            lam_col = jax.ops.segment_sum(
                flat_weights * K, flat_indices, num_segments=num_sensors)
            Lam_col = jax.ops.segment_sum(
                flat_weights * F, flat_indices, num_segments=num_sensors)
            return None, (lam_col, Lam_col)

        _, (lam_cols, Lam_cols) = jax.lax.scan(
            step, None, jnp.arange(n_grid))
        lam_grid = lam_cols.T
        Lam_grid = Lam_cols.T
        p_first_grid = lam_grid * jnp.exp(-Lam_grid)

        # ----------------------------------------------------------------
        # 99th percentile of cumulative Λ — the time by which 99% of the
        # expected detections should have occurred. Used as the natural
        # right edge of the plot range (avoids wasting space on a long
        # scattered-light tail with negligible weight).
        # ----------------------------------------------------------------
        Lam_total = Lam_grid[:, -1]
        target = 0.99 * Lam_total
        above_target = Lam_grid >= target[:, None]
        any_above_target = jnp.any(above_target, axis=1)
        first_above = jnp.argmax(above_target, axis=1)
        t99_per_sensor = jnp.where(
            any_above_target,
            t_grid_per_sensor[jnp.arange(num_sensors), first_above],
            t_grid_hi)

        # ----------------------------------------------------------------
        # Mode and support of p_first per sensor.
        # ----------------------------------------------------------------
        max_p = jnp.max(p_first_grid, axis=1)
        any_p = max_p > 0
        amax = jnp.argmax(p_first_grid, axis=1)
        mode_t = jnp.where(any_p,
                           t_grid_per_sensor[jnp.arange(num_sensors), amax],
                           jnp.nan)

        thresh = jnp.where(any_p, support_frac * max_p, 0.0)[:, None]
        above = p_first_grid > thresh
        any_above = jnp.any(above, axis=1)
        first_col = jnp.argmax(above, axis=1)
        last_col = (n_grid - 1) - jnp.argmax(above[:, ::-1], axis=1)
        keep = any_above & any_p
        support_lo = jnp.where(
            keep,
            t_grid_per_sensor[jnp.arange(num_sensors), first_col],
            jnp.nan)
        support_hi = jnp.where(
            keep,
            t_grid_per_sensor[jnp.arange(num_sensors), last_col],
            jnp.nan)

        # ----------------------------------------------------------------
        # -log λ, Λ, NLL evaluated at the observed t_obs (per sensor).
        # ----------------------------------------------------------------
        t_obs_pp = t_obs_per_sensor[flat_indices]
        x_obs = (t_obs_pp - flat_times) / sigma_tts
        log_K_obs = -0.5 * x_obs * x_obs + log_K_const
        log_w_safe = jnp.where(flat_weights > 0,
                                jnp.log(jnp.maximum(flat_weights, 1e-30)),
                                -jnp.inf)
        log_terms = log_w_safe + log_K_obs

        max_per = jax.ops.segment_max(
            log_terms, flat_indices, num_segments=num_sensors)
        max_safe = jnp.where(jnp.isfinite(max_per), max_per, 0.0)
        shifted = log_terms - max_safe[flat_indices]
        exp_shifted = jnp.where(jnp.isfinite(log_terms), jnp.exp(shifted), 0.0)
        sum_exp = jax.ops.segment_sum(
            exp_shifted, flat_indices, num_segments=num_sensors)
        log_lambda_obs = max_safe + jnp.log(jnp.maximum(sum_exp, 1e-30))

        F_obs = jax.scipy.special.ndtr(x_obs)
        Lam_obs = jax.ops.segment_sum(
            flat_weights * F_obs, flat_indices, num_segments=num_sensors)
        nll_obs = -log_lambda_obs + Lam_obs

        return {
            't_lead': t_lead,
            't_max': t_max,
            't99': t99_per_sensor,
            't_grid': t_grid_per_sensor,
            'lam_grid': lam_grid,
            'Lam_grid': Lam_grid,
            'p_first_grid': p_first_grid,
            'mode': mode_t,
            'support_lo': support_lo,
            'support_hi': support_hi,
            'log_lambda_obs': log_lambda_obs,
            'Lam_obs': Lam_obs,
            'nll_obs': nll_obs,
        }

    def evaluate(flat_times, flat_weights, flat_indices, t_obs_per_sensor):
        """Wrapper: cast to JAX, run JIT'd evaluator, return numpy."""
        out = evaluate_jit(
            jnp.asarray(flat_times, dtype=jnp.float32),
            jnp.asarray(flat_weights, dtype=jnp.float32),
            jnp.asarray(flat_indices),
            jnp.asarray(t_obs_per_sensor, dtype=jnp.float32))
        return {k: np.asarray(v) for k, v in out.items()}

    return evaluate


def per_sensor_x_range(t_pred, w_pred, t_obs, sigma_tts=SIGMA_TTS_NS,
                       perc=(2.0, 75.0), pad_sigmas=4.0):
    """Per-panel x-range focused on the leading edge / body of the photon
    swarm (where the loss has support), with enough padding around t_obs
    to keep it visible.

    The weighted percentile defaults to (2%, 75%) — keeps the leading edge
    and bulk while trimming the long scattered-light tail. ``pad_sigmas``
    is the multiplier of σ_TTS used as margin around t_obs.
    """
    pad = pad_sigmas * sigma_tts
    if t_pred.size > 5 and np.sum(w_pred) > 0:
        order = np.argsort(t_pred)
        cumw = np.cumsum(w_pred[order])
        total = cumw[-1]
        lo_idx = int(np.searchsorted(cumw, perc[0] / 100.0 * total))
        hi_idx = min(int(np.searchsorted(cumw, perc[1] / 100.0 * total)),
                     len(order) - 1)
        lo = float(t_pred[order[lo_idx]])
        hi = float(t_pred[order[hi_idx]])
    elif t_pred.size > 0:
        lo, hi = float(np.min(t_pred)), float(np.max(t_pred))
    else:
        lo, hi = t_obs - pad, t_obs + pad

    if np.isfinite(t_obs):
        lo = min(lo, t_obs - pad)
        hi = max(hi, t_obs + pad)
    if hi - lo < 1e-6:
        hi = lo + 2.0 * pad
    return lo, hi


def plot_single_sensor(sensor_id, stratum_label,
                       flat_times, weights, flat_indices,
                       total_pe, hit_counts, hit_times,
                       diag,
                       sigma_tts=SIGMA_TTS_NS,
                       n_bins=60,
                       log_y=True,
                       out_path=None):
    """Two-panel per-sensor diagnostic — uses precomputed per-sensor curves.

    ``diag`` is the dict returned by ``make_diagnostics_evaluator``;
    ``offsets`` are the grid offsets relative to the per-sensor leading
    edge. No λ / Λ / p_first / NLL recomputation happens here.

    Top:    Predicted rate λ(t) + binned bare per-photon arrivals.
    Bottom: First-arrival density p_first(t) = λ(t) · exp(-Λ(t)). NLL_s
            and the -log λ + Λ decomposition at the observed t_obs are
            annotated.
    """
    s = sensor_id
    pred_pe = float(total_pe[s])
    t_obs = float(hit_times[s])
    obs_pe = float(hit_counts[s])

    # Per-sensor precomputed curves on the data-driven grid
    t_grid = diag['t_grid'][s]
    lam_curve = diag['lam_grid'][s]
    p_first_curve = diag['p_first_grid'][s]
    mode = float(diag['mode'][s])
    log_lambda_obs = float(diag['log_lambda_obs'][s])
    Lam_obs = float(diag['Lam_obs'][s])
    nll = float(diag['nll_obs'][s])

    # Per-sensor swarm slice (for the bar histogram)
    mask = (flat_indices == s) & (weights > 0)
    t_pred = flat_times[mask]
    w_pred = weights[mask]

    # Plot range: left edge = t_lead - 5σ (start of grid). Right edge =
    # 99th-percentile of cumulative Λ + 2σ — captures essentially all
    # the expected charge without wasting space on outlier scattered
    # photons. Always include t_obs.
    t99 = float(diag['t99'][s])
    x_lo = float(t_grid[0])
    x_hi = t99 + 2.0 * sigma_tts
    if np.isfinite(t_obs):
        x_lo = min(x_lo, t_obs)
        x_hi = max(x_hi, t_obs)

    bins = np.linspace(x_lo, x_hi, n_bins + 1)
    bin_width = (x_hi - x_lo) / n_bins
    counts, _ = np.histogram(t_pred, bins=bins, weights=w_pred)
    rate_hist = counts / bin_width
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, (ax_rate, ax_lik) = plt.subplots(
        2, 1, figsize=(7.0, 6.6), sharex=True,
        gridspec_kw={'height_ratios': [1.0, 1.0], 'hspace': 0.10})

    # ---------------------------------------------------------------
    # Top panel: rate λ(t) + bare-arrival histogram
    # ---------------------------------------------------------------
    ax_rate.bar(centers, rate_hist, width=bin_width * 0.95, align='center',
                color='lightsteelblue', edgecolor='steelblue', linewidth=0.4,
                alpha=0.55, label='per-photon arrivals (binned)')
    ax_rate.plot(t_grid, lam_curve, color='navy', linewidth=2.0,
                 label=r'$\lambda(t)$')
    ax_rate.axvline(t_obs, color='crimson', linestyle='--', linewidth=1.6,
                    label=fr'$t_{{obs}} = {t_obs:.2f}$ ns', zorder=4)

    ax_rate.set_ylabel('expected PE / ns')
    if log_y:
        # Set a sensible floor so log-scale doesn't crush small values to -∞.
        nonzero_max = max(float(np.max(lam_curve)) if lam_curve.size else 1.0,
                          float(np.max(rate_hist)) if rate_hist.size else 1.0,
                          1e-3)
        floor = max(1e-4, nonzero_max * 1e-5)
        ax_rate.set_yscale('log')
        ax_rate.set_ylim(floor, nonzero_max * 3.0)
    else:
        ax_rate.set_ylim(bottom=0)
    ax_rate.grid(True, which='both', alpha=0.3, linestyle=':')
    ax_rate.legend(fontsize=8, frameon=True, loc='upper right',
                   facecolor='white', framealpha=0.85)
    ax_rate.set_title(
        f'{stratum_label}  $\\bullet$  Sensor {sensor_id}\n'
        f'pred $\\langle$PE$\\rangle$ = {pred_pe:.2f}  '
        f'$\\bullet$  obs PE = {obs_pe:.2f}',
        fontsize=11)

    # ---------------------------------------------------------------
    # Bottom panel: first-arrival density p_first(t)
    # ---------------------------------------------------------------
    ax_lik.fill_between(t_grid, 0, p_first_curve, color='seagreen', alpha=0.22,
                        linewidth=0)
    ax_lik.plot(t_grid, p_first_curve, color='darkgreen', linewidth=2.0,
                label=r'$p_{\mathrm{first}}(t)$')
    if np.isfinite(mode):
        ax_lik.axvline(mode, color='darkorange', linestyle='-', linewidth=1.4,
                       label=fr'mode = {mode:.2f} ns', zorder=3)
    ax_lik.axvline(t_obs, color='crimson', linestyle='--', linewidth=1.6,
                   label=fr'$t_{{obs}} = {t_obs:.2f}$ ns', zorder=4)

    ax_lik.set_xlim(x_lo, x_hi)
    ax_lik.set_xlabel('arrival time [ns]')
    ax_lik.set_ylabel('first-arrival pdf [1/ns]')
    if log_y:
        nonzero_max_p = float(np.max(p_first_curve)) if p_first_curve.size else 1.0
        if nonzero_max_p <= 0:
            nonzero_max_p = 1.0
        floor_p = max(1e-6, nonzero_max_p * 1e-5)
        ax_lik.set_yscale('log')
        ax_lik.set_ylim(floor_p, nonzero_max_p * 3.0)
    else:
        ax_lik.set_ylim(bottom=0)
    ax_lik.grid(True, which='both', alpha=0.3, linestyle=':')
    ax_lik.legend(fontsize=8, frameon=True, loc='upper right',
                  facecolor='white', framealpha=0.85)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
    return out_path


def render_per_sensor_plots(flat_times, weights, flat_indices,
                            total_pe, true_data, stratum_sensors, diag):
    """Generate one figure per selected sensor using precomputed
    per-sensor curves from ``diag`` (output of the JIT'd evaluator)."""
    PER_SENSOR_DIR.mkdir(parents=True, exist_ok=True)

    hit_counts = np.asarray(true_data[0])
    hit_times = np.asarray(true_data[2])

    label_map = {'high': 'High PE', 'medium': 'Medium PE', 'low': 'Low PE'}
    saved = []
    for stratum_key, sensors in stratum_sensors.items():
        for s in sensors:
            out = PER_SENSOR_DIR / f'{stratum_key}_sensor_{s}.png'
            plot_single_sensor(
                sensor_id=s,
                stratum_label=label_map[stratum_key],
                flat_times=flat_times,
                weights=weights,
                flat_indices=flat_indices,
                total_pe=total_pe,
                hit_counts=hit_counts,
                hit_times=hit_times,
                diag=diag,
                out_path=str(out),
            )
            saved.append(str(out))
    print(f'Saved {len(saved)} per-sensor plots to {PER_SENSOR_DIR}/')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    detector, num_detectors, data_sim, pred_sim = build_simulators()
    print(f'Detectors: {num_detectors}')

    key = jax.random.PRNGKey(SEED)
    (true_track, true_data, pred_data_pp,
     pos, direction, energy, _photon_data) = build_event(
        detector, data_sim, pred_sim, key)

    hit_counts = np.asarray(true_data[0])
    hit_times = np.asarray(true_data[2])
    n_hit = int(np.sum(hit_counts > 0))
    print(f'Event: E = {float(energy):.1f} MeV, hit sensors = {n_hit} / {num_detectors}')

    # Per-photon prediction output → numpy. ``total_charge`` from
    # make_hits_likelihood is exactly the per-sensor expected PE, so we
    # don't need ``aggregate_per_sensor`` for that.
    log_w, flat_times_jax, flat_indices_jax, total_charge_jax = pred_data_pp
    flat_times = np.asarray(flat_times_jax)
    flat_indices = np.asarray(flat_indices_jax)
    weights = np.exp(np.asarray(log_w))
    total_pe = np.asarray(total_charge_jax)

    n_pred_active = int(np.sum(total_pe > 0))
    n_pred_above_1pe = int(np.sum(total_pe >= 1.0))
    print(f'Predicted active sensors: {n_pred_active} '
          f'(of which {n_pred_above_1pe} have >= 1 expected PE)')

    # ---- One JIT'd call: λ, Λ, p_first on the per-sensor grid + mode +
    # support + (-log λ, Λ, NLL) at t_obs. ~sub-second on GPU after the
    # first compile. Replaces the per-sensor Python loop completely.
    print('Computing per-sensor likelihood diagnostics (JIT)...')
    import time as _time
    _t0 = _time.time()
    # Per-sensor grid spans [t_lead-5σ, t_max+5σ] of that sensor's actual
    # photon distribution. No global hand-set window — the grid follows
    # the data.
    evaluate = make_diagnostics_evaluator(
        num_sensors=num_detectors, n_grid=200, sigma_tts=SIGMA_TTS_NS)
    diag = evaluate(flat_times, weights, flat_indices, hit_times)
    print(f'  done in {_time.time() - _t0:.2f} s '
          f'(grid: {diag["t_grid"].shape[1]} samples / sensor)')

    # Predicted event display uses the mode of p_first as the per-sensor
    # predicted time — this matches what the loss "expects" the first
    # arrival to be (peak of the first-arrival pdf). Fall back to t_lead
    # for sensors where the mode isn't well-defined.
    pred_charges = total_pe
    pred_times = np.where(np.isfinite(diag['mode']),
                          diag['mode'], diag['t_lead'])
    pred_times = np.nan_to_num(pred_times, nan=0.0)

    # 2D event displays (true and sim, charge and time)
    render_event_displays(true_data, pred_charges, pred_times)

    # Per-sensor time plots — strata thresholded at >= 1 PE so we only
    # show sensors that were actually hit (not sub-PE fractional charges).
    high, medium, low = stratified_sensor_selection(
        total_pe, hit_counts, n_per_stratum=N_PER_STRATUM, hit_threshold=1.0)
    print(f'High PE   sensors: {high}')
    print(f'Medium PE sensors: {medium}')
    print(f'Low PE    sensors: {low}')

    render_per_sensor_plots(
        flat_times=flat_times, weights=weights, flat_indices=flat_indices,
        total_pe=total_pe, true_data=true_data,
        stratum_sensors={'high': high, 'medium': medium, 'low': low},
        diag=diag,
    )


if __name__ == '__main__':
    main()
