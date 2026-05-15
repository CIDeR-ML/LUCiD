"""1D parameter scans with the new Gaussian-TTS / Poisson-process loss.

Validates that loss is minimum and gradient is zero at the true parameter
values for each of (X, Y, Z, t0, theta, phi, E).

Loss:
    NLL_total =  Σ_s charge_NLL_s   (over all sensors, raw Poisson NLL)
              +  Σ_s∈hit  time_NLL_s   (Gaussian-TTS first-arrival NLL,
                                        σ_TTS = 2.5 ns)

No vertex term, no geometric-mean combiner. Both terms are summed
per-sensor log-likelihoods so per-sensor Fisher info self-balances.

t0 enters via t_obs_shifted = observed_times − t0 in the time term.

Run:
    cd good_notebooks && python parameter_scans_1D_gaussian.py
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

sys.path.append('..')

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad
from jax.scipy.special import gammaln
from matplotlib import pyplot as plt

from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.generate import read_photon_data_from_photonsim
from lucid.detector_params import ParticleParams
from lucid.optimization.utils.functions import cartesian_to_spherical


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_JSON_FILENAME = '../config/SK_geom_config.json'
PHYSICS_CONFIG = '../config/SK_physics_config.json'
DATA_FILE = '../data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root'

TEMPERATURE = 0.10
N_SCAN_POINTS = 21
K_PRED = 7
K_DATA = 20
NPHOT_PRED = 1_000_000
NPHOT_DATA = 150_000
ENTRY_IDX = 2
SEED = 44
SIGMA_TTS_NS = 2.5

OUT_DIR = Path('figures/parameter_scans_gaussian')

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------
def poisson_nll_sum(observed_counts, predicted_charge, eps=1e-30):
    """Raw per-sensor Poisson NLL summed over all sensors.

    Σ_s [λ_s − N_s · log λ_s + log Γ(N_s+1)]

    No averaging; constants kept so absolute loss values are the proper NLL.
    """
    lam = predicted_charge
    N = observed_counts
    return jnp.sum(lam - N * jnp.log(lam + eps) + gammaln(N + 1.0))


def first_arrival_nll_gaussian_per_sensor(log_w, flat_times, flat_indices,
                                           t_obs_per_sensor, sigma_tts,
                                           num_detectors):
    """Per-sensor Gaussian-TTS first-arrival NLL: −log λ_s(t_obs) + Λ_s(t_obs).

    Inhomogeneous-Poisson first-arrival model with per-photon Gaussian TTS
    convolution at width σ_TTS. Numerically stable via segment-wise
    log-sum-exp for log λ. Uses a finite −1e6 floor on invalid entries
    (matching the pattern used by the existing first_arrival_nll); −inf
    breaks autograd through segment_max ties on empty / fully-padding
    sensors and produces NaN gradients.
    """
    t_obs_pp = t_obs_per_sensor[flat_indices]
    x = (t_obs_pp - flat_times) / sigma_tts
    log_K = -0.5 * x * x - 0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_tts)

    valid = log_w > -20.0
    safe_log_w = jnp.where(valid, log_w, -1e6)
    log_terms = safe_log_w + log_K

    # log λ via segment-wise log-sum-exp. Sensors with NO entries in
    # flat_indices (no photons routed to them) get exp_summed=0 →
    # ∂log/∂exp_summed = 1/0 = inf → 0×inf = NaN even when their
    # contribution is masked out downstream. Adding ε=1e-30 inside the log
    # bounds the gradient at 1/ε; this is invisible at non-empty sensors
    # (where exp_summed ≥ 1 by construction since the max term always
    # contributes 1 after shift).
    max_per = jax.ops.segment_max(
        log_terms, flat_indices, num_segments=num_detectors)
    shifted = log_terms - max_per[flat_indices]
    exp_summed = jax.ops.segment_sum(
        jnp.exp(shifted), flat_indices, num_segments=num_detectors)
    log_lambda = max_per + jnp.log(exp_summed + 1e-30)

    # Λ = Σ_i w_i · Φ((t_obs − t_i)/σ)
    F = jax.scipy.special.ndtr(x)
    safe_w = jnp.where(valid, jnp.exp(log_w), 0.0)
    Lam = jax.ops.segment_sum(
        safe_w * F, flat_indices, num_segments=num_detectors)

    return -log_lambda + Lam


def time_conditional_nll_per_sensor(log_w, flat_times, flat_indices,
                                     t_obs_per_sensor, sigma_tts,
                                     num_detectors, lambda_bg, total_charge,
                                     observed_counts):
    """Time-conditional NLL: −log P(t_obs | N_obs, model).

    NLL_time|N = −log N − log λ(t_obs) − (N−1)·log(λ_total − Λ(t_obs))
                 + N·log λ_total

    Uses continuous extension (option b) for fractional N_obs from gain
    smearing: gammaln/log just take fractional values directly.

    Returns 0 for sensors with N_obs == 0 (mask via where).
    """
    t_obs_pp = t_obs_per_sensor[flat_indices]
    x = (t_obs_pp - flat_times) / sigma_tts
    log_K = -0.5 * x * x - 0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_tts)
    valid = log_w > -20.0
    safe_log_w = jnp.where(valid, log_w, -1e6)
    log_terms = safe_log_w + log_K

    max_per = jax.ops.segment_max(
        log_terms, flat_indices, num_segments=num_detectors)
    shifted = log_terms - max_per[flat_indices]
    exp_summed = jax.ops.segment_sum(
        jnp.exp(shifted), flat_indices, num_segments=num_detectors)
    log_lambda_signal = max_per + jnp.log(exp_summed + 1e-30)
    log_lambda_obs = jnp.logaddexp(
        log_lambda_signal, jnp.log(jnp.maximum(lambda_bg, 1e-300)))

    F = jax.scipy.special.ndtr(x)
    safe_w = jnp.where(valid, jnp.exp(log_w), 0.0)
    Lam = jax.ops.segment_sum(
        safe_w * F, flat_indices, num_segments=num_detectors)

    log_lambda_total = jnp.log(total_charge + 1e-30)
    log_remaining = jnp.log(jnp.maximum(total_charge - Lam, 1e-30))

    log_N = jnp.log(jnp.maximum(observed_counts, 1e-10))
    n_minus_1 = jnp.maximum(observed_counts - 1.0, 0.0)

    time_cond = (-log_N - log_lambda_obs - n_minus_1 * log_remaining
                  + observed_counts * log_lambda_total)
    hit_mask = observed_counts > 0
    return jnp.where(hit_mask, time_cond, 0.0)


def charge_nll_per_sensor(total_charge, observed_counts, eps=1e-30):
    """Standard Poisson charge NLL per sensor with gammaln extension for
    fractional N (option b). Includes log(N!) so the absolute value is the
    proper NLL, not a constant-shifted version."""
    return (total_charge
             - observed_counts * jnp.log(total_charge + eps)
             + gammaln(observed_counts + 1.0))


def joint_poisson_nll_per_sensor(log_w, flat_times, flat_indices,
                                  t_obs_per_sensor, sigma_tts,
                                  num_detectors, lambda_bg, total_charge,
                                  observed_counts):
    """Per-sensor JOINT Poisson-process NLL.

    NLL_s = λ_total
          − [N_obs > 0] · log λ(t_obs)
          − max(N_obs − 1, 0) · log(λ_total − Λ(t_obs))

    The (N_obs − 1) factor naturally weights time information by sensor
    activity. For N_obs ≤ 1, the second term vanishes; for N_obs >> 1,
    it gives an N_obs× scaling on the rate-body term — the proper Fisher
    info weighting from the Poisson likelihood, not arbitrary reweighting.

    Includes UNHIT sensors via the λ_total Poisson-zero term — no mask
    needed at this level.
    """
    t_obs_pp = t_obs_per_sensor[flat_indices]
    x = (t_obs_pp - flat_times) / sigma_tts
    log_K = -0.5 * x * x - 0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_tts)

    valid = log_w > -20.0
    safe_log_w = jnp.where(valid, log_w, -1e6)
    log_terms = safe_log_w + log_K

    max_per = jax.ops.segment_max(
        log_terms, flat_indices, num_segments=num_detectors)
    shifted = log_terms - max_per[flat_indices]
    exp_summed = jax.ops.segment_sum(
        jnp.exp(shifted), flat_indices, num_segments=num_detectors)
    log_lambda_signal = max_per + jnp.log(exp_summed + 1e-30)

    log_lambda_obs = jnp.logaddexp(
        log_lambda_signal,
        jnp.log(jnp.maximum(lambda_bg, 1e-300)))

    F = jax.scipy.special.ndtr(x)
    safe_w = jnp.where(valid, jnp.exp(log_w), 0.0)
    Lam = jax.ops.segment_sum(
        safe_w * F, flat_indices, num_segments=num_detectors)

    # Remaining-rate after t_obs. Floor with ε so log doesn't blow up if
    # the model says all photons should have arrived already.
    remaining = jnp.maximum(total_charge - Lam, 1e-30)
    log_remaining = jnp.log(remaining)

    hit = (observed_counts > 0).astype(jnp.float32)
    n_minus_1 = jnp.maximum(observed_counts - 1.0, 0.0)

    return (total_charge
             - hit * log_lambda_obs
             - n_minus_1 * log_remaining)


def first_arrival_nll_gaussian_per_sensor_v2(log_w, flat_times, flat_indices,
                                              t_obs_per_sensor, sigma_tts,
                                              num_detectors, lambda_bg=0.0):
    """Same as v1 plus optional dark-noise floor λ_bg added to λ_signal."""
    t_obs_pp = t_obs_per_sensor[flat_indices]
    x = (t_obs_pp - flat_times) / sigma_tts
    log_K = -0.5 * x * x - 0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma_tts)

    valid = log_w > -20.0
    safe_log_w = jnp.where(valid, log_w, -1e6)
    log_terms = safe_log_w + log_K

    max_per = jax.ops.segment_max(
        log_terms, flat_indices, num_segments=num_detectors)
    shifted = log_terms - max_per[flat_indices]
    exp_summed = jax.ops.segment_sum(
        jnp.exp(shifted), flat_indices, num_segments=num_detectors)
    log_lambda_signal = max_per + jnp.log(exp_summed + 1e-30)

    # logaddexp handles lambda_bg=0 correctly: log(0)=-inf,
    # logaddexp(a, -inf) = a → no contribution from floor.
    log_lambda = jnp.logaddexp(
        log_lambda_signal,
        jnp.log(jnp.maximum(lambda_bg, 1e-300)))

    F = jax.scipy.special.ndtr(x)
    safe_w = jnp.where(valid, jnp.exp(log_w), 0.0)
    Lam = jax.ops.segment_sum(
        safe_w * F, flat_indices, num_segments=num_detectors)

    return -log_lambda + Lam


def make_loss_fn(prediction_simulator, num_detectors,
                  sigma_tts=SIGMA_TTS_NS):
    """Returns a single JIT'd loss that is parameterized at runtime by
    weights, threshold, dark-noise floor, and combiner choice.

    Combiner: when ``gmean_flag > 0.5`` use sqrt(c*t) instead of weighted
    sum. ``charge_weight`` / ``time_weight`` still gate which terms are
    active. JIT compiles ONCE; runtime scalars don't trigger recompile.
    """
    @jit
    def likelihood_loss(params, observed_times, observed_counts, key,
                         charge_weight, time_weight,
                         hit_threshold, lambda_bg, gmean_flag,
                         joint_flag=jnp.float32(0.0),
                         factored_flag=jnp.float32(0.0)):
        position = params[:3]
        t0 = params[3]
        theta = params[4]
        phi = params[5]
        energy = params[6]

        # E/sg(E) trick — inject analytical linear E gradient
        energy_frozen = jax.lax.stop_gradient(energy)
        energy_ratio = energy / energy_frozen

        track = ParticleParams(
            energy=energy_frozen, position=position,
            theta=theta, phi=phi, t0=jnp.array(0.0))
        log_w, flat_times, flat_indices, total_charge = prediction_simulator(
            track, key)

        log_w = log_w + jnp.log(energy_ratio)
        total_charge = total_charge * energy_ratio

        charge_loss = poisson_nll_sum(observed_counts, total_charge)

        t_obs_shifted = observed_times - t0
        per_sensor_time = first_arrival_nll_gaussian_per_sensor_v2(
            log_w, flat_times, flat_indices,
            t_obs_shifted, sigma_tts, num_detectors, lambda_bg)
        hit_mask = observed_counts >= hit_threshold
        time_loss = jnp.sum(jnp.where(hit_mask, per_sensor_time, 0.0))

        # Joint Poisson-process NLL: replaces charge + time entirely.
        per_sensor_joint = joint_poisson_nll_per_sensor(
            log_w, flat_times, flat_indices,
            t_obs_shifted, sigma_tts, num_detectors, lambda_bg,
            total_charge, observed_counts)
        joint_loss = jnp.sum(jnp.where(hit_mask, per_sensor_joint, 0.0))

        # Factored: charge_pure_event + time_cond_event (independent pieces of joint).
        per_sensor_charge_pure = charge_nll_per_sensor(
            total_charge, observed_counts)
        per_sensor_time_cond = time_conditional_nll_per_sensor(
            log_w, flat_times, flat_indices,
            t_obs_shifted, sigma_tts, num_detectors, lambda_bg,
            total_charge, observed_counts)
        # Charge term: SUM over ALL sensors (unhit included via Poisson zero)
        C_event = jnp.sum(per_sensor_charge_pure)
        # Time-conditional: SUM over hit sensors meeting threshold
        T_event = jnp.sum(jnp.where(hit_mask, per_sensor_time_cond, 0.0))

        # Factored modes: weights select which to use; gmean optional.
        factored_sum   = charge_weight * C_event + time_weight * T_event
        factored_gmean = jnp.sqrt(jnp.maximum(C_event * T_event, 1e-12))
        factored_form  = jnp.where(
            gmean_flag > 0.5, factored_gmean, factored_sum)

        # Original (uses first-arrival NLL, not time-conditional)
        sum_form = charge_weight * charge_loss + time_weight * time_loss
        gmean_form = jnp.sqrt(jnp.maximum(charge_loss * time_loss, 1e-12))
        sum_or_gmean = jnp.where(gmean_flag > 0.5, gmean_form, sum_form)

        # Selector chain: factored → joint → sum_or_gmean
        result_non_joint = jnp.where(
            factored_flag > 0.5, factored_form, sum_or_gmean)
        return jnp.where(joint_flag > 0.5, joint_loss, result_non_joint)

    return likelihood_loss


# (charge_weight, time_weight, gmean_flag, joint_flag, factored_flag)
_MODE_PARAMS = {
    'charge':       (1.0, 0.0, 0.0, 0.0, 0.0),
    'time':         (0.0, 1.0, 0.0, 0.0, 0.0),
    'sum':          (1.0, 1.0, 0.0, 0.0, 0.0),
    'gmean':        (1.0, 1.0, 1.0, 0.0, 0.0),
    'both':         (1.0, 1.0, 0.0, 0.0, 0.0),
    'joint':        (0.0, 0.0, 0.0, 1.0, 0.0),
    # Properly-factored joint = charge × time_conditional. Uses gammaln
    # for fractional N (option b).
    'charge_pure':  (1.0, 0.0, 0.0, 0.0, 1.0),
    'time_cond':    (0.0, 1.0, 0.0, 0.0, 1.0),
    'gmean_joint':  (1.0, 1.0, 1.0, 0.0, 1.0),
    # Factored sum: C_event + T_event (charge_pure + time_cond, summed,
    # no gmean). Equals the joint NLL up to a constant in N_obs.
    'factored_sum': (1.0, 1.0, 0.0, 0.0, 1.0),
}


# ---------------------------------------------------------------------------
# Event setup (mirrors parameter_scans_1D_likelihood.ipynb)
# ---------------------------------------------------------------------------
def build_event(detector, data_simulator, num_detectors, seed=SEED):
    key = jax.random.PRNGKey(seed)

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
    photon_data['N'] = n

    fraction = 0.6
    r_vert = jax.random.uniform(
        key, shape=(), minval=0, maxval=detector.r * fraction)
    key, _ = jax.random.split(key)
    theta_pos = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
    key, _ = jax.random.split(key)
    z_vert = jax.random.uniform(
        key, shape=(),
        minval=-detector.H/2 * fraction, maxval=detector.H/2 * fraction)
    true_position = jnp.array([
        r_vert * jnp.cos(theta_pos),
        r_vert * jnp.sin(theta_pos),
        z_vert,
    ])

    key, _ = jax.random.split(key)
    phi_dir = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
    key, _ = jax.random.split(key)
    cos_t = jax.random.uniform(key, shape=(), minval=-1, maxval=1)
    sin_t = jnp.sqrt(1 - cos_t**2)
    true_direction = jnp.array([
        sin_t * jnp.cos(phi_dir),
        sin_t * jnp.sin(phi_dir),
        cos_t,
    ])

    orig = jnp.array([0.0, 0.0, 1.0])
    td_n = true_direction / (jnp.linalg.norm(true_direction) + 1e-8)
    rotation_axis = jnp.cross(orig, td_n)
    axis_norm = jnp.linalg.norm(rotation_axis)
    rotation_axis = jnp.where(
        axis_norm < 1e-6,
        jnp.array([1.0, 0.0, 0.0]),
        rotation_axis / (axis_norm + 1e-8))
    rotation_angle = jnp.arccos(jnp.clip(jnp.dot(orig, td_n), -1.0, 1.0))

    photon_data['rotation_axis'] = rotation_axis
    photon_data['rotation_angle'] = rotation_angle
    photon_data['apply_rotation'] = jnp.array(True)
    photon_data['apply_translation'] = jnp.array(True)
    photon_data['translation_vector'] = true_position

    true_energy = photon_data['energy']
    true_track = ParticleParams.from_cartesian(
        energy=true_energy, position=true_position,
        direction=true_direction, t0=0.0)

    key, sub = jax.random.split(key)
    true_data = jax.lax.stop_gradient(
        data_simulator(true_track, sub, photon_data))

    true_theta, true_phi = cartesian_to_spherical(true_direction)
    return (true_track, true_data,
            float(true_position[0]), float(true_position[1]), float(true_position[2]),
            float(true_theta), float(true_phi), float(true_energy))


# ---------------------------------------------------------------------------
# Parameter scan
# ---------------------------------------------------------------------------
def make_loss_and_grad(loss_fn):
    """Single JIT'd value_and_grad wrapper. Pass arrays/scalars at call
    time so JIT compiles once for fixed array shapes."""
    @jit
    def loss_and_grad(params, hit_times, hit_counts, key,
                       cw, tw, thr, bg, gmean_flag, joint_flag,
                       factored_flag):
        def f(p):
            return loss_fn(p, hit_times, hit_counts, key,
                            cw, tw, thr, bg, gmean_flag, joint_flag,
                            factored_flag)
        return value_and_grad(f)(params)
    return loss_and_grad


def perform_scan(loss_and_grad, true_param_values, true_data,
                  param_name, param_idx, scan_range, n_points=N_SCAN_POINTS,
                  mode='both', hit_threshold=0.0, lambda_bg=0.0):
    hit_counts, hit_times = true_data
    scan_key = jax.random.PRNGKey(42)

    true_value = true_param_values[param_idx]
    param_values = jnp.linspace(
        true_value - scan_range, true_value + scan_range, n_points)

    cw, tw, gm, jt, fac = _MODE_PARAMS[mode]
    cw_j = jnp.float32(cw); tw_j = jnp.float32(tw)
    thr_j = jnp.float32(hit_threshold); bg_j = jnp.float32(lambda_bg)
    gm_j = jnp.float32(gm); jt_j = jnp.float32(jt); fac_j = jnp.float32(fac)

    losses, gradients = [], []
    t0 = _time.time()
    for v in param_values:
        params = jnp.array(true_param_values).at[param_idx].set(v)
        L, g = loss_and_grad(params, hit_times, hit_counts, scan_key,
                               cw_j, tw_j, thr_j, bg_j,
                               gm_j, jt_j, fac_j)
        losses.append(float(L))
        gradients.append(float(g[param_idx]))
    print(f'  {param_name}: {_time.time() - t0:.2f} s')

    return {
        'name': param_name,
        'values': np.array(param_values),
        'true': float(true_value),
        'losses': np.array(losses),
        'gradients': np.array(gradients),
    }


def plot_scans(results, sigma_tts=SIGMA_TTS_NS, out_path=None):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.6 * n))
    if n == 1:
        axes = axes[None, :]

    for i, r in enumerate(results):
        ax_l, ax_g = axes[i, 0], axes[i, 1]
        ax_l.plot(r['values'], r['losses'], color='navy', linewidth=1.7)
        ax_l.axvline(r['true'], color='crimson', linestyle='--', linewidth=1.4,
                     label=f'true = {r["true"]:.4g}')
        ax_l.set_title(f'{r["name"]} — loss')
        ax_l.set_xlabel(r['name'])
        ax_l.set_ylabel('NLL')
        ax_l.grid(True, alpha=0.3)
        ax_l.legend(fontsize=8, loc='best')

        ax_g.plot(r['values'], r['gradients'], color='darkgreen', linewidth=1.7)
        ax_g.axvline(r['true'], color='crimson', linestyle='--', linewidth=1.4,
                     label=f'true = {r["true"]:.4g}')
        ax_g.axhline(0, color='gray', linestyle=':', linewidth=1.0)
        ax_g.set_title(f'{r["name"]} — gradient')
        ax_g.set_xlabel(r['name'])
        ax_g.set_ylabel(f'∂NLL / ∂{r["name"]}')
        ax_g.grid(True, alpha=0.3)
        ax_g.legend(fontsize=8, loc='best')

    fig.suptitle(
        f'Gaussian-TTS first-arrival NLL  +  Poisson charge NLL '
        f'(σ_TTS = {sigma_tts:.2f} ns) — sum form, no vertex term',
        fontsize=12, y=1.0)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    detector = generate_detector(DEFAULT_JSON_FILENAME)
    num_detectors = len(detector.all_points)
    print(f'Detectors: {num_detectors}')

    # temperature=None → hard sensor assignment, integer photon counts
    data_sim = setup_event_simulator(
        DEFAULT_JSON_FILENAME, NPHOT_DATA, temperature=None, K=K_DATA,
        is_data=True, is_calibration=False, apply_smearing=True,
        physics_config=PHYSICS_CONFIG, default_detector_params=True)
    pred_sim = setup_event_simulator(
        DEFAULT_JSON_FILENAME, NPHOT_PRED, TEMPERATURE,
        max_sensors_per_cell=4, K=K_PRED,
        is_data=False, hit_mode='per_photon',
        physics_config=PHYSICS_CONFIG, default_detector_params=True)

    print('Building event ...')
    (true_track, true_data, x, y, z, theta, phi, energy) = build_event(
        detector, data_sim, num_detectors)
    n_hit = int(np.sum(np.asarray(true_data[0]) > 0))
    print(f'Event: E = {energy:.1f} MeV, hit sensors = {n_hit} / {num_detectors}')
    print(f'  NPHOT_PRED = {NPHOT_PRED}, NPHOT_DATA = {NPHOT_DATA}, '
          f'K_PRED = {K_PRED}, K_DATA = {K_DATA}')

    true_param_values = [x, y, z, 0.0, theta, phi, energy]
    print(f'True parameters: x={x:.3f}, y={y:.3f}, z={z:.3f}, '
          f't0=0.0, theta={theta:.3f}, phi={phi:.3f}, E={energy:.1f}')

    scan_configs = [
        ('X',     0, 0.5),
        ('Y',     1, 0.5),
        ('Z',     2, 0.5),
        ('t0',    3, 5.0),
        ('theta', 4, 0.3),
        ('phi',   5, 0.3),
        ('E',     6, 100.0),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HIT_THRESHOLD = 2.5  # PE
    LAMBDA_BG = 0.0       # turn off floor; the threshold already kills outliers

    # Single JIT'd loss + value_and_grad wrapper. Both compile ONCE.
    # Runtime scalar args (mode weights, threshold, λ_bg) don't trigger
    # recompile.
    loss_fn = make_loss_fn(pred_sim, num_detectors)
    loss_and_grad = make_loss_and_grad(loss_fn)

    # Warmup once
    _t0 = _time.time()
    hit_counts, hit_times = true_data
    _ = loss_and_grad(
        jnp.array(true_param_values), hit_times, hit_counts,
        jax.random.PRNGKey(42),
        jnp.float32(1.0), jnp.float32(1.0),
        jnp.float32(HIT_THRESHOLD), jnp.float32(LAMBDA_BG),
        jnp.float32(0.0))
    print(f'[JIT warmup] {_time.time() - _t0:.2f} s')

    for mode_label in ('time', 'charge', 'gmean'):
        print(f'\n=== Scan mode={mode_label}, threshold ≥{HIT_THRESHOLD} PE, '
              f'λ_bg={LAMBDA_BG} ===')

        results = []
        for name, idx, rng in scan_configs:
            r = perform_scan(loss_and_grad, true_param_values, true_data,
                              name, idx, rng,
                              mode=mode_label,
                              hit_threshold=HIT_THRESHOLD,
                              lambda_bg=LAMBDA_BG)
            i_min = int(np.argmin(r['losses']))
            v_min = r['values'][i_min]
            offset = v_min - r['true']
            print(f'    min at {name}={v_min:.4g}  '
                  f'(true={r["true"]:.4g}, offset={offset:+.4g})  '
                  f'grad@true={r["gradients"][len(r["gradients"])//2]:+.3e}')
            results.append(r)

        out = OUT_DIR / f'all_params_scan_{mode_label}_thr{HIT_THRESHOLD:.1f}.png'
        plot_scans(results, out_path=str(out))
        print(f'  Saved: {out}')


if __name__ == '__main__':
    main()
