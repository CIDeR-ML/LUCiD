"""Multi-event bias study for the joint Poisson-process loss.

For each of N events (vertex/direction varied by seed; particle kinematics
fixed by PhotonSim entry):

  - 1D-scan each of (X, Y, Z, t0, theta, phi, E) around its true value
  - Record per-event:
      argmin_offset    — argmin(loss) on the grid, minus true
      grad_zero_offset — interpolated zero-crossing of ∂L/∂θ closest to 0
      grad_slope       — finite-diff slope of the gradient near 0
                         (≈ Hessian element, indicates well stiffness)

Aggregates across events: mean, std, median, IQR per (param, mode, threshold).
Plots per-param histograms of the gradient zero-crossing offset (the bias).

Sample defaults: matches the conventions in parameter_scans_1D_gaussian
(σ_TTS = 2.5 ns, T_pred = 0.10, K_pred = 7, NPHOT_pred = 1M; data hard
sampled at NPHOT = 150k, K = 20).
"""
from __future__ import annotations
import sys, time, pickle, argparse
sys.path.append('..')

import jax, jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from parameter_scans_1D_gaussian import (
    DEFAULT_JSON_FILENAME, PHYSICS_CONFIG, K_PRED, K_DATA,
    NPHOT_PRED, NPHOT_DATA, SIGMA_TTS_NS, DATA_FILE,
    make_loss_fn, make_loss_and_grad, perform_scan, _MODE_PARAMS,
)
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.generate import read_photon_data_from_photonsim
from lucid.detector_params import ParticleParams
from lucid.optimization.utils.functions import cartesian_to_spherical


def build_event_for_entry(detector, data_sim, num_detectors, entry_idx, seed):
    """Like parameter_scans_1D_gaussian.build_event but takes the
    PhotonSim entry index as a parameter (was a module-level constant)."""
    key = jax.random.PRNGKey(seed)
    photon_data = read_photon_data_from_photonsim(DATA_FILE, entry_idx)
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
             jnp.tile(jnp.array([0., 0., 1.]), (pad, 1))], axis=0)
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

    z_hat = jnp.array([0., 0., 1.])
    td_n = true_direction / (jnp.linalg.norm(true_direction) + 1e-8)
    rotation_axis = jnp.cross(z_hat, td_n)
    axis_norm = jnp.linalg.norm(rotation_axis)
    rotation_axis = jnp.where(
        axis_norm < 1e-6,
        jnp.array([1.0, 0.0, 0.0]),
        rotation_axis / (axis_norm + 1e-8))
    rotation_angle = jnp.arccos(jnp.clip(jnp.dot(z_hat, td_n), -1.0, 1.0))

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
        data_sim(true_track, sub, photon_data))
    true_theta, true_phi = cartesian_to_spherical(true_direction)
    return (true_track, true_data,
            float(true_position[0]), float(true_position[1]),
            float(true_position[2]),
            float(true_theta), float(true_phi), float(true_energy))


SCAN_SPECS = [
    ('X',     0, 0.5),
    ('Y',     1, 0.5),
    ('Z',     2, 0.5),
    ('t0',    3, 5.0),
    ('theta', 4, 0.3),
    ('phi',   5, 0.3),
    ('E',     6, 100.0),
]
PRED_TEMPERATURE = 0.10
DATA_TEMPERATURE = None  # hard sampling


def make_loss_fn_e_charge_only(prediction_simulator, num_detectors,
                                  sigma_tts=SIGMA_TTS_NS):
    """Joint Poisson-process loss with E gradient applied ONLY to the charge
    term. Time-channel quantities (λ(t_obs), Λ(t_obs)) are computed from a
    log_w that has no E gradient at all.

    Signature mirrors parameter_scans_1D_gaussian.make_loss_fn so the same
    grad_fn / secant_solve / mode flags apply. Only modes affected:
        joint:        leading λ_total has dE; everything else frozen.
        gmean_joint:  C_event has dE; T_event uses frozen log_w + frozen λ_total.
        charge_pure / time_cond: same split.
    Older modes (sum, gmean, both, etc.) fall through to the factored sum.
    """
    from jax.scipy.special import gammaln as _gammaln
    from lucid.detector_params import ParticleParams as _PP

    @jax.jit
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

        # E/sg(E) trick — but we'll only INJECT the gradient into the
        # charge term; log_w stays frozen w.r.t. E.
        energy_frozen = jax.lax.stop_gradient(energy)
        energy_ratio = energy / energy_frozen

        track = _PP(
            energy=energy_frozen, position=position,
            theta=theta, phi=phi, t0=jnp.array(0.0))
        log_w_frozen, flat_times, flat_indices, total_charge_frozen = (
            prediction_simulator(track, key))
        # log_w_frozen and total_charge_frozen have NO E gradient.

        # Charge-side: inject linear E gradient
        total_charge_with_E = total_charge_frozen * energy_ratio

        t_obs_shifted = observed_times - t0

        # ---- time-rate quantities computed from log_w_frozen (no dE) ----
        t_obs_pp = t_obs_shifted[flat_indices]
        x = (t_obs_pp - flat_times) / sigma_tts
        log_K = (-0.5 * x * x
                  - 0.5 * jnp.log(2.0 * jnp.pi)
                  - jnp.log(sigma_tts))
        valid = log_w_frozen > -20.0
        safe_log_w = jnp.where(valid, log_w_frozen, -1e6)
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
        safe_w = jnp.where(valid, jnp.exp(log_w_frozen), 0.0)
        Lam = jax.ops.segment_sum(
            safe_w * F, flat_indices, num_segments=num_detectors)

        hit = (observed_counts > 0).astype(jnp.float32)
        n_minus_1 = jnp.maximum(observed_counts - 1.0, 0.0)
        hit_mask = observed_counts >= hit_threshold

        # ---- JOINT NLL: dE only on the leading λ_total ----
        # Leading: total_charge_with_E (has dE)
        # Time:    -hit * log λ(t_obs)         (frozen → no dE)
        # Resid:   -(N-1) * log(λ_total_frozen - Λ)  (frozen → no dE)
        log_remaining_frozen = jnp.log(
            jnp.maximum(total_charge_frozen - Lam, 1e-30))
        per_sensor_joint = (total_charge_with_E
                              - hit * log_lambda_obs
                              - n_minus_1 * log_remaining_frozen)
        joint_loss = jnp.sum(jnp.where(hit_mask, per_sensor_joint, 0.0))

        # ---- FACTORED form ----
        eps = 1e-30
        # C_event: charge NLL with dE
        per_sensor_charge_pure = (
            total_charge_with_E
            - observed_counts * jnp.log(total_charge_with_E + eps)
            + _gammaln(observed_counts + 1.0))
        C_event = jnp.sum(per_sensor_charge_pure)

        # T_event: time-conditional NLL with frozen quantities
        log_lambda_total_frozen = jnp.log(total_charge_frozen + 1e-30)
        log_N = jnp.log(jnp.maximum(observed_counts, 1e-10))
        per_sensor_time_cond_frozen = (
            -log_N - log_lambda_obs
            - n_minus_1 * log_remaining_frozen
            + observed_counts * log_lambda_total_frozen)
        per_sensor_time_cond = jnp.where(
            observed_counts > 0, per_sensor_time_cond_frozen, 0.0)
        T_event = jnp.sum(jnp.where(hit_mask, per_sensor_time_cond, 0.0))

        factored_sum = charge_weight * C_event + time_weight * T_event
        factored_gmean = jnp.sqrt(jnp.maximum(C_event * T_event, 1e-12))
        factored_form = jnp.where(
            gmean_flag > 0.5, factored_gmean, factored_sum)

        # Selector chain: factored → joint → factored_sum (default)
        result_non_joint = jnp.where(
            factored_flag > 0.5, factored_form, factored_sum)
        return jnp.where(joint_flag > 0.5, joint_loss, result_non_joint)

    return likelihood_loss


def find_grad_zero_crossing(values, gradients, true_value):
    """Linear-interpolate the offset (param − true) where ∂L/∂θ = 0,
    picking the crossing closest to 0.

    Returns:
        zero_offset (float, NaN if no crossing)
        slope (float, NaN if no crossing) — ∂g/∂θ at the crossing
    """
    offsets = values - true_value
    g = np.asarray(gradients)
    o = np.asarray(offsets)

    sign = np.sign(g)
    transitions = np.where(np.diff(sign) != 0)[0]
    if len(transitions) == 0:
        return np.nan, np.nan

    candidates = []
    for i in transitions:
        x0, x1 = o[i], o[i + 1]
        g0, g1 = g[i], g[i + 1]
        if g1 == g0:
            continue
        x_zero = x0 - g0 * (x1 - x0) / (g1 - g0)
        slope = (g1 - g0) / (x1 - x0)
        candidates.append((x_zero, slope))
    if not candidates:
        return np.nan, np.nan

    # Pick the crossing closest to 0 (true value)
    candidates.sort(key=lambda c: abs(c[0]))
    return candidates[0]


def summarize(arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return dict(mean=np.nan, std=np.nan, median=np.nan,
                    iqr=np.nan, n=0)
    return dict(
        mean=float(np.mean(a)), std=float(np.std(a, ddof=1) if len(a) > 1 else 0.0),
        median=float(np.median(a)),
        iqr=float(np.percentile(a, 75) - np.percentile(a, 25)),
        n=int(len(a)),
    )


def make_grad_fn(loss_fn):
    """JIT'd 1D first-derivative dL/dα along direction `e`."""
    def L1d(alpha, base_params, e, hit_times, hit_counts, key,
             cw, tw, thr, bg, gm, jt, fac):
        params = base_params + alpha * e
        return loss_fn(params, hit_times, hit_counts, key,
                        cw, tw, thr, bg, gm, jt, fac)
    g1d = jax.grad(L1d, argnums=0)

    @jax.jit
    def grad_fn(alpha, base_params, e, hit_times, hit_counts, key,
                 cw, tw, thr, bg, gm, jt, fac):
        return g1d(alpha, base_params, e, hit_times, hit_counts, key,
                    cw, tw, thr, bg, gm, jt, fac)
    return grad_fn


def secant_solve(grad_fn, base_params, idx, hit_times, hit_counts, key,
                  flags, max_iter=8, tol_g=1e-3, tol_alpha=1e-5,
                  scale=1.0, eps0=None, noise_patience=2):
    """Secant on g(α) = ∂L/∂α. Returns (alpha, g_final, slope_estimate, n_iter).

    Uses noise-floor early stopping: if best |g| stops improving for
    `noise_patience` iterations, return the α that gave the smallest |g|.
    """
    e = jnp.zeros(7, dtype=jnp.float32).at[idx].set(1.0)
    cw, tw, thr, bg, gm, jt, fac = flags
    eps = eps0 if eps0 is not None else 0.05 * scale

    a_prev = jnp.float32(0.0)
    g_prev = float(grad_fn(a_prev, base_params, e, hit_times, hit_counts,
                             key, cw, tw, thr, bg, gm, jt, fac))
    a_cur = jnp.float32(eps)
    g_cur = float(grad_fn(a_cur, base_params, e, hit_times, hit_counts,
                            key, cw, tw, thr, bg, gm, jt, fac))

    best_alpha = float(a_cur) if abs(g_cur) < abs(g_prev) else float(a_prev)
    best_abs_g = min(abs(g_cur), abs(g_prev))
    no_improve = 0
    n_iter = 2
    last_slope = (g_cur - g_prev) / (float(a_cur) - float(a_prev) + 1e-30)

    for it in range(2, max_iter + 2):
        n_iter = it + 1
        if not np.isfinite(g_cur):
            break
        denom = g_cur - g_prev
        if abs(denom) < 1e-20:
            break
        delta = -g_cur * (float(a_cur) - float(a_prev)) / denom
        max_step = 0.5 * scale
        if abs(delta) > max_step:
            delta = np.sign(delta) * max_step
        a_next = float(a_cur) + delta

        a_prev, g_prev = a_cur, g_cur
        a_cur = jnp.float32(a_next)
        g_cur = float(grad_fn(a_cur, base_params, e, hit_times, hit_counts,
                                key, cw, tw, thr, bg, gm, jt, fac))
        last_slope = (g_cur - g_prev) / (float(a_cur) - float(a_prev) + 1e-30)
        if abs(g_cur) < best_abs_g:
            best_alpha = float(a_cur)
            best_abs_g = abs(g_cur)
            no_improve = 0
        else:
            no_improve += 1
        if abs(g_cur) < tol_g and abs(delta) < tol_alpha * scale:
            best_alpha = float(a_cur)
            break
        if no_improve >= noise_patience:
            break
    return best_alpha, best_abs_g, float(last_slope), n_iter


def run_one_event_secant(grad_fn, base_params, true_data, hit_threshold,
                           lambda_bg, mode, key=jax.random.PRNGKey(42)):
    """Returns the same per-param dict as run_one_event but via secant.
    No grid arrays — only argmin/zero offset are populated; the keys
    `values`, `losses`, `gradients` are left empty for compatibility."""
    cw, tw, gm, jt, fac = _MODE_PARAMS[mode]
    flags = (jnp.float32(cw), jnp.float32(tw),
              jnp.float32(hit_threshold), jnp.float32(lambda_bg),
              jnp.float32(gm), jnp.float32(jt), jnp.float32(fac))
    hit_counts, _hit_times_true, hit_times = true_data
    out = {}
    for name, idx, scan_rng in SCAN_SPECS:
        alpha, g_final, slope, n_iter = secant_solve(
            grad_fn, base_params, idx, hit_times, hit_counts, key,
            flags, scale=scan_rng)
        out[name] = {
            'true': float(base_params[idx]),
            'values': np.array([]),
            'losses': np.array([]),
            'gradients': np.array([]),
            'argmin_offset': float('nan'),  # not computed in secant mode
            'grad_zero_offset': float(alpha),
            'grad_slope_at_zero': float(slope),
            'range': float('nan'),
            'n_iter': int(n_iter),
            'g_final': float(g_final),
        }
    return out


def run_one_event(loss_and_grad, true_param_values, true_data,
                   mode, hit_threshold, lambda_bg, n_scan):
    out = {}
    for name, idx, scan_rng in SCAN_SPECS:
        r = perform_scan(loss_and_grad, true_param_values, true_data,
                          name, idx, scan_rng, n_points=n_scan,
                          mode=mode, hit_threshold=hit_threshold,
                          lambda_bg=lambda_bg)
        i_min = int(np.argmin(r['losses']))
        argmin_offset = float(r['values'][i_min] - r['true'])
        zero_off, slope = find_grad_zero_crossing(
            r['values'], r['gradients'], r['true'])
        rng_loss = float(r['losses'].max() - r['losses'].min())
        out[name] = {
            'true': r['true'],
            'values': r['values'],
            'losses': r['losses'],
            'gradients': r['gradients'],
            'argmin_offset': argmin_offset,
            'grad_zero_offset': float(zero_off),
            'grad_slope_at_zero': float(slope),
            'range': rng_loss,
        }
    return out


def plot_bias_histograms(per_param_zeros, mode, hit_threshold,
                          lambda_bg, n_events, out_path):
    n_p = len(SCAN_SPECS)
    fig, axes = plt.subplots(n_p, 1, figsize=(9, 1.7 * n_p), squeeze=False)
    axes = axes.ravel()
    for ax, (name, _, _) in zip(axes, SCAN_SPECS):
        vals = np.asarray(per_param_zeros[name], dtype=float)
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            ax.text(0.5, 0.5, 'no zero crossings', ha='center',
                     va='center', transform=ax.transAxes)
            ax.set_title(f'{name}', fontsize=10)
            continue
        ax.hist(finite, bins=max(8, len(finite) // 3), color='navy',
                 alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(0.0, color='red', linestyle='--', linewidth=1.0,
                    label='unbiased')
        ax.axvline(np.mean(finite), color='darkorange', linestyle='-',
                    linewidth=1.5,
                    label=f'mean = {np.mean(finite):+.4f}')
        ax.axvline(np.median(finite), color='green', linestyle=':',
                    linewidth=1.5,
                    label=f'median = {np.median(finite):+.4f}')
        ax.set_title(
            f'{name}  •  N = {len(finite)}/{n_events}  •  '
            f'std = {np.std(finite, ddof=1) if len(finite) > 1 else 0:.4f}',
            fontsize=10)
        ax.set_xlabel(f'∂L/∂{name} = 0  offset from true')
        ax.set_ylabel('events')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

    fig.suptitle(
        f'mode={mode}  •  thr ≥ {hit_threshold:.1f} PE  •  '
        f'λ_bg = {lambda_bg:.0e}  •  {n_events} events',
        fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def print_summary_table(per_param, mode, hit_threshold, lambda_bg, n_events):
    print()
    print('=' * 110)
    print(f' SUMMARY  •  mode={mode}  •  thr ≥ {hit_threshold:.1f} PE  •  '
          f'λ_bg = {lambda_bg:.0e}  •  {n_events} events')
    print('=' * 110)
    hdr = f'{"param":>6}'
    for s in ('mean', 'std', 'median', 'iqr'):
        hdr += f' | argmin_{s:>6}  zero_{s:>6}  slope_{s:>6}'
    hdr += f' | n'
    print(hdr)
    print('-' * len(hdr))

    for name, _, _ in SCAN_SPECS:
        s_arg = summarize([e['argmin_offset'] for e in per_param[name]])
        s_zer = summarize([e['grad_zero_offset'] for e in per_param[name]])
        s_slo = summarize([e['grad_slope_at_zero'] for e in per_param[name]])
        row = f'{name:>6}'
        for k in ('mean', 'std', 'median', 'iqr'):
            row += (f' | {s_arg[k]:+10.4f}  {s_zer[k]:+10.4f}  '
                    f'{s_slo[k]:+10.4f}')
        row += f' | {s_zer["n"]}'
        print(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-events', type=int, default=20)
    p.add_argument('--lambda-bg', type=float, default=1e-3)
    p.add_argument('--hit-threshold', type=float, default=1.0)
    p.add_argument('--n-scan', type=int, default=41)
    p.add_argument('--mode', type=str, default='joint',
                    help='one of: joint, charge_pure, time_cond, gmean_joint, '
                         'charge, time, sum, gmean')
    p.add_argument('--seed-base', type=int, default=44,
                    help='vertex/direction seed; per-event seed = seed_base + ev')
    p.add_argument('--start-entry', type=int, default=0,
                    help='per-event entry_idx = start_entry + ev')
    p.add_argument('--solver', type=str, default='secant',
                    choices=['scan', 'secant'])
    p.add_argument('--scan-params', type=str, default='X,Y,Z,t0,theta,phi,E',
                    help='comma-separated subset of {X,Y,Z,t0,theta,phi,E} '
                         'to scan; default scans all 7')
    p.add_argument('--wavelength-mode', type=str, default='off',
                    choices=['on', 'off'],
                    help='wavelength_mode for both sims (default off — uses '
                         'scalar absorption/scatter and scalar QE)')
    p.add_argument('--e-charge-only', action='store_true',
                    help='use loss with E gradient injected only into charge term '
                         '(time-channel sees no E gradient at all)')
    p.add_argument('--out-dir', type=str,
                    default='figures/multi_event_bias')
    p.add_argument('--tag', type=str, default='joint')
    args = p.parse_args()

    # Filter SCAN_SPECS by --scan-params
    requested = [s.strip() for s in args.scan_params.split(',')]
    global_specs = {s[0]: s for s in SCAN_SPECS}
    unknown = [r for r in requested if r not in global_specs]
    if unknown:
        raise SystemExit(f'unknown scan params: {unknown}; valid: {list(global_specs)}')
    scan_specs = [global_specs[r] for r in requested]
    # Replace module-level SCAN_SPECS used by run_one_event* and plotting.
    globals()['SCAN_SPECS'] = scan_specs

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'mode = {args.mode}')
    print(f'λ_bg = {args.lambda_bg:.0e}, hit_threshold = {args.hit_threshold:.2f}')
    print(f'n_events = {args.n_events}, n_scan = {args.n_scan}')
    print(f'data: NPHOT={NPHOT_DATA} K={K_DATA} T={DATA_TEMPERATURE} (hard)')
    print(f'pred: NPHOT={NPHOT_PRED} K={K_PRED} T={PRED_TEMPERATURE}')
    print(f'σ_TTS = {SIGMA_TTS_NS}')

    detector = generate_detector(DEFAULT_JSON_FILENAME)
    num_detectors = len(detector.all_points)

    wlmode = (args.wavelength_mode == 'on')
    print(f'wavelength_mode = {wlmode}')
    data_sim = setup_event_simulator(
        DEFAULT_JSON_FILENAME, NPHOT_DATA,
        temperature=DATA_TEMPERATURE, K=K_DATA,
        is_data=True, is_calibration=False, apply_smearing=True,
        wavelength_mode=wlmode,
        physics_config=PHYSICS_CONFIG, default_detector_params=True)
    pred_sim = setup_event_simulator(
        DEFAULT_JSON_FILENAME, NPHOT_PRED, PRED_TEMPERATURE,
        max_candidates_per_ray=4, K=K_PRED,
        is_data=False, hit_mode='per_photon',
        wavelength_mode=wlmode,
        physics_config=PHYSICS_CONFIG, default_detector_params=True)

    if args.e_charge_only:
        loss_fn = make_loss_fn_e_charge_only(pred_sim, num_detectors)
        print('loss = joint with E gradient applied ONLY to charge term '
              '(time-channel frozen w.r.t. E)')
    else:
        loss_fn = make_loss_fn(pred_sim, num_detectors)
        print('loss = joint default (E gradient flows through both charge AND time)')

    loss_and_grad = make_loss_and_grad(loss_fn)
    grad_fn = make_grad_fn(loss_fn) if args.solver == 'secant' else None

    print(f'solver = {args.solver}')

    # Per-param accumulators
    per_param = {name: [] for name, _, _ in SCAN_SPECS}

    # JIT warmup with the first event
    print('\nJIT warmup ...')
    t0 = time.time()
    (true_track, true_data, x, y, z, theta, phi, energy) = build_event_for_entry(
        detector, data_sim, num_detectors,
        entry_idx=args.start_entry, seed=args.seed_base)
    true_param_values = [x, y, z, 0.0, theta, phi, energy]
    hit_counts, _hit_times_true, hit_times = true_data
    _ = loss_and_grad(
        jnp.array(true_param_values), hit_times, hit_counts,
        jax.random.PRNGKey(42),
        jnp.float32(1.0), jnp.float32(1.0),
        jnp.float32(args.hit_threshold), jnp.float32(args.lambda_bg),
        jnp.float32(0.0), jnp.float32(1.0), jnp.float32(0.0))
    print(f'  loss_and_grad warmup: {time.time() - t0:.1f}s')
    if grad_fn is not None:
        t0 = time.time()
        _ = grad_fn(
            jnp.float32(0.0),
            jnp.array(true_param_values, dtype=jnp.float32),
            jnp.zeros(7, dtype=jnp.float32).at[0].set(1.0),
            hit_times, hit_counts, jax.random.PRNGKey(42),
            jnp.float32(1.0), jnp.float32(1.0),
            jnp.float32(args.hit_threshold), jnp.float32(args.lambda_bg),
            jnp.float32(0.0), jnp.float32(1.0), jnp.float32(0.0))
        print(f'  grad_fn warmup: {time.time() - t0:.1f}s')

    raw_events = []
    for ev in range(args.n_events):
        entry_idx = args.start_entry + ev
        seed = args.seed_base + ev
        print(f'\n=== event {ev + 1}/{args.n_events}  '
              f'(entry={entry_idx}, seed={seed}) ===')
        t_ev = time.time()

        (true_track, true_data, x, y, z, theta, phi, energy) = build_event_for_entry(
            detector, data_sim, num_detectors,
            entry_idx=entry_idx, seed=seed)
        true_param_values = [x, y, z, 0.0, theta, phi, energy]
        n_hit = int(np.sum(np.asarray(true_data[0]) > 0))
        print(f'  E={energy:.1f}  pos=({x:+.2f},{y:+.2f},{z:+.2f})  '
              f'θ={theta:+.3f} φ={phi:+.3f}  n_hit={n_hit}')

        if args.solver == 'secant':
            ev_out = run_one_event_secant(
                grad_fn,
                jnp.array(true_param_values, dtype=jnp.float32),
                true_data,
                hit_threshold=args.hit_threshold,
                lambda_bg=args.lambda_bg, mode=args.mode)
        else:
            ev_out = run_one_event(
                loss_and_grad, true_param_values, true_data,
                mode=args.mode, hit_threshold=args.hit_threshold,
                lambda_bg=args.lambda_bg, n_scan=args.n_scan)

        # Print the per-event biases inline
        if args.solver == 'secant':
            print(f'  {"param":>6}  {"zero_grad":>10}  {"slope":>12}  '
                  f'{"|g_final|":>10}  {"iters":>5}')
            for name, _, _ in SCAN_SPECS:
                r = ev_out[name]
                per_param[name].append(r)
                print(f'  {name:>6}  {r["grad_zero_offset"]:+10.4f}  '
                      f'{r["grad_slope_at_zero"]:+12.4f}  '
                      f'{r.get("g_final", float("nan")):10.4f}  '
                      f'{r.get("n_iter", 0):5d}')
        else:
            print(f'  {"param":>6}  {"argmin":>10}  {"zero_grad":>10}  '
                  f'{"slope":>10}  {"range":>9}')
            for name, _, _ in SCAN_SPECS:
                r = ev_out[name]
                per_param[name].append(r)
                print(f'  {name:>6}  {r["argmin_offset"]:+10.4f}  '
                      f'{r["grad_zero_offset"]:+10.4f}  '
                      f'{r["grad_slope_at_zero"]:+10.4f}  '
                      f'{r["range"]:9.1f}')
        raw_events.append({
            'event_idx': ev,
            'entry_idx': entry_idx,
            'seed': seed,
            'true_param_values': true_param_values,
            'n_hit': n_hit,
            'scans': ev_out,
        })
        print(f'  event time: {time.time() - t_ev:.1f}s')

    # ----- Aggregate / report / save -----
    print_summary_table(per_param, args.mode, args.hit_threshold,
                         args.lambda_bg, args.n_events)

    per_param_zeros = {
        name: [e['grad_zero_offset'] for e in per_param[name]]
        for name, _, _ in SCAN_SPECS}

    suffix_loss = '_Echargeonly' if args.e_charge_only else ''
    fig_path = out_dir / (
        f'bias_hist_{args.tag}{suffix_loss}_{args.solver}'
        f'_thr{args.hit_threshold:.1f}'
        f'_lbg{args.lambda_bg:.0e}_n{args.n_events}.png')
    plot_bias_histograms(per_param_zeros, args.mode, args.hit_threshold,
                          args.lambda_bg, args.n_events, fig_path)
    print(f'\nsaved figure: {fig_path}')

    pkl_path = out_dir / (
        f'bias_data_{args.tag}{suffix_loss}_{args.solver}'
        f'_thr{args.hit_threshold:.1f}'
        f'_lbg{args.lambda_bg:.0e}_n{args.n_events}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'raw_events': raw_events,
            'mode': args.mode,
            'solver': args.solver,
            'e_charge_only': args.e_charge_only,
            'hit_threshold': args.hit_threshold,
            'lambda_bg': args.lambda_bg,
            'n_scan': args.n_scan,
            'n_events': args.n_events,
            'sigma_tts': SIGMA_TTS_NS,
        }, f)
    print(f'saved pickle: {pkl_path}')


if __name__ == '__main__':
    main()
