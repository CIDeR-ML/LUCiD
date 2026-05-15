"""Per-sensor empirical-vs-analytical first-arrival validation.

Goal: verify that the analytical p_first(t) we use to build the
likelihood actually matches the empirical t_obs distribution produced by
the data simulator. If they agree (modulo MC noise) then the assumptions
behind the loss design are self-consistent:

  - per-photon Gaussian TTS at σ_TTS_singlePE
  - segment_min as the per-sensor first-arrival operator
  - inhomogeneous-Poisson approximation
  - Gaussian convolution kernel in the loss

Pipeline
--------
  1. Run the prediction simulator ONCE on the chosen true event →
     per-photon (log_w, t, indices) at the prediction simulator's
     soft-temperature settings.
  2. Run the JIT'd diagnostics evaluator → per-sensor analytical
     p_first(t) curve, mode, support, etc.
  3. Run the data simulator MANY times on the *same* true event with
     different RNG keys → many realizations of (per-sensor charge, t_obs)
     under the data simulator's per-photon TTS path.
  4. For each selected sensor, overlay the empirical histogram of t_obs
     (across trials, conditioned on hit) against the analytical
     p_first(t) (normalized by the hit probability, 1 - exp(-Λ_total),
     so both curves are conditional pdfs).

Run:
    cd good_notebooks && python per_sensor_p_first_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append('..')

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from per_sensor_time_overlay import (
    SIGMA_TTS_NS, N_PER_STRATUM, SEED,
    build_simulators, build_event,
    make_diagnostics_evaluator,
    stratified_sensor_selection,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRIALS = 500
OUT_DIR = Path('figures/per_sensor_validation')

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['savefig.dpi'] = 200


# ---------------------------------------------------------------------------
# Data trial loop
# ---------------------------------------------------------------------------
def run_data_trials(data_simulator, true_track, photon_data,
                    master_key, n_trials):
    """Call the (JIT'd) data simulator n_trials times with fresh keys.

    Returns:
        all_hit_counts : (n_trials, num_sensors)  per-sensor charge per trial
        all_hit_times  : (n_trials, num_sensors)  per-sensor t_obs per trial
    """
    keys = jax.random.split(master_key, n_trials)
    all_hit_counts = []
    all_hit_times = []
    for i in range(n_trials):
        td = jax.lax.stop_gradient(
            data_simulator(true_track, keys[i], photon_data))
        all_hit_counts.append(np.asarray(td[0]))
        all_hit_times.append(np.asarray(td[1]))
        if (i + 1) % 50 == 0:
            print(f'  trial {i+1}/{n_trials}', flush=True)
    return (np.stack(all_hit_counts, axis=0),
            np.stack(all_hit_times, axis=0))


# ---------------------------------------------------------------------------
# Single-sensor validation plot
# ---------------------------------------------------------------------------
def plot_validation(sensor_id, stratum_label,
                    all_hit_counts, all_hit_times, diag,
                    sigma_tts=SIGMA_TTS_NS,
                    n_bins=40, log_y=True,
                    out_path=None):
    """Empirical t_obs histogram + analytical p_first(t), conditioned on hit.

    Both curves are conditional pdfs of the per-sensor first-arrival time
    given that the sensor was hit. Empirical: counts / (n_hits·bin_width).
    Analytical: p_first(t) / (1 - exp(-Λ_total)).
    """
    s = sensor_id
    n_trials = all_hit_counts.shape[0]

    # Conditional sample: t_obs across trials where the sensor registered a hit
    hit_mask = all_hit_counts[:, s] > 0
    n_hits = int(hit_mask.sum())
    if n_hits < 5:
        print(f'  sensor {s}: only {n_hits} hits / {n_trials} trials — skip')
        return None
    t_obs_samples = all_hit_times[hit_mask, s]

    # Analytical curves on per-sensor grid
    t_grid = diag['t_grid'][s]
    p_first = diag['p_first_grid'][s]
    Lam_grid = diag['Lam_grid'][s]
    Lam_total = float(Lam_grid[-1])

    # Conditional analytical pdf
    hit_prob = 1.0 - np.exp(-Lam_total) if Lam_total > 0 else 1e-12
    p_first_cond = p_first / hit_prob

    # Plot range from p_first support, expanded to cover empirical samples.
    valid = p_first > p_first.max() * 1e-3
    if valid.any():
        idx = np.where(valid)[0]
        x_lo = float(t_grid[idx[0]])
        x_hi = float(t_grid[idx[-1]])
    else:
        x_lo, x_hi = float(t_grid[0]), float(t_grid[-1])
    x_lo = min(x_lo, float(t_obs_samples.min()))
    x_hi = max(x_hi, float(t_obs_samples.max()))
    pad = 2.0 * sigma_tts
    x_lo -= pad
    x_hi += pad

    # Empirical histogram normalized to a pdf
    bins = np.linspace(x_lo, x_hi, n_bins + 1)
    bin_width = (x_hi - x_lo) / n_bins
    counts, _ = np.histogram(t_obs_samples, bins=bins)
    hist_pdf = counts / (n_hits * bin_width)
    centers = 0.5 * (bins[:-1] + bins[1:])

    # Predicted PE for the title
    pred_pe = float(np.sum(diag['lam_grid'][s])
                    * (float(t_grid[1]) - float(t_grid[0])))

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.bar(centers, hist_pdf, width=bin_width * 0.95, align='center',
           color='lightcoral', edgecolor='crimson', alpha=0.55, linewidth=0.4,
           label=f'empirical $t_{{obs}}$ ({n_hits} hits / {n_trials} trials)')
    ax.plot(t_grid, p_first_cond, color='darkgreen', linewidth=2.0,
            label=r'analytical $p_{\mathrm{first}}(t)$ / hit prob.')
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel('arrival time [ns]')
    ax.set_ylabel('p.d.f. [1/ns]')
    ax.set_title(
        f'{stratum_label}  $\\bullet$  Sensor {sensor_id}\n'
        f'pred $\\langle$PE$\\rangle$ = {pred_pe:.2f}  '
        f'$\\bullet$  hit prob (analytic) = {hit_prob:.3f}  '
        f'$\\bullet$  empirical = {n_hits / n_trials:.3f}',
        fontsize=10)
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.legend(fontsize=8, frameon=True, loc='upper right',
              facecolor='white', framealpha=0.85)

    if log_y:
        nonzero_max = max(float(np.max(hist_pdf)) if hist_pdf.size else 0.0,
                          float(np.max(p_first_cond)) if p_first_cond.size else 0.0,
                          1e-6)
        floor = max(1e-5, nonzero_max * 1e-4)
        ax.set_yscale('log')
        ax.set_ylim(floor, nonzero_max * 3.0)
    else:
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    detector, num_detectors, data_sim, pred_sim = build_simulators()
    print(f'Detectors: {num_detectors}')

    key = jax.random.PRNGKey(SEED)

    # Initial event: gives true_track, photon_data, first true_data, pred output
    print('Building event (one prediction sim, one data sim) ...')
    (true_track, true_data, pred_data_pp,
     pos, direction, energy, photon_data) = build_event(
        detector, data_sim, pred_sim, key)

    hit_counts0 = np.asarray(true_data[0])
    n_hit0 = int(np.sum(hit_counts0 > 0))
    print(f'Event: E = {float(energy):.1f} MeV, hit sensors (trial 0) = '
          f'{n_hit0} / {num_detectors}')

    # Per-photon prediction → numpy
    log_w, flat_times_jax, flat_indices_jax, total_charge_jax = pred_data_pp
    flat_times = np.asarray(flat_times_jax)
    flat_indices = np.asarray(flat_indices_jax)
    weights = np.exp(np.asarray(log_w))
    total_pe = np.asarray(total_charge_jax)

    # Analytical diagnostics: one JIT call. t_obs from first trial is used
    # only for the in-place NLL annotation; the validation plots use the
    # whole trial distribution instead.
    print('Computing analytical p_first diagnostics (JIT) ...')
    import time as _time
    _t0 = _time.time()
    evaluate = make_diagnostics_evaluator(
        num_sensors=num_detectors, n_grid=200, sigma_tts=SIGMA_TTS_NS)
    diag = evaluate(flat_times, weights, flat_indices,
                     np.asarray(true_data[1]))
    print(f'  done in {_time.time() - _t0:.2f} s')

    # Many data trials with fresh keys on the same true event
    print(f'Running {N_TRIALS} data trials ...')
    _t0 = _time.time()
    trial_key = jax.random.PRNGKey(SEED + 1)
    all_hit_counts, all_hit_times = run_data_trials(
        data_sim, true_track, photon_data, trial_key, N_TRIALS)
    print(f'  done in {_time.time() - _t0:.2f} s')

    # Stratify on the FIRST trial's hits (just to pick representative
    # high/medium/low PE sensors).
    high, medium, low = stratified_sensor_selection(
        total_pe, hit_counts0, n_per_stratum=N_PER_STRATUM, hit_threshold=1.0)
    print(f'High PE   sensors: {high}')
    print(f'Medium PE sensors: {medium}')
    print(f'Low PE    sensors: {low}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label_map = {'high': 'High PE', 'medium': 'Medium PE', 'low': 'Low PE'}
    for stratum_key, sensors in (('high', high), ('medium', medium), ('low', low)):
        for s in sensors:
            out = OUT_DIR / f'{stratum_key}_sensor_{s}.png'
            plot_validation(
                sensor_id=s, stratum_label=label_map[stratum_key],
                all_hit_counts=all_hit_counts, all_hit_times=all_hit_times,
                diag=diag, out_path=str(out))
    print(f'Saved validation plots to {OUT_DIR}/')


if __name__ == '__main__':
    main()
