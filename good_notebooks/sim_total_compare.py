"""Direct comparison of total predicted charge vs total observed charge
across 50 events (same seed/entry pairs as the bias study).

For each event:
  1. Build the event with build_event_for_entry → data_sim(true_track) gives
     observed counts (hit_counts).
  2. Run pred_sim(true_track) at exactly the same true parameters.
  3. Report:
       sum N_data  = Σ_s hit_counts[s]
       sum N_pred  = Σ_s total_charge[s]
       ratio       = N_pred / N_data
  4. Aggregate ratio statistics across the 50 events.

The implied charge-only E bias is  E_bias / E_true  ≈ −(1 − 1/ratio).
If the secant-measured E bias matches this, we're consistent. If not,
something else is in play.
"""
from __future__ import annotations
import sys
sys.path.append('..')

import jax, jax.numpy as jnp
import numpy as np

from parameter_scans_1D_gaussian import (
    DEFAULT_JSON_FILENAME, PHYSICS_CONFIG, K_PRED, K_DATA,
    NPHOT_PRED, NPHOT_DATA,
)
from multi_event_bias_joint import build_event_for_entry
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator


N_EVENTS = 50
SEED_BASE = 44
START_ENTRY = 0


def main():
    detector = generate_detector(DEFAULT_JSON_FILENAME)
    num_detectors = len(detector.all_points)

    data_sim = setup_event_simulator(
        DEFAULT_JSON_FILENAME, NPHOT_DATA,
        temperature=None, K=K_DATA,
        is_data=True, is_calibration=False, apply_smearing=True,
        wavelength_mode=False,
        physics_config=PHYSICS_CONFIG, default_detector_params=True)
    pred_sim = setup_event_simulator(
        DEFAULT_JSON_FILENAME, NPHOT_PRED, 0.10,
        max_candidates_per_ray=4, K=K_PRED,
        is_data=False, hit_mode='per_photon',
        wavelength_mode=False,
        physics_config=PHYSICS_CONFIG, default_detector_params=True)

    sum_data_arr, sum_pred_arr, ratio_arr = [], [], []
    n_hit_arr, n_hit_thr2_arr = [], []
    pred_key = jax.random.PRNGKey(42)

    print('=' * 110)
    print(f'{"ev":>3} {"entry":>5} {"seed":>5} | '
          f'{"sumN_data":>10} {"sumN_pred":>10} '
          f'{"ratio":>8} {"impliedE_bias":>15} | '
          f'{"n_hit":>6} {"n>=2":>6}')
    print('-' * 110)

    for ev in range(N_EVENTS):
        entry_idx = START_ENTRY + ev
        seed = SEED_BASE + ev
        (true_track, true_data, x, y, z, theta, phi, energy) = build_event_for_entry(
            detector, data_sim, num_detectors, entry_idx=entry_idx, seed=seed)
        hit_counts = np.asarray(true_data[0])
        sum_data = float(np.sum(hit_counts))

        # pred_sim returns (log_w, flat_times, flat_indices, total_charge)
        pred_out = pred_sim(true_track, pred_key)
        total_charge = np.asarray(pred_out[3])
        sum_pred = float(np.sum(total_charge))

        ratio = sum_pred / sum_data
        # Charge-only E bias: E* such that ΣN_pred(E*) = ΣN_obs
        # If pred is linear in E: E* / E_true = 1/ratio → bias = E_true(1/ratio − 1)
        implied_e_bias = energy * (1.0 / ratio - 1.0)
        n_hit = int(np.sum(hit_counts > 0))
        n_hit_thr2 = int(np.sum(hit_counts >= 2.0))

        sum_data_arr.append(sum_data)
        sum_pred_arr.append(sum_pred)
        ratio_arr.append(ratio)
        n_hit_arr.append(n_hit)
        n_hit_thr2_arr.append(n_hit_thr2)

        print(f'{ev+1:>3} {entry_idx:>5} {seed:>5} | '
              f'{sum_data:>10.1f} {sum_pred:>10.1f} '
              f'{ratio:>8.4f} {implied_e_bias:>+13.1f}  | '
              f'{n_hit:>6} {n_hit_thr2:>6}')

    sum_data_arr = np.asarray(sum_data_arr)
    sum_pred_arr = np.asarray(sum_pred_arr)
    ratio_arr   = np.asarray(ratio_arr)

    print()
    print('=' * 110)
    print(' SUMMARY across 50 events')
    print('=' * 110)
    print(f'  ΣN_data: mean = {sum_data_arr.mean():.1f}, '
          f'median = {np.median(sum_data_arr):.1f}, '
          f'std = {sum_data_arr.std(ddof=1):.1f}')
    print(f'  ΣN_pred: mean = {sum_pred_arr.mean():.1f}, '
          f'median = {np.median(sum_pred_arr):.1f}, '
          f'std = {sum_pred_arr.std(ddof=1):.1f}')
    print(f'  ratio (pred/data): mean = {ratio_arr.mean():.4f}, '
          f'median = {np.median(ratio_arr):.4f}, '
          f'std = {ratio_arr.std(ddof=1):.4f}')
    print()

    implied_bias = 1050.0 * (1.0 / ratio_arr - 1.0)
    print(f'  Implied charge-only E bias (per event):')
    print(f'    mean   = {implied_bias.mean():+.2f} MeV')
    print(f'    median = {np.median(implied_bias):+.2f} MeV')
    print(f'    std    = {implied_bias.std(ddof=1):+.2f} MeV')
    print()
    print(f'  Compare: secant-measured E bias for factored_sum = -42.93 MeV')
    print(f'  Compare: secant-measured E bias for joint        = +152.73 MeV')


if __name__ == '__main__':
    main()
