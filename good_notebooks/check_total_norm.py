"""Verify whether the bias is just `tot_n_photons_normalization` calibration
or if the per-photon distribution genuinely matters.

For 50 PhotonSim entries (varying seeds + entries) at E=1050 MeV:

  1. Compute the empirical "needed" total_norm:
        ŷ(entry) = N_PhotonSim(entry) / mean_SIREN_weight
     (averaged across SIREN keys for stability).

  2. Compare to the fit value: total_norm_fit(1050) = 23.293·1050^0.909 − 1216.79

  3. Compute the empirical "needed" total_norm to match DETECTED counts
     (after propagation). If this differs from (1), the per-photon
     distribution matters beyond just the integrated yield.

  4. Apply g_cal = ŷ_emit / total_norm_fit numerically and report what
     residual bias remains in detected counts.
"""
from __future__ import annotations
import sys
sys.path.append('..')

import jax, jax.numpy as jnp
import numpy as np

from parameter_scans_1D_gaussian import (
    DEFAULT_JSON_FILENAME, PHYSICS_CONFIG, K_PRED, K_DATA,
    NPHOT_PRED, NPHOT_DATA, DATA_FILE,
)
from multi_event_bias_joint import build_event_for_entry
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.utils import unpack_photonsim_params
from lucid.siren.training.inference import SIRENPredictor
from lucid.siren.core import create_photonsim_siren_grid
from lucid.sources.siren_rays import photonsim_differentiable_get_rays
from lucid.generate import read_photon_data_from_photonsim


N_EVENTS = 50
SEED_BASE = 44
START_ENTRY = 0
ENERGY = 1050.0


def main():
    # SIREN setup
    params = unpack_photonsim_params('muon', 'water')
    a, b, c = params['tot_n_photons_normalization']
    sa, sb, sc = params['num_seeds']
    fit_total_norm = a * ENERGY**b + c
    print(f'Fit power law: y = {a:.4f}·E^{b:.4f} + {c:.2f}  (R²=0.997 reported)')
    print(f'  total_norm_fit(E={ENERGY}) = {fit_total_norm:.2f}')

    predictor = SIRENPredictor(params['siren_model_path'])
    grid_data = create_photonsim_siren_grid(predictor)
    model_params = predictor.params

    # Stable estimate of SIREN mean weight at E=1050 — average over multiple keys
    print('\nComputing SIREN mean weight (averaging over 5 keys for stability)...')
    track_origin = jnp.array([0., 0., 0.])
    track_direction = jnp.array([0., 0., 1.])
    weights = []
    for k in range(5):
        _, _, ws = photonsim_differentiable_get_rays(
            track_origin, track_direction, ENERGY, NPHOT_PRED,
            grid_data, model_params, jax.random.PRNGKey(100 + k), sa, sb, sc)
        weights.append(float(np.mean(np.asarray(ws))))
    siren_mean_weight = float(np.mean(weights))
    print(f'  SIREN mean weight: mean={siren_mean_weight:.4f},  '
          f'std across keys={float(np.std(weights, ddof=1)):.4f}')
    print(f'  SIREN total emission (predicted): '
          f'{fit_total_norm * siren_mean_weight:.1f} photons')

    # Per-entry PhotonSim N
    detector = generate_detector(DEFAULT_JSON_FILENAME)
    num_detectors = len(detector.all_points)

    data_sim = setup_event_simulator(
        DEFAULT_JSON_FILENAME, NPHOT_DATA, temperature=None, K=K_DATA,
        is_data=True, is_calibration=False, apply_smearing=True,
        wavelength_mode=False,
        physics_config=PHYSICS_CONFIG, default_detector_params=True)
    pred_sim = setup_event_simulator(
        DEFAULT_JSON_FILENAME, NPHOT_PRED, 0.10,
        max_sensors_per_cell=4, K=K_PRED,
        is_data=False, hit_mode='per_photon',
        wavelength_mode=False,
        physics_config=PHYSICS_CONFIG, default_detector_params=True)

    print()
    print('=' * 110)
    print(f'{"ev":>3} {"entry":>5} {"seed":>5} | {"PhotonSim N":>11} '
          f'{"needed_norm":>12} {"vs fit":>8} | {"sumN_data":>10} {"sumN_pred":>10} '
          f'{"need_norm_det":>14}')
    print('-' * 110)

    photonsim_Ns, sum_data_arr, sum_pred_arr = [], [], []
    pred_key = jax.random.PRNGKey(42)

    for ev in range(N_EVENTS):
        entry_idx = START_ENTRY + ev
        seed = SEED_BASE + ev
        # Read PhotonSim N for this entry
        pd = read_photon_data_from_photonsim(DATA_FILE, entry_idx)
        n_phsim = int(len(pd['photon_origins']))

        # Empirical "needed" total_norm to match emission:
        needed_norm_emit = n_phsim / siren_mean_weight
        ratio_to_fit_emit = needed_norm_emit / fit_total_norm

        # Run both sims at the same true track to get detected totals
        (true_track, true_data, x, y, z, theta, phi, energy) = build_event_for_entry(
            detector, data_sim, num_detectors, entry_idx=entry_idx, seed=seed)
        sum_data = float(np.sum(np.asarray(true_data[0])))
        out_pred = pred_sim(true_track, pred_key)
        sum_pred = float(np.sum(np.asarray(out_pred[3])))

        # Empirical "needed" total_norm to match DETECTED:
        # If we scale total_norm by g, sum_pred → g · sum_pred.
        # So g_det = sum_data / sum_pred  ⇒  needed_norm_det = fit · g_det
        needed_norm_det = fit_total_norm * (sum_data / sum_pred)

        photonsim_Ns.append(n_phsim)
        sum_data_arr.append(sum_data)
        sum_pred_arr.append(sum_pred)

        print(f'{ev+1:>3} {entry_idx:>5} {seed:>5} | {n_phsim:>11d} '
              f'{needed_norm_emit:>12.1f} {ratio_to_fit_emit:>8.4f} | '
              f'{sum_data:>10.1f} {sum_pred:>10.1f} {needed_norm_det:>14.1f}')

    photonsim_Ns = np.asarray(photonsim_Ns)
    sum_data_arr = np.asarray(sum_data_arr)
    sum_pred_arr = np.asarray(sum_pred_arr)

    needed_norm_emit_per_event = photonsim_Ns / siren_mean_weight
    needed_norm_det_per_event  = fit_total_norm * (sum_data_arr / sum_pred_arr)

    print()
    print('=' * 110)
    print(' SUMMARY  (averaged across 50 events)')
    print('=' * 110)
    print(f'  fit_total_norm(1050)          = {fit_total_norm:>10.2f}')
    print(f'  needed_norm to match EMISSION = '
          f'{needed_norm_emit_per_event.mean():>10.2f} ± '
          f'{needed_norm_emit_per_event.std(ddof=1):.2f}')
    print(f'    → fit overshoots emission by '
          f'{(fit_total_norm - needed_norm_emit_per_event.mean()) / needed_norm_emit_per_event.mean() * 100:+.2f} %')
    print()
    print(f'  needed_norm to match DETECTED = '
          f'{needed_norm_det_per_event.mean():>10.2f} ± '
          f'{needed_norm_det_per_event.std(ddof=1):.2f}')
    print(f'    → fit overshoots detection by '
          f'{(fit_total_norm - needed_norm_det_per_event.mean()) / needed_norm_det_per_event.mean() * 100:+.2f} %')
    print()
    print(f'  Difference between the two ("distribution-shape mismatch"):')
    diff_pct = (needed_norm_emit_per_event.mean() - needed_norm_det_per_event.mean()) / needed_norm_det_per_event.mean() * 100
    print(f'    needed_norm_EMIT − needed_norm_DET = '
          f'{needed_norm_emit_per_event.mean() - needed_norm_det_per_event.mean():.2f}'
          f'  ({diff_pct:+.2f} %)')
    print()
    print('Interpretation:')
    print('  • If the two needed_norms agree, the bias is purely a calibration')
    print('    fit residual at this energy → fix by adjusting tot_n_photons_normalization.')
    print('  • If they DIFFER, the per-photon distribution matters: same total emission')
    print('    but different propagation efficiency → simple gain calibration leaves a')
    print('    residual bias.')


if __name__ == '__main__':
    main()
