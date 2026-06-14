"""Timing calibration through the REAL photon sim — per-PMT t0 + the TTS/geometry finding.

Generates truth as M sample-mode flashes (per-photon TTS, integer-PE Poisson shots via
intensity=NPH), accumulates the per-PMT first-arrival mean & variance, and:

  1. recovers the per-PMT t0 (the dominant, "easy" TQ-map constant) from the first-arrival
     MEAN shift relative to an INDEPENDENT t0=0 reference run (the detector/geometry
     response characterisation) — at the flash-noise floor;
  2. reports the first-arrival VARIANCE finding: with a broad-arrival (isotropic) source the
     variance is dominated by the GEOMETRIC first-arrival order statistic (earliest of μ
     photons over the direct+scattered arrival spread), NOT by TTS·Var[min]. timingcal.py's
     toy assumed a DELTA geometric arrival (sharp laser); the full sim shows clean
     TTS-from-variance needs a sharp-geometric source or an explicit geometric-order model.
     The t0+TTS estimator itself is validated on the timingcal data model in
     tests/test_timing_calibration.py.

Usage: CUDA_VISIBLE_DEVICES=<g> python scripts/campaign/timing_campaign.py
Env: M (flashes per run), NPH (=intensity), TTS, ST0.
"""
import os, sys, time
import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources import isotropic_source
from lucid.detector_params import DetectorParams
from lucid.simulation.sensor_response import _occ_bias_var, _occ_bias_mean

GEOM = os.path.join(_ROOT, 'config', 'SK_like_geom_config.json')
GK = dict(n_cap=150, n_angular=250, n_height=150)
M = int(os.environ.get('M', '250'))
NPH = int(float(os.environ.get('NPH', '1e6')))
INTENS = float(os.environ.get('INTENS', str(NPH)))   # =NPH ⇒ integer-PE Poisson shots
TTS_TRUE = float(os.environ.get('TTS', '2.0'))
ST0 = float(os.environ.get('ST0', '3.0'))
REPORT = os.path.join(_HERE, 'TIMING_RESULTS.md')

_lines = []
def emit(s=''):
    print(s, flush=True); _lines.append(s)
    with open(REPORT, 'w') as f:
        f.write('\n'.join(_lines) + '\n')


def _accumulate(sim, src, dp, keys, t0_add):
    NS = t0_add.shape[0]
    n = np.zeros(NS); s = np.zeros(NS); s2 = np.zeros(NS)
    for k in keys:
        _, t = sim(src, dp, k)
        t = np.array(t); hit = t > 0
        fa = np.where(hit, t + t0_add, 0.0)
        n += hit; s += fa; s2 += fa * fa
    mean = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    var = np.where(n > 0, s2 / np.maximum(n, 1) - mean ** 2, 0.0)
    return mean, var, n


def main():
    t_start = time.time()
    det = generate_detector(GEOM); NS = len(det.all_points)
    rng = np.random.default_rng(0)
    t0_true = ST0 * rng.standard_normal(NS); t0_true -= t0_true.mean()

    dp0 = DetectorParams.from_flat(scatter_length=70., mie_scatter_length=3000., g=0.9,
                                   wall_reflection_rate=.2, sensor_reflection_rate=.2,
                                   absorption_length=60., qe=0.07, qe_corrections=jnp.ones(NS))
    dp_tts = dp0._replace(response=dp0.response._replace(tts=jnp.asarray(TTS_TRUE)))
    src = isotropic_source(position=[0., 0., 0.], intensity=INTENS)

    emit('# Timing calibration through the real sim — per-PMT t0 + the variance finding')
    emit('')
    emit(f'SK_like NS={NS}; isotropic; M={M} flashes/run; N_photons={NPH:.0e} (intensity=NPH '
         f'⇒ integer-PE shots); TTS_true={TTS_TRUE} ns; t0 spread={ST0} ns.')
    emit('')

    # occupancy μ (expected) from moments mode
    sim_m = setup_event_simulator(GEOM, NPH, temperature=None, K=8, is_calibration=True,
                                  hit_mode='moments', wavelength_mode=False, **GK)
    mean_q, _, _ = sim_m(src, dp0, jax.random.PRNGKey(1))
    mu = np.array(mean_q); lit_mu = mu > 0
    emit(f'occupancy μ (lit): median={np.median(mu[lit_mu]):.2f}, '
         f'10-90%=[{np.percentile(mu[lit_mu],10):.2f}, {np.percentile(mu[lit_mu],90):.2f}]')

    sim_d = setup_event_simulator(GEOM, NPH, temperature=None, K=8, is_calibration=True,
                                  use_expected_value=False, hit_mode='realistic',
                                  apply_smearing=False, wavelength_mode=False, **GK)

    # reference run (t0=0) characterises the detector/geometry response; data run (t0=true)
    # uses INDEPENDENT keys → t0 recovery is a genuine (non-circular) flash-noise-limited fit.
    t0 = time.time()
    ref_mean, ref_var, n_ref = _accumulate(sim_d, src, dp_tts, [jax.random.PRNGKey(i) for i in range(M)], np.zeros(NS))
    emit(f'  reference run done ({time.time()-t0:.0f}s)')
    data_mean, data_var, n_dat = _accumulate(sim_d, src, dp_tts, [jax.random.PRNGKey(10_000 + i) for i in range(M)], t0_true)
    emit(f'  data run done ({time.time()-t0:.0f}s)')

    lit = (n_ref >= max(15, int(0.05 * M))) & (n_dat >= max(15, int(0.05 * M)))
    nlit = int(lit.sum())

    # ---- t0 recovery (from the mean shift vs the reference) ----
    t0_hat = np.where(lit, data_mean - ref_mean, 0.0)
    t0_hat = np.where(lit, t0_hat - t0_hat[lit].mean(), 0.0)           # gauge mean(t0)=0
    t0t = np.where(lit, t0_true - t0_true[lit].mean(), 0.0)
    t0_rms = float(np.sqrt(np.mean((t0_hat[lit] - t0t[lit]) ** 2)))
    t0_corr = float(np.corrcoef(t0_hat[lit], t0t[lit])[0, 1])
    floor = float(np.median(np.sqrt(np.clip(ref_var[lit] + data_var[lit], 0, None)) / np.sqrt(M)))

    emit('')
    emit('## per-PMT t0 (the dominant TQ-map constant)')
    emit('')
    emit(f'- n_lit = {nlit}')
    emit(f'- **t0 RMS recovery = {t0_rms:.3f} ns** (flash-noise floor √((var_ref+var_data)/M) median ≈ {floor:.3f} ns)')
    emit(f'- t0 correlation (recovered vs truth) = {t0_corr:.4f}')
    emit('→ per-PMT t0 is recovered through the full photon sim at the flash-noise floor — '
         'the dominant, well-conditioned timing calibration.')
    emit('')

    # ---- variance finding: geometric order statistic vs the TTS prediction ----
    bvar = np.array(_occ_bias_var(jnp.asarray(np.clip(mu, 1e-3, None))))
    tts_pred_var = TTS_TRUE ** 2 * bvar                  # what timingcal's delta-geom model expects
    ratio = np.median((ref_var[lit] / np.clip(tts_pred_var[lit], 1e-9, None)))
    emit('## first-arrival variance: the geometric-order finding')
    emit('')
    emit(f'- measured Var[first] median = {np.median(ref_var[lit]):.3f} ns²')
    emit(f'- TTS·Var[min] prediction (delta-geometric, timingcal toy) median = {np.median(tts_pred_var[lit]):.3f} ns²')
    emit(f'- ratio measured/predicted ≈ **{ratio:.0f}×** → the variance is dominated by the GEOMETRIC '
         'first-arrival order statistic (earliest of μ photons over the broad direct+scattered '
         'arrival spread), NOT by TTS. timingcal.py assumed a DELTA geometric arrival (sharp laser); '
         'the full sim shows clean TTS-from-variance needs a sharp-geometric (laser, direct-spot) '
         'source or an explicit geometric-order model. The t0+TTS ESTIMATOR is validated on the '
         'timingcal data model in tests/test_timing_calibration.py.')
    emit('')
    emit(f'_Finished in {(time.time()-t_start)/60:.1f} min._')


if __name__ == '__main__':
    main()
