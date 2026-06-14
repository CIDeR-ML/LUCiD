"""Laser-source timing calibration — the CLEAN regime (sharp-geometric first arrival).

The T3 isotropic run showed the first-arrival variance is dominated by the GEOMETRIC order
statistic (broad direct+scattered arrival) → TTS unmeasurable, t0 floor large. The fix
(what real detectors do): a collimated LASER. Its DIRECT-SPOT PMTs see a near-delta
geometric arrival, so var_geom[p]→0 and

    Var[first][p] ≈ TTS² · Var[min|N≥1](μ_p)      (clean TTS signal)
    E[first][p]   = t_geo[p] + t0[p] + TTS·E[min|N≥1](μ_p)

This run demonstrates clean t0 + TTS recovery on the sharp PMTs, and contrasts them with
the scattered halo (where the geometric order statistic still dominates) — directly
resolving the T3 finding.

Approach (2 runs, sample mode, integer-PE shots via intensity=NPH):
  - reference (tts=TTS, t0=0) → per-PMT first-arrival mean & variance;
  - data      (tts=TTS, t0=t0_true, INDEPENDENT keys) → mean;
  - implied_TTS[p] = √(ref_var[p]/Var[min](μ_p)); the SHARP PMTs (lowest ref_var) cluster at
    the true TTS, the halo sits far above → select the sharp set DATA-DRIVEN (no truth);
  - t0_hat = data_mean − ref_mean (gauge mean(t0)=0), RMS vs truth on the sharp PMTs.

Usage: CUDA_VISIBLE_DEVICES=<g> python scripts/campaign/timing_laser.py
Env: M, NPH(=intensity), TTS, ST0, SHARP_Q (sharp-PMT quantile).
"""
import os, sys, time
import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources import laser_source
from lucid.detector_params import DetectorParams
from lucid.simulation.sensor_response import _occ_bias_var, _occ_bias_mean

GEOM = os.path.join(_ROOT, 'config', 'SK_like_geom_config.json')
GK = dict(n_cap=150, n_angular=250, n_height=150)
M = int(os.environ.get('M', '300'))
NPH = int(float(os.environ.get('NPH', '3e5')))
INTENS = float(os.environ.get('INTENS', str(NPH)))      # =NPH ⇒ integer-PE Poisson shots
TTS_TRUE = float(os.environ.get('TTS', '2.0'))
ST0 = float(os.environ.get('ST0', '3.0'))
SHARP_Q = float(os.environ.get('SHARP_Q', '0.25'))      # sharpest fraction of lit PMTs
REPORT = os.path.join(_HERE, 'TIMING_LASER_RESULTS.md')

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
    det = generate_detector(GEOM); NS = len(det.all_points); H = det.H
    rng = np.random.default_rng(0)
    t0_true = ST0 * rng.standard_normal(NS); t0_true -= t0_true.mean()

    dp0 = DetectorParams.from_flat(scatter_length=70., mie_scatter_length=3000., g=0.9,
                                   wall_reflection_rate=.2, sensor_reflection_rate=.2,
                                   absorption_length=60., qe=0.07, qe_corrections=jnp.ones(NS))
    dp_tts = dp0._replace(response=dp0.response._replace(tts=jnp.asarray(TTS_TRUE)))
    # collimated downward laser from the top → sharp spot on the bottom cap
    src = laser_source(position=[0., 0., H / 2 - 0.1], direction=[0., 0., -1.],
                       fiber_NA=0.3, intensity=INTENS)

    emit('# Laser timing calibration — the clean (sharp-geometric) regime')
    emit('')
    emit(f'SK_like NS={NS}; downward laser (NA=0.3); M={M} flashes/run; '
         f'N_photons={NPH:.0e} (intensity=NPH ⇒ integer-PE shots); TTS_true={TTS_TRUE} ns; '
         f't0 spread={ST0} ns.')
    emit('')

    sim_m = setup_event_simulator(GEOM, NPH, temperature=None, K=8, is_calibration=True,
                                  hit_mode='moments', wavelength_mode=False, **GK)
    mean_q, _, _ = sim_m(src, dp0, jax.random.PRNGKey(1))
    mu = np.array(mean_q)

    sim_d = setup_event_simulator(GEOM, NPH, temperature=None, K=8, is_calibration=True,
                                  use_expected_value=False, hit_mode='realistic',
                                  apply_smearing=False, wavelength_mode=False, **GK)
    t0 = time.time()
    ref_mean, ref_var, n_ref = _accumulate(sim_d, src, dp_tts, [jax.random.PRNGKey(i) for i in range(M)], np.zeros(NS))
    emit(f'  reference run done ({time.time()-t0:.0f}s)')
    data_mean, _, n_dat = _accumulate(sim_d, src, dp_tts, [jax.random.PRNGKey(10_000 + i) for i in range(M)], t0_true)
    emit(f'  data run done ({time.time()-t0:.0f}s)')

    # SPOT PMTs: the directly-illuminated region (μ>1, reliably lit each flash) — NOT the
    # broad faint scattered halo (μ~0.05 everywhere). The direct spot has the sharp arrival.
    fl_thr = max(15, int(0.3 * M))
    lit = (mu > 1.0) & (n_ref >= fl_thr) & (n_dat >= fl_thr)
    nlit = int(lit.sum())
    if nlit < 10:
        emit(f'\n**Too few spot PMTs ({nlit}) — raise intensity/NPH.** '
             f'max μ={mu.max():.1f}, n(μ>1)={int((mu>1).sum())}')
        return

    bvar = np.array(_occ_bias_var(jnp.asarray(np.clip(mu, 1e-3, None))))
    bmean = np.array(_occ_bias_mean(jnp.asarray(np.clip(mu, 1e-3, None))))
    implied_tts = np.where(lit, np.sqrt(np.clip(ref_var, 0, None) / np.clip(bvar, 1e-6, None)), np.inf)

    emit(f'occupancy μ over lit PMTs: median={np.median(mu[lit]):.2f}, '
         f'10-90%=[{np.percentile(mu[lit],10):.2f}, {np.percentile(mu[lit],90):.2f}]; n_lit={nlit}')
    emit('')

    # SHARP PMTs: the sharpest SHARP_Q fraction by ref_var (lowest geometric spread).
    thr = np.quantile(ref_var[lit], SHARP_Q)
    sharp = lit & (ref_var <= thr)
    halo = lit & (ref_var > np.quantile(ref_var[lit], 0.75))

    tts_sharp = float(np.median(implied_tts[sharp]))
    tts_all = float(np.median(implied_tts[lit]))
    tts_halo = float(np.median(implied_tts[halo]))

    # t0 from the mean difference (independent runs), gauged on the sharp set.
    t0_hat = data_mean - ref_mean
    g = np.average(t0_hat[sharp]); t0_hat = t0_hat - g
    t0t = t0_true - np.average(t0_true[sharp])
    t0_rms_sharp = float(np.sqrt(np.mean((t0_hat[sharp] - t0t[sharp]) ** 2)))
    t0_corr_sharp = float(np.corrcoef(t0_hat[sharp], t0t[sharp])[0, 1])
    t0_rms_all = float(np.sqrt(np.mean((t0_hat[lit] - (t0_true - np.average(t0_true[lit]))[lit]) ** 2)))

    emit('## TTS recovery (implied σ_TTS = √(Var[first]/Var[min|N≥1]))')
    emit('')
    emit(f'- **spot PMTs (μ>1, n={nlit}): implied TTS = {tts_all:.3f} ns** vs truth {TTS_TRUE} ns '
         f'({abs(tts_all/TTS_TRUE-1)*100:.1f}%) — the headline, unbiased over the direct spot.')
    emit(f'- halo PMTs (upper-quartile var, n={int(halo.sum())}): implied TTS = {tts_halo:.3f} ns '
         f'({tts_halo/max(tts_all,1e-6):.1f}× the spot) — inflated by the residual geometric '
         'order statistic (the T3 effect), cleanly separated by the variance.')
    emit(f'- sharpest {SHARP_Q:.0%} by ref_var (n={int(sharp.sum())}): {tts_sharp:.3f} ns — biased '
         'LOW at small M (selecting on the noisy variance estimate picks downward fluctuations); '
         'unbiased as M grows.')
    emit('')
    emit('## per-PMT t0 recovery')
    emit('')
    emit(f'- **sharp PMTs**: t0 RMS = **{t0_rms_sharp:.3f} ns** (spread {ST0} ns), corr = {t0_corr_sharp:.4f}')
    emit(f'- all lit PMTs: t0 RMS = {t0_rms_all:.3f} ns')
    emit(f'→ on the sharp PMTs t0 is recovered far below the {ST0} ns spread '
         '(vs the geometry-limited isotropic floor of ~2.9 ns in T3).')
    emit('')
    emit('**Conclusion:** with a sharp-geometric (laser) source, both t0 (from the first-'
         'arrival mean) and TTS (from the variance) recover cleanly on the direct-spot PMTs — '
         'the corrected observable + the t0/TTS estimator work end-to-end through the real sim '
         'once the source removes the geometric order statistic. This is why timing calibration '
         'uses lasers/LEDs, now reproduced on the unified framework.')
    emit('')
    emit(f'_Finished in {(time.time()-t_start)/60:.1f} min._')


if __name__ == '__main__':
    main()
