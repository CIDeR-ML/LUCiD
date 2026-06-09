"""Capstone: joint charge + timing multi-observable calibration + TQ map (gap 4).

Uses the gap-3 fit_charge_time on the REAL SK_like forward (moments mode → per-sensor mean
charge + first-arrival time) to recover, in ONE joint fit:
  - the optical globals + tts (global block, fed by both charge and time),
  - the per-PMT QE map k (charge Schur block, the Q-map), and
  - the per-PMT time offset t0 (time Schur block, the T-map).
This is the unified multi-observable capstone — charge gives the optics + QE map, timing
gives the T-map + tts and breaks charge-blind directions. Truth bakes a per-PMT k spread,
a per-PMT t0 spread, and tts; the fit recovers all from a perturbed start.

Usage: CUDA_VISIBLE_DEVICES=<g> python campaign/capstone.py
Env: NPH, K, STEPS, PERT, NB_H, WTIME, KSPREAD, T0SPREAD, TTS, GRID.
"""
import os, sys, time
import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources import laser_source, isotropic_source
from lucid.detector_params import DetectorParams, _flatten_detector_params, _nest_flat_kwargs
from lucid.fitting import fit_charge_time, ChargeTimeModel

GEOM = os.path.join(_ROOT, 'config', 'SK_like_geom_config.json')
GK = dict(n_cap=150, n_angular=250, n_height=150) if os.environ.get('GRID', '0') == '1' \
    else dict(n_cap=100, n_angular=150, n_height=100)
NPH = int(float(os.environ.get('NPH', '5e5')))
K = int(os.environ.get('K', '8'))
STEPS = int(os.environ.get('STEPS', '120'))
PERT = float(os.environ.get('PERT', '0.15'))
NB_H = int(os.environ.get('NB_H', '2'))
WTIME = float(os.environ.get('WTIME', '0.3'))
KSPREAD = float(os.environ.get('KSPREAD', '0.12'))
T0SPREAD = float(os.environ.get('T0SPREAD', '3.0'))
TTS = float(os.environ.get('TTS', '2.0'))
INTENS = float(os.environ.get('INTENS', '1e8'))
REPORT = os.path.join(_HERE, 'CAPSTONE_RESULTS.md')

FIELDS = ['scatter_length', 'absorption_length', 'wall_reflection_rate',
          'sensor_reflection_rate', 'qe', 'tts']

_lines = []
def emit(s=''):
    print(s, flush=True); _lines.append(s)
    with open(REPORT, 'w') as f:
        f.write('\n'.join(_lines) + '\n')


def main():
    t0 = time.time()
    det = generate_detector(GEOM); NS = len(det.all_points); H = det.H
    rng = np.random.default_rng(0)
    k_true = np.exp(KSPREAD * rng.standard_normal(NS)); k_true /= np.exp(np.mean(np.log(k_true)))
    t0_true = T0SPREAD * rng.standard_normal(NS); t0_true -= t0_true.mean()

    base = dict(scatter_length=70., mie_scatter_length=3000., g=0.9,
                wall_reflection_rate=.2, sensor_reflection_rate=.2, absorption_length=60.,
                qe=0.07, spe_width=0.35, tts=TTS)
    dp_true = DetectorParams.from_flat(qe_corrections=jnp.asarray(k_true),
                                       t0=jnp.asarray(t0_true), **base)
    flat_true = {kk: np.asarray(v) for kk, v in _flatten_detector_params(dp_true).items()}

    sim = setup_event_simulator(GEOM, NPH, temperature=None, K=K, is_calibration=True,
                                hit_mode='moments', wavelength_mode=False, **GK)
    srcs = [laser_source(position=[0., 0., H/2 - .1], intensity=INTENS),
            isotropic_source(position=[0., 0., 0.], intensity=INTENS)]

    theta0 = np.log(np.array([float(flat_true[f]) for f in FIELDS]))

    def unravel(theta, k_value=1.0):
        flat = {kk: jnp.asarray(v) for kk, v in flat_true.items()}
        for i, f in enumerate(FIELDS):
            flat[f] = jnp.exp(theta[i])
        flat['qe_corrections'] = jnp.asarray(k_value) * jnp.ones(NS)   # k Schur baseline
        flat['t0'] = jnp.zeros(NS)                                     # t0 Schur baseline
        return _nest_flat_kwargs(flat)

    def make_forward(src):
        def forward(theta, ek, pk):
            mean, var, tarr = sim(src, unravel(theta, 1.0), ek)
            return mean, tarr
        return forward
    models = [ChargeTimeModel(make_forward(s)) for s in srcs]

    # truth observables (k + t0 + tts baked into dp_true)
    truth_c = [np.asarray(sim(s, dp_true, jax.random.PRNGKey(1))[0]) for s in srcs]
    truth_t = [np.asarray(sim(s, dp_true, jax.random.PRNGKey(1))[2]) for s in srcs]

    emit('# Capstone — joint charge + timing calibration + TQ map (gap 4)')
    emit('')
    emit(f'SK_like NS={NS}, N_photons={NPH:.0e}, K={K}, grid={GK}, moments mode, '
         f'sources=[laser_down, iso], steps={STEPS}, w_time={WTIME}, start ±{PERT:.0%}.')
    emit(f'Truth: optics(L_R70/L_abs60/wall=sensor.2/qe.07) + tts={TTS}ns + per-PMT k '
         f'spread {KSPREAD:.0%} + per-PMT t0 spread {T0SPREAD}ns.')
    emit('')

    start = theta0 + rng.uniform(-PERT, PERT, theta0.shape)
    emit(f'joint fit_charge_time of {FIELDS} + per-PMT k (Q-map) + per-PMT t0 (T-map) …')
    res = fit_charge_time(models, truth_c, truth_t, start, NS, steps=STEPS, refresh=15,
                          w_time=WTIME, nb_h=NB_H, step_max=0.1, kstep_max=0.3, t0step_max=1.5)
    emit(f'  fit done ({time.time()-t0:.0f}s)')
    emit('')
    emit('| global | truth | start | recovered | frac err |')
    emit('|---|---|---|---|---|')
    for i, f in enumerate(FIELDS):
        tv = float(flat_true[f]); sv = float(np.exp(start[i])); rv = float(res['theta'][i])
        emit(f'| {f} | {tv:.3g} | {sv:.3g} | {rv:.4g} | {abs(rv/tv-1)*100:.1f}% |')
    kh = res['k']; t0h = res['t0']
    lit_c = truth_c[1] > 0; lit_t = truth_t[1] > 0
    emit('')
    emit(f'**Q-map** (per-PMT k): corr(k̂, k_true) = {np.corrcoef(kh[lit_c], k_true[lit_c])[0,1]:.4f}, '
         f'RMS frac = {np.sqrt(np.mean((kh[lit_c]/k_true[lit_c]-1)**2))*100:.1f}% (spread {KSPREAD:.0%})')
    t0t = t0_true - t0_true[lit_t].mean(); t0hh = t0h - t0h[lit_t].mean()
    emit(f'**T-map** (per-PMT t0): corr(t̂0, t0_true) = {np.corrcoef(t0hh[lit_t], t0t[lit_t])[0,1]:.4f}, '
         f'RMS = {np.sqrt(np.mean((t0hh[lit_t]-t0t[lit_t])**2)):.3f} ns (spread {T0SPREAD}ns)')
    emit('')
    emit('Joint charge+timing in ONE fit: charge → optics + Q-map; timing → T-map + tts. '
         'The unified multi-observable capstone on the exact GN recipe (charge Schur k + time Schur t0).')
    emit('')
    emit(f'_Finished in {(time.time()-t0)/60:.1f} min._')


if __name__ == '__main__':
    main()
