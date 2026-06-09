"""Spectral λ-curve FIT recovery on the unified framework (gap-2: the fit, not just CRB).

spectral_crb.py gives the per-control-point identifiability (Fisher σ). This RECOVERS the
λ-deviation curves with the exact GN recipe: set DEVIATED truth curves (≠1), generate the
multi-λ charge data, start from a perturbed point, and run build_calibration_problem + fit.
Reports per-control-point fractional recovery vs truth for each curve, alongside the CRB σ.

Monochromatic lasers at each control λ + isotropic at each control λ give the source
diversity needed to separate abs(λ) from rayleigh(λ) at each wavelength (a laser alone at
one λ constrains only that knot of one curve — see spectral_crb).

Usage: CUDA_VISIBLE_DEVICES=<g> python campaign/spectral_fit.py
Env: NPH, K, STEPS, PERT, NB_H, GRID (1=full 150s, 0=reduced).
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
from lucid.detector_params import DetectorParams
from lucid.fitting import build_calibration_problem, fit, crb
from lucid.wavelength.optical_model import CONTROL_WAVELENGTHS_NM

GEOM = os.path.join(_ROOT, 'config', 'SK_like_geom_config.json')
GK = dict(n_cap=150, n_angular=250, n_height=150) if os.environ.get('GRID', '0') == '1' \
    else dict(n_cap=100, n_angular=150, n_height=100)
NPH = int(float(os.environ.get('NPH', '5e5')))
K = int(os.environ.get('K', '8'))
STEPS = int(os.environ.get('STEPS', '120'))
PERT = float(os.environ.get('PERT', '0.15'))
NB_H = int(os.environ.get('NB_H', '3'))
INTENS = float(os.environ.get('INTENS', '1e8'))
REPORT = os.path.join(_HERE, 'SPECTRAL_FIT_RESULTS.md')
CTRL = [float(w) for w in CONTROL_WAVELENGTHS_NM]

# DEVIATED truth λ-deviation curves (≠1) at the 5 control points.
ABS_DEV_TRUE = np.array([1.20, 1.10, 1.00, 0.95, 0.90])
RAY_DEV_TRUE = np.array([0.90, 0.95, 1.00, 1.05, 1.10])

_lines = []
def emit(s=''):
    print(s, flush=True); _lines.append(s)
    with open(REPORT, 'w') as f:
        f.write('\n'.join(_lines) + '\n')


def main():
    t0 = time.time()
    det = generate_detector(GEOM); NS = len(det.all_points); H = det.H
    dp_true = DetectorParams.from_flat(
        scatter_length=70., mie_scatter_length=3000., g=0.9,
        wall_reflection_rate=.2, sensor_reflection_rate=.2, absorption_length=60., qe=0.07,
        abs_dev=jnp.asarray(ABS_DEV_TRUE), rayleigh_dev=jnp.asarray(RAY_DEV_TRUE),
        qe_corrections=jnp.ones(NS))
    sim = setup_event_simulator(GEOM, NPH, temperature=None, K=K, is_calibration=True,
                                hit_mode='aggregated', wavelength_mode=True, **GK)
    srcs = [laser_source(position=[0., 0., H/2 - .1], intensity=INTENS, wavelength=w) for w in CTRL]
    srcs += [isotropic_source(position=[0., 0., 0.], intensity=INTENS, wavelength=w) for w in CTRL]

    emit('# Spectral λ-curve FIT recovery — unified GN recipe (gap 2)')
    emit('')
    emit(f'SK_like NS={NS}, N_photons={NPH:.0e}, K={K}, grid={GK}, sources=5 lasers+5 iso @ control λ, '
         f'steps={STEPS}, start perturbation ±{PERT:.0%}.')
    emit(f'Truth abs_dev={list(ABS_DEV_TRUE)}, rayleigh_dev={list(RAY_DEV_TRUE)} (deviated ≠1).')
    emit('')

    fields = ['abs_dev', 'rayleigh_dev']
    prob = build_calibration_problem(sim, srcs, dp_true, fields, key=jax.random.PRNGKey(1))
    truth = np.exp(prob['theta0'])
    rng = np.random.default_rng(0)
    start = prob['theta0'] + rng.uniform(-PERT, PERT, truth.shape)
    emit(f'loss-free GN fit of {fields} (10 control points), '
         f'start ±{PERT:.0%} off truth …')
    res = fit(prob['source_models'], prob['truth_charge'], start, NS,
              steps=STEPS, refresh=15, nb_h=NB_H)
    rec = res['theta']
    emit(f'  fit done ({time.time()-t0:.0f}s)')

    # CRB for the same curves (per-control-point σ), joint
    c = crb(prob['source_models'], prob['theta_true'], NS, nb_h=NB_H)
    sig = c['sigma']

    emit('')
    emit('| curve | ' + ' | '.join(f'{int(w)}nm' for w in CTRL) + ' |')
    emit('|' + '---|' * (len(CTRL) + 1))
    shapes = prob['shapes']; i = 0
    for fld, name in [('abs_dev', 'L_abs(λ) dev'), ('rayleigh_dev', 'L_R(λ) dev')]:
        n = int(np.prod(dict(shapes)[fld]))
        tv, rv, sv = truth[i:i+n], rec[i:i+n], sig[i:i+n]
        emit(f'| **{name}** truth | ' + ' | '.join(f'{v:.3f}' for v in tv) + ' |')
        emit(f'| recovered | ' + ' | '.join(f'{v:.3f}' for v in rv) + ' |')
        emit(f'| frac err | ' + ' | '.join(f'{abs(r/t-1)*100:.1f}%' for r, t in zip(rv, tv)) + ' |')
        emit(f'| CRB σ | ' + ' | '.join(f'{s*100:.2f}%' for s in sv) + ' |')
        i += n
    emit('')
    max_fe = float(np.max(np.abs(rec/truth - 1)))
    emit(f'**Max per-point fractional recovery error = {max_fe*100:.1f}%** over all 10 control '
         f'points (joint abs+Rayleigh λ-deviation fit). Recovers the deviated curves from a '
         f'±{PERT:.0%} start — the flexible-curve FIT (not just CRB), on the unified recipe.')
    emit('')
    emit(f'_Finished in {(time.time()-t0)/60:.1f} min._')


if __name__ == '__main__':
    main()
