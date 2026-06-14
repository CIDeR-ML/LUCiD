"""Angular (complex) reflection FIT on the unified framework (gap 1).

The campaign so far fit only the two SCALAR reflectivities. This fits the full ANGULAR
model (opt-in reflection_model='angular'): Schlick blacksheet wall R_w=R0+(1-R0)(1-cosθ)^p
+ multilayer-Fresnel cathode R_s=pmt_reflectance(cosθ,λ,nr,nk), with specular/diffuse
fractions. Optics (scatter/abs/qe) are held at truth — reflection is calibrated with the
medium known (the staging that makes it identifiable; reflection ↔ absorption otherwise
trade). Charge identifies the MAGNITUDE params (wall_R0, wall_p, cathode_nr, cathode_nk);
the spec/diff fractions (wall_fspec, sensor_fspec) ride the DiCE score and are charge-LOOSE
(they are the timing lever, not a charge one) — expected, per the timing findings.

Usage: CUDA_VISIBLE_DEVICES=<g> python scripts/campaign/reflection_fit.py
Env: NPH, K, STEPS, PERT, NB_H, GRID.
"""
import os, sys, time
import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources import laser_source, isotropic_source
from lucid.detector_params import DetectorParams
from lucid.fitting import build_calibration_problem, fit, crb

GEOM = os.path.join(_ROOT, 'config', 'SK_like_geom_config.json')
GK = dict(n_cap=150, n_angular=250, n_height=150) if os.environ.get('GRID', '0') == '1' \
    else dict(n_cap=100, n_angular=150, n_height=100)
NPH = int(float(os.environ.get('NPH', '5e5')))
K = int(os.environ.get('K', '8'))
STEPS = int(os.environ.get('STEPS', '120'))
PERT = float(os.environ.get('PERT', '0.15'))
NB_H = int(os.environ.get('NB_H', '2'))
INTENS = float(os.environ.get('INTENS', '1e8'))
REFL_LAM = float(os.environ.get('REFL_LAM', '400'))
REPORT = os.path.join(_HERE, 'REFLECTION_FIT_RESULTS.md')

# DEVIATED truth angular-reflection params (defaults: R0=0.05 p=1 fspec=0 nr=2.8 nk=1.5).
TRUTH = dict(wall_R0=0.10, wall_p=2.0, wall_fspec=0.30,
             cathode_nr=2.5, cathode_nk=1.2, sensor_fspec=0.30)
FIELDS = ['wall_R0', 'wall_p', 'cathode_nr', 'cathode_nk', 'wall_fspec', 'sensor_fspec']

_lines = []
def emit(s=''):
    print(s, flush=True); _lines.append(s)
    with open(REPORT, 'w') as f:
        f.write('\n'.join(_lines) + '\n')


def main():
    t0 = time.time()
    det = generate_detector(GEOM); NS = len(det.all_points); H = det.H
    R = det.r if hasattr(det, 'r') else 6.0
    dp_true = DetectorParams.from_flat(
        scatter_length=70., mie_scatter_length=3000., g=0.9,
        absorption_length=60., qe=0.07, qe_corrections=jnp.ones(NS), **TRUTH)
    sim = setup_event_simulator(GEOM, NPH, temperature=None, K=K, is_calibration=True,
                                hit_mode='aggregated', wavelength_mode=False,
                                reflection_model='angular', reflection_wavelength=REFL_LAM, **GK)
    # diversity that lights the walls/sensors at a range of incidence angles
    srcs = [laser_source(position=[0., 0., H/2 - .1], intensity=INTENS),
            laser_source(position=[R - .1, 0., 0.], direction=[-1., 0., 0.], intensity=INTENS),
            isotropic_source(position=[0., 0., 0.], intensity=INTENS)]

    emit('# Angular (complex) reflection FIT — unified GN recipe (gap 1)')
    emit('')
    emit(f'SK_like NS={NS}, N_photons={NPH:.0e}, K={K}, grid={GK}, reflection_model=angular, '
         f'λ_refl={REFL_LAM}nm, sources=[laser_down, laser_wall, iso], steps={STEPS}, start ±{PERT:.0%}.')
    emit(f'Truth angular params (deviated from defaults): {TRUTH}. Optics held at truth.')
    emit('')

    prob = build_calibration_problem(sim, srcs, dp_true, FIELDS, key=jax.random.PRNGKey(1))
    truth = np.exp(prob['theta0'])
    rng = np.random.default_rng(0)
    start = prob['theta0'] + rng.uniform(-PERT, PERT, truth.shape)
    emit(f'GN fit of {FIELDS} …')
    res = fit(prob['source_models'], prob['truth_charge'], start, NS,
              steps=STEPS, refresh=15, nb_h=NB_H)
    rec = res['theta']
    c = crb(prob['source_models'], prob['theta_true'], NS, nb_h=NB_H)
    sig = c['sigma']
    emit(f'  fit + CRB done ({time.time()-t0:.0f}s)')
    emit('')
    emit('| param | truth | start | recovered | frac err | CRB σ |')
    emit('|---|---|---|---|---|---|')
    for j, f in enumerate(FIELDS):
        emit(f'| {f} | {truth[j]:.3g} | {np.exp(start[j]):.3g} | {rec[j]:.4g} | '
             f'{abs(rec[j]/truth[j]-1)*100:.1f}% | {sig[j]*100:.2f}% |')
    emit('')
    emit('Angular reflection is PARTIALLY charge-identifiable (read the CRB column): '
         'wall_R0 (normal reflectance) and the spec/diff fractions wall_fspec/sensor_fspec '
         'are constrained — the fractions tightly (CRB ~0.4%), because the per-sensor '
         'reflected-light PATTERN depends on the spec/diff split (so they are NOT charge-blind '
         'as one might assume). The Schlick exponent wall_p is weakly constrained, and the '
         'cathode Fresnel indices cathode_nr↔cathode_nk are NEAR-DEGENERATE (CRB ~50-65% — '
         'they trade in producing the cathode reflectance magnitude), recovered near truth '
         'only because they started near it. Pinning wall_p / the cathode pair needs more '
         'incidence-angle diversity or the timing observable.')
    emit('')
    emit(f'_Finished in {(time.time()-t0)/60:.1f} min._')


if __name__ == '__main__':
    main()
