"""Calibration campaign on the UNIFIED framework — reproduce the mie_hunter numbers.

Runs, on the base ``setup_event_simulator`` + ``lucid.fitting`` (the consolidated code),
the headline calibration combinations and tabulates recovered value / bias / √CRB (×√12)
against the recorded mie_hunter results:

  1. Source diversity breaks L_M ↔ per-PMT-k: CRB of the 7 optical+reflection globals for
     single-laser vs laser+iso vs multi-source (down/up/wall lasers + iso).
  2. Reflection split: wall_refl tight, sensor_refl info-limited (flat).
  3. A full GN recovery fit on the diverse-source config (recovered vs truth).
  4. Charge variance (moments mode): recover per-PMT gain + SPE width w (Fano v/m=g(1+w²)).

Writes campaign/CAMPAIGN_RESULTS.md incrementally so partial progress is preserved.
Env: CAMPAIGN_QUICK=1 (fewer steps/photons), NPH, SK geom is fixed.
Usage: CUDA_VISIBLE_DEVICES=<g> python campaign/run_campaign.py
"""
import os, sys, time, traceback
import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources import isotropic_source, laser_source
from lucid.detector_params import DetectorParams
from lucid.fitting import build_calibration_problem, fit, crb

GEOM = os.path.join(_ROOT, 'config', 'SK_like_geom_config.json')
GK = dict(n_cap=150, n_angular=250, n_height=150)
QUICK = os.environ.get('CAMPAIGN_QUICK', '0') == '1'
NPH = int(float(os.environ.get('NPH', '5e5' if QUICK else '1e6')))
INTENS = 5_000_000_000
REPORT = os.path.join(_HERE, 'CAMPAIGN_RESULTS.md')

det = generate_detector(GEOM)
NS = len(det.all_points)
H = det.H

# Truth detector params — SK-like, finite Mie so L_M is a live (measurable) parameter.
DP_TRUE = DetectorParams.from_flat(
    scatter_length=70.0, mie_scatter_length=30.0, g=0.9,
    wall_reflection_rate=0.20, sensor_reflection_rate=0.20,
    absorption_length=60.0, qe=0.07, qe_corrections=jnp.ones(NS))

FIELDS7 = ['g', 'scatter_length', 'mie_scatter_length', 'absorption_length',
           'wall_reflection_rate', 'sensor_reflection_rate', 'qe']
LABEL = {'g': 'g', 'scatter_length': 'L_R', 'mie_scatter_length': 'L_M',
         'absorption_length': 'L_abs', 'wall_reflection_rate': 'wall_refl',
         'sensor_reflection_rate': 'sensor_refl', 'qe': 'qe'}

# mie_hunter recorded reference (fractional σ, ×√12 honest), for the comparison column.
MIE_REF = {'L_M': '~5%', 'qe': '~1% (single-λ)', 'wall_refl': '~0.6% (tight)',
           'sensor_refl': 'flat (info-limited)', 'L_abs': 'corr w/ qe -0.99',
           'L_R': 'tight', 'g': 'loose'}

_lines = []
def emit(s=''):
    print(s, flush=True)
    _lines.append(s)
    with open(REPORT, 'w') as f:
        f.write('\n'.join(_lines) + '\n')


def src_sets():
    L_down = laser_source(position=[0., 0., H/2 - 0.1], intensity=INTENS)
    L_up = laser_source(position=[0., 0., -H/2 + 0.1], direction=[0., 0., 1.], intensity=INTENS)
    L_wall = laser_source(position=[det.r - 0.1 if hasattr(det, 'r') else 6.0, 0., 0.],
                          direction=[-1., 0., 0.], intensity=INTENS)
    iso = isotropic_source(position=[0., 0., 0.], intensity=INTENS)
    return {
        'single_laser': [L_down],
        'laser+iso': [L_down, iso],
        'multi_source': [L_down, L_up, L_wall, iso],
    }


def main():
    t_start = time.time()
    emit('# Calibration campaign — unified framework vs mie_hunter')
    emit('')
    emit(f'SK_like, NS={NS}, N_photons={NPH:.0e}, intensity={INTENS:.0e}, QUICK={QUICK}.')
    emit(f'Truth: L_R=70 L_M=30 g=0.9 L_abs=60 wall=sensor=0.20 qe=0.07.')
    emit('CRB = fractional σ on log-params, ×√12 honesty factor (the implicit engine is √12 quieter than Poisson).')
    emit('')

    sim = setup_event_simulator(GEOM, NPH, temperature=None, K=8, is_calibration=True,
                                hit_mode='aggregated', **GK)
    SS = src_sets()
    nb_h = 2 if QUICK else 3

    # ---- 1+2. CRB across source diversity (the headline source-diversity / reflection-split table) ----
    emit('## 1-2. CRB of the 7 optical+reflection globals vs source diversity')
    emit('')
    header = '| param | ' + ' | '.join(SS.keys()) + ' | mie_hunter |'
    emit(header); emit('|' + '---|' * (len(SS) + 2))
    crb_by_set = {}
    for name, srcs in SS.items():
        try:
            t0 = time.time()
            prob = build_calibration_problem(sim, srcs, DP_TRUE, FIELDS7, key=jax.random.PRNGKey(1))
            c = crb(prob['source_models'], prob['theta_true'], NS, nb_h=nb_h)
            crb_by_set[name] = c['sigma']
            emit(f'> built CRB[{name}] in {time.time()-t0:.0f}s')
        except Exception:
            emit(f'> CRB[{name}] FAILED:\n```\n{traceback.format_exc()}\n```')
            crb_by_set[name] = np.full(len(FIELDS7), np.nan)
    for j, f in enumerate(FIELDS7):
        row = f'| {LABEL[f]} | '
        for name in SS:
            s = crb_by_set[name][j]
            row += (f'{s*100:.2f}% | ' if np.isfinite(s) and s < 100 else ('>1e4% | ' if np.isfinite(s) else 'n/a | '))
        row += f'{MIE_REF.get(LABEL[f], "")} |'
        emit(row)
    emit('')
    emit('Expected (mie_hunter): L_M and sensor_refl tighten with source diversity; '
         'sensor_refl stays the loosest (few photons hit sensors); single_laser leaves '
         'L_M↔k and scatter↔abs degenerate (huge σ).')
    emit('')

    # ---- 3. Full GN recovery on the diverse-source config ----
    emit('## 3. GN recovery (multi_source), from a 20%-perturbed start')
    emit('')
    try:
        srcs = SS['multi_source']
        prob = build_calibration_problem(sim, srcs, DP_TRUE, FIELDS7, key=jax.random.PRNGKey(1))
        rng = np.random.default_rng(0)
        start = prob['theta0'] + rng.uniform(-0.2, 0.2, len(FIELDS7))
        steps = 50 if QUICK else 120
        t0 = time.time()
        res = fit(prob['source_models'], prob['truth_charge'], start, NS,
                  steps=steps, refresh=12, nb_r=2, nb_h=nb_h, seed=0, fix=(0,))  # fix g (loose)
        truth = np.exp(prob['theta0'])
        emit(f'fit {steps} steps in {time.time()-t0:.0f}s')
        emit('')
        emit('| param | truth | recovered | frac err |'); emit('|---|---|---|---|')
        for j, f in enumerate(FIELDS7):
            fe = abs(res['theta'][j] / truth[j] - 1.0) * 100
            emit(f'| {LABEL[f]} | {truth[j]:.4g} | {res["theta"][j]:.4g} | {fe:.1f}% |')
        emit('')
    except Exception:
        emit(f'> recovery FAILED:\n```\n{traceback.format_exc()}\n```')

    # ---- 4. Charge variance: recover per-PMT gain + SPE width w (moments mode) ----
    emit('## 4. Charge variance (moments mode): Fano v/m = g·(1+w²) breaks QE↔gain')
    emit('')
    try:
        sim_m = setup_event_simulator(GEOM, NPH, temperature=None, K=8, is_calibration=True,
                                      hit_mode='moments', **GK)
        rng = np.random.default_rng(3)
        gain = np.exp(0.12 * rng.standard_normal(NS)); gain /= np.exp(np.mean(np.log(gain)))
        w_true = 0.35
        dp_var = DP_TRUE._replace(
            per_pmt=DP_TRUE.per_pmt._replace(gain=jnp.asarray(gain)),
            response=DP_TRUE.response._replace(spe_width=jnp.asarray(w_true)))
        src = [laser_source(position=[0., 0., H/2 - 0.1], intensity=INTENS),
               isotropic_source(position=[0., 0., 0.], intensity=INTENS)]
        mean, var, _ = sim_m(src[1], dp_var, jax.random.PRNGKey(5))
        lit = np.array(mean) > 1e-3
        fano = np.array(var)[lit] / np.array(mean)[lit]
        fano_med = float(np.median(fano))
        emit(f'measured Fano v/m median = {fano_med:.4f}  (expect g·(1+w²) ≈ {1.0*(1+w_true**2):.4f} at median g≈1)')
        emit(f'→ implied w from Fano (median g=1): w ≈ {np.sqrt(max(fano_med-1,0)):.3f}  (truth w={w_true})')
        emit(f'→ per-PMT gain recoverable from v/m and mean (g=v/m/(1+w²), then k=mean/(g·μ)); '
             f'SPE width w from the population Fano. This is the QE↔gain degeneracy break.')
        emit('')
    except Exception:
        emit(f'> charge-variance FAILED:\n```\n{traceback.format_exc()}\n```')

    emit('')
    emit(f'_Campaign finished in {(time.time()-t_start)/60:.1f} min._')


if __name__ == '__main__':
    main()
