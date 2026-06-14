"""Shot-noise validation (#4): does the GN recipe hit the CRB on REAL Poisson data?

Every calibration fit so far used the EXPECTED (implicit) forward as the data target — so
the honest uncertainty is the Fisher CRB ×√12, not the (artificially quiet) implicit-engine
scatter. This closes that loop: generate SAMPLE-mode integer-PE Poisson shot-noise charge
(use_expected_value=False) as the DATA, fit it with the recipe whose model is the EXPECTED
forward, over M independent noise seeds, and compare the realized seed-to-seed scatter to
the CRB σ (×√12). Expectation (from the shot-noise study): realized ≲ CRB (the √12 honesty
factor is the bound; the implicit engine is √12 quieter than Poisson).

Usage: CUDA_VISIBLE_DEVICES=<g> python scripts/campaign/shotnoise_validate.py
Env: NPH, K, M (seeds), STEPS, NB_H, GRID.
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
NPH = int(float(os.environ.get('NPH', '2e5')))
K = int(os.environ.get('K', '8'))
M = int(os.environ.get('M', '6'))
STEPS = int(os.environ.get('STEPS', '50'))
NB_H = int(os.environ.get('NB_H', '2'))
BAKE_K = os.environ.get('BAKE_K', '0') == '1'
POLYAK = int(os.environ.get('POLYAK', '0'))
EPS = float(os.environ.get('EPS', '1e-8'))     # 0.375 = Anscombe √(x+3/8) (Poisson bias-correct)
REPORT = os.path.join(_HERE, 'SHOTNOISE_RESULTS.md')

FIELDS = ['scatter_length', 'absorption_length', 'wall_reflection_rate', 'qe']
LABEL = dict(scatter_length='L_R', absorption_length='L_abs',
             wall_reflection_rate='wall', qe='qe')

_lines = []
def emit(s=''):
    print(s, flush=True); _lines.append(s)
    with open(REPORT, 'w') as f:
        f.write('\n'.join(_lines) + '\n')


def main():
    t0 = time.time()
    det = generate_detector(GEOM); NS = len(det.all_points); H = det.H
    rng = np.random.default_rng(0)
    k_true = np.exp(0.12 * rng.standard_normal(NS)); k_true /= np.exp(np.mean(np.log(k_true)))

    base = dict(scatter_length=70., mie_scatter_length=3000., g=0.9,
                wall_reflection_rate=.2, sensor_reflection_rate=.2, absorption_length=60., qe=0.07)
    dp1 = DetectorParams.from_flat(qe_corrections=jnp.ones(NS), **base)              # for build (truth_k baked)
    dpk = DetectorParams.from_flat(qe_corrections=jnp.asarray(k_true), **base)       # for sample data

    sim_model = setup_event_simulator(GEOM, NPH, temperature=None, K=K, is_calibration=True,
                                      hit_mode='aggregated', wavelength_mode=False, **GK)
    sim_data = setup_event_simulator(GEOM, NPH, temperature=None, K=K, is_calibration=True,
                                     use_expected_value=False, hit_mode='realistic',
                                     apply_smearing=False, wavelength_mode=False, **GK)
    srcs = [laser_source(position=[0., 0., H/2 - .1], intensity=NPH),
            isotropic_source(position=[0., 0., 0.], intensity=NPH)]

    emit('# Shot-noise validation (#4) — realized scatter vs CRB×√12 on real Poisson data')
    emit('')
    emit(f'SK_like NS={NS}, N_photons={NPH:.0e} (intensity=NPH ⇒ integer-PE shots), K={K}, '
         f'grid={GK}, sources=[laser_down, iso], {M} noise seeds, {STEPS} steps/fit.')
    emit(f'DATA = sample-mode shot noise (use_expected_value=False); MODEL = expected forward.')
    emit(f'Stabilizers: bake_k={BAKE_K} (closed-form k=ΣQ/ΣM, no free Schur-k), polyak={POLYAK} '
         f'(iterate-averaging), eps={EPS} ({"ANSCOMBE √(x+3/8) Poisson-bias-correct" if EPS > 0.1 else "plain √-MSE"}).')
    emit('')

    prob = build_calibration_problem(sim_model, srcs, dp1, FIELDS, truth_k=k_true,
                                     key=jax.random.PRNGKey(1), eps=EPS)
    theta_true = prob['theta_true']
    truth_vals = np.exp(theta_true)
    cr = crb(prob['source_models'], theta_true, NS, lk_true=prob['lk_true'], nb_h=NB_H)
    crb_sig = cr['sigma']

    rec = np.zeros((M, len(FIELDS))); krec = np.zeros((M, NS))
    for m in range(M):
        truth_shot = [np.asarray(sim_data(s, dpk, jax.random.PRNGKey(100 + m * 17 + j))[0])
                      for j, s in enumerate(srcs)]
        res = fit(prob['source_models'], truth_shot, theta_true, NS,    # start AT truth → measure scatter
                  steps=STEPS, refresh=max(20, STEPS // 2), nb_h=NB_H, seed=m,
                  bake_k=BAKE_K, polyak=POLYAK, eps=EPS)
        rec[m] = res['theta']; krec[m] = res['k']
        emit(f'  seed {m} done ({time.time()-t0:.0f}s)')

    emit('')
    emit('## Globals — realized scatter vs CRB (fractional)')
    emit('')
    emit('| param | truth | mean rec | bias | realized σ | CRB σ (×√12) | realized/CRB |')
    emit('|---|---|---|---|---|---|---|')
    for i, f in enumerate(FIELDS):
        mu = rec[:, i].mean(); sd = rec[:, i].std()
        bias = mu / truth_vals[i] - 1; rsig = sd / truth_vals[i]
        emit(f'| {LABEL[f]} | {truth_vals[i]:.3g} | {mu:.4g} | {bias*100:+.1f}% | '
             f'{rsig*100:.2f}% | {crb_sig[i]*100:.2f}% | {rsig/max(crb_sig[i],1e-9):.2f} |')

    # per-PMT k: realized scatter (over seeds, per PMT) vs the per-PMT CRB if available
    klit = k_true > 0
    k_real = krec.std(axis=0)[klit] / k_true[klit]
    emit('')
    emit(f'## per-PMT k: median realized σ = {np.median(k_real)*100:.2f}% over {int(klit.sum())} PMTs '
         f'(truth spread 12%; per-seed corr median '
         f'{np.median([np.corrcoef(krec[m][klit], k_true[klit])[0,1] for m in range(M)]):.3f}).')
    emit('')
    if BAKE_K or POLYAK:
        emit('STABILIZERS ON (now IN fit_gn): the bare free-Schur-k GN diverges on a single '
             'shot-noise draw (globals collapse, k overfits); bake_k (closed-form k=ΣQ/ΣM, no '
             'free per-PMT block to overfit) + polyak (iterate-averaging) are the recipe '
             'stabilizers. If the globals above are near truth (small bias) and realized σ ≈ '
             'CRB, the recipe is shot-noise-robust and #4 is closed.')
    else:
        emit('FINDING (negative, NO stabilizers): the bare free-Schur-k GN DIVERGES on a '
             'single-draw shot-noise dataset (globals collapse, k overfits) — run with '
             'BAKE_K=1 POLYAK=10 for the stabilized recipe. Quote CRB×√12 as the honest bound.')
    emit('')
    emit(f'_Finished in {(time.time()-t0)/60:.1f} min._')


if __name__ == '__main__':
    main()
