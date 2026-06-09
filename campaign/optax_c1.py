"""C1 scalar-charge calibration through the DEFAULT pytree-native optax fit, on the REAL
SK_like photon forward — recover the 7 optical/reflection globals + per-PMT k from a
perturbed start, and compare to the fit_gn campaign reference.

This is the end-to-end proof that the simple ``loss + jax.grad + optax`` path (reverse-mode
through the DiCE custom_vjp forward) calibrates the real detector — no flatten, no Schur,
no subset. Two sources (laser + iso) break the per-PMT-k ↔ global degeneracy. Fit EVERY
leaf: the dead ones (dev curves in mono mode, angular-refl, gain/t0 under a charge loss)
get ≈0 gradient and self-hold; only the 7 live globals + qe_corrections move.

Usage: CUDA_VISIBLE_DEVICES=<g> python campaign/optax_c1.py
Env: NPH, K, STEPS, LR, KSPREAD (truth per-PMT spread), PERT (global start perturbation).
"""
import os, sys, time
import numpy as np
import jax
import jax.numpy as jnp
import optax

_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources import isotropic_source, laser_source
from lucid.detector_params import DetectorParams
from lucid.fitting import make_loss, fit, charge, gauge_mean_log

GEOM = os.path.join(_ROOT, 'config', 'SK_like_geom_config.json')
GK = dict(n_cap=100, n_angular=150, n_height=100)
NPH = int(float(os.environ.get('NPH', '3e5')))
K = int(os.environ.get('K', '6'))
STEPS = int(os.environ.get('STEPS', '500'))
LR = float(os.environ.get('LR', '2e-2'))
KSPREAD = float(os.environ.get('KSPREAD', '0.12'))
PERT = float(os.environ.get('PERT', '0.20'))
REPORT = os.path.join(_HERE, 'OPTAX_C1_RESULTS.md')

# fit_gn campaign reference (CAMPAIGN_RESULTS, full-stats): well-determined → ~CRB.
GN_REF = {'L_abs': '~4.9%', 'qe': '~1.9%', 'sensor_refl': '~2.8%', 'wall_refl': '~3%',
          'L_R': 'tight', 'L_M': '~17% (thin Mie, hard)', 'g': 'loose'}

_lines = []
def emit(s=''):
    print(s, flush=True); _lines.append(s)
    with open(REPORT, 'w') as f:
        f.write('\n'.join(_lines) + '\n')

_GLOBALS = [('L_R', lambda d: d.scattering.scatter_length),
            ('L_M', lambda d: d.scattering.mie_scatter_length),
            ('g', lambda d: d.scattering.g),
            ('L_abs', lambda d: d.absorption.absorption_length),
            ('wall_refl', lambda d: d.reflection.wall_reflection_rate),
            ('sensor_refl', lambda d: d.reflection.sensor_reflection_rate),
            ('qe', lambda d: d.response.qe)]


def _project(dp):
    """Gauge per-PMT k (mean(log k)=0) + clip globals to physical/positive ranges."""
    dp = gauge_mean_log(dp, 'qe_corrections')
    s, a, r, rp = dp.scattering, dp.absorption, dp.reflection, dp.response
    return dp._replace(
        scattering=s._replace(scatter_length=jnp.clip(s.scatter_length, 1., None),
                              mie_scatter_length=jnp.clip(s.mie_scatter_length, 1., None),
                              g=jnp.clip(s.g, 0., 0.999)),
        absorption=a._replace(absorption_length=jnp.clip(a.absorption_length, 1., None)),
        reflection=r._replace(wall_reflection_rate=jnp.clip(r.wall_reflection_rate, 0., 0.99),
                              sensor_reflection_rate=jnp.clip(r.sensor_reflection_rate, 0., 0.99)),
        response=rp._replace(qe=jnp.clip(rp.qe, 1e-3, 1.)))


def main():
    t0 = time.time()
    det = generate_detector(GEOM); NS = len(det.all_points); H = det.H
    rng = np.random.default_rng(0)
    k_true = np.exp(KSPREAD * rng.standard_normal(NS)); k_true /= np.exp(np.mean(np.log(k_true)))

    dp_true = DetectorParams.from_flat(
        scatter_length=70., mie_scatter_length=3000., g=0.9,
        wall_reflection_rate=0.20, sensor_reflection_rate=0.20,
        absorption_length=60., qe=0.07, qe_corrections=jnp.asarray(k_true))

    sim = setup_event_simulator(GEOM, NPH, temperature=None, K=K, is_calibration=True,
                                hit_mode='aggregated', wavelength_mode=False, **GK)
    srcs = [laser_source(position=[0., 0., H/2 - 0.1], intensity=NPH),
            isotropic_source(position=[0., 0., 0.], intensity=NPH)]
    obs = [sim(s, dp_true, jax.random.PRNGKey(1)) for s in srcs]
    loss = make_loss(sim, srcs, obs, terms=[charge])

    # perturbed start: globals ×(1±PERT), per-PMT k = 1
    pf = {n: float(np.asarray(g(dp_true))) * (1 + PERT * (1 if i % 2 else -1))
          for i, (n, g) in enumerate(_GLOBALS)}
    dp0 = DetectorParams.from_flat(
        scatter_length=pf['L_R'], mie_scatter_length=pf['L_M'], g=min(pf['g'], 0.98),
        wall_reflection_rate=pf['wall_refl'], sensor_reflection_rate=pf['sensor_refl'],
        absorption_length=pf['L_abs'], qe=pf['qe'], qe_corrections=jnp.ones(NS))

    emit('# C1 scalar-charge calibration — pytree-native optax fit on the real SK_like forward')
    emit('')
    emit(f'NS={NS}, N_photons={NPH:.0e}, K={K}, sources=[laser_down, iso], '
         f'steps={STEPS}, lr={LR}, optimizer=adam, project=gauge_mean_log+clip.')
    emit(f'Truth per-PMT k: log-normal spread {KSPREAD:.0%} (gauged mean(log k)=0). '
         f'Global start perturbation ±{PERT:.0%}.')
    emit('')
    emit(f'loss at truth = {float(loss(dp_true, jax.random.PRNGKey(1))):.3e}; '
         f'at start = {float(loss(dp0, jax.random.PRNGKey(1))):.3e}')
    emit('')

    # fit with periodic logging
    opt = optax.adam(LR)
    state = opt.init(dp0)
    dp = dp0

    @jax.jit
    def step(dp, state, k):
        g = jax.grad(loss)(dp, k)
        upd, state = opt.update(g, state, dp)
        dp = optax.apply_updates(dp, upd)
        return _project(dp), state

    log_every = max(1, STEPS // 10)
    for s in range(STEPS):
        dp, state = step(dp, state, jax.random.PRNGKey(1))
        if s % log_every == 0 or s == STEPS - 1:
            l = float(loss(dp, jax.random.PRNGKey(1)))
            emit(f'  step {s:4d}: loss={l:.3e}  L_abs={float(dp.absorption.absorption_length):6.2f}  '
                 f'qe={float(dp.response.qe):.4f}  L_R={float(dp.scattering.scatter_length):6.2f}  '
                 f'({time.time()-t0:.0f}s)')

    emit('')
    emit('## Recovery (optax) vs fit_gn campaign reference')
    emit('')
    emit('| param | truth | start | optax fit | frac err | fit_gn ref |')
    emit('|---|---|---|---|---|---|')
    for n, g in _GLOBALS:
        tv = float(np.asarray(g(dp_true))); sv = pf[n]; hv = float(np.asarray(g(dp)))
        fe = abs(hv / tv - 1.0)
        emit(f'| {n} | {tv:.3g} | {sv:.3g} | {hv:.4g} | {fe*100:.1f}% | {GN_REF[n]} |')
    k_hat = np.asarray(dp.per_pmt.qe_corrections)
    emit('')
    emit(f'per-PMT k: corr(k̂, k_true) = {np.corrcoef(k_hat, k_true)[0,1]:.4f}, '
         f'RMS frac = {np.sqrt(np.mean((k_hat/k_true - 1)**2))*100:.1f}% '
         f'(truth spread {KSPREAD:.0%})')
    emit('')
    emit(f'_Finished in {(time.time()-t0)/60:.1f} min._')


if __name__ == '__main__':
    main()
