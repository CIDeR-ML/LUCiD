"""Rigorous per-item AD-vs-FD analysis.
A) wavelength_mode: is the SCALAR scatter_length truly inert in λ-mode (bit-identical under a big
   perturbation)? + categorize the λ-mode DEVIATION curves (abs_dev pathwise vs rayleigh_dev discrete).
B) h-scan (single key): pathwise param -> FD->AD as h->0 (TRUNCATION, O(h^2)); discrete param ->
   FD(h) plateaus at a value != AD (ESTIMATOR difference, not truncation).
C) key-scan n=1..256: do AD (score) and FD (reparam) share an EXPECTATION? (both unbiased <=> converge)
"""
import os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.sources.calibration_sources import laser_source
from lucid.fitting import build_calibration_problem

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
NPHOT = int(os.environ.get('NPHOT', '30000')); K = 8
det = generate_detector(GEOM); ND = len(det.all_points); Hd = float(getattr(det, 'H', 36.0))
dp = load_detector_params(PHYS, num_sensors=ND)
src = laser_source(position=[0.0, 0.0, Hd / 2 - 0.1], intensity=NPHOT)


def make(field, wlmode):
    sim = setup_event_simulator(GEOM, NPHOT, temperature=0.2, K=K, is_calibration=True,
                                physics_config=PHYS, hit_mode='aggregated', wavelength_mode=wlmode)
    prob = build_calibration_problem(sim, [src], dp, [field], key=jax.random.PRNGKey(0))
    fwd = prob['source_models'][0].forward; th0 = jnp.asarray(prob['theta0'], float)
    tgt = jax.lax.stop_gradient(fwd(th0, jax.random.PRNGKey(0), jax.random.PRNGKey(0)))
    Wm = jnp.asarray((np.asarray(tgt) > 0).astype(float))

    def loss(t, key):
        M = fwd(t, key, key)
        return 0.5 * jnp.sum(Wm * (jnp.sqrt(M + 1e-8) - jnp.sqrt(tgt + 1e-8)) ** 2)
    return fwd, th0, loss, tgt


# ---------- A) wavelength_mode: scalars inert? + deviation-curve categories ----------
print('=== A) wavelength_mode ===')
fwd, th0, _, tgt = make('scatter_length', True)
big = th0.at[0].add(1.0)                       # ×e perturbation to scatter_length (log-space)
M0 = np.asarray(fwd(th0, jax.random.PRNGKey(5), jax.random.PRNGKey(5)))
M1 = np.asarray(fwd(big, jax.random.PRNGKey(5), jax.random.PRNGKey(5)))
print(f'  λ-mode forward |ΔM| under ×e scatter_length pert (same key): {np.abs(M0-M1).max():.3e}  '
      f'-> scalar is {"INERT (dead)" if np.abs(M0-M1).max() < 1e-9 else "LIVE"}')
for f in ['abs_dev', 'rayleigh_dev', 'qe_dev']:
    fwd, th0, loss, _ = make(f, True)
    k = jax.random.PRNGKey(7); ga = float(jax.grad(loss)(th0, k)[0])
    e = jnp.zeros_like(th0).at[0].set(1e-3); gf = float((loss(th0 + e, k) - loss(th0 - e, k)) / 2e-3)
    print(f'  λ-mode dev curve {f:14s}: AD={ga:+.4f} FD={gf:+.4f}  {"PATHWISE" if abs(ga-gf)<0.02*abs(gf)+1e-6 else "score/disc"}')

# ---------- B) h-scan: truncation (pathwise) vs estimator (discrete) ----------
print('\n=== B) h-scan (single key, mono mode): FD(h) vs AD ===')
for f in ['absorption_length', 'scatter_length']:
    fwd, th0, loss, _ = make(f, False); k = jax.random.PRNGKey(11)
    ad = float(jax.grad(loss)(th0, k)[0])
    row = []
    for h in [1e-2, 1e-3, 1e-4, 1e-5]:
        e = jnp.zeros_like(th0).at[0].set(h)
        row.append(f'h={h:.0e}:{float((loss(th0+e,k)-loss(th0-e,k))/(2*h)):+8.3f}')
    print(f'  {f:18s} AD={ad:+8.3f} | ' + '  '.join(row))

# ---------- C) key-scan: do AD and FD share an expectation? ----------
print('\n=== C) key-scan (mono mode): AD vs FD averaged over n keys ===')
for f in ['scatter_length', 'mie_scatter_length']:
    fwd, th0, loss, _ = make(f, False)
    gad = jax.jit(jax.grad(loss))
    def gf(key, h=1e-3):
        e = jnp.zeros_like(th0).at[0].set(h); return float((loss(th0+e,key)-loss(th0-e,key))/(2*h))
    row = []
    for n in [1, 16, 64, 256]:
        ks = [jax.random.PRNGKey(2000 + i) for i in range(n)]
        a = np.mean([float(gad(th0, kk)[0]) for kk in ks]); b = np.mean([gf(kk) for kk in ks])
        row.append(f'n={n:<3d} AD={a:+7.2f} FD={b:+8.2f}')
    print(f'  {f:18s}\n    ' + '\n    '.join(row))
