"""DECISIVE bias-vs-variance test for the discrete-decision params.
Clean LINEAR functional L(θ,k)=Σ cᵢ·Mᵢ(θ,k), fixed random c → grad = Σ cᵢ ∂Mᵢ/∂θ, well-defined
nonzero expectation. Compare AD (score) vs FD (CRN reparam) means + std over many keys: if they
converge to the SAME mean → both unbiased, AD just lower-variance; if DIFFERENT → one is biased."""
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
FIELD = os.environ.get('FIELD', 'scatter_length'); NMAX = int(os.environ.get('NMAX', '2048'))
det = generate_detector(GEOM); ND = len(det.all_points); Hd = float(getattr(det, 'H', 36.0))
dp = load_detector_params(PHYS, num_sensors=ND)
src = laser_source(position=[0.0, 0.0, Hd / 2 - 0.1], intensity=NPHOT)
sim = setup_event_simulator(GEOM, NPHOT, temperature=0.2, K=K, is_calibration=True,
                            physics_config=PHYS, hit_mode='aggregated', wavelength_mode=False)
prob = build_calibration_problem(sim, [src], dp, [FIELD], key=jax.random.PRNGKey(0))
fwd = prob['source_models'][0].forward; th0 = jnp.asarray(prob['theta0'], float)
c = jnp.asarray(np.random.default_rng(0).standard_normal(ND))     # fixed linear functional


def L(t, key):
    return jnp.sum(c * fwd(t, key, key))


gad = jax.jit(jax.grad(L))
H = 1e-3


def gfd(key):
    e = jnp.zeros_like(th0).at[0].set(H)
    return float((L(th0 + e, key) - L(th0 - e, key)) / (2 * H))


print(f'field={FIELD} Nphot={NPHOT} | linear functional grad, AD (score) vs FD (CRN)')
A = np.array([float(gad(th0, jax.random.PRNGKey(5000 + i))[0]) for i in range(NMAX)])
F = np.array([gfd(jax.random.PRNGKey(5000 + i)) for i in range(NMAX)])
for n in [1, 16, 64, 256, 1024, NMAX]:
    am, asd = A[:n].mean(), A[:n].std() / np.sqrt(n)
    fm, fsd = F[:n].mean(), F[:n].std() / np.sqrt(n)
    print(f'  n={n:<5d} AD={am:+8.3f}±{asd:5.3f}   FD={fm:+9.3f}±{fsd:6.3f}   '
          f'|AD-FD|/σ_FD={abs(am-fm)/(fsd+1e-9):5.1f}')
print(f'\n  per-key std: AD={A.std():.3f}  FD={F.std():.3f}  -> FD variance is {(F.std()/A.std())**2:.0f}x AD')
print(f'  CONVERGED MEANS: AD={A.mean():+.3f}  FD={F.mean():+.3f}  '
      f'-> {"SAME (both unbiased; AD lower-var)" if abs(A.mean()-F.mean())<2*F.std()/np.sqrt(NMAX) else "DIFFERENT (bias!)"}')
