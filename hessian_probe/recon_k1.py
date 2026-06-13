"""Pin the K=1 BASE breaker (eps-independent indefinite Hessian, 8/9 neg-diag at K=1).
Sweep the deposit Gaussian TEMPERATURE at K=1 (surf/wall eps off, they don't fire at K=1).
Hypothesis: the soft-overlap deposit exp(-d^2/2 temp^2) has curvature ~1/temp^4 -> if min-eig
scales as temp^-4 the deposit Gaussian IS the base breaker (not a bug, the soft-overlap width).
Also reports grad_rel. Single config per process via env TEMP, K."""
import os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
import lucid.simulation.simulator as SIM, lucid.simulation.photon_step as PS
def plain_factory(reflection_fn=PS.scalar_reflection):
    def step(p, d, t, sd, n, sl, ml, g, rp, al, hs, lam, rk, cc):
        return PS.photon_iteration_update_factors(p, d, t, sd, n, sl, ml, g, rp, al, hs, lam, rk, cc, reflection_fn=reflection_fn)
    return step
SIM.make_photon_iteration_update_factors_safe = plain_factory
from lucid.fitting.recon import track_from_vec9, vec9_from_track
from lucid.geometry import generate_detector
GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
NPH = 40000; K = int(os.environ.get('K', '1')); MC = 4; TEMP = float(os.environ.get('TEMP', '0.1'))
GRID = dict(n_cap=80, n_angular=120, n_height=80)
ND = len(generate_detector(GEOM).all_points)
c = jnp.asarray(np.random.default_rng(0).standard_normal(ND))
t9 = jnp.asarray(vec9_from_track(1050., [2.0, -1.0, 3.0], [0.2, 0.1, 0.97], 0.0), float)
pred = SIM.setup_event_simulator(GEOM, NPH, temperature=TEMP, K=K, hit_mode='per_photon',
    physics_config=PHYS, default_detector_params=True, particle='muon', wavelength_mode=True,
    pos_grad_threshold=K, n_grad_iters=K, max_sensors_per_cell=MC, **GRID)
def L(t): return jnp.sum(c * pred(track_from_vec9(t), jax.random.PRNGKey(3))[3])
H = np.asarray(jax.hessian(L)(t9)); g = np.asarray(jax.grad(L)(t9))
h = 1e-3; gF = np.array([float((L(t9.at[i].add(h)) - L(t9.at[i].add(-h)))/(2*h)) for i in range(9)])
ev = np.sort(np.linalg.eigvalsh(0.5*(H+H.T)))
print(f'TEMP={TEMP} K={K} | L={float(L(t9)):.4f} eig[min,max]=[{ev.min():.2e},{ev.max():.2e}] '
      f'neg-eig={int((ev<-1.0).sum())}/9 neg-diag={int((np.diag(H)<0).sum())}/9 |H|={np.linalg.norm(H):.1e} '
      f'grad_rel={np.linalg.norm(np.nan_to_num(g)-gF)/(np.linalg.norm(gF)+1e-9):.3f}')
