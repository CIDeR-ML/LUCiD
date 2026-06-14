"""Single-eps-per-process test (no jit-cache contamination): set _CYL_SQRT_EPS from env BEFORE any
build, confirm it changes the forward (loss), then report the K=8 AD Hessian min-eig/cond/neg-diag."""
import os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
import lucid.propagation.cylinder as CYL
EPS = float(os.environ.get('EPS', '0.0')); CYL._CYL_SQRT_EPS = EPS    # set FIRST, before any trace
import lucid.simulation.simulator as SIM, lucid.simulation.photon_step as PS
def plain_factory(reflection_fn=PS.scalar_reflection):
    def step(p, d, t, sd, n, sl, ml, g, rp, al, hs, lam, rk, cc):
        return PS.photon_iteration_update_factors(p, d, t, sd, n, sl, ml, g, rp, al, hs, lam, rk, cc, reflection_fn=reflection_fn)
    return step
SIM.make_photon_iteration_update_factors_safe = plain_factory
from lucid.fitting.recon import track_from_vec9, vec9_from_track
from lucid.geometry import generate_detector
GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
NPH = 40000; K = int(os.environ.get('K', '8')); MC = 4; GRID = dict(n_cap=80, n_angular=120, n_height=80)
ND = len(generate_detector(GEOM).all_points)
c = jnp.asarray(np.random.default_rng(0).standard_normal(ND))
t9 = jnp.asarray(vec9_from_track(1050., [2.0, -1.0, 3.0], [0.2, 0.1, 0.97], 0.0), float)
pred = SIM.setup_event_simulator(GEOM, NPH, temperature=0.1, K=K, hit_mode='per_photon',
    physics_config=PHYS, default_detector_params=True, particle='muon', wavelength_mode=True,
    pos_grad_threshold=K, n_grad_iters=K, max_candidates_per_ray=MC, **GRID)
def L(t): return jnp.sum(c * pred(track_from_vec9(t), jax.random.PRNGKey(3))[3])
H = np.asarray(jax.hessian(L)(t9)); g = np.asarray(jax.grad(L)(t9))
h = 1e-3; gF = np.array([float((L(t9.at[i].add(h)) - L(t9.at[i].add(-h)))/(2*h)) for i in range(9)])
ev = np.linalg.eigvalsh(0.5*(H+H.T))
print(f'EPS={EPS:.1e} K={K} | L={float(L(t9)):.5f} | min-eig={ev.min():.2e} max-eig={ev.max():.2e} '
      f'neg-diag={int((np.diag(H)<0).sum())}/9 cond={ev.max()/(abs(ev).min()+1e-30):.1e} '
      f'grad_rel={np.linalg.norm(np.nan_to_num(g)-gF)/(np.linalg.norm(gF)+1e-9):.3f}')
