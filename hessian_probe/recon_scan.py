"""Systematically locate (1) the Hessian breaker and (2) the NaN trigger in the RECON path.
(1) MC (max_sensors_per_cell) x K scan: AD Hessian min-eig / neg-diag / cond + AD-vs-FD grad rel.
(2) NaN hunt: PLAIN step over many keys/tracks -> any NaN in grad or Hessian?
FD Hessian (central diff of the loss) as the 'good' reference for conditioning."""
import os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
import lucid.simulation.simulator as SIM, lucid.simulation.photon_step as PS
def plain_factory(reflection_fn=PS.scalar_reflection):
    def step(p, d, t, sd, n, sl, ml, g, rp, al, hs, lam, rk, cc):
        return PS.photon_iteration_update_factors(p, d, t, sd, n, sl, ml, g, rp, al, hs, lam, rk, cc, reflection_fn=reflection_fn)
    return step
SIM.make_photon_iteration_update_factors_safe = plain_factory     # PLAIN everywhere (no scrub)
from lucid.fitting.recon import track_from_vec9, vec9_from_track
from lucid.geometry import generate_detector

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
NPH = int(os.environ.get('NPH', '40000')); GRID = dict(n_cap=80, n_angular=120, n_height=80)
ND = len(generate_detector(GEOM).all_points)
c = jnp.asarray(np.random.default_rng(0).standard_normal(ND))
t9 = jnp.asarray(vec9_from_track(1050., [2.0, -1.0, 3.0], [0.2, 0.1, 0.97], 0.0), float)

def build(MC, K):
    return SIM.setup_event_simulator(GEOM, NPH, temperature=0.1, K=K, hit_mode='per_photon',
        physics_config=PHYS, default_detector_params=True, particle='muon', wavelength_mode=True,
        pos_grad_threshold=K, n_grad_iters=K, max_sensors_per_cell=MC, **GRID)

def analyze(pred, key):
    def L(t): return jnp.sum(c * pred(track_from_vec9(t), key)[3])
    g = np.asarray(jax.grad(L)(t9)); H = np.asarray(jax.hessian(L)(t9))
    h = 1e-3; gF = np.array([float((L(t9.at[i].add(h)) - L(t9.at[i].add(-h)))/(2*h)) for i in range(9)])
    ev = np.linalg.eigvalsh(0.5*(H+H.T))
    return dict(gNaN=int(np.sum(~np.isfinite(g))), HNaN=int(np.sum(~np.isfinite(H))),
                mineig=ev.min(), negdiag=int((np.diag(H)<0).sum()),
                cond=ev.max()/(abs(ev).min()+1e-30), grel=np.linalg.norm(np.nan_to_num(g)-gF)/(np.linalg.norm(gF)+1e-9))

print(f'NPH={NPH} ND={ND}\n=== (1) Hessian breaker: MC x K scan (1 key) ===')
print(f'{"MC":>3s} {"K":>2s} | {"gNaN":>4s} {"HNaN":>4s} {"min-eig":>10s} {"neg-diag":>8s} {"cond":>9s} {"grad rel(AD,FD)":>14s}')
for MC in [4, 16, 64]:
    for K in [8]:
        r = analyze(build(MC, K), jax.random.PRNGKey(3))
        print(f'{MC:>3d} {K:>2d} | {r["gNaN"]:>4d} {r["HNaN"]:>4d} {r["mineig"]:>10.2e} {r["negdiag"]:>8d} {r["cond"]:>9.1e} {r["grel"]:>14.3f}')
for MC in [4]:
    for K in [1, 2]:
        r = analyze(build(MC, K), jax.random.PRNGKey(3))
        print(f'{MC:>3d} {K:>2d} | {r["gNaN"]:>4d} {r["HNaN"]:>4d} {r["mineig"]:>10.2e} {r["negdiag"]:>8d} {r["cond"]:>9.1e} {r["grel"]:>14.3f}')

print('\n=== (2) NaN hunt: PLAIN step, MC=4 K=8, 24 keys x 2 tracks ===')
pred = build(4, 8); nfound = 0
tracks = [t9, jnp.asarray(vec9_from_track(800., [-5., 4., -8.], [0.5, -0.3, 0.81], 0.0), float)]
for ti, tk in enumerate(tracks):
    for i in range(24):
        key = jax.random.PRNGKey(500 + i)
        def L(t): return jnp.sum(c * pred(track_from_vec9(t), key)[3])
        g = np.asarray(jax.grad(L)(tk))
        if not np.all(np.isfinite(g)):
            nfound += 1
            if nfound <= 3: print(f'  NaN grad at track{ti} key{i}: nNaN={np.sum(~np.isfinite(g))}/9')
print(f'  NaN-grad found in {nfound}/48 (track,key) combos')
