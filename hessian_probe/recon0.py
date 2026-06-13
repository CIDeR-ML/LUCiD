"""RECON path: reproduce the AD-gradient NaN + AD-Hessian breakage that motivated the custom_vjp.
Track params t9=[E,x,y,z,sinθ,cosθ,sinφ,cosφ,t0] flow PATHWISE through the SIREN emitter + the
GEOMETRY (ray-trace, surface distances, sensor assignment, max_sensors_per_cell). Compare SAFE step
(custom_vjp+nan_to_num) vs PLAIN step (no custom_vjp): does AD grad go NaN? Is the AD Hessian
finite/PSD? FD reference. Loss = Σ c·total_charge (clean linear functional of per-PMT charge)."""
import os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
PLAIN = os.environ.get('PLAIN', '0') == '1'
import lucid.simulation.simulator as SIM
import lucid.simulation.photon_step as PS
if PLAIN:
    def plain_factory(reflection_fn=PS.scalar_reflection):
        def step(p, d, t, sd, n, sl, ml, g, rp, al, hs, lam, rk, cc):
            return PS.photon_iteration_update_factors(p, d, t, sd, n, sl, ml, g, rp, al, hs, lam, rk, cc,
                                                      reflection_fn=reflection_fn)
        return step
    SIM.make_photon_iteration_update_factors_safe = plain_factory
from lucid.fitting.recon import track_from_vec9, vec9_from_track

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
NPH = int(os.environ.get('NPH', '60000')); K = int(os.environ.get('K', '8'))
MC = int(os.environ.get('MC', '4'))
GRID = dict(n_cap=80, n_angular=120, n_height=80)
pred = SIM.setup_event_simulator(GEOM, NPH, temperature=0.1, K=K, hit_mode='per_photon',
                                 physics_config=PHYS, default_detector_params=True, particle='muon',
                                 wavelength_mode=True, pos_grad_threshold=K, n_grad_iters=K,
                                 max_sensors_per_cell=MC, **GRID)
ND = SIM.generate_detector(GEOM).all_points.shape[0] if hasattr(SIM, 'generate_detector') else None
from lucid.geometry import generate_detector
ND = len(generate_detector(GEOM).all_points)
t9 = jnp.asarray(vec9_from_track(1050., [2.0, -1.0, 3.0], [0.2, 0.1, 0.97], t0=0.0), float)
c = jnp.asarray(np.random.default_rng(0).standard_normal(ND))
print(f'PLAIN={PLAIN} NPH={NPH} K={K} MC={MC} ND={ND}  t9={np.round(np.asarray(t9),3)}')


def L(t, key):
    out = pred(track_from_vec9(t), key)
    return jnp.sum(c * out[3])            # total_charge


k0 = jax.random.PRNGKey(3)
val = float(L(t9, k0)); print(f'L = {val:.3f}  finite={np.isfinite(val)}')
gA = np.asarray(jax.grad(L)(t9, k0))
print(f'AD grad finite={np.all(np.isfinite(gA))}  nNaN={np.sum(~np.isfinite(gA))}/9  grad={np.round(gA,3)}')
# FD grad
h = 1e-3; gF = np.array([float((L(t9.at[i].add(h), k0) - L(t9.at[i].add(-h), k0)) / (2*h)) for i in range(9)])
print(f'FD grad finite={np.all(np.isfinite(gF))}  grad={np.round(gF,3)}')
print(f'  AD-vs-FD grad rel (finite entries) = {np.linalg.norm(np.nan_to_num(gA)-gF)/(np.linalg.norm(gF)+1e-9):.2e}')
# AD Hessian
try:
    HA = np.asarray(jax.hessian(L)(t9, k0))
    ev = np.linalg.eigvalsh(0.5*(HA+HA.T))
    print(f'AD Hessian finite={np.all(np.isfinite(HA))}  nNaN={np.sum(~np.isfinite(HA))}/81  '
          f'eig[min,max]=[{ev.min():.2e},{ev.max():.2e}]  negdiag={int((np.diag(HA)<0).sum())}/9  cond={ev.max()/ (abs(ev).min()+1e-30):.1e}')
except Exception as e:
    print(f'AD Hessian ERROR: {type(e).__name__}: {str(e)[:90]}')
