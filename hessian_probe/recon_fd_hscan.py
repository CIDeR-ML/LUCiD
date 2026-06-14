"""Is FD itself trustworthy (not just 'agrees with AD on one h')? If FD-CRN had truncation bias the
AD-vs-FD agreement would DRIFT with h. Sweep h; AD is h-independent (computed once). Report, per h,
the FD mean for representative params (x,z,sinphi) and the max per-param z(AD,FD). Stable small z
across 2 decades of h => FD is a consistent estimator, agreement is not an h-artifact. K=8, NK keys."""
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
NPH = 40000; K = 8; MC = 4; NK = int(os.environ.get('NK', '150')); GRID = dict(n_cap=80, n_angular=120, n_height=80)
ND = len(generate_detector(GEOM).all_points)
c = jnp.asarray(np.random.default_rng(0).standard_normal(ND))
t9 = jnp.asarray(vec9_from_track(1050., [2.0,-1.0,3.0], [0.2,0.1,0.97], 0.0), float)
LBL = ['E','x','y','z','sinth','costh','sinph','cosph','t0']
pred = SIM.setup_event_simulator(GEOM, NPH, temperature=0.1, K=K, hit_mode='per_photon',
    physics_config=PHYS, default_detector_params=True, particle='muon', wavelength_mode=True,
    pos_grad_threshold=K, n_grad_iters=K, max_candidates_per_ray=MC, **GRID)
def L(t, k): return jnp.sum(c * pred(track_from_vec9(t), k)[3])
gad = jax.jit(jax.grad(L))
keys = [jax.random.PRNGKey(11000+i) for i in range(NK)]
GA = np.array([np.asarray(gad(t9, k)) for k in keys]); AM = GA.mean(0); AE = GA.std(0)/np.sqrt(NK)
print(f'ND={ND} K={K} NK={NK}')
print(f'AD mean: x={AM[1]:+.2f} z={AM[3]:+.2f} sinph={AM[6]:+.2f}')
print(f'{"h":>8} {"FD x":>10} {"FD z":>10} {"FD sinph":>10} {"maxz(AD,FD)":>12} {"argmax":>7}')
for h in [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
    def gfd(k): return np.array([float((L(t9.at[d].add(h), k) - L(t9.at[d].add(-h), k))/(2*h)) for d in range(9)])
    GF = np.array([gfd(k) for k in keys]); FM = GF.mean(0); FE = GF.std(0)/np.sqrt(NK)
    z = np.abs(AM-FM)/np.sqrt(AE**2+FE**2+1e-12)
    print(f'{h:>8.0e} {FM[1]:>+10.2f} {FM[3]:>+10.2f} {FM[6]:>+10.2f} {z.max():>12.1f} {LBL[int(z.argmax())]:>7}')
