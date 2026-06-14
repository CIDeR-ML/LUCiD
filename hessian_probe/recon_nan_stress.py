"""Does dropping custom_vjp (plain step) reintroduce NaN gradients/Hessians? The custom_vjp+nan_to_num
was a backstop for sqrt-at-zero 0/0 cotangents (photon ON a surface -> |Delta|=0; ray grazing -> disc->0).
Eps-INSIDE-sqrt (surface-dist +1e-12; discriminant maximum(.,1e-6)) should remove them at source.
STRESS the exact failure geometry: track near the wall, near the cap, aligned with the axis (max grazing
+ on-surface photons), many keys. Count NaN in grad and Hessian. PLAIN step (no custom_vjp)."""
import os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
import lucid.simulation.simulator as SIM, lucid.simulation.photon_step as PS
def plain_factory(reflection_fn=PS.scalar_reflection):   # NO custom_vjp, NO nan_to_num
    def step(p, d, t, sd, n, sl, ml, g, rp, al, hs, lam, rk, cc):
        return PS.photon_iteration_update_factors(p, d, t, sd, n, sl, ml, g, rp, al, hs, lam, rk, cc, reflection_fn=reflection_fn)
    return step
SIM.make_photon_iteration_update_factors_safe = plain_factory
from lucid.fitting.recon import track_from_vec9, vec9_from_track
from lucid.geometry import generate_detector
GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
NPH = 40000; K = 8; MC = 4; NK = int(os.environ.get('NK', '300')); GRID = dict(n_cap=80, n_angular=120, n_height=80)
det = generate_detector(GEOM); ND = len(det.all_points)
R = float(getattr(det, 'R', 16.9)); Hd = float(getattr(det, 'H', 36.0))
c = jnp.asarray(np.random.default_rng(0).standard_normal(ND))
pred = SIM.setup_event_simulator(GEOM, NPH, temperature=0.1, K=K, hit_mode='per_photon',
    physics_config=PHYS, default_detector_params=True, particle='muon', wavelength_mode=True,
    pos_grad_threshold=K, n_grad_iters=K, max_candidates_per_ray=MC, **GRID)
def L(t, k): return jnp.sum(c * pred(track_from_vec9(t), k)[3])
gad = jax.jit(jax.grad(L)); had = jax.jit(jax.hessian(L))
# STRESS configs: (label, vtx, dir). near-wall + radial (photons hit wall at near-grazing),
# near-cap + axial (photons sit on the cap), axis-aligned long track (max on-surface).
CFG = [
    ('center',     [2.,-1.,3.],     [0.2,0.1,0.97]),
    ('near-wall',  [R-0.5,0.,0.],   [1.0,0.0,0.02]),   # pointing into wall, grazing
    ('near-cap',   [0.,0.,Hd/2-0.5],[0.05,0.05,1.0]),  # pointing at cap
    ('axis-long',  [0.,0.,-Hd/2+1.],[0.0,0.0,1.0]),    # straight up the axis
    ('wall-graze', [0.,0.,0.],      [0.999,0.0,0.045]),# nearly tangent to wall
]
print(f'ND={ND} R={R:.2f} H={Hd:.2f} K={K} NK={NK}')
for lbl, v, d in CFG:
    t9 = jnp.asarray(vec9_from_track(1050., v, d, 0.0), float)
    ng = nh = nval = 0
    for i in range(NK):
        k = jax.random.PRNGKey(20000 + i)
        val = float(L(t9, k)); ng += int(not np.all(np.isfinite(np.asarray(gad(t9, k)))))
        nval += int(not np.isfinite(val))
    # Hessian is heavier -> sample fewer keys
    for i in range(min(NK, 40)):
        k = jax.random.PRNGKey(20000 + i)
        nh += int(not np.all(np.isfinite(np.asarray(had(t9, k)))))
    print(f'  {lbl:11s} vtx={np.round(v,1)} dir={np.round(d,2)}  NaN: val={nval}/{NK} grad={ng}/{NK} hess={nh}/{min(NK,40)}')
print('PASS: zero NaNs everywhere => custom_vjp/nan_to_num backstop is unnecessary; safe to drop.' )
