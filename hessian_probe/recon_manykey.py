"""RECON path, the rigorous test (the calibration-grade one, never yet run for recon):
is the AD GRADIENT biased, or just high single-key variance reconciled in expectation?
And is the AD HESSIAN == FD Hessian in expectation (indefinite is FINE for a random-c loss;
the question is whether AD matches the true 2nd derivative, NOT whether it is PSD).

For each param: AD mean +/- SE vs FD mean +/- SE over NK keys, z-score. AGREE (z<3) => AD unbiased.
GT Hessian diagonal = d/dtheta E[grad] (FD of the EXPECTED AD gradient) vs E[AD H_jj]. Env: K, NK."""
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
NPH = 40000; K = int(os.environ.get('K', '8')); MC = 4; NK = int(os.environ.get('NK', '200')); h = 1e-3
GRID = dict(n_cap=80, n_angular=120, n_height=80)
ND = len(generate_detector(GEOM).all_points)
c = jnp.asarray(np.random.default_rng(0).standard_normal(ND))
t9 = jnp.asarray(vec9_from_track(1050., [2.0, -1.0, 3.0], [0.2, 0.1, 0.97], 0.0), float)
pred = SIM.setup_event_simulator(GEOM, NPH, temperature=0.1, K=K, hit_mode='per_photon',
    physics_config=PHYS, default_detector_params=True, particle='muon', wavelength_mode=True,
    pos_grad_threshold=K, n_grad_iters=K, max_candidates_per_ray=MC, **GRID)
LBL = ['E','x','y','z','sinth','costh','sinph','cosph','t0']
def L(t, k): return jnp.sum(c * pred(track_from_vec9(t), k)[3])
gad = jax.jit(jax.grad(L)); had = jax.jit(jax.hessian(L))
keys = [jax.random.PRNGKey(11000 + i) for i in range(NK)]
print(f'ND={ND} K={K} keys={NK} h={h}  t9={np.round(np.asarray(t9),3)}')

# ---- GRADIENT: AD vs FD in expectation ----
GA = np.array([np.asarray(gad(t9, k)) for k in keys])
def gfd(k): return np.array([float((L(t9.at[d].add(h), k) - L(t9.at[d].add(-h), k))/(2*h)) for d in range(9)])
GF = np.array([gfd(k) for k in keys])
print('\n=== GRADIENT (many-key): AD mean vs FD mean (+/-SE) ===')
for d in range(9):
    am, ae = GA[:,d].mean(), GA[:,d].std()/np.sqrt(NK); fm, fe = GF[:,d].mean(), GF[:,d].std()/np.sqrt(NK)
    z = abs(am-fm)/np.sqrt(ae**2+fe**2+1e-12)
    print(f'  {LBL[d]:6s} AD={am:+10.3f}+/-{ae:6.3f}  FD={fm:+10.3f}+/-{fe:6.3f}  z={z:5.1f}  '
          f'stdAD={GA[:,d].std():8.2f} stdFD={GF[:,d].std():8.2f}  {"AGREE" if z<3 else "*** DIFFER (AD BIASED)"}')

# ---- HESSIAN diagonal: AD vs GT = d/dtheta E[grad] ----
print('\n=== HESSIAN DIAG: E[AD H_jj] vs GT=d/dtheta E[grad] (FD of expected AD grad) ===')
for j in range(9):
    gp = np.mean([np.asarray(gad(t9.at[j].add(h), k))[j] for k in keys])
    gm = np.mean([np.asarray(gad(t9.at[j].add(-h), k))[j] for k in keys])
    sp = np.std([np.asarray(gad(t9.at[j].add(h), k))[j] for k in keys])/np.sqrt(NK)
    gt = (gp-gm)/(2*h); gte = np.sqrt(2)*sp/(2*h)
    HJ = np.array([np.asarray(had(t9, k))[j,j] for k in keys])
    ad, ade = HJ.mean(), HJ.std()/np.sqrt(NK)
    z = abs(ad-gt)/np.sqrt(ade**2+gte**2+1e-12)
    print(f'  {LBL[j]:6s} AD_H={ad:+11.1f}+/-{ade:7.1f}  GT={gt:+11.1f}+/-{gte:7.1f}  z={z:5.1f}  '
          f'{"AGREE" if z<3 else "*** DIFFER (2nd-order BUG)"}')
