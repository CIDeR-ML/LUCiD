"""HIGH-STATISTICS AD-vs-FD match for the recon path (plain step, no custom_vjp). Gradient over NK_G
keys (cheap), Hessian diagonal over NK_H keys (heavy). Per-param AD mean +/- SE vs FD mean +/- SE,
z-score, AGREE/DIFFER. Hessian: E[AD H_jj] vs GT = d/dtheta E[grad]. h=1e-3 (validated window).
Env: K, NK_G, NK_H."""
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
NPH = 40000; K = int(os.environ.get('K','8')); MC = 4; h = 1e-3
NKG = int(os.environ.get('NK_G','2000')); NKH = int(os.environ.get('NK_H','500'))
GRID = dict(n_cap=80, n_angular=120, n_height=80)
ND = len(generate_detector(GEOM).all_points)
c = jnp.asarray(np.random.default_rng(0).standard_normal(ND))
t9 = jnp.asarray(vec9_from_track(1050.,[2.0,-1.0,3.0],[0.2,0.1,0.97],0.0), float)
LBL = ['E','x','y','z','sinth','costh','sinph','cosph','t0']
pred = SIM.setup_event_simulator(GEOM, NPH, temperature=0.1, K=K, hit_mode='per_photon',
    physics_config=PHYS, default_detector_params=True, particle='muon', wavelength_mode=True,
    pos_grad_threshold=K, n_grad_iters=K, max_candidates_per_ray=MC, **GRID)
def L(t,k): return jnp.sum(c * pred(track_from_vec9(t), k)[3])
gad = jax.jit(jax.grad(L)); had = jax.jit(jax.hessian(L))
MODE = os.environ.get('MODE','both')  # 'grad' | 'hess' | 'both'
print(f'### K={K}  ND={ND}  NK_G={NKG} NK_H={NKH} h={h} MODE={MODE}', flush=True)
nbad = 0

# GRADIENT
if MODE in ('grad','both'):
    kg = [jax.random.PRNGKey(11000+i) for i in range(NKG)]
    GA = np.array([np.asarray(gad(t9,k)) for k in kg])
    def gfd(k): return np.array([float((L(t9.at[d].add(h),k)-L(t9.at[d].add(-h),k))/(2*h)) for d in range(9)])
    GF = np.array([gfd(k) for k in kg])
    print('=== GRADIENT high-stat ===', flush=True)
    for d in range(9):
        am,ae = GA[:,d].mean(), GA[:,d].std()/np.sqrt(NKG); fm,fe = GF[:,d].mean(), GF[:,d].std()/np.sqrt(NKG)
        z = abs(am-fm)/np.sqrt(ae**2+fe**2+1e-12); ok = z<3; nbad += (not ok)
        print(f'  {LBL[d]:6s} AD={am:+10.3f}+/-{ae:6.3f}  FD={fm:+10.3f}+/-{fe:6.3f}  z={z:5.2f}  {"AGREE" if ok else "*** DIFFER"}', flush=True)

# HESSIAN diag vs GT
if MODE in ('hess','both'):
    kh = [jax.random.PRNGKey(11000+i) for i in range(NKH)]
    print('=== HESSIAN DIAG high-stat ===', flush=True)
    for j in range(9):
        gp = np.mean([np.asarray(gad(t9.at[j].add(h),k))[j] for k in kh])
        gm = np.mean([np.asarray(gad(t9.at[j].add(-h),k))[j] for k in kh])
        sp = np.std([np.asarray(gad(t9.at[j].add(h),k))[j] for k in kh])/np.sqrt(NKH)
        gt = (gp-gm)/(2*h); gte = np.sqrt(2)*sp/(2*h)
        HJ = np.array([np.asarray(had(t9,k))[j,j] for k in kh]); ad,ade = HJ.mean(), HJ.std()/np.sqrt(NKH)
        z = abs(ad-gt)/np.sqrt(ade**2+gte**2+1e-12); ok = z<3; nbad += (not ok)
        print(f'  {LBL[j]:6s} AD_H={ad:+12.1f}+/-{ade:8.1f}  GT={gt:+12.1f}+/-{gte:8.1f}  z={z:5.2f}  {"AGREE" if ok else "*** DIFFER"}', flush=True)
print(f'### K={K} RESULT: {"ALL AGREE" if nbad==0 else str(nbad)+" DIFFER"}', flush=True)
