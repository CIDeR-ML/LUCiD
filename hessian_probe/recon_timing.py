"""Baseline wall-clock of ONE recon (self-contained, in-repo, SIREN-sampled truth — no external ROOT).
Current recon = AD gradient (jax.grad) + FD Fisher (ReconModel.fisher, 9x2xnkeys forward evals/refresh).
Breaks down: sim build, data gen (compile+run), grad compile+steady, fisher compile+steady, a short
timed fit, then EXTRAPOLATES to campaign settings (nkeys, niters, refresh). Env: GRID_S(mall/big), NITERS_FIT."""
import os, sys, time
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.fitting import ReconModel, fit_track, vec9_from_track, vec9_dir, track_from_vec9, SCALE9
def clk(lab, t): dt=time.time()-t; print(f'  {lab:38s} {dt:8.2f}s', flush=True); return time.time()
GEOM, PHYS, K = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json', 8
BIG = os.environ.get('GRID','big')=='big'
GRID = dict(n_cap=80, n_angular=120, n_height=80) if BIG else dict(n_cap=40, n_angular=80, n_height=40)
NPH = int(os.environ.get('NPH','250000')); NITERS_FIT = int(os.environ.get('NITERS_FIT','40'))
NKEYS, REFRESH, NITERS_CAMP = 4, 8, 250
ND = len(generate_detector(GEOM).all_points)
T=time.time(); t=T
pred = setup_event_simulator(GEOM, NPH, temperature=0.1, K=K, hit_mode='per_photon',
    physics_config=PHYS, default_detector_params=True, particle='muon', wavelength_mode=True,
    pos_grad_threshold=K, n_grad_iters=K, **GRID)
dp_data = load_detector_params(PHYS, num_sensors=ND)._replace(
    response=load_detector_params(PHYS, num_sensors=ND).response._replace(tts=jnp.asarray(2.5)))
data_sim = setup_event_simulator(GEOM, NPH, temperature=None, K=K, use_expected_value=False,
    hit_mode='realistic', apply_smearing=False, particle='muon', physics_config=PHYS,
    default_detector_params=dp_data, wavelength_mode=True, **GRID)
model = ReconModel(pred, ND, sigma=2.5, delta=1.0)
t=clk('build sims (no compile)', t)
print(f'config: GRID={"big" if BIG else "small"} NPH={NPH} ND={ND} K={K}', flush=True)

th9 = vec9_from_track(1050., [1.5,-0.8,2.0], [0.3,0.1,0.95], 0.)
c, tt = jax.lax.stop_gradient(data_sim(track_from_vec9(jnp.asarray(th9)), jax.random.PRNGKey(0)))
c.block_until_ready(); oc=jnp.asarray(np.asarray(c)); ot=jnp.asarray(np.where(np.asarray(c)>0, np.asarray(tt), 0.))
t=clk('data_sim FIRST call (compile+gen truth)', t)

keys=[jax.random.PRNGKey(s) for s in range(NKEYS)]; fdh=0.4*SCALE9
# grad compile + steady
g=model.grad(th9, oc, ot, keys[0]); jax.block_until_ready(g); t=clk('grad FIRST (compile)', t)
ts=[]
for k in keys: t0=time.time(); jax.block_until_ready(model.grad(th9,oc,ot,k)); ts.append(time.time()-t0)
tg=np.median(ts); print(f'  grad steady (per call)                  {1000*tg:8.1f} ms', flush=True)
# fisher compile + steady
t=time.time(); F=model.fisher(th9, oc, ot, keys, fdh); t=clk('fisher FIRST (compile, nkeys)', t)
t0=time.time(); F=model.fisher(th9, oc, ot, keys, fdh); tf=time.time()-t0
print(f'  fisher steady (per refresh, nkeys)      {1000*tf:8.1f} ms', flush=True)
# short timed fit
start = th9 + 2.5*SCALE9*np.random.default_rng(1).uniform(-1,1,9)
t=time.time(); res=fit_track(model, oc, ot, start, nkeys=NKEYS, niters=NITERS_FIT); tfit=time.time()-t
clk(f'fit_track {NITERS_FIT}it x{NKEYS}keys (warm)', time.time()-tfit)
fv=np.linalg.norm((res-th9)[1:4])*100; print(f'  -> {NITERS_FIT}-iter fit vtx {fv:.1f}cm', flush=True)

# extrapolate to campaign
camp = NITERS_CAMP*NKEYS*tg + (NITERS_CAMP//REFRESH+1)*tf
print(f'\n=== EXTRAPOLATION to campaign (niters={NITERS_CAMP}, nkeys={NKEYS}, refresh={REFRESH}) ===', flush=True)
print(f'  grad: {NITERS_CAMP}*{NKEYS}*{1000*tg:.1f}ms = {NITERS_CAMP*NKEYS*tg:6.1f}s', flush=True)
print(f'  fisher: {NITERS_CAMP//REFRESH+1}*{1000*tf:.1f}ms = {(NITERS_CAMP//REFRESH+1)*tf:6.1f}s', flush=True)
print(f'  => fit ~{camp:.0f}s (warm)  + one-time compile (grad+fisher+data) ~{0:.0f}', flush=True)
print(f'### DONE total wall {time.time()-T:.0f}s', flush=True)
