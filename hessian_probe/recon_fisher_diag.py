"""Why does the recon AD-Fisher differ from FD-Fisher (non-uniform, azimuth 137x smaller) and break the
FD-tuned fit? Hypothesis: F[d,d]=Sum_s J[s,d]^2, and E[Sum J^2]=Sum(E[J]^2+var) -> a NOISY per-sensor J
(FD secant) INFLATES the Fisher diagonal by Sum var; AD (low-var) gives the truer smaller curvature.
TEST (a): Fisher diag vs nkeys -- FD should DECREASE toward AD as averaging kills the var term; AD stable.
TEST (b): re-tune AD-Fisher fit with smaller lr -> does vtx recover to ~15cm? Env: GRID,NPH."""
import os, sys, time
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.fitting import ReconModel, fit_track, vec9_from_track, vec9_dir, track_from_vec9, SCALE9
GEOM, PHYS, K = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json', 8
GRID = dict(n_cap=80,n_angular=120,n_height=80); NPH=int(os.environ.get('NPH','250000'))
LBL=['E','x','y','z','sinth','costh','sinph','cosph','t0']
ND=len(generate_detector(GEOM).all_points)
pred=setup_event_simulator(GEOM,NPH,temperature=0.1,K=K,hit_mode='per_photon',physics_config=PHYS,
    default_detector_params=True,particle='muon',wavelength_mode=True,pos_grad_threshold=K,n_grad_iters=K,**GRID)
dpd=load_detector_params(PHYS,num_sensors=ND); dpd=dpd._replace(response=dpd.response._replace(tts=jnp.asarray(2.5)))
data_sim=setup_event_simulator(GEOM,NPH,temperature=None,K=K,use_expected_value=False,hit_mode='realistic',
    apply_smearing=False,particle='muon',physics_config=PHYS,default_detector_params=dpd,wavelength_mode=True,**GRID)
model=ReconModel(pred,ND,sigma=2.5,delta=1.0)
th9=vec9_from_track(1050.,[1.5,-0.8,2.0],[0.3,0.1,0.95],0.)
c,tt=jax.lax.stop_gradient(data_sim(track_from_vec9(jnp.asarray(th9)),jax.random.PRNGKey(0))); c.block_until_ready()
oc=jnp.asarray(np.asarray(c)); ot=jnp.asarray(np.where(np.asarray(c)>0,np.asarray(tt),0.)); fdh=0.4*SCALE9
allkeys=[jax.random.PRNGKey(s) for s in range(16)]
print(f'ND={ND} NPH={NPH}', flush=True)

print('=== (a) Fisher diag vs nkeys: FD (expect DECREASE->AD) vs AD (expect STABLE) ===', flush=True)
print(f'{"nkeys":>6} | ' + ' '.join(f'{l:>10s}' for l in ['x(FD)','x(AD)','cosph(FD)','cosph(AD)']), flush=True)
for nk in [1,2,4,8,16]:
    ks=allkeys[:nk]
    Ffd=model.fisher(th9,oc,ot,ks,fdh); Fad=model.fisher_ad(th9,oc,ot,ks)
    print(f'{nk:>6} | {Ffd[1,1]:>10.2e} {Fad[1,1]:>10.2e} {Ffd[7,7]:>10.2e} {Fad[7,7]:>10.2e}', flush=True)

print('=== (b) AD-Fisher fit, lr sweep (niters=60) — does a smaller lr recover ~15cm? ===', flush=True)
start=th9+2.5*SCALE9*np.random.default_rng(1).uniform(-1,1,9); td=vec9_dir(th9)
for mode,lr in [('fd',8.0),('ad',8.0),('ad',4.0),('ad',2.0),('ad',1.0)]:
    r=fit_track(model,oc,ot,start,nkeys=4,niters=60,lr=lr,fisher_mode=mode)
    fv=np.linalg.norm((r-th9)[1:4])*100; fd=np.degrees(np.arccos(np.clip(vec9_dir(r)@td,-1,1)))
    print(f'  fisher={mode} lr={lr:>4}: vtx {fv:6.1f}cm  dir {fd:4.2f}deg  E {r[0]-1050:+7.1f}MeV', flush=True)
print('### DONE', flush=True)
