"""Do we need nkeys=4? nkeys averages the stochastic gradient + Fisher. Measure (a) per-key gradient
noise and how the SE falls with nkeys; (b) fit vtx vs nkeys for AD-Fisher(lr=1) over a few seeds (so
it's not one lucky draw). If vtx holds at nkeys=1-2, we can cut grad+fisher cost proportionally."""
import os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.fitting import ReconModel, fit_track, vec9_from_track, vec9_dir, track_from_vec9, SCALE9
GEOM, PHYS, K = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json', 8
GRID=dict(n_cap=80,n_angular=120,n_height=80); NPH=int(os.environ.get('NPH','250000')); NITERS=int(os.environ.get('NITERS','80'))
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
oc=jnp.asarray(np.asarray(c)); ot=jnp.asarray(np.where(np.asarray(c)>0,np.asarray(tt),0.)); td=vec9_dir(th9)
print(f'ND={ND} NPH={NPH} NITERS={NITERS}', flush=True)

# (a) per-key gradient noise (relative to step scale SCALE9): how big is one-key noise vs the SE?
G=np.array([np.asarray(model.grad(th9,oc,ot,jax.random.PRNGKey(s))) for s in range(32)])
print('=== (a) per-key gradient: std (1 key) and implied SE at nkeys=1,2,4,8 (scaled by SCALE9) ===', flush=True)
for d in [0,1,3,7,8]:
    s1=G[:,d].std()*SCALE9[d]
    print(f'  {LBL[d]:6s} std@1={s1:9.3f}  SE@2={s1/np.sqrt(2):9.3f}  SE@4={s1/2:9.3f}  SE@8={s1/np.sqrt(8):9.3f}', flush=True)

# (b) fit vtx vs nkeys, AD-Fisher lr=1, multiple seeds + one FD nkeys=4 baseline
print('=== (b) fit vtx (cm) vs nkeys [AD-Fisher lr=1], 3 seeds ===', flush=True)
for nk in [1,2,4]:
    vs=[]; ds=[]
    for sd in [0,1,2]:
        r=fit_track(model,oc,ot,th9+2.5*SCALE9*np.random.default_rng(sd).uniform(-1,1,9),
                    nkeys=nk,niters=NITERS,lr=1.0,fisher_mode='ad',seed=sd)
        vs.append(np.linalg.norm((r-th9)[1:4])*100); ds.append(np.degrees(np.arccos(np.clip(vec9_dir(r)@td,-1,1))))
    print(f'  nkeys={nk}: vtx={np.round(vs,1)} (med {np.median(vs):.1f})  dir med {np.median(ds):.2f}deg', flush=True)
r=fit_track(model,oc,ot,th9+2.5*SCALE9*np.random.default_rng(0).uniform(-1,1,9),nkeys=4,niters=NITERS,lr=8.0,fisher_mode='fd',seed=0)
print(f'  [ref] FD nkeys=4 lr=8: vtx {np.linalg.norm((r-th9)[1:4])*100:.1f}cm', flush=True)
print('### DONE', flush=True)
