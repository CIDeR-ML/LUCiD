"""Recon FD-Fisher vs AD-Fisher: TIMING (compile+steady) + VALUES (metric matrix agreement) + a fit
with each. Track params E/x/y/z/dir/t0 are all pathwise through SIREN+geometry (no discrete scatter/
mie/g here), so AD-Fisher should closely match FD-Fisher AND be faster. Env: GRID, NPH, NITERS_FIT."""
import os, sys, time
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.fitting import ReconModel, fit_track, vec9_from_track, vec9_dir, track_from_vec9, SCALE9
def clk(lab,t): dt=time.time()-t; print(f'  {lab:36s} {dt:8.2f}s', flush=True); return time.time()
GEOM, PHYS, K = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json', 8
BIG = os.environ.get('GRID','big')=='big'
GRID = dict(n_cap=80,n_angular=120,n_height=80) if BIG else dict(n_cap=40,n_angular=80,n_height=40)
NPH=int(os.environ.get('NPH','250000')); NITERS_FIT=int(os.environ.get('NITERS_FIT','40')); NKEYS,REFRESH,NCAMP=4,8,250
LBL=['E','x','y','z','sinth','costh','sinph','cosph','t0']
ND=len(generate_detector(GEOM).all_points); T=time.time(); t=T
pred=setup_event_simulator(GEOM,NPH,temperature=0.1,K=K,hit_mode='per_photon',physics_config=PHYS,
    default_detector_params=True,particle='muon',wavelength_mode=True,pos_grad_threshold=K,n_grad_iters=K,**GRID)
dpd=load_detector_params(PHYS,num_sensors=ND); dpd=dpd._replace(response=dpd.response._replace(tts=jnp.asarray(2.5)))
data_sim=setup_event_simulator(GEOM,NPH,temperature=None,K=K,use_expected_value=False,hit_mode='realistic',
    apply_smearing=False,particle='muon',physics_config=PHYS,default_detector_params=dpd,wavelength_mode=True,**GRID)
model=ReconModel(pred,ND,sigma=2.5,delta=1.0); t=clk('build sims',t)
th9=vec9_from_track(1050.,[1.5,-0.8,2.0],[0.3,0.1,0.95],0.)
c,tt=jax.lax.stop_gradient(data_sim(track_from_vec9(jnp.asarray(th9)),jax.random.PRNGKey(0))); c.block_until_ready()
oc=jnp.asarray(np.asarray(c)); ot=jnp.asarray(np.where(np.asarray(c)>0,np.asarray(tt),0.)); t=clk('gen truth',t)
keys=[jax.random.PRNGKey(s) for s in range(NKEYS)]; fdh=0.4*SCALE9

print('=== FISHER timing (per refresh, nkeys=4) ===', flush=True)
t=time.time(); Ffd=model.fisher(th9,oc,ot,keys,fdh); t=clk('FD fisher FIRST (compile)',t)
t0=time.time(); Ffd=model.fisher(th9,oc,ot,keys,fdh); tffd=time.time()-t0; print(f'  FD steady {1000*tffd:8.1f} ms', flush=True)
t=time.time(); Fad=model.fisher_ad(th9,oc,ot,keys); t=clk('AD fisher FIRST (compile)',t)
t0=time.time(); Fad=model.fisher_ad(th9,oc,ot,keys); tfad=time.time()-t0; print(f'  AD steady {1000*tfad:8.1f} ms   (FD/AD = {tffd/tfad:.1f}x)', flush=True)

print('=== FISHER values: FD vs AD ===', flush=True)
print(f'  matrix rel ||Fad-Ffd||/||Ffd|| = {np.linalg.norm(Fad-Ffd)/(np.linalg.norm(Ffd)+1e-30):.4f}', flush=True)
evf=np.sort(np.linalg.eigvalsh(0.5*(Ffd+Ffd.T))); eva=np.sort(np.linalg.eigvalsh(0.5*(Fad+Fad.T)))
print(f'  eig FD [{evf[0]:.2e},{evf[-1]:.2e}]  eig AD [{eva[0]:.2e},{eva[-1]:.2e}]', flush=True)
for j in range(9): print(f'    diag {LBL[j]:6s} FD={Ffd[j,j]:+.3e}  AD={Fad[j,j]:+.3e}', flush=True)

print('=== FIT with each (niters={}) ==='.format(NITERS_FIT), flush=True)
start=th9+2.5*SCALE9*np.random.default_rng(1).uniform(-1,1,9)
t0=time.time(); rfd=fit_track(model,oc,ot,start,nkeys=NKEYS,niters=NITERS_FIT,fisher_mode='fd'); tfit_fd=time.time()-t0
t0=time.time(); rad=fit_track(model,oc,ot,start,nkeys=NKEYS,niters=NITERS_FIT,fisher_mode='ad'); tfit_ad=time.time()-t0
td=vec9_dir(th9)
for lab,r,tf in [('FD',rfd,tfit_fd),('AD',rad,tfit_ad)]:
    fv=np.linalg.norm((r-th9)[1:4])*100; fd=np.degrees(np.arccos(np.clip(vec9_dir(r)@td,-1,1)))
    print(f'  {lab}: vtx {fv:5.1f}cm  dir {fd:4.2f}deg  E {r[0]-1050:+6.1f}MeV  | {NITERS_FIT}it {tf:5.1f}s', flush=True)
camp_fd=NCAMP*NKEYS*0+ (NCAMP//REFRESH+1)*tffd; camp_ad=(NCAMP//REFRESH+1)*tfad
print(f'\n=== campaign fisher cost (niters={NCAMP}): FD ~{camp_fd:.0f}s  AD ~{camp_ad:.0f}s ===', flush=True)
print(f'### DONE total {time.time()-T:.0f}s', flush=True)
