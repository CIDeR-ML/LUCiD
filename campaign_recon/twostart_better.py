"""Test: bias-corrected SINGLE time seed vs the two-start, on inward tail events.
time-multilateration (transverse-perfect, forward-biased) -> cone dir -> shift vtx backward by
DELTA along dir -> ONE fit. Compare to out_ms100 (two-start). EVENT_LIST, DELTA(m) env."""
import os, sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,'/sdf/group/neutrino/omara/LUCiD_unification')
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.sources.event_io import read_photon_data_from_photonsim, pad_photon_data
from lucid.fitting import ReconModel, fit_track, track_from_vec9, vec9_from_track, vec9_dir, seed_vertex_time
from lucid.optimization.utils.functions import hierarchical_direction_search_cone, energy_scan_optimization
from campaign_recon.truth_exact import exact_truth
ROOT='/sdf/group/neutrino/omara/LUCiD_recon/data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root'
GEOM='config/SK_like_geom_config.json'; PHYS='config/SK_like_physics_config.json'
K,NBUF,FIDR,FIDZ=8,400_000,12.,12.; GRID=dict(n_cap=80,n_angular=120,n_height=80)
DELTA=float(os.environ.get('DELTA','2.5')); EVS=[int(x) for x in os.environ['EVENT_LIST'].split(',')]
det=generate_detector(GEOM); ND=len(det.all_points); POS=np.asarray(det.all_points)
dp=load_detector_params(PHYS,num_sensors=ND); dp=dp._replace(response=dp.response._replace(tts=jnp.asarray(2.5)))
data_sim=setup_event_simulator(GEOM,NBUF,temperature=None,K=K,is_data=True,hit_mode='realistic',physics_config=PHYS,
        default_detector_params=dp,particle='muon',wavelength_mode=True,apply_smearing=False,**GRID)
pred=setup_event_simulator(GEOM,250_000,temperature=0.1,K=K,hit_mode='per_photon',physics_config=PHYS,
        default_detector_params=True,particle='muon',wavelength_mode=True,pos_grad_threshold=K,n_grad_iters=K,**GRID)
model=ReconModel(pred,ND,sigma=2.5,delta=1.0)
def _rotax(u,deg):
    a=np.radians(deg);ca,sa=np.cos(a),np.sin(a);u=u/np.linalg.norm(u)
    ux=np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]]);return np.eye(3)*ca+sa*ux+(1-ca)*np.outer(u,u)
def rand_tf(raw,ev):
    rng=np.random.default_rng(100003+ev);beta=np.degrees(np.arccos(rng.uniform(-1,1)));al=rng.uniform(0,2*np.pi)
    axis=np.array([-np.sin(al),np.cos(al),0.]);rr=FIDR*np.sqrt(rng.uniform());ph=rng.uniform(0,2*np.pi)
    sh=np.array([rr*np.cos(ph),rr*np.sin(ph),rng.uniform(-FIDZ,FIDZ)])*100.
    raw=dict(raw);O=np.asarray(raw['photon_origins']).astype(float);D=np.asarray(raw['photon_directions']).astype(float)
    R=_rotax(axis,beta);c=O.mean(0);raw['photon_origins']=(O-c)@R.T+c+sh;raw['photon_directions']=D@R.T;return raw
ET=exact_truth(EVS)
print(f'DELTA={DELTA}m | ev: corrected-single fit vs two-start(out_ms100)')
for ev in EVS:
    raw=rand_tf(read_photon_data_from_photonsim(ROOT,ev),ev); vtx_true,dir_true=ET[ev]
    pd,_=pad_photon_data(raw,NBUF)
    cc,tt=jax.lax.stop_gradient(data_sim(track_from_vec9(jnp.array([1050.,0,0,0,0.,1.,0.,1.,0.])),jax.random.PRNGKey(7000+ev),pd))
    oc=np.asarray(cc);ot=np.where(oc>0,np.asarray(tt),0.);ocf,otf,POSf=jnp.asarray(oc),jnp.asarray(ot),jnp.asarray(POS)
    e0=energy_scan_optimization(pred,jnp.zeros(3),jnp.arccos(1/jnp.sqrt(3)),jnp.pi/4,0.,POSf,otf,ocf,(ocf,otf),1000.,700.,12,0)['best_energy']
    vtx_t,t0g=seed_vertex_time(POS,oc,ot)
    c2=hierarchical_direction_search_cone(pred,jnp.asarray(vtx_t),t0g,POSf,otf,ocf,(ocf,otf),e0,3,8,90.,0.5,0)
    dg=np.array([np.sin(c2['best_theta'])*np.cos(c2['best_phi']),np.sin(c2['best_theta'])*np.sin(c2['best_phi']),np.cos(c2['best_theta'])])
    vtx_corr=np.asarray(vtx_t)-DELTA*dg                          # backward shift along cone dir
    seed=vec9_from_track(e0,vtx_corr,dg,t0=t0g)
    res=fit_track(model,oc,ot,seed,nkeys=4,niters=250)
    fv=np.linalg.norm(res[1:4]-vtx_true)*100; sv=np.linalg.norm(np.asarray(vtx_t)-vtx_true)*100; cv=np.linalg.norm(vtx_corr-vtx_true)*100
    ts=np.load(f'campaign_recon/out_ms100/ev{ev:03d}.npz')
    import numpy as _np; tsfit=min(_np.linalg.norm(ts['fitA'][1:4]-vtx_true),_np.linalg.norm(ts['fitB'][1:4]-vtx_true))*100
    print(f'  ev{ev:03d}: time-seed {sv:5.0f} -> corrected {cv:5.0f} -> single-fit {fv:6.1f}cm | two-start best {tsfit:6.1f}cm',flush=True)
