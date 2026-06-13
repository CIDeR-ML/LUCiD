"""EXHAUSTIVE NaN sweep, plain step (NO custom_vjp / NO nan_to_num). If a single NaN appears in value,
gradient, or Hessian anywhere, the eps-inside-sqrt floors are insufficient and custom_vjp cannot be
dropped. Coverage: many RANDOM tracks across the whole fiducial volume (random vtx/dir/energy) PLUS the
on-surface/grazing STRESS geometries (the exact 0/0 firing conditions), many keys each, val+grad+Hess,
at K=8 AND K=16 (more bounces => more on-surface photons => more chances to 0/0). Env: NT, NKR, NKS."""
import os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
import lucid.simulation.simulator as SIM, lucid.simulation.photon_step as PS
def plain_factory(reflection_fn=PS.scalar_reflection):    # NO custom_vjp, NO nan_to_num
    def step(p, d, t, sd, n, sl, ml, g, rp, al, hs, lam, rk, cc):
        return PS.photon_iteration_update_factors(p, d, t, sd, n, sl, ml, g, rp, al, hs, lam, rk, cc, reflection_fn=reflection_fn)
    return step
SIM.make_photon_iteration_update_factors_safe = plain_factory
from lucid.fitting.recon import track_from_vec9, vec9_from_track
from lucid.geometry import generate_detector
GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
NPH = 40000; MC = 4; GRID = dict(n_cap=80, n_angular=120, n_height=80)
NT = int(os.environ.get('NT','40')); NKR = int(os.environ.get('NKR','150')); NKS = int(os.environ.get('NKS','400'))
det = generate_detector(GEOM); ND = len(det.all_points)
R = float(getattr(det,'R',16.9)); Hd = float(getattr(det,'H',36.2))
c = jnp.asarray(np.random.default_rng(0).standard_normal(ND))
rng = np.random.default_rng(123)

def rand_tracks(n):
    out = []
    for _ in range(n):
        rr = (R-0.3)*np.sqrt(rng.uniform()); ph = rng.uniform(0,2*np.pi)
        v = [rr*np.cos(ph), rr*np.sin(ph), rng.uniform(-(Hd/2-0.3), Hd/2-0.3)]
        d = rng.standard_normal(3); d = (d/np.linalg.norm(d)).tolist()
        E = float(rng.uniform(200., 1500.)); out.append((E, v, d))
    return out
STRESS = [('center',[2.,-1.,3.],[0.2,0.1,0.97]), ('near-wall',[R-0.5,0.,0.],[1.,0.,0.02]),
          ('near-cap',[0.,0.,Hd/2-0.5],[0.05,0.05,1.]), ('axis-long',[0.,0.,-Hd/2+1.],[0.,0.,1.]),
          ('wall-graze',[0.,0.,0.],[0.999,0.,0.045])]
print(f'ND={ND} R={R:.2f} H={Hd:.2f} NT={NT} NKR={NKR} NKS={NKS}', flush=True)

KS = [int(x) for x in os.environ.get('KS', '8,16').split(',')]
TOTAL = dict(val=0, grad=0, hess=0, nval=0, ngrad=0, nhess=0)
for K in KS:
    pred = SIM.setup_event_simulator(GEOM, NPH, temperature=0.1, K=K, hit_mode='per_photon',
        physics_config=PHYS, default_detector_params=True, particle='muon', wavelength_mode=True,
        pos_grad_threshold=K, n_grad_iters=K, max_sensors_per_cell=MC, **GRID)
    def L(t,k): return jnp.sum(c * pred(track_from_vec9(t), k)[3])
    gad = jax.jit(jax.grad(L)); had = jax.jit(jax.hessian(L))
    nv=ng=nh=cv=cg=ch=0
    jobs = [('rand', E, v, d, NKR) for (E,v,d) in rand_tracks(NT)] + [(lbl,1050.,v,d,NKS) for (lbl,v,d) in STRESS]
    for (lbl,E,v,d,nk) in jobs:
        t9 = jnp.asarray(vec9_from_track(E, v, d, 0.0), float)
        for i in range(nk):
            k = jax.random.PRNGKey(30000+i)
            cv+=1; cg+=1
            if not np.isfinite(float(L(t9,k))): nv+=1
            if not np.all(np.isfinite(np.asarray(gad(t9,k)))): ng+=1
        for i in range(min(nk,25)):
            k = jax.random.PRNGKey(30000+i); ch+=1
            if not np.all(np.isfinite(np.asarray(had(t9,k)))): nh+=1
    print(f'  K={K:2d}: checked val={cv} grad={cg} hess={ch}  ->  NaN val={nv} grad={ng} hess={nh}', flush=True)
    TOTAL['val']+=cv; TOTAL['grad']+=cg; TOTAL['hess']+=ch; TOTAL['nval']+=nv; TOTAL['ngrad']+=ng; TOTAL['nhess']+=nh
tot_nan = TOTAL['nval']+TOTAL['ngrad']+TOTAL['nhess']
print(f'### TOTAL checked: val={TOTAL["val"]} grad={TOTAL["grad"]} hess={TOTAL["hess"]}', flush=True)
print(f'### TOTAL NaN: val={TOTAL["nval"]} grad={TOTAL["ngrad"]} hess={TOTAL["nhess"]}  ->  '
      f'{"PASS: ZERO NaN without custom_vjp" if tot_nan==0 else "FAIL: "+str(tot_nan)+" NaN"}', flush=True)
