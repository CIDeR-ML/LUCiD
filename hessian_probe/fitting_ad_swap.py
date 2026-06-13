"""Validate the FD->AD Jacobian swap in gauss_newton + fisher, and time it. NO monkeypatch:
the code itself is now custom_vjp-free, so this is the end-to-end test that jacfwd works in the
real fitter. Checks: (1) per-param J_AD vs J_FD single-key (pathwise match exactly; discrete
scatter/mie/g differ per-key = score-vs-secant); (2) many-key MEAN J_AD vs MEAN J_FD (all params
reconcile) + per-key VARIANCE (AD should be lower); (3) downstream GN normal matrix JtWJ and the
CRB sigma, AD vs FD; (4) timing compile+steady, AD vs FD."""
import os, sys, time
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
from lucid.geometry import generate_detector
from lucid.detector_params import load_detector_params
from lucid.sources.calibration_sources import laser_source
from lucid.fitting import build_calibration_problem
from lucid.fitting.gauss_newton import _keys, make_constrained_schur
import lucid.simulation.simulator as SIM
GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
NPHOT = int(os.environ.get('NPHOT','40000')); K = 8; NK = int(os.environ.get('NK','60'))
det = generate_detector(GEOM); ND = len(det.all_points); Hd = float(getattr(det,'H',36.0))
dp = load_detector_params(PHYS, num_sensors=ND)
src = laser_source(position=[0.0,0.0,Hd/2-0.1], intensity=NPHOT)
sim = SIM.setup_event_simulator(GEOM, NPHOT, temperature=0.2, K=K, is_calibration=True,
                                physics_config=PHYS, hit_mode='aggregated', wavelength_mode=False)
GLOB = ['absorption_length','wall_reflection_rate','sensor_reflection_rate','qe',
        'scatter_length','mie_scatter_length','g']
PATH = {0,1,2,3}  # pathwise; 4,5,6 = scatter/mie/g (discrete, DiCE-score)
prob = build_calibration_problem(sim, [src], dp, GLOB, key=jax.random.PRNGKey(0))
S = prob['source_models'][0]; th0 = jnp.asarray(prob['theta0'], float); P = th0.shape[0]
lk = jnp.zeros(ND); W = np.ones(ND)
print(f'ND={ND} P={P} K={K} NPHOT={NPHOT} NK={NK}', flush=True)

# (1) single-key per-param column match
ek, pk = _keys(1234)
Ja = S.ad_jacobian(th0, lk, ek, pk); Jf = S.fd_jacobian(th0, lk, ek, pk)
print('=== (1) single-key J_AD vs J_FD, per-param column rel ===', flush=True)
for d in range(P):
    rel = np.linalg.norm(Ja[:,d]-Jf[:,d])/(np.linalg.norm(Jf[:,d])+1e-12)
    print(f'  {GLOB[d]:22s} rel={rel:8.4f}  {"pathwise (expect ~0)" if d in PATH else "discrete (per-key differs; reconcile in mean)"}', flush=True)

# (2) many-key mean + variance
AJ = np.zeros((NK,ND,P)); FJ = np.zeros((NK,ND,P))
for j in range(NK):
    ek, pk = _keys(5000+j)
    AJ[j] = S.ad_jacobian(th0, lk, ek, pk); FJ[j] = S.fd_jacobian(th0, lk, ek, pk)
Am, Fm = AJ.mean(0), FJ.mean(0)
print(f'=== (2) {NK}-key MEAN J_AD vs MEAN J_FD + per-key column-norm variance ===', flush=True)
for d in range(P):
    relmean = np.linalg.norm(Am[:,d]-Fm[:,d])/(np.linalg.norm(Fm[:,d])+1e-12)
    cn_a = np.linalg.norm(AJ[:,:,d],axis=1); cn_f = np.linalg.norm(FJ[:,:,d],axis=1)
    sa, sf = cn_a.std()/(cn_a.mean()+1e-12), cn_f.std()/(cn_f.mean()+1e-12)
    print(f'  {GLOB[d]:22s} mean-rel={relmean:8.4f}  colnorm CV: AD={sa:7.4f} FD={sf:7.4f}  '
          f'FD/AD var={ (sf/(sa+1e-12)):5.1f}x', flush=True)

# (3) downstream GN normal matrix JtWJ and CRB sigma (AD vs FD), averaged over NK
Htt_a = (Am*W[:,None]).T @ Am; Htt_f = (Fm*W[:,None]).T @ Fm
print(f'=== (3) GN normal matrix JtWJ rel(AD,FD) = {np.linalg.norm(Htt_a-Htt_f)/(np.linalg.norm(Htt_f)+1e-12):.4f} ===', flush=True)
def crb_sigma(Jmean):
    mi = np.array(S.m(th0, lk, *_keys(9999))); Jk = 0.5*mi
    Htt = (Jmean*W[:,None]).T @ Jmean; Htk = (Jmean*W[:,None]).T * Jk[None,:]; Hkk = W*(Jk*Jk)+1e-12
    Minv = make_constrained_schur(Hkk); F = 4.0*(Htt - Htk @ Minv(Htk.T))
    cov = np.linalg.inv(F + 1e-9*np.median(np.diag(F))*np.eye(P))
    return np.sqrt(np.clip(np.diag(cov),0,None))
sa, sf = crb_sigma(Am), crb_sigma(Fm)
print('  CRB fractional sigma (AD vs FD):', flush=True)
for d in range(P):
    print(f'    {GLOB[d]:22s} AD={sa[d]:.4e}  FD={sf[d]:.4e}  rel={abs(sa[d]-sf[d])/(sf[d]+1e-12):7.4f}', flush=True)

# (4) timing: compile (1st) + steady (median over reps, fresh keys)
def bench(fn, label, reps=8):
    ek,pk=_keys(1); t=time.perf_counter(); jax.block_until_ready(jnp.asarray(fn(ek,pk))); comp=time.perf_counter()-t
    ts=[]
    for r in range(reps):
        ek,pk=_keys(100+r); t=time.perf_counter(); jax.block_until_ready(jnp.asarray(fn(ek,pk))); ts.append(time.perf_counter()-t)
    print(f'  {label:18s} compile={comp:6.2f}s  steady={1000*np.median(ts):7.1f} ms', flush=True)
print('=== (4) timing (single Jacobian eval) ===', flush=True)
bench(lambda ek,pk: S.ad_jacobian(th0,lk,ek,pk), 'AD jacfwd')
bench(lambda ek,pk: S.fd_jacobian(th0,lk,ek,pk), f'FD ({P}+1 fwd)')
print('### DONE', flush=True)
