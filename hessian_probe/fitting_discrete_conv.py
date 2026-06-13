"""Confirm AD is the LOW-VARIANCE (better) per-sensor estimator for discrete scatter/mie/g:
(a) AD self-consistency: cos(AD@NKE, AD@NKG) -> 1 and ||AD|| stable if AD is converged/low-var;
(b) FD converges TOWARD AD as keys grow: ||FD@N|| decreases with N toward ||AD||, cos(FD@N, AD@NKG) rises.
If both hold, the per-sensor 'GT-via-secant' was secant-noise and AD is the trustworthy column.
Env: NKG (big), NKE (small)."""
import os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
from lucid.geometry import generate_detector
from lucid.detector_params import load_detector_params
from lucid.sources.calibration_sources import laser_source
from lucid.fitting import build_calibration_problem
from lucid.fitting.gauss_newton import _keys
import lucid.simulation.simulator as SIM
GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
NPHOT = 40000; K = 8; NKG = int(os.environ.get('NKG','500')); NKE = int(os.environ.get('NKE','100'))
det = generate_detector(GEOM); ND = len(det.all_points); Hd = float(getattr(det,'H',36.0))
dp = load_detector_params(PHYS, num_sensors=ND)
src = laser_source(position=[0.0,0.0,Hd/2-0.1], intensity=NPHOT)
sim = SIM.setup_event_simulator(GEOM, NPHOT, temperature=0.2, K=K, is_calibration=True,
                                physics_config=PHYS, hit_mode='aggregated', wavelength_mode=False)
GLOB = ['absorption_length','wall_reflection_rate','sensor_reflection_rate','qe',
        'scatter_length','mie_scatter_length','g']
prob = build_calibration_problem(sim, [src], dp, GLOB, key=jax.random.PRNGKey(0))
S = prob['source_models'][0]; th0 = jnp.asarray(prob['theta0'], float); lk = jnp.zeros(ND)
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
print(f'ND={ND} K={K} NKG={NKG} NKE={NKE}', flush=True)
for d in [4,5,6]:
    AD_all = np.array([S.ad_jacobian(th0, lk, *_keys(5000+j))[:,d] for j in range(NKG)])
    FD_all = np.array([S.fd_jacobian(th0, lk, *_keys(5000+j))[:,d] for j in range(NKG)])
    ADbig = AD_all.mean(0); ADsmall = AD_all[:NKE].mean(0)
    print(f'\n--- {GLOB[d]} ---', flush=True)
    print(f'  AD self-consistency: ||AD@{NKE}||={np.linalg.norm(ADsmall):.2f}  ||AD@{NKG}||={np.linalg.norm(ADbig):.2f}  '
          f'cos(AD@{NKE},AD@{NKG})={cos(ADsmall,ADbig):+.3f}', flush=True)
    for N in [50, 100, 250, NKG]:
        FDn = FD_all[:N].mean(0)
        print(f'  FD@{N:4d}: ||FD||={np.linalg.norm(FDn):8.2f}  cos(FD,AD@{NKG})={cos(FDn,ADbig):+.3f}', flush=True)
print('\n### DONE', flush=True)
