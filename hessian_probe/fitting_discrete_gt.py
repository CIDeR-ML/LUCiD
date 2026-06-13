"""Adjudicate the discrete-param (scatter/mie/g) per-sensor Jacobian: AD-score vs FD-secant vs a
HIGH-KEY ground truth GT[s] = (E_key[m_s(th+h)] - E_key[m_s(th-h)])/2h (both AD and FD are unbiased
estimators of this same dE[m]/dtheta, so whichever converges to GT is the trustworthy one for the
per-sensor Fisher/GN). Report, per discrete param: SUM over sensors (scalar, expected to agree) and
PER-SENSOR cosine + rel of {AD-mean, FD-mean} vs GT. Env: NKG (GT keys), NKE (estimator keys)."""
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
NPHOT = int(os.environ.get('NPHOT','40000')); K = 8
NKG = int(os.environ.get('NKG','400')); NKE = int(os.environ.get('NKE','120')); h = 1e-3
det = generate_detector(GEOM); ND = len(det.all_points); Hd = float(getattr(det,'H',36.0))
dp = load_detector_params(PHYS, num_sensors=ND)
src = laser_source(position=[0.0,0.0,Hd/2-0.1], intensity=NPHOT)
sim = SIM.setup_event_simulator(GEOM, NPHOT, temperature=0.2, K=K, is_calibration=True,
                                physics_config=PHYS, hit_mode='aggregated', wavelength_mode=False)
GLOB = ['absorption_length','wall_reflection_rate','sensor_reflection_rate','qe',
        'scatter_length','mie_scatter_length','g']
prob = build_calibration_problem(sim, [src], dp, GLOB, key=jax.random.PRNGKey(0))
S = prob['source_models'][0]; th0 = jnp.asarray(prob['theta0'], float); P = th0.shape[0]
lk = jnp.zeros(ND)
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def rel(a,b): return float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-30))
print(f'ND={ND} P={P} K={K} NPHOT={NPHOT} NKG={NKG} NKE={NKE} h={h}', flush=True)
for d in [4,5,6]:
    th_p = th0.at[d].add(h); th_m = th0.at[d].add(-h)
    # GROUND TRUTH: high-key mean of m at +/-h, secant per sensor
    mp = np.mean([np.asarray(S.m(th_p, lk, *_keys(80000+j))) for j in range(NKG)], 0)
    mm = np.mean([np.asarray(S.m(th_m, lk, *_keys(80000+j))) for j in range(NKG)], 0)
    GT = (mp - mm) / (2*h)
    # ESTIMATORS: NKE-key mean of the per-key AD and FD columns
    AD = np.mean([S.ad_jacobian(th0, lk, *_keys(5000+j))[:,d] for j in range(NKE)], 0)
    FD = np.mean([S.fd_jacobian(th0, lk, *_keys(5000+j))[:,d] for j in range(NKE)], 0)
    print(f'\n--- {GLOB[d]} (d={d}) ---', flush=True)
    print(f'  SUM over sensors:  GT={GT.sum():+.4f}   AD={AD.sum():+.4f}   FD={FD.sum():+.4f}', flush=True)
    print(f'  per-sensor vs GT:  AD cos={cos(AD,GT):+.3f} rel={rel(AD,GT):.3f}   '
          f'FD cos={cos(FD,GT):+.3f} rel={rel(FD,GT):.3f}', flush=True)
    print(f'  ||GT||={np.linalg.norm(GT):.3f}  ||AD||={np.linalg.norm(AD):.3f}  ||FD||={np.linalg.norm(FD):.3f}', flush=True)
print('\n### DONE', flush=True)
