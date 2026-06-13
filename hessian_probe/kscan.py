"""WHY scatter/mie/g differ from absorption: the DEPOSIT (Rao-Blackwellised reach, pathwise) vs the
TRAJECTORY (sampled hard decisions, score/flip). Prediction: at K=1 there is no downstream, so the
free-path/angle DECISIONS can't affect the charge -> scatter/mie are pure-pathwise (AD==FD, low var)
and g~0; the AD-vs-FD gap and the FD variance must GROW WITH K (more trajectory). Measured: AD & FD
mean/std over keys at K=1,2,4,8, on a clean linear functional."""
import os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.sources.calibration_sources import laser_source
from lucid.fitting import build_calibration_problem

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
NPHOT = int(os.environ.get('NPHOT', '20000')); NK = int(os.environ.get('NKEYS', '48')); H = 1e-3
det = generate_detector(GEOM); ND = len(det.all_points); Hd = float(getattr(det, 'H', 36.0))
dp = load_detector_params(PHYS, num_sensors=ND)
src = laser_source(position=[0.0, 0.0, Hd / 2 - 0.1], intensity=NPHOT)
c = jnp.asarray(np.random.default_rng(0).standard_normal(ND))
keys = [jax.random.PRNGKey(5000 + i) for i in range(NK)]

print(f'Nphot={NPHOT} keys={NK} | AD(score) & FD(CRN) grad mean±std vs K  (linear functional)\n')
print(f'{"field":20s} {"K":>2s} | {"AD mean":>9s} {"AD std":>8s} | {"FD mean":>9s} {"FD std":>9s}  var(FD)/var(AD)')
for field in ['scatter_length', 'mie_scatter_length', 'g', 'absorption_length']:
    for K in [1, 2, 4, 8]:
        sim = setup_event_simulator(GEOM, NPHOT, temperature=0.2, K=K, is_calibration=True,
                                    physics_config=PHYS, hit_mode='aggregated', wavelength_mode=False)
        prob = build_calibration_problem(sim, [src], dp, [field], key=jax.random.PRNGKey(0))
        fwd = prob['source_models'][0].forward; th0 = jnp.asarray(prob['theta0'], float)
        gad = jax.jit(jax.grad(lambda t, k: jnp.sum(c * fwd(t, k, k))))
        def gf(k):
            e = jnp.zeros_like(th0).at[0].set(H)
            return float((jnp.sum(c*fwd(th0+e,k,k)) - jnp.sum(c*fwd(th0-e,k,k))) / (2*H))
        A = np.array([float(gad(th0, k)[0]) for k in keys]); F = np.array([gf(k) for k in keys])
        ratio = (F.std()**2) / (A.std()**2 + 1e-30)
        print(f'{field:20s} {K:>2d} | {A.mean():+9.3f} {A.std():8.3f} | {F.mean():+9.3f} {F.std():9.3f}  {ratio:10.1f}')
    print()
