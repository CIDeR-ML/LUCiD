"""Do we get AD for ALL parameters, and what's the bottleneck? With the custom_vjp removed:
  - jacfwd over ALL optical globals: finite? no OOM? matches reverse grad? cost (P passes)?
  - jax.hessian over all globals: works? finite?
  - jacrev over the full ND-sensor output: OOM (the reason jacfwd is needed)?
  - per-PMT k (qe_corrections, 10764-dim): jacfwd would be 10764 passes (infeasible) -> but it's
    LINEAR (analytic Jk=0.5 m) so no AD needed; confirm it's linear.
  - NaN robustness: evaluate the AD Jacobian at perturbed param points (away from truth)."""
import os, sys, time
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
import lucid.simulation.simulator as SIM
import lucid.simulation.photon_step as PS
def plain_factory(reflection_fn=PS.scalar_reflection):
    def step(position, direction, time_, surface_distance, normal, scatter_length, mie_scatter_length,
             g, refl_params, absorption_length, hit_sensor, lam, rng_key, speed_of_light):
        return PS.photon_iteration_update_factors(position, direction, time_, surface_distance, normal,
            scatter_length, mie_scatter_length, g, refl_params, absorption_length, hit_sensor, lam,
            rng_key, speed_of_light, reflection_fn=reflection_fn)
    return step
SIM.make_photon_iteration_update_factors_safe = plain_factory

from lucid.geometry import generate_detector
from lucid.detector_params import load_detector_params
from lucid.sources.calibration_sources import laser_source
from lucid.fitting import build_calibration_problem

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
NPHOT = 30000; K = 8
det = generate_detector(GEOM); ND = len(det.all_points); Hd = float(getattr(det, 'H', 36.0))
dp = load_detector_params(PHYS, num_sensors=ND)
src = laser_source(position=[0.0, 0.0, Hd / 2 - 0.1], intensity=NPHOT)
sim = SIM.setup_event_simulator(GEOM, NPHOT, temperature=0.2, K=K, is_calibration=True,
                                physics_config=PHYS, hit_mode='aggregated', wavelength_mode=False)
GLOB = ['scatter_length', 'mie_scatter_length', 'g', 'absorption_length',
        'wall_reflection_rate', 'sensor_reflection_rate', 'qe']
prob = build_calibration_problem(sim, [src], dp, GLOB, key=jax.random.PRNGKey(0))
fwd = prob['source_models'][0].forward; th0 = jnp.asarray(prob['theta0'], float); P = th0.shape[0]
k0 = jax.random.PRNGKey(9)
print(f'ND={ND} P(globals)={P}')

# 1) jacfwd over ALL globals — the efficient calib Jacobian
t = time.time(); Jf = np.asarray(jax.jit(jax.jacfwd(lambda t: fwd(t, k0, k0)))(th0)); dt = time.time() - t
print(f'jacfwd FULL ({Jf.shape}) finite={np.all(np.isfinite(Jf))}  ||col||={np.round(np.linalg.norm(Jf,axis=0),3)}  [{dt:.1f}s]')
# match reverse grad of a linear functional?
c = jnp.asarray(np.random.default_rng(0).standard_normal(ND))
gR = np.asarray(jax.grad(lambda t: jnp.sum(c * fwd(t, k0, k0)))(th0))
print(f'  jacfwd vs reverse grad (Σc·M): max|Δ|={np.max(np.abs(c.__array__()@Jf - gR)):.2e}  -> {"MATCH" if np.allclose(c.__array__()@Jf, gR, rtol=1e-4) else "MISMATCH"}')

# 2) jax.hessian over all globals (forward-over-reverse) — finite? no OOM?
try:
    t = time.time(); Hh = np.asarray(jax.hessian(lambda t: jnp.sum(c * fwd(t, k0, k0)))(th0)); dt = time.time() - t
    print(f'jax.hessian ({Hh.shape}) finite={np.all(np.isfinite(Hh))}  sym|Δ|={np.max(np.abs(Hh-Hh.T)):.1e}  [{dt:.1f}s]')
except Exception as e:
    print(f'jax.hessian BLOCKED: {type(e).__name__}: {str(e)[:90]}')

# 3) jacrev over full ND output — the OOM that forces jacfwd
try:
    Jr = jax.jit(jax.jacrev(lambda t: fwd(t, k0, k0)))(th0); print(f'jacrev FULL ran ({np.asarray(Jr).shape})')
except Exception as e:
    print(f'jacrev FULL: {type(e).__name__}: {str(e)[:70]} -> OOM as expected (why jacfwd is needed)')

# 4) per-PMT k linearity: M(k) must be linear in k (so Jk analytic, no AD over 10764 params)
prob2 = build_calibration_problem(sim, [src], dp, ['scatter_length'], key=jax.random.PRNGKey(0))
unrav = prob2['unravel']
def Mk(kval): return sim(src, unrav(prob2['theta0'], kval), k0)[0]
M1, M2 = np.asarray(Mk(1.0)), np.asarray(Mk(2.0))
lit = M1 > 1e-6
print(f'per-PMT k linearity: M(2k)/M(k) on lit sensors mean={np.mean((M2[lit]/M1[lit])):.4f} (==2 => LINEAR, Jk=0.5m analytic, no AD needed)')

# 5) NaN robustness at perturbed param points (away from truth)
bad = 0
for s in [-1.5, -0.8, +0.8, +1.5]:
    Jp = np.asarray(jax.jacfwd(lambda t: fwd(t, k0, k0))(th0 + s * jnp.ones(P) * 0.3))
    if not np.all(np.isfinite(Jp)): bad += 1
print(f'NaN robustness: jacfwd finite at {4-bad}/4 perturbed param points')
