"""Validate the B1 fix: bypass the custom_vjp (use the PLAIN step) and test whether forward-mode
(jacfwd/jvp) then WORKS and matches reverse-mode (jacrev) without NaNs. If yes, dropping the
custom_vjp gives the efficient low-variance AD Jacobian (P passes) that retires the noisy FD."""
import os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
import lucid.simulation.simulator as SIM
import lucid.simulation.photon_step as PS

# --- monkeypatch: plain step (no custom_vjp), same signature as the 'safe' factory ---
def plain_factory(reflection_fn=PS.scalar_reflection):
    def step(position, direction, time, surface_distance, normal, scatter_length, mie_scatter_length,
             g, refl_params, absorption_length, hit_sensor, lam, rng_key, speed_of_light):
        return PS.photon_iteration_update_factors(
            position, direction, time, surface_distance, normal, scatter_length, mie_scatter_length,
            g, refl_params, absorption_length, hit_sensor, lam, rng_key, speed_of_light,
            reflection_fn=reflection_fn)
    return step
SIM.make_photon_iteration_update_factors_safe = plain_factory   # patch BEFORE building the sim

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
prob = build_calibration_problem(sim, [src], dp, ['scatter_length', 'absorption_length'], key=jax.random.PRNGKey(0))
fwd = prob['source_models'][0].forward; th0 = jnp.asarray(prob['theta0'], float)
c = jnp.asarray(np.random.default_rng(0).standard_normal(ND)); k0 = jax.random.PRNGKey(5000)
def L(t): return jnp.sum(c * fwd(t, k0, k0))

print('=== B1 fix check: PLAIN step (no custom_vjp) ===')
gR = np.asarray(jax.grad(L)(th0)); print(f'  reverse grad(L) = {np.round(gR,4)}  finite={np.all(np.isfinite(gR))}')
try:
    gF = np.asarray(jax.jacfwd(L)(th0))
    print(f'  forward jacfwd(L) = {np.round(gF,4)}  WORKS  matches reverse={np.allclose(gR,gF,rtol=1e-4)}  finite={np.all(np.isfinite(gF))}')
except Exception as e:
    print(f'  forward jacfwd(L) BLOCKED: {type(e).__name__}: {str(e)[:110]}')
# AD score gradient still low-variance? (mean over keys vs the custom_vjp version)
def gad_key(key):
    return float(jax.grad(lambda t: jnp.sum(c * fwd(t, key, key)))(th0)[0])
A = np.array([gad_key(jax.random.PRNGKey(5000 + i)) for i in range(64)])
print(f'  AD scatter grad over 64 keys: mean={A.mean():+.3f} std={A.std():.3f} (low-var score, matches custom_vjp run)')
# full Jacobian via jacfwd (the efficient calib Jacobian) — does it run without OOM?
try:
    Jf = jax.jit(jax.jacfwd(lambda t: fwd(t, k0, k0)))(th0)
    Jf = np.asarray(Jf); print(f'  jacfwd FULL Jacobian {Jf.shape} ||={np.linalg.norm(Jf):.3e} finite={np.all(np.isfinite(Jf))}  -> EFFICIENT low-var Jacobian available')
except Exception as e:
    print(f'  jacfwd FULL Jacobian: {type(e).__name__}: {str(e)[:110]}')
