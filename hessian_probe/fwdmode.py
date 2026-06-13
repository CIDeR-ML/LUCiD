"""Does FORWARD-MODE (jvp/jacfwd) work through the calibration custom_vjp? If yes, the efficient
low-variance AD Jacobian (P forward passes) is available and FD is unnecessary. Test: jacfwd(L) vs
grad(L) (reverse) — both pure AD, must be identical; and jax.jacfwd(forward) on the full Jacobian."""
import os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.sources.calibration_sources import laser_source
from lucid.fitting import build_calibration_problem

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
NPHOT = 30000; K = 8
det = generate_detector(GEOM); ND = len(det.all_points); Hd = float(getattr(det, 'H', 36.0))
dp = load_detector_params(PHYS, num_sensors=ND)
src = laser_source(position=[0.0, 0.0, Hd / 2 - 0.1], intensity=NPHOT)
sim = setup_event_simulator(GEOM, NPHOT, temperature=0.2, K=K, is_calibration=True,
                            physics_config=PHYS, hit_mode='aggregated', wavelength_mode=False)
prob = build_calibration_problem(sim, [src], dp, ['scatter_length', 'absorption_length'], key=jax.random.PRNGKey(0))
fwd = prob['source_models'][0].forward; th0 = jnp.asarray(prob['theta0'], float)
c = jnp.asarray(np.random.default_rng(0).standard_normal(ND))
k0 = jax.random.PRNGKey(5000)


def L(t):
    return jnp.sum(c * fwd(t, k0, k0))


print('=== forward-mode availability through the calibration custom_vjp ===')
gR = np.asarray(jax.grad(L)(th0))
print(f'  reverse  grad(L)      = {np.round(gR,4)}')
try:
    gF = np.asarray(jax.jacfwd(L)(th0))
    print(f'  forward  jacfwd(L)    = {np.round(gF,4)}   '
          f'-> forward-mode WORKS; matches reverse: {np.allclose(gR, gF, rtol=1e-4)}')
except Exception as e:
    print(f'  forward  jacfwd(L)    BLOCKED: {type(e).__name__}: {str(e)[:120]}')
try:
    v = jnp.array([1.0, 0.0])
    _, jvp = jax.jvp(L, (th0,), (v,))
    print(f'  jvp(L, e0)            = {float(jvp):+.4f}   -> jvp WORKS')
except Exception as e:
    print(f'  jvp(L, e0)            BLOCKED: {type(e).__name__}: {str(e)[:120]}')
# full per-sensor Jacobian via jacfwd (P forward passes) — the efficient calib Jacobian
try:
    Jf = np.asarray(jax.jacfwd(lambda t: fwd(t, k0, k0))(th0))   # (ND, P)
    print(f'  jacfwd(forward) full Jacobian shape {Jf.shape}, ||={np.linalg.norm(Jf):.3e}  -> WORKS (P passes)')
except Exception as e:
    print(f'  jacfwd(forward) full  BLOCKED: {type(e).__name__}: {str(e)[:120]}')
