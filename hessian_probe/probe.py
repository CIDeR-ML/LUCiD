"""ANCHOR probe: AD vs FD gradient AND Hessian of the calibration loss over OPTICAL globals.

Builds the real calibration forward (build_calibration_problem) over a chosen set of optical
globals, then compares — per single key and averaged over many keys:
  * AD gradient (jax.grad, reverse-mode through the custom_vjp) vs FD gradient (central, CRN);
  * AD Hessian via jacrev∘jacrev (reverse-over-reverse) AND jax.hessian (jacfwd∘jacrev — expected
    to hit the custom_vjp forward-mode block) vs FD Hessian (central diff of the AD gradient).
The custom_vjp only scrubs NaN cotangents (B1); it blocks forward-mode. Loss = sum of mean charge
(clean scalar) by default; --mse uses the √-MSE calibration loss.

Env: NPHOT, K, FIELDS (comma list), NKEYS, H (fd step), LOSS=sum|mse.
"""
import os, sys, time
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian')
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.sources.calibration_sources import laser_source
from lucid.fitting import build_calibration_problem

GEOM = 'config/SK_like_geom_config.json'; PHYS = 'config/SK_like_physics_config.json'
NPHOT = int(os.environ.get('NPHOT', '200000')); K = int(os.environ.get('K', '8'))
FIELDS = os.environ.get('FIELDS', 'scatter_length,absorption_length').split(',')
NKEYS = int(os.environ.get('NKEYS', '8')); H = float(os.environ.get('H', '1e-3'))
LOSS = os.environ.get('LOSS', 'sum')
os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')

det = generate_detector(GEOM); ND = len(det.all_points)
dp = load_detector_params(PHYS, num_sensors=ND)
Hd = float(det.H) if hasattr(det, 'H') else 18.0
sim = setup_event_simulator(GEOM, NPHOT, temperature=0.2, K=K, is_calibration=True,
                            physics_config=PHYS, hit_mode='aggregated', wavelength_mode=os.environ.get('WLMODE','1')=='1')
src = laser_source(position=[0.0, 0.0, Hd / 2 - 0.1], intensity=NPHOT)
prob = build_calibration_problem(sim, [src], dp, FIELDS, key=jax.random.PRNGKey(0))
fwd = prob['source_models'][0].forward            # forward(theta_log, ek, pk) -> (ND,) mean charge
theta0 = jnp.asarray(prob['theta0'], float)        # log truth globals
P = theta0.shape[0]
target = jax.lax.stop_gradient(fwd(theta0, jax.random.PRNGKey(0), jax.random.PRNGKey(0)))
W = (np.asarray(target) > 0).astype(float)         # lit-sensor mask
print(f'fields={FIELDS} P={P} ND={ND} Nphot={NPHOT} K={K} loss={LOSS} | theta0={np.round(np.asarray(theta0),4)}')


def scalar(theta, key):
    M = fwd(theta, key, key)
    if LOSS == 'mse':
        eps = 1e-8
        return 0.5 * jnp.sum(jnp.asarray(W) * (jnp.sqrt(M + eps) - jnp.sqrt(target + eps)) ** 2)
    return jnp.sum(M)


g_ad = jax.jit(jax.grad(scalar))
H_rr = jax.jit(jax.jacrev(jax.jacrev(scalar)))     # reverse-over-reverse (should pass custom_vjp)


def g_fd(theta, key, h=H):
    out = np.zeros(P)
    for d in range(P):
        e = jnp.zeros(P).at[d].set(h)
        out[d] = float((scalar(theta + e, key) - scalar(theta - e, key)) / (2 * h))
    return out


def H_fd(theta, key, h=H):                          # central diff of the AD gradient
    M = np.zeros((P, P))
    for d in range(P):
        e = jnp.zeros(P).at[d].set(h)
        M[:, d] = (np.asarray(g_ad(theta + e, key)) - np.asarray(g_ad(theta - e, key))) / (2 * h)
    return 0.5 * (M + M.T)


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b); n = np.linalg.norm(a - b); d = np.linalg.norm(b) + 1e-30
    return n / d


keys = [jax.random.PRNGKey(100 + i) for i in range(NKEYS)]
# ---- single-key ----
k0 = keys[0]
gA, gF = np.asarray(g_ad(theta0, k0)), g_fd(theta0, k0)
print(f'\n[single key] AD grad {np.round(gA,4)}  FD grad {np.round(gF,4)}  rel={rel(gA,gF):.2e}')
HArr, HF = np.asarray(H_rr(theta0, k0)), H_fd(theta0, k0)
print(f'[single key] AD Hess (rev-rev):\n{np.round(HArr,4)}\nFD Hess:\n{np.round(HF,4)}  rel={rel(HArr,HF):.2e}')
try:
    HAfr = np.asarray(jax.hessian(scalar)(theta0, k0))
    print(f'[single key] jax.hessian (jacfwd∘jacrev) rel-to-FD={rel(HAfr,HF):.2e}  (forward-mode WORKED)')
except Exception as e:
    print(f'[single key] jax.hessian (jacfwd∘jacrev) BLOCKED: {type(e).__name__}: {str(e)[:90]}')
# ---- multi-key average ----
gA_m = np.mean([np.asarray(g_ad(theta0, k)) for k in keys], 0)
gF_m = np.mean([g_fd(theta0, k) for k in keys], 0)
HArr_m = np.mean([np.asarray(H_rr(theta0, k)) for k in keys], 0)
HF_m = np.mean([H_fd(theta0, k) for k in keys], 0)
print(f'\n[{NKEYS}-key avg] grad rel(AD,FD)={rel(gA_m,gF_m):.2e}   Hess rel(AD,FD)={rel(HArr_m,HF_m):.2e}')
print(f'[{NKEYS}-key avg] AD grad {np.round(gA_m,4)} | FD grad {np.round(gF_m,4)}')
print(f'[{NKEYS}-key avg] AD Hess:\n{np.round(HArr_m,4)}\nFD Hess:\n{np.round(HF_m,4)}')
