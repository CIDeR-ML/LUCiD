"""Per-sensor Jacobian ∂M_i/∂theta — AD (jacrev through the custom_vjp) vs FD (CRN), element-wise.
This is what SourceModel actually uses. Reports norms + cosine so an AD gradient-flow break shows
up as ||J_AD|| ~ 0 while ||J_FD|| is large. Env: NPHOT, K, FIELDS, NKEYS, H."""
import os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.sources.calibration_sources import laser_source
from lucid.fitting import build_calibration_problem

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
NPHOT = int(os.environ.get('NPHOT', '200000')); K = int(os.environ.get('K', '8'))
FIELDS = os.environ.get('FIELDS', 'scatter_length,absorption_length').split(',')
NKEYS = int(os.environ.get('NKEYS', '6')); H = float(os.environ.get('H', '1e-3'))
WLMODE = os.environ.get('WLMODE', '1') == '1'

det = generate_detector(GEOM); ND = len(det.all_points); Hd = float(getattr(det, 'H', 36.0))
dp = load_detector_params(PHYS, num_sensors=ND)
sim = setup_event_simulator(GEOM, NPHOT, temperature=0.2, K=K, is_calibration=True,
                            physics_config=PHYS, hit_mode='aggregated', wavelength_mode=WLMODE)
print(f'wavelength_mode={WLMODE}')
src = laser_source(position=[0.0, 0.0, Hd / 2 - 0.1], intensity=NPHOT)
prob = build_calibration_problem(sim, [src], dp, FIELDS, key=jax.random.PRNGKey(0))
fwd = prob['source_models'][0].forward
theta0 = jnp.asarray(prob['theta0'], float); P = theta0.shape[0]
print(f'fields={FIELDS} P={P} ND={ND} Nphot={NPHOT} K={K}')

jac_ad = jax.jit(lambda th, k: jax.jacrev(lambda t: fwd(t, k, k))(th))   # (ND, P) AD Jacobian


def jac_fd(th, k, h=H):
    base = np.asarray(fwd(th, k, k)); cols = []
    for d in range(P):
        e = jnp.zeros(P).at[d].set(h)
        cols.append((np.asarray(fwd(th + e, k, k)) - base) / h)          # forward diff (matches SourceModel)
    return np.stack(cols, 1)                                              # (ND, P)


def cmp(A, B, lab):
    A, B = np.asarray(A), np.asarray(B)
    for d in range(P):
        a, b = A[:, d], B[:, d]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        print(f'  {lab} d{d}({FIELDS[d]:18s}): ||AD||={np.linalg.norm(a):9.3e} ||FD||={np.linalg.norm(b):9.3e} '
              f'cos={cos:+.4f} rel={np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-30):.3e}')


keys = [jax.random.PRNGKey(100 + i) for i in range(NKEYS)]
print('\n[single key]'); cmp(jac_ad(theta0, keys[0]), jac_fd(theta0, keys[0]), 'k0')
Aad = np.mean([np.asarray(jac_ad(theta0, k)) for k in keys], 0)
Afd = np.mean([jac_fd(theta0, k) for k in keys], 0)
print(f'\n[{NKEYS}-key avg]'); cmp(Aad, Afd, 'avg')
# where is the AD mass? how many sensors have nonzero AD vs FD Jacobian?
for d in range(P):
    nz_ad = int((np.abs(Aad[:, d]) > 1e-9).sum()); nz_fd = int((np.abs(Afd[:, d]) > 1e-9).sum())
    print(f'  d{d}: sensors with |J|>1e-9  AD={nz_ad}  FD={nz_fd}  (of {ND})')
