"""COMPREHENSIVE AD-vs-FD for JACOBIAN and HESSIAN, every param, key-averaged. custom_vjp removed.
Linear functional L=Σ cᵢ Mᵢ -> grad = Σ c ∂M (per-param Jacobian dotted with c), Hess = Σ c ∂²M.
Reports per-param grad match + the full PxP Hessian match map, key-averaged (so the discrete params'
score-vs-reparam shows as a per-key mismatch that the expectation reconciles)."""
import os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
import lucid.simulation.simulator as SIM
import lucid.simulation.photon_step as PS
def plain_factory(reflection_fn=PS.scalar_reflection):
    def step(p, d, t, sd, n, sl, ml, g, rp, al, hs, lam, rk, c):
        return PS.photon_iteration_update_factors(p, d, t, sd, n, sl, ml, g, rp, al, hs, lam, rk, c,
                                                  reflection_fn=reflection_fn)
    return step
SIM.make_photon_iteration_update_factors_safe = plain_factory
from lucid.geometry import generate_detector
from lucid.detector_params import load_detector_params
from lucid.sources.calibration_sources import laser_source
from lucid.fitting import build_calibration_problem

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
NPHOT = int(os.environ.get('NPHOT', '40000')); K = 8; NK = int(os.environ.get('NKEYS', '64')); H = 1e-3
det = generate_detector(GEOM); ND = len(det.all_points); Hd = float(getattr(det, 'H', 36.0))
dp = load_detector_params(PHYS, num_sensors=ND)
src = laser_source(position=[0.0, 0.0, Hd / 2 - 0.1], intensity=NPHOT)
sim = SIM.setup_event_simulator(GEOM, NPHOT, temperature=0.2, K=K, is_calibration=True,
                                physics_config=PHYS, hit_mode='aggregated', wavelength_mode=False)
GLOB = ['absorption_length', 'wall_reflection_rate', 'sensor_reflection_rate', 'qe',
        'scatter_length', 'mie_scatter_length', 'g']
prob = build_calibration_problem(sim, [src], dp, GLOB, key=jax.random.PRNGKey(0))
fwd = prob['source_models'][0].forward; th0 = jnp.asarray(prob['theta0'], float); P = th0.shape[0]
c = jnp.asarray(np.random.default_rng(0).standard_normal(ND))
def L(t, k): return jnp.sum(c * fwd(t, k, k))
keys = [jax.random.PRNGKey(7000 + i) for i in range(NK)]

# ---- GRADIENT / JACOBIAN-dot-c, per param ----
gad = jax.jit(jax.grad(L))
def gfd(k):
    return np.array([float((L(th0.at[d].add(H), k) - L(th0.at[d].add(-H), k)) / (2*H)) for d in range(P)])
GA = np.mean([np.asarray(gad(th0, k)) for k in keys], 0)
GF = np.mean([gfd(k) for k in keys], 0)
GAs = np.std([np.asarray(gad(th0, k)) for k in keys], 0); GFs = np.std([gfd(k) for k in keys], 0)
print(f'ND={ND} P={P} K={K} Nphot={NPHOT} keys={NK}\n=== GRADIENT (Σc·∂M): AD vs FD, key-averaged ===')
print(f'{"param":22s} {"AD":>10s} {"FD":>10s} {"|AD-FD|/σ_FD":>12s}  {"std AD":>8s} {"std FD":>9s}')
for d in range(P):
    sigF = GFs[d] / np.sqrt(NK)
    flag = 'MATCH' if abs(GA[d]-GF[d]) < 0.02*abs(GF[d])+1e-6 else ('~exp' if abs(GA[d]-GF[d]) < 3*sigF else 'MISMATCH')
    print(f'{GLOB[d]:22s} {GA[d]:+10.3f} {GF[d]:+10.3f} {abs(GA[d]-GF[d])/(sigF+1e-9):12.1f}  {GAs[d]:8.2f} {GFs[d]:9.2f}  {flag}')

# ---- HESSIAN (Σc·∂²M), full PxP ----
had = jax.jit(jax.hessian(L))
def hfd(k):                                       # central diff of the AD gradient
    M = np.zeros((P, P))
    for d in range(P):
        M[:, d] = (np.asarray(gad(th0.at[d].add(H), k)) - np.asarray(gad(th0.at[d].add(-H), k))) / (2*H)
    return 0.5 * (M + M.T)
HA = np.mean([np.asarray(had(th0, k)) for k in keys], 0)
HF = np.mean([hfd(k) for k in keys], 0)
HAs = np.std([np.asarray(had(th0, k)) for k in keys], 0)
print(f'\n=== HESSIAN (Σc·∂²M): per-entry AD vs FD match map, key-averaged ===')
print('  rows/cols order:', GLOB)
PATH = set(['absorption_length','wall_reflection_rate','sensor_reflection_rate','qe'])
match = np.zeros((P,P), '<U4')
for i in range(P):
    for j in range(P):
        both_path = (GLOB[i] in PATH) and (GLOB[j] in PATH)
        rel = abs(HA[i,j]-HF[i,j]) / (abs(HF[i,j]) + 1e-6)
        match[i,j] = ' ok ' if rel < 0.05 else ('~exp' if abs(HA[i,j]-HF[i,j]) < 3*HAs[i,j]/np.sqrt(NK)+1e-3 else 'BAD ')
print('  match map (ok=AD==FD; BAD=mismatch beyond noise):')
for i in range(P):
    print(f'   {GLOB[i][:8]:8s} ' + ' '.join(match[i]))
print(f'\n  pathwise 4x4 block AD-vs-FD rel = {np.linalg.norm(HA[:4,:4]-HF[:4,:4])/(np.linalg.norm(HF[:4,:4])+1e-9):.2e}')
print(f'  discrete 3x3 block AD-vs-FD rel = {np.linalg.norm(HA[4:,4:]-HF[4:,4:])/(np.linalg.norm(HF[4:,4:])+1e-9):.2e}')
print(f'  AD Hessian finite={np.all(np.isfinite(HA))} symmetric|Δ|={np.max(np.abs(HA-HA.T)):.1e}')
