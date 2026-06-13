"""Is the DiCE/AD Hessian the UNBIASED Hessian (DiCE by construction), or is there a 2nd-order bug?
Ground truth ∇²E[L] = d/dθ E[∇L]: FD the EXPECTED AD gradient (low-variance -> precise with many keys),
NOT the pure-forward 2nd difference (astronomically noisy). Compare to E[AD Hessian] over many keys.
If they AGREE -> DiCE Hessian correct, my earlier FD-Hessian 'BAD' was just FD noise. If not -> bug.
First: confirm AD gradient == FD gradient in expectation (many keys) for ALL params."""
import os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
import lucid.simulation.simulator as SIM
import lucid.simulation.photon_step as PS
def plain_factory(reflection_fn=PS.scalar_reflection):
    def step(p, d, t, sd, n, sl, ml, g, rp, al, hs, lam, rk, cc):
        return PS.photon_iteration_update_factors(p, d, t, sd, n, sl, ml, g, rp, al, hs, lam, rk, cc,
                                                  reflection_fn=reflection_fn)
    return step
SIM.make_photon_iteration_update_factors_safe = plain_factory
from lucid.geometry import generate_detector
from lucid.detector_params import load_detector_params
from lucid.sources.calibration_sources import laser_source
from lucid.fitting import build_calibration_problem

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
NPHOT = int(os.environ.get('NPHOT', '40000')); K = 8
NK = int(os.environ.get('NKEYS', '2000')); h = float(os.environ.get('H', '0.02'))
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
gad = jax.jit(jax.grad(L)); had = jax.jit(jax.hessian(L))
keys = [jax.random.PRNGKey(11000 + i) for i in range(NK)]
print(f'ND={ND} P={P} K={K} Nphot={NPHOT} keys={NK} h={h}')

# ---- 1) GRADIENT: AD vs FD in expectation (many keys) ----
GA = np.array([np.asarray(gad(th0, k)) for k in keys])          # (NK, P)
def gfd(k): return np.array([float((L(th0.at[d].add(h), k) - L(th0.at[d].add(-h), k)) / (2*h)) for d in range(P)])
GF = np.array([gfd(k) for k in keys])
print('\n=== GRADIENT (many-key): AD mean vs FD mean (±SE) ===')
for d in range(P):
    am, ae = GA[:,d].mean(), GA[:,d].std()/np.sqrt(NK); fm, fe = GF[:,d].mean(), GF[:,d].std()/np.sqrt(NK)
    z = abs(am-fm)/np.sqrt(ae**2+fe**2+1e-12)
    print(f'  {GLOB[d]:22s} AD={am:+9.3f}±{ae:5.3f}  FD={fm:+9.3f}±{fe:6.3f}  z={z:4.1f}  {"AGREE" if z<3 else "DIFFER"}')

# ---- 2) HESSIAN DIAGONAL: DiCE/AD  vs  ground truth (FD of expected AD gradient) ----
HA = np.array([np.asarray(had(th0, k)) for k in keys])           # (NK, P, P)
print('\n=== HESSIAN DIAGONAL: DiCE/AD  vs  GT=d/dθ E[∇L] (FD of expected AD grad) ===')
for j in range(P):
    gp = np.mean([np.asarray(gad(th0.at[j].add(h), k))[j] for k in keys])   # E[∂L/∂θj] at +h
    gm = np.mean([np.asarray(gad(th0.at[j].add(-h), k))[j] for k in keys])  # at -h
    sp = np.std([np.asarray(gad(th0.at[j].add(h), k))[j] for k in keys])/np.sqrt(NK)
    gt = (gp - gm) / (2*h); gt_err = np.sqrt(2)*sp/(2*h)
    ad = HA[:,j,j].mean(); ad_err = HA[:,j,j].std()/np.sqrt(NK)
    z = abs(ad-gt)/np.sqrt(ad_err**2+gt_err**2+1e-12)
    print(f'  {GLOB[j]:22s} DiCE H_jj={ad:+9.2f}±{ad_err:6.2f}   GT={gt:+9.2f}±{gt_err:6.2f}   z={z:4.1f}  {"AGREE" if z<3 else "*** DIFFER (2nd-order BUG)"}')
