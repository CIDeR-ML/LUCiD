"""Map AD-vs-FD per single optical parameter, with key-averaging to test score convergence.
Each param fit alone (1-D), mse loss, wavelength_mode=False. Reports AD grad, FD grad averaged
over increasing #keys + the AD/FD ratio. A pathwise param -> AD==FD at 1 key; a discrete-decision
param -> AD (score) != FD per key, converging (or not) as keys grow."""
import os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.sources.calibration_sources import laser_source
from lucid.fitting import build_calibration_problem

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
NPHOT = int(os.environ.get('NPHOT', '40000')); K = int(os.environ.get('K', '8')); H = 1e-3
det = generate_detector(GEOM); ND = len(det.all_points); Hd = float(getattr(det, 'H', 36.0))
dp = load_detector_params(PHYS, num_sensors=ND)
sim = setup_event_simulator(GEOM, NPHOT, temperature=0.2, K=K, is_calibration=True,
                            physics_config=PHYS, hit_mode='aggregated', wavelength_mode=False)
src = laser_source(position=[0.0, 0.0, Hd / 2 - 0.1], intensity=NPHOT)

FIELDS = ['absorption_length', 'scatter_length', 'mie_scatter_length', 'g',
          'wall_reflection_rate', 'sensor_reflection_rate', 'qe']
NSETS = [1, 8, 32]
print(f'Nphot={NPHOT} K={K} | per-param AD vs FD grad (mse loss), averaged over keys\n')
print(f'{"param":22s} {"category":10s} | ' + '  '.join(f'n={n:<3d}AD/FD' for n in NSETS))
for f in FIELDS:
    prob = build_calibration_problem(sim, [src], dp, [f], key=jax.random.PRNGKey(0))
    fwd = prob['source_models'][0].forward; th0 = jnp.asarray(prob['theta0'], float)
    target = jax.lax.stop_gradient(fwd(th0, jax.random.PRNGKey(0), jax.random.PRNGKey(0)))
    Wm = (np.asarray(target) > 0).astype(float)

    def loss(t, key):
        M = fwd(t, key, key)
        return 0.5 * jnp.sum(jnp.asarray(Wm) * (jnp.sqrt(M + 1e-8) - jnp.sqrt(target + 1e-8)) ** 2)
    gad = jax.jit(jax.grad(loss))

    def gfd(key):
        e = jnp.array([H])
        return float((loss(th0 + e, key) - loss(th0 - e, key)) / (2 * H))
    cells = []
    for n in NSETS:
        ks = [jax.random.PRNGKey(1000 + i) for i in range(n)]
        a = np.mean([float(gad(th0, k)[0]) for k in ks]); b = np.mean([gfd(k) for k in ks])
        cells.append(f'{a:+8.2f}/{b:+8.2f}')
    a1 = float(gad(th0, jax.random.PRNGKey(1000))[0]); b1 = gfd(jax.random.PRNGKey(1000))
    cat = 'PATHWISE' if abs(a1 - b1) < 1e-6 + 0.02 * abs(b1) else 'score/disc'
    print(f'{f:22s} {cat:10s} | ' + '  '.join(cells))
