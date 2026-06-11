"""hello_calibrate — recover the detector's optical parameters from calibration light.

The validated recipe (the one the calibration campaign ran on): build_calibration_problem
-> Fisher/CRB at truth -> Gauss-Newton fit. Seven global optical parameters are recovered
from a perturbed start while all 10764 per-PMT QE/gain factors are marginalised analytically
by a Schur complement. Reports the fit error against the Cramer-Rao bound.

Fast on GPU (~1 min); on CPU it is much slower — set JAX_PLATFORM_NAME if needed.
Run:  python examples/hello_calibrate.py
"""
import jax, jax.numpy as jnp, numpy as np
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources import laser_source, isotropic_source
from lucid.detector_params import DetectorParams
from lucid.fitting import build_calibration_problem, fit, crb

GEOM = 'config/SK_like_geom_config.json'
det = generate_detector(GEOM); NS = len(det.all_points); top, bot, R = det.H/2-.1, -det.H/2+.1, det.r
dp = DetectorParams.from_flat(scatter_length=70., mie_scatter_length=3000., g=0.9,
                              wall_reflection_rate=.2, sensor_reflection_rate=.2,
                              absorption_length=60., qe=0.07, qe_corrections=jnp.ones(NS))
sources = [laser_source(position=[0, 0, top], direction=[0, 0, -1], intensity=1e6),   # down
           laser_source(position=[0, 0, bot], direction=[0, 0, 1], intensity=1e6),    # up
           laser_source(position=[R-.1, 0, 0], direction=[-1, 0, 0], intensity=1e6),  # wall
           isotropic_source(position=[0, 0, 0], intensity=1e6)]                        # flasher
sim = setup_event_simulator(GEOM, 1_000_000, temperature=None, K=8, is_calibration=True,
                            hit_mode='aggregated', wavelength_mode=False,
                            n_cap=100, n_angular=150, n_height=100)

FIELDS = ['g', 'scatter_length', 'mie_scatter_length', 'absorption_length',
          'wall_reflection_rate', 'sensor_reflection_rate', 'qe']
prob = build_calibration_problem(sim, sources, dp, FIELDS, key=jax.random.PRNGKey(1))
sigma = crb(prob['source_models'], prob['theta_true'], NS)['sigma']           # Cramer-Rao bound
start = prob['theta0'] + np.random.default_rng(0).uniform(-.15, .15, prob['theta0'].shape)
res = fit(prob['source_models'], prob['truth_charge'], start, NS, steps=100, refresh=15, nb_h=2)

truth = np.exp(prob['theta0'])
print(f'{"param":22s}{"truth":>9s}{"start":>9s}{"fit":>9s}{"err":>8s}{"CRB":>7s}')
for i, f in enumerate(FIELDS):
    print(f'{f:22s}{truth[i]:9.3f}{np.exp(start[i]):9.3f}{res["theta"][i]:9.3f}'
          f'{res["theta"][i]/truth[i]-1:+7.1%}{sigma[i]:7.1%}')
