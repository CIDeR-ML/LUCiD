"""Diagnostic: does calibrate_optics recover g/scatter/Mie with MORE STEPS, or is it the
k<->scatter degeneracy? Fit at increasing step counts, report fractional error per field."""
import sys; sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_unification')
import jax, jax.numpy as jnp, numpy as np
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources import laser_source, isotropic_source
from lucid.detector_params import DetectorParams
from lucid.fitting import build_calibration_problem, fit, crb

GEOM = 'config/SK_like_geom_config.json'
det = generate_detector(GEOM); NS = len(det.all_points); top, bot, R = det.H / 2 - .1, -det.H / 2 + .1, det.r
dp = DetectorParams.from_flat(scatter_length=70., mie_scatter_length=3000., g=0.9, wall_reflection_rate=.2,
                              sensor_reflection_rate=.2, absorption_length=60., qe=0.07, qe_corrections=jnp.ones(NS))
sources = [laser_source(position=[0, 0, top], direction=[0, 0, -1], intensity=1e6),
           laser_source(position=[0, 0, bot], direction=[0, 0, 1], intensity=1e6),
           laser_source(position=[R - .1, 0, 0], direction=[-1, 0, 0], intensity=1e6),
           isotropic_source(position=[0, 0, 0], intensity=1e6)]
sim = setup_event_simulator(GEOM, 1_000_000, temperature=None, K=8, is_calibration=True, hit_mode='aggregated',
                            wavelength_mode=False, n_cap=100, n_angular=150, n_height=100)
FIELDS = ['g', 'scatter_length', 'mie_scatter_length', 'absorption_length',
          'wall_reflection_rate', 'sensor_reflection_rate', 'qe']
prob = build_calibration_problem(sim, sources, dp, FIELDS, key=jax.random.PRNGKey(1))
sig = np.asarray(crb(prob['source_models'], prob['theta_true'], NS)['sigma'])
truth = np.exp(prob['theta0'])
start = prob['theta0'] + np.random.default_rng(0).uniform(-.15, .15, prob['theta0'].shape)
STEPS = [100, 400, 1000, 2500]
res = {s: fit(prob['source_models'], prob['truth_charge'], start, NS, steps=s, refresh=15, nb_h=2) for s in STEPS}
print('STEPS', STEPS)
print('%-20s %6s %7s | err vs truth (%) at each step count' % ('field', 'CRB%', 'start%'))
for i, f in enumerate(FIELDS):
    err = lambda s: 100 * (np.asarray(res[s]['theta'])[i] / truth[i] - 1)
    row = ' '.join('%+7.1f' % err(s) for s in STEPS)
    print('%-20s %6.2f %+7.1f | %s' % (f, 100 * sig[i], 100 * (np.exp(start[i]) / truth[i] - 1), row))
