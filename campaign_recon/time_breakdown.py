"""Per-stage wall-clock breakdown of ONE seeded recon event — find where the time goes.
Times: sim build/compile, data gen, each seeder stage, the fit. CUDA_VISIBLE_DEVICES picks GPU."""
import os, sys, time
import numpy as np, jax, jax.numpy as jnp
_R = '/sdf/group/neutrino/omara/LUCiD_unification'; sys.path.insert(0, _R)
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.sources.event_io import read_photon_data_from_photonsim
from lucid.fitting import ReconModel, fit_track, vec9_dir, track_from_vec9
from lucid.fitting.recon import vec9_from_track
from lucid.optimization.grid_search import hierarchical_position_grid_search, get_detector_bounds
from lucid.optimization.utils.functions import hierarchical_direction_search_cone, energy_scan_optimization

def clk(lab, t):
    dt = time.time() - t; print(f'  {lab:32s} {dt:7.1f}s', flush=True); return time.time()

ROOT = '/sdf/group/neutrino/omara/LUCiD_recon/data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root'
GEOM = os.path.join(_R, 'config/SK_like_geom_config.json'); PHYS = os.path.join(_R, 'config/SK_like_physics_config.json')
K, NBUF = 8, 400_000
GRID = dict(n_cap=80, n_angular=120, n_height=80)
NPH_PRED = int(os.environ.get('NPH_PRED', '250000')); NITERS = int(os.environ.get('NITERS', '250')); NKEYS = int(os.environ.get('NKEYS', '4'))
T = time.time(); t = T
det = generate_detector(GEOM); ND = len(det.all_points); POS = np.asarray(det.all_points); bounds = get_detector_bounds(det)
dp_data = load_detector_params(PHYS, num_sensors=ND); dp_data = dp_data._replace(response=dp_data.response._replace(tts=jnp.asarray(2.5)))
data_sim = setup_event_simulator(GEOM, NBUF, temperature=None, K=K, is_data=True, hit_mode='realistic',
    physics_config=PHYS, default_detector_params=dp_data, particle='muon', wavelength_mode=True, apply_smearing=False, **GRID)
pred = setup_event_simulator(GEOM, NPH_PRED, temperature=0.1, K=K, hit_mode='per_photon',
    physics_config=PHYS, default_detector_params=True, particle='muon', wavelength_mode=True, pos_grad_threshold=K, n_grad_iters=K, **GRID)
model = ReconModel(pred, ND, sigma=2.5, delta=1.0)
t = clk('build sims (no compile yet)', t)
print(f'config: NPH_PRED={NPH_PRED} NITERS={NITERS} NKEYS={NKEYS} grid={GRID} ND={ND}', flush=True)

# replicate truth prep
def _rotax(u, deg):
    a = np.radians(deg); ca, sa = np.cos(a), np.sin(a); u = u / np.linalg.norm(u)
    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]); return np.eye(3)*ca+sa*ux+(1-ca)*np.outer(u, u)
raw = read_photon_data_from_photonsim(ROOT, 0); rng = np.random.default_rng(100003)
beta = np.degrees(np.arccos(rng.uniform(-1, 1))); al = rng.uniform(0, 2*np.pi); axis = np.array([-np.sin(al), np.cos(al), 0.])
rr = 12.*np.sqrt(rng.uniform()); ph = rng.uniform(0, 2*np.pi); sh = np.array([rr*np.cos(ph), rr*np.sin(ph), rng.uniform(-12, 12)])*100.
O = np.asarray(raw['photon_origins']).astype(float); D = np.asarray(raw['photon_directions']).astype(float); R = _rotax(axis, beta); cc = O.mean(0)
raw = dict(raw); raw['photon_origins'] = (O-cc)@R.T+cc+sh; raw['photon_directions'] = D@R.T
P = np.asarray(raw['photon_origins'])/100.; tt = np.asarray(raw['photon_times']); C = P-P.mean(0); _, _, vt = np.linalg.svd(C, full_matrices=False); d = vt[0]
if np.corrcoef(C@d, tt)[0, 1] < 0: d = -d
vtx0 = P.mean(0)+(C@d).min()*d; pol = np.arccos(np.clip(d[2], -1, 1)); az = np.arctan2(d[1], d[0])
th9 = np.array([float(raw['energy']), vtx0[0], vtx0[1], vtx0[2], np.sin(pol), np.cos(pol), np.sin(az), np.cos(az), 0.])
n = int(P.shape[0]); reps = int(np.ceil(NBUF/n))
pd = {'photon_origins': jnp.asarray(np.tile(raw['photon_origins'], (reps, 1))[:NBUF]),
      'photon_directions': jnp.asarray(np.tile(raw['photon_directions'], (reps, 1))[:NBUF]),
      'photon_times': jnp.asarray(np.tile(raw['photon_times'], reps)[:NBUF]), 'N': jnp.asarray(n), 'apply_rotation': False,
      'rotation_axis': jnp.array([1., 0., 0.]), 'rotation_angle': jnp.array(0.), 'apply_translation': False, 'translation_vector': jnp.zeros(3)}
if 'wavelengths' in raw: pd['wavelengths'] = jnp.asarray(np.tile(raw['wavelengths'], reps)[:NBUF])
t = clk('truth prep', t)

c, tt2 = jax.lax.stop_gradient(data_sim(track_from_vec9(jnp.asarray(th9)), jax.random.PRNGKey(7000), pd))
c.block_until_ready(); oc = np.asarray(c); ot = np.where(oc > 0, np.asarray(tt2), 0.)
t = clk('data_sim FIRST call (compile+run)', t)
ocf, otf, POSf = jnp.asarray(oc), jnp.asarray(ot), jnp.asarray(POS)
e0 = energy_scan_optimization(pred, jnp.zeros(3), jnp.arccos(1/jnp.sqrt(3)), jnp.pi/4, 0., POSf, otf, ocf, (ocf, otf), 1000., 700., 12, 0)['best_energy']
t = clk('seed S0 energy (pred compile+12 fwd)', t)
p1 = hierarchical_position_grid_search(POSf, otf, ocf, jnp.asarray(th9[1:4]), 0.0, 0.0, bounds, n_div=5, t0_n_div=5, levels=6, verbosity=0)
vtx, t0g = np.asarray(p1['best_position']), float(p1['best_t0']); t = clk('seed S1 vertex grid (geometric)', t)
c2 = hierarchical_direction_search_cone(pred, jnp.asarray(vtx), t0g, POSf, otf, ocf, (ocf, otf), e0, 3, 8, 90., 0.5, 0)
dirg = np.array([np.sin(c2['best_theta'])*np.cos(c2['best_phi']), np.sin(c2['best_theta'])*np.sin(c2['best_phi']), np.cos(c2['best_theta'])])
t = clk('seed S2 cone direction (fwd)', t)
seed = vec9_from_track(e0, vtx, dirg, t0=t0g)
res, H = fit_track(model, oc, ot, seed, nkeys=NKEYS, niters=NITERS, hist=True); t = clk(f'FIT {NITERS}it x {NKEYS}keys', t)
fv = np.linalg.norm((res-th9)[1:4])*100; print(f'  -> fit vtx {fv:.1f}cm  | TOTAL {time.time()-T:.0f}s', flush=True)
