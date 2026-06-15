"""Multi-event TRUTH-SEED optimization: measure the forward-model bias (loss-minimum location)
under corrected vs uncorrected emitter config.

Goal: does the CORRECTED forward (cherenkov_photon_norm=1.66 yield + ray_sampling.threshold=0.001
longitudinal/ring) make the fit UNBIASED — i.e. does the loss minimum sit at truth across events?
Fitting from the truth seed isolates the loss-min location (the forward bias) from seeder/basin
effects. We also fit from one perturbed seed per event to confirm the basin still contains truth.

Env: EVENTS (comma list), NPH, K, NKEYS, NITERS, CHERENKOV_NORM, THRESH (ray_sampling override
via monkeypatch; '' = leave config default 0.05).
"""
import os, sys, time
import numpy as np, jax, jax.numpy as jnp
_ROOT = '/sdf/group/neutrino/omara/LUCiD_unification'; sys.path.insert(0, _ROOT)
from lucid.geometry import generate_detector
import lucid.simulation.simulator as SIM
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.sources.event_io import read_photon_data_from_photonsim
from lucid.fitting import ReconModel, fit_track, vec9_dir, track_from_vec9
from lucid.fitting.recon import vec9_from_track

ROOT = '/sdf/group/neutrino/omara/LUCiD_recon/data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root'
GEOM = os.path.join(_ROOT, 'config/SK_like_geom_config.json')
PHYS = os.path.join(_ROOT, 'config/SK_like_physics_config.json')
K = int(os.environ.get('K', '8')); NBUF = 400_000; FIDR = FIDZ = 12.
NPH = int(os.environ.get('NPH', '150000')); NKEYS = int(os.environ.get('NKEYS', '8'))
NITERS = int(os.environ.get('NITERS', '200'))
THRESH = os.environ.get('THRESH', '')   # '' -> default config (0.05); else override
EVENTS = [int(x) for x in os.environ.get('EVENTS', '0,1,2,3,4').split(',')]
GRID = dict(n_cap=80, n_angular=120, n_height=80)

# --- threshold override via monkeypatch on the name the simulator calls (no config commit) ---
if THRESH != '':
    _orig = SIM.unpack_siren_params
    def _patched(particle, material):
        cfg = dict(_orig(particle, material))
        cfg['ray_sampling'] = {**cfg.get('ray_sampling', {}), 'threshold': float(THRESH)}
        return cfg
    SIM.unpack_siren_params = _patched
    print(f'ray_sampling.threshold OVERRIDE -> {THRESH}', flush=True)

det = generate_detector(GEOM); ND = len(det.all_points)
dp = load_detector_params(PHYS, num_sensors=ND)
dp = dp._replace(response=dp.response._replace(tts=jnp.asarray(2.5)))
data_sim = setup_event_simulator(GEOM, NBUF, temperature=None, K=K, is_data=True, hit_mode='realistic',
                                 physics_config=PHYS, default_detector_params=dp, particle='muon',
                                 wavelength_mode=True, apply_smearing=False, **GRID)
pred = setup_event_simulator(GEOM, NPH, temperature=0.1, K=K, hit_mode='per_photon',
                             physics_config=PHYS, default_detector_params=True, particle='muon',
                             wavelength_mode=True, pos_grad_threshold=K, n_grad_iters=K, **GRID)
model = ReconModel(pred, ND, sigma=2.5, delta=1.0)
print(f'CONFIG: THRESH={THRESH or "0.05(default)"} NPH={NPH} K={K} NKEYS={NKEYS} '
      f'NITERS={NITERS} EVENTS={EVENTS}', flush=True)


def _rotax(u, deg):
    a = np.radians(deg); ca, sa = np.cos(a), np.sin(a); u = u / np.linalg.norm(u)
    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) * ca + sa * ux + (1 - ca) * np.outer(u, u)


def rand_tf(raw, ev):
    rng = np.random.default_rng(100003 + ev); beta = np.degrees(np.arccos(rng.uniform(-1, 1)))
    al = rng.uniform(0, 2 * np.pi); axis = np.array([-np.sin(al), np.cos(al), 0.])
    rr = FIDR * np.sqrt(rng.uniform()); ph = rng.uniform(0, 2 * np.pi)
    sh = np.array([rr * np.cos(ph), rr * np.sin(ph), rng.uniform(-FIDZ, FIDZ)]) * 100.0
    raw = dict(raw); O = np.asarray(raw['photon_origins']).astype(float)
    D = np.asarray(raw['photon_directions']).astype(float); R = _rotax(axis, beta); c = O.mean(0)
    raw['photon_origins'] = (O - c) @ R.T + c + sh; raw['photon_directions'] = D @ R.T
    vtx_true = ((np.zeros(3) - c) @ R.T + c + sh) / 100.0
    dir_true = np.array([0., 0., 1.]) @ R.T
    return raw, vtx_true, dir_true


def pad_event(raw):
    n = int(np.asarray(raw['photon_origins']).shape[0])
    def tile(a):
        a = np.asarray(a); reps = int(np.ceil(NBUF / n))
        return jnp.asarray(np.tile(a, (reps,) + (1,) * (a.ndim - 1))[:NBUF])
    pd = {'photon_origins': tile(raw['photon_origins']), 'photon_directions': tile(raw['photon_directions']),
          'photon_times': tile(raw['photon_times']), 'N': jnp.asarray(n), 'apply_rotation': False,
          'rotation_axis': jnp.array([1., 0., 0.]), 'rotation_angle': jnp.array(0.),
          'apply_translation': False, 'translation_vector': jnp.zeros(3)}
    if 'wavelengths' in raw: pd['wavelengths'] = tile(raw['wavelengths'])
    return pd


def errs(x, th9, d):
    dv = np.linalg.norm((x - th9)[1:4]) * 100
    dd = float(np.degrees(np.arccos(np.clip(vec9_dir(x) @ d, -1, 1))))
    return dv, dd, float(x[0] - th9[0]), float(x[8] - th9[8])


rows = []
for ev in EVENTS:
    t0 = time.time()
    raw, vtx_true, dir_true = rand_tf(read_photon_data_from_photonsim(ROOT, ev), ev)
    pol = float(np.arccos(np.clip(dir_true[2], -1, 1))); az = float(np.arctan2(dir_true[1], dir_true[0]))
    E_true = float(raw['energy'])
    th9 = np.array([E_true, vtx_true[0], vtx_true[1], vtx_true[2],
                    np.sin(pol), np.cos(pol), np.sin(az), np.cos(az), 0.])
    pd = pad_event(raw)
    c, t = jax.lax.stop_gradient(data_sim(track_from_vec9(jnp.asarray(th9)), jax.random.PRNGKey(7000 + ev), pd))
    oc = np.asarray(c); ot = np.where(oc > 0, np.asarray(t), 0.)
    truth_seed = vec9_from_track(E_true, vtx_true, dir_true, t0=0.0)
    # perturbed seed: +40cm vtx, +5deg dir, +120 MeV, +1.5 ns
    pert = th9.copy(); pert[0] += 120.; pert[1] += 0.40
    Rp = _rotax(np.array([1., 0., 0.]), 5.0); dd = Rp @ dir_true; dd /= np.linalg.norm(dd)
    p2 = np.arccos(np.clip(dd[2], -1, 1)); a2 = np.arctan2(dd[1], dd[0])
    pert[4:8] = [np.sin(p2), np.cos(p2), np.sin(a2), np.cos(a2)]; pert[8] += 1.5
    rT = fit_track(model, oc, ot, truth_seed, nkeys=NKEYS, niters=NITERS, lr=8.0, fisher_mode='fd')
    rP = fit_track(model, oc, ot, pert, nkeys=NKEYS, niters=NITERS, lr=8.0, fisher_mode='fd')
    eT = errs(rT, th9, dir_true); eP = errs(rP, th9, dir_true)
    rows.append((ev, eT, eP, oc.sum()))
    print(f'ev{ev:03d} q_tot{oc.sum():6.0f} | TRUTHseed-> vtx{eT[0]:6.1f}cm dir{eT[1]:4.2f} dE{eT[2]:+6.0f} dt0{eT[3]:+5.2f}'
          f' || PERTseed-> vtx{eP[0]:6.1f}cm dir{eP[1]:4.2f} dE{eP[2]:+6.0f} dt0{eP[3]:+5.2f} [{time.time()-t0:.0f}s]', flush=True)

# summary over events (truth-seed = forward-bias measurement)
A = np.array([r[1] for r in rows])
print(f'\n=== TRUTH-SEED summary over {len(rows)} ev (forward-bias / loss-min location) ===', flush=True)
print(f'vtx  cm: median {np.median(A[:,0]):.1f}  mean {A[:,0].mean():.1f}', flush=True)
print(f'dir deg: median {np.median(A[:,1]):.2f}  mean {A[:,1].mean():.2f}', flush=True)
print(f'dE  MeV: median {np.median(A[:,2]):+.0f}  mean {A[:,2].mean():+.0f}', flush=True)
print(f'dt0  ns: median {np.median(A[:,3]):+.2f}  mean {A[:,3].mean():+.2f}', flush=True)
print('DONE', flush=True)
