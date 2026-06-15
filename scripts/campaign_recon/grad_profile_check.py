"""Definitive gradient validation via noise-averaged 1D PROFILE slopes (not single-h CRN-FD).

The single-h central FD of a stochastic loss is noise-inflated (documented: raw CRN-FD geom
gradient unreliable). The correct reference is a many-key, multi-point local profile: scan each
smooth param over a small window, average the loss over a FIXED key set (CRN) at every point,
least-squares-fit the slope, and compare to the AD gradient (same keys). R² reports local
linearity. Angles are known-kinky and skipped here (covered in grad_opt_check).

Env: EV, NPH, K, NKEYS (keys averaged), NPTS, same event/truth construction as the worker.
"""
import os, sys, time
import numpy as np, jax, jax.numpy as jnp
_ROOT = '/sdf/group/neutrino/omara/LUCiD_unification'; sys.path.insert(0, _ROOT)
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.sources.event_io import read_photon_data_from_photonsim
from lucid.fitting import ReconModel, track_from_vec9

ROOT = '/sdf/group/neutrino/omara/LUCiD_recon/data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root'
GEOM = os.path.join(_ROOT, 'config/SK_like_geom_config.json')
PHYS = os.path.join(_ROOT, 'config/SK_like_physics_config.json')
K = int(os.environ.get('K', '8')); NBUF = 400_000; FIDR = FIDZ = 12.
NPH = int(os.environ.get('NPH', '150000')); NKEYS = int(os.environ.get('NKEYS', '16'))
NPTS = int(os.environ.get('NPTS', '7')); EV = int(os.environ.get('EV', '0'))
GRID = dict(n_cap=80, n_angular=120, n_height=80)

det = generate_detector(GEOM); ND = len(det.all_points)
dp = load_detector_params(PHYS, num_sensors=ND)._replace()
dp = dp._replace(response=dp.response._replace(tts=jnp.asarray(2.5)))
data_sim = setup_event_simulator(GEOM, NBUF, temperature=None, K=K, is_data=True, hit_mode='realistic',
                                 physics_config=PHYS, default_detector_params=dp, particle='muon',
                                 wavelength_mode=True, apply_smearing=False, **GRID)
pred = setup_event_simulator(GEOM, NPH, temperature=0.1, K=K, hit_mode='per_photon',
                             physics_config=PHYS, default_detector_params=True, particle='muon',
                             wavelength_mode=True, pos_grad_threshold=K, n_grad_iters=K, **GRID)
model = ReconModel(pred, ND, sigma=2.5, delta=1.0)


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


raw, vtx_true, dir_true = rand_tf(read_photon_data_from_photonsim(ROOT, EV), EV)
pol = float(np.arccos(np.clip(dir_true[2], -1, 1))); az = float(np.arctan2(dir_true[1], dir_true[0]))
E_true = float(raw['energy'])
th9 = np.array([E_true, vtx_true[0], vtx_true[1], vtx_true[2],
                np.sin(pol), np.cos(pol), np.sin(az), np.cos(az), 0.])
pd = pad_event(raw)
c, t = jax.lax.stop_gradient(data_sim(track_from_vec9(jnp.asarray(th9)), jax.random.PRNGKey(7000 + EV), pd))
oc = np.asarray(c); ot = np.where(oc > 0, np.asarray(t), 0.)
ocj, otj = jnp.asarray(oc), jnp.asarray(ot)
keys = [jax.random.PRNGKey(1000 + i) for i in range(NKEYS)]
print(f'EV={EV} NPH={NPH} K={K} NKEYS={NKEYS} NPTS={NPTS} | n_hit={int((oc>0).sum())} q_tot={oc.sum():.0f}', flush=True)

# smooth params: index, name, half-window
SMOOTH = [(0, 'E', 30.0), (1, 'x', 0.10), (2, 'y', 0.10), (3, 'z', 0.10), (8, 't0', 0.40)]
print(f'\n{"param":>5} {"AD":>12} {"profile":>12} {"AD/prof":>8} {"R^2":>7} {"win":>7}', flush=True)
for j, nm, w in SMOOTH:
    t0 = time.time()
    g_ad = np.mean([float(model.grad(th9, ocj, otj, k)[j]) for k in keys])
    xs = np.linspace(-w, w, NPTS); ys = np.zeros(NPTS)
    for i, dx in enumerate(xs):
        th = th9.copy(); th[j] += dx
        ys[i] = np.mean([float(model.loss(th, ocj, otj, k)) for k in keys])   # CRN: same keys every point
    A = np.vstack([xs, np.ones_like(xs)]).T
    (slope, b), *_ = np.linalg.lstsq(A, ys, rcond=None)
    yhat = A @ np.array([slope, b]); ss_res = ((ys - yhat) ** 2).sum()
    ss_tot = ((ys - ys.mean()) ** 2).sum(); r2 = 1 - ss_res / max(ss_tot, 1e-30)
    r = g_ad / slope if abs(slope) > 1e-6 else np.nan
    print(f'{nm:>5} {g_ad:12.3f} {slope:12.3f} {r:8.3f} {r2:7.4f} {w:7.2f}  [{time.time()-t0:.0f}s]', flush=True)
print('\nDONE', flush=True)
