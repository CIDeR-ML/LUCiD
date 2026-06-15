"""Recon gradient + Fisher + optimization sanity check on the merged engine.

For one GEANT4 event at the SK_like fiducial:
  (1) AD gradient (ReconModel.grad) vs common-random-number central FD of the loss, per param.
  (2) AD Fisher (jacfwd) vs FD Fisher, compared on the diagonal + as full matrices.
  (3) Optimization: fit_track from (a) the TRUTH seed (self-consistency / loss-min location)
      and (b) a perturbed seed (convergence), reporting loss-decrease + final errors.

Env: EV (event index), NPH, K, NKEYS, THRESH (ray_sampling
override applied via siren_params on the fly is NOT done here; we read the committed config).
"""
import os, sys, time
import numpy as np, jax, jax.numpy as jnp
_ROOT = '/sdf/group/neutrino/omara/LUCiD_unification'; sys.path.insert(0, _ROOT)
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.sources.event_io import read_photon_data_from_photonsim
from lucid.fitting import ReconModel, fit_track, vec9_dir, track_from_vec9, seed_vertex_time
from lucid.fitting.recon import vec9_from_track, SCALE9
from lucid.optimization.grid_search import get_detector_bounds

ROOT = '/sdf/group/neutrino/omara/LUCiD_recon/data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root'
GEOM = os.path.join(_ROOT, 'config/SK_like_geom_config.json')
PHYS = os.path.join(_ROOT, 'config/SK_like_physics_config.json')
K = int(os.environ.get('K', '8')); NBUF = 400_000; FIDR = FIDZ = 12.
NPH = int(os.environ.get('NPH', '250000')); NKEYS = int(os.environ.get('NKEYS', '6'))
EV = int(os.environ.get('EV', '0'))
GRID = dict(n_cap=80, n_angular=120, n_height=80)

det = generate_detector(GEOM); ND = len(det.all_points); POS = np.asarray(det.all_points)
dp = load_detector_params(PHYS, num_sensors=ND)
dp = dp._replace(response=dp.response._replace(tts=jnp.asarray(2.5)))
data_sim = setup_event_simulator(GEOM, NBUF, temperature=None, K=K, is_data=True, hit_mode='realistic',
                                 physics_config=PHYS, default_detector_params=dp, particle='muon',
                                 wavelength_mode=True, apply_smearing=False, **GRID)
pred = setup_event_simulator(GEOM, NPH, temperature=0.1, K=K, hit_mode='per_photon',
                             physics_config=PHYS, default_detector_params=True, particle='muon',
                             wavelength_mode=True, pos_grad_threshold=K, n_grad_iters=K, **GRID)
model = ReconModel(pred, ND, sigma=2.5, delta=1.0)
print(f'EV={EV} NPH={NPH} K={K} NKEYS={NKEYS}', flush=True)


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


# ---- build event + data ----
raw, vtx_true, dir_true = rand_tf(read_photon_data_from_photonsim(ROOT, EV), EV)
pol = float(np.arccos(np.clip(dir_true[2], -1, 1))); az = float(np.arctan2(dir_true[1], dir_true[0]))
E_true = float(raw['energy'])
th9 = np.array([E_true, vtx_true[0], vtx_true[1], vtx_true[2],
                np.sin(pol), np.cos(pol), np.sin(az), np.cos(az), 0.])
pd = pad_event(raw)
c, t = jax.lax.stop_gradient(data_sim(track_from_vec9(jnp.asarray(th9)), jax.random.PRNGKey(7000 + EV), pd))
oc = np.asarray(c); ot = np.where(oc > 0, np.asarray(t), 0.)
ocj, otj = jnp.asarray(oc), jnp.asarray(ot)
print(f'data: n_hit={int((oc>0).sum())} q_tot={oc.sum():.0f} E_true={E_true:.0f}', flush=True)
keys = [jax.random.PRNGKey(1000 + i) for i in range(NKEYS)]

# ===================== (1) AD gradient vs CRN central FD =====================
print('\n=== (1) GRADIENT: AD vs common-random-number central FD (at truth) ===', flush=True)
names = ['E', 'x', 'y', 'z', 'sinP', 'cosP', 'sinA', 'cosA', 't0']
t0 = time.time()
g_ad = np.mean([np.asarray(model.grad(th9, ocj, otj, k)) for k in keys], 0)
fdh = 0.4 * SCALE9
g_fd = np.zeros(9)
for j in range(9):
    ej = np.eye(9)[j]; h = fdh[j]
    fp = np.mean([float(model.loss(th9 + h * ej, ocj, otj, k)) for k in keys], 0)
    fm = np.mean([float(model.loss(th9 - h * ej, ocj, otj, k)) for k in keys], 0)
    g_fd[j] = (fp - fm) / (2 * h)
print(f'{"param":>5} {"AD":>13} {"FD":>13} {"AD/FD":>8}')
for j in range(9):
    r = g_ad[j] / g_fd[j] if abs(g_fd[j]) > 1e-6 else np.nan
    print(f'{names[j]:>5} {g_ad[j]:13.4f} {g_fd[j]:13.4f} {r:8.3f}')
# scaled-coordinate cosine (the metric the optimizer sees)
gs_ad = g_ad * SCALE9; gs_fd = g_fd * SCALE9
cos = float(gs_ad @ gs_fd / (np.linalg.norm(gs_ad) * np.linalg.norm(gs_fd) + 1e-30))
print(f'scaled-grad cosine(AD,FD) = {cos:.4f}   [{time.time()-t0:.0f}s]', flush=True)

# ===================== (2) AD Fisher vs FD Fisher =====================
print('\n=== (2) FISHER: AD (jacfwd) vs FD, at truth ===', flush=True)
t0 = time.time()
F_ad = model.fisher_ad(th9, ocj, otj, keys); F_fd = model.fisher(th9, ocj, otj, keys, fdh)
da, df = np.diag(F_ad), np.diag(F_fd)
print(f'{"param":>5} {"F_ad diag":>14} {"F_fd diag":>14} {"ad/fd":>8}')
for j in range(9):
    r = da[j] / df[j] if abs(df[j]) > 1e-12 else np.nan
    print(f'{names[j]:>5} {da[j]:14.4f} {df[j]:14.4f} {r:8.3f}')
Sm = SCALE9[:, None] * SCALE9[None, :]
ea = np.linalg.eigvalsh(F_ad * Sm); ef = np.linalg.eigvalsh(F_fd * Sm)
print(f'scaled eig(F_ad): min {ea.min():.3e} max {ea.max():.3e} cond {ea.max()/max(ea.min(),1e-30):.2e}')
print(f'scaled eig(F_fd): min {ef.min():.3e} max {ef.max():.3e} cond {ef.max()/max(ef.min(),1e-30):.2e}')
fro = np.linalg.norm((F_ad - F_fd) * Sm) / (np.linalg.norm(F_fd * Sm) + 1e-30)
print(f'rel ||F_ad-F_fd||_F (scaled) = {fro:.3f}   [{time.time()-t0:.0f}s]', flush=True)

# ===================== (3) optimization =====================
print('\n=== (3) OPTIMIZATION ===', flush=True)
truth_seed = vec9_from_track(E_true, vtx_true, dir_true, t0=0.0)
# perturbed seed: +40cm vertex, +5deg direction tilt, +120 MeV, +1.5 ns
pert = np.array(th9, float); pert[0] += 120.; pert[1] += 0.40
dd = dir_true.copy()
Rp = _rotax(np.array([1., 0., 0.]) if abs(dir_true[2]) < 0.99 else np.array([1., 0., 0.]), 5.0)
dd = (Rp @ dir_true); dd /= np.linalg.norm(dd)
pol2 = np.arccos(np.clip(dd[2], -1, 1)); az2 = np.arctan2(dd[1], dd[0])
pert[4], pert[5], pert[6], pert[7] = np.sin(pol2), np.cos(pol2), np.sin(az2), np.cos(az2)
pert[8] += 1.5
for tag, seed in [('TRUTH', truth_seed), ('PERTURBED', pert)]:
    t0 = time.time()
    res, H = fit_track(model, oc, ot, seed, nkeys=NKEYS, niters=int(os.environ.get('NITERS', '200')),
                       lr=8.0, fisher_mode='fd', hist=True)
    l0 = np.mean([float(model.loss(seed, ocj, otj, k)) for k in keys])
    lf = np.mean([float(model.loss(res, ocj, otj, k)) for k in keys])
    e_s = errs(seed, th9, dir_true); e_f = errs(res, th9, dir_true)
    print(f'[{tag}] seed vtx{e_s[0]:6.1f}cm dir{e_s[1]:4.2f} dE{e_s[2]:+6.0f} dt0{e_s[3]:+5.2f}  ->  '
          f'fit vtx{e_f[0]:6.1f}cm dir{e_f[1]:4.2f} dE{e_f[2]:+6.0f} dt0{e_f[3]:+5.2f} '
          f'| loss {l0:.1f}->{lf:.1f} best_iter={int(H["best_iter"])} '
          f'gnorm0={float(H["gnorm"][0]):.2e}->min{float(H["gnorm"][int(H["best_iter"])]):.2e} [{time.time()-t0:.0f}s]',
          flush=True)
print('\nDONE', flush=True)
