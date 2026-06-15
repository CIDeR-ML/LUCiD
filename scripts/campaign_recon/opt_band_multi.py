"""Band-consistent truth-seed optimization over many events, any particle/energy.

Uses the new ROOT (OpticalPhotonsRaw + PhotonWavelength) so QE applies to true λ, the
cherenkov_emission_band fix (no scalar norm), and threshold=0.001. Reports per-event q_tot
ratio + truth-seed bias, and a summary over many events.

Env: PARTICLE (muon|electron), ENERGY (500|1000|1500), EVENTS, NPH, K, NKEYS, NITERS,
THRESH, CHER_BAND (lo,hi). NEWROOT auto-resolved from PARTICLE/ENERGY under LUCiD_dlcheck.
"""
import os, sys, time
import numpy as np, jax, jax.numpy as jnp
_ROOT = '/sdf/group/neutrino/omara/LUCiD_unification'; sys.path.insert(0, _ROOT)
import uproot
from lucid.geometry import generate_detector
import lucid.simulation.simulator as SIM
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.fitting import ReconModel, fit_track, vec9_dir, track_from_vec9
from lucid.fitting.recon import vec9_from_track

PARTICLE = os.environ.get('PARTICLE', 'muon')
ENERGY = int(os.environ.get('ENERGY', '1000'))
NEWROOT = os.environ.get('NEWROOT',
    f'/sdf/group/neutrino/omara/LUCiD_dlcheck/data/water/{PARTICLE}/{ENERGY}MeV_100events.root')
GEOM = os.path.join(_ROOT, 'config/SK_like_geom_config.json')
PHYS = os.path.join(_ROOT, 'config/SK_like_physics_config.json')
K = int(os.environ.get('K', '8')); NBUF = 600_000; FIDR = FIDZ = 12.
NPH = int(os.environ.get('NPH', '150000')); NKEYS = int(os.environ.get('NKEYS', '8'))
NITERS = int(os.environ.get('NITERS', '200'))
THRESH = os.environ.get('THRESH', '0.001')
_cb = os.environ.get('CHER_BAND', '274.91,673.83')   # PhotonSim optical cutoff [4.51,1.84] eV
CHER_BAND = tuple(float(x) for x in _cb.split(',')) if _cb else None
EVENTS = [int(x) for x in os.environ.get('EVENTS', ','.join(str(i) for i in range(20))).split(',')]
GRID = dict(n_cap=80, n_angular=120, n_height=80)


def read_new_root(path, ev):
    f = uproot.open(path); raw = f['OpticalPhotonsRaw']
    eid = raw['EventID'].array(library='np'); idx = np.where(eid == ev)[0]
    lo, hi = int(idx.min()), int(idx.max()) + 1
    br = ['PhotonPosX', 'PhotonPosY', 'PhotonPosZ', 'PhotonDirX', 'PhotonDirY', 'PhotonDirZ',
          'PhotonTime', 'PhotonWavelength']
    d = raw.arrays(br, entry_start=lo, entry_stop=hi, library='np')
    cat = lambda b: np.concatenate([np.asarray(x, dtype=np.float64) for x in d[b]])
    pos = np.column_stack((cat('PhotonPosX') / 10.0, cat('PhotonPosY') / 10.0, cat('PhotonPosZ') / 10.0))
    dr = np.column_stack((cat('PhotonDirX'), cat('PhotonDirY'), cat('PhotonDirZ')))
    energy = float(f['OpticalPhotons']['PrimaryEnergy'].array(library='np')[ev])
    return {'photon_origins': jnp.asarray(pos), 'photon_directions': jnp.asarray(dr),
            'photon_times': jnp.asarray(cat('PhotonTime')), 'energy': energy,
            'wavelengths': jnp.asarray(cat('PhotonWavelength'))}


if THRESH != '':
    _orig = SIM.unpack_siren_params
    def _patched(p, m):
        c = dict(_orig(p, m)); c['ray_sampling'] = {**c.get('ray_sampling', {}), 'threshold': float(THRESH)}; return c
    SIM.unpack_siren_params = _patched

det = generate_detector(GEOM); ND = len(det.all_points)
dp = load_detector_params(PHYS, num_sensors=ND)
dp = dp._replace(response=dp.response._replace(tts=jnp.asarray(2.5)))
data_sim = setup_event_simulator(GEOM, NBUF, temperature=None, K=K, is_data=True, hit_mode='realistic',
                                 physics_config=PHYS, default_detector_params=dp, particle=PARTICLE,
                                 wavelength_mode=True, apply_smearing=False, **GRID)
pred = setup_event_simulator(GEOM, NPH, temperature=0.1, K=K, hit_mode='per_photon',
                             physics_config=PHYS, default_detector_params=True, particle=PARTICLE,
                             wavelength_mode=True, pos_grad_threshold=K, n_grad_iters=K,
                             cherenkov_emission_band=CHER_BAND, **GRID)
model = ReconModel(pred, ND, sigma=2.5, delta=1.0)
print(f'OPT-BAND | {PARTICLE} {ENERGY}MeV | CHER_BAND={CHER_BAND} THRESH={THRESH} NPH={NPH} '
      f'K={K} NKEYS={NKEYS} NITERS={NITERS} | {len(EVENTS)} events', flush=True)


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
          'apply_translation': False, 'translation_vector': jnp.zeros(3),
          'wavelengths': tile(raw['wavelengths'])}
    return pd


def errs(x, th9, d):
    dv = np.linalg.norm((x - th9)[1:4]) * 100
    dd = float(np.degrees(np.arccos(np.clip(vec9_dir(x) @ d, -1, 1))))
    return dv, dd, float(x[0] - th9[0]), float(x[8] - th9[8])


OUT = os.environ.get('OUT', os.path.join(_ROOT, 'scripts/campaign_recon/plots'))
os.makedirs(OUT, exist_ok=True)
ratios = []; biases = []; evdone = []; trajs = []
for ev in EVENTS:
    t0 = time.time()
    try:
        raw, vtx_true, dir_true = rand_tf(read_new_root(NEWROOT, ev), ev)
        pol = float(np.arccos(np.clip(dir_true[2], -1, 1))); az = float(np.arctan2(dir_true[1], dir_true[0]))
        E_true = float(raw['energy'])
        th9 = np.array([E_true, vtx_true[0], vtx_true[1], vtx_true[2],
                        np.sin(pol), np.cos(pol), np.sin(az), np.cos(az), 0.])
        pd = pad_event(raw)
        c, t = jax.lax.stop_gradient(data_sim(track_from_vec9(jnp.asarray(th9)), jax.random.PRNGKey(7000 + ev), pd))
        oc = np.asarray(c); ot = np.where(oc > 0, np.asarray(t), 0.)
        data_q = float(oc.sum())
        mq = np.mean([float(np.asarray(model.perpmt(th9, jnp.asarray(oc), jnp.asarray(ot), jax.random.PRNGKey(50 + i))[0]).sum())
                      for i in range(4)])
        ratio = data_q / mq; ratios.append(ratio)
        truth_seed = vec9_from_track(E_true, vtx_true, dir_true, t0=0.0)
        rT, H = fit_track(model, oc, ot, truth_seed, nkeys=NKEYS, niters=NITERS, lr=8.0, fisher_mode='fd', hist=True)
        eT = errs(rT, th9, dir_true); biases.append(eT); evdone.append(ev)
        # per-iteration trajectory metrics (vtx-err cm, dir deg, dE, dt0) + scaled gnorm
        per_iter = np.array([errs(x, th9, dir_true) for x in H['traj']])   # (niters+1, 4)
        trajs.append({'err': per_iter, 'gnorm': np.asarray(H['gnorm']), 'best_iter': int(H['best_iter'])})
        print(f'ev{ev:03d} n_hit{int((oc>0).sum()):5d} q{data_q:7.0f} mq{mq:7.0f} R{ratio:5.3f} '
              f'| vtx{eT[0]:6.1f}cm dir{eT[1]:4.2f} dE{eT[2]:+6.0f}({100*eT[2]/E_true:+5.1f}%) dt0{eT[3]:+5.2f} [{time.time()-t0:.0f}s]', flush=True)
    except Exception as e:
        print(f'ev{ev:03d} FAILED: {type(e).__name__}: {e} [{time.time()-t0:.0f}s]', flush=True)

R = np.array(ratios); B = np.array(biases)
# save per-event metrics + per-iteration trajectories (rectangular: all share NITERS)
_tag = f'{PARTICLE}_{ENERGY}'
np.savez(os.path.join(OUT, f'optb_{_tag}.npz'),
         particle=PARTICLE, energy=ENERGY, events=np.array(evdone), ratio=R, bias=B,  # bias cols: vtx_cm,dir_deg,dE_MeV,dt0_ns
         traj_err=np.array([tr['err'] for tr in trajs]),       # (n_ev, niters+1, 4)
         traj_gnorm=np.array([tr['gnorm'] for tr in trajs]),   # (n_ev, niters+1)
         best_iter=np.array([tr['best_iter'] for tr in trajs]))
print(f'saved {OUT}/optb_{_tag}.npz', flush=True)
print(f'\n=== SUMMARY {PARTICLE} {ENERGY}MeV ({len(B)} ev, no norm, band={CHER_BAND}, thr={THRESH}) ===', flush=True)
print(f'q_tot ratio: median {np.median(R):.3f} mean {R.mean():.3f} std {R.std():.3f}', flush=True)
print(f'vtx cm : median {np.median(B[:,0]):.1f} mean {B[:,0].mean():.1f} RMS {np.sqrt((B[:,0]**2).mean()):.1f}', flush=True)
print(f'dir deg: median {np.median(B[:,1]):.2f} mean {B[:,1].mean():.2f}', flush=True)
print(f'dE  MeV: median {np.median(B[:,2]):+.0f} mean {B[:,2].mean():+.0f} ({100*B[:,2].mean()/ENERGY:+.1f}%)', flush=True)
print(f'dt0  ns: median {np.median(B[:,3]):+.2f} mean {B[:,3].mean():+.2f}', flush=True)
print('DONE', flush=True)
