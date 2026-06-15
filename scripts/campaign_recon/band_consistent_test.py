"""Band-consistent recon test: fit the NEW GEANT4 ROOT (carries true PhotonWavelength) so the
data path QE-weights by TRUE lambda. Expectation from the wavelength analysis: the data q_tot
matches the model q_tot with NO cherenkov_photon_norm (the 1.66x on the old ROOT was a
data-path artifact — undetectable blue/red photons laundered into the detection band because
the old ROOT lacked per-photon wavelengths).

Reports, per event: data q_tot vs model q_tot@truth (ratio should be ~1.0, NOT 1.66), then a
truth-seed fit to measure the residual forward bias under the corrected (threshold) emitter.

Env: NEWROOT, EVENTS, NPH, K, NKEYS, NITERS, THRESH (emitter ray_sampling override), CHERENKOV_NORM.
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

NEWROOT = os.environ.get('NEWROOT',
    '/sdf/group/neutrino/omara/LUCiD_dlcheck/data/water/muon/1000MeV_100events.root')
GEOM = os.path.join(_ROOT, 'config/SK_like_geom_config.json')
PHYS = os.path.join(_ROOT, 'config/SK_like_physics_config.json')
K = int(os.environ.get('K', '8')); NBUF = 400_000; FIDR = FIDZ = 12.
NPH = int(os.environ.get('NPH', '150000')); NKEYS = int(os.environ.get('NKEYS', '8'))
NITERS = int(os.environ.get('NITERS', '200'))
THRESH = os.environ.get('THRESH', '0.001')
_cb = os.environ.get('CHER_BAND', '274.91,673.83')   # PhotonSim band [4.51,1.84] eV; sample model λ here
CHER_BAND = tuple(float(x) for x in _cb.split(',')) if _cb else None
EVENTS = [int(x) for x in os.environ.get('EVENTS', '0,1,2,3,4').split(',')]
GRID = dict(n_cap=80, n_angular=120, n_height=80)


def read_new_root(path, ev):
    """Loader for the OpticalPhotonsRaw (chunked) schema. Returns the same contract as
    read_photon_data_from_photonsim: cm positions, directions, times(ns), energy(MeV),
    wavelengths(nm). Chunks for one event are contiguous."""
    f = uproot.open(path)
    raw = f['OpticalPhotonsRaw']
    eid = raw['EventID'].array(library='np')
    idx = np.where(eid == ev)[0]
    lo, hi = int(idx.min()), int(idx.max()) + 1
    br = ['PhotonPosX', 'PhotonPosY', 'PhotonPosZ', 'PhotonDirX', 'PhotonDirY', 'PhotonDirZ',
          'PhotonTime', 'PhotonWavelength']
    d = raw.arrays(br, entry_start=lo, entry_stop=hi, library='np')
    cat = lambda b: np.concatenate([np.asarray(x, dtype=np.float64) for x in d[b]])
    pos = np.column_stack((cat('PhotonPosX') / 10.0, cat('PhotonPosY') / 10.0, cat('PhotonPosZ') / 10.0))  # mm->cm
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
                                 physics_config=PHYS, default_detector_params=dp, particle='muon',
                                 wavelength_mode=True, apply_smearing=False, **GRID)
pred = setup_event_simulator(GEOM, NPH, temperature=0.1, K=K, hit_mode='per_photon',
                             physics_config=PHYS, default_detector_params=True, particle='muon',
                             wavelength_mode=True, pos_grad_threshold=K, n_grad_iters=K,
                             cherenkov_emission_band=CHER_BAND, **GRID)
model = ReconModel(pred, ND, sigma=2.5, delta=1.0)
print(f'BAND-CONSISTENT TEST | NEWROOT (true wavelengths) | THRESH={THRESH or "default"} '
      f'CHER_BAND={CHER_BAND} NPH={NPH} K={K} EVENTS={EVENTS}', flush=True)


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


ratios = []; biases = []
for ev in EVENTS:
    t0 = time.time()
    raw, vtx_true, dir_true = rand_tf(read_new_root(NEWROOT, ev), ev)
    pol = float(np.arccos(np.clip(dir_true[2], -1, 1))); az = float(np.arctan2(dir_true[1], dir_true[0]))
    E_true = float(raw['energy'])
    th9 = np.array([E_true, vtx_true[0], vtx_true[1], vtx_true[2],
                    np.sin(pol), np.cos(pol), np.sin(az), np.cos(az), 0.])
    pd = pad_event(raw)
    c, t = jax.lax.stop_gradient(data_sim(track_from_vec9(jnp.asarray(th9)), jax.random.PRNGKey(7000 + ev), pd))
    oc = np.asarray(c); ot = np.where(oc > 0, np.asarray(t), 0.)
    data_q = float(oc.sum())
    # model q_tot at truth (no norm), averaged over a few keys
    mq = np.mean([float(np.asarray(model.perpmt(th9, jnp.asarray(oc), jnp.asarray(ot), jax.random.PRNGKey(50 + i))[0]).sum())
                  for i in range(4)])
    ratio = data_q / mq; ratios.append(ratio)
    # truth-seed fit
    truth_seed = vec9_from_track(E_true, vtx_true, dir_true, t0=0.0)
    rT = fit_track(model, oc, ot, truth_seed, nkeys=NKEYS, niters=NITERS, lr=8.0, fisher_mode='fd')
    eT = errs(rT, th9, dir_true); biases.append(eT)
    print(f'ev{ev:03d} n_hit{int((oc>0).sum()):5d} data_q{data_q:7.0f} model_q{mq:7.0f} RATIO {ratio:5.3f} '
          f'| TRUTHseed-> vtx{eT[0]:6.1f}cm dir{eT[1]:4.2f} dE{eT[2]:+6.0f}({100*eT[2]/E_true:+.1f}%) dt0{eT[3]:+5.2f} [{time.time()-t0:.0f}s]', flush=True)

R = np.array(ratios); B = np.array(biases)
print(f'\n=== SUMMARY (NEW GEANT4, true wavelengths, NO norm, threshold={THRESH}) ===', flush=True)
print(f'data/model q_tot ratio: median {np.median(R):.3f} mean {R.mean():.3f}  (1.66 = old-ROOT artifact; ~1.0 = resolved)', flush=True)
print(f'TRUTH-seed bias: vtx median {np.median(B[:,0]):.1f}cm | dir {np.median(B[:,1]):.2f} | '
      f'dE {np.median(B[:,2]):+.0f}MeV | dt0 {np.median(B[:,3]):+.2f}ns', flush=True)
print('DONE', flush=True)
