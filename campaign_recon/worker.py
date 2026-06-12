"""Per-event seeded track reconstruction on GEANT4 data (the CLEAN truth path).

For each event: load GEANT4/PhotonSim photons (is_data=True — unit-weight integer photons, NOT
the SIREN importance-weighted emitter), randomize geometry into the fiducial volume, generate
realistic data with 2.5 ns per-photon TTS, run the data-driven 3-stage seeder, then the full
Fisher-GN fit_track keeping the FULL trajectory. Saves one npz per event.

Env: EVENT_START, EVENT_COUNT, OUT, CUDA_VISIBLE_DEVICES. Geometry SK_like; truth energy 1050 MeV.
"""
import os, sys, time
import numpy as np, jax, jax.numpy as jnp
_ROOT_DIR = '/sdf/group/neutrino/omara/LUCiD_unification'
sys.path.insert(0, _ROOT_DIR)
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import load_detector_params
from lucid.sources.event_io import read_photon_data_from_photonsim
from lucid.fitting import (ReconModel, fit_track_multistart, vec9_dir, track_from_vec9,
                           seed_vertex_time)
from lucid.fitting.recon import vec9_from_track
from lucid.optimization.grid_search import hierarchical_position_grid_search, get_detector_bounds
from lucid.optimization.utils.functions import (
    hierarchical_direction_search_cone, energy_scan_optimization)

ROOT = '/sdf/group/neutrino/omara/LUCiD_recon/data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root'
GEOM = os.path.join(_ROOT_DIR, 'config/SK_like_geom_config.json')
PHYS = os.path.join(_ROOT_DIR, 'config/SK_like_physics_config.json')
K, NBUF, FIDR, FIDZ = 8, 400_000, 12., 12.
GRID = dict(n_cap=80, n_angular=120, n_height=80)
E0 = int(os.environ.get('EVENT_START', '0')); EC = int(os.environ.get('EVENT_COUNT', '20'))
EVLIST = os.environ.get('EVENT_LIST', '')                          # comma-list overrides the range
EVENTS = [int(x) for x in EVLIST.split(',')] if EVLIST else list(range(E0, E0 + EC))
OUT = os.environ.get('OUT', os.path.join(_ROOT_DIR, 'campaign_recon/out')); os.makedirs(OUT, exist_ok=True)

det = generate_detector(GEOM); ND = len(det.all_points); POS = np.asarray(det.all_points)
bounds = get_detector_bounds(det)
dp_data = load_detector_params(PHYS, num_sensors=ND)
dp_data = dp_data._replace(response=dp_data.response._replace(tts=jnp.asarray(2.5)))
data_sim = setup_event_simulator(GEOM, NBUF, temperature=None, K=K, is_data=True, hit_mode='realistic',
                                 physics_config=PHYS, default_detector_params=dp_data, particle='muon',
                                 wavelength_mode=True, apply_smearing=False, **GRID)
pred = setup_event_simulator(GEOM, 250_000, temperature=0.1, K=K, hit_mode='per_photon',
                             physics_config=PHYS, default_detector_params=True, particle='muon',
                             wavelength_mode=True, pos_grad_threshold=K, n_grad_iters=K, **GRID)
model = ReconModel(pred, ND, sigma=2.5, delta=1.0)


def _rotax(u, deg):
    a = np.radians(deg); ca, sa = np.cos(a), np.sin(a); u = u / np.linalg.norm(u)
    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) * ca + sa * ux + (1 - ca) * np.outer(u, u)


def rand_tf(raw, ev):                                    # randomize geometry into the fiducial volume
    rng = np.random.default_rng(100003 + ev); beta = np.degrees(np.arccos(rng.uniform(-1, 1)))
    al = rng.uniform(0, 2 * np.pi); axis = np.array([-np.sin(al), np.cos(al), 0.])
    rr = FIDR * np.sqrt(rng.uniform()); ph = rng.uniform(0, 2 * np.pi)
    sh = np.array([rr * np.cos(ph), rr * np.sin(ph), rng.uniform(-FIDZ, FIDZ)]) * 100.0
    raw = dict(raw); O = np.asarray(raw['photon_origins']).astype(float)
    D = np.asarray(raw['photon_directions']).astype(float); R = _rotax(axis, beta); c = O.mean(0)
    raw['photon_origins'] = (O - c) @ R.T + c + sh; raw['photon_directions'] = D @ R.T
    return raw


def derive_truth(raw):                                   # truth vertex/dir via PCA of photon origins
    P = np.asarray(raw['photon_origins']) / 100.0; tt = np.asarray(raw['photon_times']); C = P - P.mean(0)
    _, _, vt = np.linalg.svd(C, full_matrices=False); d = vt[0]; proj = C @ d
    if np.corrcoef(proj, tt)[0, 1] < 0: d = -d; proj = -proj
    vtx = P.mean(0) + proj.min() * d
    pol = float(np.arccos(np.clip(d[2], -1, 1))); az = float(np.arctan2(d[1], d[0]))
    return np.array([float(raw['energy']), vtx[0], vtx[1], vtx[2],
                     np.sin(pol), np.cos(pol), np.sin(az), np.cos(az), 0.]), d


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
    return [dv, dd, float(x[0] - th9[0]), float(x[8] - th9[8])]      # vtx cm, dir deg, dE MeV, dt0 ns


for ev in EVENTS:
    t0 = time.time()
    try:
        raw = rand_tf(read_photon_data_from_photonsim(ROOT, ev), ev)
        th9, d = derive_truth(raw); pd = pad_event(raw)
        c, t = jax.lax.stop_gradient(data_sim(track_from_vec9(jnp.asarray(th9)),
                                              jax.random.PRNGKey(7000 + ev), pd))
        oc = np.asarray(c); ot = np.where(oc > 0, np.asarray(t), 0.)
        # FULL fixed-shape (n_sensors,) arrays for the forward seed stages (energy/direction)
        # so the @jit'd losses compile ONCE (not once per event's variable n_hit).
        ocf, otf, POSf = jnp.asarray(oc), jnp.asarray(ot), jnp.asarray(POS)
        # --- TWO complementary data-driven seeds (energy shared); cone direction per vertex ---
        e0 = energy_scan_optimization(pred, jnp.zeros(3), jnp.arccos(1 / jnp.sqrt(3)), jnp.pi / 4, 0.,
                                      POSf, otf, ocf, (ocf, otf), 1000., 700., 12, 0)['best_energy']

        def make_seed(vtx, t0g):                               # vertex+t0 -> cone direction -> 9-vec
            c2 = hierarchical_direction_search_cone(pred, jnp.asarray(vtx), t0g, POSf, otf, ocf,
                                                    (ocf, otf), e0, 3, 8, 90., 0.5, 0)
            dg = np.array([np.sin(c2['best_theta']) * np.cos(c2['best_phi']),
                           np.sin(c2['best_theta']) * np.sin(c2['best_phi']), np.cos(c2['best_theta'])])
            return vec9_from_track(e0, np.asarray(vtx), dg, t0=t0g)
        # seed A: charge-grid (good longitudinally on most events; loses inward-pointing tracks)
        p1 = hierarchical_position_grid_search(POSf, otf, ocf, jnp.zeros(3), 0.0, 0.0,   # zeros = no truth
                                               bounds, n_div=5, t0_n_div=5, levels=6, verbosity=0)
        seedA = make_seed(np.asarray(p1['best_position']), float(p1['best_t0']))
        # seed B: TIME multilateration (transverse-perfect; forward-biased -> rescues inward tracks)
        seedB = make_seed(*seed_vertex_time(POS, oc, ot))
        # --- two-start Fisher-GN fit: keep the lower-loss basin ---
        res, MS = fit_track_multistart(model, oc, ot, [seedA, seedB], nkeys=4, niters=250)
        H = MS['per_seed'][MS['which']][1]
        seA, seB = errs(seedA, th9, d), errs(seedB, th9, d)
        fe = errs(res, th9, d); fA = errs(MS['per_seed'][0][0], th9, d); fB = errs(MS['per_seed'][1][0], th9, d)
        np.savez(os.path.join(OUT, f'ev{ev:03d}.npz'), truth=th9, tdir=d, seedA=seedA, seedB=seedB, fit=res,
                 traj=H['traj'], gnorm=H['gnorm'], best_iter=H['best_iter'], which=MS['which'],
                 losses=np.array(MS['losses']), seedA_err=np.array(seA), seedB_err=np.array(seB),
                 fitA_err=np.array(fA), fitB_err=np.array(fB), fit_err=np.array(fe),
                 n_hit=int((oc > 0).sum()), q_tot=float(oc.sum()))
        print(f'ev{ev:03d} seedA vtx{seA[0]:5.0f} B vtx{seB[0]:5.0f} | fitA{fA[0]:6.1f} fitB{fB[0]:6.1f} '
              f'-> WIN={"AB"[MS["which"]]} vtx{fe[0]:5.1f}cm dir{fe[1]:4.2f} E{fe[2]:+6.1f} t0{fe[3]:+5.2f} [{time.time()-t0:.0f}s]', flush=True)
    except Exception as e:
        print(f'ev{ev:03d} FAILED: {type(e).__name__}: {e} [{time.time()-t0:.0f}s]', flush=True)
