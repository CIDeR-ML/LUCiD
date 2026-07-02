"""Config-driven single-track reconstruction pipeline (the two-start Fisher-GN).

Generalises ``scripts/campaign_recon/worker.py`` into a reusable, config-parameterised
pipeline for the tracking studies (nrays / energy / geom, muon / electron). For each event:

  1. load GEANT4/PhotonSim photons (``is_data=True`` — unit-weight integer photons),
  2. randomize the geometry into the fiducial volume (reproducible per (config, event)),
  3. generate realistic per-PMT (charge, first-arrival time) with per-photon TTS,
  4. run the data-driven seeder (energy scan -> charge-grid vertex ‖ time-multilateration
     vertex -> cone direction for each -> FUSED third seed, :func:`fuse_seeds`),
  5. run ``fit_track_multistart`` from all three starts, keeping the FULL per-iteration
     trajectory of each.

The exact truth is known analytically: the gun fires from the ORIGIN in +z, and we apply the
SAME random transform to it — so ``(vtx_true, dir_true)`` is that transform of ``(0, +z)`` (NOT
a PCA of the photon origins). See ``scripts/campaign_recon/RESULTS.md``.

``tot_n_scale`` defaults to 1.0 for both particles (no scalar charge norm); the model λ is
sampled over the PhotonSim Cherenkov emission band so QE applies to the true band.

Usage
-----
    from analysis.tracking.pipeline import TrackingPipeline, load_config
    cfg  = load_config('config_00.json')
    pipe = TrackingPipeline(cfg)          # builds detector + simulators ONCE
    rec  = pipe.reconstruct(ev=0)         # dict of truth / seeds / trajectories / errors
"""
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]      # analysis/tracking/pipeline.py -> LUCiD/

# 9-vector physical-parameter order the studies log: [E, x,y,z, sinθ,cosθ, sinφ,cosφ, t0].
# The user-facing tuple is (x, y, z, phi, theta, t0, E) — see vec9_to_phys.
PHYS_NAMES = ['x', 'y', 'z', 'phi', 'theta', 't0', 'E']


# --------------------------------------------------------------------------- config

DEFAULT_CONFIG = {
    'name': 'config',
    'particle': 'muon',                  # 'muon' | 'electron'
    'study': 'nrays',                    # 'nrays' | 'energy' | 'geom'
    'geom_config': 'config/SK_like_geom_config.json',
    'phys_config': 'config/SK_like_physics_config.json',
    'root_file': None,                   # PhotonSim ROOT with the primary photons
    'energy_nominal_MeV': 1000,          # nominal (true energy is read per-event from the ROOT)
    'n_events': 100,
    'event_start': 0,
    'n_rays': 250_000,                   # NPH: per-photon predictor photon budget (the "nrays" axis)
    'K': 8,
    'nbuf': 'auto',                      # data buffer: 'auto' = max photons/event measured from
                                         # the ROOT file (a fixed int overrides)
    'fidr': 12.0,                        # fiducial radius (m) for random vertex placement
    'fidz': 12.0,                        # fiducial half-height (m)
    'containment_margin': 0.95,          # accept a placement only if vertex + s_max(E)·dir is
                                         # inside this fraction of the detector (None disables)
    'tts': 2.5,                          # transit-time spread (ns) baked into the data sim
    'grid': {'n_cap': 80, 'n_angular': 120, 'n_height': 80},
    'cherenkov_band': [274.91, 673.83],  # model λ band (nm); None disables
    'gn': {'nkeys': 4, 'niters': 250, 'lr': 8.0, 'fisher_mode': 'fd',
           'sigma': 2.5, 'delta': 1.0, 'tot_n_scale': 1.0},
    # seeder knobs: one generous energy-scan window covering every study energy
    # (200..3000 MeV in 50 MeV steps -> seed-E quantization ~25 MeV), and the charge-grid t0
    # search grid (keep SYMMETRIC even for asymmetric true-t0 samples; must cover true_t0_range).
    'seed': {'energy_center': 1600.0, 'energy_delta': 1400.0, 'energy_steps': 57,
             't0_min': -20.0, 't0_max': 20.0, 't0_n_div': 9},
    # true t0 per event, drawn uniform in [lo, hi] (ns) and injected by shifting the photon
    # times; [0, 0] pins the historical fixed t0=0.
    'true_t0_range': [-15.0, 15.0],
    'placement_seed_base': 100003,
    'data_seed_base': 7000,
}


def _deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path):
    """Load a study config JSON, merged over :data:`DEFAULT_CONFIG`."""
    with open(path) as f:
        return _deep_merge(DEFAULT_CONFIG, json.load(f))


def _resolve(p):
    """Resolve a path against the repo root if it is not absolute."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


# --------------------------------------------------------------------------- geometry / truth

def load_smax_m(particle, material='water'):
    """Per-particle ``s_max(E_MeV) -> meters`` from the trained-SIREN metadata.

    Uses the canonical parametrization LUCiD ships with each trained model
    (``data/<material>/<particle>/siren_training/trained_model/photonsim_siren_metadata.json``,
    the same fit PhotonSim's ``smax_fit.csv`` carries) via :func:`lucid.siren.core.make_smax_fn`
    — NOT the legacy range parametrization in ``check_track_endpoint_in_detector``. Both muon
    and electron models carry the block.
    """
    import json as _json
    from lucid.siren.core import make_smax_fn
    meta = _json.loads((REPO_ROOT / 'data' / material / particle / 'siren_training' /
                        'trained_model' / 'photonsim_siren_metadata.json').read_text())
    fn = make_smax_fn(meta['smax'])
    return lambda E: float(fn(float(E))) / 1000.0          # mm -> m


def _rotax(u, deg):
    a = np.radians(deg); ca, sa = np.cos(a), np.sin(a); u = u / np.linalg.norm(u)
    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) * ca + sa * ux + (1 - ca) * np.outer(u, u)


def rand_tf(raw, ev, fidr, fidz, seed_base, *, track_len_m=None, bounds=None, margin=0.95,
            max_tries=200):
    """Randomize the gun geometry into the fiducial volume (reproducible per (config, event)).

    Draws an isotropic rotation (uniform-in-cosine polar tilt about an in-plane axis) and a
    uniform-in-volume shift inside the (fidr, fidz) cylinder. Applies it to the photons AND to
    the gun's exact (origin, +z), so the returned ``(vtx_true, dir_true)`` is exact truth.

    CONTAINMENT: if ``track_len_m`` (the s_max(E) projected track length) and ``bounds`` (from
    ``get_detector_bounds``) are given, placements whose endpoint ``vtx + track_len·dir`` falls
    outside ``margin`` × the detector are rejected and redrawn (same rng stream, so the accepted
    placement is still reproducible per (config, event)). After ``max_tries`` the last draw is
    kept with a warning — only reachable if (fidr, fidz, margin) leave almost no phase space.
    """
    from lucid.optimization.grid_search import is_point_inside_detector
    rng = np.random.default_rng(seed_base + ev)
    check = track_len_m is not None and bounds is not None and margin is not None
    O = np.asarray(raw['photon_origins']).astype(float)
    c = O.mean(0)                                               # rotation pivot: photon centroid (cm)
    for attempt in range(max_tries):
        beta = np.degrees(np.arccos(rng.uniform(-1, 1)))
        al = rng.uniform(0, 2 * np.pi); axis = np.array([-np.sin(al), np.cos(al), 0.])
        rr = fidr * np.sqrt(rng.uniform()); ph = rng.uniform(0, 2 * np.pi)
        sh = np.array([rr * np.cos(ph), rr * np.sin(ph), rng.uniform(-fidz, fidz)]) * 100.0  # cm
        R = _rotax(axis, beta)
        vtx = ((np.zeros(3) - c) @ R.T + c + sh) / 100.0       # the gun origin, transformed (m)
        d = np.array([0., 0., 1.]) @ R.T
        if not check or is_point_inside_detector(vtx + track_len_m * d, bounds, margin):
            break
    else:
        print(f"  WARNING ev{ev}: no contained placement in {max_tries} tries "
              f"(len={track_len_m:.1f} m, margin={margin}); keeping last draw", flush=True)
    raw = dict(raw)
    D = np.asarray(raw['photon_directions']).astype(float)
    raw['photon_origins'] = (O - c) @ R.T + c + sh
    raw['photon_directions'] = D @ R.T
    return raw, vtx, d


def truth9(vtx, d, energy, t0=0.0):
    """Physical (vertex m, unit direction, energy MeV, t0 ns) -> fitter 9-vector + numpy dir."""
    pol = float(np.arccos(np.clip(d[2], -1, 1))); az = float(np.arctan2(d[1], d[0]))
    return (np.array([float(energy), vtx[0], vtx[1], vtx[2],
                      np.sin(pol), np.cos(pol), np.sin(az), np.cos(az), float(t0)]),
            np.asarray(d))


def vec9_to_phys(v):
    """9-vector ``[E, x,y,z, sinθ,cosθ, sinφ,cosφ, t0]`` -> ``[x, y, z, phi, theta, t0, E]``."""
    v = np.asarray(v, float)
    st, ct, sp, cp = v[4], v[5], v[6], v[7]
    nt = np.hypot(st, ct) + 1e-12; npp = np.hypot(sp, cp) + 1e-12
    theta = np.arctan2(st / nt, ct / nt); phi = np.arctan2(sp / npp, cp / npp)
    return np.array([v[1], v[2], v[3], phi, theta, v[8], v[0]])


def traj_to_phys(traj):
    """Convert a ``(niters+1, 9)`` trajectory to ``(niters+1, 7)`` physical params."""
    return np.stack([vec9_to_phys(v) for v in np.asarray(traj)])


def _dir_from_vec9(v):
    """Unit direction (numpy, no JAX) from a 9-vector — mirrors lucid.fitting.vec9_dir."""
    v = np.asarray(v, float); st, ct, sp, cp = v[4], v[5], v[6], v[7]
    nt = np.hypot(st, ct); npp = np.hypot(sp, cp)
    st, ct, sp, cp = st / nt, ct / nt, sp / npp, cp / npp
    return np.array([st * cp, st * sp, ct])


# per-seed error vector order stored by the seed study (see analysis/tracking/seed_study.py).
SEED_ERR_NAMES = ['vtx_cm', 'vtx_trans_cm', 'vtx_long_cm', 'dir_deg', 'dE_MeV', 'dt0_ns']


def fuse_seeds(seedA, seedB, t0_mode='avg'):
    """Fuse the two seeds to get the best of both — all from seed-available quantities (no truth).

    Vertex: TRANSVERSE component from seedB (time-multilateration is transverse-excellent),
    LONGITUDINAL component from seedA (charge-grid is longitudinally unbiased), decomposed along
    **seedA's direction** ``dA``:  ``vtx = vtxB + ((vtxA - vtxB)·dA) dA``. Direction + energy
    from seedA (the better direction). t0 = mean of the two (their biases are opposite — seedA
    early, seedB late — so the average largely cancels): ``t0_mode`` ``'avg'`` | ``'A'`` | ``'B'``.
    """
    a = np.asarray(seedA, float); b = np.asarray(seedB, float)
    dA = _dir_from_vec9(a)
    vf = b[1:4] + float(np.dot(a[1:4] - b[1:4], dA)) * dA
    out = a.copy(); out[1:4] = vf                          # direction/energy/sincos inherited from A
    out[8] = {'avg': 0.5 * (a[8] + b[8]), 'A': a[8], 'B': b[8]}[t0_mode]
    return out


def seed_errors(seed, th9, d):
    """Per-component seed errors vs truth. Vertex split transverse/longitudinal along truth dir
    ``d`` (longitudinal is SIGNED: + = ahead of the true vertex along the track). Returns an
    array in :data:`SEED_ERR_NAMES` order."""
    seed = np.asarray(seed, float); th9 = np.asarray(th9, float)
    dv = seed[1:4] - th9[1:4]                              # meters
    long = float(np.dot(dv, d)); trans = float(np.linalg.norm(dv - long * d))
    ddeg = float(np.degrees(np.arccos(np.clip(_dir_from_vec9(seed) @ d, -1, 1))))
    return np.array([np.linalg.norm(dv) * 100, trans * 100, long * 100,
                     ddeg, float(seed[0] - th9[0]), float(seed[8] - th9[8])])


# --------------------------------------------------------------------------- pipeline

class TrackingPipeline:
    """Builds the detector + data/predictor simulators + ReconModel ONCE from a config,
    then reconstructs individual events. Reuse a single instance across a config's events so
    the jitted forwards compile once."""

    def __init__(self, config, verbose=True):
        # imports are local so importing this module (e.g. for load_config / vec9_to_phys in a
        # ROOT-free analysis context) does not pull in JAX / the simulator stack.
        from lucid.geometry import generate_detector
        from lucid.simulation import setup_event_simulator
        from lucid.detector_params import load_detector_params
        from lucid.fitting import ReconModel
        from lucid.optimization.grid_search import get_detector_bounds

        self.cfg = config
        geom = str(_resolve(config['geom_config'])); phys = str(_resolve(config['phys_config']))
        grid = config['grid']; K = config['K']; part = config['particle']

        self.det = generate_detector(geom)
        self.ND = len(self.det.all_points)
        self.POS = np.asarray(self.det.all_points)
        self.bounds = get_detector_bounds(self.det)
        # per-particle s_max(E) -> m, for the placement containment check
        self.smax_m = load_smax_m(part) if config.get('containment_margin') else None

        # nbuf 'auto': size the data buffer to the LARGEST event in the ROOT file, so no event
        # is silently truncated by _pad_event (461k photons/event at 2.1 GeV vs the old fixed
        # 400k). The resolved int is written back into the config for provenance.
        if config['nbuf'] in ('auto', None):
            import uproot
            with uproot.open(str(_resolve(config['root_file']))) as f:
                nmax = int(f['OpticalPhotons']['NOpticalPhotons'].array(library='np').max())
            config['nbuf'] = nmax
            if verbose:
                print(f"[pipeline] nbuf auto -> {nmax} (max photons/event in ROOT)", flush=True)

        # grid params are GEOMETRY-SPECIFIC (cylinder: n_cap/n_angular/n_height; sphere:
        # n_divisions; box: n_x/n_y/n_z). The defaults are the cylinder trio, so on a
        # non-cylinder drop them and let configure_grid() derive its own from the sensor
        # count — put the right keys in the config's 'grid' to override explicitly.
        grid = dict(grid or {})
        if self.bounds['type'] != 'cylinder':
            for k in ('n_cap', 'n_angular', 'n_height'):
                grid.pop(k, None)

        dp = load_detector_params(phys, num_sensors=self.ND)
        dp = dp._replace(response=dp.response._replace(tts=jnp.asarray(config['tts'])))
        self.data_sim = setup_event_simulator(
            geom, config['nbuf'], temperature=None, K=K, is_data=True, hit_mode='realistic',
            physics_config=phys, default_detector_params=dp, particle=part,
            wavelength_mode=True, apply_smearing=False, **grid)

        pred_kw = dict(temperature=0.1, K=K, hit_mode='per_photon', physics_config=phys,
                       default_detector_params=True, particle=part, wavelength_mode=True,
                       pos_grad_threshold=K, n_grad_iters=K, **grid)
        if config.get('cherenkov_band'):
            pred_kw['cherenkov_emission_band'] = tuple(config['cherenkov_band'])
        self.pred = setup_event_simulator(geom, config['n_rays'], **pred_kw)

        gn = config['gn']
        self.model = ReconModel(self.pred, self.ND, sigma=gn['sigma'], delta=gn['delta'],
                                tot_n_scale=gn['tot_n_scale'])
        if verbose:
            print(f"[pipeline] {part} | {self.ND} PMTs | n_rays={config['n_rays']} | "
                  f"geom={Path(geom).name} | root={Path(str(config['root_file'])).name}", flush=True)

    # -- one event ---------------------------------------------------------------------------

    def _pad_event(self, raw):
        NBUF = self.cfg['nbuf']
        n = int(np.asarray(raw['photon_origins']).shape[0])

        def tile(a):
            a = np.asarray(a); reps = int(np.ceil(NBUF / n))
            return jnp.asarray(np.tile(a, (reps,) + (1,) * (a.ndim - 1))[:NBUF])

        pd = {'photon_origins': tile(raw['photon_origins']),
              'photon_directions': tile(raw['photon_directions']),
              'photon_times': tile(raw['photon_times']), 'N': jnp.asarray(n),
              'apply_rotation': False, 'rotation_axis': jnp.array([1., 0., 0.]),
              'rotation_angle': jnp.array(0.), 'apply_translation': False,
              'translation_vector': jnp.zeros(3)}
        if 'wavelengths' in raw:
            pd['wavelengths'] = tile(raw['wavelengths'])
        return pd

    def _prepare_event(self, ev):
        """Load, place in the FV, simulate realistic data, and build BOTH seeds — NO Gauss-Newton.

        Shared by :meth:`reconstruct` and :meth:`seed_event`. Returns a dict with the truth
        9-vector + dir, the observables ``(oc, ot)``, both seed 9-vectors, and provenance.
        """
        from lucid.sources.event_io import read_photon_data_from_photonsim
        from lucid.fitting import track_from_vec9, seed_vertex_time
        from lucid.fitting.recon import vec9_from_track
        from lucid.optimization.grid_search import hierarchical_position_grid_search
        from lucid.optimization.utils.functions import (hierarchical_direction_search_cone,
                                                         energy_scan_optimization)
        cfg = self.cfg; POS = self.POS
        t_start = time.time()

        raw = read_photon_data_from_photonsim(str(_resolve(cfg['root_file'])), ev)
        track_len = self.smax_m(raw['energy']) if self.smax_m is not None else None
        raw, vtx_true, dir_true = rand_tf(
            raw, ev, cfg['fidr'], cfg['fidz'], cfg['placement_seed_base'],
            track_len_m=track_len, bounds=self.bounds,
            margin=cfg.get('containment_margin'))
        # true t0: uniform in true_t0_range, injected by SHIFTING the photon emission times so
        # the realistic data actually moves (the is_data sim takes times from the photons, not
        # the track). Own rng stream (base+7919) so placement draws stay unchanged vs t0=[0,0].
        lo, hi = cfg['true_t0_range']
        t0_true = float(np.random.default_rng(cfg['placement_seed_base'] + 7919 + ev).uniform(lo, hi)) \
            if hi > lo else float(lo)
        if t0_true != 0.0:
            raw = dict(raw)
            raw['photon_times'] = np.asarray(raw['photon_times'], float) + t0_true
        th9, d = truth9(vtx_true, dir_true, raw['energy'], t0=t0_true)
        pd = self._pad_event(raw)

        c, t = jax.lax.stop_gradient(self.data_sim(
            track_from_vec9(jnp.asarray(th9)),
            jax.random.PRNGKey(cfg['data_seed_base'] + ev), pd))
        oc = np.asarray(c); ot = np.where(oc > 0, np.asarray(t), 0.)
        ocf, otf, POSf = jnp.asarray(oc), jnp.asarray(ot), jnp.asarray(POS)

        # --- seeds: shared energy scan, two complementary vertices, cone direction each ------
        sd = cfg['seed']
        e0 = energy_scan_optimization(self.pred, jnp.zeros(3), jnp.arccos(1 / jnp.sqrt(3)),
                                      jnp.pi / 4, 0., POSf, otf, ocf, (ocf, otf),
                                      sd['energy_center'], sd['energy_delta'],
                                      sd['energy_steps'], 0)['best_energy']

        def make_seed(vtx, t0g):
            c2 = hierarchical_direction_search_cone(self.pred, jnp.asarray(vtx), t0g, POSf, otf,
                                                    ocf, (ocf, otf), e0, 3, 8, 90., 0.5, 0)
            dg = np.array([np.sin(c2['best_theta']) * np.cos(c2['best_phi']),
                           np.sin(c2['best_theta']) * np.sin(c2['best_phi']),
                           np.cos(c2['best_theta'])])
            return vec9_from_track(e0, np.asarray(vtx), dg, t0=t0g)

        p1 = hierarchical_position_grid_search(POSf, otf, ocf, jnp.zeros(3), 0.0, 0.0,
                                               self.bounds, n_div=5, t0_n_div=sd['t0_n_div'],
                                               levels=6, t0_min=sd['t0_min'], t0_max=sd['t0_max'],
                                               verbosity=0)
        seedA = make_seed(np.asarray(p1['best_position']), float(p1['best_t0']))   # charge-grid
        seedB = make_seed(*seed_vertex_time(POS, oc, ot))                          # time-multilateration

        return dict(ev=int(ev), energy_true=float(raw['energy']), th9=th9, d=d,
                    oc=oc, ot=ot, seedA=np.asarray(seedA), seedB=np.asarray(seedB),
                    n_hit=int((oc > 0).sum()), q_tot=float(oc.sum()),
                    seed_seconds=float(time.time() - t_start))

    def seed_event(self, ev, sel_margin=0.01):
        """Seed-only evaluation (no GN): both seeds vs truth + the data-loss selector.

        Returns per-component errors for seedA / seedB (:data:`SEED_ERR_NAMES` order), the data
        loss at each seed, and the loss-based pick — plain argmin and the margin-gated rule
        ``fit_track_multistart`` uses (prefer seedA unless seedB beats it by ``sel_margin``·|loss|).
        """
        gn = self.cfg['gn']
        P = self._prepare_event(ev)
        seedF = fuse_seeds(P['seedA'], P['seedB'])                    # fused (transverse-B + long-A, t0 avg)
        keys = [jax.random.PRNGKey(s) for s in range(gn['nkeys'])]   # match fit_track_multistart seed=0

        def dloss(seed):
            return float(np.mean([float(self.model.loss(seed, P['oc'], P['ot'], k)) for k in keys]))
        lossA, lossB, lossF = dloss(P['seedA']), dloss(P['seedB']), dloss(seedF)

        # margin-gated pick, prefer seedA; 2-way (A vs B, as fit_track_multistart today) and
        # 3-way (A vs B vs fused) — does adding the fused start change the selection?
        thr = lossA - sel_margin * abs(lossA)
        pick_gated = 1 if lossB < thr else 0
        losses3 = [lossA, lossB, lossF]
        cand = [i for i in (1, 2) if losses3[i] < thr]
        pick3 = min(cand, key=lambda i: losses3[i]) if cand else 0   # 0=A, 1=B, 2=fused
        return dict(
            ev=P['ev'], energy_true=P['energy_true'],
            truth_vec9=P['th9'], truth_phys=vec9_to_phys(P['th9']), tdir=P['d'],
            seedA_vec9=P['seedA'], seedB_vec9=P['seedB'], seedF_vec9=seedF,
            seedA_phys=vec9_to_phys(P['seedA']), seedB_phys=vec9_to_phys(P['seedB']),
            seedF_phys=vec9_to_phys(seedF),
            seedA_err=seed_errors(P['seedA'], P['th9'], P['d']),
            seedB_err=seed_errors(P['seedB'], P['th9'], P['d']),
            seedF_err=seed_errors(seedF, P['th9'], P['d']),
            lossA=lossA, lossB=lossB, lossF=lossF,
            loss_pick=(0 if lossA <= lossB else 1), loss_pick_gated=pick_gated, loss_pick3=pick3,
            n_hit=P['n_hit'], q_tot=P['q_tot'], seconds=P['seed_seconds'])

    def reconstruct(self, ev):
        """Reconstruct one PhotonSim event (seeds + three-start Fisher-GN). Returns numpy dict.

        Starts: seedA (charge-grid), seedB (time-multilateration), seedF (their FUSION —
        transverse vertex from B, longitudinal from A along A's direction, averaged t0).
        The fused seed was validated across geometry / energy / t0-range / particle / JUNO-sphere
        (50-event seed studies): best-or-equal vertex in every regime, t0 RMS ~7 ns -> ~2 ns, and
        the post-GN margin-gated loss pick (prefer=A) rescues the rare fusion failures (JUNO).
        """
        from lucid.fitting import fit_track_multistart, vec9_dir
        gn = self.cfg['gn']
        P = self._prepare_event(ev)
        th9, d, oc, ot = P['th9'], P['d'], P['oc'], P['ot']
        seedA, seedB = P['seedA'], P['seedB']
        seedF = fuse_seeds(seedA, seedB)
        t_start = time.time()

        # --- three-start Fisher-GN fit; keep the lowest-loss basin, retain all trajectories ---
        res, MS = fit_track_multistart(self.model, oc, ot, [seedA, seedB, seedF],
                                       nkeys=gn['nkeys'], niters=gn['niters'], lr=gn['lr'],
                                       fisher_mode=gn['fisher_mode'])
        HA = MS['per_seed'][0][1]; HB = MS['per_seed'][1][1]; HF = MS['per_seed'][2][1]
        H = MS['per_seed'][MS['which']][1]

        def _errs(x):
            dv = float(np.linalg.norm((np.asarray(x) - th9)[1:4]) * 100)          # vertex cm
            dd = float(np.degrees(np.arccos(np.clip(vec9_dir(x) @ d, -1, 1))))     # direction deg
            return np.array([dv, dd, float(x[0] - th9[0]), float(x[8] - th9[8])])  # +dE MeV, +dt0 ns

        return dict(
            ev=int(ev), energy_true=P['energy_true'],
            truth_vec9=th9, truth_phys=vec9_to_phys(th9), tdir=d,
            seedA_vec9=np.asarray(seedA), seedB_vec9=np.asarray(seedB),
            seedF_vec9=np.asarray(seedF),
            seedA_phys=vec9_to_phys(seedA), seedB_phys=vec9_to_phys(seedB),
            seedF_phys=vec9_to_phys(seedF),
            fit_vec9=np.asarray(res), fit_phys=vec9_to_phys(res),
            fitA_vec9=np.asarray(MS['per_seed'][0][0]), fitB_vec9=np.asarray(MS['per_seed'][1][0]),
            fitF_vec9=np.asarray(MS['per_seed'][2][0]),
            trajA=HA['traj'], trajB=HB['traj'], trajF=HF['traj'], traj_win=H['traj'],
            traj_win_phys=traj_to_phys(H['traj']),
            gnormA=HA['gnorm'], gnormB=HB['gnorm'], gnormF=HF['gnorm'], gnorm_win=H['gnorm'],
            best_iterA=int(HA['best_iter']), best_iterB=int(HB['best_iter']),
            best_iterF=int(HF['best_iter']),
            best_iter_win=int(H['best_iter']), which=int(MS['which']),
            losses=np.asarray(MS['losses']),
            seedA_err=_errs(seedA), seedB_err=_errs(seedB), seedF_err=_errs(seedF),
            fit_err=_errs(res),
            n_hit=P['n_hit'], q_tot=P['q_tot'],
            seconds=P['seed_seconds'] + float(time.time() - t_start))
