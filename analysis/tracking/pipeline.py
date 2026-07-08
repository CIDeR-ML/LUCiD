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
           'sigma': 2.5, 'delta': 1.0, 'tot_n_scale': 1.0,
           # energy_scale_mode (new-lib only): 'nphot' (DEFAULT since the paired 250k test:
           # physics identical to simtotal within 0.4 MeV p68 per event, 1.8x faster; single
           # pass, analytic nphot(E) scale, nphot_fn built from the emitter context) or
           # 'simtotal' (2nd forward pass, exact dA/dE).
           'energy_scale_mode': 'nphot'},
    # seeder knobs: one generous energy-scan window covering every study energy
    # (200..3000 MeV in 50 MeV steps -> seed-E quantization ~25 MeV), and the charge-grid t0
    # search grid (keep SYMMETRIC even for asymmetric true-t0 samples; must cover true_t0_range).
    'seed': {'energy_center': 1600.0, 'energy_delta': 1400.0, 'energy_steps': 57,
             't0_min': -20.0, 't0_max': 20.0, 't0_n_div': 9},
    # true t0 per event, drawn uniform in [lo, hi] (ns) and injected by shifting the photon
    # times; [0, 0] pins the historical fixed t0=0.
    'true_t0_range': [-15.0, 15.0],
    # fix the energy to its TRUE value: seeds get E_true (energy scan skipped) and the GN step
    # in E is exactly zero (gradient + Fisher row/col masked). Diagnostic mode.
    'fix_energy': False,
    # opposite diagnostic: pin ALL params at truth except E — single start from the truth
    # 9-vector, only the energy moves.
    'fix_geometry': False,
    # override the SIREN emitter's ray_sampling.threshold. LEGACY-ONLY since upstream a270a55:
    # the default seed_mode='importance' keeps the full emission domain and ignores threshold
    # (it only matters if ray_sampling.seed_mode is set back to 'uniform').
    'siren_threshold': None,
    # False (default): build all three seeds, pick ONE by the pre-GN margin-gated data loss
    # (prefer seedA unless another beats it by 1%) and run a single GN fit (~2.2x faster).
    # True: full three-start fit_track_multistart with post-GN selection.
    'multistart': False,
    # E-refit stage: after the main fit, freeze ALL params at the fit values and re-optimize E
    # alone for this many 1-D Gauss-Newton iterations on a DIFFERENT loss — the total-charge
    # match L = 0.5 (sum mu(E) - sum q_obs)^2, whose AD gradient is faithful and whose minimum
    # sits at the unbiased energy (grad_vs_fd_qtot). 0 disables.
    'e_refit_iters': 0,
    # force the single-track start: None (default) = pre-GN margin-gated loss pick among
    # A/B/F; 'A'|'B'|'F' = always track that seed (skips the pick's 3x nkeys loss evals).
    'force_seed': None,
    'placement_seed_base': 100003,
    # placement rng seed = base + event * stride. stride=1 is OUR historical scheme; stride=1000
    # matches the upstream sweep's pose_seed (lucid/fitting/sweep.py POSE_STRIDE, pose=0).
    # To reproduce the upstream truth treatment EXACTLY set, per config:
    #   {"placement_seed_stride": 1000, "containment_margin": null, "true_t0_range": [0, 0]}
    # and to come back to ours simply omit them (defaults: stride 1, containment 0.95,
    # t0 free in [-15, 15]).
    'placement_seed_stride': 1,
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


class _FixedParamsModel:
    """Freeze a subset of the 9 track params in the Fisher-GN without touching lucid.fitting.

    Delegates to a ReconModel but zeroes the gradient entries and Fisher rows/cols of the fixed
    indices. In fit_track's step ``A = Fs + lam·diag(Fs) + rI + 1e-9I`` the fixed rows reduce to
    ``[Aii, 0, ...]`` with ``gs[i]=0``, so ``du[i] = 0`` EXACTLY — fixed params stay at their
    seed values — and the min-||g|| readout scores the free parameters only.
    """

    def __init__(self, base, fixed_idx):
        self._m = base; self.ND = base.ND; self._idx = list(fixed_idx)
        # pass through so fit_track's trust='auto' sees the energy_from_scale flag (97c33ea)
        self.energy_from_scale = getattr(base, 'energy_from_scale', False)

    def loss(self, *a, **k): return self._m.loss(*a, **k)
    def perpmt(self, *a, **k): return self._m.perpmt(*a, **k)

    def grad(self, *a, **k):
        g = np.array(self._m.grad(*a, **k), float); g[self._idx] = 0.0; return g

    def fisher(self, *a, **k):
        F = np.array(self._m.fisher(*a, **k), float)
        F[self._idx, :] = 0.0; F[:, self._idx] = 0.0; return F

    def fisher_ad(self, *a, **k):
        F = np.array(self._m.fisher_ad(*a, **k), float)
        F[self._idx, :] = 0.0; F[:, self._idx] = 0.0; return F


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

        # siren_threshold: patch the ray_sampling block the simulator loads internally
        # (setup_event_simulator has no kwarg for it). Affects only the SIREN emitter (pred);
        # the is_data sim never samples from the SIREN.
        if config.get('siren_threshold') is not None:
            import lucid.simulation.simulator as _sim
            _orig_unpack = _sim.unpack_siren_params
            thr = float(config['siren_threshold'])

            def _patched(particle_type='muon', material='water'):
                d = _orig_unpack(particle_type, material)
                d['ray_sampling'] = dict(d['ray_sampling'], threshold=thr)
                return d
            _sim.unpack_siren_params = _patched
            if verbose:
                print(f"[pipeline] siren ray_sampling.threshold -> {thr}", flush=True)

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

        pred_kw = dict(temperature=config.get('pred_temperature', 0.1), K=K, hit_mode='per_photon', physics_config=phys,
                       default_detector_params=True, particle=part, wavelength_mode=True,
                       pos_grad_threshold=K, n_grad_iters=K, **grid)
        if config.get('cherenkov_band'):
            pred_kw['cherenkov_emission_band'] = tuple(config['cherenkov_band'])
        self.pred = setup_event_simulator(geom, config['n_rays'], **pred_kw)

        gn = config['gn']
        if gn.get('tts_walk_corr'):
            # CLOSURE-BIAS TEST: correct the observed first-arrival for the TTS
            # order-statistic early walk, tobs_corr = tobs - tts*E[min of n std normals]
            # (E[min_n] < 0, so the correction shifts tobs LATER, back to the geometric
            # reference). Wraps lucid.fitting.recon.first_arrival_window_nll BEFORE the
            # ReconModel is built so the model's jitted loss picks up the wrapper.
            import lucid.fitting.recon as _rec
            from scipy.special import erf as _serf
            _tts = float(config.get('tts', 2.5))
            _xs = np.linspace(-9, 9, 6001)
            _phi = np.exp(-_xs**2 / 2) / np.sqrt(2 * np.pi)
            _Phi = 0.5 * (1 + _serf(_xs / np.sqrt(2)))
            _ns = np.unique(np.round(np.logspace(0, 3.4, 240)).astype(int))
            _mn = np.array([np.trapezoid(_xs * nn * _phi * (1 - _Phi)**(nn - 1), _xs) for nn in _ns])
            _lnj = jnp.asarray(np.log(_ns.astype(float))); _mnj = jnp.asarray(_mn)
            _cond = gn.get('tts_walk_cond', 'n')
            if _cond == 'mu':
                # MODEL-side conditioning: Poisson-mixed, N>=1-conditioned E[min] evaluated at
                # the PREDICTED per-PMT total mu (smooth in the track parameters, so the shift
                # injects NO data noise -- the flaw of conditioning on the observed count n).
                from scipy.stats import poisson as _poisson
                _mus = np.logspace(-2, 3.6, 300)
                _mbar = np.zeros_like(_mus)
                for _i, _m in enumerate(_mus):
                    _nmax = int(max(20, _m + 8 * np.sqrt(_m) + 5))
                    _ns2 = np.arange(1, _nmax + 1)
                    _pn = _poisson.pmf(_ns2, _m) / max(1e-300, 1.0 - np.exp(-_m))
                    _mn_at = np.interp(np.log(_ns2.astype(float)), np.log(_ns.astype(float)), _mn)
                    _mbar[_i] = float(np.sum(_pn * _mn_at))
                _lmuj = jnp.asarray(np.log(_mus)); _mbarj = jnp.asarray(_mbar)
            _orig_fawn = _rec.first_arrival_window_nll
            if not getattr(_rec, '_fawn_wrapped', False):
                _lam = float(gn.get('tts_walk_scale', 1.0))
                if _cond == 'mu':
                    def _fawn_corr(log_w, flat_times, flat_indices, t_obs, mu_total, obs_counts,
                                   num_detectors, sigma=2.5, delta=1.0):
                        mu = jax.lax.stop_gradient(jnp.maximum(mu_total, 1e-2))
                        mn = jnp.interp(jnp.log(mu), _lmuj, _mbarj)
                        return _orig_fawn(log_w, flat_times, flat_indices,
                                          t_obs - _lam * _tts * mn, mu_total, obs_counts,
                                          num_detectors, sigma=sigma, delta=delta)
                else:
                    def _fawn_corr(log_w, flat_times, flat_indices, t_obs, mu_total, obs_counts,
                                   num_detectors, sigma=2.5, delta=1.0):
                        nn = jnp.maximum(obs_counts, 1.0)
                        mn = jnp.interp(jnp.log(nn), _lnj, _mnj)
                        return _orig_fawn(log_w, flat_times, flat_indices,
                                          t_obs - _lam * _tts * mn, mu_total, obs_counts,
                                          num_detectors, sigma=sigma, delta=delta)
                _rec.first_arrival_window_nll = _fawn_corr
                _rec._fawn_wrapped = True
            if verbose:
                print('[pipeline] tts_walk_corr ACTIVE (order-statistic tobs correction)', flush=True)
        _esm = gn.get('energy_scale_mode', 'simtotal')
        _model_kw = {}
        if _esm != 'simtotal':
            # 'nphot': single-pass energy scale from the analytic nphot(E) curve — build the
            # emitter context (same metadata the predictor uses) to get n_photons_fn.
            from lucid.utils import unpack_siren_params
            from lucid.siren.core import build_cherenkov_context
            from lucid.siren.training.inference import SIRENPredictor
            _scfg = unpack_siren_params(part, 'water')
            _ctx = build_cherenkov_context(SIRENPredictor(_scfg['siren_model_path']),
                                           _scfg['ray_sampling'])
            _model_kw = dict(energy_scale_mode='nphot', nphot_fn=_ctx.n_photons_fn)
            if verbose:
                print(f"[pipeline] energy_scale_mode -> nphot (single-pass)", flush=True)
        if gn.get('time_weight', 1.0) != 1.0:
            _model_kw['time_weight'] = float(gn['time_weight'])
            if verbose:
                print(f"[pipeline] time_weight -> {gn['time_weight']}", flush=True)
        self.model = ReconModel(self.pred, self.ND, sigma=gn['sigma'], delta=gn['delta'],
                                tot_n_scale=gn['tot_n_scale'], **_model_kw)
        self._qtot_vg = None                       # lazy jit for the E-refit stage
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

    def _siren_pseudodata(self, raw, ev):
        """CLOSURE TEST: replace the PhotonSim photons with photons sampled from the SIREN
        emitter itself (same energy, same photon count, gun frame: origin 0 along +z).
        Origins on the track axis, directions from the emitter, emission times from the
        model's own predict_t0 curve. Reconstruction of these events tests the estimator
        with data drawn from its OWN model - any surviving bias is intrinsic to the
        likelihood/detection machinery, not data-model mismatch."""
        import jax
        if getattr(self, '_pseudo_ctx', None) is None:
            from lucid.utils import unpack_siren_params, unpack_t0_params
            from lucid.siren.core import build_cherenkov_context
            from lucid.siren.training.inference import SIRENPredictor
            from lucid.sources.siren_rays import make_cherenkov_surrogate_fn
            scfg = unpack_siren_params(self.cfg['particle'], 'water')
            pred = SIRENPredictor(scfg['siren_model_path'])
            ctx = build_cherenkov_context(pred, dict(scfg['ray_sampling']))
            self._pseudo_ctx = (ctx, make_cherenkov_surrogate_fn(ctx), pred,
                                unpack_t0_params(self.cfg['particle'], 'water'))
        ctx, get_rays, spred, (a_c, l_c, b_c) = self._pseudo_ctx
        E = float(raw['energy']); n_phot = len(np.asarray(raw['photon_origins']))
        vec, org, inten = get_rays(jnp.zeros(3), jnp.array([0., 0., 1.]), jnp.asarray(E),
                                   250_000, spred.params, jax.random.PRNGKey(900_000 + ev))
        w = np.asarray(inten, float); w = np.clip(w, 0, None); w /= w.sum()
        rng = np.random.default_rng(910_000 + ev)
        idx = rng.choice(len(w), size=n_phot, p=w)
        org = np.asarray(org, float)[idx]; vec = np.asarray(vec, float)[idx]
        s_mm = np.clip(org[:, 2], 0, None) * 1000.0
        x = np.log10(E)
        def _cub(c):
            return c[0] + c[1]*x + c[2]*x*x + c[3]*x*x*x
        A = 10.0 ** _cub(np.asarray(a_c)); lam = 10.0 ** _cub(np.asarray(l_c))
        beta = _cub(np.asarray(b_c))
        t = s_mm / 299.792 + A * np.expm1(np.power(np.clip(s_mm / lam, 1e-12, None), beta))
        out = dict(raw)
        out['photon_origins'] = org * 100.0          # m -> cm (gun frame convention)
        out['photon_directions'] = vec
        out['photon_times'] = t
        return out

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
        if cfg.get('pseudodata'):
            raw = self._siren_pseudodata(raw, ev)
        track_len = self.smax_m(raw['energy']) if self.smax_m is not None else None
        # placement seed: base + ev*stride (stride 1000 + no containment + t0 [0,0]
        # reproduces the upstream sweep poses exactly; see DEFAULT_CONFIG comment).
        pseed = cfg['placement_seed_base'] + ev * (cfg.get('placement_seed_stride', 1) - 1)
        raw, vtx_true, dir_true = rand_tf(
            raw, ev, cfg['fidr'], cfg['fidz'], pseed,
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

        if cfg.get('fix_geometry'):
            # free-E diagnostic starts from truth — no seeder needed
            return dict(ev=int(ev), energy_true=float(raw['energy']), th9=th9, d=d,
                        oc=oc, ot=ot, seedA=None, seedB=None,
                        n_hit=int((oc > 0).sum()), q_tot=float(oc.sum()),
                        seed_seconds=float(time.time() - t_start))

        # --- seeds: shared energy scan, two complementary vertices, cone direction each ------
        sd = cfg['seed']
        if cfg.get('fix_energy'):
            e0 = float(raw['energy'])                      # E pinned to truth; no scan
        else:
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

    def _split_fns(self):
        """Jitted per-term (grad, jacobian) builders replicating ReconModel._perpmt exactly.

        nphot energy-scale mode only (the campaign default): mu carries the analytic escale,
        the survival denominator stays unscaled -- byte-matched to recon.py so the projected
        fitter optimizes the SAME objective, just with an anisotropic time-term contribution.
        """
        if getattr(self, '_splitf', None) is not None:
            return self._splitf
        from lucid.fitting import track_from_vec9
        from lucid.losses import counts_loss, first_arrival_window_nll
        gn = self.cfg['gn']
        if gn.get('energy_scale_mode', 'nphot') != 'nphot':
            raise NotImplementedError('time_soft_mode=project requires energy_scale_mode=nphot')
        sigma, delta = float(gn['sigma']), float(gn['delta'])
        tns = float(gn['tot_n_scale']); ND = self.ND
        nphot_fn = self.model.nphot_fn

        def perpmt(t9, oc, ot, key):
            lw, ft, fi, tot = self.pred(track_from_vec9(t9), key)
            npn = jnp.maximum(nphot_fn(t9[0]), 1e3)
            npd = jnp.maximum(nphot_fn(jax.lax.stop_gradient(t9[0])), 1e3)
            mu = jnp.maximum(tot * (npn / npd) * tns, 1e-8)
            mu_surv = jnp.maximum(tot, 1e-8)
            tnll = first_arrival_window_nll(lw, ft, fi, ot - t9[8], mu_surv, oc, ND,
                                            sigma=sigma, delta=delta)
            return mu, tnll

        def LQ(t9, oc, ot, key):
            mu, _ = perpmt(t9, oc, ot, key)
            return counts_loss(oc, mu, eps=0.0, normalize=False)

        def LT(t9, oc, ot, key):
            _, tnll = perpmt(t9, oc, ot, key)
            return jnp.sum(tnll)

        self._splitf = dict(
            gQ=jax.jit(jax.grad(LQ)), gT=jax.jit(jax.grad(LT)),
            pjac=jax.jit(jax.jacfwd(perpmt, argnums=0)),
            perpmt=jax.jit(perpmt))
        return self._splitf

    def _fit_track_projected(self, oc, ot, start):
        """fit_track (ad recipe) with the TIME term's soft-mode component projected out.

        Soft mode v = (0.285 m along the CURRENT direction, +1 ns of t0) in SCALE9 coords --
        the measured vertex-t0 degeneracy. Per iteration: g = gQ + P gT, F = FQ + P FT P with
        P = I - vv^T. The time term keeps full transverse/direction/stiff-t0 power but cannot
        pull along the degeneracy. Recipe knobs mirror fit_track (lr 4->1.5, lam .01,
        ridge_i .1, refresh 8, trust 3, NaN guard, Polyak-40).
        """
        from lucid.fitting.recon import SCALE9
        gn = self.cfg['gn']
        nkeys, niters = gn['nkeys'], gn['niters']
        lr, lr_final, lam, ridge_i, refresh = float(gn['lr']), 1.5, 0.01, 0.1, 8
        trust, polyak_w = 3.0, 40
        F = self._split_fns()
        ocj, otj = jnp.asarray(oc), jnp.asarray(ot)
        keys = [jax.random.PRNGKey(s) for s in range(nkeys)]
        S = SCALE9

        def grads(th):
            t9 = jnp.asarray(th)
            gq = np.mean([np.asarray(F['gQ'](t9, ocj, otj, k)) for k in keys], 0)
            gt = np.mean([np.asarray(F['gT'](t9, ocj, otj, k)) for k in keys], 0)
            return gq, gt

        def fishers(th):
            t9 = jnp.asarray(th)
            FQ = np.zeros((9, 9)); FT = np.zeros((9, 9))
            for k in keys:
                Jmu, Jl = F['pjac'](t9, ocj, otj, k)
                mu, _ = F['perpmt'](t9, ocj, otj, k)
                Jmu = np.asarray(Jmu); Jl = np.asarray(Jl); mu = np.asarray(mu)
                FQ += (Jmu / np.clip(mu, 1e-8, None)[:, None]).T @ Jmu
                FT += np.asarray(Jl).T @ np.asarray(Jl)
            return FQ / nkeys, FT / nkeys

        def soft_P(th):
            st, ct, sp, cp = th[4], th[5], th[6], th[7]
            nt = np.hypot(st, ct); npp = np.hypot(sp, cp)
            stn, ctn, spn, cpn = st / nt, ct / nt, sp / npp, cp / npp
            u = np.array([stn * cpn, stn * spn, ctn])
            v = np.zeros(9); v[1:4] = 0.285 * u; v[8] = 1.0
            vs = v / S; vs = vs / np.linalg.norm(vs)
            return np.eye(9) - np.outer(vs, vs)

        th = np.asarray(start, float)
        gq, gt = grads(th)
        traj = [th.copy()]; gnorms = []
        FQ = FT = None; since = 0
        for it in range(niters):
            if FQ is None or since >= refresh:
                FQ, FT = fishers(th); since = 0
            since += 1
            P = soft_P(th)
            gs = S * gq + P @ (S * gt)
            Fs = (S[:, None] * FQ * S[None, :]) + P @ (S[:, None] * FT * S[None, :]) @ P
            marq = np.diag(lam * np.diag(Fs))
            rI = ridge_i * np.median(np.clip(np.diag(Fs), 1e-12, None)) * np.eye(9)
            lr_it = lr + (lr_final - lr) * (it / max(1, niters - 1))
            du = -lr_it * np.linalg.solve(Fs + marq + rI + 1e-9 * np.eye(9), gs)
            du = np.clip(du, -trust, trust)
            th_new = th + S * du
            gq_n, gt_n = grads(th_new)
            if np.isfinite(th_new).all() and np.isfinite(gq_n).all() and np.isfinite(gt_n).all():
                th, gq, gt = th_new, gq_n, gt_n
            gn_ = float(np.linalg.norm(S * (gq + gt)))
            traj.append(th.copy()); gnorms.append(gn_)
        out = np.mean(np.array(traj)[-polyak_w:], axis=0)
        return out, dict(traj=np.array(traj), gnorm=np.array(gnorms),
                         best_iter=int(np.argmin(gnorms)))

    def reconstruct_free_E(self, ev):
        """Diagnostic: geometry+t0 pinned at TRUTH, only E free — single GN start from truth."""
        from lucid.fitting import fit_track, vec9_dir
        gn = self.cfg['gn']
        P = self._prepare_event(ev)
        th9, d, oc, ot = P['th9'], P['d'], P['oc'], P['ot']
        t_start = time.time()
        model = _FixedParamsModel(self.model, list(range(1, 9)))     # E (idx 0) free
        res, H = fit_track(model, oc, ot, th9, nkeys=gn['nkeys'], niters=gn['niters'],
                           lr=gn['lr'], fisher_mode=gn['fisher_mode'], hist=True)

        def _errs(x):
            dv = float(np.linalg.norm((np.asarray(x) - th9)[1:4]) * 100)
            dd = float(np.degrees(np.arccos(np.clip(vec9_dir(x) @ d, -1, 1))))
            return np.array([dv, dd, float(x[0] - th9[0]), float(x[8] - th9[8])])

        return dict(
            ev=int(ev), energy_true=P['energy_true'],
            truth_vec9=th9, truth_phys=vec9_to_phys(th9), tdir=d,
            fit_vec9=np.asarray(res), fit_phys=vec9_to_phys(res),
            traj_win=H['traj'], traj_win_phys=traj_to_phys(H['traj']),
            gnorm_win=H['gnorm'], best_iter_win=int(H['best_iter']), which=0,
            fit_err=_errs(res), n_hit=P['n_hit'], q_tot=P['q_tot'],
            seconds=P['seed_seconds'] + float(time.time() - t_start))

    def _e_refit(self, t9, oc, niters):
        """Stage-2 energy refit: all params FROZEN at the fit, E alone re-optimized on the
        total-charge loss L = 0.5 (Qpred(E) - Qobs)^2 by 1-D Gauss-Newton
        (E <- E - (Qpred - Qobs)/ (dQpred/dE)), key-averaged. Returns (t9', E history)."""
        from lucid.fitting import track_from_vec9
        gn = self.cfg['gn']; tns = float(gn['tot_n_scale'])
        if self._qtot_vg is None:
            def qtot(t9j, key):
                _, _, _, tot = self.pred(track_from_vec9(t9j), key)
                return jnp.sum(jnp.maximum(tot * tns, 1e-8))
            self._qtot_vg = jax.jit(jax.value_and_grad(qtot))
        keys = [jax.random.PRNGKey(gn.get('seed', 0) + s) for s in range(gn['nkeys'])]
        q_obs = float(np.sum(oc))
        t = np.array(t9, float); hist = [t[0]]
        for _ in range(int(niters)):
            qs, gs = [], []
            for k in keys:
                q, g = self._qtot_vg(jnp.asarray(t), k)
                qs.append(float(q)); gs.append(float(np.asarray(g)[0]))
            Q = float(np.mean(qs)); dQdE = float(np.mean(gs))
            if abs(dQdE) < 1e-9:
                break
            t[0] -= np.clip((Q - q_obs) / dQdE, -200.0, 200.0)   # 1-D GN step, clipped
            hist.append(t[0])
        return t, np.array(hist)

    def reconstruct(self, ev):
        """Reconstruct one PhotonSim event (seeds + three-start Fisher-GN). Returns numpy dict.

        Starts: seedA (charge-grid), seedB (time-multilateration), seedF (their FUSION —
        transverse vertex from B, longitudinal from A along A's direction, averaged t0).
        The fused seed was validated across geometry / energy / t0-range / particle / JUNO-sphere
        (50-event seed studies): best-or-equal vertex in every regime, t0 RMS ~7 ns -> ~2 ns, and
        the post-GN margin-gated loss pick (prefer=A) rescues the rare fusion failures (JUNO).
        """
        from lucid.fitting import fit_track, fit_track_multistart, vec9_dir
        if self.cfg.get('fix_geometry'):
            return self.reconstruct_free_E(ev)
        gn = self.cfg['gn']
        P = self._prepare_event(ev)
        th9, d, oc, ot = P['th9'], P['d'], P['oc'], P['ot']
        seedA, seedB = P['seedA'], P['seedB']
        seedF = fuse_seeds(seedA, seedB)
        starts = [seedA, seedB, seedF]
        t_start = time.time()
        model = _FixedParamsModel(self.model, [0]) if self.cfg.get('fix_energy') else self.model

        def _errs(x):
            dv = float(np.linalg.norm((np.asarray(x) - th9)[1:4]) * 100)          # vertex cm
            dd = float(np.degrees(np.arccos(np.clip(vec9_dir(x) @ d, -1, 1))))     # direction deg
            return np.array([dv, dd, float(x[0] - th9[0]), float(x[8] - th9[8])])  # +dE MeV, +dt0 ns

        rec = dict(
            ev=int(ev), energy_true=P['energy_true'],
            truth_vec9=th9, truth_phys=vec9_to_phys(th9), tdir=d,
            seedA_vec9=np.asarray(seedA), seedB_vec9=np.asarray(seedB),
            seedF_vec9=np.asarray(seedF),
            seedA_phys=vec9_to_phys(seedA), seedB_phys=vec9_to_phys(seedB),
            seedF_phys=vec9_to_phys(seedF),
            seedA_err=_errs(seedA), seedB_err=_errs(seedB), seedF_err=_errs(seedF),
            n_hit=P['n_hit'], q_tot=P['q_tot'])

        if self.cfg.get('multistart'):
            # --- three-start Fisher-GN; post-GN margin-gated selection, all trajectories ---
            res, MS = fit_track_multistart(model, oc, ot, starts,
                                           nkeys=gn['nkeys'], niters=gn['niters'], lr=gn['lr'],
                                           fisher_mode=gn['fisher_mode'])
            HA = MS['per_seed'][0][1]; HB = MS['per_seed'][1][1]; HF = MS['per_seed'][2][1]
            H = MS['per_seed'][MS['which']][1]
            rec.update(
                fitA_vec9=np.asarray(MS['per_seed'][0][0]),
                fitB_vec9=np.asarray(MS['per_seed'][1][0]),
                fitF_vec9=np.asarray(MS['per_seed'][2][0]),
                trajA=HA['traj'], trajB=HB['traj'], trajF=HF['traj'],
                gnormA=HA['gnorm'], gnormB=HB['gnorm'], gnormF=HF['gnorm'],
                best_iterA=int(HA['best_iter']), best_iterB=int(HB['best_iter']),
                best_iterF=int(HF['best_iter']),
                which=int(MS['which']), losses=np.asarray(MS['losses']))
        else:
            # --- single tracking (default): pre-GN margin-gated loss pick, ONE GN fit -------
            # Same gate as the validated seed study: prefer seedA unless another seed's data
            # loss beats it by margin x |loss| (the pick went to fused on 92-100% there).
            # force_seed 'A'|'B'|'F' bypasses the pick entirely.
            if self.cfg.get('force_seed'):
                pick = {'A': 0, 'B': 1, 'F': 2}[self.cfg['force_seed']]
                losses = [0.0, 0.0, 0.0]
            else:
                keys = [jax.random.PRNGKey(gn.get('seed', 0) + s) for s in range(gn['nkeys'])]

                def dloss(s):
                    return float(np.mean([float(self.model.loss(np.asarray(s), oc, ot, k))
                                          for k in keys]))
                losses = [dloss(s) for s in starts]
                gate = losses[0] - 0.01 * abs(losses[0])
                cand = [i for i in (1, 2) if losses[i] < gate]
                pick = min(cand, key=lambda i: losses[i]) if cand else 0
            if gn.get('time_soft_mode') == 'project':
                res, H = self._fit_track_projected(oc, ot, starts[pick])
            else:
                res, H = fit_track(model, oc, ot, starts[pick], nkeys=gn['nkeys'],
                                   niters=gn['niters'], lr=gn['lr'],
                                   fisher_mode=gn['fisher_mode'], hist=True)
            rec.update(which=int(pick), losses=np.asarray(losses))

        # optional stage 2: freeze everything at the fit, refit E alone on the Qtot loss
        if self.cfg.get('e_refit_iters', 0) > 0 and not self.cfg.get('fix_energy'):
            rec['fit_preErefit_vec9'] = np.asarray(res)
            res, ehist = self._e_refit(res, oc, self.cfg['e_refit_iters'])
            rec['e_refit_hist'] = ehist

        rec.update(
            fit_vec9=np.asarray(res), fit_phys=vec9_to_phys(res), fit_err=_errs(res),
            traj_win=H['traj'], traj_win_phys=traj_to_phys(H['traj']),
            gnorm_win=H['gnorm'], best_iter_win=int(H['best_iter']),
            seconds=P['seed_seconds'] + float(time.time() - t_start))
        return rec
