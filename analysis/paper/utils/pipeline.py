"""Config-driven single-track reconstruction pipeline for the tracking studies.

Generalises ``scripts/campaign_recon/worker.py`` into a reusable, config-parameterised
pipeline for the tracking studies (nrays / energy / geom, muon / electron). For each event:

  1. load GEANT4/PhotonSim photons (``source='photons'`` — unit-weight integer photons) and the
     primary's recorded vertex and direction,
  2. place the event (:mod:`lucid.sources.placement`): a uniform rotation about the primary
     vertex, the vertex uniform in the fiducial cylinder, redrawn until the track end point is
     contained; per-event streams from :func:`lucid.sources.seed_utils.derive_event_keys`
     (the seeder's ``seed_key`` and the fit's ``fit_key`` among them),
  3. generate realistic per-PMT (charge, first-arrival time) with per-photon TTS, the true t0
     added to the hit times,
  4. reconstruct with :func:`lucid.fitting.reconstruct`: data-driven seeds A, B and their
     fusion F, a pre-fit pick by data loss, then one Fisher-GN fit, keeping the full
     per-iteration trajectory.

The truth is exact: the placement moves the recorded primary with the photons, so
``(vtx_true, dir_true)`` is the placed vertex and the rotated primary direction.

There is no scalar charge normalisation; the model λ is
sampled over the PhotonSim Cherenkov emission band so QE applies to the true band.

Usage
-----
    from analysis.paper.utils import studies
    from analysis.paper.utils.pipeline import TrackingPipeline
    cfg  = studies.named_config('nrays_mu_250k', n_events=10)
    pipe = TrackingPipeline(cfg)          # builds detector + simulators ONCE
    rec  = pipe.reconstruct(ev=0)         # dict of truth / seeds / trajectories / errors
"""
import json
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from analysis.paper.utils import studies

REPO_ROOT = Path(__file__).resolve().parents[3]      # analysis/paper/utils/pipeline.py -> LUCiD/

# --------------------------------------------------------------------------- config

def load_config(path):
    """A complete study config, as ``studies`` writes them (see ``studies.base_config``)."""
    with open(path) as f:
        return json.load(f)


def _check_complete(config):
    """Every key a published study config has; a hand-built config without one is refused here
    rather than by a KeyError deep in the constructor."""
    missing = sorted(set(studies.base_config('muon', 0, 0, 0, '', '')) - set(config))
    if missing:
        raise KeyError(f'config {config.get("name")!r} lacks {missing}; build it with '
                       f'studies.base_config or studies.named_config')


def _resolve(p):
    """Resolve a path against the repo root if it is not absolute."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


# --------------------------------------------------------------------------- pipeline

class TrackingPipeline:
    """Builds the reconstruction model and the data simulator ONCE from a config,
    then reconstructs individual events. Reuse a single instance across a config's events so
    the jitted forwards compile once."""

    def __init__(self, config, verbose=True):
        # imports are local so importing this module (e.g. for load_config in a ROOT-free
        # analysis context) does not pull in the simulator stack.
        from lucid.simulation import setup_event_simulator
        from lucid.detector_params import load_detector_params
        from lucid.fitting import FitRecipe, ModelRecipe, ReconModel, SeedRecipe
        from lucid.geometry.volume import CylinderVolume

        # Not copied: __init__ resolves 'nbuf' into the caller's dict, and run_local records that
        # same dict as the run's provenance.
        _check_complete(config)
        self.cfg = config
        geom = str(_resolve(config['geom_config'])); phys = str(_resolve(config['phys_config']))
        part = config['particle']
        self.seed, self.fit = SeedRecipe(**config['seed']), FitRecipe(**config['fit'])
        self.model = ReconModel.build(geom, phys, part, sigma=config['tts'],
                                      recipe=ModelRecipe(**config['model']), grid=config['grid'])
        self.det = self.model.pred.det_geom.detector
        self.ND = self.model.pred.det_geom.num_sensors
        # placement: the vertex is uniform in the fiducial cylinder, and the track end point
        # vertex + s_max(E)·dir must lie in this shrunk copy of the detector
        self.vertex_region = CylinderVolume(config['fidr'], -config['fidz'], config['fidz'])
        self.containment = self.det.volume.scaled(config['containment_margin'])
        self.smax_m = self.model.pred.emitter.s_max_m    # s_max(E) in m

        # nbuf 'auto': size the data buffer to the LARGEST event in the ROOT file (461k
        # photons/event at 2.1 GeV). The resolved int is written back into the config for
        # provenance.
        if config['nbuf'] in ('auto', None):
            from lucid.sources.root_reader import get_max_photons_per_particle
            nmax = get_max_photons_per_particle(str(_resolve(config['root_file'])))
            config['nbuf'] = nmax
            if verbose:
                print(f"[pipeline] nbuf auto -> {nmax} (max photons/event in ROOT)", flush=True)

        dp = load_detector_params(phys, num_sensors=self.ND)
        dp = dp._replace(response=dp.response._replace(tts=jnp.asarray(config['tts'])))
        self.data_sim = setup_event_simulator(
            geom, config['nbuf'], source='photons', temperature=None, K=config['model']['K'],
            physics_config=phys, default_detector_params=dp, particle=part,
            wavelength_mode=True, charge_resolution=config['charge_resolution'], **config['grid'])
        if verbose:
            print(f"[pipeline] {part} | {self.ND} PMTs | n_rays={config['model']['n_rays']} | "
                  f"geom={Path(geom).name} | root={Path(str(config['root_file'])).name}", flush=True)

    # -- one event ---------------------------------------------------------------------------

    def prepare_event(self, ev):
        """Load, place and simulate realistic data — no seeding, no fit.

        Returns a dict with the truth 9-vector + dir, the observables ``(oc, ot)``, the event's
        seeding and fit keys and provenance.
        """
        from lucid.sources.root_reader import read_photon_data_from_photonsim
        from lucid.sources.data_event import place_event, simulate_data_event
        from lucid.fitting import vec9_from_track
        cfg = self.cfg
        root = str(_resolve(cfg['root_file']))

        raw = read_photon_data_from_photonsim(root, ev)                 # metres
        track_len = self.smax_m(raw['energy'])
        raw, vtx_true, dir_true, keys = place_event(
            root, ev, raw, region=self.vertex_region, event_seed=cfg['event_seed'],
            accept=lambda v, d: bool(self.containment.contains(v + track_len * d)),
            max_tries=cfg['placement_max_tries'])
        lo, hi = cfg['true_t0_range']
        t0_true = float(np.random.default_rng(keys['t0_seed']).uniform(lo, hi))
        th9 = vec9_from_track(raw['energy'], vtx_true, dir_true, t0=t0_true)
        oc, ot = simulate_data_event(self.data_sim, raw, keys['sim_key'], nbuf=cfg['nbuf'], t0=t0_true)
        return dict(ev=int(ev), energy_true=float(raw['energy']), th9=th9,
                    d=np.asarray(dir_true), oc=oc, ot=ot, seed_key=keys['seed_key'],
                    fit_key=keys['fit_key'],
                    n_hit=int((oc > 0).sum()), q_tot=float(oc.sum()))

    def reconstruct(self, ev):
        """Reconstruct one PhotonSim event with :func:`lucid.fitting.reconstruct`. Returns a
        numpy dict: truth, the three seeds, the pick, and the fit with its trajectory.

        Starts: seedA (first-light grid), seedB (time multilateration), seedF (their FUSION —
        transverse vertex from B, longitudinal from A along A's direction, averaged t0).
        """
        from lucid.fitting import reconstruct, track_errors
        t_start = time.time()
        P = self.prepare_event(ev)
        th9, d, oc, ot = P['th9'], P['d'], P['oc'], P['ot']
        res, info = reconstruct(self.model, oc, ot, seed=self.seed, fit=self.fit,
                                seed_key=P['seed_key'], fit_key=P['fit_key'])
        seedA, seedB, seedF = (info['seeds'][k] for k in 'ABF')
        H = info['history']

        return dict(
            ev=int(ev), energy_true=P['energy_true'], truth_vec9=th9, tdir=d,
            seedA_vec9=np.asarray(seedA), seedB_vec9=np.asarray(seedB),
            seedF_vec9=np.asarray(seedF),
            n_hit=P['n_hit'], q_tot=P['q_tot'],
            which=int(info['which']), losses=np.asarray(info['losses']),
            fit_vec9=np.asarray(res), fit_err=track_errors(res, th9, d),
            traj_win=H['traj'], gnorm_win=H['gnorm'], best_iter_win=int(H['best_iter']),
            n_rejected=int(H['n_rejected']),        # refused steps on the fitted start
            seconds=float(time.time() - t_start))
