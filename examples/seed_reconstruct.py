"""seed_reconstruct — reconstruct a muon or electron track with the published method.

The start is built from the observed per-PMT charge and first-arrival time alone, then refined by
the Fisher-Gauss-Newton fit, through ``lucid.fitting.reconstruct`` at the published working point
(``MODEL_RECIPE``, ``SEED_RECIPE``, ``FIT_RECIPE``; TTS 2.1 ns):

  energy   — total-charge scan along a probe track through the detector centre      [forward]
  Seed A   — first-light grid: vertex/t0 grid on arrival-time residuals + cone dir  [geometric+forward]
  Seed B   — time multilateration (seed_vertex_time) + cone direction               [geometric+forward]
  Seed F   — fuse_seeds(A, B): B's transverse vertex, A's longitudinal vertex and direction
  Pick     — A unless B or F has a data loss lower by pick_margin; then one fit_track.

The event is sampled from the SIREN emitter at a known track (default), or, with ``--geant4``, is
a GEANT4 event of the 1 GeV muon PhotonSim slice in ``data/`` (``python scripts/download_data.py``),
placed at a random vertex in the tank so its truth is known. ``--particle`` is the particle that made
the event (a GEANT4 event is a muon), ``--hypothesis`` the model fitted to it. The last line is the data loss at the fit: fit one
event under both hypotheses and the lower loss names the particle.

Run:  python examples/seed_reconstruct.py [--geant4 [--event N]] [--particle mu|e]
                                          [--hypothesis mu|e] [--n-rays N] [--steps N]
                                          [--quick]

At the defaults the fit's forward-mode Fisher matrix needs more GPU memory than JAX takes by
default (75%) on an 11 GB card: there, run with XLA_PYTHON_CLIENT_MEM_FRACTION=.95 (about 8 min
on an RTX 2080 Ti).
"""
import argparse
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from lucid.detector_params import load_detector_params
from lucid.fitting import (ReconModel, FIT_RECIPE, MODEL_RECIPE, SEED_RECIPE, reconstruct,
                           vec9_from_track, vec9_dir, track_from_vec9)
from lucid.simulation import setup_event_simulator

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
# the published propagation grid, of the data simulator and the model: 8 candidate sensors per ray
# (the library default is 4), so the soft deposit reaches a sensor of a neighbouring cell
GRID = dict(n_cap=80, n_angular=120, n_height=80, max_candidates_per_ray=8)
TTS = 2.1                     # the data's transit-time spread (ns); the likelihood's sigma matches
PARTICLE = {'mu': 'muon', 'e': 'electron'}
# --quick: a few energy and direction trials instead of SEED_RECIPE's ~460 forwards, and 2 fit keys
# instead of 8, for a smoke run (CI, CPU). Not the published recipe.
QUICK_SEED = replace(SEED_RECIPE, energy_steps=15, cone_levels=2, cone_initial_div=4, cone_rings=2)
QUICK_NKEYS = 2


def siren_event(particle, dp_data, n_rays):
    """A sampled event from the SIREN emitter at a fixed track: ``(oc, ot, truth 9-vector)``."""
    data_sim = setup_event_simulator(GEOM, n_rays, source='particle', transport='sampled',
                                     detection='sampled', hits='first_arrival',
                                     temperature=None, K=MODEL_RECIPE.K,
                                     charge_resolution=None, particle=particle, physics_config=PHYS,
                                     default_detector_params=dp_data, wavelength_mode=True, **GRID)
    th9 = vec9_from_track(1050., [1.5, -0.8, 2.0], [0.3, 0.1, 0.95], t0=0.)
    c, t = jax.lax.stop_gradient(data_sim(track_from_vec9(jnp.asarray(th9)), jax.random.PRNGKey(0)))
    oc = np.asarray(c)
    return oc, np.where(oc > 0, np.asarray(t), 0.), th9


def geant4_event(dp_data, event):
    """GEANT4 event ``event`` of the bundled 1 GeV muon slice, at a random vertex in the tank."""
    from lucid.geometry.volume import CylinderVolume
    from lucid.sources.data_event import place_event, simulate_data_event
    from lucid.sources.root_reader import read_photon_data_from_photonsim
    root = 'data/water/muon/1000MeV_5events.root'
    nbuf = 400_000                                     # holds the event's photons
    data_sim = setup_event_simulator(GEOM, nbuf, source='photons', temperature=None,
                                     K=MODEL_RECIPE.K, charge_resolution=None,
                                     particle='muon', physics_config=PHYS,
                                     default_detector_params=dp_data, wavelength_mode=True, **GRID)
    raw = read_photon_data_from_photonsim(root, event)
    placed, vertex, direction, keys = place_event(root, event, raw,
                                                  region=CylinderVolume(12.0, -12.0, 12.0),
                                                  event_seed=100003, accept=lambda v, d: True,
                                                  max_tries=1)
    th9 = vec9_from_track(float(raw['energy']), vertex.tolist(), direction.tolist(), t0=0.)
    oc, ot = simulate_data_event(data_sim, placed, keys['sim_key'], nbuf=nbuf, t0=0.0)
    return oc, ot, th9


def main(geant4=False, particle='mu', hypothesis=None, event=0,
         n_rays=MODEL_RECIPE.n_rays, steps=FIT_RECIPE.steps, quick=False):
    if geant4 and particle != 'mu':
        raise SystemExit('the bundled GEANT4 slice is muons: --geant4 takes --particle mu')
    hypothesis = hypothesis or particle
    seed = QUICK_SEED if quick else SEED_RECIPE
    fit = replace(FIT_RECIPE, steps=steps, nkeys=QUICK_NKEYS if quick else FIT_RECIPE.nkeys)
    # --8<-- [start:model]
    # the published model: a per-photon, wavelength-consistent forward (MODEL_RECIPE) and the
    # likelihood; sigma must match the data's TTS (ns)
    model = ReconModel.build(GEOM, PHYS, PARTICLE[hypothesis], sigma=TTS,
                             recipe=replace(MODEL_RECIPE, n_rays=n_rays), grid=GRID)
    # --8<-- [end:model]
    dp_data = load_detector_params(PHYS, num_sensors=model.pred.det_geom.num_sensors)
    dp_data = dp_data._replace(response=dp_data.response._replace(tts=jnp.asarray(TTS)))
    if geant4:
        oc, ot, th9 = geant4_event(dp_data, event)
    else:
        oc, ot, th9 = siren_event(PARTICLE[particle], dp_data, n_rays)
    source = f'GEANT4 event {event}' if geant4 else 'SIREN-sampled event'
    print(f'data: {source}, {PARTICLE[particle]}, truth E {th9[0]:.0f} MeV; '
          f'{(oc > 0).sum()} hit PMTs, total charge {oc.sum():.0f} pe; '
          f'hypothesis {PARTICLE[hypothesis]}\n', flush=True)

    # --8<-- [start:reconstruct]
    # oc, ot: the event's per-PMT charge and first-hit time, (model.pred.det_geom.num_sensors,)
    res, info = reconstruct(model, oc, ot, seed=seed, fit=fit,
                            seed_key=jax.random.PRNGKey(1),   # the event's seeding and fit streams
                            fit_key=jax.random.PRNGKey(2))
    # res: the 9-vector [E, x,y,z, sinθ,cosθ, sinφ,cosφ, t0] (track_from_vec9 -> ParticleParams);
    # info: the three seeds, their pre-fit losses, the one fitted ('which') and the fit history
    # --8<-- [end:reconstruct]

    td = vec9_dir(th9)
    for name in 'ABF':
        s = info['seeds'][name]
        dv = np.linalg.norm(s[1:4] - th9[1:4]) * 100
        dd = np.degrees(np.arccos(np.clip(vec9_dir(s) @ td, -1, 1)))
        print(f'seed {name}: vtx |Δ| {dv:5.1f} cm  dir {dd:4.1f}°  E {s[0]:.0f} MeV  '
              f'pre-fit loss {info["losses"]["ABF".index(name)]:.5e}')
    fv = np.linalg.norm((res - th9)[1:4]) * 100
    fd = np.degrees(np.arccos(np.clip(vec9_dir(res) @ td, -1, 1)))
    print(f'\nfit (from seed {"ABF"[info["which"]]}): vtx {fv:5.1f} cm   '
          f'E {res[0] - th9[0]:+6.1f} MeV   dir {fd:4.2f}°   t0 {res[8] - th9[8]:+.2f} ns')
    # The data loss at the fit, on the fit's keys: the metric the pre-fit pick ranks starts by.
    keys = fit.keys(jax.random.PRNGKey(2))
    loss = np.mean([float(model.loss(jnp.asarray(res), oc, ot, k)) for k in keys])
    print(f'data loss at the fit under the {PARTICLE[hypothesis]} hypothesis: {loss:.5e}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--geant4', action='store_true',
                    help='fit a GEANT4 event of the bundled slice instead of a SIREN-sampled one')
    ap.add_argument('--event', type=int, default=0, help='GEANT4 event index, 0-4 [0]')
    ap.add_argument('--particle', choices=PARTICLE, default='mu', help='particle of the event [mu; a GEANT4 event is a muon]')
    ap.add_argument('--hypothesis', choices=PARTICLE, default=None,
                    help='particle model to fit with [the event\'s particle]')
    ap.add_argument('--n-rays', type=int, default=MODEL_RECIPE.n_rays,
                    help=f'rays per forward, model and sampled event [{MODEL_RECIPE.n_rays}]')
    ap.add_argument('--steps', type=int, default=FIT_RECIPE.steps,
                    help=f'Gauss-Newton steps [{FIT_RECIPE.steps}]')
    ap.add_argument('--quick', action='store_true',
                    help='a coarse seed search and 2 fit keys, for a smoke run (not the published '
                         'recipe)')
    main(**vars(ap.parse_args()))
