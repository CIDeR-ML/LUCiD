"""hello_telescope — a muon track and a cascade in an IceCube-style string array (ice).

The same differentiable forward, on a neutrino-telescope geometry: DOMs on vertical strings in
ice. Shows the two canonical telescope event topologies —
  * a through-going muon TRACK (particle sim), and
  * a CASCADE / shower (a `cascade_source`) —
and prints the expected charge each puts on the DOMs. Ice has no emitters of its own: its
material file (``config/materials/ice.json``, ``"siren": "water"``) runs water's.

Run:  python examples/hello_telescope.py [--n-rays N]
"""
import argparse

import jax
import numpy as np

from lucid.detector_params import ParticleParams
from lucid.simulation import setup_event_simulator
from lucid.sources.cascade import cascade_source

GEOM = 'config/IceCube86_full_geom_config.json'
PHYS = 'config/IceCube86_ice_physics_config.json'
K = 15   # ice scatters strongly (p_scat~0.77); K=15 keeps the multiple-scatter tail (K=4 dropped ~35%)


def main(n_rays=500_000):
    # --- through-going muon track (particle sim) ---
    trk_sim = setup_event_simulator(GEOM, n_rays, source='particle', hits='first_arrival', K=K,
                                    temperature=None, physics_config=PHYS,
                                    default_detector_params=True, wavelength_mode=False,
                                    particle='muon')
    centroid = np.asarray(trk_sim.det_geom.detector.all_points).mean(0).tolist()  # in the array
    track = ParticleParams.from_cartesian(energy=100_000., position=centroid,
                                          direction=[1., 0., 0.], t0=0.)
    q_trk = np.asarray(trk_sim(track, jax.random.PRNGKey(0)).charge)
    print(f'muon track (100 GeV, horizontal): expected charge {q_trk.sum():.0f} pe, '
          f'nonzero on {(q_trk > 0).sum()} of {q_trk.size} DOMs')

    # --- cascade / shower (source sim) ---
    cas_sim = setup_event_simulator(GEOM, n_rays, source='light_source', K=K, temperature=0.2,
                                    physics_config=PHYS, default_detector_params=True,
                                    wavelength_mode=False)
    src = cascade_source(position=centroid, direction=[0., 0., 1.], energy_mev=100_000.)
    q_cas = np.asarray(cas_sim(src, jax.random.PRNGKey(1)).charge)
    print(f'cascade    (100 GeV, EM shower):  expected charge {q_cas.sum():.0f} pe, '
          f'nonzero on {(q_cas > 0).sum()} of {q_cas.size} DOMs')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-rays', type=int, default=500_000, help='rays per event [500000]')
    main(**vars(ap.parse_args()))
