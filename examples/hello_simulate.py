"""hello_simulate — propagate one muon through SK-like water and draw the event.

The differentiable forward: ParticleParams (truth track) -> expected charge per PMT, shown with
LUCiD's canonical unrolled-cylinder event display (barrel rectangle + top/bottom cap discs).
Run:  python examples/hello_simulate.py [--n-rays N]   (writes hello_simulate.png)
"""
import argparse

import jax
import numpy as np

from lucid.detector_params import ParticleParams
from lucid.simulation import setup_event_simulator
from lucid.visualization import create_detector_display

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'


def main(n_rays=200_000, out='hello_simulate.png'):
    # Explicit propagation-grid cell counts (a grid keyword, like max_candidates_per_ray); without
    # them the grid is sized from the sensor spacing.
    sim = setup_event_simulator(GEOM, n_rays, source='particle', K=4, physics_config=PHYS,
                                default_detector_params=True, particle='muon',
                                n_cap=150, n_angular=250, n_height=150)

    track = ParticleParams.from_cartesian(energy=1000., position=[0., 0., 0.],
                                          direction=[1., 0., 0.], t0=0.)  # horizontal → barrel ring
    charge = np.asarray(sim(track, jax.random.PRNGKey(0)).charge)         # (n_sensors,) expected pe
    print(f'expected charge: {charge.sum():.0f} pe over {charge.size} PMTs')

    display = create_detector_display(GEOM, sparse=False)                 # unrolled cylinder
    display(charge, np.zeros_like(charge), file_name=out, perc_min=0.0, perc_max=99.5)
    print(f'wrote {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-rays', type=int, default=200_000, help='rays per event [200000]')
    ap.add_argument('--out', default='hello_simulate.png', help='display file [hello_simulate.png]')
    main(**vars(ap.parse_args()))
