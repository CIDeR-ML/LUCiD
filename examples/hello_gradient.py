"""hello_gradient — differentiate the forward model with jax.grad, batch it with jax.vmap.

The simulator is a JAX function of the track, so the usual transforms apply to it directly:

* ``jax.grad`` of the total expected charge with respect to the energy (pe per MeV), and of a
  Poisson loss against an event at a shifted vertex with respect to the vertex (it points from
  the trial vertex toward the event's);
* ``jax.vmap`` of the total expected charge over a batch of energies, one call.

SK_like water, one muon, a coarse propagation grid and few rays so it runs in about a minute on a
CPU. Run:  python examples/hello_gradient.py [--n-rays N]
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np

from lucid.detector_params import ParticleParams
from lucid.simulation import setup_event_simulator

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
# coarse: fast to build, enough for gradients. A coarse cell holds up to 15 PMTs, so it needs as
# many candidates per ray (stated, rather than raised from the default 4 with a warning).
GRID = dict(n_cap=30, n_angular=40, n_height=30, max_candidates_per_ray=15)


def main(n_rays=20_000):
    # --8<-- [start:gradient]
    sim = setup_event_simulator(GEOM, n_rays, source='particle', K=4, physics_config=PHYS,
                                default_detector_params=True, particle='muon', **GRID)
    key = jax.random.PRNGKey(0)
    direction = jnp.array([1., 0., 0.])

    def expected_charge(energy, position):
        """Per-PMT expected charge: ``.charge`` of the track forward's ``PhotonList``."""
        track = ParticleParams.from_cartesian(energy=energy, position=position,
                                              direction=direction, t0=0.)
        return sim(track, key).charge

    vertex = jnp.array([0., 0., 0.])
    dq_de = jax.grad(lambda e: expected_charge(e, vertex).sum())(1000.)
    print(f'd(total expected charge)/dE at 1000 MeV: {float(dq_de):.2f} pe/MeV')

    # An "event" at a vertex 0.5 m along +x, fitted from the origin: the descent direction
    # -grad points toward the event's vertex.
    observed = jax.lax.stop_gradient(expected_charge(1000., jnp.array([0.5, 0., 0.])))

    def poisson_loss(position):
        mu = expected_charge(1000., position) + 1e-6
        return jnp.sum(mu - observed * jnp.log(mu))

    g = jax.grad(poisson_loss)(vertex)
    print(f'-d(Poisson loss)/d(vertex) at the origin: {np.round(-np.asarray(g), 1)} per m '
          '(event vertex at +0.5 m in x)')

    energies = jnp.array([500., 1000., 1500.])
    totals = jax.vmap(lambda e: expected_charge(e, vertex).sum())(energies)
    print('total expected charge, vmapped over energy:',
          ', '.join(f'{float(e):.0f} MeV -> {float(q):.0f} pe' for e, q in zip(energies, totals)))
    # --8<-- [end:gradient]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-rays', type=int, default=20_000, help='rays per forward [20000]')
    main(**vars(ap.parse_args()))
