"""hello_simulate — propagate one muon through SK-like water and draw the event.

The differentiable forward: ParticleParams (truth track) -> per-sensor charge.
Run:  python examples/hello_simulate.py   (CPU is fine; writes hello_simulate.png)
"""
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import ParticleParams

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
# (small grid + photon count keep this a ~10s demo; production uses ~1e6 photons, full grid)
sim = setup_event_simulator(GEOM, 200_000, K=4, physics_config=PHYS,
                            default_detector_params=True, particle='muon',
                            n_cap=40, n_angular=80, n_height=40)

track = ParticleParams.from_cartesian(energy=1000., position=[0., 0., 0.],
                                      direction=[0., 0., 1.], t0=0.)
charge = np.asarray(sim(track, jax.random.PRNGKey(0))[3])      # (n_sensors,) per-PMT charge
print(f'{(charge > 0).sum()} of {charge.size} PMTs lit, total charge {charge.sum():.0f} pe')

P = np.asarray(generate_detector(GEOM).all_points)            # sensor xyz
phi, lit = np.arctan2(P[:, 1], P[:, 0]), charge > 0           # unrolled-cylinder display
plt.scatter(phi[lit], P[lit, 2], c=charge[lit], s=6, cmap='inferno')
plt.colorbar(label='charge [pe]'); plt.xlabel('azimuth φ'); plt.ylabel('z [m]')
plt.title('hello_simulate: 1 GeV muon (+z)'); plt.tight_layout()
plt.savefig('hello_simulate.png', dpi=110); print('wrote hello_simulate.png')
