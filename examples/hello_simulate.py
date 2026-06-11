"""hello_simulate — propagate one muon through SK-like water and draw the event.

The differentiable forward: ParticleParams (truth track) -> per-PMT charge, shown with LUCiD's
canonical unrolled-cylinder event display (barrel rectangle + top/bottom cap discs).
Run:  python examples/hello_simulate.py   (writes hello_simulate.png)
"""
import jax, jax.numpy as jnp, numpy as np
from lucid.simulation import setup_event_simulator
from lucid.detector_params import ParticleParams
from lucid.visualization import create_detector_display

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
# Full grid so every PMT is reachable (a coarse grid drops sensors → exact-zero → white holes).
sim = setup_event_simulator(GEOM, 200_000, K=4, physics_config=PHYS,
                            default_detector_params=True, particle='muon',
                            n_cap=150, n_angular=250, n_height=150)

track = ParticleParams.from_cartesian(energy=1000., position=[0., 0., 0.],
                                      direction=[1., 0., 0.], t0=0.)       # horizontal → barrel ring
charge = np.asarray(sim(track, jax.random.PRNGKey(0))[3])                  # (n_sensors,) per-PMT charge
print(f'{(charge > 0).sum()} of {charge.size} PMTs lit, total charge {charge.sum():.0f} pe')

display = create_detector_display(GEOM, sparse=False)                      # canonical unrolled display
display(charge, np.zeros_like(charge), file_name='hello_simulate.png', perc_min=0.0, perc_max=99.5)
print('wrote hello_simulate.png')
