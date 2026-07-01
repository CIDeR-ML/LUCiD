"""hello_telescope — a muon track and a cascade in an IceCube-style string array (ice).

The same differentiable forward, on a neutrino-telescope geometry: DOMs on vertical strings in
ice. Shows the two canonical telescope event topologies —
  * a through-going muon TRACK (particle sim), and
  * a CASCADE / shower (a `cascade_source`) —
and prints how many DOMs each lights. Uses the water-trained SIREN emitter for ice for now
(data/ice symlinks to water; a dedicated ice emitter can drop in later).

Run:  python examples/hello_telescope.py
"""
import jax, numpy as np
from lucid.simulation import setup_event_simulator
from lucid.detector_params import ParticleParams
from lucid.sources.cascade import cascade_source

GEOM = 'config/IceCube86_full_geom_config.json'
PHYS = 'config/IceCube86_ice_physics_config.json'

# --- through-going muon track (particle sim) ---
trk_sim = setup_event_simulator(GEOM, 500_000, K=4, temperature=None, is_calibration=False,
                                detector_type='string', physics_config=PHYS,
                                default_detector_params=True, hit_mode='aggregated',
                                wavelength_mode=False, particle='muon', use_expected_value=True)
centroid = np.asarray(trk_sim.det_geom.detector.all_points).mean(0).tolist()   # place inside the array
track = ParticleParams.from_cartesian(energy=100_000., position=centroid, direction=[1., 0., 0.], t0=0.)
q_trk = np.asarray(jax.lax.stop_gradient(trk_sim(track, jax.random.PRNGKey(0)))[0])
print(f'muon track (100 GeV, horizontal): {(q_trk > 0).sum():4d} / {q_trk.size} DOMs lit, '
      f'total {q_trk.sum():.0f} pe')

# --- cascade / shower (source sim) ---
cas_sim = setup_event_simulator(GEOM, 500_000, K=4, temperature=0.2, is_calibration=True,
                                detector_type='string', physics_config=PHYS,
                                default_detector_params=True, hit_mode='aggregated',
                                wavelength_mode=False)
src = cascade_source(position=centroid, direction=[0., 0., 1.], energy_mev=100_000.)
q_cas = np.asarray(jax.lax.stop_gradient(cas_sim(src, jax.random.PRNGKey(1)))[0])
print(f'cascade    (100 GeV, EM shower):  {(q_cas > 0).sum():4d} / {q_cas.size} DOMs lit, '
      f'total {q_cas.sum():.0f} pe')
