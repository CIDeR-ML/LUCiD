"""Phase-6 re-validation — energy-gradient gate on the new s/s_max SIREN emitter.

The refactor-v2 Cherenkov emitter sets each photon's intensity to
``pmf(angle, s/s_max) × n_photons_fn(E)`` (the photon budget comes straight from the
trained-model ``nphot`` block — no manual reweighting). The author flagged the risk
that an inconsistent ``n_photons`` function would give a wrong-signed or AD≠FD energy
gradient. This pins, on a water track forward:

  - d q_tot / dE > 0 (more energy ⇒ more light), and
  - reverse-mode AD agrees with a common-random-number central finite difference
    (the emitter's energy dependence is differentiable, not score-noisy).

Needs the downloaded Cherenkov SIREN net; skipped when absent.
"""
import os
import pytest
import jax
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEOM = os.path.join(ROOT, 'config/SK_like_geom_config.json')
PHYS = os.path.join(ROOT, 'config/SK_like_physics_config.json')
_NET = os.path.join(ROOT, 'data/water/muon/siren_training/trained_model/photonsim_siren_weights.npz')

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.path.exists(_NET),
                       reason="water Cherenkov SIREN net not downloaded"),
]


def test_emitter_energy_gradient_ad_matches_fd():
    from lucid.simulation.simulator import setup_event_simulator
    from lucid.detector_params import ParticleParams

    sim = setup_event_simulator(GEOM, 30000, K=4, physics_config=PHYS,
                                wavelength_mode=False, default_detector_params=True)
    pp = ParticleParams(energy=jnp.array(500.0), position=jnp.array([0., 0., 0.]),
                        theta=jnp.array(0.1), phi=jnp.array(0.2), t0=jnp.array(0.0))
    key = jax.random.PRNGKey(0)   # fixed key ⇒ common random numbers for AD and FD

    def q_tot(E):
        return jnp.sum(sim(pp._replace(energy=E), key)[3])   # hits[3] = per-PMT charge

    E0 = jnp.array(500.0)
    g_ad = float(jax.grad(q_tot)(E0))
    h = 5.0
    g_fd = float((q_tot(E0 + h) - q_tot(E0 - h)) / (2 * h))

    assert g_ad > 0.0                      # more energy ⇒ more light
    assert g_fd > 0.0
    # AD matches CRN-FD to a few percent (sampling-noise floor at this photon count).
    assert abs(g_ad - g_fd) <= 0.05 * abs(g_fd) + 1e-3, f"AD {g_ad} vs FD {g_fd}"


def test_emitter_charge_scales_with_energy():
    """q_tot grows monotonically across the trained energy band."""
    from lucid.simulation.simulator import setup_event_simulator
    from lucid.detector_params import ParticleParams

    sim = setup_event_simulator(GEOM, 30000, K=4, physics_config=PHYS,
                                wavelength_mode=False, default_detector_params=True)
    pp = ParticleParams(energy=jnp.array(500.0), position=jnp.array([0., 0., 0.]),
                        theta=jnp.array(0.1), phi=jnp.array(0.2), t0=jnp.array(0.0))
    key = jax.random.PRNGKey(1)
    qs = [float(jnp.sum(sim(pp._replace(energy=jnp.array(E)), key)[3]))
          for E in (300.0, 600.0, 1200.0)]
    assert qs[0] < qs[1] < qs[2]
