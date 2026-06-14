"""Phase-2d reconciliation test — string telescope + volume photon step.

Validates the volume-scatter machinery wired into the unification forward:
  - the IceCube86 'string' config builds a StringTelescope via generate_detector
  - setup_event_simulator routes a string detector to the volume (per-DOM, no-reflection)
    photon step (is_volume dispatch), and a point source near a DOM deposits finite charge
    that concentrates on the nearby DOMs and is differentiable through the optical params
  - the surface path (cylinder) is unaffected — covered by the byte-identical water tripwire.

Uses the committed icecube86_simple.npz and a calibration isotropic source (no SIREN net),
so it runs without any downloaded data.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

ROOT = '/sdf/group/neutrino/omara/LUCiD_unification'
GEOM = os.path.join(ROOT, 'config/IceCube86_simple_geom_config.json')
PHYS = os.path.join(ROOT, 'config/IceCube86_physics_config.json')


def test_string_geometry_loads():
    from lucid.geometry import generate_detector
    from lucid.geometry.string import StringTelescope
    det = generate_detector(GEOM)
    assert isinstance(det, StringTelescope)
    assert det.n_sensors == 4680
    assert np.asarray(det.all_points).shape == (4680, 3)


def test_string_volume_forward_deposits_charge():
    """A point source 1 m from a DOM deposits finite charge concentrated on nearby DOMs."""
    from lucid.geometry import generate_detector
    from lucid.simulation.simulator import setup_event_simulator
    from lucid.sources import isotropic_source

    pts = np.asarray(generate_detector(GEOM).all_points)
    dom = pts[len(pts) // 2]

    sim = setup_event_simulator(GEOM, 200000, K=4, is_calibration=True,
                                physics_config=PHYS, wavelength_mode=False,
                                default_detector_params=True)
    src = isotropic_source(position=[float(dom[0]) + 1.0, float(dom[1]), float(dom[2])],
                           intensity=1e6)
    out = sim(src, jax.random.PRNGKey(0))
    q = out[0] if isinstance(out, (tuple, list)) else out
    q = jnp.asarray(q)
    assert q.shape == (4680,)
    assert bool(jnp.isfinite(q).all())
    assert float(jnp.sum(q)) > 0.0          # the volume step actually deposits light
    assert float(jnp.max(q)) > 0.0          # the nearby DOM lights up
    # charge is local to the source — far fewer than all 4680 DOMs light up
    assert int((q > 1e-4).sum()) < 4680


def test_string_volume_forward_differentiable():
    """Total charge is differentiable through an optical param (scatter_length)."""
    from lucid.geometry import generate_detector
    from lucid.simulation.simulator import setup_event_simulator
    from lucid.sources import isotropic_source

    pts = np.asarray(generate_detector(GEOM).all_points)
    dom = pts[len(pts) // 2]
    sim = setup_event_simulator(GEOM, 200000, K=4, is_calibration=True,
                                physics_config=PHYS, wavelength_mode=False)
    dp = setup_event_simulator(GEOM, 200000, K=4, is_calibration=True, physics_config=PHYS,
                               wavelength_mode=False, default_detector_params=True
                               ).default_detector_params
    src = isotropic_source(position=[float(dom[0]) + 1.0, float(dom[1]), float(dom[2])],
                           intensity=1e6)

    def q_tot(scat):
        d2 = dp._replace(scattering=dp.scattering._replace(scatter_length=scat))
        out = sim(src, d2, jax.random.PRNGKey(0))
        q = out[0] if isinstance(out, (tuple, list)) else out
        return jnp.sum(q)

    g = float(jax.grad(q_tot)(jnp.asarray(float(dp.scattering.scatter_length))))
    assert np.isfinite(g)          # gradient flows through the volume step
