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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def _string_sim_and_dp(K):
    from lucid.simulation.simulator import setup_event_simulator
    sim = setup_event_simulator(GEOM, 200000, K=K, is_calibration=True,
                                physics_config=PHYS, wavelength_mode=False)
    dp = setup_event_simulator(GEOM, 200000, K=K, is_calibration=True, physics_config=PHYS,
                               wavelength_mode=False, default_detector_params=True
                               ).default_detector_params
    return sim, dp


def _string_src():
    from lucid.geometry import generate_detector
    from lucid.sources import isotropic_source
    pts = np.asarray(generate_detector(GEOM).all_points)
    dom = pts[len(pts) // 2]
    return isotropic_source(position=[float(dom[0]) + 1.0, float(dom[1]), float(dom[2])],
                            intensity=1e6)


def _qsum(sim, src, d, key):
    out = sim(src, d, key)
    q = out[0] if isinstance(out, (tuple, list)) else out   # sim returns (charges, times)
    return jnp.sum(q)


def test_string_volume_mie_channel_present():
    """The Rayleigh+Mie mixture is actually wired into the volume phase function:
    the forward output responds to BOTH the Mie scatter length and the asymmetry g
    (a Rayleigh-only step would be invariant to them)."""
    sim, dp = _string_sim_and_dp(K=4)
    src = _string_src()
    base = float(_qsum(sim, src, dp, jax.random.PRNGKey(0)))
    dp_lm = dp._replace(scattering=dp.scattering._replace(
        mie_scatter_length=jnp.asarray(float(dp.scattering.mie_scatter_length) * 0.05)))  # stronger Mie
    dp_g = dp._replace(scattering=dp.scattering._replace(g=jnp.asarray(0.0)))              # isotropic Mie
    q_lm = float(_qsum(sim, src, dp_lm, jax.random.PRNGKey(0)))
    q_g = float(_qsum(sim, src, dp_g, jax.random.PRNGKey(0)))
    assert abs(q_lm - base) > 1e-6 * abs(base), "forward does not respond to mie_scatter_length"
    assert abs(q_g - base) > 1e-6 * abs(base), "forward does not respond to the Mie asymmetry g"


def test_string_volume_optical_gradient_ad_matches_fd_single_step():
    """B-fit gate (single-step): the per-DOM volume DEPOSIT gradient w.r.t. the ice
    optical lengths is the EXACT reparam gradient — AD == FD at K=1 for scatter_length
    (Rayleigh), mie_scatter_length (Mie), and absorption_length, all of which enter the
    deposit through the rate-combined effective length / absorption weight.

    NOTE: this validates the deposit (single-step) gradient. The MULTI-STEP (K>1)
    trajectory gradient is NOT yet AD-faithful — the discrete per-step DOM-candidate
    selection in the string propagator is non-differentiable (AD diverges from FD as K
    grows). Making the full multi-bounce ice forward AD==FD (a differentiable candidate
    selection) is the open 'volume → DiCE-forward citizen' work and the ice-recon gate.
    """
    sim, dp = _string_sim_and_dp(K=1)
    src = _string_src()
    KEY = jax.random.PRNGKey(0)
    s, a = dp.scattering, dp.absorption

    def q_of(which):
        def f(x):
            if which == 'scat':
                d = dp._replace(scattering=s._replace(scatter_length=x))
            elif which == 'mie':
                d = dp._replace(scattering=s._replace(mie_scatter_length=x))
            else:
                d = dp._replace(absorption=a._replace(absorption_length=x))
            return _qsum(sim, src, d, KEY)
        return f

    for which, x0 in [('scat', float(s.scatter_length)),
                      ('mie', float(s.mie_scatter_length)),
                      ('abs', float(a.absorption_length))]:
        f = q_of(which)
        ad = float(jax.grad(f)(jnp.asarray(x0)))
        h = abs(x0) * 0.02
        fd = float((f(jnp.asarray(x0 + h)) - f(jnp.asarray(x0 - h))) / (2 * h))
        assert np.isfinite(ad) and np.isfinite(fd)
        # central-difference reparam gradient: AD and FD agree to a few %
        np.testing.assert_allclose(ad, fd, rtol=0.05, atol=1e-8 * (abs(fd) + 1.0),
                                   err_msg=f"AD!=FD for {which}: AD={ad:.4e} FD={fd:.4e}")
