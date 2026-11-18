"""Tests for wavelength integration into the simulation loop.

Verifies that:
1. The gradient is the derivative of the forward in both modes (central differences)
2. QE enters once in wavelength mode (closure with scalar mode at 400 nm, linear in qe)
3. Wavelength reaches the transport step (300 nm scatters out of the beam spot more than 500 nm)
4. Every shipped physics config survives wavelength mode
"""
import os
import pytest
import jax
import jax.numpy as jnp
from jax import value_and_grad, jit

from tests.helpers.params import calib_dp, central_difference, leaf, with_leaf
from tests.helpers.paths import REPO

pytestmark = pytest.mark.slow

GEOM = os.path.join(REPO, 'config', 'WCTE_like_geom_config.json')
PHYSICS = os.path.join(REPO, 'config', 'SK_physics_config.json')


@pytest.fixture(scope="module")
def detector():
    from lucid.geometry import generate_detector
    return generate_detector(GEOM)


@pytest.fixture(scope="module")
def base_dp(detector):
    return calib_dp(len(detector.all_points))


# ── Gradient tests ─────────────────────────────────────────────────

class TestGradients:
    """The autodiff gradient is the derivative of the forward, in both modes."""

    @pytest.mark.parametrize("wavelength_mode,wavelength", [(False, None), (True, 405.0)],
                             ids=["scalar", "wavelength"])
    def test_gradient_matches_central_differences(self, detector, base_dp, wavelength_mode,
                                                  wavelength):
        """d loss / d qe and d loss / d wall_reflection_rate equal central finite differences
        of the same jitted loss at a fixed key, away from the truth, and point back to it.

        The data are the forward at ``base_dp`` with the same key, and the gradient is taken at
        qe 0.2 -> 0.25 and wall reflection 0.2 -> 0.3, where it is not zero. The charge is
        linear in qe and polynomial in the reflection rate, so the central difference at a 1 %
        step is good to 1e-4 (measured 9e-5 for qe, 1.3e-5 for the reflection rate) against a
        0.5 % tolerance.

        In wavelength mode the scatter and absorption scalars are replaced by the medium curves,
        so their gradient must be exactly zero there; in scalar mode it must not be.
        """
        from lucid.simulation import setup_event_simulator
        from lucid.sources import laser_source
        from lucid.losses import WC_smooth_loss

        sim = setup_event_simulator(
            GEOM, 5000, source='light_source', temperature=None, K=3,
            wavelength_mode=wavelength_mode)
        sim_ref = setup_event_simulator(
            GEOM, 5000, source='light_source', temperature=None, K=3,
            default_detector_params=base_dp,
            wavelength_mode=wavelength_mode)

        source = laser_source(position=[0., 0., detector.H / 2 - 0.1], intensity=1e8,
                              wavelength=wavelength)
        sp = jnp.array(detector.all_points)
        key = jax.random.PRNGKey(42)
        true_data = jax.lax.stop_gradient(sim_ref(source, key))

        @jit
        def loss_fn(dp):
            pred = sim(source, dp, key)
            return WC_smooth_loss(sp, *true_data, *pred,
                                  lambda_poisson=1.0, lambda_time=0.0, tau=2.0)

        start = with_leaf(with_leaf(base_dp, 'qe', 0.25), 'wall_reflection_rate', 0.3)
        loss, grads = value_and_grad(loss_fn)(start)
        assert jnp.isfinite(loss)
        for field, truth in (('qe', 0.2), ('wall_reflection_rate', 0.2)):
            fd = central_difference(loss_fn, start, field)
            ad = leaf(grads, field)
            assert abs(ad - fd) <= 5e-3 * abs(fd), (field, ad, fd)
            assert ad * (leaf(start, field) - truth) > 0, \
                f'{field}: the gradient must point back to the truth'
        for field in ('scatter_length', 'absorption_length'):
            g = leaf(grads, field)
            if wavelength_mode:
                assert g == 0.0, f'{field} leaks into wavelength mode (d loss/d {field} = {g})'
            else:
                assert g != 0.0, f'{field} must reach the scalar-mode forward'


# ── QE handling ────────────────────────────────────────────────────

class TestQEHandling:
    """In wavelength mode, QE is baked into intensities."""


    def test_wavelength_mode_uses_qe_weighting(self, detector, base_dp):
        """Charge in wavelength mode should reflect QE curve shape."""
        from lucid.simulation import setup_event_simulator
        from lucid.sources import laser_source

        sim = setup_event_simulator(
            GEOM, 10000, source='light_source', temperature=None, K=4,
            default_detector_params=base_dp,
            physics_config=PHYSICS,
            wavelength_mode=True)

        key = jax.random.PRNGKey(42)

        # At 375nm QE peaks (~21%), at 600nm QE is very low (~1%)
        source_peak = laser_source(position=[0., 0., detector.H / 2 - 0.1],
                                   intensity=1e8, wavelength=375.0)
        source_low = laser_source(position=[0., 0., detector.H / 2 - 0.1],
                                  intensity=1e8, wavelength=600.0)

        c_peak, _ = sim(source_peak, key)
        c_low, _ = sim(source_low, key)

        # Peak QE wavelength should produce more detected charge
        # (even though scatter/absorption also differ, QE dominates)
        assert float(jnp.sum(c_peak)) > float(jnp.sum(c_low)), \
            f"Peak QE charge ({float(jnp.sum(c_peak)):.0f}) should exceed " \
            f"low QE charge ({float(jnp.sum(c_low)):.0f})"

    @staticmethod
    def _laser_charge(detector, dp, wavelength_mode):
        from lucid.simulation import setup_event_simulator
        from lucid.sources import laser_source

        sim = setup_event_simulator(
            GEOM, 100000, source='light_source', temperature=None, K=4,
            default_detector_params=dp,
            physics_config=PHYSICS, wavelength_mode=wavelength_mode)
        source = laser_source(position=[0., 0., detector.H / 2 - 0.1],
                              intensity=1e8, wavelength=400.0)
        charges, _ = sim(source, jax.random.PRNGKey(7))
        return float(jnp.sum(charges))

    def test_modes_agree_at_reference_wavelength(self, detector):
        """QE enters once in both modes, so a 400 nm laser gives the same charge.

        Scalar mode uses the scalars projected from the curves at 400 nm and wavelength
        mode evaluates those curves at 400 nm. Counting QE twice in wavelength mode would
        scale its charge by qe_fn(400 nm) ~ 0.23.
        """
        from lucid.detector_params import load_physics_config
        dp, _, _ = load_physics_config(PHYSICS, num_sensors=len(detector.all_points))
        scalar = self._laser_charge(detector, dp, wavelength_mode=False)
        per_photon = self._laser_charge(detector, dp, wavelength_mode=True)
        assert scalar > 0
        assert abs(per_photon / scalar - 1.0) < 0.03, (scalar, per_photon)

    def test_explicit_qe_rescales_wavelength_mode(self, detector):
        """In wavelength mode qe is still the QE at 400 nm: halving it halves the charge."""
        from lucid.detector_params import load_physics_config
        dp, _, _ = load_physics_config(PHYSICS, num_sensors=len(detector.all_points))
        half = dp._replace(response=dp.response._replace(qe=dp.response.qe * 0.5))
        full = self._laser_charge(detector, dp, wavelength_mode=True)
        halved = self._laser_charge(detector, half, wavelength_mode=True)
        assert abs(halved / full - 0.5) < 0.05, (full, halved)


# ── Physics consistency ────────────────────────────────────────────

class TestPhysicsConsistency:
    """Different wavelengths produce physically consistent results."""

    def test_short_wavelength_more_scattering(self, detector, base_dp):
        """At 300 nm (L_scat 42 m) the charge pattern is less concentrated than at 500 nm (504 m).

        Scattering redirects photons out of the beam spot, so the spot's share
        of the total falls. That share -- max(charges) / sum(charges) -- is the
        claim; the old version of this test asserted only that each wavelength
        produced some charge, which is true of every wavelength in the curve
        and distinguishes nothing.
        """
        from lucid.simulation import setup_event_simulator
        from lucid.sources import laser_source

        key = jax.random.PRNGKey(42)
        sim = setup_event_simulator(
            GEOM, 5000, source='light_source', temperature=None, K=6,
            default_detector_params=base_dp,
            wavelength_mode=True)

        charge = {}
        for wl in (300.0, 500.0):
            source = laser_source(position=[0., 0., detector.H / 2 - 0.1],
                                  intensity=1e8, wavelength=wl)
            q, _ = sim(source, key)
            assert float(jnp.sum(q)) > 0, f"no charge at all at {wl} nm"
            charge[wl] = q

        # The two must not be the same array: wavelength has to reach the step.
        assert not bool(jnp.allclose(charge[300.0], charge[500.0])), \
            "300 nm and 500 nm gave identical charge: wavelength_mode did nothing"

        spot = {wl: float(jnp.max(q) / jnp.sum(q)) for wl, q in charge.items()}
        assert spot[300.0] < spot[500.0], (
            f"beam-spot share should be smaller at 300 nm: "
            f"300 nm -> {spot[300.0]:.5f}, 500 nm -> {spot[500.0]:.5f}")


# ── Shipped real-detector configs in wavelength mode ───────────────

CONFIG_DIR = os.path.join(REPO, 'config')
JUNO_GEOM = os.path.join(CONFIG_DIR, 'JUNO_geom_config.json')
JUNO_PHYS = os.path.join(CONFIG_DIR, 'JUNO_physics_config.json')
WCTE_PHYS = os.path.join(CONFIG_DIR, 'WCTE_like_physics_config.json')
SK_GEOM = os.path.join(CONFIG_DIR, 'SK_geom_config.json')
SK_PHYS = os.path.join(CONFIG_DIR, 'SK_physics_config.json')

SMOKE_NPHOT = 5_000
SMOKE_K = 3


class TestShippedConfigsInWavelengthMode:
    """Wavelength mode must survive each shipped physics config's missing fields.

    These four cases were the whole of ``tests/e2e/``, where they printed PASS
    or FAIL through a ``report()`` helper and asserted nothing, so they could
    not fail. They are the only coverage of the JUNO and WCTE config shapes and
    of the SK sensor count, which is why they are here rather than deleted with
    the rest of that directory.
    """

    def test_juno_has_medium_model_but_no_qe_curve(self):
        """JUNO_physics_config.json has `medium_model` and no `qe_curve`."""
        from lucid.simulation import setup_event_simulator
        from lucid.sources import laser_source

        sim = setup_event_simulator(
            JUNO_GEOM, SMOKE_NPHOT, source='light_source', temperature=None, K=SMOKE_K,
            physics_config=JUNO_PHYS, default_detector_params=True,
            wavelength_mode=True)
        charges, _ = sim(laser_source(position=[0., 0., 0.], intensity=1e8,
                                      wavelength=405.0),
                         jax.random.PRNGKey(42))

        assert jnp.all(jnp.isfinite(charges))
        assert float(jnp.sum(charges)) > 0

    def test_wcte_has_no_medium_model_and_falls_back(self):
        """WCTE_like_physics_config.json has neither field; water.json is the fallback."""
        from lucid.simulation import setup_event_simulator
        from lucid.sources import laser_source

        sim = setup_event_simulator(
            GEOM, SMOKE_NPHOT, source='light_source', temperature=None, K=SMOKE_K,
            physics_config=WCTE_PHYS,
            default_detector_params=True, wavelength_mode=True)
        charges, _ = sim(laser_source(position=[0., 0., 1.5], intensity=1e8,
                                      wavelength=405.0),
                         jax.random.PRNGKey(42))

        assert jnp.all(jnp.isfinite(charges))
        assert float(jnp.sum(charges)) > 0

    @pytest.mark.parametrize('wavelength', [200.0, 800.0])
    def test_extreme_wavelengths_stay_finite(self, detector, base_dp, wavelength):
        """200 nm and 800 nm are off the ends of every tabulated curve.

        Charge may legitimately be zero there — the QE curve is zero outside
        its support — so finiteness is the whole claim. A NaN here is an
        extrapolation reaching past the end of a table.
        """
        from lucid.simulation import setup_event_simulator
        from lucid.sources import laser_source

        sim = setup_event_simulator(
            GEOM, SMOKE_NPHOT, source='light_source', temperature=None, K=SMOKE_K,
            default_detector_params=base_dp,
            wavelength_mode=True)
        charges, times = sim(
            laser_source(position=[0., 0., detector.H / 2 - 0.1],
                         intensity=1e8, wavelength=wavelength),
            jax.random.PRNGKey(42))

        assert jnp.all(jnp.isfinite(charges))
        assert jnp.all(jnp.isfinite(times))

    def test_sk_npz_geometry_is_11096_sensors_and_runs(self):
        """The real SK geometry loads 11096 PMTs from config/sk_geometry.npz."""
        from lucid.geometry import generate_detector
        from lucid.simulation import setup_event_simulator
        from lucid.sources import laser_source

        det = generate_detector(SK_GEOM)
        assert len(det.all_points) == 11096

        sim = setup_event_simulator(
            SK_GEOM, SMOKE_NPHOT, source='light_source', temperature=None, K=SMOKE_K,
            physics_config=SK_PHYS, default_detector_params=True,
            wavelength_mode=True)
        charges, _ = sim(laser_source(position=[0., 0., 10.0], intensity=1e8,
                                      wavelength=405.0),
                         jax.random.PRNGKey(42))

        assert charges.shape == (11096,)
        assert jnp.all(jnp.isfinite(charges))
        assert float(jnp.sum(charges)) > 0
