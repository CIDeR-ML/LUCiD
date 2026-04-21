"""Tests for the wavelength module (Phase 5)."""
import jax
import jax.numpy as jnp
import numpy.testing as npt

import os

from lucid.wavelength.medium import MediumProperties, make_medium, compute_effective_properties, load_qe_curve
from lucid.wavelength.spectrum import sample_cherenkov_wavelengths

_SK_QE_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'pmt', 'SK_QE.json')
from lucid.wavelength.scattering import (
    compute_rayleigh_scatter_direction,
    hg_sample_cos_theta,
    compute_mie_scatter_direction,
)


class TestMediumProperties:
    def test_make_medium_water_scalar(self):
        m = make_medium("water")
        assert m.material == "water"
        assert abs(m.refractive_index - 1.33) < 1e-6
        npt.assert_allclose(m.speed_of_light, 0.299792 / 1.33, atol=1e-5)
        assert m.wavelength_grid is None
        assert m.scatter_coeff is None

    def test_make_medium_with_wavelength_grid(self):
        wl = jnp.linspace(300.0, 700.0, 50)
        m = make_medium("water", wavelength_grid=wl)
        assert m.wavelength_grid is not None
        assert m.wavelength_grid.shape == (50,)
        assert m.scatter_coeff.shape == (50,)
        assert m.absorption_coeff.shape == (50,)
        assert m.mie_scatter_coeff.shape == (50,)
        assert m.mie_asymmetry == 0.95

    def test_scatter_coeff_positive(self):
        wl = jnp.linspace(300.0, 700.0, 50)
        m = make_medium("water", wavelength_grid=wl)
        assert jnp.all(m.scatter_coeff > 0)
        assert jnp.all(m.absorption_coeff > 0)
        assert jnp.all(m.mie_scatter_coeff > 0)

    def test_scatter_increases_with_shorter_wavelength(self):
        """Rayleigh scattering goes as 1/lambda^4 — shorter λ scatters more."""
        wl = jnp.linspace(300.0, 700.0, 50)
        m = make_medium("water", wavelength_grid=wl)
        # scatter_coeff = alpha (1/m), so larger at shorter wavelengths
        assert float(m.scatter_coeff[0]) > float(m.scatter_coeff[-1])

    def test_unknown_material_raises(self):
        import pytest  # noqa: F811
        with pytest.raises(ValueError, match="Unknown material"):
            make_medium("ice")


class TestEffectiveProperties:
    def test_monochromatic_passthrough(self):
        """With wavelengths=None, effective properties equal scalar inputs."""
        from lucid.detector_params import DetectorParams
        dp = DetectorParams(
            scatter_length=50.0, wall_reflection_rate=0.5,
            sensor_reflection_rate=0.3, absorption_length=100.0,
            qe=0.2, qe_corrections=jnp.ones(10),
        )
        m = make_medium("water")
        eff_s, eff_a, eff_qe = compute_effective_properties(dp, m)
        assert eff_s == 50.0
        assert eff_a == 100.0
        assert eff_qe == 0.2

    def test_wavelength_active_returns_arrays(self):
        from lucid.detector_params import DetectorParams
        dp = DetectorParams(
            scatter_length=50.0, wall_reflection_rate=0.5,
            sensor_reflection_rate=0.3, absorption_length=100.0,
            qe=0.2, qe_corrections=jnp.ones(10),
        )
        wl = jnp.linspace(300.0, 700.0, 100)
        m = make_medium("water", wavelength_grid=wl)
        wavelengths = jnp.array([350.0, 400.0, 500.0, 600.0])
        eff_s, eff_a, eff_qe = compute_effective_properties(dp, m, wavelengths=wavelengths)
        assert eff_s.shape == (4,)
        assert eff_a.shape == (4,)


class TestCherenkovSpectrum:
    def test_basic_sampling(self):
        key = jax.random.PRNGKey(42)
        wl = sample_cherenkov_wavelengths(key, 10000)
        assert wl.shape == (10000,)
        assert jnp.all(wl >= 300.0)
        assert jnp.all(wl <= 700.0)

    def test_distribution_skews_blue(self):
        """Cherenkov 1/lambda^2 means more photons at short wavelengths."""
        key = jax.random.PRNGKey(42)
        wl = sample_cherenkov_wavelengths(key, 100000)
        median = jnp.median(wl)
        midpoint = (300.0 + 700.0) / 2.0
        assert float(median) < midpoint, "Cherenkov spectrum should skew toward shorter wavelengths"

    def test_custom_range(self):
        key = jax.random.PRNGKey(0)
        wl = sample_cherenkov_wavelengths(key, 1000, lambda_min=400.0, lambda_max=500.0)
        assert jnp.all(wl >= 400.0)
        assert jnp.all(wl <= 500.0)


class TestScattering:
    def test_rayleigh_output_is_unit(self):
        key = jax.random.PRNGKey(42)
        inc = jnp.array([0.0, 0.0, 1.0])
        result = compute_rayleigh_scatter_direction(inc, key)
        npt.assert_allclose(jnp.linalg.norm(result), 1.0, atol=1e-5)

    def test_hg_isotropic_at_g_near_zero(self):
        """For g~0, HG reduces to nearly isotropic."""
        keys = jax.random.split(jax.random.PRNGKey(0), 10000)
        u_vals = jax.random.uniform(jax.random.PRNGKey(1), (10000,))
        cos_thetas = jax.vmap(hg_sample_cos_theta, in_axes=(0, None))(u_vals, 0.001)
        # Mean should be near 0 for isotropic
        npt.assert_allclose(jnp.mean(cos_thetas), 0.0, atol=0.05)

    def test_hg_forward_peaked(self):
        """For g=0.95, most scattering is forward (cos_theta near 1)."""
        u_vals = jax.random.uniform(jax.random.PRNGKey(42), (10000,))
        cos_thetas = jax.vmap(hg_sample_cos_theta, in_axes=(0, None))(u_vals, 0.95)
        assert float(jnp.mean(cos_thetas)) > 0.8

    def test_mie_output_is_unit(self):
        key = jax.random.PRNGKey(42)
        inc = jnp.array([1.0, 0.0, 0.0])
        result = compute_mie_scatter_direction(inc, key, g=0.95)
        npt.assert_allclose(jnp.linalg.norm(result), 1.0, atol=1e-5)


class TestQECurve:
    def test_load_default(self):
        qe_fn = load_qe_curve(_SK_QE_PATH)
        # Peak QE around 370-380nm should be ~21%
        peak_qe = qe_fn(375.0)
        assert 0.15 < float(peak_qe) < 0.25

    def test_zero_outside_range(self):
        qe_fn = load_qe_curve(_SK_QE_PATH)
        assert float(qe_fn(200.0)) == 0.0
        assert float(qe_fn(800.0)) == 0.0

    def test_vectorized(self):
        qe_fn = load_qe_curve(_SK_QE_PATH)
        wls = jnp.array([300.0, 400.0, 500.0, 600.0])
        result = qe_fn(wls)
        assert result.shape == (4,)
        # QE should decrease from blue to red
        assert float(result[1]) > float(result[3])

    def test_returns_fraction_not_percent(self):
        qe_fn = load_qe_curve(_SK_QE_PATH)
        # All values should be < 1.0 (fractions, not percent)
        wls = jnp.linspace(300.0, 650.0, 50)
        result = qe_fn(wls)
        assert jnp.all(result <= 1.0)
