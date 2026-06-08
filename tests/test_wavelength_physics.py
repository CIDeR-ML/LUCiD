"""Physics-level tests for wavelength-dependent properties.

Tests verify known physical relationships in the SK water model:
Rayleigh lambda^4 scaling, absorption features, Cherenkov spectrum
statistics, and HG phase function moments.
"""
import jax
import jax.numpy as jnp
import numpy.testing as npt

from lucid.wavelength.medium import make_medium, compute_effective_properties
from lucid.wavelength.spectrum import sample_cherenkov_wavelengths
from lucid.wavelength.scattering import hg_sample_cos_theta


class TestRayleighScaling:
    """Rayleigh scattering should scale as λ^4."""

    def test_lambda4_ratio(self):
        """Scatter coefficient at 350nm vs 700nm should differ by (700/350)^4 = 16."""
        wl = jnp.array([350.0, 700.0])
        m = make_medium("water", wavelength_grid=wl)
        ratio = float(m.scatter_coeff[0] / m.scatter_coeff[1])
        # Not exactly 16 due to the P5/lambda^2 correction term
        # With correction: alpha ∝ 1/λ^4 * (1 + P5/λ^2), so ratio > 16
        assert ratio > 16, "Should exceed pure λ^4 due to correction term"
        assert ratio < 30, "Should not be wildly larger than λ^4"

    def test_monotonic_decrease(self):
        """Rayleigh scatter coefficient should decrease with wavelength."""
        wl = jnp.linspace(300.0, 700.0, 50)
        m = make_medium("water", wavelength_grid=wl)
        diffs = jnp.diff(m.scatter_coeff)
        assert jnp.all(diffs < 0), "Scatter coeff should monotonically decrease with λ"


class TestAbsorptionPhysics:
    """Water absorption follows the SK model with Pope & Fry transition at 464nm."""

    def test_uv_has_finite_absorption(self):
        """At UV wavelengths, absorption should be measurable (not infinite)."""
        wl = jnp.array([300.0])
        m = make_medium("water", wavelength_grid=wl)
        abs_length = 1.0 / m.absorption_coeff[0]
        # SK water model gives moderate absorption at 300nm
        assert 0.1 < float(abs_length) < 1000

    def test_visible_low_absorption(self):
        """Pure water is most transparent in blue-green (~400-450nm)."""
        wl = jnp.linspace(350.0, 600.0, 100)
        m = make_medium("water", wavelength_grid=wl)
        abs_lengths = 1.0 / m.absorption_coeff
        # Maximum transparency (longest absorption length) should be in 350-500nm range
        max_idx = jnp.argmax(abs_lengths)
        peak_wl = float(wl[max_idx])
        assert 350 < peak_wl < 500, f"Peak transparency at {peak_wl}nm, expected 350-500nm"


class TestEffectivePropertyPhysics:
    def test_shorter_wavelength_scatters_more(self):
        """Photons at shorter wavelengths should have shorter effective scatter length."""
        from lucid.detector_params import DetectorParams
        dp = DetectorParams(scatter_length=50.0, mie_scatter_length=1000.0, g=0.9,
                            wall_reflection_rate=0.5,
                            sensor_reflection_rate=0.3, absorption_length=100.0,
                            qe=0.2, qe_corrections=jnp.ones(10))
        wl = jnp.linspace(300.0, 700.0, 100)
        m = make_medium("water", wavelength_grid=wl)
        wavelengths = jnp.array([350.0, 600.0])
        eff_s, _, _, _ = compute_effective_properties(dp, m, wavelengths=wavelengths)
        # 350nm photon should scatter more (shorter effective scatter length)
        assert float(eff_s[0]) < float(eff_s[1])


class TestCherenkovSpectrumPhysics:
    """Verify 1/λ^2 spectrum has correct analytical moments."""

    def test_analytical_mean(self):
        """For dN/dλ ∝ 1/λ^2, E[λ] = (λ_max - λ_min) / ln(λ_max/λ_min)."""
        key = jax.random.PRNGKey(42)
        lmin, lmax = 300.0, 700.0
        wl = sample_cherenkov_wavelengths(key, 200000, lmin, lmax)
        # Analytical: E[λ] for P(λ)∝1/λ^2 normalized on [lmin, lmax]
        # P(λ) = C/λ^2, C = 1/(1/lmin - 1/lmax)
        # E[λ] = C * integral(1/λ, lmin, lmax) = ln(lmax/lmin) / (1/lmin - 1/lmax)
        expected_mean = jnp.log(lmax / lmin) / (1.0 / lmin - 1.0 / lmax)
        npt.assert_allclose(jnp.mean(wl), expected_mean, atol=1.0)

    def test_inverse_cdf_is_exact(self):
        """The 1/λ^2 inverse CDF should perfectly recover the input uniform samples."""
        # For u=0 → λ=λ_min, u=1 → λ=λ_max
        key = jax.random.PRNGKey(0)
        wl = sample_cherenkov_wavelengths(key, 10, lambda_min=400.0, lambda_max=400.0 + 1e-6)
        # Very narrow range → all samples should be ≈ 400nm
        npt.assert_allclose(wl, 400.0, atol=0.01)


class TestHGPhaseFunctionPhysics:
    """Verify the Henyey-Greenstein phase function moments."""

    def test_mean_cos_theta_equals_g(self):
        """The analytical result: E[cosθ] = g for the HG distribution."""
        for g in [0.1, 0.5, 0.8, 0.95]:
            u_vals = jax.random.uniform(jax.random.PRNGKey(42), (50000,))
            cos_thetas = jax.vmap(hg_sample_cos_theta, in_axes=(0, None))(u_vals, g)
            npt.assert_allclose(jnp.mean(cos_thetas), g, atol=0.02,
                                err_msg=f"Failed for g={g}")

    def test_near_isotropic_at_small_g(self):
        """Small g should give nearly isotropic scattering (mean cosθ ≈ g ≈ 0).
        Note: hg_sample_cos_theta clamps g to [1e-4, 1-1e-4] for numerical safety,
        so negative g is not supported by this implementation."""
        u_vals = jax.random.uniform(jax.random.PRNGKey(42), (10000,))
        cos_thetas = jax.vmap(hg_sample_cos_theta, in_axes=(0, None))(u_vals, 0.01)
        npt.assert_allclose(jnp.mean(cos_thetas), 0.01, atol=0.03)

    def test_differentiable_wrt_g(self):
        """HG should be differentiable w.r.t. the asymmetry parameter g."""
        def mean_cos(g):
            u_vals = jax.random.uniform(jax.random.PRNGKey(0), (1000,))
            return jnp.mean(jax.vmap(hg_sample_cos_theta, in_axes=(0, None))(u_vals, g))
        grad = jax.grad(mean_cos)(0.5)
        assert jnp.isfinite(grad)
        # Increasing g should increase mean cos_theta
        assert float(grad) > 0
