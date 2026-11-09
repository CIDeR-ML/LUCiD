"""Tests for the per-photon optical model seam (lucid.wavelength.optical_model).

These lock the two regimes of ``evaluate_optical_model`` and the exact lengths it
must reproduce so the simulator's ``_get_optical_arrays`` stays byte-identical to the
pre-extraction inline code.
"""
import os

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

import jax

from lucid.wavelength.medium import make_medium, load_qe_curve
from lucid.wavelength.optical_model import (
    evaluate_optical_model, OpticalArrays, CONTROL_WAVELENGTHS_NM, N_CONTROL,
)

from tests.helpers.params import lab_dp
from tests.helpers.paths import REPO

_SK_QE_PATH = os.path.join(REPO, 'config', 'pmt', 'SK_QE.json')


class TestScalarMode:
    def test_broadcasts_scalars(self):
        """wavelengths=None → DetectorParams scalars broadcast to (n,), qe None."""
        dp = lab_dp()
        m = make_medium("water")
        n = 7
        oa = evaluate_optical_model(dp, None, m, n)
        assert isinstance(oa, OpticalArrays)
        assert oa.scatter_len.shape == (n,)
        assert oa.mie_len.shape == (n,)
        assert oa.abs_len.shape == (n,)
        npt.assert_allclose(oa.scatter_len, 50.0)
        npt.assert_allclose(oa.mie_len, 1000.0)
        npt.assert_allclose(oa.abs_len, 100.0)
        assert oa.qe is None


class TestWavelengthMode:
    def test_lengths_are_inverse_medium_coeff(self):
        """length(λ) = 1 / coeff(λ); matches the medium reference exactly."""
        dp = lab_dp()
        wl_grid = jnp.linspace(300.0, 700.0, 200)
        m = make_medium("water", wavelength_grid=wl_grid)
        wavelengths = jnp.array([350.0, 400.0, 500.0, 600.0])
        oa = evaluate_optical_model(dp, wavelengths, m, wavelengths.shape[0])

        sc = jnp.interp(wavelengths, m.wavelength_grid, m.scatter_coeff)
        ac = jnp.interp(wavelengths, m.wavelength_grid, m.absorption_coeff)
        asym = jnp.interp(wavelengths, m.wavelength_grid, m.mie_scatter_coeff)
        npt.assert_allclose(oa.scatter_len, 1.0 / (sc + 1e-30), rtol=1e-6)
        npt.assert_allclose(oa.abs_len, 1.0 / (ac + 1e-30), rtol=1e-6)
        npt.assert_allclose(oa.mie_len, 1.0 / (asym + 1e-30), rtol=1e-6)
        assert oa.qe is None

    def test_clamps_to_grid(self):
        """λ outside the grid is clamped to the grid endpoints (no extrapolation)."""
        dp = lab_dp()
        wl_grid = jnp.linspace(300.0, 700.0, 200)
        m = make_medium("water", wavelength_grid=wl_grid)
        oa_lo = evaluate_optical_model(dp, jnp.array([100.0]), m, 1)
        oa_edge = evaluate_optical_model(dp, jnp.array([300.0]), m, 1)
        npt.assert_allclose(oa_lo.scatter_len, oa_edge.scatter_len, rtol=1e-6)

    def test_qe_fn_applied_per_photon(self):
        """A qe_fn is evaluated at the (clamped) per-photon wavelengths."""
        dp = lab_dp()
        wl_grid = jnp.linspace(300.0, 700.0, 200)
        m = make_medium("water", wavelength_grid=wl_grid)
        qe_fn = load_qe_curve(_SK_QE_PATH)
        wavelengths = jnp.array([350.0, 400.0, 500.0])
        oa = evaluate_optical_model(dp, wavelengths, m, 3, qe_fn=qe_fn)
        assert oa.qe is not None
        wl_clamped = jnp.clip(wavelengths, m.wavelength_grid[0], m.wavelength_grid[-1])
        npt.assert_allclose(oa.qe, qe_fn(wl_clamped), rtol=1e-6)


class TestDeviationCurves:
    def test_default_curves_are_unit(self):
        """Default DetectorParams carry all-ones (≡1) λ-deviation curves."""
        dp = lab_dp()
        assert dp.scattering.rayleigh_dev.shape == (N_CONTROL,)
        npt.assert_allclose(dp.scattering.rayleigh_dev, 1.0)
        npt.assert_allclose(dp.scattering.mie_dev, 1.0)
        npt.assert_allclose(dp.absorption.abs_dev, 1.0)
        npt.assert_allclose(dp.response.qe_dev, 1.0)

    # One distinct, non-uniform curve per property, so a knot misalignment or a swap of two
    # properties' curves changes the answer.
    _DEV = {'rayleigh_dev': (1.5, 0.8, 2.0, 1.2, 0.6), 'mie_dev': (0.7, 1.9, 1.1, 0.5, 1.4),
            'abs_dev': (1.3, 0.9, 0.6, 2.2, 1.0), 'qe_dev': (0.9, 1.2, 0.7, 1.1, 1.6)}

    def _deviated(self):
        base = lab_dp()
        dev = {k: jnp.asarray(v) for k, v in self._DEV.items()}
        return base._replace(
            scattering=base.scattering._replace(rayleigh_dev=dev['rayleigh_dev'],
                                                mie_dev=dev['mie_dev']),
            absorption=base.absorption._replace(abs_dev=dev['abs_dev']),
            response=base.response._replace(qe_dev=dev['qe_dev']))

    def test_each_curve_scales_its_own_property_at_and_between_knots(self):
        """length = reference / dev(lambda) with dev linear between the control knots: at
        knot k it is curve[k], midway between knots k and k+1 it is their mean. Rayleigh,
        Mie and absorption lengths divide by their own curves; QE multiplies by qe_dev."""
        c = np.asarray(CONTROL_WAVELENGTHS_NM)
        lam = np.concatenate([c, 0.5 * (c[:-1] + c[1:])])
        expected_dev = {k: np.concatenate([np.asarray(v), 0.5 * (np.asarray(v)[:-1] + np.asarray(v)[1:])])
                        for k, v in self._DEV.items()}
        m = make_medium("water", wavelength_grid=jnp.linspace(300.0, 700.0, 200))
        qe_fn = load_qe_curve(_SK_QE_PATH)
        ref = evaluate_optical_model(lab_dp(), jnp.asarray(lam), m, len(lam), qe_fn=qe_fn)
        got = evaluate_optical_model(self._deviated(), jnp.asarray(lam), m, len(lam), qe_fn=qe_fn)
        npt.assert_allclose(got.scatter_len, np.asarray(ref.scatter_len) / expected_dev['rayleigh_dev'], rtol=1e-5)
        npt.assert_allclose(got.mie_len, np.asarray(ref.mie_len) / expected_dev['mie_dev'], rtol=1e-5)
        npt.assert_allclose(got.abs_len, np.asarray(ref.abs_len) / expected_dev['abs_dev'], rtol=1e-5)
        npt.assert_allclose(got.qe, np.asarray(ref.qe) * expected_dev['qe_dev'], rtol=1e-5)

    def test_gradient_wrt_the_deviation_knots_is_the_closed_form(self):
        """d/d dev_k of sum_lambda abs_len(lambda), at dev = 1, is -sum_lambda ref(lambda)
        w_k(lambda), with ref = 1/coeff(lambda) and w_k the linear-interpolation weight of knot k
        at lambda (abs_len = ref / sum_k w_k dev_k).

        The weights are taken by interpolating each unit vector over the knots in NumPy, so
        a curve applied to the wrong knots, or a deviation read from another property, gives
        another gradient. The wavelengths leave the 398 nm knot untouched (360 nm sits between
        337 and 375, 440 nm between 405 and 445, and 530 nm clamps to 445), so its derivative
        is exactly zero.
        """
        wl_grid = jnp.linspace(300.0, 700.0, 200)
        m = make_medium("water", wavelength_grid=wl_grid)
        lam = jnp.array([360.0, 440.0, 530.0])

        def loss(dev):
            dp = lab_dp()._replace(
                absorption=lab_dp().absorption._replace(abs_dev=dev))
            oa = evaluate_optical_model(dp, lam, m, len(lam))
            return jnp.sum(oa.abs_len)

        g = np.asarray(jax.grad(loss)(jnp.ones(N_CONTROL)), np.float64)
        ref = np.asarray(evaluate_optical_model(lab_dp(), lam, m, len(lam)).abs_len, np.float64)
        knots = np.asarray(CONTROL_WAVELENGTHS_NM, np.float64)
        w = np.stack([np.interp(np.asarray(lam), knots, np.eye(N_CONTROL)[k])
                      for k in range(N_CONTROL)])                     # (N_CONTROL, n_lambda)
        npt.assert_allclose(g, -(w @ ref), rtol=1e-5)
        assert g[2] == 0.0 and np.all(g[[0, 1, 3, 4]] < 0)
