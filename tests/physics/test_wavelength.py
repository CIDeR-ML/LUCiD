"""Tests for the wavelength module (Phase 5)."""
import jax.numpy as jnp
import numpy.testing as npt

import os

from lucid.wavelength.medium import make_medium, load_qe_curve
from tests.helpers.paths import REPO

_SK_QE_PATH = os.path.join(REPO, 'config', 'pmt', 'SK_QE.json')


class TestMediumProperties:
    def test_make_medium_water_scalar(self):
        """n = 1.33 and the medium speed is the PHASE velocity c/n = 0.22541 m/ns (c*n would
        be 0.399). The group velocity (0.2167 m/ns, which lucid/fitting/seed.py uses) is not
        what the simulation propagates at; which one it should be is an open physics
        question pending the owner's decision, so this pins today's choice."""
        m = make_medium("water")
        assert m.material == "water"
        assert abs(m.refractive_index - 1.33) < 1e-6
        assert m.speed_of_light == 0.299792458 / 1.33
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


    def test_unknown_material_raises(self):
        # water/ice/wbls are now real materials (multi-material merge); a material with
        # no config/materials/<name>.json still raises.
        import pytest  # noqa: F811
        with pytest.raises(ValueError, match="No material config"):
            make_medium("unobtainium")

    def test_ice_index(self):
        # Emission dispatch per material is tested in physics/test_scintillation_emission.
        assert make_medium("ice").refractive_index == 1.309

    def test_every_material_states_the_si_vacuum_speed(self):
        # c = 299 792 458 m/s exactly (the SI definition of the metre); light moves at c/n.
        for material in ("water", "ice", "wbls"):
            m = make_medium(material)
            assert m.speed_of_light == 0.299792458 / m.refractive_index, material


class TestQECurve:
    def test_load_default(self):
        """SK's 20-inch PMT QE peaks at ~21 % near 375 nm (Fukuda et al., NIM A 501 (2003)
        418), returned as a fraction, not a percent."""
        qe_fn = load_qe_curve(_SK_QE_PATH)
        assert 0.15 < float(qe_fn(375.0)) < 0.25
        assert jnp.all(qe_fn(jnp.linspace(300.0, 650.0, 50)) <= 1.0)

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

    def test_peak_is_the_curve_maximum(self):
        curve = load_qe_curve(_SK_QE_PATH)
        lo, hi = curve.bounds
        values = curve(jnp.linspace(lo, hi, 2001))
        peak = curve.peak
        assert float(jnp.max(values)) <= peak + 1e-7
        npt.assert_allclose(float(jnp.max(values)), peak, rtol=1e-3)


class TestLaserWavelength:
    """A laser with wavelength= carries it (wavelength mode uses it for every photon)."""

    def test_laser_wavelength_field(self):
        from lucid.sources import laser_source
        assert float(laser_source(position=[0., 0., 0.], wavelength=405.0).wavelength) == 405.0
        assert laser_source(position=[0., 0., 0.]).wavelength is None
