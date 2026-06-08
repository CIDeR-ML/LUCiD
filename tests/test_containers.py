"""Tests for Phase 6 container types."""
import pytest
import jax.numpy as jnp
import numpy.testing as npt

from lucid.simulation.config import SimConfig
from lucid.geometry.detector_geometry import DetectorGeometry

pytestmark = pytest.mark.slow


class TestSimConfig:
    def test_defaults(self):
        cfg = SimConfig()
        assert cfg.n_photons == 1_000_000
        assert cfg.K == 7
        assert cfg.mode == 'track'
        assert cfg.use_expected_value is True

    def test_track_mode_grad_iters(self):
        # Track mode lets direction gradient flow all K bounces; the
        # reflection-normal curvature issue that used to require n_grad_iters=0
        # is now fixed at the normal level.
        cfg = SimConfig(mode='track')
        assert cfg.effective_n_grad_iters == cfg.K

    def test_calibration_mode_grad_iters(self):
        # Calibration mode now lets the direction gradient flow through all K
        # bounces (same as track mode); the old n_grad_iters=2 cap was removed
        # after the reflection-normal curvature fix.
        cfg = SimConfig(mode='calibration')
        assert cfg.effective_n_grad_iters == cfg.K

    def test_data_mode_grad_iters(self):
        cfg = SimConfig(mode='data')
        assert cfg.effective_n_grad_iters == 0

    def test_explicit_override(self):
        cfg = SimConfig(mode='track', n_grad_iters=5)
        assert cfg.effective_n_grad_iters == 5

    def test_is_data(self):
        assert SimConfig(mode='data').is_data is True
        assert SimConfig(mode='track').is_data is False

    def test_is_calibration(self):
        assert SimConfig(mode='calibration').is_calibration is True
        assert SimConfig(mode='track').is_calibration is False


class TestDetectorGeometry:
    def test_from_config_cylinder(self, small_cylinder_config):
        dg = DetectorGeometry.from_config(
            small_cylinder_config, temperature=0.2,
            detector_type='Cylinder')
        assert dg.detector_type == 'Cylinder'
        assert dg.num_sensors == len(dg.sensor_points)
        assert dg.sensor_points.shape[1] == 3
        npt.assert_allclose(dg.speed_of_light, 0.299792 / 1.33, atol=1e-5)
        assert dg.medium.material == 'water'
        assert dg.propagator is not None
        assert dg.detector is not None

    def test_bounds_check_via_detector(self, small_cylinder_config):
        dg = DetectorGeometry.from_config(small_cylinder_config, detector_type='Cylinder')
        pts = jnp.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])
        result = dg.detector.bounds_check(pts)
        assert result[0] == True
        assert result[1] == False

    def test_no_qe_curve_in_geometry(self, small_cylinder_config):
        dg = DetectorGeometry.from_config(small_cylinder_config, detector_type='Cylinder')
        # QE curve is no longer part of DetectorGeometry — it lives in physics config
        assert not hasattr(dg, 'qe_curve')

    def test_invalid_detector_type(self, small_cylinder_config):
        import pytest
        with pytest.raises(ValueError, match="detector_type"):
            DetectorGeometry.from_config(small_cylinder_config, detector_type='Triangle')
