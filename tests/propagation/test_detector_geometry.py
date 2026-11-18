"""Tests for Phase 6 container types."""
import json

import pytest
import jax.numpy as jnp

from lucid.geometry.detector_geometry import DetectorGeometry



@pytest.mark.slow                 # builds detectors
class TestDetectorGeometry:
    def test_from_config_cylinder(self, small_cylinder_config):
        dg = DetectorGeometry.from_config(small_cylinder_config, temperature=0.2)
        # the field holds the value the config carries (shipped configs spell it lowercase)
        assert dg.detector_type == 'cylinder'
        assert dg.num_sensors == len(dg.sensor_points)
        assert dg.sensor_points.shape[1] == 3
        assert dg.speed_of_light == 0.299792458 / 1.33      # c/n, c exact (SI)
        assert dg.medium.material == 'water'
        assert dg.propagator is not None
        assert dg.detector is not None

    def test_bounds_check_via_detector(self, small_cylinder_config):
        dg = DetectorGeometry.from_config(small_cylinder_config)
        pts = jnp.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])
        result = dg.detector.bounds_check(pts)
        assert result[0] == True
        assert result[1] == False

    def test_no_qe_curve_in_geometry(self, small_cylinder_config):
        dg = DetectorGeometry.from_config(small_cylinder_config)
        # QE curve is no longer part of DetectorGeometry — it lives in physics config
        assert not hasattr(dg, 'qe_curve')

    def test_unknown_detector_type_in_config_raises(self, small_cylinder_config, tmp_path):
        cfg = json.loads(open(small_cylinder_config).read())
        cfg['detector_type'] = 'triangle'
        bad = tmp_path / 'triangle_geom_config.json'
        bad.write_text(json.dumps(cfg))
        with pytest.raises(ValueError, match="Unknown detector type 'triangle'"):
            DetectorGeometry.from_config(str(bad))
