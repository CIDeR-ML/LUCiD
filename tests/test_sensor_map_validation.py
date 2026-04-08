"""Tests for sensor map validation and grid auto-derivation.

Verifies that configure_grid() produces adequate grids and that
validate_sensor_map catches problems.
"""
import warnings
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.geometry import generate_detector
from lucid.propagation.shared import create_propagator, validate_sensor_map


class TestAutoGridDerivation:
    """Verify configure_grid() produces sensible defaults per geometry."""

    def test_cylinder_auto_grid(self):
        det = generate_detector("config/WCTE_geom_config.json")
        det.configure_grid()
        n_sensors = len(det.all_points)
        total_cells = det._n_angular * det._n_height + 2 * det._n_cap**2
        # Should be roughly 1:1 cells to sensors
        ratio = total_cells / n_sensors
        assert 0.5 < ratio < 3.0, f"Cell/sensor ratio {ratio:.1f} outside [0.5, 3.0]"

    def test_sphere_auto_grid(self):
        det = generate_detector("config/JUNO_geom_config.json")
        det.configure_grid()
        n_sensors = len(det.all_points)
        total_cells = det._n_divisions * 2 * det._n_divisions
        ratio = total_cells / n_sensors
        assert 0.5 < ratio < 3.0, f"Cell/sensor ratio {ratio:.1f} outside [0.5, 3.0]"

    def test_box_auto_grid(self):
        det = generate_detector("config/nuSCOPE_geom_config.json")
        det.configure_grid()
        n_sensors = len(det.all_points)
        total_cells = det.total_grid_cells()
        ratio = total_cells / n_sensors
        assert 0.5 < ratio < 3.0, f"Cell/sensor ratio {ratio:.1f} outside [0.5, 3.0]"

    def test_explicit_overrides_auto(self):
        """Explicit grid params should override auto-derivation."""
        det = generate_detector("config/WCTE_geom_config.json")
        det.configure_grid(n_cap=10, n_angular=20, n_height=15)
        assert det._n_cap == 10
        assert det._n_angular == 20
        assert det._n_height == 15

    def test_partial_override(self):
        """Only specified params override; others auto-derive."""
        det = generate_detector("config/WCTE_geom_config.json")
        det.configure_grid(n_angular=100)
        assert det._n_angular == 100
        assert det._n_cap is not None  # auto-derived
        assert det._n_height is not None  # auto-derived


class TestValidationCatchesProblems:
    """Verify validate_sensor_map warns about real issues."""

    def test_no_warnings_with_good_grid(self):
        """Well-configured grid should produce no warnings."""
        det = generate_detector("config/WCTE_geom_config.json")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prop = create_propagator(
                det, jnp.array(det.all_points), det.S_radius,
                n_cap=150, n_angular=250, n_height=150)
            sensor_warnings = [x for x in w if 'Sensor map' in str(x.message)]
            assert len(sensor_warnings) == 0, \
                f"Unexpected warnings: {[str(x.message) for x in sensor_warnings]}"

    def test_warns_on_overcrowding(self):
        """Very coarse grid should warn about overcrowding."""
        det = generate_detector("config/WCTE_geom_config.json")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Very coarse grid → many sensors per cell
            prop = create_propagator(
                det, jnp.array(det.all_points), det.S_radius,
                n_cap=5, n_angular=5, n_height=5,
                max_sensors_per_cell=2)
            sensor_warnings = [x for x in w if 'Sensor map' in str(x.message)]
            assert len(sensor_warnings) > 0, "Should warn about overcrowding"

    def test_warns_on_missing_sensors(self):
        """Coarse grid with low max_sensors_per_cell should lose sensors."""
        det = generate_detector("config/WCTE_geom_config.json")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prop = create_propagator(
                det, jnp.array(det.all_points), det.S_radius,
                n_cap=5, n_angular=5, n_height=5,
                max_sensors_per_cell=1)
            msgs = [str(x.message) for x in w if 'Sensor map' in str(x.message)]
            has_visibility_warning = any('do not appear' in m for m in msgs)
            has_overflow_warning = any('overflow' in m or 'Increase' in m for m in msgs)
            assert has_visibility_warning or has_overflow_warning


class TestAutoGridWithPropagator:
    """Verify auto-derived grids produce working propagators."""

    def test_cylinder_auto_propagator_runs(self):
        det = generate_detector("config/WCTE_geom_config.json")
        prop = create_propagator(det, jnp.array(det.all_points), det.S_radius)
        result = prop(jnp.zeros((2, 3)), jnp.array([[1., 0., 0.], [0., 1., 0.]]))
        assert 'sensor_weights' in result
        assert jnp.any(result['sensor_weights'] > 0)

    def test_sphere_auto_propagator_runs(self):
        det = generate_detector("config/JUNO_geom_config.json")
        prop = create_propagator(det, jnp.array(det.all_points), det.S_radius)
        result = prop(jnp.zeros((2, 3)), jnp.array([[1., 0., 0.], [0., 0., 1.]]))
        assert 'sensor_weights' in result

    def test_box_auto_propagator_runs(self):
        det = generate_detector("config/nuSCOPE_geom_config.json")
        prop = create_propagator(det, jnp.array(det.all_points), det.S_radius)
        result = prop(jnp.zeros((2, 3)), jnp.array([[1., 0., 0.], [0., 1., 0.]]))
        assert 'sensor_weights' in result
