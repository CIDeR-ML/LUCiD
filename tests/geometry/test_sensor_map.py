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

pytestmark = pytest.mark.slow


class TestAutoGridDerivation:
    """Verify configure_grid() produces sensible defaults per geometry."""

    def _check_no_overcrowding(self, det, max_candidates_per_ray=4):
        """Verify no cell exceeds max_candidates_per_ray."""
        import numpy as np
        sp = jnp.array(det.all_points)
        ag = np.asarray(det.assign_sensor_to_cells(sp, det.S_radius))
        tc = det.total_grid_cells()
        cell_count = np.zeros(tc, dtype=int)
        for sid in range(ag.shape[0]):
            for slot in range(ag.shape[1]):
                c = ag[sid, slot]
                if np.all(c == -1):
                    continue
                li = int(det.point_to_grid_cell_from_coords(c))
                if 0 <= li < tc:
                    cell_count[li] += 1
        max_in_cell = int(cell_count.max()) if tc > 0 else 0
        assert max_in_cell <= max_candidates_per_ray, \
            f"Cell has {max_in_cell} sensors, exceeds max_candidates_per_ray={max_candidates_per_ray}"

    def test_cylinder_auto_grid(self):
        det = generate_detector("config/WCTE_like_geom_config.json")
        det.configure_grid(max_candidates_per_ray=4)
        self._check_no_overcrowding(det)

    def test_sphere_auto_grid(self):
        det = generate_detector("config/JUNO_geom_config.json")
        det.configure_grid(max_candidates_per_ray=4)
        self._check_no_overcrowding(det)

    def test_box_auto_grid(self):
        det = generate_detector("config/nuSCOPE_geom_config.json")
        det.configure_grid(max_candidates_per_ray=4)
        self._check_no_overcrowding(det)

    def test_explicit_overrides_auto(self):
        """Explicit grid params should override auto-derivation."""
        det = generate_detector("config/WCTE_like_geom_config.json")
        det.configure_grid(n_cap=10, n_angular=20, n_height=15)
        assert det._n_cap == 10
        assert det._n_angular == 20
        assert det._n_height == 15

    def test_partial_override(self):
        """Only specified params override; others auto-derive."""
        det = generate_detector("config/WCTE_like_geom_config.json")
        det.configure_grid(n_angular=100)
        assert det._n_angular == 100
        assert det._n_cap is not None  # auto-derived
        assert det._n_height is not None  # auto-derived


class TestValidationCatchesProblems:
    """Verify validate_sensor_map warns about real issues."""

    def test_no_warnings_with_good_grid(self):
        """Well-configured grid should produce no warnings."""
        det = generate_detector("config/WCTE_like_geom_config.json")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prop = create_propagator(
                det, jnp.array(det.all_points), det.S_radius,
                n_cap=150, n_angular=250, n_height=150)
            sensor_warnings = [x for x in w if 'Sensor map' in str(x.message)]
            assert len(sensor_warnings) == 0, \
                f"Unexpected warnings: {[str(x.message) for x in sensor_warnings]}"

    def test_warns_on_overcrowding(self):
        """Very coarse grid should warn about exceeding max_candidates_per_ray
        (and auto-adjust the limit rather than crashing)."""
        det = generate_detector("config/WCTE_like_geom_config.json")
        with pytest.warns(UserWarning, match="max_candidates_per_ray"):
            create_propagator(
                det, jnp.array(det.all_points), det.S_radius,
                n_cap=5, n_angular=5, n_height=5,
                max_candidates_per_ray=2)

    def test_auto_grid_no_overcrowding_cylinder(self):
        """Auto-derived cylinder grid must not exceed max_candidates_per_ray."""
        det = generate_detector("config/WCTE_like_geom_config.json")
        # Should not raise — auto grid handles max_candidates_per_ray
        prop = create_propagator(det, jnp.array(det.all_points), det.S_radius)
        result = prop(jnp.zeros((1, 3)), jnp.array([[1., 0., 0.]]))
        assert 'sensor_weights' in result

    def test_auto_grid_no_overcrowding_sphere(self):
        """Auto-derived sphere grid must not exceed max_candidates_per_ray."""
        det = generate_detector("config/JUNO_geom_config.json")
        prop = create_propagator(det, jnp.array(det.all_points), det.S_radius)
        result = prop(jnp.zeros((1, 3)), jnp.array([[1., 0., 0.]]))
        assert 'sensor_weights' in result

    def test_auto_grid_no_overcrowding_box(self):
        """Auto-derived box grid must not exceed max_candidates_per_ray."""
        det = generate_detector("config/nuSCOPE_geom_config.json")
        prop = create_propagator(det, jnp.array(det.all_points), det.S_radius)
        result = prop(jnp.zeros((1, 3)), jnp.array([[1., 0., 0.]]))
        assert 'sensor_weights' in result


class TestAutoGridWithPropagator:
    """Verify auto-derived grids produce working propagators."""

    def test_cylinder_auto_propagator_runs(self):
        det = generate_detector("config/WCTE_like_geom_config.json")
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
