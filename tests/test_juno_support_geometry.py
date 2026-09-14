"""Deterministic tests for the JUNO support-rod geometry."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from lucid.geometry.detector import generate_detector
from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.geometry.juno_support import (
    SupportRodLayout,
    build_support_candidate_map,
    intersect_capped_cylinders,
    load_juno_support_layout,
)
from lucid.simulation.photon_step_factory import make_photon_step
from lucid.simulation.reflection import ScalarReflection


def _one_rod():
    """Return one radial rod centred on the positive x axis."""
    return SupportRodLayout(
        copy_numbers=np.array([0], dtype=np.int32),
        family_codes=np.array([0], dtype=np.int8),
        centers_m=np.array([[5.0, 0.0, 0.0]]),
        inward_axes=np.array([[-1.0, 0.0, 0.0]]),
        radii_m=np.array([0.1]),
        half_lengths_m=np.array([1.0]),
    )


def test_axis_and_side_rays_hit_capped_cylinder():
    """Both end-cap and barrel intersections return the nearest surface."""
    layout = _one_rod()
    candidates = jnp.array([[0]], dtype=jnp.int32)

    cap = intersect_capped_cylinders(
        jnp.array([[0.0, 0.0, 0.0]]),
        jnp.array([[1.0, 0.0, 0.0]]),
        layout,
        candidates,
    )
    side = intersect_capped_cylinders(
        jnp.array([[5.0, 1.0, 0.0]]),
        jnp.array([[0.0, -1.0, 0.0]]),
        layout,
        candidates,
    )

    npt.assert_allclose(cap.distance_m, [4.0], atol=1e-6)
    npt.assert_allclose(side.distance_m, [0.9], atol=1e-6)
    npt.assert_array_equal(cap.valid, [True])
    npt.assert_array_equal(side.valid, [True])
    # The transport normal is defined so -normal points into surrounding water.
    npt.assert_allclose(cap.transport_normals[0], [1.0, 0.0, 0.0])
    npt.assert_allclose(side.transport_normals[0], [0.0, -1.0, 0.0])


def test_miss_and_geometry_gradient_are_well_defined():
    """A miss is flagged and an off-axis side distance is differentiable."""
    layout = _one_rod()
    candidates = jnp.array([[0]], dtype=jnp.int32)
    miss = intersect_capped_cylinders(
        jnp.array([[0.0, 1.0, 0.0]]),
        jnp.array([[1.0, 0.0, 0.0]]),
        layout,
        candidates,
    )
    assert not bool(miss.valid[0])

    def distance(offset):
        result = intersect_capped_cylinders(
            jnp.array([[5.0, 1.0, offset]]),
            jnp.array([[0.0, -1.0, 0.0]]),
            layout,
            candidates,
        )
        return result.distance_m[0]

    derivative = jax.grad(distance)(jnp.asarray(0.03))
    assert bool(jnp.isfinite(derivative))


def test_checked_in_layout_matches_junosw_j26():
    """The portable table retains both J26.3.2 support families."""
    layout = load_juno_support_layout("config/JUNO_support_rods.csv")

    assert len(layout.copy_numbers) == 590
    npt.assert_array_equal(
        np.bincount(layout.family_codes), np.array([370, 220])
    )
    npt.assert_allclose(
        np.unique(layout.radii_m), [0.0300, 0.0425], atol=1e-12
    )
    npt.assert_allclose(
        np.linalg.norm(layout.inward_axes, axis=1), 1.0, atol=1e-12
    )
    radial_dot = np.sum(layout.centers_m * layout.inward_axes, axis=1)
    assert np.all(radial_dot < 0.0)


def test_candidate_map_contains_nearest_rod():
    """Each support-centre direction can retrieve its own rod candidate."""
    layout = _one_rod()
    candidates = np.asarray(
        build_support_candidate_map(layout, 8, 2)
    )
    assert candidates.shape == (128, 2)
    assert np.all(np.any(candidates == 0, axis=1))


def test_detector_loader_support_on_off(tmp_path):
    """The support override is independent of the configured PMT geometry."""
    positions = tmp_path / "pmts.csv"
    positions.write_text(
        "#CopyNo X Y Z Orientation_theta Orientation_phi\n"
        "0 0 0 10000 0 0\n",
        encoding="utf-8",
    )
    supports = tmp_path / "supports.csv"
    supports.write_text(
        "copy_no,family,center_x_m,center_y_m,center_z_m,"
        "radius_m,half_length_m\n"
        "0,main,5,0,0,0.1,1\n",
        encoding="utf-8",
    )
    config = {
        "material": "water",
        "detector_type": "nested_sphere",
        "geometry_definitions": {
            "inner_radius": 8.0,
            "outer_radius": 10.0,
            "n_sensors": 1,
            "sensor_radius": 0.254,
            "pmt_file_path": positions.name,
            "support_rod_file_path": supports.name,
            "support_rod_geometry": "juno_radial_cylinders",
        },
    }
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    enabled = generate_detector(path, support_geometry_enabled=True)
    disabled = generate_detector(path, support_geometry_enabled=False)

    assert enabled.support_geometry == "juno_radial_cylinders"
    assert enabled.support_rods is not None
    assert disabled.support_geometry is None
    assert disabled.support_rods is None


def test_support_is_selected_before_outer_surface(tmp_path):
    """A water-side ray sees the rod before the outer detector boundary."""
    positions = tmp_path / "pmts.csv"
    positions.write_text(
        "#CopyNo X Y Z Orientation_theta Orientation_phi\n"
        "0 0 0 10000 0 0\n",
        encoding="utf-8",
    )
    supports = tmp_path / "supports.csv"
    supports.write_text(
        "copy_no,family,center_x_m,center_y_m,center_z_m,"
        "radius_m,half_length_m\n"
        "0,main,9,0,0,0.1,0.5\n",
        encoding="utf-8",
    )
    config = {
        "material": "water",
        "inner_material": "water",
        "outer_material": "water",
        "detector_type": "nested_sphere",
        "geometry_definitions": {
            "inner_radius": 8.0,
            "outer_radius": 10.0,
            "n_sensors": 1,
            "sensor_radius": 0.254,
            "pmt_file_path": positions.name,
            "support_rod_file_path": supports.name,
            "support_rod_geometry": "juno_radial_cylinders",
        },
    }
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    geometry = DetectorGeometry.from_config(
        str(path), detector_type="nested_sphere", temperature=None,
        max_candidates_per_ray=1, n_divisions=8,
    )

    result = geometry.propagator(
        jnp.array([[8.2, 0.0, 0.0]]),
        jnp.array([[1.0, 0.0, 0.0]]),
    )

    npt.assert_array_equal(result["hit_support"], [True])
    npt.assert_array_equal(result["support_index"], [0])
    npt.assert_allclose(result["positions"], [[8.5, 0.0, 0.0]], atol=1e-5)
    npt.assert_allclose(result["normals"], [[1.0, 0.0, 0.0]], atol=1e-6)
    assert not bool(result["hit_interface"][0])
    assert float(jnp.sum(result["sensor_weights"])) == 0.0


def test_support_step_reports_reflection_branch():
    """An opaque support hit reports mutually consistent surface counts."""
    step = make_photon_step(
        "sample", has_interface=True, multi_interface=True,
        has_support=True,
    )
    result = step(
        jnp.array([0.0, 0.0, 0.0]),
        jnp.array([1.0, 0.0, 0.0]),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.array([1.0, 0.0, 0.0]),
        jnp.asarray(1e12),
        jnp.asarray(1e12),
        jnp.asarray(0.0),
        ScalarReflection(jnp.asarray(0.0), jnp.asarray(0.0)),
        jnp.asarray(1e12),
        jnp.asarray(False),
        jnp.asarray(532.0),
        jax.random.PRNGKey(17),
        jnp.asarray(0.225),
        jnp.asarray(False),
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray(1.5),
        jnp.asarray(1.33),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray([1.5, 1.49, 1.33]),
        jnp.asarray(True),
        jnp.asarray(1.0),
        jnp.asarray(1.0),
    )

    events = np.asarray(result[8], dtype=np.int32)
    npt.assert_array_equal(events, [1, 0, 1, 1, 0])
    assert float(result[5]) == 1.0
    assert float(result[1][0]) < 0.0
