"""Deterministic tests for JUNO chimney and Acrylic-bowl geometry."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from lucid.geometry.detector import generate_detector
from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.geometry.juno_structures import (
    AcrylicBowlLayout,
    BOWL_INTERFACE_OFFSET_M,
    ChimneyGeometry,
    intersect_acrylic_bowls,
    intersect_chimney,
    load_juno_acrylic_bowls,
)
from lucid.simulation.photon_step_factory import make_photon_step
from lucid.simulation.reflection import ScalarReflection


def test_checked_in_bowls_match_junosw_j26():
    """The portable table retains all J26.3.2 Acrylic additions."""
    bowls = load_juno_acrylic_bowls(
        "config/JUNO_acrylic_bowls.csv"
    )

    assert len(bowls.copy_numbers) == 590
    npt.assert_allclose(
        np.linalg.norm(bowls.attachment_points_m, axis=1),
        17.824,
        atol=2e-6,
    )
    npt.assert_allclose(np.unique(bowls.base_radii_m), [0.45])
    npt.assert_allclose(np.unique(bowls.tip_radii_m), [0.20])
    npt.assert_allclose(np.unique(bowls.lengths_m), [0.14])


def test_axis_ray_hits_bowl_tip_and_inside_ray_is_ignored():
    """The effective frustum is hit once, then treated as a thin surface."""
    bowls = AcrylicBowlLayout(
        copy_numbers=np.array([0], dtype=np.int32),
        attachment_points_m=np.array([[5.0, 0.0, 0.0]]),
        outward_axes=np.array([[1.0, 0.0, 0.0]]),
        base_radii_m=np.array([0.4]),
        tip_radii_m=np.array([0.2]),
        lengths_m=np.array([1.0]),
    )
    candidates = jnp.array([[0]], dtype=jnp.int32)
    hit = intersect_acrylic_bowls(
        jnp.array([[7.0, 0.0, 0.0]]),
        jnp.array([[-1.0, 0.0, 0.0]]),
        bowls,
        candidates,
    )
    inside = intersect_acrylic_bowls(
        jnp.array([[5.5, 0.0, 0.0]]),
        jnp.array([[1.0, 0.0, 0.0]]),
        bowls,
        candidates,
    )

    npt.assert_allclose(hit.distance_m, [1.0], atol=1e-6)
    npt.assert_array_equal(hit.valid, [True])
    npt.assert_allclose(hit.transport_normals[0], [-1.0, 0.0, 0.0])
    assert not bool(inside.valid[0])

    after_sphere = intersect_acrylic_bowls(
        jnp.array([[5.0001, 0.0, 0.0]]),
        jnp.array([[1.0, 0.0, 0.0]]),
        bowls,
        candidates,
        surface_offset_m=BOWL_INTERFACE_OFFSET_M,
    )
    npt.assert_allclose(after_sphere.distance_m, [0.0004], atol=1e-7)


def test_chimney_inner_wall_and_material_code():
    """A bore-side ray hits the Acrylic lower-chimney wall."""
    chimney = ChimneyGeometry.junosw_j26(outer_limit_m=19.434)
    hit = intersect_chimney(
        jnp.array([[0.0, 0.0, 18.0]]),
        jnp.array([[1.0, 0.0, 0.0]]),
        chimney,
    )

    npt.assert_allclose(hit.distance_m, [0.4], atol=1e-6)
    npt.assert_array_equal(hit.valid, [True])
    npt.assert_array_equal(hit.surface_codes, [1])
    npt.assert_allclose(hit.transport_normals[0], [1.0, 0.0, 0.0])


def _write_minimal_config(tmp_path):
    """Write a one-PMT nested detector with both new components."""
    positions = tmp_path / "pmts.csv"
    positions.write_text(
        "#CopyNo X Y Z Orientation_theta Orientation_phi\n"
        "0 0 0 19434 0 0\n",
        encoding="utf-8",
    )
    bowls = tmp_path / "bowls.csv"
    bowls.write_text(
        "copy_no,x_m,y_m,z_m,base_radius_m,tip_radius_m,length_m\n"
        "0,17.824,0,0,0.4,0.2,1.0\n",
        encoding="utf-8",
    )
    config = {
        "material": "water",
        "inner_material": "water",
        "acrylic_material": "water",
        "outer_material": "water",
        "detector_type": "nested_sphere",
        "geometry_definitions": {
            "inner_radius": 17.7,
            "acrylic_radius": 17.824,
            "outer_radius": 19.434,
            "n_sensors": 1,
            "sensor_radius": 0.254,
            "pmt_file_path": positions.name,
            "chimney_geometry": "juno_lower_chimney",
            "acrylic_bowl_file_path": bowls.name,
            "acrylic_bowl_geometry": "juno_addition_frusta",
        },
    }
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_detector_loader_independent_structure_overrides(tmp_path):
    """Chimney and bowl switches do not alter one another."""
    path = _write_minimal_config(tmp_path)

    chimney_only = generate_detector(
        path,
        chimney_geometry_enabled=True,
        acrylic_bowl_geometry_enabled=False,
    )
    bowls_only = generate_detector(
        path,
        chimney_geometry_enabled=False,
        acrylic_bowl_geometry_enabled=True,
    )

    assert chimney_only.chimney is not None
    assert chimney_only.acrylic_bowls is None
    assert bowls_only.chimney is None
    assert bowls_only.acrylic_bowls is not None


def test_chimney_opens_top_spherical_interfaces(tmp_path):
    """An axial LS ray passes through the sphere opening when enabled."""
    path = _write_minimal_config(tmp_path)
    enabled = DetectorGeometry.from_config(
        str(path),
        detector_type="nested_sphere",
        temperature=None,
        max_candidates_per_ray=1,
        n_divisions=8,
        chimney_geometry_enabled=True,
        acrylic_bowl_geometry_enabled=False,
    )
    disabled = DetectorGeometry.from_config(
        str(path),
        detector_type="nested_sphere",
        temperature=None,
        max_candidates_per_ray=1,
        n_divisions=8,
        chimney_geometry_enabled=False,
        acrylic_bowl_geometry_enabled=False,
    )
    origin = jnp.array([[0.0, 0.0, 0.0]])
    direction = jnp.array([[0.0, 0.0, 1.0]])

    open_result = enabled.propagator(origin, direction)
    closed_result = disabled.propagator(origin, direction)

    assert not bool(open_result["hit_interface"][0])
    npt.assert_allclose(open_result["positions"][0, 2], 19.18, atol=1e-4)
    assert bool(closed_result["hit_interface"][0])
    npt.assert_allclose(closed_result["positions"][0, 2], 17.7, atol=1e-4)


def test_chimney_region_is_ls_inside_bore(tmp_path):
    """The source-region classifier retains LS in the opened chimney."""
    detector = generate_detector(
        _write_minimal_config(tmp_path),
        chimney_geometry_enabled=True,
        acrylic_bowl_geometry_enabled=False,
    )

    regions = detector.region_of(jnp.array([
        [0.0, 0.0, 18.0],
        [0.6, 0.0, 18.0],
    ]))

    npt.assert_array_equal(regions, [0, 2])


def test_bowl_competes_after_acrylic_water_interface(tmp_path):
    """The numerical offset exposes a bowl after the sphere crossing."""
    geometry = DetectorGeometry.from_config(
        str(_write_minimal_config(tmp_path)),
        detector_type="nested_sphere",
        temperature=None,
        max_candidates_per_ray=1,
        n_divisions=8,
        chimney_geometry_enabled=False,
        acrylic_bowl_geometry_enabled=True,
    )
    result = geometry.propagator(
        jnp.array([[17.8241, 0.0, 0.0]]),
        jnp.array([[1.0, 0.0, 0.0]]),
    )

    npt.assert_array_equal(result["hit_acrylic_bowl"], [True])
    npt.assert_array_equal(result["engineering_surface_code"], [3])
    npt.assert_allclose(result["positions"][0, 0], 17.8245, atol=1e-6)


def test_transparent_engineering_surface_continues_without_detection():
    """A unit-transmission surface preserves both forward modes."""
    results = {}
    for mode in ("sample", "update_factors"):
        step = make_photon_step(
            mode,
            has_interface=True,
            multi_interface=True,
            has_engineering_surfaces=True,
        )
        results[mode] = step(
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
            jax.random.PRNGKey(31),
            jnp.asarray(0.225),
            jnp.asarray(False),
            jnp.asarray(2, dtype=jnp.int32),
            jnp.asarray(1.5),
            jnp.asarray(1.33),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray([1.5, 1.49, 1.33]),
            jnp.asarray(True),
            jnp.asarray(0.0),
            jnp.asarray(1.0),
            jnp.asarray(0.5),
        )

    for result in results.values():
        assert float(result[3]) == 0.0
        assert float(result[5]) == 1.0
        npt.assert_allclose(result[1], [1.0, 0.0, 0.0], atol=1e-6)
    npt.assert_array_equal(
        np.asarray(results["sample"][8], dtype=np.int32),
        [1, 0, 0, 0, 0, 1],
    )
