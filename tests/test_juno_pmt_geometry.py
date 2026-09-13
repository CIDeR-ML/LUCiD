"""Deterministic tests for the JUNO 20-inch PMT head geometry."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from lucid.geometry.detector import generate_detector
from lucid.geometry.juno_pmt import (
    HAMAMATSU,
    HIGH_QE_NNVT,
    NNVT,
    build_juno_pmt_dimensions,
    intersect_ellipsoid_heads,
    load_juno_pmt_layout,
)
from lucid.propagation.base import compute_sensor_intersections_base
from lucid.propagation.nested_sphere import create_nested_sphere_propagator
from lucid.simulation.reflection import (
    PmtSurfaceReflection,
    pmt_surface_reflection,
)


def _write_layout(tmp_path):
    """Write a minimal position/type pair in the JUNOSW text format."""
    positions = tmp_path / "positions.csv"
    positions.write_text(
        "#CopyNo X Y Z Orientation_theta Orientation_phi\n"
        "0 0 0 10000 0 0\n"
        "1 10000 0 0 90 0\n"
        "2 0 10000 0 90 90\n",
        encoding="utf-8",
    )
    types = tmp_path / "types.csv"
    types.write_text(
        "#CopyNo PMTType\n"
        "2 HighQENNVT\n"
        "0 Hamamatsu\n"
        "1 NNVT\n",
        encoding="utf-8",
    )
    return positions, types


def test_layout_preserves_copy_number_orientation_and_type(tmp_path):
    """The two official tables must join by CopyNo, not row position."""
    positions, types = _write_layout(tmp_path)
    layout = load_juno_pmt_layout(positions, types)

    npt.assert_array_equal(layout.copy_numbers, [0, 1, 2])
    npt.assert_allclose(layout.positions_m[0], [0.0, 0.0, 10.0])
    npt.assert_allclose(layout.inward_axes[0], [0.0, 0.0, -1.0])
    npt.assert_allclose(
        layout.inward_axes[1], [-1.0, 0.0, 0.0], atol=1e-12
    )
    npt.assert_array_equal(
        layout.type_codes,
        [HAMAMATSU, NNVT, HIGH_QE_NNVT],
    )


def test_dimensions_match_junosw_front_heads_and_glass_shell():
    """P1a dimensions reproduce the J26.3.2 front ellipsoids."""
    dimensions = build_juno_pmt_dimensions(
        np.array([HAMAMATSU, NNVT, HIGH_QE_NNVT])
    )

    npt.assert_allclose(dimensions.outer_radial_m, 0.254)
    npt.assert_allclose(dimensions.outer_axial_m, [0.190, 0.184, 0.184])
    npt.assert_allclose(dimensions.inner_radial_m, 0.249)
    npt.assert_allclose(dimensions.inner_axial_m, [0.185, 0.179, 0.179])


def test_axis_ray_hits_outer_and_inner_front_surfaces():
    """The glass entry/exit surfaces are separated by 5 mm on axis."""
    origins = np.array([[0.0, 0.0, 0.0]])
    directions = np.array([[0.0, 0.0, 1.0]])
    centers = np.array([[0.0, 0.0, 10.0]])
    axes = np.array([[0.0, 0.0, -1.0]])

    outer = intersect_ellipsoid_heads(
        origins, directions, centers, axes, 0.254, 0.190
    )
    inner = intersect_ellipsoid_heads(
        origins, directions, centers, axes, 0.249, 0.185
    )

    assert bool(outer.valid[0])
    assert bool(inner.valid[0])
    npt.assert_allclose(outer.distance_m, [9.810], atol=1e-5)
    npt.assert_allclose(inner.distance_m, [9.815], atol=1e-5)
    npt.assert_allclose(outer.water_normals[0], [0.0, 0.0, -1.0])
    npt.assert_allclose(
        outer.detector_outward_normals[0], [0.0, 0.0, 1.0]
    )


def test_pmt_diffuse_reflection_returns_to_detector_interior():
    """The transport normal sends a PMT reflection back into the detector."""
    hit = intersect_ellipsoid_heads(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, 10.0]]),
        np.array([[0.0, 0.0, -1.0]]),
        0.254,
        0.190,
    )
    params = PmtSurfaceReflection(
        wall_R0=jnp.asarray(0.05),
        wall_p=jnp.asarray(1.5),
        wall_fspec=jnp.asarray(0.1),
        sensor_rate=jnp.asarray(0.2),
        sensor_fspec=jnp.asarray(0.0),
    )
    _, reflected, _ = pmt_surface_reflection(
        jnp.array([0.0, 0.0, 1.0]),
        hit.detector_outward_normals[0],
        True,
        params,
        jnp.asarray(532.0),
        jax.random.PRNGKey(7),
    )

    detector_inward = jnp.array([0.0, 0.0, -1.0])
    assert float(jnp.dot(reflected, detector_inward)) > 0.0


def test_detector_loader_can_disable_configured_pmt_geometry(tmp_path):
    """The ablation override changes only the PMT intersection geometry."""
    positions, types = _write_layout(tmp_path)
    config = {
        "material": "water",
        "detector_type": "nested_sphere",
        "geometry_definitions": {
            "inner_radius": 8.0,
            "acrylic_radius": 8.1,
            "outer_radius": 10.0,
            "n_sensors": 3,
            "sensor_radius": 0.254,
            "pmt_file_path": positions.name,
            "pmt_type_file_path": types.name,
            "pmt_geometry": "juno_20inch_front",
        },
    }
    config_path = tmp_path / "geometry.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    enabled = generate_detector(config_path, pmt_geometry_enabled=True)
    disabled = generate_detector(config_path, pmt_geometry_enabled=False)

    assert enabled.pmt_geometry == "juno_20inch_front"
    assert enabled.pmt_inward_axes is not None
    assert disabled.pmt_geometry is None
    assert disabled.pmt_inward_axes is None


def test_front_half_rejects_a_ray_that_only_meets_the_back_half():
    """Only the detector-facing half of the bulb is active in P1a."""
    origins = np.array([[0.0, 0.0, 10.1]])
    directions = np.array([[0.0, 0.0, 1.0]])
    centers = np.array([[0.0, 0.0, 10.0]])
    axes = np.array([[0.0, 0.0, -1.0]])
    result = intersect_ellipsoid_heads(
        origins, directions, centers, axes, 0.254, 0.190
    )

    assert not bool(result.valid[0])


def test_base_sensor_kernel_uses_pmt_head_instead_of_full_sphere():
    """The shared sensor kernel must return the 190 mm front apex."""
    origins = np.array([[0.0, 0.0, 0.0]])
    directions = np.array([[0.0, 0.0, 1.0]])
    positions = np.array([[0.0, 0.0, 10.0]])
    axes = np.array([[0.0, 0.0, -1.0]])
    indices = np.array([0], dtype=np.int32)

    result = compute_sensor_intersections_base(
        indices,
        positions,
        0.254,
        origins,
        directions,
        lambda points: np.ones(len(points), dtype=bool),
        lambda distance: np.ones_like(distance),
        sensor_inward_axes=axes,
        sensor_radial_radii=np.array([0.254]),
        sensor_axial_radii=np.array([0.190]),
    )

    npt.assert_allclose(result[1], [[9.810]], atol=1e-5)
    npt.assert_array_equal(result[4], [True])
    npt.assert_allclose(result[5], [[0.0, 0.0, 9.810]], atol=1e-5)


def test_ellipsoid_distance_has_finite_geometry_gradient():
    """An off-axis hit keeps the P1a intersection differentiable."""
    def distance(offset):
        result = intersect_ellipsoid_heads(
            jnp.array([[offset, 0.0, 0.0]]),
            jnp.array([[0.0, 0.0, 1.0]]),
            jnp.array([[0.0, 0.0, 10.0]]),
            jnp.array([[0.0, 0.0, -1.0]]),
            0.254,
            0.190,
        )
        return result.distance_m[0]

    derivative = jax.grad(distance)(jnp.asarray(0.1))
    assert bool(jnp.isfinite(derivative))
    assert float(derivative) > 0.0


def test_nested_propagator_rejects_partial_pmt_geometry():
    """Axes and both sets of radii form one indivisible geometry contract."""
    with pytest.raises(ValueError, match="supplied together"):
        create_nested_sphere_propagator(
            jnp.array([[10.0, 0.0, 0.0]]),
            0.254,
            8.0,
            10.0,
            sensor_inward_axes=jnp.array([[-1.0, 0.0, 0.0]]),
        )
