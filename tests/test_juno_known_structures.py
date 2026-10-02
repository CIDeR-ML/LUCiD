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
    BOTTOM_CHIMNEY_ACRYLIC,
    BOTTOM_CHIMNEY_STEEL,
    BOWL_CANDIDATES_PER_CELL,
    BOWL_INTERFACE_OFFSET_M,
    BottomChimneyGeometry,
    ChimneyGeometry,
    acrylic_bowl_radius,
    acrylic_bowl_slope,
    build_bowl_candidate_map,
    intersect_acrylic_bowls,
    intersect_bottom_chimney,
    intersect_chimney,
    load_juno_acrylic_bowls,
)
from lucid.simulation.photon_step_factory import make_photon_step
from lucid.simulation.reflection import ScalarReflection


def test_checked_in_bowls_match_junosw_j26():
    """The portable table retains all J26.5.1 Acrylic additions."""
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


def test_twelve_candidates_cover_every_checked_in_bowl_surface():
    """The restored 12-candidate map retains every sampled bowl owner."""
    assert BOWL_CANDIDATES_PER_CELL == 12
    bowls = load_juno_acrylic_bowls(
        "config/JUNO_acrylic_bowls.csv"
    )
    n_divisions = 132
    candidates = np.asarray(
        build_bowl_candidate_map(
            bowls, n_divisions, BOWL_CANDIDATES_PER_CELL
        )
    )
    axes = bowls.outward_axes
    helpers = np.tile([0.0, 0.0, 1.0], (len(axes), 1))
    helpers[np.abs(axes[:, 2]) > 0.9] = [0.0, 1.0, 0.0]
    first = np.cross(axes, helpers)
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second = np.cross(axes, first)
    owners = []
    points = []
    for axial in (0.0, 0.07, 0.14):
        radius = float(acrylic_bowl_radius(axial))
        for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            radial = np.cos(angle) * first + np.sin(angle) * second
            points.append(
                bowls.attachment_points_m
                + axial * axes
                + radius * radial
            )
            owners.append(np.arange(len(axes)))
    points = np.concatenate(points)
    owners = np.concatenate(owners)
    norms = np.linalg.norm(points, axis=1)
    theta = np.arccos(np.clip(points[:, 2] / norms, -1.0, 1.0))
    phi = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    theta_index = np.floor(
        theta / np.pi * n_divisions
    ).astype(np.int32)
    phi_index = np.floor(
        phi / (2.0 * np.pi) * (2 * n_divisions)
    ).astype(np.int32)
    cell = theta_index * (2 * n_divisions) + phi_index

    npt.assert_array_equal(
        np.any(candidates[cell] == owners[:, None], axis=1),
        np.ones(len(points), dtype=bool),
    )


def _nominal_bowl(attachment_m=5.0):
    """Return one radially aligned JUNO Acrylic bowl."""
    return AcrylicBowlLayout(
        copy_numbers=np.array([0], dtype=np.int32),
        attachment_points_m=np.array([[attachment_m, 0.0, 0.0]]),
        outward_axes=np.array([[1.0, 0.0, 0.0]]),
        base_radii_m=np.array([0.45]),
        tip_radii_m=np.array([0.20]),
        lengths_m=np.array([0.14]),
    )


def test_yumiao_arc_profile_replaces_the_linear_side():
    """The production side follows the dormant JUNOSW curved profile."""
    axial = jnp.array([0.0, 0.07, 0.14])
    radii = acrylic_bowl_radius(axial)

    npt.assert_allclose(
        radii,
        [0.450013840033, 0.298417111001, 0.200006485152],
        atol=1e-10,
    )
    assert float(radii[1]) < 0.325


def test_curved_bowl_side_has_analytic_intersection_and_normal():
    """A transverse ray hits the single arc and receives its smooth normal."""
    bowls = _nominal_bowl()
    axial_m = 0.07
    radius_m = float(acrylic_bowl_radius(axial_m))
    hit = intersect_acrylic_bowls(
        jnp.array([[5.0 + axial_m, 1.0, 0.0]]),
        jnp.array([[0.0, -1.0, 0.0]]),
        bowls,
        jnp.array([[0]], dtype=jnp.int32),
    )

    slope = float(acrylic_bowl_slope(axial_m))
    outward = np.array([-slope, 1.0, 0.0])
    outward /= np.linalg.norm(outward)
    npt.assert_array_equal(hit.valid, [True])
    npt.assert_allclose(hit.distance_m, [1.0 - radius_m], atol=1e-7)
    npt.assert_allclose(hit.transport_normals[0], -outward, atol=1e-6)


def test_curved_bowl_side_is_jittable_and_differentiable():
    """Fixed Newton updates compile and preserve a local position gradient."""
    bowls = _nominal_bowl()
    candidates = jnp.array([[0]], dtype=jnp.int32)
    direction = jnp.array([[0.0, -1.0, 0.0]])

    @jax.jit
    def distance(y_position):
        origin = jnp.array([[5.07, y_position, 0.0]])
        return intersect_acrylic_bowls(
            origin, direction, bowls, candidates
        ).distance_m[0]

    npt.assert_allclose(distance(1.0), 1.0 - 0.298417111001, atol=1e-7)
    npt.assert_allclose(jax.grad(distance)(1.0), 1.0, atol=1e-7)


def test_curved_bowl_newton_tracks_the_arc_across_its_length():
    """One smooth-interface solve reproduces transverse analytic hits."""
    bowls = _nominal_bowl()
    axial = jnp.linspace(0.002, 0.138, 17)
    origins = jnp.column_stack([
        5.0 + axial,
        jnp.ones_like(axial),
        jnp.zeros_like(axial),
    ])
    directions = jnp.tile(jnp.array([[0.0, -1.0, 0.0]]), (17, 1))
    candidates = jnp.zeros((17, 1), dtype=jnp.int32)
    hit = intersect_acrylic_bowls(
        origins, directions, bowls, candidates
    )

    npt.assert_array_equal(hit.valid, np.ones(17, dtype=bool))
    npt.assert_allclose(
        hit.distance_m,
        1.0 - acrylic_bowl_radius(axial),
        atol=1e-7,
    )


def test_curved_bowl_oblique_hit_matches_quartic_reference():
    """The nearest Newton root agrees with an independent torus solution."""
    direction = np.array([0.05, -1.0, -0.2])
    direction /= np.linalg.norm(direction)
    hit = intersect_acrylic_bowls(
        jnp.array([[5.07, 0.8, 0.1]]),
        jnp.asarray(direction[None, :]),
        _nominal_bowl(),
        jnp.array([[0]], dtype=jnp.int32),
    )

    npt.assert_array_equal(hit.valid, [True])
    npt.assert_allclose(
        hit.distance_m,
        [0.5553468637053137],
        atol=1e-10,
    )


def _quartic_bowl_distance(origin, direction):
    """Return an independent NumPy-roots distance to one nominal bowl."""
    radial_centre = 0.69764
    arc_radius = 0.776
    axial_centre = 0.73543
    length = 0.14

    def profile_radius(axial):
        relative = axial_centre - axial
        return radial_centre - np.sqrt(
            arc_radius**2 - relative**2
        )

    axial = origin[0]
    rho2 = origin[1] ** 2 + origin[2] ** 2
    if 0.0 <= axial <= length:
        if rho2 <= profile_radius(axial) ** 2:
            return np.inf

    centred = np.array([
        axial - axial_centre,
        origin[1],
        origin[2],
    ])
    q0 = (
        np.dot(centred, centred)
        + radial_centre**2
        - arc_radius**2
    )
    q1 = 2.0 * np.dot(centred, direction)
    h0 = rho2
    h1 = 2.0 * np.dot(origin[1:], direction[1:])
    h2 = np.dot(direction[1:], direction[1:])
    coefficients = np.array([
        q0**2 - 4.0 * radial_centre**2 * h0,
        2.0 * q0 * q1 - 4.0 * radial_centre**2 * h1,
        q1**2 + 2.0 * q0 - 4.0 * radial_centre**2 * h2,
        2.0 * q1,
        1.0,
    ])
    distances = []
    for root in np.roots(coefficients[::-1]):
        if abs(root.imag) > 1e-8 or root.real <= 1e-6:
            continue
        distance = root.real
        point = origin + distance * direction
        hit_rho = np.linalg.norm(point[1:])
        if -1e-7 <= point[0] <= length + 1e-7:
            if hit_rho <= radial_centre:
                distances.append(distance)

    if abs(direction[0]) > 1e-12:
        for cap_axial in (0.0, length):
            distance = (cap_axial - axial) / direction[0]
            point = origin + distance * direction
            if distance > 1e-6:
                if np.linalg.norm(point[1:]) <= profile_radius(cap_axial):
                    distances.append(distance)
    return min(distances, default=np.inf)


def test_curved_bowl_matches_random_quartic_reference():
    """Fixed-iteration hits match independent roots for random rays."""
    rng = np.random.default_rng(20260924)
    count = 512
    origins = np.column_stack([
        rng.uniform(-0.35, 0.55, count),
        rng.uniform(-0.9, 0.9, count),
        rng.uniform(-0.9, 0.9, count),
    ])
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    expected = np.asarray([
        _quartic_bowl_distance(origin, direction)
        for origin, direction in zip(origins, directions)
    ])
    hit = intersect_acrylic_bowls(
        jnp.asarray(origins),
        jnp.asarray(directions),
        _nominal_bowl(attachment_m=0.0),
        jnp.zeros((count, 1), dtype=jnp.int32),
    )
    expected_valid = np.isfinite(expected)
    axial = origins[:, 0]
    radial = np.linalg.norm(origins[:, 1:], axis=1)
    started_inside = (
        (axial >= 0.0)
        & (axial <= 0.14)
        & (radial <= np.asarray(acrylic_bowl_radius(axial)))
    )

    npt.assert_array_equal(
        np.asarray(hit.valid)[~started_inside],
        expected_valid[~started_inside],
    )
    npt.assert_allclose(
        np.asarray(hit.distance_m)[expected_valid & ~started_inside],
        expected[expected_valid & ~started_inside],
        atol=2e-6,
        rtol=0.0,
    )


def test_axis_ray_hits_bowl_tip_from_both_sides():
    """The closed bowl returns entry and exit intersections."""
    bowls = _nominal_bowl()
    candidates = jnp.array([[0]], dtype=jnp.int32)
    hit = intersect_acrylic_bowls(
        jnp.array([[6.0, 0.0, 0.0]]),
        jnp.array([[-1.0, 0.0, 0.0]]),
        bowls,
        candidates,
    )
    inside = intersect_acrylic_bowls(
        jnp.array([[5.07, 0.0, 0.0]]),
        jnp.array([[1.0, 0.0, 0.0]]),
        bowls,
        candidates,
    )

    npt.assert_allclose(hit.distance_m, [0.86], atol=1e-6)
    npt.assert_array_equal(hit.valid, [True])
    npt.assert_allclose(hit.transport_normals[0], [-1.0, 0.0, 0.0])
    npt.assert_allclose(inside.distance_m, [0.07], atol=1e-6)
    npt.assert_array_equal(inside.valid, [True])
    npt.assert_array_equal(inside.inside_before, [True])

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
    npt.assert_array_equal(hit.inside_before, [False])


def test_chimney_acrylic_origin_reaches_outer_exit():
    """A ray in the chimney wall receives an Acrylic exit boundary."""
    chimney = ChimneyGeometry.junosw_j26(outer_limit_m=19.434)
    hit = intersect_chimney(
        jnp.array([[0.45, 0.0, 18.0]]),
        jnp.array([[1.0, 0.0, 0.0]]),
        chimney,
    )

    npt.assert_allclose(hit.distance_m, [0.074], atol=1e-6)
    npt.assert_array_equal(hit.valid, [True])
    npt.assert_array_equal(hit.inside_before, [True])
    npt.assert_allclose(hit.outward_normals[0], [1.0, 0.0, 0.0])


def test_bottom_chimney_matches_junosw_principal_dimensions():
    """The south opening retains the J26.5.1 bore and flange scales."""
    bottom = BottomChimneyGeometry.junosw_j26()

    npt.assert_allclose(bottom.aperture_radii_m, [0.500, 0.580])
    npt.assert_allclose(
        bottom.acrylic.inner_radii_m,
        [0.500, 0.580],
    )
    npt.assert_allclose(
        bottom.acrylic.outer_radii_m,
        [0.580, 0.675],
    )
    npt.assert_allclose(bottom.acrylic.z_min_m, [-18.700, -18.700])
    npt.assert_allclose(bottom.acrylic.z_max_m, [-17.700, -18.600])
    npt.assert_allclose(bottom.steel_upper.inner_radii_m, [0.500, 0.506])
    npt.assert_allclose(bottom.steel_upper.outer_radii_m, [0.506, 0.675])
    npt.assert_allclose(bottom.steel_upper.z_min_m, [-18.765, -18.750])
    npt.assert_allclose(bottom.steel_upper.z_max_m, [-18.700, -18.700])
    npt.assert_allclose(bottom.steel_centre_z_m, -18.765)
    npt.assert_allclose(bottom.steel_inner_axes_m, [0.500, 0.250])
    npt.assert_allclose(bottom.steel_outer_axes_m, [0.502, 0.255])


def test_bottom_chimney_acrylic_wall_is_transparent_interface():
    """A bore-side ray first enters the bottom Acrylic tube wall."""
    bottom = BottomChimneyGeometry.junosw_j26()
    hit = intersect_bottom_chimney(
        jnp.array([[0.0, 0.0, -18.0]]),
        jnp.array([[1.0, 0.0, 0.0]]),
        bottom,
    )

    npt.assert_array_equal(hit.valid, [True])
    npt.assert_allclose(hit.distance_m, [0.500], atol=1e-7)
    npt.assert_array_equal(hit.surface_codes, [BOTTOM_CHIMNEY_ACRYLIC])
    npt.assert_array_equal(hit.inside_before, [False])


def test_bottom_chimney_intersection_is_jittable_and_differentiable():
    """The effective south wall preserves a local origin gradient."""
    bottom = BottomChimneyGeometry.junosw_j26()

    @jax.jit
    def distance(x_position):
        hit = intersect_bottom_chimney(
            jnp.array([[x_position, 0.0, -18.0]]),
            jnp.array([[1.0, 0.0, 0.0]]),
            bottom,
        )
        return hit.distance_m[0]

    npt.assert_allclose(distance(0.100), 0.400, atol=1e-7)
    npt.assert_allclose(jax.grad(distance)(0.100), -1.0, atol=1e-7)


def test_bottom_chimney_axis_reaches_steel_enclosure():
    """The open south axis terminates on the effective steel bowl."""
    bottom = BottomChimneyGeometry.junosw_j26()
    hit = intersect_bottom_chimney(
        jnp.array([[0.0, 0.0, -18.0]]),
        jnp.array([[0.0, 0.0, -1.0]]),
        bottom,
    )

    npt.assert_array_equal(hit.valid, [True])
    npt.assert_allclose(hit.positions[0, 2], -19.015, atol=1e-6)
    npt.assert_array_equal(hit.surface_codes, [BOTTOM_CHIMNEY_STEEL])


def test_bottom_chimney_acrylic_union_has_no_internal_end_cap():
    """The tube wall continues through the flange without a fake face."""
    bottom = BottomChimneyGeometry.junosw_j26()
    hit = intersect_bottom_chimney(
        jnp.array([[0.540, 0.0, -18.500]]),
        jnp.array([[0.0, 0.0, -1.0]]),
        bottom,
    )

    npt.assert_array_equal(hit.valid, [True])
    npt.assert_allclose(hit.positions[0, 2], -18.700, atol=1e-7)
    npt.assert_array_equal(hit.surface_codes, [BOTTOM_CHIMNEY_ACRYLIC])
    npt.assert_array_equal(hit.inside_before, [True])


def test_bottom_chimney_acrylic_union_skips_shared_radial_face():
    """The tube/flange partition is not reported as an optical interface."""
    bottom = BottomChimneyGeometry.junosw_j26()
    hit = intersect_bottom_chimney(
        jnp.array([[0.540, 0.0, -18.650]]),
        jnp.array([[1.0, 0.0, 0.0]]),
        bottom,
    )

    npt.assert_array_equal(hit.valid, [True])
    npt.assert_allclose(hit.positions[0, 0], 0.675, atol=1e-7)
    npt.assert_array_equal(hit.surface_codes, [BOTTOM_CHIMNEY_ACRYLIC])
    npt.assert_array_equal(hit.inside_before, [True])


def test_bowl_acrylic_origin_reaches_curved_exit():
    """A ray in one bowl receives the curved Acrylic exit boundary."""
    bowls = _nominal_bowl()
    candidates = jnp.array([[0]], dtype=jnp.int32)
    hit = intersect_acrylic_bowls(
        jnp.array([[5.07, 0.0, 0.0]]),
        jnp.array([[0.0, 1.0, 0.0]]),
        bowls,
        candidates,
    )

    npt.assert_array_equal(hit.valid, [True])
    npt.assert_array_equal(hit.inside_before, [True])
    assert 0.2 < float(hit.distance_m[0]) < 0.5


def test_transparent_chimney_reports_acrylic_target_medium(tmp_path):
    """The propagator labels entry and exit media for chimney Acrylic."""
    geometry = DetectorGeometry.from_config(
        str(_write_minimal_config(tmp_path)),
        detector_type="nested_sphere",
        temperature=None,
        max_candidates_per_ray=1,
        n_divisions=8,
        chimney_geometry_enabled=True,
        acrylic_bowl_geometry_enabled=False,
    )
    entering = geometry.propagator(
        jnp.array([[0.0, 0.0, 18.0]]),
        jnp.array([[1.0, 0.0, 0.0]]),
    )
    exiting = geometry.propagator(
        jnp.array([[0.45, 0.0, 18.0]]),
        jnp.array([[1.0, 0.0, 0.0]]),
    )

    npt.assert_array_equal(entering["hit_interface"], [True])
    npt.assert_array_equal(entering["engineering_entering"], [True])
    npt.assert_array_equal(entering["interface_target_medium_id"], [1])
    npt.assert_array_equal(exiting["hit_interface"], [True])
    npt.assert_array_equal(exiting["engineering_entering"], [False])
    npt.assert_array_equal(exiting["interface_target_medium_id"], [2])


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
        "0,17.824,0,0,0.45,0.2,0.14\n",
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
            "bottom_chimney_geometry": "juno_bottom_chimney",
            "acrylic_bowl_file_path": bowls.name,
            "acrylic_bowl_geometry": "juno_addition_curved_arc",
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


def test_bottom_chimney_has_independent_loader_override(tmp_path):
    """The south opening can be ablated without changing the top chimney."""
    path = _write_minimal_config(tmp_path)
    enabled = generate_detector(
        path,
        chimney_geometry_enabled=False,
        bottom_chimney_geometry_enabled=True,
        acrylic_bowl_geometry_enabled=False,
    )
    disabled = generate_detector(
        path,
        chimney_geometry_enabled=True,
        bottom_chimney_geometry_enabled=False,
        acrylic_bowl_geometry_enabled=False,
    )

    assert enabled.chimney is None
    assert enabled.bottom_chimney is not None
    assert disabled.chimney is not None
    assert disabled.bottom_chimney is None


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


def test_bottom_chimney_opens_south_spherical_interfaces(tmp_path):
    """An axial LS ray reaches the steel bowl through the south opening."""
    path = _write_minimal_config(tmp_path)
    geometry = DetectorGeometry.from_config(
        str(path),
        detector_type="nested_sphere",
        temperature=None,
        max_candidates_per_ray=1,
        n_divisions=8,
        chimney_geometry_enabled=False,
        bottom_chimney_geometry_enabled=True,
        acrylic_bowl_geometry_enabled=False,
    )
    result = geometry.propagator(
        jnp.array([[0.0, 0.0, 0.0]]),
        jnp.array([[0.0, 0.0, -1.0]]),
    )

    npt.assert_array_equal(result["hit_interface"], [False])
    npt.assert_array_equal(result["hit_bottom_chimney"], [True])
    npt.assert_array_equal(
        result["engineering_surface_code"],
        [BOTTOM_CHIMNEY_STEEL],
    )
    npt.assert_allclose(result["positions"][0, 2], -19.015, atol=1e-5)


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
        npt.assert_allclose(result[3], 0.0, atol=1e-12)
        npt.assert_allclose(result[5], 1.0, atol=1e-12)
        npt.assert_allclose(result[1], [1.0, 0.0, 0.0], atol=1e-6)
    npt.assert_array_equal(
        np.asarray(results["sample"][8], dtype=np.int32),
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    )


def test_acrylic_interface_transmits_to_explicit_target_medium():
    """An engineering interface updates the medium without surface loss."""
    step = make_photon_step(
        "sample",
        has_interface=True,
        multi_interface=True,
        has_engineering_surfaces=True,
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
        jax.random.PRNGKey(37),
        jnp.asarray(0.225),
        jnp.asarray(True),
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray(1.5),
        jnp.asarray(1.5),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray([1.5, 1.5, 1.5]),
        jnp.asarray(True),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(True),
        jnp.asarray(-1, dtype=jnp.int32),
    )

    npt.assert_array_equal(result[7], 1)
    npt.assert_array_equal(
        np.asarray(result[8], dtype=np.int32),
        [1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    )


def test_acrylic_volume_reports_bulk_absorption_and_path():
    """A terminated in-volume step records bulk rather than surface loss."""
    step = make_photon_step(
        "sample",
        has_interface=True,
        multi_interface=True,
        has_engineering_surfaces=True,
    )
    result = step(
        jnp.array([0.0, 0.0, 0.0]),
        jnp.array([1.0, 0.0, 0.0]),
        jnp.asarray(0.0),
        jnp.asarray(0.2),
        jnp.array([1.0, 0.0, 0.0]),
        jnp.asarray(1e12),
        jnp.asarray(1e12),
        jnp.asarray(0.0),
        ScalarReflection(jnp.asarray(0.0), jnp.asarray(0.0)),
        jnp.asarray(1e-9),
        jnp.asarray(False),
        jnp.asarray(532.0),
        jax.random.PRNGKey(41),
        jnp.asarray(0.2),
        jnp.asarray(True),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(1.5),
        jnp.asarray(1.33),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray([1.5, 1.5, 1.5]),
        jnp.asarray(True),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray(False),
        jnp.asarray(3, dtype=jnp.int32),
    )

    events = np.asarray(result[8])
    npt.assert_allclose(events[8], 1.0)
    npt.assert_allclose(events[9], 0.2, atol=1e-6)
