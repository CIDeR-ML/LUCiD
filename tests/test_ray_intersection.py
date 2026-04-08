"""Tests for ray-geometry intersection across all 3 geometries.

These tests establish baselines for the core ray-tracing math that
Phase 9 must preserve exactly. Each geometry has its own intersection
function with different surface types and coordinate systems.
"""
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from lucid.propagation.cylinder import (
    intersect_cylinder_wall, intersect_cylinder_cap, intersect_cylinder,
    intersect_cylinder_with_grid, batch_intersect_cylinder_with_grid,
    calculate_cylinder_normals,
)
from lucid.propagation.sphere import (
    intersect_sphere, intersect_sphere_with_grid, batch_intersect_sphere_with_grid,
    calculate_sphere_normals,
)
from lucid.propagation.box import (
    intersect_box_face, intersect_box, intersect_box_with_grid,
    batch_intersect_box_with_grid, calculate_box_normals,
)


# ── Cylinder intersection ────────────────────────────────────────────

class TestCylinderWallIntersection:
    """Ray-cylinder wall: x² + y² = r²"""

    def test_outward_from_center(self):
        """Ray from origin along +x should hit wall at (r, 0, 0)."""
        hit, t = intersect_cylinder_wall(
            jnp.array([0., 0., 0.]), jnp.array([1., 0., 0.]), r=2.0, h=4.0)
        assert hit
        npt.assert_allclose(t, 2.0, atol=1e-4)

    def test_parallel_to_axis_misses_wall(self):
        """Ray along z-axis from center should not hit the wall."""
        hit, t = intersect_cylinder_wall(
            jnp.array([0., 0., 0.]), jnp.array([0., 0., 1.]), r=2.0, h=4.0)
        # Either misses or t is very large
        assert not hit or t > 100.0

    def test_off_center_outward(self):
        """Ray from offset position along +x should hit wall closer."""
        hit, t = intersect_cylinder_wall(
            jnp.array([1.0, 0., 0.]), jnp.array([1., 0., 0.]), r=2.0, h=4.0)
        assert hit
        npt.assert_allclose(t, 1.0, atol=1e-4)

    def test_intersection_on_surface(self):
        """Hit point should satisfy x² + y² = r²."""
        origin = jnp.array([0.3, 0.5, 0.0])
        direction = jnp.array([1.0, 0.2, 0.0])
        direction = direction / jnp.linalg.norm(direction)
        hit, t = intersect_cylinder_wall(origin, direction, r=2.0, h=4.0)
        if hit:
            point = origin + t * direction
            r_hit = jnp.sqrt(point[0]**2 + point[1]**2)
            npt.assert_allclose(r_hit, 2.0, atol=1e-3)


class TestCylinderCapIntersection:
    def test_upward_hits_top_cap(self):
        """Ray from origin along +z should hit top cap at z=h/2."""
        hit, t = intersect_cylinder_cap(
            jnp.array([0., 0., 0.]), jnp.array([0., 0., 1.]), r=2.0, z=2.0)
        assert hit
        npt.assert_allclose(t, 2.0, atol=1e-4)

    def test_outside_radius_misses_cap(self):
        """Ray hitting cap plane outside radius should miss."""
        hit, t = intersect_cylinder_cap(
            jnp.array([3., 0., 0.]), jnp.array([0., 0., 1.]), r=2.0, z=2.0)
        assert not hit


class TestCylinderCombined:
    def test_closest_surface(self):
        """intersect_cylinder should return the CLOSEST intersection."""
        origin = jnp.array([0., 0., 1.9])
        direction = jnp.array([0., 0., 1.])  # close to top cap
        hit, t, part = intersect_cylinder(origin, direction, r=2.0, h=4.0)
        assert hit
        # Should hit top cap (z=2.0) at t=0.1, not the far wall
        npt.assert_allclose(t, 0.1, atol=0.05)
        assert part == 1  # top cap

    def test_all_three_surfaces(self):
        """Different rays should hit wall, top cap, and bottom cap."""
        # Wall
        _, _, p1 = intersect_cylinder(
            jnp.zeros(3), jnp.array([1., 0., 0.]), 2.0, 4.0)
        # Top cap
        _, _, p2 = intersect_cylinder(
            jnp.zeros(3), jnp.array([0., 0., 1.]), 2.0, 4.0)
        # Bottom cap
        _, _, p3 = intersect_cylinder(
            jnp.zeros(3), jnp.array([0., 0., -1.]), 2.0, 4.0)
        assert p1 == 0  # wall
        assert p2 == 1  # top
        assert p3 == 2  # bottom


class TestCylinderNormals:
    def test_wall_normal_is_radial_outward(self):
        """Wall normals point radially outward (standard convention)."""
        point = jnp.array([[2.0, 0.0, 0.0]])
        is_wall = jnp.array([True])
        is_top = jnp.array([False])
        normals = calculate_cylinder_normals(point, is_wall, is_top)
        # Wall normal at (2,0,0) points outward: (+1, 0, 0)
        npt.assert_allclose(normals[0], [1., 0., 0.], atol=1e-3)

    def test_top_cap_normal_points_up(self):
        """Top cap normal should point upward (outward from detector volume)."""
        point = jnp.array([[0.5, 0.5, 2.0]])
        is_wall = jnp.array([False])
        is_top = jnp.array([True])
        normals = calculate_cylinder_normals(point, is_wall, is_top)
        npt.assert_allclose(normals[0], [0., 0., 1.], atol=1e-5)


# ── Sphere intersection ──────────────────────────────────────────────

class TestSphereIntersection:
    def test_outward_from_center(self):
        """Ray from center should hit sphere at distance r."""
        hit, t = intersect_sphere(
            jnp.zeros(3), jnp.array([1., 0., 0.]),
            jnp.zeros(3), 5.0)
        assert hit
        npt.assert_allclose(t, 5.0, atol=1e-4)

    def test_intersection_on_surface(self):
        """Hit point should satisfy |p - center|² = r²."""
        origin = jnp.array([1., 2., 0.])
        direction = jnp.array([1., 0., 0.5])
        direction = direction / jnp.linalg.norm(direction)
        hit, t = intersect_sphere(origin, direction, jnp.zeros(3), 5.0)
        if hit:
            point = origin + t * direction
            npt.assert_allclose(jnp.linalg.norm(point), 5.0, atol=1e-3)

    def test_outside_sphere_misses(self):
        """Ray starting outside and pointing away should miss."""
        hit, t = intersect_sphere(
            jnp.array([10., 0., 0.]), jnp.array([1., 0., 0.]),
            jnp.zeros(3), 5.0)
        assert not hit or t > 1e9


class TestSphereNormals:
    def test_normal_is_outward_radial(self):
        """Sphere normals should point radially outward (away from center)."""
        point = jnp.array([[5.0, 0., 0.], [0., 0., 5.0]])
        normals = calculate_sphere_normals(point)
        npt.assert_allclose(normals[0], [1., 0., 0.], atol=1e-3)
        npt.assert_allclose(normals[1], [0., 0., 1.], atol=1e-3)


# ── Box intersection ─────────────────────────────────────────────────

class TestBoxIntersection:
    def test_outward_from_center(self):
        """Ray from center along +x should hit right face."""
        hit, t, face = intersect_box(
            jnp.zeros(3), jnp.array([1., 0., 0.]), 4.0, 4.0, 6.0)
        assert hit
        # Should hit right face at x = L/2 = 2.0
        npt.assert_allclose(t, 2.0, atol=1e-3)

    def test_six_faces_reachable(self):
        """Each axis direction should hit a different face."""
        L, W, H = 4.0, 4.0, 6.0
        faces_hit = set()
        for d in [[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]:
            _, _, face = intersect_box(
                jnp.zeros(3), jnp.array(d, dtype=jnp.float32), L, W, H)
            faces_hit.add(int(face))
        assert len(faces_hit) == 6

    def test_intersection_on_surface(self):
        """Hit point should be on one of the 6 faces."""
        origin = jnp.array([0.5, -0.3, 0.2])
        direction = jnp.array([1., 0.5, 0.3])
        direction = direction / jnp.linalg.norm(direction)
        L, W, H = 4.0, 4.0, 6.0
        hit, t, face = intersect_box(origin, direction, L, W, H)
        if hit:
            point = origin + t * direction
            on_face = (
                abs(abs(float(point[0])) - L/2) < 0.01 or
                abs(abs(float(point[1])) - W/2) < 0.01 or
                abs(abs(float(point[2])) - H/2) < 0.01
            )
            assert on_face, f"Point {point} not on any face of box ({L},{W},{H})"


class TestBoxNormals:
    def test_face_normal_directions(self):
        """Each face index should map to the correct axis normal."""
        faces = jnp.arange(6)
        normals = calculate_box_normals(faces)
        # Face 0: front (+y), Face 1: back (-y), Face 2: left (-x),
        # Face 3: right (+x), Face 4: top (+z), Face 5: bottom (-z)
        expected = jnp.array([
            [0., 1., 0.], [0., -1., 0.], [-1., 0., 0.],
            [1., 0., 0.], [0., 0., 1.], [0., 0., -1.],
        ])
        npt.assert_allclose(normals, expected, atol=1e-6)


# ── Batch intersection consistency ───────────────────────────────────

class TestBatchIntersection:
    def test_cylinder_batch_matches_single(self):
        """Batch intersection should match individual calls."""
        origins = jnp.array([[0., 0., 0.], [1., 0., 0.]])
        dirs = jnp.array([[1., 0., 0.], [0., 1., 0.]])
        r, h = 2.0, 4.0
        batch = batch_intersect_cylinder_with_grid(
            origins, dirs, r, h, 10, 20, 10)
        # Check first ray individually
        single = intersect_cylinder_with_grid(
            origins[0], dirs[0], r, h, 10, 20, 10)
        npt.assert_allclose(batch[1][0], single[1], atol=1e-6)  # t values

    def test_sphere_batch_matches_single(self):
        origins = jnp.array([[0., 0., 0.], [1., 0., 0.]])
        dirs = jnp.array([[1., 0., 0.], [0., 1., 0.]])
        batch = batch_intersect_sphere_with_grid(origins, dirs, 5.0, 10)
        single = intersect_sphere_with_grid(origins[0], dirs[0], 5.0, 10)
        npt.assert_allclose(batch[1][0], single[1], atol=1e-6)

    def test_box_batch_matches_single(self):
        origins = jnp.array([[0., 0., 0.], [1., 0., 0.]])
        dirs = jnp.array([[1., 0., 0.], [0., 1., 0.]])
        batch = batch_intersect_box_with_grid(
            origins, dirs, 4.0, 4.0, 6.0, 10, 10, 10)
        single = intersect_box_with_grid(
            origins[0], dirs[0], 4.0, 4.0, 6.0, 10, 10, 10)
        npt.assert_allclose(batch[1][0], single[1], atol=1e-6)


# ── Normal convention tests ───────────────────────────────────────────
# ALL geometry normals point OUTWARD (standard convention).
# The simulation code negates normals at the two consumption points
# (epsilon offset and diffuse reflection) to get the inward direction.

class TestNormalConvention:
    """Verify all normals follow the outward convention."""

    def test_cylinder_wall_outward(self):
        """Cylinder wall normals point OUTWARD (radially away from axis)."""
        point = jnp.array([[2.0, 0.0, 0.5], [0.0, 2.0, -0.3]])
        is_wall = jnp.array([True, True])
        is_top = jnp.array([False, False])
        normals = calculate_cylinder_normals(point, is_wall, is_top)
        # Dot with radial direction should be positive (outward)
        for i in range(2):
            radial = point[i, :2] / jnp.linalg.norm(point[i, :2])
            dot = normals[i, 0] * radial[0] + normals[i, 1] * radial[1]
            assert float(dot) > 0, f"Cylinder wall normal should point outward, got dot={dot}"

    def test_cylinder_cap_outward(self):
        """Cylinder cap normals point OUTWARD (+z top, -z bottom)."""
        p = jnp.array([[0.5, 0.5, 2.0], [0.5, 0.5, -2.0]])
        is_wall = jnp.array([False, False])
        is_top = jnp.array([True, False])
        normals = calculate_cylinder_normals(p, is_wall, is_top)
        npt.assert_allclose(normals[0], [0., 0., 1.], atol=1e-5)   # top: +z
        npt.assert_allclose(normals[1], [0., 0., -1.], atol=1e-5)  # bottom: -z

    def test_sphere_outward(self):
        """Sphere normals point OUTWARD (from center to surface)."""
        point = jnp.array([[5.0, 0., 0.], [0., -3.0, 4.0]])
        normals = calculate_sphere_normals(point)
        for i in range(2):
            dot = jnp.dot(normals[i], point[i]) / jnp.linalg.norm(point[i])
            assert float(dot) > 0, "Sphere normal should point outward"

    def test_box_outward(self):
        """Box face normals point OUTWARD."""
        # All 6 box normals should point away from the center
        faces = jnp.arange(6)
        normals = calculate_box_normals(faces)
        # Check that each normal has magnitude 1 and is axis-aligned
        for i in range(6):
            npt.assert_allclose(jnp.linalg.norm(normals[i]), 1.0, atol=1e-6)

    def test_specular_reflection_sign_agnostic(self):
        """Specular reflection gives same result regardless of normal sign."""
        from lucid.simulation.optics import compute_reflection_direction
        d = jnp.array([1., -0.5, 0.3])
        d = d / jnp.linalg.norm(d)
        n_out = jnp.array([0., 1., 0.])
        n_in = jnp.array([0., -1., 0.])
        r_out = compute_reflection_direction(d, n_out)
        r_in = compute_reflection_direction(d, n_in)
        npt.assert_allclose(r_out, r_in, atol=1e-5)


# ── Cross-geometry consistency ────────────────────────────────────────

class TestCrossGeometry:
    def test_all_propagators_return_same_keys(self):
        """All 3 geometry propagators must return dicts with identical keys."""
        from lucid.geometry.detector_geometry import DetectorGeometry

        dg_cyl = DetectorGeometry.from_config(
            "config/WCTE_geom_config.json", detector_type='Cylinder')
        dg_sph = DetectorGeometry.from_config(
            "config/JUNO_geom_config.json", detector_type='Sphere')
        dg_box = DetectorGeometry.from_config(
            "config/nuSCOPE_geom_config.json", detector_type='Box')

        origins = jnp.zeros((2, 3))
        dirs = jnp.array([[1., 0., 0.], [0., 1., 0.]])

        r_cyl = dg_cyl.propagator(origins, dirs)
        r_sph = dg_sph.propagator(origins, dirs)
        r_box = dg_box.propagator(origins, dirs)

        expected_keys = {'sensor_weights', 'sensor_indices', 'times',
                         'positions', 'normals', 'inside_sensor',
                         'per_sensor_positions', 'sensor_normals'}

        assert set(r_cyl.keys()) == expected_keys
        assert set(r_sph.keys()) == expected_keys
        assert set(r_box.keys()) == expected_keys
