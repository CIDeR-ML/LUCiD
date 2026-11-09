"""Ray-geometry intersection for the three surface shapes, against hand geometry.

Every expected value is worked out from the shape's defining equation for a ray chosen so the
arithmetic is short: x^2 + y^2 = r^2 with |z| <= h/2 for the barrel, z = +-h/2 inside the disk
for the caps, |p| = R for the sphere, the six planes for the box. Grid cell indices are worked
out from the grid definition (equal bins in azimuth, height, polar angle and cap x/y, counted
from phi = 0, z = -h/2, theta = 0 and x, y = -r).

No detector is built: these run by default.
"""
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

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
    intersect_box, intersect_box_with_grid,
    batch_intersect_box_with_grid, calculate_box_normals,
)

R, H = 2.0, 4.0                # the test cylinder: radius 2, height 4 (caps at z = +-2)


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


# ── Cylinder intersection ────────────────────────────────────────────

class TestCylinderWallIntersection:
    """Ray-cylinder wall: x² + y² = r², |z| <= h/2"""

    def test_outward_from_center(self):
        """Ray from origin along +x should hit wall at (r, 0, 0)."""
        hit, t = intersect_cylinder_wall(
            jnp.array([0., 0., 0.]), jnp.array([1., 0., 0.]), r=R, h=H)
        assert hit
        npt.assert_allclose(t, 2.0, atol=1e-6)

    def test_parallel_to_axis_misses_wall(self):
        """Ray along z-axis from center should not hit the wall."""
        hit, t = intersect_cylinder_wall(
            jnp.array([0., 0., 0.]), jnp.array([0., 0., 1.]), r=R, h=H)
        assert not hit and t > 1e9

    def test_off_center_outward(self):
        """Ray from offset position along +x should hit wall closer."""
        hit, t = intersect_cylinder_wall(
            jnp.array([1.0, 0., 0.]), jnp.array([1., 0., 0.]), r=R, h=H)
        assert hit
        npt.assert_allclose(t, 1.0, atol=1e-6)

    def test_intersection_on_surface(self):
        """Hit point satisfies x² + y² = r², lies within the height, and is ahead of the origin."""
        origin = jnp.array([0.3, 0.5, 0.0])
        direction = jnp.asarray(_unit([1.0, 0.2, 0.3]), jnp.float32)
        hit, t = intersect_cylinder_wall(origin, direction, r=R, h=H)
        assert hit, "this ray starts inside the cylinder, so it must hit the wall"
        assert t > 0
        point = origin + t * direction
        npt.assert_allclose(jnp.hypot(point[0], point[1]), R, atol=1e-5)
        assert abs(float(point[2])) <= H / 2

    def test_wall_rejects_a_crossing_above_the_rim(self):
        """The ray toward (2, 0, 2.2) crosses r = 2 at z = 2.2 > h/2: that is not a wall hit
        (the photon has already left through the cap)."""
        hit, t = intersect_cylinder_wall(jnp.zeros(3), jnp.asarray(_unit([2, 0, 2.2]),
                                                                      jnp.float32), r=R, h=H)
        assert not hit and t > 1e9

    def test_wall_accepts_a_crossing_just_below_the_rim(self):
        hit, t = intersect_cylinder_wall(jnp.zeros(3), jnp.asarray(_unit([2, 0, 1.99]),
                                                                      jnp.float32), r=R, h=H)
        assert hit
        npt.assert_allclose(t, np.hypot(2, 1.99), rtol=1e-6)


class TestCylinderCapIntersection:
    def test_upward_hits_top_cap(self):
        """Ray from origin along +z should hit top cap at z=h/2."""
        hit, t = intersect_cylinder_cap(
            jnp.array([0., 0., 0.]), jnp.array([0., 0., 1.]), r=R, z=2.0)
        assert hit
        npt.assert_allclose(t, 2.0, atol=1e-6)

    def test_outside_radius_misses_cap(self):
        """Ray hitting cap plane outside radius should miss."""
        hit, t = intersect_cylinder_cap(
            jnp.array([3., 0., 0.]), jnp.array([0., 0., 1.]), r=R, z=2.0)
        assert not hit


class TestCylinderCombined:
    def test_closest_surface(self):
        """intersect_cylinder should return the CLOSEST intersection."""
        hit, t, part = intersect_cylinder(jnp.array([0., 0., 1.9]), jnp.array([0., 0., 1.]),
                                          r=R, h=H)
        assert hit
        # The top cap (z = 2.0) at t = 0.1, not the bottom or the wall.
        npt.assert_allclose(t, 0.1, atol=1e-5)
        assert part == 1  # top cap

    def test_all_three_surfaces(self):
        """Axis rays and oblique rays, including crossings within a few cm of the rim, go to the
        surface the geometry says, at the distance it says.

        From the origin toward P, the first surface is the one whose plane or wall the ray reaches
        first; for P on the cylinder that is P itself, at t = |P|.
        """
        cases = [
            ([1, 0, 0], 0, 2.0),
            ([0, 0, 1], 1, 2.0),
            ([0, 0, -1], 2, 2.0),
            ([2.0, 0.0, 1.95], 0, np.hypot(2, 1.95)),          # wall, 5 cm below the top rim
            ([1.95, 0.0, 2.0], 1, np.hypot(1.95, 2)),          # top cap, 5 cm inside the rim
            ([0.0, -2.0, -1.97], 0, np.hypot(2, 1.97)),        # wall, near the bottom rim
            ([1.0, 0.5, -2.0], 2, np.linalg.norm([1, 0.5, 2])),  # bottom cap, oblique
        ]
        for P, part_want, t_want in cases:
            hit, t, part = intersect_cylinder(jnp.zeros(3), jnp.asarray(_unit(P), jnp.float32),
                                              R, H)
            assert hit and int(part) == part_want, (P, int(part))
            npt.assert_allclose(float(t), t_want, rtol=1e-6, err_msg=str(P))


# ── Sphere intersection ──────────────────────────────────────────────

class TestSphereIntersection:
    def test_outward_from_center(self):
        """Ray from center should hit sphere at distance r."""
        hit, t = intersect_sphere(
            jnp.zeros(3), jnp.array([1., 0., 0.]),
            jnp.zeros(3), 5.0)
        assert hit
        npt.assert_allclose(t, 5.0, atol=1e-5)

    def test_intersection_on_surface(self):
        """Hit point should satisfy |p - center|² = r², ahead of the origin (the exit root)."""
        origin = jnp.array([1., 2., 0.])
        direction = jnp.asarray(_unit([1., 0., 0.5]), jnp.float32)
        hit, t = intersect_sphere(origin, direction, jnp.zeros(3), 5.0)
        assert hit, "this ray starts inside the sphere, so it must hit the surface"
        assert t > 0
        point = origin + t * direction
        npt.assert_allclose(jnp.linalg.norm(point), 5.0, atol=1e-5)

    def test_outside_sphere_misses(self):
        """Ray starting outside and pointing away should miss."""
        hit, t = intersect_sphere(
            jnp.array([10., 0., 0.]), jnp.array([1., 0., 0.]),
            jnp.zeros(3), 5.0)
        assert not hit or t > 1e9


# ── Box intersection ─────────────────────────────────────────────────

class TestBoxIntersection:
    L, W, Hb = 4.0, 4.0, 6.0

    def test_outward_from_center(self):
        """Ray from center along +x should hit right face."""
        hit, t, face = intersect_box(
            jnp.zeros(3), jnp.array([1., 0., 0.]), self.L, self.W, self.Hb)
        assert hit
        # Should hit right face at x = L/2 = 2.0
        npt.assert_allclose(t, 2.0, atol=1e-6)
        assert int(face) == 3

    def test_intersection_on_surface(self):
        """The ray from (0.5, -0.3, 0.2) along (1, 0.5, 0.3) reaches x = L/2 first, at
        t = 1.5 / d_x, and lands inside that face."""
        origin = np.array([0.5, -0.3, 0.2])
        direction = _unit([1., 0.5, 0.3])
        hit, t, face = intersect_box(jnp.asarray(origin, jnp.float32),
                                     jnp.asarray(direction, jnp.float32), self.L, self.W, self.Hb)
        assert hit and int(face) == 3
        npt.assert_allclose(float(t), (self.L / 2 - origin[0]) / direction[0], rtol=1e-6)
        point = origin + float(t) * direction
        assert abs(point[1]) <= self.W / 2 and abs(point[2]) <= self.Hb / 2


# ── Grid cell indices, ray side ──────────────────────────────────────

class TestGridIndices:
    def test_cylinder_wall_and_cap_cells(self):
        """n_cap = 10, n_angular = 20, n_height = 10 on r = 2, h = 4.

        Wall point at phi = 100 deg, z = 0.5: angular floor(100/360 * 20) = 5,
        height floor((0.5 + 2)/4 * 10) = 6. Top-cap point (0.7, -1.1): x floor(2.7/4 * 10) = 6,
        y floor(0.9/4 * 10) = 2. Bottom-cap point (-1.3, 0.25): x 1, y 5."""
        phi = np.deg2rad(100.0)
        o = jnp.array([[0., 0., 0.5], [0.7, -1.1, 0.], [-1.3, 0.25, 0.]])
        d = jnp.array([[np.cos(phi), np.sin(phi), 0.], [0., 0., 1.], [0., 0., -1.]])
        hit, t, is_wall, is_top, wall_idx, cap_idx, pt = batch_intersect_cylinder_with_grid(
            o, d, R, H, 10, 20, 10)
        assert np.asarray(hit).all()
        npt.assert_array_equal(np.asarray(is_wall), [True, False, False])
        npt.assert_array_equal(np.asarray(is_top), [False, True, False])
        npt.assert_array_equal(np.asarray(wall_idx)[0], [5, 6])
        npt.assert_array_equal(np.asarray(cap_idx)[1], [6, 2])
        npt.assert_array_equal(np.asarray(cap_idx)[2], [1, 5])
        npt.assert_allclose(np.asarray(t), [2.0, 2.0, 2.0], atol=1e-6)

    def test_sphere_cells(self):
        """n_divisions = 10: 10 polar bins over [0, pi], 20 azimuth bins over [0, 2 pi).
        theta = 50 deg, phi = 200 deg: theta floor(50/180 * 10) = 2, phi floor(200/360 * 20) = 11.
        """
        th, ph = np.deg2rad(50.0), np.deg2rad(200.0)
        d = jnp.asarray([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)],
                        jnp.float32)
        hit, t, ti, pi_, pt = intersect_sphere_with_grid(jnp.zeros(3), d, 5.0, 10)
        assert hit
        assert (int(ti), int(pi_)) == (2, 11)
        npt.assert_allclose(float(t), 5.0, rtol=1e-6)

    def test_box_cells(self):
        """Box 4 x 4 x 6 with 10 bins per axis. +x face point (2, 0.7, -1.3): y bin
        floor(2.7/4 * 10) = 6, z bin floor(1.7/6 * 10) = 2. Top face point (0.5, -1.5, 3): x bin 6,
        y bin 1."""
        d = jnp.asarray(np.stack([_unit([2, 0.7, -1.3]), _unit([0.5, -1.5, 3])]), jnp.float32)
        hit, t, face, idx, pt = batch_intersect_box_with_grid(jnp.zeros((2, 3)), d,
                                                              4.0, 4.0, 6.0, 10, 10, 10)
        npt.assert_array_equal(np.asarray(face), [3, 4])
        npt.assert_array_equal(np.asarray(idx), [[6, 2], [6, 1]])
        npt.assert_allclose(np.asarray(t), [np.linalg.norm([2, 0.7, -1.3]),
                                            np.linalg.norm([0.5, -1.5, 3])], rtol=1e-6)


# ── Batch intersection consistency ───────────────────────────────────

def _assert_batch_equals_single(batch, singles):
    """Every output of the vmapped call, for every ray, equals the single-ray call."""
    for i, single in enumerate(singles):
        for b, s in zip(batch, single):
            npt.assert_allclose(np.asarray(b)[i], np.asarray(s), rtol=0, atol=1e-6)


_O = jnp.array([[0., 0., 0.], [1., 0., 0.], [0.2, -0.4, 0.9]])
_D = jnp.asarray(np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([-0.3, 0.2, 0.9])]),
                 jnp.float32)


class TestBatchIntersection:
    def test_cylinder_batch_matches_single(self):
        batch = batch_intersect_cylinder_with_grid(_O, _D, R, H, 10, 20, 10)
        _assert_batch_equals_single(batch, [intersect_cylinder_with_grid(
            _O[i], _D[i], R, H, 10, 20, 10) for i in range(len(_O))])

    def test_sphere_batch_matches_single(self):
        batch = batch_intersect_sphere_with_grid(_O, _D, 5.0, 10)
        _assert_batch_equals_single(batch, [intersect_sphere_with_grid(_O[i], _D[i], 5.0, 10)
                                            for i in range(len(_O))])

    def test_box_batch_matches_single(self):
        batch = batch_intersect_box_with_grid(_O, _D, 4.0, 4.0, 6.0, 10, 10, 10)
        _assert_batch_equals_single(batch, [intersect_box_with_grid(
            _O[i], _D[i], 4.0, 4.0, 6.0, 10, 10, 10) for i in range(len(_O))])


# ── Normal convention tests ───────────────────────────────────────────
# ALL geometry normals point OUTWARD (standard convention).
# The simulation code negates normals at the two consumption points
# (epsilon offset and diffuse reflection) to get the inward direction.

class TestNormalConvention:
    """Verify all normals follow the outward convention."""

    def test_cylinder_wall_outward(self):
        """Cylinder wall normals are the outward radial unit vector (x, y, 0) / r."""
        point = jnp.array([[2.0, 0.0, 0.5], [0.0, 2.0, -0.3], [-1.2, 1.6, 1.0]])
        normals = calculate_cylinder_normals(point, jnp.array([True] * 3), jnp.array([False] * 3))
        want = np.asarray(point).copy()
        want[:, 2] = 0
        want /= np.linalg.norm(want, axis=1, keepdims=True)
        npt.assert_allclose(normals, want, atol=1e-6)

    def test_cylinder_cap_outward(self):
        """Cylinder cap normals point OUTWARD (+z top, -z bottom)."""
        p = jnp.array([[0.5, 0.5, 2.0], [0.5, 0.5, -2.0]])
        is_wall = jnp.array([False, False])
        is_top = jnp.array([True, False])
        normals = calculate_cylinder_normals(p, is_wall, is_top)
        npt.assert_allclose(normals[0], [0., 0., 1.], atol=1e-6)   # top: +z
        npt.assert_allclose(normals[1], [0., 0., -1.], atol=1e-6)  # bottom: -z

    def test_sphere_outward(self):
        """Sphere normals are p / |p| (from the centre out)."""
        point = jnp.array([[5.0, 0., 0.], [0., -3.0, 4.0], [0., 0., 5.0]])
        normals = calculate_sphere_normals(point)
        npt.assert_allclose(normals, np.asarray(point) / 5.0, atol=1e-6)

    def test_box_outward(self):
        """The normal of the face a ray leaves through points along that ray.

        Outwardness is a statement about the face the intersection actually
        returned, so the test has to go through `intersect_box` to make it:
        reading the six normals straight out of the table and checking their
        norms says nothing about direction, and said nothing before.
        """
        L, W, Hb = 4.0, 4.0, 6.0
        for d in ([1, 0, 0], [-1, 0, 0], [0, 1, 0],
                  [0, -1, 0], [0, 0, 1], [0, 0, -1]):
            direction = jnp.array(d, dtype=jnp.float32)
            hit, _, face = intersect_box(jnp.zeros(3), direction, L, W, Hb)
            assert hit, f"a ray from the centre along {d} must leave the box"
            normal = calculate_box_normals(face)
            # Outward: the exit face's normal agrees with the outgoing ray.
            npt.assert_allclose(normal, direction, atol=1e-6)

    def test_specular_reflection(self):
        """r = d - 2 (d.n) n, the same for either sign of n.

        d = (1, -0.5, 0.3)/|.| off the plane y = 0 reflects to (1, 0.5, 0.3)/|.|: the normal
        component flips, the tangential ones stay, so the angle of incidence equals the angle of
        reflection."""
        from lucid.simulation.optics import compute_reflection_direction
        d = jnp.asarray(_unit([1., -0.5, 0.3]), jnp.float32)
        want = _unit([1., 0.5, 0.3])
        for n in (jnp.array([0., 1., 0.]), jnp.array([0., -1., 0.])):
            r = np.asarray(compute_reflection_direction(d, n))
            npt.assert_allclose(r, want, atol=1e-6)
            npt.assert_allclose(abs(r @ np.asarray(n)), abs(np.asarray(d) @ np.asarray(n)),
                                atol=1e-6)
