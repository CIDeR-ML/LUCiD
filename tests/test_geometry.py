"""Tests for detector geometry construction and properties."""
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from lucid.geometry import generate_detector


class TestCylinderDetector:
    def test_construction(self, cylinder_detector):
        det = cylinder_detector
        # n_sensors is the requested count; all_points is actual placed
        assert det.n_sensors == 2500
        assert det.all_points.shape == (2444, 3)
        assert det.all_points.shape[0] == len(det.ID_to_case)

    def test_first_sensors(self, cylinder_detector):
        expected = [
            [2.0, 0.0, -1.94],
            [1.9929857184990087, 0.16735568666463096, -1.94],
            [1.97199207414101, 0.3335374934322045, -1.94],
            [1.9371663222572622, 0.4973797743297095, -1.94],
            [1.8887527404749622, 0.6577332934771665, -1.94],
        ]
        npt.assert_allclose(cylinder_detector.all_points[:5], expected, atol=1e-5)

    def test_last_sensors(self, cylinder_detector):
        expected = [
            [1.7243407703905513, -0.8889594521511766, -2.0],
            [1.8010337900511804, -0.7210251639810364, -2.0],
            [1.8614163688121248, -0.5465611602723465, -2.0],
            [1.9049416726896506, -0.3671474140591966, -2.0],
            [1.931215529791784, -0.18440872401011585, -2.0],
        ]
        npt.assert_allclose(cylinder_detector.all_points[-5:], expected, atol=1e-4)

    def test_sensors_on_surface(self, cylinder_detector):
        """All barrel sensors should be at radius r=2.0, cap sensors at z=+/-2."""
        det = cylinder_detector
        pts = det.all_points
        for idx in range(min(50, len(pts))):
            case = det.ID_to_case[idx]
            p = pts[idx]
            if case == 0:  # barrel
                r = np.sqrt(p[0]**2 + p[1]**2)
                npt.assert_allclose(r, det.r, atol=0.01)
            elif case == 1:  # top cap
                npt.assert_allclose(abs(p[2]), det.H / 2, atol=0.1)

    def test_id_to_case_coverage(self, cylinder_detector):
        det = cylinder_detector
        cases = set(det.ID_to_case.values())
        assert cases.issubset({0, 1, 2}), f"Unexpected cases: {cases}"


class TestSphereDetector:
    def test_construction(self):
        det = generate_detector("config/JUNO_geom_config.json")
        assert det.all_points.shape[1] == 3
        assert det.n_sensors > 0

    def test_sensors_on_sphere(self):
        det = generate_detector("config/JUNO_geom_config.json")
        radii = np.linalg.norm(det.all_points, axis=1)
        npt.assert_allclose(radii, det.r, atol=0.01)


class TestBoxDetector:
    def test_construction(self):
        det = generate_detector("config/nuSCOPE_geom_config.json")
        assert det.all_points.shape[1] == 3
        assert det.n_sensors > 0

    def test_sensors_on_faces(self):
        det = generate_detector("config/nuSCOPE_geom_config.json")
        pts = det.all_points
        # each sensor should be on one of the 6 faces
        for idx in range(min(20, len(pts))):
            p = pts[idx]
            on_face = (
                abs(abs(p[0]) - det.L / 2) < 0.01 or
                abs(abs(p[1]) - det.W / 2) < 0.01 or
                abs(abs(p[2]) - det.H / 2) < 0.01
            )
            assert on_face, f"Sensor {idx} at {p} not on any face"


class TestBoundsCheck:
    def test_cylinder_bounds(self, cylinder_detector):
        from lucid.propagation.cylinder import cylinder_bounds_check
        det = cylinder_detector
        pts = jnp.array([
            [0.0, 0.0, 0.0],       # center → inside
            [10.0, 10.0, 10.0],     # far away → outside
            [1.9, 0.0, 0.0],        # near barrel → inside
        ])
        result = cylinder_bounds_check(pts, det.r, det.H)
        assert result[0] == True
        assert result[1] == False
        assert result[2] == True

    def test_cylinder_bounds_method(self, cylinder_detector):
        """detector.bounds_check() matches standalone cylinder_bounds_check()."""
        from lucid.propagation.cylinder import cylinder_bounds_check
        det = cylinder_detector
        pts = jnp.array([
            [0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [1.9, 0.0, 0.0],
            [0.0, 0.0, 1.99], [0.0, 0.0, 2.01],
        ])
        expected = cylinder_bounds_check(pts, det.r, det.H)
        result = det.bounds_check(pts)
        npt.assert_array_equal(result, expected)

    def test_sphere_bounds_method(self):
        det = generate_detector("config/JUNO_geom_config.json")
        pts = jnp.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])
        result = det.bounds_check(pts)
        assert result[0] == True
        assert result[1] == False

    def test_box_bounds_method(self):
        det = generate_detector("config/nuSCOPE_geom_config.json")
        pts = jnp.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])
        result = det.bounds_check(pts)
        assert result[0] == True
        assert result[1] == False
