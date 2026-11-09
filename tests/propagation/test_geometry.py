"""Detector geometry construction: every sensor on the surface its case names, and fits there.

The independent truth is the shape: a barrel sensor sits at r_xy = r with its disk inside the
barrel's height, a cap sensor at z = +-H/2 with its disk inside the cap, a sphere sensor at
|p| = R, a box sensor on the face its case code names and inside that face. Placement pins
(the first and last positions, the placed count) record the algorithm's output so a change to it
is seen; they are regression pins, not physics.
"""
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

import pytest
from lucid.geometry import generate_detector, Cylinder, Sphere, Box

# Slow by behaviour: builds the WCTE_like, JUNO and nuSCOPE detectors.
pytestmark = pytest.mark.slow


class TestCylinderDetector:
    def test_construction(self, cylinder_detector):
        det = cylinder_detector
        assert isinstance(det, Cylinder)
        # n_sensors is the requested count; all_points is actual placed
        assert det.n_sensors == 2500
        assert det.all_points.shape == (2444, 3)
        assert det.all_points.shape[0] == len(det.ID_to_case)

    def test_placement_pins(self, cylinder_detector):
        """Regression pins: the first barrel and the last bottom-cap positions."""
        first = [
            [2.0, 0.0, -1.94],
            [1.9929857184990087, 0.16735568666463096, -1.94],
            [1.97199207414101, 0.3335374934322045, -1.94],
            [1.9371663222572622, 0.4973797743297095, -1.94],
            [1.8887527404749622, 0.6577332934771665, -1.94],
        ]
        last = [
            [1.7243407703905513, -0.8889594521511766, -2.0],
            [1.8010337900511804, -0.7210251639810364, -2.0],
            [1.8614163688121248, -0.5465611602723465, -2.0],
            [1.9049416726896506, -0.3671474140591966, -2.0],
            [1.931215529791784, -0.18440872401011585, -2.0],
        ]
        npt.assert_allclose(cylinder_detector.all_points[:5], first, atol=1e-9)
        npt.assert_allclose(cylinder_detector.all_points[-5:], last, atol=1e-9)

    def test_sensors_on_surface(self, cylinder_detector):
        """Every barrel sensor at r_xy = r with its disk within the height; every cap sensor on its
        cap plane with its disk inside the cap."""
        det = cylinder_detector
        pts = np.asarray(det.all_points)
        case = np.array([det.ID_to_case[i] for i in range(len(pts))])
        rxy = np.hypot(pts[:, 0], pts[:, 1])
        b, t, bo = case == 0, case == 1, case == 2
        npt.assert_allclose(rxy[b], det.r, atol=1e-9)
        assert np.all(np.abs(pts[b, 2]) + det.S_radius <= det.H / 2 + 1e-9)
        npt.assert_allclose(pts[t, 2], det.H / 2, atol=1e-9)
        npt.assert_allclose(pts[bo, 2], -det.H / 2, atol=1e-9)
        assert np.all(rxy[t | bo] + det.S_radius <= det.r + 1e-9)

    def test_id_to_case_coverage(self, cylinder_detector):
        """All three cases are used, in the barrel/top/bottom blocks of all_points, and the two
        caps get the same number of sensors."""
        det = cylinder_detector
        case = np.array([det.ID_to_case[i] for i in range(len(det.all_points))])
        nb, nt, nbo = len(det.barr_points), len(det.tcap_points), len(det.bcap_points)
        assert min(nb, nt, nbo) > 0 and nt == nbo
        npt.assert_array_equal(case, [0] * nb + [1] * nt + [2] * nbo)


class TestSphereDetector:
    def test_construction_and_pins(self):
        det = generate_detector("config/JUNO_geom_config.json")
        assert isinstance(det, Sphere)
        assert det.all_points.shape == (10000, 3)
        npt.assert_allclose(det.all_points[0], [0.0, 0.0, 17.5], atol=1e-3)

    def test_sensors_on_sphere(self):
        det = generate_detector("config/JUNO_geom_config.json")
        radii = np.linalg.norm(det.all_points, axis=1)
        npt.assert_allclose(radii, det.r, rtol=1e-12)


class TestBoxDetector:
    def test_construction_and_pins(self):
        det = generate_detector("config/nuSCOPE_geom_config.json")
        assert isinstance(det, Box)
        assert det.all_points.shape == (4928, 3)
        npt.assert_allclose(det.all_points[0], [-10.88, 2.0, -1.88], atol=1e-2)

    def test_sensors_on_faces(self):
        """Every sensor lies on the face its case code names (0 +y, 1 -y, 2 -x, 3 +x, 4 +z,
        5 -z, as in intersect_box) and its disk fits inside that face."""
        det = generate_detector("config/nuSCOPE_geom_config.json")
        pts = np.asarray(det.all_points)
        half = np.array([det.L / 2, det.W / 2, det.H / 2])
        face_axis = {0: (1, 1), 1: (1, -1), 2: (0, -1), 3: (0, 1), 4: (2, 1), 5: (2, -1)}
        case = np.array([det.ID_to_case[i] for i in range(len(pts))])
        assert set(case) == set(range(6))
        for c, (ax, sign) in face_axis.items():
            p = pts[case == c]
            npt.assert_allclose(p[:, ax], sign * half[ax], atol=1e-9, err_msg=f"face {c}")
            others = [a for a in range(3) if a != ax]
            assert np.all(np.abs(p[:, others]) + det.S_radius <= half[others] + 1e-9), c


class TestBoundsCheck:
    def test_cylinder_bounds(self, cylinder_detector):
        """Inside/outside against the cylinder's own radius and half-height, barrel and caps."""
        det = cylinder_detector
        pts = jnp.array([
            [0.0, 0.0, 0.0],                        # center: inside
            [10.0, 10.0, 10.0],                     # far away: outside
            [det.r - 0.1, 0.0, 0.0],                # just inside the barrel
            [det.r + 0.1, 0.0, 0.0],                # just outside the barrel
            [0.0, 0.0, det.H / 2 - 0.01],           # just inside the cap
            [0.0, 0.0, det.H / 2 + 0.01],           # just outside the cap
        ])
        npt.assert_array_equal(det.bounds_check(pts), [True, False, True, False, True, False])
