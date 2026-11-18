"""`lucid.sources.placement`: one rule for placing a simulated event in a detector.

A rigid motion about the recorded primary vertex: the placed vertex is uniform in the caller's
region, the placed direction is isotropic, and every distance from the vertex and every angle is
kept. Production's event generator draws its rotation from the same `random_rotation`.
"""
import numpy as np
import pytest

from tests.helpers import paths
from lucid.geometry.volume import BoxVolume, CylinderVolume, SphereVolume
from lucid.sources.placement import draw_placement, random_rotation, rigid_motion

REGIONS = [CylinderVolume(12.0, -12.0, 12.0), CylinderVolume(3.0, 1.0, 5.0), SphereVolume(7.0),
           BoxVolume(2.0, 3.0, 4.0)]


def test_rotations_are_proper():
    rng = np.random.default_rng(0)
    for _ in range(50):
        R = random_rotation(rng)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize('region', REGIONS, ids=lambda r: type(r).__name__)
def test_from_unit_lands_inside_and_fills_the_region(region):
    rng = np.random.default_rng(1)
    pts = np.stack([region.from_unit(rng.uniform(0, 1, 3)) for _ in range(4000)])
    assert region.contains(pts).all()
    lo, hi = region.bounding_box()
    np.testing.assert_allclose(pts.mean(0), 0.5 * (lo + hi), atol=0.05 * np.max(hi - lo))


def test_cylinder_vertices_are_uniform_in_volume():
    """Uniform in r², not r: half the vertices inside r = R/sqrt(2)."""
    region = CylinderVolume(12.0, -12.0, 12.0)
    rng = np.random.default_rng(2)
    pts = np.stack([region.from_unit(rng.uniform(0, 1, 3)) for _ in range(20000)])
    inner = np.hypot(pts[:, 0], pts[:, 1]) < 12.0 / np.sqrt(2)
    assert inner.mean() == pytest.approx(0.5, abs=0.015)


def _draw(n, accept=lambda v, d: True, region=REGIONS[0], seed=3):
    rot, ver = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    return [draw_placement([0., 0., 1.], region=region, rotation_rng=rot, vertex_rng=ver,
                           accept=accept, max_tries=50) for _ in range(n)]


def test_placed_directions_are_isotropic():
    d = np.stack([p[2] for p in _draw(20000)])
    np.testing.assert_allclose(d.mean(0), 0.0, atol=0.02)
    assert np.mean(np.abs(d[:, 2]) < 0.5) == pytest.approx(0.5, abs=0.015)   # cos θ uniform


def test_the_placed_direction_is_the_rotated_primary():
    for R, v, d in _draw(20):
        np.testing.assert_allclose(d, R @ [0., 0., 1.])


def test_rejected_draws_are_redrawn_and_exhaustion_raises():
    up = _draw(200, accept=lambda v, d: d[2] > 0)
    assert all(d[2] > 0 for _, _, d in up)
    with pytest.raises(RuntimeError, match='no accepted placement in 50 draws'):
        _draw(1, accept=lambda v, d: False)


def test_rigid_motion_is_rigid_about_the_pivot():
    rng = np.random.default_rng(4)
    pivot, vertex, R = rng.normal(size=3), rng.normal(size=3) * 5, random_rotation(rng)
    pts, dirs = rng.normal(size=(30, 3)), rng.normal(size=(30, 3))
    p2, d2 = rigid_motion(pts, dirs, pivot=pivot, rotation=R, vertex=vertex)
    np.testing.assert_allclose(np.linalg.norm(p2 - vertex, axis=1), np.linalg.norm(pts - pivot, axis=1))
    np.testing.assert_allclose(d2 @ d2.T, dirs @ dirs.T, atol=1e-12)
    np.testing.assert_allclose(
        rigid_motion(pivot[None], dirs[:1], pivot=pivot, rotation=R, vertex=vertex)[0][0], vertex)


@pytest.mark.requires_data(paths.MUON_1GEV_EVENTS)
def test_the_repo_events_record_their_primary():
    from lucid.sources.root_reader import read_primary
    root = str(paths.data_file(paths.MUON_1GEV_EVENTS))
    for ev in range(2):                    # the two events both shipped files hold
        v0, d0 = read_primary(root, ev)
        np.testing.assert_array_equal(v0, 0.0)
        np.testing.assert_array_equal(d0, [0., 0., 1.])
