"""Analytic tests for the hard SK 20-inch PMT geometry oracle."""

import numpy as np
import numpy.testing as npt
import pytest

from lucid.propagation import intersect_sk20inch_pmt_hard
from lucid.propagation.sk_pmt import (
    BARREL_INACTIVE_BAND_Z,
    SPHERE_CENTER_Z,
    SPHERE_RADIUS,
    SPHERE_TO_TORUS_Z,
    TORUS_MAJOR_RADIUS,
    TORUS_MINOR_RADIUS,
)


ORIGIN = np.zeros(3)
AXIS = np.array([0.0, 0.0, 1.0])


def _vertical_ray(x, y=0.0):
    return intersect_sk20inch_pmt_hard(
        [x, y, 1.0], [0.0, 0.0, -1.0], ORIGIN, AXIS)


def test_sk_constants_reproduce_fortran_transition_height():
    expected = 0.150 * 0.127 / (0.315 - 0.150)
    assert SPHERE_TO_TORUS_Z == pytest.approx(expected)


def test_normal_incidence_hits_spherical_apex():
    hit = _vertical_ray(0.0)
    apex_z = SPHERE_CENTER_Z + SPHERE_RADIUS

    assert hit.intersects and hit.active
    assert hit.surface == "sphere"
    assert hit.distance == pytest.approx(1.0 - apex_z)
    npt.assert_allclose(hit.local_position, [0.0, 0.0, apex_z], atol=1e-12)
    npt.assert_allclose(hit.local_normal, [0.0, 0.0, 1.0], atol=1e-12)
    assert hit.incidence_cosine == pytest.approx(1.0)


def test_off_axis_front_hit_obeys_large_sphere_equation():
    x = 0.10
    hit = _vertical_ray(x)
    expected_z = SPHERE_CENTER_Z + np.sqrt(SPHERE_RADIUS**2 - x**2)

    assert hit.surface == "sphere"
    assert hit.local_position[2] == pytest.approx(expected_z)
    sphere_residual = (
        hit.local_position[0]**2
        + hit.local_position[1]**2
        + (hit.local_position[2] - SPHERE_CENTER_Z)**2
        - SPHERE_RADIUS**2
    )
    assert abs(sphere_residual) < 1e-12
    assert 0.0 < hit.incidence_cosine < 1.0


def test_outer_edge_hits_toroidal_transition():
    x = 0.24
    hit = _vertical_ray(x)
    expected_z = np.sqrt(
        TORUS_MINOR_RADIUS**2 - (x - TORUS_MAJOR_RADIUS)**2)

    assert hit.intersects and hit.active
    assert hit.surface == "torus"
    assert 0.0 < hit.local_position[2] < SPHERE_TO_TORUS_Z
    assert hit.local_position[2] == pytest.approx(expected_z, abs=1e-10)
    torus_residual = (
        (np.hypot(*hit.local_position[:2]) - TORUS_MAJOR_RADIUS)**2
        + hit.local_position[2]**2
        - TORUS_MINOR_RADIUS**2
    )
    assert abs(torus_residual) < 1e-12
    npt.assert_allclose(np.linalg.norm(hit.local_normal), 1.0, atol=1e-12)


def test_two_centimetre_band_is_inactive_only_for_barrel_pmts():
    # Near the 25.4 cm rim, the torus height is below the SK barrel band cut.
    x = 0.253
    barrel_hit = intersect_sk20inch_pmt_hard(
        [x, 0.0, 1.0], [0.0, 0.0, -1.0], ORIGIN, AXIS,
        barrel=True,
    )
    cap_hit = intersect_sk20inch_pmt_hard(
        [x, 0.0, 1.0], [0.0, 0.0, -1.0], ORIGIN, AXIS,
        barrel=False,
    )

    assert barrel_hit.intersects and not barrel_hit.active
    assert barrel_hit.local_position[2] < BARREL_INACTIVE_BAND_Z
    assert cap_hit.intersects and cap_hit.active
    npt.assert_allclose(barrel_hit.position, cap_hit.position)


def test_ray_outside_25_4_cm_envelope_misses():
    hit = _vertical_ray(0.255)

    assert not hit.intersects
    assert not hit.active
    assert hit.surface == "miss"
    assert np.isinf(hit.distance)
    assert np.all(np.isnan(hit.position))


def test_world_transform_uses_inward_pmt_axis():
    # A PMT whose inward axis is world -x has its apex at x=-18.8 cm.
    hit = intersect_sk20inch_pmt_hard(
        ray_origin=[-1.0, 2.0, 3.0],
        ray_direction=[1.0, 0.0, 0.0],
        pmt_position=[0.0, 2.0, 3.0],
        pmt_direction=[-2.0, 0.0, 0.0],  # normalization is internal
    )

    assert hit.intersects and hit.surface == "sphere"
    npt.assert_allclose(hit.position, [-0.188, 2.0, 3.0], atol=1e-12)
    npt.assert_allclose(hit.normal, [-1.0, 0.0, 0.0], atol=1e-12)
    assert hit.distance == pytest.approx(0.812)
    assert hit.incidence_cosine == pytest.approx(1.0)


def test_roots_behind_ray_are_rejected():
    hit = intersect_sk20inch_pmt_hard(
        [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], ORIGIN, AXIS)
    assert not hit.intersects


def _skdpoint(origin, direction, center, radius):
    """Independent translation of SKDONUTS' SKDPOINT helper."""
    relative = center - origin
    along = float(np.dot(direction, relative))
    closest = origin + along * direction
    distance = float(np.linalg.norm(closest - center))
    if distance > radius:
        return closest, distance
    return (closest - np.sqrt(max(radius**2 - distance**2, 0.0)) * direction,
            distance)


def _skdonuts_millimetre_reference(origin, direction):
    """Return SKDONUTS' hit and local point before reflection handling."""
    large_center = np.array([0.0, 0.0, SPHERE_CENTER_Z])
    point, line_distance = _skdpoint(
        origin, direction, large_center, SPHERE_RADIUS)
    if line_distance > SPHERE_RADIUS:
        return False, None

    # hthrg is the z intersection between the large and GEANT spheres.
    hthrg = (
        SPHERE_RADIUS**2 - 0.254**2 - (-SPHERE_CENTER_Z)**2
    ) / (2.0 * -SPHERE_CENTER_Z)
    if point[2] < hthrg:
        point, line_distance = _skdpoint(
            origin, direction, np.zeros(3), 0.254)
        if line_distance > 0.254 or point[2] >= hthrg:
            return False, None

    displacement = point - origin
    if (np.dot(direction, displacement) < 0.0
            and np.linalg.norm(displacement) > 0.01):
        return False, None
    if point[2] >= SPHERE_TO_TORUS_Z:
        return True, point

    was_inside_geant = False
    for step in range(1, 509):
        candidate = point + direction * (step * 0.001)
        if candidate[2] < 0.0:
            return False, None
        rho = np.hypot(candidate[0], candidate[1])
        torus_implicit = np.sqrt(
            (rho - TORUS_MAJOR_RADIUS)**2 + candidate[2]**2
        ) - TORUS_MINOR_RADIUS
        if torus_implicit <= 0.0:
            return True, candidate
        if candidate[2] >= hthrg:
            large_implicit = (
                np.dot(candidate - large_center, candidate - large_center)
                - SPHERE_RADIUS**2
            )
            if large_implicit > 0.0:
                return False, None
            was_inside_geant = True
        else:
            geant_implicit = np.dot(candidate, candidate) - 0.254**2
            if geant_implicit <= 0.0:
                was_inside_geant = True
            elif was_inside_geant:
                return False, None
    return False, None


def test_oracle_matches_skdonuts_millimetre_search_for_oblique_rays():
    rng = np.random.default_rng(20260918)
    compared_hits = 0
    for _ in range(500):
        origin = np.array([
            rng.uniform(-0.35, 0.35),
            rng.uniform(-0.35, 0.35),
            0.80,
        ])
        direction = np.array([
            rng.uniform(-0.8, 0.8),
            rng.uniform(-0.8, 0.8),
            -1.0,
        ])
        direction /= np.linalg.norm(direction)

        reference_hit, reference_point = _skdonuts_millimetre_reference(
            origin, direction)
        oracle = intersect_sk20inch_pmt_hard(
            origin, direction, ORIGIN, AXIS)

        assert oracle.intersects == reference_hit
        if reference_hit:
            compared_hits += 1
            # SKDONUTS accepts the first 1 mm step inside the torus; sphere
            # hits are analytic. The exact oracle must therefore lie no more
            # than one step behind the reported SK point along the ray.
            along_ray = np.dot(reference_point - oracle.local_position,
                               direction)
            transverse = np.linalg.norm(
                (reference_point - oracle.local_position)
                - along_ray * direction)
            assert -1e-10 <= along_ray <= 0.001 + 1e-10
            assert transverse < 1e-9

    assert compared_hits > 50


@pytest.mark.parametrize(
    "argument,value,error",
    [
        ("ray_origin", [0.0, 0.0], "shape"),
        ("ray_direction", [0.0, 0.0, 0.0], "non-zero"),
        ("pmt_direction", [0.0, 0.0, 0.0], "non-zero"),
    ],
)
def test_invalid_vectors_are_rejected(argument, value, error):
    inputs = {
        "ray_origin": [0.0, 0.0, 1.0],
        "ray_direction": [0.0, 0.0, -1.0],
        "pmt_position": ORIGIN,
        "pmt_direction": AXIS,
    }
    inputs[argument] = value
    with pytest.raises(ValueError, match=error):
        intersect_sk20inch_pmt_hard(**inputs)
