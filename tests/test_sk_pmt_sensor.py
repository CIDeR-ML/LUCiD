"""Integration tests for SK 20-inch PMTs in the shared propagator."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from lucid.geometry.cylinder import Cylinder
from lucid.propagation.shared import create_propagator
from lucid.propagation.sk_pmt_lookup import build_sk20inch_coverage_lookup
from lucid.propagation.sk_pmt_sensor import (
    compute_sk20inch_sensor_intersections,
)


@pytest.fixture(scope="module")
def lookup():
    return build_sk20inch_coverage_lookup(
        0.030,
        n_incidence=9,
        grid_spacing=0.010,
        tail_sigma=3.0,
        raster_oversample=2,
        cache=False,
    )


def _candidate_result(origin, direction, *, barrel=False, lookup=None):
    return compute_sk20inch_sensor_intersections(
        sensor_indices=jnp.array([0]),
        sensor_positions=jnp.zeros((1, 3)),
        sensor_directions=jnp.array([[0.0, 0.0, 1.0]]),
        sensor_is_barrel=jnp.array([barrel]),
        ray_origins=jnp.asarray(origin)[None, :],
        ray_directions=jnp.asarray(direction)[None, :],
        coverage_lookup=lookup,
    )


def test_hard_candidate_uses_exact_bulb_root_and_transport_normal():
    result = _candidate_result(
        origin=[0.0, 0.0, 1.0], direction=[0.0, 0.0, -1.0])
    weights, times, indices, normals, inside, positions = result

    assert float(weights[0]) == 1.0
    assert bool(inside[0])
    assert int(indices[0]) == 0
    # Sphere apex: center at -0.127 m plus radius 0.315 m.
    assert float(times[0, 0]) == pytest.approx(0.812, abs=2e-6)
    npt.assert_allclose(positions[0], [0.0, 0.0, 0.188], atol=2e-6)
    # The geometry oracle points into water; transport expects out of water.
    npt.assert_allclose(normals[0], [0.0, 0.0, -1.0], atol=2e-6)


def test_hard_barrel_band_intersects_but_collects_no_charge():
    cap = _candidate_result(
        origin=[0.253, 0.0, 1.0], direction=[0.0, 0.0, -1.0],
        barrel=False)
    barrel = _candidate_result(
        origin=[0.253, 0.0, 1.0], direction=[0.0, 0.0, -1.0],
        barrel=True)

    assert bool(cap[4][0]) and bool(barrel[4][0])
    assert float(cap[0][0]) == 1.0
    assert float(barrel[0][0]) == 0.0
    npt.assert_allclose(cap[5], barrel[5], atol=2e-6)


def test_invalid_candidate_is_zeroed():
    result = compute_sk20inch_sensor_intersections(
        sensor_indices=jnp.array([-1]),
        sensor_positions=jnp.zeros((1, 3)),
        sensor_directions=jnp.array([[0.0, 0.0, 1.0]]),
        sensor_is_barrel=jnp.array([False]),
        ray_origins=jnp.array([[0.0, 0.0, 1.0]]),
        ray_directions=jnp.array([[0.0, 0.0, -1.0]]),
    )
    assert float(result[0][0]) == 0.0
    assert not bool(result[4][0])


def test_smooth_candidate_has_edge_gradient(lookup):
    direction = jnp.array([0.0, 0.0, -1.0])

    def coverage(offset):
        return _candidate_result(
            origin=jnp.array([offset, 0.0, 1.0]),
            direction=direction,
            lookup=lookup,
        )[0][0]

    value = coverage(0.254)
    derivative = jax.grad(coverage)(0.254)
    assert float(value) == pytest.approx(0.5, abs=0.04)
    assert np.isfinite(float(derivative))
    assert float(derivative) < -5.0


def _one_pmt_detector(tmp_path):
    path = tmp_path / "one_pmt.npz"
    np.savez(
        path,
        positions_mm=np.array([[1000.0, 0.0, 0.0]]),
        directions=np.array([[-1.0, 0.0, 0.0]]),
        surfaces=np.array(["barrel"]),
        pmt_id=np.array([1]),
        radius=np.asarray(1.0),
        height=np.asarray(2.0),
        sensor_radius=np.asarray(0.254),
    )
    return Cylinder.from_pmt_file(path)


def _two_overlapping_pmt_detector(tmp_path):
    path = tmp_path / "two_overlapping_pmts.npz"
    np.savez(
        path,
        positions_mm=np.array([
            [1000.0, -150.0, 0.0],
            [1000.0, 150.0, 0.0],
        ]),
        directions=np.array([
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]),
        surfaces=np.array(["barrel", "barrel"]),
        pmt_id=np.array([1, 2]),
        radius=np.asarray(1.0),
        height=np.asarray(2.0),
        sensor_radius=np.asarray(0.254),
    )
    return Cylinder.from_pmt_file(path)


def test_shared_propagator_reaches_curved_pmt_before_wall(tmp_path):
    detector = _one_pmt_detector(tmp_path)
    assert detector.surfaces.tolist() == ["barrel"]
    propagator = create_propagator(
        detector,
        jnp.asarray(detector.all_points),
        detector.S_radius,
        temperature=None,
        sensor_shape="sk20inch",
        max_candidates_per_ray=1,
    )
    result = propagator(
        jnp.array([[0.0, 0.0, 0.0]]),
        jnp.array([[1.0, 0.0, 0.0]]),
    )

    assert bool(jnp.any(result["inside_sensor"][:, 0]))
    assert float(jnp.max(result["sensor_weights"][:, 0])) == 1.0
    # PMT local apex is 18.8 cm inward from its wall-mounted origin.
    npt.assert_allclose(result["positions"][0], [0.812, 0.0, 0.0], atol=2e-6)
    npt.assert_allclose(result["normals"][0], [1.0, 0.0, 0.0], atol=2e-6)


def test_wall_routing_selects_one_pmt_before_curved_intersection(tmp_path):
    origin = jnp.array([[0.0, -0.03, 0.0]])
    direction = jnp.array([[1.0, 0.0, 0.0]])

    outputs = {}
    for mode in ("all", "wall"):
        detector = _two_overlapping_pmt_detector(tmp_path)
        propagator = create_propagator(
            detector,
            jnp.asarray(detector.all_points),
            detector.S_radius,
            temperature=None,
            sensor_shape="sk20inch",
            sensor_candidate_selection=mode,
            max_candidates_per_ray=2,
        )
        outputs[mode] = propagator(origin, direction)

    # Both curved bulbs intercept the ray in the legacy all-candidate mode.
    assert float(jnp.sum(outputs["all"]["sensor_weights"])) == 2.0
    # The nominal wall crossing is closer to PMT 0, so wall routing deposits
    # only there even though the ray also intersects PMT 1's bulb.
    wall = outputs["wall"]
    assert float(jnp.sum(wall["sensor_weights"])) == 1.0
    nonzero_slot = int(jnp.argmax(wall["sensor_weights"][:, 0]))
    assert int(wall["sensor_indices"][nonzero_slot, 0]) == 0
    assert int(jnp.sum(wall["inside_sensor"])) == 1


def test_smooth_shared_propagator_supports_forward_and_reverse_autodiff(
    tmp_path, lookup,
):
    """The integrated production path differentiates the forward value."""
    detector = _one_pmt_detector(tmp_path)
    propagator = create_propagator(
        detector,
        jnp.asarray(detector.all_points),
        detector.S_radius,
        temperature=float(lookup.sigma) / detector.S_radius,
        sensor_shape="sk20inch",
        sk_pmt_lookup=lookup,
        max_candidates_per_ray=1,
    )

    def deposited_weight(offset):
        result = propagator(
            jnp.array([[0.0, offset, 0.0]]),
            jnp.array([[1.0, 0.0, 0.0]]),
        )
        return jnp.sum(result["sensor_weights"])

    offset = 0.24
    reverse = jax.grad(deposited_weight)(offset)
    forward = jax.jvp(deposited_weight, (offset,), (1.0,))[1]
    step = 1e-4
    finite_difference = (
        deposited_weight(offset + step)
        - deposited_weight(offset - step)
    ) / (2.0 * step)

    assert np.isfinite(float(reverse))
    assert abs(float(reverse)) > 1.0
    assert float(reverse) == pytest.approx(float(forward), rel=2e-6)
    assert float(reverse) == pytest.approx(
        float(finite_difference), rel=2e-4, abs=2e-3)


def test_wall_routing_has_hard_forward_and_smooth_backward(tmp_path, lookup):
    detector = _two_overlapping_pmt_detector(tmp_path)
    propagator = create_propagator(
        detector,
        jnp.asarray(detector.all_points),
        detector.S_radius,
        temperature=float(lookup.sigma) / detector.S_radius,
        sensor_shape="sk20inch",
        sk_pmt_lookup=lookup,
        sensor_candidate_selection="wall",
        max_candidates_per_ray=2,
    )

    def deposited_weight(offset):
        result = propagator(
            jnp.array([[0.0, offset, 0.0]]),
            jnp.array([[1.0, 0.0, 0.0]]),
        )
        return jnp.sum(result["sensor_weights"])

    offset = -0.03
    value = deposited_weight(offset)
    reverse = jax.grad(deposited_weight)(offset)
    forward = jax.jvp(deposited_weight, (offset,), (1.0,))[1]

    # The discrete wall assignment is exact in the forward pass, while the
    # lookup coverage still supplies a useful derivative in both AD modes.
    assert float(value) == 1.0
    assert np.isfinite(float(reverse))
    assert abs(float(reverse)) > 0.1
    assert float(reverse) == pytest.approx(float(forward), rel=1e-5)


def test_sk_shape_requires_measured_axes_and_surfaces():
    detector = Cylinder(1.0, 2.0, 20, 0.10)
    with pytest.raises(ValueError, match="pmt_directions"):
        create_propagator(
            detector,
            jnp.asarray(detector.all_points),
            detector.S_radius,
            temperature=None,
            sensor_shape="sk20inch",
        )


def test_unknown_sensor_shape_is_rejected(tmp_path):
    detector = _one_pmt_detector(tmp_path)
    with pytest.raises(ValueError, match="sensor_shape"):
        create_propagator(
            detector,
            jnp.asarray(detector.all_points),
            detector.S_radius,
            sensor_shape="disc",
        )


def test_unknown_candidate_selection_is_rejected(tmp_path):
    detector = _one_pmt_detector(tmp_path)
    with pytest.raises(ValueError, match="sensor_candidate_selection"):
        create_propagator(
            detector,
            jnp.asarray(detector.all_points),
            detector.S_radius,
            sensor_shape="sk20inch",
            sensor_candidate_selection="nearest-ish",
        )


def test_wall_candidate_selection_requires_sk_shape(tmp_path):
    detector = _one_pmt_detector(tmp_path)
    with pytest.raises(ValueError, match="requires"):
        create_propagator(
            detector,
            jnp.asarray(detector.all_points),
            detector.S_radius,
            sensor_shape="sphere",
            sensor_candidate_selection="wall",
        )


def test_prebuilt_lookup_must_match_temperature(tmp_path, lookup):
    detector = _one_pmt_detector(tmp_path)
    with pytest.raises(ValueError, match="sigma"):
        create_propagator(
            detector,
            jnp.asarray(detector.all_points),
            detector.S_radius,
            temperature=0.2,
            sensor_shape="sk20inch",
            sk_pmt_lookup=lookup,
            max_candidates_per_ray=1,
        )


def test_lookup_options_cannot_override_physical_width(tmp_path):
    detector = _one_pmt_detector(tmp_path)
    with pytest.raises(ValueError, match="must not set sigma"):
        create_propagator(
            detector,
            jnp.asarray(detector.all_points),
            detector.S_radius,
            temperature=0.2,
            sensor_shape="sk20inch",
            sk_pmt_lookup_options={"sigma": 0.01},
            max_candidates_per_ray=1,
        )
