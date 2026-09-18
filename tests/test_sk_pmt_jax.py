"""Forward-value and gradient tests for the JAX SK PMT intersection."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from lucid.propagation.sk_pmt import intersect_sk20inch_pmt_hard
from lucid.propagation.sk_pmt_jax import (
    SK_PMT_SURFACE_MISS,
    SK_PMT_SURFACE_SPHERE,
    SK_PMT_SURFACE_TORUS,
    intersect_sk20inch_pmt_jax,
)


ORIGIN = jnp.zeros(3)
AXIS = jnp.array([0.0, 0.0, 1.0])
DIRECTION = jnp.array([0.0, 0.0, -1.0])


@pytest.mark.parametrize(
    "x,expected_surface",
    [
        (0.0, SK_PMT_SURFACE_SPHERE),
        (0.10, SK_PMT_SURFACE_SPHERE),
        (0.20, SK_PMT_SURFACE_SPHERE),
        (0.24, SK_PMT_SURFACE_TORUS),
        (0.253, SK_PMT_SURFACE_TORUS),
        (0.255, SK_PMT_SURFACE_MISS),
    ],
)
def test_known_forward_values_match_hard_oracle(x, expected_surface):
    ray_origin = np.array([x, 0.0, 1.0])
    hard = intersect_sk20inch_pmt_hard(
        ray_origin, DIRECTION, ORIGIN, AXIS)
    differentiable = intersect_sk20inch_pmt_jax(
        ray_origin, DIRECTION, ORIGIN, AXIS)

    assert bool(differentiable.intersects) == hard.intersects
    assert int(differentiable.surface) == expected_surface
    if hard.intersects:
        assert float(differentiable.distance) == pytest.approx(
            hard.distance, abs=2e-6)
        npt.assert_allclose(
            differentiable.position, hard.position, atol=2e-6)
        npt.assert_allclose(
            differentiable.normal, hard.normal, atol=2e-5)
        assert float(differentiable.incidence_cosine) == pytest.approx(
            hard.incidence_cosine, abs=2e-5)


def test_jit_and_vmap_match_hard_oracle_for_random_oblique_rays():
    rng = np.random.default_rng(20260918)
    origins = np.column_stack([
        rng.uniform(-0.35, 0.35, 500),
        rng.uniform(-0.35, 0.35, 500),
        np.full(500, 0.80),
    ])
    directions = np.column_stack([
        rng.uniform(-0.8, 0.8, 500),
        rng.uniform(-0.8, 0.8, 500),
        np.full(500, -1.0),
    ])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    batched_intersection = jax.jit(jax.vmap(
        lambda origin, direction: intersect_sk20inch_pmt_jax(
            origin, direction, ORIGIN, AXIS)
    ))
    jax_hits = batched_intersection(
        jnp.asarray(origins), jnp.asarray(directions))
    hard_hits = [
        intersect_sk20inch_pmt_hard(origin, direction, np.zeros(3),
                                    np.array([0.0, 0.0, 1.0]))
        for origin, direction in zip(origins, directions)
    ]
    hard_mask = np.array([hit.intersects for hit in hard_hits])

    npt.assert_array_equal(np.asarray(jax_hits.intersects), hard_mask)
    assert hard_mask.sum() > 50
    hard_distances = np.array([hit.distance for hit in hard_hits])[hard_mask]
    npt.assert_allclose(
        np.asarray(jax_hits.distance)[hard_mask],
        hard_distances,
        atol=3e-5,
        rtol=2e-5,
    )
    hard_positions = np.stack(
        [hit.position for hit in hard_hits if hit.intersects])
    npt.assert_allclose(
        np.asarray(jax_hits.position)[hard_mask],
        hard_positions,
        atol=3e-5,
        rtol=2e-5,
    )
    hard_normals = np.stack(
        [hit.normal for hit in hard_hits if hit.intersects])
    npt.assert_allclose(
        np.asarray(jax_hits.normal)[hard_mask],
        hard_normals,
        atol=8e-5,
        rtol=5e-5,
    )
    hard_surfaces = np.array([
        SK_PMT_SURFACE_SPHERE if hit.surface == "sphere"
        else SK_PMT_SURFACE_TORUS if hit.surface == "torus"
        else SK_PMT_SURFACE_MISS
        for hit in hard_hits
    ])
    npt.assert_array_equal(np.asarray(jax_hits.surface), hard_surfaces)


def _vertical_distance(x):
    hit = intersect_sk20inch_pmt_jax(
        jnp.array([x, 0.0, 1.0]), DIRECTION, ORIGIN, AXIS)
    return hit.distance


def test_sphere_root_gradient_matches_analytic_derivative():
    x = 0.10
    expected = x / np.sqrt(0.315**2 - x**2)
    derivative = jax.grad(_vertical_distance)(x)

    assert np.isfinite(derivative)
    assert float(derivative) == pytest.approx(expected, rel=2e-5, abs=2e-6)


def test_torus_newton_root_gradient_matches_implicit_derivative():
    x = 0.24
    z = np.sqrt(0.150**2 - (x - 0.104)**2)
    expected = (x - 0.104) / z
    derivative = jax.grad(_vertical_distance)(x)

    assert np.isfinite(derivative)
    assert float(derivative) == pytest.approx(expected, rel=3e-5, abs=3e-5)


def test_position_and_normal_jacobians_are_finite_on_both_surfaces():
    def position_and_normal(x):
        hit = intersect_sk20inch_pmt_jax(
            jnp.array([x, 0.0, 1.0]), DIRECTION, ORIGIN, AXIS)
        return jnp.concatenate([hit.position, hit.normal])

    for x in (0.0, 0.10, 0.24):
        jacobian = jax.jacrev(position_and_normal)(x)
        assert bool(jnp.all(jnp.isfinite(jacobian)))


def test_gradients_flow_through_world_transform():
    def distance_from_pmt_x(pmt_x):
        hit = intersect_sk20inch_pmt_jax(
            ray_origin=jnp.array([-1.0, 2.0, 3.0]),
            ray_direction=jnp.array([1.0, 0.0, 0.0]),
            pmt_position=jnp.array([pmt_x, 2.0, 3.0]),
            pmt_direction=jnp.array([-1.0, 0.0, 0.0]),
        )
        return hit.distance

    derivative = jax.grad(distance_from_pmt_x)(0.0)
    assert float(derivative) == pytest.approx(1.0, abs=2e-6)


def test_barrel_band_changes_active_flag_without_changing_intersection():
    ray_origin = jnp.array([0.253, 0.0, 1.0])
    cap = intersect_sk20inch_pmt_jax(
        ray_origin, DIRECTION, ORIGIN, AXIS, barrel=False)
    barrel = intersect_sk20inch_pmt_jax(
        ray_origin, DIRECTION, ORIGIN, AXIS, barrel=True)

    assert bool(cap.intersects) and bool(cap.active)
    assert bool(barrel.intersects) and not bool(barrel.active)
    npt.assert_allclose(cap.position, barrel.position)


@pytest.mark.parametrize("x", [0.10, 0.24])
def test_full_ray_and_pmt_geometry_jacobian_is_finite(x):
    """Origin, direction, PMT position, and PMT axis all carry gradients."""
    parameters = jnp.array([
        x, 0.02, 1.0,          # ray origin
        0.01, -0.02, -1.0,    # ray direction
        0.0, 0.0, 0.0,        # PMT origin
        0.01, 0.02, 1.0,      # inward PMT axis
    ])

    def observables(values):
        hit = intersect_sk20inch_pmt_jax(
            values[0:3], values[3:6], values[6:9], values[9:12])
        return jnp.concatenate([
            jnp.atleast_1d(hit.distance),
            hit.position,
            hit.normal,
            jnp.atleast_1d(hit.incidence_cosine),
        ])

    jacobian = jax.jacrev(observables)(parameters)
    assert bool(jnp.all(jnp.isfinite(jacobian)))
