"""Tests for the cached production SK PMT coverage lookup."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from lucid.propagation import (
    build_sk20inch_coverage_lookup,
    create_sk20inch_surface_quadrature,
    sk20inch_gaussian_coverage,
    sk20inch_lookup_coordinates,
    sk20inch_lookup_coverage,
)


@pytest.fixture(scope="module")
def lookup():
    return build_sk20inch_coverage_lookup(
        0.030,
        n_incidence=31,
        grid_spacing=0.0075,
        tail_sigma=4.0,
        raster_oversample=4,
        cache=False,
    )


@pytest.fixture(scope="module")
def reference_quadrature():
    return create_sk20inch_surface_quadrature(48, 96)


def _ray(angle_degrees, parallel_offset, perpendicular_offset):
    angle = np.deg2rad(angle_degrees)
    direction = jnp.array([np.sin(angle), 0.0, -np.cos(angle)])
    parallel = jnp.array([np.cos(angle), 0.0, np.sin(angle)])
    perpendicular = jnp.array([0.0, -1.0, 0.0])
    origin = (
        -direction
        + parallel_offset * parallel
        + perpendicular_offset * perpendicular
    )
    return origin, direction


@pytest.mark.parametrize(
    "angle,parallel,perpendicular",
    [
        (0.0, 0.0, 0.0),
        (0.0, 0.254, 0.0),
        (37.0, 0.10, 0.03),
        (60.0, 0.15, 0.04),
        (75.0, 0.25, 0.0),
        (83.0, -0.10, 0.05),
    ],
)
def test_lookup_matches_surface_quadrature(
    lookup, reference_quadrature, angle, parallel, perpendicular,
):
    origin, direction = _ray(angle, parallel, perpendicular)
    table_value = sk20inch_lookup_coverage(
        origin, direction, jnp.zeros(3), jnp.array([0.0, 0.0, 1.0]),
        False, lookup)
    reference = sk20inch_gaussian_coverage(
        origin, direction, jnp.zeros(3), jnp.array([0.0, 0.0, 1.0]),
        0.030, reference_quadrature)

    assert float(table_value) == pytest.approx(
        float(reference), abs=2.5e-3, rel=0.06)


def test_lookup_gradient_matches_reference_at_photocathode_edge(
    lookup, reference_quadrature,
):
    direction = jnp.array([0.0, 0.0, -1.0])
    axis = jnp.array([0.0, 0.0, 1.0])

    def table_value(offset):
        return sk20inch_lookup_coverage(
            jnp.array([offset, 0.0, 1.0]), direction,
            jnp.zeros(3), axis, False, lookup)

    def reference_value(offset):
        return sk20inch_gaussian_coverage(
            jnp.array([offset, 0.0, 1.0]), direction,
            jnp.zeros(3), axis, 0.030, reference_quadrature)

    table_gradient = jax.grad(table_value)(0.254)
    reference_gradient = jax.grad(reference_value)(0.254)
    assert float(table_gradient) == pytest.approx(
        float(reference_gradient), rel=0.015)


def test_lookup_coordinates_are_world_rotation_invariant():
    origin, direction = _ray(53.0, 0.13, 0.06)
    pmt_position = jnp.array([1.0, -2.0, 0.5])
    origin = origin + pmt_position
    axis = jnp.array([0.0, 0.0, 1.0])
    angle = np.deg2rad(31.0)
    rotation = jnp.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])

    original = sk20inch_lookup_coordinates(
        origin, direction, pmt_position, axis)
    rotated = sk20inch_lookup_coordinates(
        rotation @ origin, rotation @ direction,
        rotation @ pmt_position, rotation @ axis)

    npt.assert_allclose(original, rotated, atol=2e-6)


def test_barrel_table_removes_inactive_rim(lookup):
    origin, direction = _ray(0.0, 0.253, 0.0)
    cap = sk20inch_lookup_coverage(
        origin, direction, jnp.zeros(3), jnp.array([0.0, 0.0, 1.0]),
        False, lookup)
    barrel = sk20inch_lookup_coverage(
        origin, direction, jnp.zeros(3), jnp.array([0.0, 0.0, 1.0]),
        True, lookup)
    assert float(barrel) < float(cap)


def test_cache_round_trip(tmp_path):
    options = dict(
        sigma=0.050,
        n_incidence=3,
        grid_spacing=0.025,
        tail_sigma=3.0,
        raster_oversample=1,
        cache=True,
        cache_dir=tmp_path,
    )
    generated = build_sk20inch_coverage_lookup(**options)
    loaded = build_sk20inch_coverage_lookup(**options)

    files = list(tmp_path.glob("sk20inch_*.npz"))
    assert len(files) == 1
    npt.assert_array_equal(generated.cap_coverage, loaded.cap_coverage)
    npt.assert_array_equal(generated.barrel_coverage, loaded.barrel_coverage)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"sigma": 0.0}, "sigma"),
        ({"sigma": 0.03, "n_incidence": 1}, "n_incidence"),
        ({"sigma": 0.03, "grid_spacing": 0.0}, "grid_spacing"),
        ({"sigma": 0.03, "tail_sigma": 2.0}, "tail_sigma"),
        ({"sigma": 0.03, "raster_oversample": 0}, "raster_oversample"),
    ],
)
def test_invalid_lookup_configuration_is_rejected(kwargs, error):
    with pytest.raises(ValueError, match=error):
        build_sk20inch_coverage_lookup(cache=False, **kwargs)
