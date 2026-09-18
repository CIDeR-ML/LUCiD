"""Tests for smooth Gaussian coverage of the curved SK photocathode."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from scipy.stats import ncx2

from lucid.propagation import (
    create_sk20inch_surface_quadrature,
    intersect_sk20inch_pmt_jax,
    sk20inch_gaussian_coverage,
    sk20inch_projected_area,
)


AXIS = jnp.array([0.0, 0.0, 1.0])
PMT_POSITION = jnp.zeros(3)


def _direction(angle_degrees):
    angle = np.deg2rad(angle_degrees)
    return jnp.array([np.sin(angle), 0.0, -np.cos(angle)])


def test_normal_projected_area_is_aperture_area():
    cap = create_sk20inch_surface_quadrature(48, 96, barrel=False)
    barrel = create_sk20inch_surface_quadrature(48, 96, barrel=True)

    cap_area = sk20inch_projected_area(_direction(0.0), AXIS, cap)
    barrel_area = sk20inch_projected_area(_direction(0.0), AXIS, barrel)

    cap_radius = 0.254
    barrel_radius = 0.104 + np.sqrt(0.150**2 - 0.020**2)
    assert float(cap_area) == pytest.approx(
        np.pi * cap_radius**2, rel=2e-6)
    assert float(barrel_area) == pytest.approx(
        np.pi * barrel_radius**2, rel=2e-6)


def test_projected_area_matches_hard_ray_raster_at_oblique_angles():
    quadrature = create_sk20inch_surface_quadrature(40, 96)
    grid_size = 180
    side = 0.70
    coordinate = (
        np.linspace(-side / 2, side / 2, grid_size, endpoint=False)
        + side / (2 * grid_size)
    )
    first_coordinate, second_coordinate = np.meshgrid(
        coordinate, coordinate, indexing="ij")

    for angle_degrees in (0.0, 30.0, 60.0, 75.0, 90.0):
        angle = np.deg2rad(angle_degrees)
        direction = np.array([np.sin(angle), 0.0, -np.cos(angle)])
        first = np.array([np.cos(angle), 0.0, np.sin(angle)])
        second = np.array([0.0, 1.0, 0.0])
        origins = (
            -direction[None, :]
            + first_coordinate.reshape(-1, 1) * first[None, :]
            + second_coordinate.reshape(-1, 1) * second[None, :]
        )
        raster_fn = jax.jit(jax.vmap(
            lambda origin: intersect_sk20inch_pmt_jax(
                origin, jnp.asarray(direction), PMT_POSITION, AXIS).active
        ))
        raster_area = (
            np.asarray(raster_fn(jnp.asarray(origins))).mean() * side**2
        )
        surface_area = float(sk20inch_projected_area(
            jnp.asarray(direction), AXIS, quadrature))

        assert surface_area == pytest.approx(raster_area, rel=0.012)


def test_normal_gaussian_coverage_matches_exact_circular_aperture():
    quadrature = create_sk20inch_surface_quadrature(48, 96)
    sigma = 0.030
    radius = 0.254

    for offset in (0.0, 0.20, 0.254, 0.30):
        coverage = sk20inch_gaussian_coverage(
            ray_origin=jnp.array([offset, 0.0, 1.0]),
            ray_direction=_direction(0.0),
            pmt_position=PMT_POSITION,
            pmt_direction=AXIS,
            sigma=sigma,
            quadrature=quadrature,
        )
        # Squared radial distance of a displaced isotropic Gaussian follows a
        # noncentral chi-square distribution with two degrees of freedom.
        exact = ncx2.cdf((radius / sigma)**2, 2, (offset / sigma)**2)
        assert float(coverage) == pytest.approx(exact, abs=3e-6)


def test_oblique_gaussian_coverage_matches_convolved_hard_silhouette():
    quadrature = create_sk20inch_surface_quadrature(48, 96)
    grid_size = 260
    side = 0.70
    coordinate = (
        np.linspace(-side / 2, side / 2, grid_size, endpoint=False)
        + side / (2 * grid_size)
    )
    first_coordinate, second_coordinate = np.meshgrid(
        coordinate, coordinate, indexing="ij")
    angle = np.deg2rad(60.0)
    direction = np.array([np.sin(angle), 0.0, -np.cos(angle)])
    first = np.array([np.cos(angle), 0.0, np.sin(angle)])
    second = np.array([0.0, 1.0, 0.0])
    origins = (
        -direction[None, :]
        + first_coordinate.reshape(-1, 1) * first[None, :]
        + second_coordinate.reshape(-1, 1) * second[None, :]
    )
    raster_fn = jax.jit(jax.vmap(
        lambda origin: intersect_sk20inch_pmt_jax(
            origin, jnp.asarray(direction), PMT_POSITION, AXIS).active
    ))
    hard_silhouette = np.asarray(
        raster_fn(jnp.asarray(origins))).reshape(grid_size, grid_size)
    cell_area = (side / grid_size)**2
    sigma = 0.040

    for first_offset, second_offset in ((0.0, 0.0), (0.15, 0.04),
                                         (0.25, 0.0)):
        density = np.exp(-0.5 * (
            (first_coordinate - first_offset)**2
            + (second_coordinate - second_offset)**2
        ) / sigma**2) / (2.0 * np.pi * sigma**2)
        raster_coverage = np.sum(hard_silhouette * density) * cell_area
        ray_origin = (
            -direction + first_offset * first + second_offset * second)
        surface_coverage = sk20inch_gaussian_coverage(
            jnp.asarray(ray_origin), jnp.asarray(direction),
            PMT_POSITION, AXIS, sigma, quadrature)

        assert float(surface_coverage) == pytest.approx(
            raster_coverage, rel=0.004, abs=5e-4)


def test_backward_gradient_matches_forward_finite_difference_at_edge():
    quadrature = create_sk20inch_surface_quadrature(48, 96)
    sigma = 0.030

    def coverage(offset):
        return sk20inch_gaussian_coverage(
            jnp.array([offset, 0.0, 1.0]), _direction(0.0),
            PMT_POSITION, AXIS, sigma, quadrature)

    offset = 0.254
    autodiff = float(jax.grad(coverage)(offset))
    step = 2e-4
    finite_difference = float(
        (coverage(offset + step) - coverage(offset - step)) / (2 * step)
    )

    assert autodiff < 0.0
    assert autodiff == pytest.approx(finite_difference, rel=2e-3, abs=2e-3)


def test_all_continuous_geometry_inputs_and_sigma_have_finite_gradients():
    quadrature = create_sk20inch_surface_quadrature(24, 48)
    parameters = jnp.array([
        0.20, 0.03, 1.0,       # ray origin
        0.20, -0.05, -1.0,    # ray direction
        0.0, 0.0, 0.0,        # PMT position
        0.02, 0.01, 1.0,      # PMT direction
        0.035,                 # Gaussian sigma
    ])

    def coverage(values):
        return sk20inch_gaussian_coverage(
            values[0:3], values[3:6], values[6:9], values[9:12],
            values[12], quadrature)

    gradient = jax.grad(coverage)(parameters)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_rigid_rotation_leaves_coverage_unchanged():
    quadrature = create_sk20inch_surface_quadrature(36, 96)
    angle = np.deg2rad(37.0)
    rotation = jnp.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    origin = jnp.array([0.20, 0.04, 1.0])
    direction = jnp.array([0.10, -0.03, -1.0])
    sigma = 0.035

    reference = sk20inch_gaussian_coverage(
        origin, direction, PMT_POSITION, AXIS, sigma, quadrature)
    rotated = sk20inch_gaussian_coverage(
        rotation @ origin,
        rotation @ direction,
        rotation @ PMT_POSITION,
        rotation @ AXIS,
        sigma,
        quadrature,
    )

    assert float(rotated) == pytest.approx(float(reference), rel=2e-5)


@pytest.mark.parametrize("n_polar,n_azimuth", [(0, 8), (8, 0), (2.5, 8)])
def test_invalid_quadrature_orders_are_rejected(n_polar, n_azimuth):
    with pytest.raises(ValueError, match="positive integer"):
        create_sk20inch_surface_quadrature(n_polar, n_azimuth)
