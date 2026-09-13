"""Tests for the minimal whole-PMT surface reflection model."""

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy.testing as npt

from lucid.simulation.reflection import (
    PmtSurfaceReflection,
    get_reflection_model,
    pmt_surface_reflection,
)


_DIRECTION = jnp.asarray([0.0, 0.0, 1.0])
_NORMAL = jnp.asarray([0.0, 0.0, 1.0])
_ROOT = Path(__file__).resolve().parents[1]


def _params(sensor_rate, sensor_fspec):
    """Return minimal PMT-surface parameters for deterministic tests."""
    return PmtSurfaceReflection(
        wall_R0=jnp.asarray(0.05),
        wall_p=jnp.asarray(1.0),
        wall_fspec=jnp.asarray(0.0),
        sensor_rate=jnp.asarray(sensor_rate),
        sensor_fspec=jnp.asarray(sensor_fspec),
    )


def test_pmt_reflectance_is_effective_and_not_internal_optics():
    """The whole-PMT rate is independent of angle and wavelength."""
    params = _params(sensor_rate=0.23, sensor_fspec=1.0)
    values = []
    for direction, wavelength in (
        (jnp.asarray([0.0, 0.0, 1.0]), 400.0),
        (jnp.asarray([0.8, 0.0, 0.6]), 532.0),
        (jnp.asarray([0.99, 0.0, 0.141067]), 650.0),
    ):
        reflectance, _, _ = pmt_surface_reflection(
            direction,
            _NORMAL,
            True,
            params,
            jnp.asarray(wavelength),
            jax.random.PRNGKey(4),
        )
        values.append(float(reflectance))
    npt.assert_allclose(values, 0.23, atol=1e-7)


def test_specular_fraction_one_gives_mirror_direction():
    """A PMT reflection is exactly specular when its fraction is one."""
    _, reflected, _ = pmt_surface_reflection(
        _DIRECTION,
        _NORMAL,
        True,
        _params(sensor_rate=1.0, sensor_fspec=1.0),
        jnp.asarray(532.0),
        jax.random.PRNGKey(8),
    )
    npt.assert_allclose(reflected, [0.0, 0.0, -1.0], atol=1e-7)


def test_specular_fraction_zero_gives_inward_diffuse_direction():
    """A diffuse PMT reflection returns to the detector interior."""
    _, reflected, _ = pmt_surface_reflection(
        _DIRECTION,
        _NORMAL,
        True,
        _params(sensor_rate=1.0, sensor_fspec=0.0),
        jnp.asarray(532.0),
        jax.random.PRNGKey(9),
    )
    npt.assert_allclose(jnp.linalg.norm(reflected), 1.0, atol=1e-6)
    assert float(jnp.dot(reflected, -_NORMAL)) > 0.0


def test_registry_builds_only_effective_surface_parameters():
    """The registry reads total PMT reflectance and specular fraction."""
    function, builder = get_reflection_model("pmt_surface")
    detector_params = SimpleNamespace(
        reflection=SimpleNamespace(
            wall_R0=jnp.asarray(0.07),
            wall_p=jnp.asarray(2.0),
            wall_fspec=jnp.asarray(0.1),
            sensor_reflection_rate=jnp.asarray(0.31),
            sensor_fspec=jnp.asarray(0.65),
        )
    )
    parameters = builder(detector_params)
    assert function is pmt_surface_reflection
    assert isinstance(parameters, PmtSurfaceReflection)
    npt.assert_allclose(parameters.sensor_rate, 0.31)
    npt.assert_allclose(parameters.sensor_fspec, 0.65)


def test_effective_pmt_reflectance_has_a_finite_gradient():
    """The PMT total reflection rate remains directly differentiable."""
    def reflectance(sensor_rate):
        parameters = _params(
            sensor_rate=sensor_rate,
            sensor_fspec=0.5,
        )
        value, _, _ = pmt_surface_reflection(
            _DIRECTION,
            _NORMAL,
            True,
            parameters,
            jnp.asarray(532.0),
            jax.random.PRNGKey(10),
        )
        return value

    derivative = jax.grad(reflectance)(jnp.asarray(0.2))
    npt.assert_allclose(derivative, 1.0, atol=1e-7)


def test_juno_laser_baseline_selects_the_minimal_pmt_model():
    """The direct batch entry uses the reviewed whole-PMT baseline."""
    with (_ROOT / "config/JUNO_laser_config.json").open() as stream:
        run_config = json.load(stream)
    with (_ROOT / "config/JUNO_laser_physics_config.json").open() as stream:
        physics_config = json.load(stream)

    assert run_config["transport"]["reflection_model"] == "pmt_surface"
    assert physics_config["sensor_reflection_rate"] == 0.2
    assert physics_config["sensor_fspec"] == 0.5
