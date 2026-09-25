"""Deterministic tests for the JUNO green diffuser-ball source."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lucid.sources import isotropic_source
from lucid.sources.diffuser_ball import (
    JUNO_LS_PHASE_WEIGHTS,
    DiffuserBallSource,
    diffuser_ball_source,
    sample_source,
)
from lucid.juno_laser.source import (
    build_juno_laser_source,
    rotate_source_directions,
)
from lucid.juno_laser.config import JunoLaserConfig


def test_nominal_diffuser_matches_junosw_moments():
    """Weighted proposal rays reproduce the J26.5.1 LSPhase moments."""
    source = diffuser_ball_source(
        [0.0, 0.0, 0.0], intensity=2.5e6
    )
    batch = source(200_000, jax.random.PRNGKey(42))
    normals = batch.positions / source.radius
    local_cosine = jnp.sum(normals * batch.directions, axis=1)
    weighted_normal_z = jnp.sum(batch.weights * normals[:, 2])
    weighted_normal_z /= jnp.sum(batch.weights)
    weighted_direction_z = jnp.sum(
        batch.weights * batch.directions[:, 2]
    )
    weighted_direction_z /= jnp.sum(batch.weights)

    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(batch.positions), axis=1),
        0.010,
        rtol=2e-5,
        atol=2e-7,
    )
    assert float(jnp.min(local_cosine)) >= -2e-6
    assert float(jnp.max(local_cosine)) <= 1.0 + 2e-6
    np.testing.assert_allclose(
        float(jnp.mean(local_cosine)), 0.5, atol=3e-3
    )
    np.testing.assert_allclose(weighted_normal_z, -0.05341225, atol=3e-3)
    np.testing.assert_allclose(
        weighted_direction_z, -0.02670612, atol=3e-3
    )
    np.testing.assert_allclose(jnp.sum(batch.weights), 2.5e6, rtol=2e-6)
    np.testing.assert_allclose(jnp.std(batch.times), 0.23, atol=3e-3)


def test_diffuser_sampling_is_reproducible():
    """The same explicit PRNG key returns the same source phase space."""
    source = diffuser_ball_source([1.0, -2.0, 3.0])
    first = source(128, jax.random.PRNGKey(7))
    second = source(128, jax.random.PRNGKey(7))
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)


def test_diffuser_phase_space_remains_float32_with_x64_enabled():
    """The source must not promote the full transport chain to float64."""
    assert jax.config.x64_enabled
    source = diffuser_ball_source([0.0, 0.0, 0.0])
    batch = source(128, jax.random.PRNGKey(13))

    for value in batch:
        assert value.dtype == jnp.float32


def test_angular_logits_have_finite_nonzero_gradient():
    """Angular coefficients change shape without changing total intensity."""
    source = diffuser_ball_source([0.0, 0.0, 0.0])
    key = jax.random.PRNGKey(11)

    def weighted_mean(logits):
        varied = DiffuserBallSource(
            source.center,
            source.intensity,
            source.axis,
            source.radius,
            source.time_sigma,
            logits,
            source.wavelength,
        )
        batch = varied(4096, key)
        normals = batch.positions / source.radius
        return jnp.sum(batch.weights * normals[:, 2]) / jnp.sum(
            batch.weights
        )

    gradient = jax.grad(weighted_mean)(source.angular_logits)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 1e-3


def test_legacy_source_adapter_adds_zero_emission_times():
    """Existing three-array sources retain their historical semantics."""
    source = isotropic_source([0.0, 0.0, 0.0], intensity=100.0)
    batch = sample_source(source, 64, jax.random.PRNGKey(3))
    np.testing.assert_array_equal(batch.times, np.zeros(64))
    np.testing.assert_allclose(jnp.sum(batch.weights), 100.0)


def test_nominal_weights_are_positive_and_normalized():
    """The Bernstein representation has three independent shape degrees."""
    weights = np.asarray(JUNO_LS_PHASE_WEIGHTS)
    assert weights.shape == (4,)
    assert np.all(weights > 0.0)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-7)


def test_juno_source_factory_and_rotation_policy():
    """Config selection keeps the diffuser asymmetry fixed to its axis."""
    config = {
        "model": "juno_diffuser_ball",
        "axis": [0.0, 0.0, 1.0],
        "radius_m": 0.010,
        "time_sigma_ns": 0.23,
        "angular_weights": list(JUNO_LS_PHASE_WEIGHTS),
    }
    source = build_juno_laser_source(
        config,
        [0.0, 0.0, 1.0],
        intensity=1000.0,
        wavelength_nm=532.0,
    )
    assert isinstance(source, DiffuserBallSource)
    directions = jnp.eye(3)
    actual = rotate_source_directions(
        source, directions, jax.random.PRNGKey(1)
    )
    np.testing.assert_array_equal(actual, directions)


def test_repository_config_selects_diffuser_baseline():
    """The production JUNO laser config enables the new source baseline."""
    root = Path(__file__).resolve().parents[1]
    config = JunoLaserConfig.from_file(
        root / "config" / "JUNO_laser_config.json"
    )
    assert config.data["source"]["model"] == "juno_diffuser_ball"
