"""Analytic checks for censored exponential free-path gradients."""

import jax
import jax.numpy as jnp
import numpy as np

from lucid.simulation.photon_step_factory import (
    free_path_log_probability,
    make_photon_step,
)
from lucid.simulation.reflection import ScalarReflection


REFLECTION = ScalarReflection(
    wall_rate=jnp.asarray(0.2),
    sensor_rate=jnp.asarray(0.2),
)


def _analytic_constant_gradient(log_length, distance, reach, scatter):
    """Return d E[reward] / d log(scattering length)."""
    rate = np.exp(-log_length)
    optical_depth = rate * distance
    return (
        (reach - scatter)
        * optical_depth
        * np.exp(-optical_depth)
    )


def _analytic_linear_gradient(log_length, distance, reach):
    """Return the analytic gradient for scatter reward equal to distance."""
    rate = np.exp(-log_length)
    x_value = rate * distance
    truncated = 1.0 - (x_value + 1.0) * np.exp(-x_value)
    scatter_gradient = (
        truncated / rate
        - rate * distance**2 * np.exp(-x_value)
    )
    return reach * x_value * np.exp(-x_value) + scatter_gradient


def _estimate(log_length, uniforms, distance, reach, mode):
    """Evaluate one score/pathwise Monte Carlo reward estimator."""
    rate = jnp.exp(-log_length)
    live_distance = -jnp.log1p(-uniforms) / rate
    sampled_distance = jax.lax.stop_gradient(live_distance)
    is_scatter = sampled_distance < distance
    log_probability = free_path_log_probability(
        is_scatter,
        rate,
        sampled_distance,
        distance,
    )
    stop = jax.lax.stop_gradient
    magic = jnp.exp(log_probability - stop(log_probability))
    if mode == "constant_score":
        reward = jnp.where(is_scatter, 0.4, reach)
        return jnp.mean(reward * magic)
    if mode == "linear_score":
        reward = jnp.where(is_scatter, sampled_distance, reach)
        return jnp.mean(reward * magic)
    if mode == "linear_combined":
        reward = jnp.where(is_scatter, live_distance, reach)
        return jnp.mean(reward * magic)
    raise ValueError(mode)


def test_free_path_score_matches_constant_reward_derivative():
    """The production score must reproduce an analytic derivative."""
    log_length = np.log(2.3)
    surface_distance = 1.7
    uniforms = jax.random.uniform(
        jax.random.PRNGKey(7),
        (1_000_000,),
    )
    gradient = jax.grad(_estimate)(
        log_length,
        uniforms,
        surface_distance,
        1.2,
        "constant_score",
    )
    expected = _analytic_constant_gradient(
        log_length,
        surface_distance,
        1.2,
        0.4,
    )
    np.testing.assert_allclose(gradient, expected, rtol=8.0e-3)


def test_full_score_and_live_distance_double_count_linear_reward():
    """A full score plus live free path is biased for time rewards."""
    log_length = np.log(2.3)
    surface_distance = 1.7
    reach_reward = 1.2
    uniforms = jax.random.uniform(
        jax.random.PRNGKey(11),
        (1_000_000,),
    )
    expected = _analytic_linear_gradient(
        log_length,
        surface_distance,
        reach_reward,
    )
    score = jax.grad(_estimate)(
        log_length,
        uniforms,
        surface_distance,
        reach_reward,
        "linear_score",
    )
    combined = jax.grad(_estimate)(
        log_length,
        uniforms,
        surface_distance,
        reach_reward,
        "linear_combined",
    )
    np.testing.assert_allclose(score, expected, rtol=1.5e-2)
    assert abs(float(combined) - expected) > 0.05


def test_production_free_path_uses_score_without_time_pathwise_gradient():
    """Production must retain the score and detach sampled arrival time."""
    step = make_photon_step("update_factors", False)
    fixed = (
        jnp.zeros(3),
        jnp.asarray([1.0, 0.0, 0.0]),
        jnp.asarray(2.0),
        jnp.asarray(1.0e6),
        jnp.asarray([-1.0, 0.0, 0.0]),
    )
    tail = (
        jnp.asarray(1.0e9),
        jnp.asarray(0.0),
        REFLECTION,
        jnp.asarray(1.0e9),
        jnp.asarray(False),
        jnp.asarray(532.0),
        jax.random.PRNGKey(17),
        jnp.asarray(0.225),
    )

    def evaluate(log_length):
        """Evaluate one production photon step."""
        return step(*fixed, jnp.exp(log_length), *tail)

    coordinate = jnp.log(jnp.asarray(30.0))
    time_gradient = jax.grad(lambda value: evaluate(value)[2])(
        coordinate
    )
    score_gradient = jax.grad(lambda value: evaluate(value)[6])(
        coordinate
    )
    assert float(time_gradient) == 0.0
    assert abs(float(score_gradient)) > 0.0
