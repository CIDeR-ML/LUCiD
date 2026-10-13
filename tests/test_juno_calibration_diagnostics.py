"""Tests for the minimal closure AD--FD audit."""

import jax.numpy as jnp
import numpy as np
import pytest

from lucid.juno_laser.diagnostics import objective_ad_fd_comparison


def test_ad_fd_comparison_matches_quadratic_gradient():
    """Centered differences reproduce AD for both fit coordinates."""
    matrix = jnp.asarray((1.0, 2.0))

    def objective(theta, keys):
        del keys
        return jnp.sum(matrix * theta**2)

    theta = jnp.asarray((0.4, -0.8))
    result = objective_ad_fd_comparison(
        objective,
        theta,
        jnp.zeros((1, 2), dtype=jnp.uint32),
        relative_step=1.0e-2,
    )
    expected = 2.0 * np.asarray(matrix) * np.asarray(theta)
    np.testing.assert_allclose(result["ad_gradient"], expected, rtol=1e-6)
    np.testing.assert_allclose(
        result["finite_difference_gradient"], expected, rtol=2e-4
    )


def test_ad_fd_rejects_nonpositive_step():
    """Audit step sizes are explicit positive controls."""
    with pytest.raises(ValueError, match="steps must be positive"):
        objective_ad_fd_comparison(
            lambda theta, _keys: jnp.sum(theta**2),
            jnp.ones(2),
            jnp.zeros((1, 2), dtype=jnp.uint32),
            relative_step=0.0,
        )
