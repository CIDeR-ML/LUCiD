"""Shared fixtures for the LUCiD test suite."""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import pytest
import jax
import jax.numpy as jnp


@pytest.fixture(scope="session")
def key():
    """Fixed PRNGKey for reproducible stochastic tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="session")
def small_cylinder_config():
    """Path to a small cylinder detector config."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "config", "WCTE_geom_config.json")


@pytest.fixture(scope="session")
def cylinder_detector(small_cylinder_config):
    """Build WCTE cylinder detector once per session."""
    from lucid.geometry import generate_detector
    return generate_detector(small_cylinder_config)


@pytest.fixture(scope="session")
def fixed_flat_hits():
    """Fixed flat arrays for make_hits tests."""
    return dict(
        flat_weights=jnp.array([0.5, 0.3, 0.8, 0.1, 0.6]),
        flat_indices=jnp.array([0, 2, 5, 5, 10]),
        flat_times=jnp.array([10.0, 15.0, 12.0, 20.0, 8.0]),
        num_detectors=20,
        qe_corrections=jnp.ones(20),
    )
