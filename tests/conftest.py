"""Shared fixtures for the LUCiD test suite."""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import pytest
import jax
import jax.numpy as jnp

# Files marked @pytest.mark.slow — skip importing them unless --slow is passed.
_SLOW_FILES = [
    "test_containers.py",
    "test_e2e_wavelength.py",
    "test_integration.py",
    "test_optics_physics.py",
    "test_photon_step.py",
    "test_photon_step_physics.py",
    "test_propagation_differentiability.py",
    "test_propagator_output.py",
    "test_ray_intersection.py",
    "test_sensor_map_validation.py",
    "test_shared_propagator.py",
    "test_shared_propagator_differentiability.py",
    "test_shotgun_waveform.py",
    "test_sk_like_integration.py",
    "test_wavelength_integration.py",
    "test_qe_importance_sampling.py",
]


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False,
                     help="Include slow tests (detector/propagator/simulation)")


def pytest_ignore_collect(collection_path, config):
    if config.getoption("--slow", default=False):
        return False
    return collection_path.name in _SLOW_FILES


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
