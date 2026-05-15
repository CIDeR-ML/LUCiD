"""Shared fixtures for the LUCiD test suite."""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import pytest
import jax
import jax.numpy as jnp

# Files marked @pytest.mark.slow — skip importing them unless --slow is passed.
_SLOW_FILES = [
    "unit/test_containers.py",
    "physics/test_optics_physics.py",
    "physics/test_photon_step_physics.py",
    "propagation/test_photon_step.py",
    "propagation/test_propagator_output.py",
    "propagation/test_shared_propagator.py",
    "propagation/test_shared_propagator_diff.py",
    "propagation/test_propagation_diff.py",
    "geometry/test_ray_intersection.py",
    "geometry/test_sensor_map.py",
    "integration/test_integration.py",
    "integration/test_sk_like_integration.py",
    "integration/test_wavelength_integration.py",
    "integration/test_qe_importance_sampling.py",
    "integration/test_shotgun_waveform.py",
    "e2e/test_e2e_wavelength.py",
]


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False,
                     help="Include slow tests (detector/propagator/simulation)")


def pytest_ignore_collect(collection_path, config):
    if config.getoption("--slow", default=False):
        return False
    # Match relative path from tests/ directory
    tests_root = collection_path.parent
    while tests_root.name != "tests" and tests_root != tests_root.parent:
        tests_root = tests_root.parent
    try:
        rel = str(collection_path.relative_to(tests_root))
    except ValueError:
        return False
    return rel in _SLOW_FILES


@pytest.fixture(scope="session")
def key():
    """Fixed PRNGKey for reproducible stochastic tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="session")
def small_cylinder_config():
    """Path to a small generic cylinder detector config (WCTE-shaped,
    algorithmic placement). The real-WCTE config (``WCTE_geom_config``)
    loads measured PMT positions from a separate npz file and is not
    suitable as a generic cylinder fixture."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "config", "WCTE_like_geom_config.json")


@pytest.fixture(scope="session")
def cylinder_detector(small_cylinder_config):
    """Build the small generic cylinder detector once per session."""
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
