"""Fixtures for the propagation and geometry tests."""
import pytest


@pytest.fixture(scope="session")
def cylinder_detector(small_cylinder_config):
    """Build the small generic cylinder detector once per session."""
    from lucid.geometry import generate_detector
    return generate_detector(small_cylinder_config)
