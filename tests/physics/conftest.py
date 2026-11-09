"""Fixtures for the physics tests: the tiny detector's other session simulators.

``tiny_sim`` is in ``tests/conftest.py``, because tests outside ``physics/`` use it too.
"""
import pytest


@pytest.fixture(scope="session")
def tiny_sampled_sim():
    """Calibration source, Bernoulli-sampled transport and detection, one step."""
    from tests.helpers import tiny
    return tiny.simulator(transport='sampled', detection='sampled')


@pytest.fixture(scope="session")
def tiny_data_sim():
    """Given photons, expected first-arrival charge, one step."""
    from tests.helpers import tiny
    return tiny.data_simulator()


@pytest.fixture(scope="session")
def tiny_detector(tiny_sim):
    """The tiny detector's geometry (barrel, caps, radius, sensor positions)."""
    return tiny_sim.det_geom.detector
