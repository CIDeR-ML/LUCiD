"""Light arriving at a sensor is detected at the QE, whatever the sensor reflects.

PMT QE is quoted per photon arriving at the PMT, so the photon step's sensor reflection must not
reduce it a second time: a deposited photon converts at QE / (1 - R0). A pencil beam on a barrel
PMT of the tiny detector (``tests/helpers/tiny.py``), through water made transparent and with a
single transport step, isolates that encounter.
"""
import jax
import numpy as np
import pytest

from tests.helpers import tiny

N, QE = tiny.N, tiny.QE


@pytest.fixture(scope="module")
def beam(tiny_detector):
    """A pencil beam from the axis onto a mid-height barrel PMT, at normal incidence."""
    from lucid.sources import laser_source
    target, centre, out = tiny.barrel_target(tiny_detector)
    source = laser_source(position=[0.0, 0.0, centre[2]], direction=out.tolist(),
                          intensity=N, fiber_NA=1e-6)
    return target, source


def _params(sim, sensor_rate, qe=QE):
    """Water made transparent, so the sensor encounter is the only thing that acts."""
    return tiny.transparent(sim.det_geom.num_sensors, sensor_reflection_rate=sensor_rate, qe=qe)


def _target_charge(sim, beam, sensor_rate):
    target, source = beam
    charges, _ = sim(source, _params(sim, sensor_rate), jax.random.PRNGKey(3))
    charges = np.asarray(charges)
    assert charges[target] > 0.999 * charges.sum()      # the beam lands on the target
    return float(charges[target])


def test_expected_charge_is_qe_whatever_the_sensor_reflects(tiny_sim, beam):
    bare, reflecting = (_target_charge(tiny_sim, beam, r) for r in (0.0, 0.25))
    np.testing.assert_allclose(bare, N * QE, rtol=1e-3)
    np.testing.assert_allclose(reflecting, bare, rtol=1e-5)


def test_direct_charge_gradient_ignores_the_sensor_reflectance(tiny_sim, beam):
    target, source = beam
    grad = jax.grad(lambda dp: tiny_sim(source, dp, jax.random.PRNGKey(3))[0][target])(
        _params(tiny_sim, 0.25))
    assert abs(float(grad.reflection.sensor_reflection_rate)) < 1e-4 * N * QE
    np.testing.assert_allclose(float(grad.response.qe), N, rtol=1e-3)


def test_sampled_detection_is_qe_whatever_the_sensor_reflects(tiny_sampled_sim, beam):
    five_sigma = 5 * np.sqrt(N * QE * (1 - QE))
    for r in (0.0, 0.25):
        assert abs(_target_charge(tiny_sampled_sim, beam, r) - N * QE) < five_sigma, r


def test_angular_model_detects_qe_at_normal_incidence(beam):
    sim = tiny.simulator(reflection_model='angular')
    np.testing.assert_allclose(_target_charge(sim, beam, 0.0), N * QE, rtol=1e-3)


def test_setup_bounds_qe_by_the_non_reflected_fraction(tiny_sim):
    tiny.simulator(default_detector_params=_params(tiny_sim, 0.25, qe=0.75))
    with pytest.raises(ValueError, match="1 - R0"):
        tiny.simulator(default_detector_params=_params(tiny_sim, 0.25, qe=0.8))
