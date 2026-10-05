"""Light arriving at a sensor is detected at the QE, whatever the sensor reflects.

PMT QE is quoted per photon arriving at the PMT, so the photon step's sensor reflection must not
reduce it a second time: a deposited photon converts at QE / (1 - R0). A pencil beam on a barrel
PMT, through water made transparent and with a single transport step, isolates that encounter.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.slow

GEOM = os.path.join(os.path.dirname(__file__), '..', 'config', 'SK_like_geom_config.json')
N = 100_000
QE = 0.2


@pytest.fixture(scope="module")
def beam():
    """A pencil beam from the axis onto a mid-height barrel PMT, at normal incidence."""
    from lucid.geometry import generate_detector
    from lucid.sources import laser_source
    det = generate_detector(GEOM)
    barrel = np.asarray(det.barr_points)
    target = int(np.argmin(np.abs(barrel[:, 2])))      # barrel PMTs lead all_points
    x, y, z = barrel[target]
    rho = np.hypot(x, y)
    source = laser_source(position=[0.0, 0.0, z], direction=[x / rho, y / rho, 0.0],
                          intensity=N, fiber_NA=1e-6)
    return det, target, source


def _params(det, sensor_rate, qe=QE):
    """Water made transparent, so the sensor encounter is the only thing that acts."""
    from lucid.detector_params import DetectorParams
    return DetectorParams.from_flat(
        scatter_length=1e9, mie_scatter_length=1e9, absorption_length=1e9,
        wall_reflection_rate=0.0, sensor_reflection_rate=sensor_rate,
        qe=qe, qe_corrections=jnp.ones(len(det.all_points)))


def _simulator(**setup):
    from lucid.simulation import setup_event_simulator
    return setup_event_simulator(GEOM, N, temperature=None, K=1, is_calibration=True,
                                 wavelength_mode=False, **setup)


def _target_charge(sim, beam, sensor_rate):
    det, target, source = beam
    charges, _ = sim(source, _params(det, sensor_rate), jax.random.PRNGKey(3))
    charges = np.asarray(charges)
    assert charges[target] > 0.999 * charges.sum()      # the beam lands on the target
    return float(charges[target])


def test_expected_charge_is_qe_whatever_the_sensor_reflects(beam):
    sim = _simulator()
    bare, reflecting = (_target_charge(sim, beam, r) for r in (0.0, 0.25))
    np.testing.assert_allclose(bare, N * QE, rtol=1e-3)
    np.testing.assert_allclose(reflecting, bare, rtol=1e-5)


def test_direct_charge_gradient_ignores_the_sensor_reflectance(beam):
    det, target, source = beam
    sim = _simulator()
    grad = jax.grad(lambda dp: sim(source, dp, jax.random.PRNGKey(3))[0][target])(
        _params(det, 0.25))
    assert abs(float(grad.reflection.sensor_reflection_rate)) < 1e-4 * N * QE
    np.testing.assert_allclose(float(grad.response.qe), N, rtol=1e-3)


def test_sampled_detection_is_qe_whatever_the_sensor_reflects(beam):
    sim = _simulator(use_expected_value=False, hit_mode='realistic')
    five_sigma = 5 * np.sqrt(N * QE * (1 - QE))
    for r in (0.0, 0.25):
        assert abs(_target_charge(sim, beam, r) - N * QE) < five_sigma, r


def test_angular_model_detects_qe_at_normal_incidence(beam):
    sim = _simulator(reflection_model='angular')
    np.testing.assert_allclose(_target_charge(sim, beam, 0.0), N * QE, rtol=1e-3)


def test_setup_bounds_qe_by_the_non_reflected_fraction(beam):
    det = beam[0]
    _simulator(default_detector_params=_params(det, 0.25, qe=0.75))
    with pytest.raises(ValueError, match="1 - R0"):
        _simulator(default_detector_params=_params(det, 0.25, qe=0.8))
