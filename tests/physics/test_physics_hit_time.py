"""End-to-end photon arrival time for a pencil beam: t = |x_sensor - x_0| n / c.

A laser on the axis of the tiny detector (``tests/helpers/tiny.py``) fires a pencil beam at
normal incidence onto a mid-height barrel PMT through water made transparent, so every photon
takes the straight path to the PMT's front surface, (r_barrel - r_PMT) from the axis. No transport
step is involved in the last leg: its time is the simulator's own ``hit distance / speed of
light`` (``simulator.py``, the ``times_ns`` line), which no step-level test covers.

The speed is the PHASE velocity c/n (n = 1.33 and the exact SI c from
``config/materials/water``). Whether the simulation should propagate at the group velocity
instead (0.2167 m/ns, 3.9 % slower, which ``lucid/fitting/seed.py`` already uses for its
multilateration seed) is an open physics question pending the owner's decision. These tests
pin what the simulator does today; when that decision is made, ``V_SIM`` changes with it.
"""
import jax
import numpy as np
import pytest

from tests.helpers import tiny

N, QE = tiny.N, tiny.QE
C_VACUUM_M_PER_NS = 0.299792458        # SI definition of the metre
N_WATER = 1.33                          # water's refractive_index in the material JSON
V_SIM = C_VACUUM_M_PER_NS / N_WATER     # phase velocity, 0.22541 m/ns: the material's c/n
V_GROUP = 0.2167                        # the seeder's group velocity, for contrast
OFFSET_M = 0.5                          # the second source position, this far out from the axis


@pytest.fixture(scope="module")
def target(tiny_detector):
    return tiny.barrel_target(tiny_detector)


def _beam(det, target, offset):
    """The pencil beam starting ``offset`` metres out from the axis toward the target."""
    from lucid.sources import laser_source
    _, centre, out = target
    start = np.array([0.0, 0.0, centre[2]]) + offset * out
    path = np.hypot(centre[0], centre[1]) - offset - float(det.S_radius)   # to the PMT's front
    return laser_source(position=start.tolist(), direction=out.tolist(), intensity=N,
                        fiber_NA=1e-6), path


def _target_hit(sim, det, target, offset, tts=0.0, key=3):
    index = target[0]
    source, path = _beam(det, target, offset)
    params = tiny.transparent(sim.det_geom.num_sensors, tts=tts)
    charges, times = (np.asarray(o) for o in sim(source, params, jax.random.PRNGKey(key)))
    assert charges[index] > 0.999 * charges.sum()      # the beam lands on the target
    return float(charges[index]), float(times[index]), path


@pytest.mark.parametrize("offset", [0.0, OFFSET_M])
def test_sampled_first_arrival_is_path_over_phase_velocity(tiny_sampled_sim, tiny_detector,
                                                           target, offset):
    """With no TTS the first photon arrives at exactly path / (c/n). Float32 time over a
    ~1 m path is good to ~1e-6 relative; rtol 1e-4 is 0.4 ps. The vacuum speed would be
    1.06 ns early, and the group velocity 0.17 ns late on the 0.96 m path."""
    _, t, path = _target_hit(tiny_sampled_sim, tiny_detector, target, offset)
    np.testing.assert_allclose(t, path / V_SIM, rtol=1e-4)
    assert abs(t - path / V_GROUP) > 0.03 * path / V_GROUP   # pins the current choice


def test_arrival_time_scales_with_the_path(tiny_sampled_sim, tiny_detector, target):
    """Moving the source 0.5 m toward the PMT moves the arrival 0.5 m / (c/n) = 2.218 ns
    earlier: the time is linear in the path, with no offset absorbing a speed error."""
    t = [_target_hit(tiny_sampled_sim, tiny_detector, target, off)[1] for off in (0.0, OFFSET_M)]
    np.testing.assert_allclose(t[0] - t[1], OFFSET_M / V_SIM, rtol=1e-4)


def test_sampled_first_arrival_with_tts(tiny_sampled_sim, tiny_detector, target):
    """With a per-photon Gaussian TTS sigma, the first of n detected photons arrives at
    t_geo + sigma * min(z_1..z_n). E[min of n N(0,1)] is computed here by quadrature of
    the order-statistic density n phi(x) S(x)^(n-1); for n ~ 800 it is -3.1 with a spread
    of 0.37, so 4 x 0.37 sigma bounds one draw."""
    from scipy.integrate import trapezoid
    from scipy.special import log_ndtr
    sigma = 1.0
    q, t, path = _target_hit(tiny_sampled_sim, tiny_detector, target, 0.0, tts=sigma)
    n = int(round(q))
    x = np.linspace(-10.0, 5.0, 30001)
    logf = np.log(n) - 0.5 * x * x - 0.5 * np.log(2 * np.pi) + (n - 1) * log_ndtr(-x)
    f = np.exp(logf)
    f /= trapezoid(f, x)
    mean_min = trapezoid(x * f, x)
    sd_min = np.sqrt(trapezoid((x - mean_min) ** 2 * f, x))
    t_geo = path / V_SIM
    assert abs(t - (t_geo + sigma * mean_min)) < 4 * sigma * sd_min, (t, t_geo, mean_min, n)


def test_expected_value_time_of_coincident_arrivals_is_their_arrival(tiny_sim, tiny_detector,
                                                                      target):
    """The expected-value forward reports the mean of the first-arrival distribution. Every ray
    of the pencil beam reaches the target at t_geo, so with no transit-time spread the first
    photo-electron arrives at t_geo exactly, however many there are."""
    _, t, path = _target_hit(tiny_sim, tiny_detector, target, 0.0)
    np.testing.assert_allclose(t, path / V_SIM, atol=1e-4)
