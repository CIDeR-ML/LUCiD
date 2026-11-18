"""Light emitted outside a closed detector never reaches its sensors.

The inner detector is optically sealed, so a photon born outside it (by a track that leaves the
detector once the event is moved to its vertex) deposits nothing, even aimed straight through a
PMT from behind. On the tiny detector (``tests/helpers/tiny.py``).
"""
import jax
import numpy as np
import pytest

from tests.helpers import tiny

N, QE = tiny.N, tiny.QE


@pytest.fixture(scope="module")
def pmt(tiny_detector):
    """A mid-height barrel PMT: its index, its centre on the wall, and the outward normal there."""
    target, centre, out = tiny.barrel_target(tiny_detector)
    wall = np.array([tiny_detector.r * out[0], tiny_detector.r * out[1], centre[2]])
    return target, wall, out


def test_laser_behind_a_pmt_deposits_nothing(tiny_sim, pmt):
    from lucid.sources import laser_source
    target, centre, out = pmt
    # Reflecting walls and PMTs: whatever the surfaces do, nothing born outside gets in.
    params = tiny.transparent(tiny_sim.det_geom.num_sensors, wall_reflection_rate=0.05,
                              sensor_reflection_rate=0.25)

    def charge(origin, direction):
        source = laser_source(position=origin, direction=direction, intensity=N, fiber_NA=1e-6)
        return np.asarray(tiny_sim(source, params, jax.random.PRNGKey(5))[0])

    assert charge(centre + 0.5 * out, -out).sum() == 0.0              # outside, aimed in through it
    # Inside, aimed at its face: through transparent water every photon reaches the PMT and is
    # detected at the QE, whatever the PMT reflects (see test_incident_qe), so Q = N * QE.
    np.testing.assert_allclose(charge(centre - 0.5 * out, out)[target], N * QE, rtol=1e-3)


def test_photons_moved_outside_by_the_vertex_shift_go_dark(tiny_data_sim, pmt):
    """Data mode with known photons: N photons 0.5 m inside the PMT. Translated 0.5 m outward
    and aimed back in, they deposit nothing; left in place and aimed at the PMT, all N land on
    it and give exactly N x QE."""
    target, centre, out = pmt
    params = tiny.transparent(tiny_data_sim.det_geom.num_sensors)
    origins = np.tile(centre - 0.5 * out, (N, 1))

    def charge(direction, shift):
        data = tiny.photon_data(origins + shift, np.tile(direction, (N, 1)))
        return np.asarray(tiny_data_sim(params, jax.random.PRNGKey(5), data)[0])

    assert charge(-out, out).sum() == 0.0                          # moved 0.5 m outside, aimed in
    q = charge(out, np.zeros(3))                                    # left inside, aimed at its face
    np.testing.assert_allclose(q[target], N * QE, rtol=1e-4)   # float32 sum of N weights
    assert q.sum() == pytest.approx(q[target])
