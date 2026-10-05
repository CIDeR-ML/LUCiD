"""Light emitted outside a closed detector never reaches its sensors.

The inner detector is optically sealed, so a photon born outside it (by a track that leaves the
detector once the event is moved to its vertex) deposits nothing, even aimed straight through a
PMT from behind.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.slow

GEOM = os.path.join(os.path.dirname(__file__), '..', 'config', 'SK_like_geom_config.json')
N = 4096
QE = 0.2


@pytest.fixture(scope="module")
def pmt():
    """A mid-height barrel PMT: its index, its centre on the wall, and the outward normal there."""
    from lucid.geometry import generate_detector
    det = generate_detector(GEOM)
    barrel = np.asarray(det.barr_points)
    target = int(np.argmin(np.abs(barrel[:, 2])))      # barrel PMTs lead all_points
    x, y, z = barrel[target]
    out = np.array([x, y, 0.0]) / np.hypot(x, y)
    return det, target, np.array([det.r * out[0], det.r * out[1], z]), out


def _params(det):
    """Water made transparent, so only the surfaces act."""
    from lucid.detector_params import DetectorParams
    return DetectorParams.from_flat(
        scatter_length=1e9, mie_scatter_length=1e9, absorption_length=1e9,
        wall_reflection_rate=0.05, sensor_reflection_rate=0.25,
        qe=QE, qe_corrections=jnp.ones(len(det.all_points)))


def test_laser_behind_a_pmt_deposits_nothing(pmt):
    from lucid.simulation import setup_event_simulator
    from lucid.sources import laser_source
    det, target, centre, out = pmt
    sim = setup_event_simulator(GEOM, N, temperature=None, K=4, is_calibration=True,
                                wavelength_mode=False)

    def charge(origin, direction):
        source = laser_source(position=origin, direction=direction, intensity=N, fiber_NA=1e-6)
        return np.asarray(sim(source, _params(det), jax.random.PRNGKey(5))[0])

    assert charge(centre + 0.5 * out, -out).sum() == 0.0              # outside, aimed in through it
    assert charge(centre - 0.5 * out, out)[target] > 0.5 * N * QE      # inside, aimed at its face


def test_photons_moved_outside_by_the_vertex_shift_go_dark(pmt):
    from lucid.detector_params import ParticleParams
    from lucid.simulation import setup_event_simulator
    det, target, centre, out = pmt
    sim = setup_event_simulator(GEOM, 0, K=12, is_data=True, temperature=0.0, wavelength_mode=False,
                                default_detector_params=_params(det), hit_mode='realistic',
                                deposit_leg_bound=True)
    particle = ParticleParams.from_cartesian(1000.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])

    def charge(direction, shift):
        data = dict(
            photon_origins=jnp.asarray(np.tile((centre - 0.5 * out) * 100.0, (N, 1)), jnp.float32),
            photon_directions=jnp.asarray(np.tile(direction, (N, 1)), jnp.float32),
            photon_times=jnp.zeros(N), wavelengths=jnp.full(N, 400.0), N=jnp.int32(N),
            apply_rotation=False, rotation_axis=jnp.array([0.0, 0.0, 1.0]), rotation_angle=0.0,
            apply_translation=True, translation_vector=jnp.asarray(shift, jnp.float32))
        return np.asarray(sim(particle, jax.random.PRNGKey(5), data)[0])

    assert charge(-out, out).sum() == 0.0                         # moved 0.5 m outside, aimed in
    assert charge(out, np.zeros(3))[target] > 0.5 * N * QE         # left inside, aimed at its face
