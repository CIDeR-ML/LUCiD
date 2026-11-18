"""The photon propagator's output contract, and what simple rays must return.

The propagator is the JIT-compiled function that takes photon positions and directions and
returns intersection data. Shapes and keys are checked for all three surface geometries; the
values for rays whose answer is plain geometry. Rays aimed at individual sensors, with the
overlap weight against its closed form, are in tests/propagation/test_propagation_aimed_rays.py.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from lucid.geometry import generate_detector
from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.propagation.shared import create_propagator
from tests.helpers.paths import REPO

pytestmark = pytest.mark.slow

EXPECTED_KEYS = {'sensor_weights', 'sensor_indices', 'times', 'positions',
                 'normals', 'inside_sensor', 'per_sensor_positions', 'sensor_normals'}


def _cfg(name):
    return os.path.join(REPO, 'config', f'{name}_geom_config.json')


@pytest.fixture(scope="module")
def dg():
    return DetectorGeometry.from_config(_cfg('WCTE_like'), temperature=0.2)


_CONTRACT = {}


def _contract_case(name):
    """(det, prop, origins, dirs, out) for 7 random rays; built once per geometry."""
    if name not in _CONTRACT:
        det = generate_detector(_cfg(name))
        # temperature=None: the contract does not depend on the overlap kernel; it needs no table
        prop, _ = create_propagator(det, jnp.asarray(det.all_points, jnp.float32),
                                    float(det.S_radius), temperature=None)
        dirs = jax.random.normal(jax.random.PRNGKey(42), (7, 3))
        dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
        origins = 0.1 * jax.random.normal(jax.random.PRNGKey(1), (7, 3))
        _CONTRACT[name] = (det, prop, origins, dirs, prop(origins, dirs))
    return _CONTRACT[name]


GEOMETRIES = ['WCTE_like', 'TAO', 'nuSCOPE']       # cylinder, sphere, box


@pytest.mark.parametrize("name", GEOMETRIES)
def test_output_contract(name):
    """Same keys and shapes for cylinder, sphere and box; indices are real sensors; the deposit
    over a ray's candidates is a probability."""
    det, prop, origins, dirs, out = _contract_case(name)
    n_sensors, n = len(det.all_points), len(origins)
    assert set(out) == EXPECTED_KEYS
    # K candidate slots: the default 4, raised by create_propagator when a cell holds more
    K = out['sensor_indices'].shape[0]
    assert K >= 4
    for k in ('sensor_weights', 'sensor_indices', 'inside_sensor'):
        assert out[k].shape == (K, n), (k, out[k].shape)
    assert out['times'].shape[:2] == (K, n)
    for k in ('per_sensor_positions', 'sensor_normals'):
        assert out[k].shape == (K, n, 3), (k, out[k].shape)
    for k in ('positions', 'normals'):
        assert out[k].shape == (n, 3), (k, out[k].shape)
    idx = np.asarray(out['sensor_indices'])
    assert idx.min() >= 0 and idx.max() < n_sensors
    w = np.asarray(out['sensor_weights'])
    assert w.min() >= 0 and w.sum(axis=0).max() <= 1 + 1e-6


@pytest.mark.parametrize("name", GEOMETRIES)
def test_a_single_unbatched_ray_is_refused(name):
    """The propagator takes (n_rays, 3) batches; the old 1-D path returned slot 0 only."""
    _, prop, origins, dirs, _ = _contract_case(name)
    with pytest.raises(ValueError, match="takes batches"):
        prop(origins[3], dirs[3])


def test_outward_ray_hits_barrel(dg):
    """A ray from the centre along +x stops on the barrel or on a barrel PMT in front of it:
    on the ray (y = z = 0) and at r - r_s <= x <= r."""
    out = dg.propagator(jnp.zeros((1, 3)), jnp.array([[1.0, 0.0, 0.0]]))
    p = np.asarray(out['positions'][0])
    r, r_s = dg.detector.r, float(dg.detector.S_radius)
    npt.assert_allclose(p[1:], 0.0, atol=1e-6)
    assert r - r_s - 1e-6 <= p[0] <= r + 1e-6, p


def _aimed_rays(dg, n=16):
    """Rays from the centre at the first n sensor centres: each enters its sensor."""
    S = np.asarray(dg.detector.all_points, np.float64)[:: len(dg.detector.all_points) // n][:n]
    d = S / np.linalg.norm(S, axis=1, keepdims=True)
    return jnp.zeros((n, 3)), jnp.asarray(d, jnp.float32)


def test_times_are_distances_along_the_ray(dg):
    """For every candidate the ray enters, the time is the distance from the origin to the
    entry point (the propagator's times are path lengths along a unit ray, in metres)."""
    o, d = _aimed_rays(dg)
    out = dg.propagator(o, d)
    entered = np.asarray(out['inside_sensor'])
    assert entered.any(axis=0).all(), "every ray aimed at a sensor centre must enter it"
    t = np.asarray(out['times']).reshape(entered.shape)
    dist = np.linalg.norm(np.asarray(out['per_sensor_positions']) - np.asarray(o)[None], axis=-1)
    npt.assert_allclose(t[entered], dist[entered], rtol=1e-6, atol=1e-6)
    assert (t[entered] > 0).all()


def test_normals_are_outward_unit_vectors(dg):
    """|n| = 1, and n points out of the volume along the ray (n . d > 0), for both the barrel and
    caps (axis-aligned rays) and sensor hits (rays at sensor centres)."""
    o1, d1 = _aimed_rays(dg)
    d2 = jnp.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [0, 0, -1.], [0.6, 0.0, 0.8]])
    for o, d in ((o1, d1), (jnp.zeros((5, 3)), d2)):
        n = np.asarray(dg.propagator(o, d)['normals'])
        npt.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)
        assert (np.sum(n * np.asarray(d), axis=1) > 0).all(), n
