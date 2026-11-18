"""The string (per-DOM volume) propagator, against geometry worked out here.

The string propagator weights every candidate DOM independently by the disk-under-Gaussian
overlap of its closest approach to the ray (no first-hit product), so for a ray whose closest
approach to a DOM centre is rho the DOM's weight is exactly

    f(rho) = ncx2.cdf((r/sigma)^2, 2, (rho/sigma)^2),   sigma = temperature * r

(see tests/propagation/test_propagation_aimed_rays.py), and the reported ``sensor_distances`` is the
closest-approach parameter t_c = (c - o).d, clamped at 0.

Verifies:
  1. Output dict has the correct keys and shapes
  2. A ray aimed past a DOM lists that DOM with weight f(rho) and distance t_c, and its
     gradient is f'(rho) -- for one hand-placed ray and for 400 random ones (the DOM snap)
  3. Photons far from DOMs get zero weight and stop on the instrumented envelope
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import ncx2

from lucid.geometry.string import StringTelescope
from lucid.propagation.string.string_propagator import create_string_propagator
from tests.helpers.paths import REPO

PROJECT_ROOT = str(REPO)
NPZ = os.path.join(PROJECT_ROOT, "config", "icecube86_simple.npz")
TEMPERATURE = 0.2


@pytest.fixture(scope="module")
def built():
    det = StringTelescope.from_npz(NPZ)
    prop = create_string_propagator(det, det.S_radius, temperature=TEMPERATURE)
    return det, prop


def _f(rho, r):
    s = TEMPERATURE * r
    return ncx2.cdf((r / s) ** 2, 2, (np.asarray(rho) / s) ** 2)


def _f_slope(rho, r):
    s = TEMPERATURE * r
    x, lam = (r / s) ** 2, (rho / s) ** 2
    return -(rho / s ** 2) * (ncx2.cdf(x, 2, lam) - ncx2.cdf(x, 4, lam))


def _dom(det, string, k):
    """Index and centre of DOM k on a string, located by position (not by the id tables)."""
    c = np.asarray(det.string_anchors[string]) + np.array([0, 0, float(det.dom_s_offsets[string, k])])
    idx = int(np.argmin(np.linalg.norm(np.asarray(det.all_points) - c, axis=1)))
    assert np.linalg.norm(np.asarray(det.all_points)[idx] - c) < 1e-6
    return idx, c


def test_output_contract(built):
    det, prop = built
    n_cand = prop.n_cand
    n = 8
    o = jnp.tile(jnp.array([0.0, 0.1, -1950.0]), (n, 1))
    d = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (n, 1))
    r = prop(o, d)
    expected = {'sensor_weights', 'sensor_indices', 'sensor_distances',
                'positions', 'normals', 'inside_sensor',
                'per_sensor_positions', 'sensor_normals',
                'envelope_exit_t'}
    assert set(r.keys()) == expected, f"missing keys: {expected - set(r.keys())}"
    assert r['sensor_weights'].shape == (n_cand, n), f"weights: {r['sensor_weights'].shape}"
    assert r['sensor_indices'].shape == (n_cand, n), f"indices: {r['sensor_indices'].shape}"
    assert r['sensor_distances'].shape == (n_cand, n, 1), f"distances: {r['sensor_distances'].shape}"
    assert r['positions'].shape == (n, 3)
    assert r['normals'].shape == (n, 3)
    assert r['inside_sensor'].shape == (n_cand, n)
    assert r['per_sensor_positions'].shape == (n_cand, n, 3)
    assert r['sensor_normals'].shape == (n_cand, n, 3)
    assert r['envelope_exit_t'].shape == (n,)


def test_near_dom_hits(built):
    """A ray along +x from 5 m before DOM 30 of string 0, offset 0.1 m in y: the DOM is the
    heaviest candidate, with weight f(0.1 m), distance 5 m, and d w / d y = f'(0.1 m)."""
    det, prop = built
    r_s = float(det.S_radius)
    idx, c = _dom(det, 0, 30)
    d = jnp.array([[1.0, 0.0, 0.0]])

    def run(y):
        return prop(jnp.array([[c[0] - 5.0, y, c[2]]]), d)

    out = run(c[1] + 0.1)
    w = np.asarray(out['sensor_weights'])[:, 0]
    ids = np.asarray(out['sensor_indices'])[:, 0]
    slot = int(np.argmax(w))
    assert ids[slot] == idx, (ids, w)
    np.testing.assert_allclose(w[slot], _f(0.1, r_s), atol=1e-4)
    np.testing.assert_allclose(float(out['sensor_distances'][slot, 0, 0]), 5.0, atol=1e-4)
    # Every candidate, not only the target, carries f of its own closest approach (this ray
    # also passes DOM 30 of string 1, 125 m on, at the same 0.1 m).
    S = np.asarray(det.all_points, np.float64)
    o = np.array([c[0] - 5.0, c[1] + 0.1, c[2]])
    v = S[ids] - o
    tc = np.maximum(v[:, 0], 0.0)
    rho = np.linalg.norm(v - tc[:, None] * np.array([1.0, 0, 0]), axis=1)
    np.testing.assert_allclose(w, _f(rho, r_s), atol=1e-4)

    g = float(jax.grad(lambda y: run(y)['sensor_weights'][slot, 0])(jnp.float32(c[1] + 0.1)))
    np.testing.assert_allclose(g, _f_slope(0.1, r_s), rtol=3e-2)


def test_aimed_rays_find_their_dom(built):
    """400 rays, each aimed past a random DOM at impact <= 0.8 r from 20 m away in a random
    direction (not near the string axis), must list that DOM with weight f(rho) and distance
    t_c. This is the DOM snap: the candidate pair is chosen from the ray's position along the
    nearest strings, and a snap one DOM off loses the target."""
    det, prop = built
    S = np.asarray(det.all_points, np.float64)
    r_s = float(det.S_radius)
    rng = np.random.default_rng(0)
    k = rng.choice(len(S), size=400, replace=False)
    u = rng.normal(size=(400, 3))
    u[:, 2] *= 0.5                                           # keep away from the string axis
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    e = np.cross(u, rng.normal(size=(400, 3)))
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    rho = 0.8 * r_s * np.sqrt(rng.uniform(size=400))
    o = S[k] + rho[:, None] * e - 20.0 * u
    out = prop(jnp.asarray(o, jnp.float32), jnp.asarray(u, jnp.float32))
    ids = np.asarray(out['sensor_indices'])
    hit = ids == k[None, :]
    assert hit.any(axis=0).all(), f"{(~hit.any(axis=0)).sum()}/400 rays miss their DOM"
    w = np.sum(np.where(hit, np.asarray(out['sensor_weights']), 0), axis=0)
    t = np.sum(np.where(hit, np.asarray(out['sensor_distances'])[..., 0], 0), axis=0)
    # float32 at |o| ~ 2 km: closest-approach distance good to ~1e-4 m, i.e. ~5e-3 in f at the
    # steepest point; the median is far tighter
    assert np.max(np.abs(w - _f(rho, r_s))) < 1e-2
    assert np.median(np.abs(w - _f(rho, r_s))) < 1e-3
    np.testing.assert_allclose(t, 20.0, atol=2e-3)


def test_far_misses_stop_on_the_envelope(built):
    """A vertical ray between strings touches no DOM and stops on the envelope's top cap, at
    t = z_max - z0; a ray from far outside the array gets no weight at all."""
    det, prop = built
    anchors = np.asarray(det.string_anchors)[:, :2]
    # the grid point inside the array farthest from every string
    g = np.stack(np.meshgrid(np.linspace(-300, 300, 61), np.linspace(-300, 300, 61)), -1).reshape(-1, 2)
    far = np.min(np.linalg.norm(g[:, None] - anchors[None], axis=-1), axis=1)
    xy = g[np.argmax(far)]
    assert far.max() > 10.0
    z0 = 0.5 * (det.envelope_z_min + det.envelope_z_max)
    out = prop(jnp.array([[xy[0], xy[1], z0], [5000.0, 5000.0, -1950.0]], jnp.float32),
               jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))
    assert float(jnp.max(out['sensor_weights'])) < 1e-6
    np.testing.assert_allclose(float(out['envelope_exit_t'][0]), det.envelope_z_max - z0,
                               rtol=1e-6)
    np.testing.assert_allclose(np.asarray(out['positions'][0]),
                               [xy[0], xy[1], det.envelope_z_max], atol=1e-3)
