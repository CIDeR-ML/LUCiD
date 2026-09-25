"""Invariants of the surface deposit that no aggregate check can see.

Each propagator test guards one piece of the deposit: without the first-hit survival product,
grazing photons deposit more than they carry; without the ahead gate, a photon leaving a PMT
deposits on sensors behind it (the tube it left and its neighbours); without first-entry, a
grazing photon stops on no sensor's surface. They use the small WCTE_like tank to stay cheap, and
each checks its population is non-empty so a geometry change cannot make it pass vacuously.
"""
import os

import jax.numpy as jnp
import numpy as np
import pytest

from lucid.propagation.shared import first_hit_survival

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- first_hit_survival, directly -------------------------------------------------------------

def test_single_candidate_keeps_its_probability():
    w = jnp.array([[0.7]]); t = jnp.array([[1.0]])
    np.testing.assert_allclose(np.asarray(first_hit_survival(w, t)), [[0.7]], rtol=1e-6)


def test_earlier_candidate_keeps_full_weight_later_one_is_attenuated():
    w = jnp.array([[0.6], [0.5]]); t = jnp.array([[2.0], [1.0]])     # slot 1 arrives first
    out = np.asarray(first_hit_survival(w, t))[:, 0]
    np.testing.assert_allclose(out, [0.6 * (1 - 0.5), 0.5], rtol=1e-6)


def test_exact_time_tie_still_caps_at_one():
    """Two live candidates at IDENTICAL times. A strict `t_j < t_i` makes neither earlier, so both
    keep 0.9 and the photon deposits 1.8. The slot tie-break must bring it to 0.9 + 0.9*0.1."""
    w = jnp.array([[0.9], [0.9]]); t = jnp.array([[1.5], [1.5]])
    out = np.asarray(first_hit_survival(w, t))[:, 0]
    assert out.sum() <= 1.0 + 1e-6, out
    np.testing.assert_allclose(out.sum(), 0.9 + 0.9 * 0.1, rtol=1e-5)


def test_step_mode_gradient_survives_the_cap():
    """In step mode a candidate the ray passes inside has forward overlap EXACTLY 1, and its gradient
    comes from the straight-through surrogate. The cap clips p below 1 to keep log1p finite; a plain
    jnp.clip has zero derivative there and would zero that gradient. d(total)/dp must be 1."""
    import jax
    w = jnp.array([[1.0], [0.0]]); t = jnp.array([[1.0], [2.0]])
    g = np.asarray(jax.grad(lambda w_: first_hit_survival(w_, t).sum())(w))
    assert g[0, 0] > 0.5, g


def test_cap_holds_on_random_inputs_including_ties():
    rng = np.random.default_rng(0)
    C, N = 8, 20000
    w = rng.uniform(0, 1, (C, N)).astype(np.float32)
    t = rng.integers(0, 4, (C, N)).astype(np.float32)                # coarse times: ties everywhere
    tot = np.asarray(first_hit_survival(jnp.asarray(w), jnp.asarray(t))).sum(0)
    assert tot.max() <= 1.0 + 1e-5, tot.max()


# ---- through the real propagator ----------------------------------------------------------------

@pytest.fixture(scope='module')
def wcte():
    from lucid.geometry.detector_geometry import DetectorGeometry
    dg = DetectorGeometry.from_config(os.path.join(ROOT, 'config/WCTE_like_geom_config.json'),
                                      temperature=0.2)
    return dg, np.asarray(dg.sensor_points), float(dg.sensor_radius)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _run(dg, o, d):
    out = dg.propagator(jnp.asarray(o, jnp.float32), jnp.asarray(d, jnp.float32))
    w = np.asarray(out['sensor_weights']); idx = np.asarray(out['sensor_indices'])
    ins = np.asarray(out['inside_sensor'])
    return w, idx, ins, np.asarray(out['positions'])


@pytest.mark.slow
def test_grazing_photons_never_deposit_more_than_they_carry(wcte):
    dg, S, rS = wcte
    det = dg.detector; R, H = float(det.r), float(det.H)
    rng = np.random.default_rng(1); N = 20000
    phi = rng.uniform(0, 2 * np.pi, N); z = rng.uniform(-H / 4, H / 4, N)
    o = np.stack([(R - 0.05) * np.cos(phi), (R - 0.05) * np.sin(phi), z], -1)
    d = _unit(np.stack([-np.sin(phi), np.cos(phi), np.zeros(N)], -1) + 0.02 * rng.normal(size=(N, 3)))
    w, _, _, _ = _run(dg, o, d)
    assert int(np.sum((w > 0).sum(0) >= 2)) > 100, 'too few photons reach two sensors to test a cap'
    assert w.sum(0).max() <= 1.0 + 1e-5, w.sum(0).max()


@pytest.mark.slow
def test_photon_leaving_a_pmt_deposits_nothing_behind_it(wcte):
    """The population the ahead gate exists for: on a PMT sphere, nudged 1e-4 outward exactly as
    photon_step does after a sensor reflection, moving away."""
    dg, S, rS = wcte
    rng = np.random.default_rng(2); N = 20000
    k = rng.integers(0, len(S), N)
    n = _unit(rng.normal(size=(N, 3)))
    n = np.where((np.sum(n * (-S[k]), -1) < 0)[:, None], -n, n)       # the water-side half
    o = S[k] + (rS + 1e-4) * n
    d = _unit(rng.normal(size=(N, 3))); d = np.where((np.sum(d * n, -1) < 0)[:, None], -d, d)
    w, idx, _, _ = _run(dg, o, d)
    # float32, like the gate itself: in float64 a candidate at |t| ~ 1e-6 could flip sign
    S32, o32, d32 = S.astype(np.float32), o.astype(np.float32), d.astype(np.float32)
    t_closest = np.sum((S32[np.clip(idx, 0, len(S) - 1)] - o32[None]) * d32[None], -1)
    assert int(np.sum(t_closest <= 0)) > 100, 'too few candidates behind the photon to test the gate'
    assert int(np.sum((w > 0) & (t_closest <= 0))) == 0


@pytest.mark.slow
def test_a_photon_that_enters_a_sensor_stops_on_its_surface(wcte):
    """With several spheres entered, averaging their entry points puts the stop INSIDE a sphere."""
    from scipy.spatial import cKDTree
    dg, S, rS = wcte
    det = dg.detector; R, H = float(det.r), float(det.H)
    rng = np.random.default_rng(3); N = 20000
    phi = rng.uniform(0, 2 * np.pi, N); z = rng.uniform(-H / 4, H / 4, N)
    o = np.stack([(R - 0.05) * np.cos(phi), (R - 0.05) * np.sin(phi), z], -1)
    d = _unit(np.stack([-np.sin(phi), np.cos(phi), np.zeros(N)], -1) + 0.02 * rng.normal(size=(N, 3)))
    _, _, ins, pos = _run(dg, o, d)
    entered = ins.any(0)
    assert entered.sum() > 100, 'probe reached too few sensors to mean anything'
    dist, _ = cKDTree(S).query(pos[entered])
    assert int(np.sum(np.abs(dist - rS) > 1e-3)) == 0
