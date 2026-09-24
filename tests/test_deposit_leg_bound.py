"""The optional leg bound on the surface deposit (`deposit_leg_bound`).

Without it a candidate is weighted by its distance from the photon's ray LINE, which does not stop
at the wall: at grazing incidence the line passes within a sensor radius of sensors further along
the wall than the photon ever reached. With it, the distance is measured to the leg the photon
actually travels. It is off by default and must then change nothing at all.
"""
import os

import jax.numpy as jnp
import numpy as np
import pytest

from lucid.geometry.detector_geometry import DetectorGeometry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WCTE = os.path.join(ROOT, 'config/WCTE_like_geom_config.json')


def _grazing(dg, n=20000, seed=11):
    """Photons just inside the barrel, moving almost along the wall -- where the line and the leg
    disagree most."""
    det = dg.detector
    R, H = float(det.r), float(det.H)
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, n); z = rng.uniform(-H / 4, H / 4, n)
    o = np.stack([(R - 0.05) * np.cos(phi), (R - 0.05) * np.sin(phi), z], -1)
    d = np.stack([-np.sin(phi), np.cos(phi), np.zeros(n)], -1) + 0.05 * rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return o.astype(np.float32), d.astype(np.float32)


def _run(dg, o, d):
    out = dg.propagator(jnp.asarray(o), jnp.asarray(d))
    return {k: np.asarray(v) for k, v in out.items()}


@pytest.mark.slow
def test_off_is_the_default_and_changes_nothing():
    """Passing False explicitly and not passing it at all must give the same arrays, every one."""
    a = DetectorGeometry.from_config(WCTE, temperature=0.2)
    b = DetectorGeometry.from_config(WCTE, temperature=0.2, deposit_leg_bound=False)
    o, d = _grazing(a, n=4000)
    ra, rb = _run(a, o, d), _run(b, o, d)
    assert ra.keys() == rb.keys()
    for k in ra:
        assert np.array_equal(ra[k], rb[k]), k


@pytest.mark.slow
def test_charge_only_where_the_travelled_leg_reaches_the_sensor():
    """Step mode, where the forward overlap is exactly 1 inside a sensor radius and 0 outside: with
    the bound on, a candidate carries charge only if the SEGMENT from the photon to where it lands
    passes within a sensor radius of it. Without the bound the same photons put charge on sensors
    only the infinite line comes near."""
    dg_on = DetectorGeometry.from_config(WCTE, temperature=None, deposit_leg_bound=True)
    dg_off = DetectorGeometry.from_config(WCTE, temperature=None)
    S = np.asarray(dg_on.sensor_points); rS = float(dg_on.sensor_radius)
    o, d = _grazing(dg_on)

    # The leg the code bounds to ends at the WALL -- t_geometry, from the same intersect_ray the
    # propagator calls -- not at a sensor the photon may enter first. (A first draft used the hit
    # position, which is the sensor entry for photons that hit one, and so flagged sensors that
    # legitimately lie on the travelled leg.)
    t_land = np.asarray(dg_on.detector.intersect_ray(jnp.asarray(o), jnp.asarray(d))[1]).reshape(-1)

    def reach(out):
        """(charged candidates the leg does NOT reach, uncharged candidates it DOES reach)."""
        w = out['sensor_weights']; idx = np.clip(out['sensor_indices'], 0, len(S) - 1)
        c = S[idx]                                                      # (C, N, 3)
        t_c = np.sum((c - o[None]) * d[None], -1)                       # closest approach on line
        perp = np.linalg.norm(c - (o[None] + t_c[..., None] * d[None]), axis=-1)
        seg = np.sqrt(perp ** 2 + np.maximum(t_c - t_land[None], 0.0) ** 2)
        beyond = int(np.sum((w > 0) & (seg > rS * (1 + 1e-3))))
        within = (seg < rS * (1 - 1e-3)) & (t_c > 0)                    # ahead, and on the leg
        return beyond, int(np.sum(within)), int(np.sum(within & (w <= 0)))

    off_beyond, _, _ = reach(_run(dg_off, o, d))
    assert off_beyond > 100, 'the population does not exercise the line/leg difference'
    on_beyond, on_within, on_starved = reach(_run(dg_on, o, d))
    assert on_beyond == 0
    # and the converse, so a bound that simply removed charge could not pass: every candidate the
    # travelled leg does reach still carries some
    assert on_within > 100 and on_starved == 0, (on_within, on_starved)
