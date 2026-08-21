"""Every contact point the propagator reports must lie inside the detector.

A photon stops at the first surface it meets, so the point where it lands is either on the
wall or on the part of a photosensor inside the wall — never beyond. This used to fail: the
ray-sphere solve treats the ray as an infinite line and was not clipped against the wall, so
a photon could leave the detector and then enter the part of a sensor sphere sitting outside
it (every sensor sphere straddles the wall; an offset photocathode sits mostly outside). It
cost 5.7% of deposited charge on an SK-like cylinder and 20% with SK's offset sphere.

The invariant is enforced in ``compute_sensor_intersections_base`` by
``reachable = intersects & (t_intersect <= wall_t)`` plus a fallback contact point at the
wall, so these tests pin the property rather than the implementation.
"""

import os

import numpy as np
import jax.numpy as jnp
import pytest

from lucid.geometry import generate_detector
from lucid.propagation.shared import create_propagator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# float32 at detector scale: ~2e-6 m at 17 m, plus a few ulps through the panel maths.
# Contact points legitimately sit ON the boundary, so the test is one-sided.
TOL = 1e-4

GEOMS = ['SK_like_geom_config.json', 'SK_geom_config.json',
         'JUNO_geom_config.json', 'MidBox_geom_config.json']


def _detector(cfg):
    return generate_detector(os.path.join(REPO, 'config', cfg))


def _hostile_rays(det, n, seed=17):
    """Rays chosen to stress the wall clip: isotropic, launched off sensors, and grazing."""
    rng = np.random.default_rng(seed)
    P = np.asarray(det.all_points)
    o1 = np.tile(np.array([0.2, -0.3, 0.5]), (n // 3, 1))
    d1 = rng.normal(size=(n // 3, 3))
    j = rng.integers(0, len(P), n // 3)
    o2, d2 = P[j] * 0.985, rng.normal(size=(n // 3, 3))
    j = rng.integers(0, len(P), n - 2 * (n // 3))
    o3 = P[j] * 0.99
    d3 = rng.normal(size=(len(j), 3))
    d3 -= (np.einsum('ij,ij->i', d3, P[j])
           / np.einsum('ij,ij->i', P[j], P[j]))[:, None] * P[j]      # tangential
    o = np.concatenate([o1, o2, o3])
    d = np.concatenate([d1, d2, d3])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    keep = np.asarray(det.bounds_check(jnp.asarray(o)))
    return o[keep], d[keep]


def _worst_outside(det, result):
    """Most-negative signed distance over every contact point in a propagator result."""
    worst = 0.0
    n_out = 0
    for key in ('per_sensor_positions', 'positions'):
        q = np.asarray(result[key]).reshape(-1, 3)
        sdf = np.asarray(det.boundary_signed_distance(jnp.asarray(q)))
        n_out += int((sdf < -TOL).sum())
        worst = min(worst, float(sdf.min()))
    return n_out, worst


@pytest.mark.parametrize('cfg', GEOMS)
@pytest.mark.parametrize('temperature', [None, 0.1], ids=['hard', 'soft'])
def test_contact_points_are_inside(cfg, temperature):
    det = _detector(cfg)
    o, d = _hostile_rays(det, 30_000)
    prop = create_propagator(det, jnp.asarray(det.all_points), float(det.S_radius),
                             temperature=temperature)
    n_out, worst = _worst_outside(det, prop(jnp.asarray(o), jnp.asarray(d)))
    assert n_out == 0, f"{n_out} contact points outside {cfg}, worst {worst:.3e} m"


@pytest.mark.parametrize('cfg', ['SK_like_geom_config.json', 'SK_geom_config.json'])
@pytest.mark.parametrize('hit_mode,temperature',
                         [('realistic', None), ('per_photon', 0.1)],
                         ids=['data', 'prediction'])
def test_contact_points_inside_through_the_real_simulator(cfg, hit_mode, temperature):
    """Run the actual simulator and inspect every contact point at every scan step.

    The propagator DetectorGeometry hands over is wrapped, then ``setup_event_simulator``
    runs unmodified: real Cherenkov rays, real wavelength sampling, real reflection draws,
    all K steps. Reconstructing the scan body here instead would test the reconstruction —
    an earlier version of this test did exactly that, hand-rolled the reflection, and
    reported a failure that existed only in the test.

    ``jax.debug.callback`` fires at runtime, so the check reaches inside the jitted scan.
    """
    import jax
    from lucid.geometry.detector_geometry import DetectorGeometry
    from lucid.simulation import setup_event_simulator
    from lucid.fitting import track_from_vec9

    K, n_rays = 4, 20_000
    tally = [0, 0, 0.0]        # calls, violations, worst sdf

    def instrument(det, prop):
        def wrapped(o, d):
            res = prop(o, d)

            def _chk(psens, pos):
                tally[0] += 1
                for q in (np.asarray(psens).reshape(-1, 3), np.asarray(pos).reshape(-1, 3)):
                    sdf = np.asarray(det.boundary_signed_distance(jnp.asarray(q)))
                    tally[1] += int((sdf < -TOL).sum())
                    tally[2] = min(tally[2], float(sdf.min()))
            jax.debug.callback(_chk, res['per_sensor_positions'], res['positions'])
            return res
        return wrapped

    original = DetectorGeometry.from_config
    try:
        def patched(json_filename, *a, **k):
            dg = original(json_filename, *a, **k)
            return dg._replace(propagator=instrument(dg.detector, dg.propagator))
        DetectorGeometry.from_config = staticmethod(patched)

        th, ph = np.pi / 3, np.pi / 4
        t9 = jnp.asarray([1000.0, 0.0, 0.0, 0.0, np.sin(th), np.cos(th),
                          np.sin(ph), np.cos(ph), 0.0], dtype=float)
        sim = setup_event_simulator(
            os.path.join(REPO, 'config', cfg), n_rays, temperature=temperature, K=K,
            hit_mode=hit_mode,
            physics_config=os.path.join(REPO, 'config', 'SK_like_physics_config.json'),
            default_detector_params=True, particle='muon', wavelength_mode=True,
            cherenkov_emission_band=(274.91, 673.83))
        jax.block_until_ready(sim(track_from_vec9(t9), jax.random.PRNGKey(0)))
    finally:
        DetectorGeometry.from_config = original

    assert tally[0] == K, f"instrumented {tally[0]} propagator calls, expected K={K}"
    assert tally[1] == 0, (f"{tally[1]} contact points outside {cfg} ({hit_mode}), "
                           f"worst {tally[2]:.3e} m")


@pytest.mark.parametrize('cfg', GEOMS)
def test_signed_distance_sign_matches_bounds_check(cfg):
    """boundary_signed_distance is what the charge gate differentiates, so its sign must
    agree with the boolean the forward pass uses."""
    det = _detector(cfg)
    rng = np.random.default_rng(3)
    # Sphere has no H, Box has no r — size the sampling box from whatever the shape exposes
    # and pad it so points land both inside and outside.
    lim = getattr(det, 'circumradius', None) or getattr(det, 'r', None) or max(
        getattr(det, 'L', 1.0), getattr(det, 'W', 1.0))
    half_z = getattr(det, 'H', 2.0 * lim) / 2.0
    q = rng.uniform(-1.3, 1.3, (100_000, 3)) * np.array([lim, lim, half_z])
    hard = np.asarray(det.bounds_check(jnp.asarray(q)))
    sdf = np.asarray(det.boundary_signed_distance(jnp.asarray(q)))
    mismatch = hard != (sdf >= 0)
    # only float32 ties at |sdf| ~ 0 may disagree
    assert not mismatch.any() or np.abs(sdf[mismatch]).max() < TOL, (
        f"{int(mismatch.sum())} sign mismatches, worst |sdf| "
        f"{np.abs(sdf[mismatch]).max():.3e} m")
