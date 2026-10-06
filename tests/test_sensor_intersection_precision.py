"""Sensor hit points must be accurate to well within the reflection nudge at detector distances.

`compute_sensor_intersections_base` places a photon on a PMT sphere. In float32 the textbook
discriminant `b**2 - 4*a*c` cancels catastrophically when the photon starts metres away: both terms
are ~|oc|^2 while their difference is ~4 r^2. The hit point can then land up to mm inside the
sphere, deeper than the 1e-4 nudge `photon_step` applies along the inward normal, so a photon that
reflects off a PMT starts its next leg inside it and meets the same PMT again.

The tests aim float32 rays at SK-sized PMTs from points inside an SK-sized tank, compare the hit
distance with a float64 reference computed from the same float32 inputs, and replay the reflection
the way `photon_step` does. The last test checks that the textbook discriminant does fail on this
geometry, so the first two can detect the bug.
"""
import jax.numpy as jnp
import numpy as np

from lucid.propagation.base import compute_sensor_intersections_base
from lucid.simulation.optics import compute_reflection_direction

R_SENSOR = 0.254                    # SK 20" PMT radius [m] (config/sk_geometry.npz)
R_TANK, H_TANK = 16.96, 36.38       # SK inner detector [m]
NUDGE = 1e-4                        # photon_step's epsilon along the inward normal [m]
N_RAYS = 20_000


def _bounds(points):
    return jnp.ones(points.shape[0], dtype=bool)


def _overlap(d):
    return jnp.exp(-(d / R_SENSOR) ** 2)


def _rays(seed=0):
    """Rays from random points in the tank, each aimed at its own wall PMT with impact parameter
    up to 0.99 R (the grazing edge is excluded: there the hit point is ill-conditioned in any
    precision). Inputs are float32; geometry quantities derived from them are float64."""
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, N_RAYS)
    centers = np.column_stack([R_TANK * np.cos(phi), R_TANK * np.sin(phi),
                               rng.uniform(-H_TANK / 2, H_TANK / 2, N_RAYS)])
    r_o = R_TANK * np.sqrt(rng.uniform(0, 1, N_RAYS))
    phi_o = rng.uniform(0, 2 * np.pi, N_RAYS)
    origins = np.column_stack([r_o * np.cos(phi_o), r_o * np.sin(phi_o),
                               rng.uniform(-H_TANK / 2, H_TANK / 2, N_RAYS)])
    axis = centers - origins
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    side = np.cross(axis, rng.normal(size=(N_RAYS, 3)))
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    rho = 0.99 * R_SENSOR * np.sqrt(rng.uniform(0, 1, N_RAYS))
    aim = centers + rho[:, None] * side
    directions = aim - origins
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    o32, d32, c32 = (x.astype(np.float32) for x in (origins, directions, centers))
    o, d, c = (x.astype(np.float64) for x in (o32, d32, c32))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    oc = o - c
    t_c = -np.sum(oc * d, axis=1)
    perp2 = np.sum((oc + t_c[:, None] * d) ** 2, axis=1)
    t_ref = t_c - np.sqrt(R_SENSOR ** 2 - perp2)          # entry distance, float64
    return o32, d32, c32, t_ref


def _intersect(origins, directions, centers):
    idx = jnp.arange(origins.shape[0])
    _, times, _, normals, hit, points = compute_sensor_intersections_base(
        idx, jnp.asarray(centers), R_SENSOR, jnp.asarray(origins), jnp.asarray(directions),
        _bounds, _overlap)
    return np.asarray(times)[:, 0], np.asarray(normals), np.asarray(hit), np.asarray(points)


def test_hit_distance_matches_float64_within_the_nudge():
    o, d, c, t_ref = _rays()
    t, _, hit, _ = _intersect(o, d, c)
    assert hit.all(), f'{np.sum(~hit)} aimed rays missed their sensor'
    err = np.abs(t.astype(np.float64) - t_ref)
    assert err.max() < NUDGE, (
        f'hit distance off by up to {err.max() * 1e3:.3f} mm (p99 {np.percentile(err, 99) * 1e3:.3f} mm); '
        f'{np.mean(err > NUDGE) * 100:.1f}% of hits exceed the {NUDGE * 1e3:.1f} mm reflection nudge')


def test_reflected_photon_does_not_meet_its_sensor_again():
    o, d, c, _ = _rays(seed=1)
    t, normals, hit, _ = _intersect(o, d, c)
    assert hit.all()
    # As photon_step: surface point plus the nudge along the inward normal, specular direction.
    pos = jnp.asarray(o) + jnp.asarray(t)[:, None] * jnp.asarray(d) - NUDGE * jnp.asarray(normals)
    new_dir = compute_reflection_direction(jnp.asarray(d), jnp.asarray(normals))
    _, _, again, _ = _intersect(np.asarray(pos), np.asarray(new_dir), c)
    assert not again.any(), f'{np.sum(again)} of {again.size} reflected photons re-entered their sensor'


def test_textbook_discriminant_fails_here():
    """The geometry is hard enough: `b**2 - 4*a*c` in float32 misses the nudge on these rays."""
    o, d, c, t_ref = _rays()
    oc = o - c
    a = np.sum(d * d, axis=1)
    b = np.float32(2) * np.sum(oc * d, axis=1)
    cc = np.sum(oc * oc, axis=1) - np.float32(R_SENSOR) ** 2
    disc = b * b - np.float32(4) * a * cc
    t_old = (-b - np.sqrt(np.maximum(disc, 0))) / (np.float32(2) * a)
    assert np.abs(t_old.astype(np.float64) - t_ref).max() > NUDGE
