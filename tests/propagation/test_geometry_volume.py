"""`lucid.geometry.volume`: one membership test per shape, shared by the kernel and the callers.

Each detector's ``volume`` must agree with the kernel's ``bounds_check`` (which now delegates to
it) on numpy and JAX input alike, and a scaled region must be the caller's shrink about the
volume's own centre — the paper's placement cut is ``detector.volume.scaled(0.95)``.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from lucid.geometry.volume import BoxVolume, CylinderVolume, SphereVolume

VOLUMES = [CylinderVolume(16.9, -18.1, 18.1), SphereVolume(17.7), BoxVolume(3.0, 6.0, 2.5),
           CylinderVolume(560.0, -2503.0, -1400.0)]          # last: an off-centre string envelope


def _points(vol, n=20000, seed=0):
    lo, hi = vol.bounding_box()
    c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
    return c + np.random.default_rng(seed).uniform(-1.2, 1.2, (n, 3)) * h


def _reference(vol, p):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    if isinstance(vol, CylinderVolume):
        return (x * x + y * y <= vol.radius ** 2) & (z >= vol.z_min) & (z <= vol.z_max)
    if isinstance(vol, SphereVolume):
        return np.sqrt(x * x + y * y + z * z) <= vol.radius
    return (np.abs(x) <= vol.half_x) & (np.abs(y) <= vol.half_y) & (np.abs(z) <= vol.half_z)


@pytest.mark.parametrize('vol', VOLUMES, ids=lambda v: type(v).__name__)
def test_contains_is_the_shape(vol):
    p = _points(vol)
    inside = vol.contains(p)
    np.testing.assert_array_equal(inside, _reference(vol, p))
    assert 0.2 < inside.mean() < 0.9                      # the sample straddles the wall


@pytest.mark.parametrize('vol', VOLUMES, ids=lambda v: type(v).__name__)
def test_numpy_and_jax_agree(vol):
    p = _points(vol).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(vol.contains(jnp.asarray(p))), vol.contains(p))


@pytest.mark.parametrize('vol', VOLUMES, ids=lambda v: type(v).__name__)
def test_scaled_is_a_shrink_about_the_centre(vol):
    lo, hi = vol.bounding_box()
    c = 0.5 * (lo + hi)
    p = _points(vol, seed=1)
    np.testing.assert_array_equal(vol.scaled(0.5).contains(p), vol.contains(c + (p - c) / 0.5))
    slo, shi = vol.scaled(0.5).bounding_box()
    np.testing.assert_allclose(0.5 * (slo + shi), c)


def _small_detectors():
    """One small detector per shape, built directly (no config file, a few hundred sensors)."""
    from lucid.geometry import Cylinder, Sphere, Box
    return [
        (Cylinder(2.0, 4.0, 300, 0.04), CylinderVolume(2.0, -2.0, 2.0)),
        (Sphere(1.8, 300, 0.05), SphereVolume(1.8)),
        (Box(6.0, 12.0, 5.0, 300, 0.075), BoxVolume(3.0, 6.0, 2.5)),
    ]


@pytest.mark.parametrize("i", range(3), ids=['cylinder', 'sphere', 'box'])
def test_detectors_use_their_volume(i):
    """Each detector's volume is the one its own dimensions define (half-lengths for the box),
    and its bounds_check is that volume's membership, written out independently."""
    det, want = _small_detectors()[i]
    assert det.volume == want
    p = _points(det.volume).astype(np.float32)
    got = np.asarray(det.bounds_check(jnp.asarray(p)))
    np.testing.assert_array_equal(got, _reference(want, p.astype(np.float64)))
    assert 0.2 < got.mean() < 0.8             # the sample straddles the surface


@pytest.mark.parametrize('vol', VOLUMES, ids=lambda v: type(v).__name__)
def test_clip_is_the_nearest_point_of_the_volume(vol):
    """Inside, unchanged; outside, a point of the volume (to rounding) that no point drawn
    uniformly inside the volume beats for distance."""
    p = _points(vol, n=4000, seed=2)
    inside = vol.contains(p)
    got = vol.clip(p)
    np.testing.assert_array_equal(got[inside], p[inside])
    out, near = p[~inside][:200], got[~inside][:200]
    assert vol.scaled(1 + 1e-12).contains(near).all()
    rng = np.random.default_rng(3)
    q = np.array([vol.from_unit(u) for u in rng.uniform(size=(10000, 3))])
    brute = np.min(np.linalg.norm(out[:, None] - q[None], axis=-1), axis=1)
    assert (np.linalg.norm(out - near, axis=-1) <= brute + 1e-9).all()
    np.testing.assert_allclose(vol.clip(near), near, rtol=1e-12, atol=0)
