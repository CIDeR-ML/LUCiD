"""The sensor deposit must have a FINITE GRADIENT when a ray passes through a sphere centre.

Why this exists
---------------
A gradient census over 960 PRNG keys found one that produced `nan` in every geometry component of
the reconstruction gradient -- X, Y, Z, phi, theta -- while t0, E and the LOSS ITSELF stayed
finite. Finite forward, non-finite backward is the signature of a value that is masked out by a
`jnp.where` after it has already gone bad: `where` selects the value, it does not stop the
cotangent, so the branch that was not taken is still differentiated.

The source is the norm feeding the deposit weight:

    distance = jnp.linalg.norm(to_sensor, axis=1)        # -> overlap_prob(distance) -> deposit

`jnp.linalg.norm(v)` differentiates to `v / |v|`, which is 0/0 at `v == 0`. Adding an epsilon
AFTERWARDS -- as the neighbouring lines do -- protects the division that follows and never the
norm's own derivative. `simulator.py` already documents this exact trap for the surface distance
("SAFE norm: eps INSIDE the sqrt"); these call sites were missed.

`to_sensor == 0` is reachable two ways, and the second is the one that bites:

  * a ray aimed exactly at a real sensor's centre;
  * an INVALID candidate slot. `compute_sensor_intersections_base` sets `sphere_centers` to the
    ORIGIN for `sensor_idx == -1`, so `to_sensor` becomes the ray's closest approach to (0,0,0) --
    and a ray through the detector centre is an ordinary thing for a track to emit. The weight is
    then discarded by `jnp.where(valid, ...)`, which is exactly the mask that hides the value and
    passes the NaN.

It was rare -- 1 key in 960 at 0.84 rad off truth in azimuth, 0 in 960 at truth -- and fatal
where it landed, because `ReconProblem.grad_metric_loss` averages the gradient over `nkeys` draws
and steps on the mean, so a single poisoned draw makes the whole Gauss-Newton step `nan`.

These tests construct the degenerate geometry directly instead of hunting for the seed again, and
the last one asserts that the OLD form really did fail, so the test is known to be able to catch
the bug rather than merely passing.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.propagation.base import compute_sensor_intersections_base

R = 0.25                                     # sensor radius, metres


def _bounds(points):
    return jnp.ones(points.shape[0], dtype=bool)


def _overlap(d):
    """A smooth stand-in for the real overlap kernel; only its d-dependence matters here."""
    return jnp.exp(-(d / R) ** 2)


def _deposit(origins, directions, sensor_idx, sensor_positions):
    w = compute_sensor_intersections_base(
        sensor_idx, sensor_positions, R, origins, directions, _bounds, _overlap)[0]
    return jnp.sum(w)


# (label, sensor_idx, sensor_positions, origin, direction) — each aims a ray so that its closest
# approach lands exactly on the sphere centre the code will use.
CASES = [
    # a real sensor at (0,0,5), ray fired straight at its centre from the origin
    ('valid sensor, ray through its centre',
     jnp.array([0]), jnp.array([[0.0, 0.0, 5.0]]),
     jnp.array([[0.0, 0.0, 0.0]]), jnp.array([[0.0, 0.0, 1.0]])),
    # an INVALID slot: `sphere_centers` becomes the ORIGIN, and this ray passes through it
    ('invalid slot, ray through the origin',
     jnp.array([-1]), jnp.array([[0.0, 0.0, 5.0]]),
     jnp.array([[0.0, 0.0, -5.0]]), jnp.array([[0.0, 0.0, 1.0]])),
]


@pytest.mark.parametrize('label,idx,pos,origin,direction', CASES)
def test_deposit_gradient_is_finite_at_zero_distance(label, idx, pos, origin, direction):
    g_o, g_d = jax.grad(_deposit, argnums=(0, 1))(origin, direction, idx, pos)
    assert np.isfinite(np.asarray(g_o)).all(), f'{label}: non-finite d/d(origin) {g_o}'
    assert np.isfinite(np.asarray(g_d)).all(), f'{label}: non-finite d/d(direction) {g_d}'


@pytest.mark.parametrize('label,idx,pos,origin,direction', CASES)
def test_the_forward_value_was_never_the_problem(label, idx, pos, origin, direction):
    """Finite forward, non-finite backward was the whole difficulty — pin the forward half."""
    assert np.isfinite(float(_deposit(origin, direction, idx, pos)))


def test_a_generic_ray_still_differentiates():
    """The guard must not have flattened the gradient everywhere it used to work."""
    origin = jnp.array([[0.1, -0.2, 0.0]])
    direction = jnp.array([[0.02, 0.03, 1.0]])
    pos = jnp.array([[0.0, 0.0, 5.0]])
    g_o = jax.grad(_deposit, argnums=0)(origin, direction, jnp.array([0]), pos)
    g_o = np.asarray(g_o)
    assert np.isfinite(g_o).all()
    assert np.abs(g_o).max() > 0.0, 'deposit lost its dependence on the ray origin'


def test_the_unsafe_norm_really_does_produce_nan():
    """Control: the form that shipped must FAIL, or these tests prove nothing.

    Without this, the tests above only show that the current code is finite — they would pass
    just as happily against an implementation that never had the bug, and could not tell anyone
    what they are for.
    """
    def unsafe(v):
        return jnp.sum(jnp.linalg.norm(v, axis=1))          # eps outside, or absent, as shipped

    def safe(v):
        return jnp.sum(jnp.sqrt(jnp.sum(v ** 2, axis=1) + 1e-12))

    zero = jnp.zeros((1, 3))
    assert not np.isfinite(np.asarray(jax.grad(unsafe)(zero))).all()
    assert np.isfinite(np.asarray(jax.grad(safe)(zero))).all()
