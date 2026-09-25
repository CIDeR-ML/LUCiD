"""The sensor deposit must have a finite gradient when a ray passes through a sphere centre.

`compute_sensor_intersections_base` feeds `|to_sensor|` into the overlap kernel. `jnp.linalg.norm(v)`
differentiates to `v / |v|`, which is 0/0 at `v == 0`, and the later `jnp.where(valid, ...)` mask
selects the value but still passes the NaN cotangent: the loss stays finite while the geometry
gradient is `nan`. The epsilon must sit INSIDE the sqrt, not after the norm -- the same convention
`simulator.py` documents for the surface-distance norm.

`to_sensor == 0` is reached by a ray aimed at a real sensor's centre, or by an invalid slot
(`sensor_idx == -1`, whose sphere centre is set to the origin) on a ray through the origin. One such
draw is enough to break a fit, because `grad_metric_loss` averages the gradient over `nkeys` draws.

The tests build the degenerate geometry directly; the last one checks that the unsafe norm does
produce NaN, so these tests can detect the bug.
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
    """The failure mode is finite forward, non-finite backward; pin the forward half."""
    assert np.isfinite(float(_deposit(origin, direction, idx, pos)))


def test_a_generic_ray_still_differentiates():
    """The guard must not zero the gradient for an ordinary, non-degenerate ray."""
    origin = jnp.array([[0.1, -0.2, 0.0]])
    direction = jnp.array([[0.02, 0.03, 1.0]])
    pos = jnp.array([[0.0, 0.0, 5.0]])
    g_o = jax.grad(_deposit, argnums=0)(origin, direction, jnp.array([0]), pos)
    g_o = np.asarray(g_o)
    assert np.isfinite(g_o).all()
    assert np.abs(g_o).max() > 0.0, 'deposit lost its dependence on the ray origin'


def test_the_unsafe_norm_really_does_produce_nan():
    """Control: the unguarded norm must give a NaN gradient at zero, or the tests above prove nothing."""
    def unsafe(v):
        return jnp.sum(jnp.linalg.norm(v, axis=1))          # eps outside the sqrt, or absent

    def safe(v):
        return jnp.sum(jnp.sqrt(jnp.sum(v ** 2, axis=1) + 1e-12))

    zero = jnp.zeros((1, 3))
    assert not np.isfinite(np.asarray(jax.grad(unsafe)(zero))).all()
    assert np.isfinite(np.asarray(jax.grad(safe)(zero))).all()
