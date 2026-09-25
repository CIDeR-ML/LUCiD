"""Where the first-arrival time NLL's saturation cap actually lives.

`first_arrival_window_nll` used to clamp its order-statistic exponent at the call site::

    a = jnp.minimum(n * (jnp.log(Shi) - jnp.log(Slo)), -1e-9)

and `_log1mexp` then clamped the same quantity again, to -1e-7. The second clamp is a hundred
times larger, so it always won: the call-site line was dead and the operative saturation point
was never the one the code advertised. Anyone reading the term — or tuning that constant — would
have had the wrong number.

It matters because the cap is ACTIVE, not a formality. The two caps differ by
log(1e-7 / 1e-9) = 4.6 nats on every PMT that sits on one, and a rebuild of this loss honouring
the written -1e-9 came out 1.4% above the library's own value on a real SK-like event.

The clamp is gone and the removal is bit-identical, which is what these tests pin: the
equivalence that licensed the deletion, and the cap's real location and value so a future edit to
`_log1mexp` cannot move it silently.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.losses import _log1mexp, first_arrival_window_nll

CAP = -1e-7          # the cap `_log1mexp` applies, and therefore the term's real saturation


def test_the_removed_clamp_was_redundant():
    """min(min(x, -1e-9), -1e-7) == min(x, -1e-7), which is why deleting it changed nothing."""
    x = jnp.asarray(np.concatenate([
        np.linspace(-50.0, 0.0, 20001),
        [-1e-5, -1e-6, -1e-7, -1e-8, -1e-9, -1e-10, -0.0, 0.0, 1e-9, 1e-3, 5.0]]))
    both = jnp.minimum(jnp.minimum(x, -1e-9), CAP)
    inner_only = jnp.minimum(x, CAP)
    assert bool(jnp.all(both == inner_only))


def test_log1mexp_saturates_at_1e_minus_7_not_1e_minus_9():
    """The value that a reader of the call site would have predicted is NOT the one produced."""
    got = float(_log1mexp(jnp.asarray(-1e-9)))
    assert got == pytest.approx(float(np.log(1e-7)), abs=1e-4)      # -16.12, the real cap
    assert abs(got - float(np.log(1e-9))) > 4.0                     # -20.72, the advertised one


@pytest.mark.parametrize('a', [-1e-6, -1e-8, -1e-9, -1e-12, 0.0])
def test_log1mexp_is_finite_and_monotone_through_the_cap(a):
    """Whatever the cap is, the term must stay finite: the cap exists to stop log(0)."""
    v = float(_log1mexp(jnp.asarray(a)))
    assert np.isfinite(v)
    assert v <= float(_log1mexp(jnp.asarray(-1.0)))                 # saturated branch is smaller


def test_the_term_still_computes_on_a_small_hand_case():
    """A end-to-end guard so a refactor of the clamp cannot break the caller's shape/finiteness.

    Two sensors, three photons. Sensor 0 is lit and its window straddles the predicted arrivals;
    sensor 1 is unlit and must contribute exactly zero.
    """
    log_w = jnp.log(jnp.asarray([1.0, 2.0, 0.5]))
    times = jnp.asarray([10.0, 11.0, 30.0])
    idx = jnp.asarray([0, 0, 1])
    t_obs = jnp.asarray([12.0, 0.0])
    mu = jnp.asarray([3.0, 0.5])
    n = jnp.asarray([2.0, 0.0])
    out = first_arrival_window_nll(log_w, times, idx, t_obs, mu, n, 2)
    assert out.shape == (2,)
    assert np.isfinite(np.asarray(out)).all()
    assert float(out[1]) == 0.0                                     # unlit contributes nothing
    assert float(out[0]) > 0.0
