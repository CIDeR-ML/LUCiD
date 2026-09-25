"""The first-arrival time NLL's saturation cap lives in `_log1mexp`, at -1e-7.

`first_arrival_window_nll` passes its order-statistic exponent to `_log1mexp` unclamped, so that
cap is the term's only saturation point. A caller-side clamp closer to zero than -1e-7 (e.g.
-1e-9) would be dead code that advertises the wrong number, since `_log1mexp`'s own cap always
dominates. The cap is active, not a formality: -1e-7 and -1e-9 differ by log(100) = 4.6 nats on
every PMT that sits on it.

These tests pin that equivalence and the cap's location and value, so an edit to `_log1mexp`
cannot move it silently.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.losses import _log1mexp, first_arrival_window_nll

CAP = -1e-7          # the cap `_log1mexp` applies, and therefore the term's real saturation


def test_the_removed_clamp_was_redundant():
    """min(min(x, -1e-9), -1e-7) == min(x, -1e-7): a caller-side -1e-9 clamp is a no-op."""
    x = jnp.asarray(np.concatenate([
        np.linspace(-50.0, 0.0, 20001),
        [-1e-5, -1e-6, -1e-7, -1e-8, -1e-9, -1e-10, -0.0, 0.0, 1e-9, 1e-3, 5.0]]))
    both = jnp.minimum(jnp.minimum(x, -1e-9), CAP)
    inner_only = jnp.minimum(x, CAP)
    assert bool(jnp.all(both == inner_only))


def test_log1mexp_saturates_at_1e_minus_7_not_1e_minus_9():
    """`_log1mexp` saturates at log(1e-7), not at the log(1e-9) a -1e-9 clamp would suggest."""
    got = float(_log1mexp(jnp.asarray(-1e-9)))
    assert got == pytest.approx(float(np.log(1e-7)), abs=1e-4)      # -16.12, the real cap
    assert abs(got - float(np.log(1e-9))) > 4.0                     # -20.72, what a -1e-9 cap would give


@pytest.mark.parametrize('a', [-1e-6, -1e-8, -1e-9, -1e-12, 0.0])
def test_log1mexp_is_finite_and_monotone_through_the_cap(a):
    """Whatever the cap is, the term must stay finite: the cap exists to stop log(0)."""
    v = float(_log1mexp(jnp.asarray(a)))
    assert np.isfinite(v)
    assert v <= float(_log1mexp(jnp.asarray(-1.0)))                 # saturated branch is smaller


def test_the_term_still_computes_on_a_small_hand_case():
    """End-to-end guard: a refactor of the cap must not break the caller's shape or finiteness.

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
