"""The first-arrival time NLL's saturation cap lives in `_log1mexp`, at -1e-7.

`first_arrival_window_nll` passes its order-statistic exponent to `_log1mexp` unclamped, so that
cap is the term's only saturation point. A caller-side clamp closer to zero than -1e-7 (e.g.
-1e-9) would be dead code that advertises the wrong number, since `_log1mexp`'s own cap always
dominates. The cap is active, not a formality: -1e-7 and -1e-9 differ by log(100) = 4.6 nats on
every PMT that sits on it.

These tests pin the cap's location and value, so an edit to `_log1mexp` cannot move it silently,
and check the term itself against its documented order-statistic closed form.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.losses import _log1mexp, first_arrival_window_nll

CAP = -1e-7          # the cap `_log1mexp` applies, and therefore the term's real saturation


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


def test_the_term_matches_its_closed_form_on_a_small_hand_case():
    """Two sensors, three photons. Sensor 0 is lit: with Gaussian per-photon resolution sigma,
    R(t) = sum w_i Phi((t - t_i)/sigma) and S = (mu - R)/mu is the survival of one arrival,
    so the earliest of n arrivals falls in the window [t_obs -+ delta/2] with probability
    S(t_lo)^n - S(t_hi)^n, and the term is minus its log (evaluated here in float64 with
    scipy's normal CDF). Sensor 1 is unlit and contributes exactly zero.

    (The code computes -n log S_lo - log(1 - (S_hi/S_lo)^n), which is that. Its docstring,
    losses.py:285, writes the second term as log(S_lo^n - S_hi^n), which would count
    -n log S_lo twice: 4.47 instead of 2.49 here.)"""
    from scipy.special import ndtr
    log_w = jnp.log(jnp.asarray([1.0, 2.0, 0.5]))
    times = jnp.asarray([10.0, 11.0, 30.0])
    idx = jnp.asarray([0, 0, 1])
    t_obs = jnp.asarray([12.0, 0.0])
    mu = jnp.asarray([3.0, 0.5])
    n = jnp.asarray([2.0, 0.0])
    sigma, delta = 2.5, 1.0
    out = np.asarray(first_arrival_window_nll(log_w, times, idx, t_obs, mu, n, 2,
                                              sigma=sigma, delta=delta))
    w, t = np.array([1.0, 2.0]), np.array([10.0, 11.0])

    def S(edge):
        return (3.0 - np.sum(w * ndtr((edge - t) / sigma))) / 3.0
    s_lo, s_hi = S(12.0 - delta / 2), S(12.0 + delta / 2)
    expected = -np.log(s_lo**2 - s_hi**2)
    np.testing.assert_allclose(out[0], expected, rtol=1e-5)
    assert out[1] == 0.0
