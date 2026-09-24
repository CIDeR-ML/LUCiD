"""lucid.fitting.calib — the (source × wavelength) assembly, and the estimator's two functions.

Scope: what the class ADDS on top of the simulator — the key arithmetic, the (S, NS) row
ordering, draw averaging, and the equivalence of the serial and mapped dispatch. The physics comes
from the simulator and is not what is tested here -- the library's agreement with the paper's
calibration engine was verified when it was extracted and is not continuously guarded -- so these tests
use a stub simulator and run in milliseconds.

The stub is deliberately sensitive to all three inputs the assembly must route correctly — source,
wavelength (through the DetectorParams the parameterisation builds) and key. A stub that ignored
one of them would let a transposed layout or a swapped key pass.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from lucid.fitting.params import CalibrationParams
from lucid.fitting.calib import CalibrationForward, profile_gains, neyman_residual

W, NS, NSRC = 3, 6, 4


class _Src:
    """Minimal source stand-in: a pytree with one distinguishing scalar."""

    def __init__(self, tag):
        self.tag = jnp.asarray(float(tag))


jax.tree_util.register_pytree_node(
    _Src, lambda s: ((s.tag,), None), lambda _, c: _Src.__new__(_Src).__class__.__new__(_Src) or None)


# A plain pytree is simpler and avoids registration subtleties: sources are dicts.
def _sources(n):
    return [{'tag': jnp.asarray(float(i))} for i in range(n)]


def _stub_sim(source, dp, key):
    """(charge, time). Depends on source, on the wavelength via dp, and on the key."""
    seed = jax.random.uniform(key, ()) * 1e-3
    per_sensor = jnp.arange(NS, dtype=jnp.float32)
    q = source['tag'] * 100.0 + dp.scattering.scatter_length + per_sensor + seed
    return q, jnp.zeros(NS)


def _params():
    return CalibrationParams(n_wavelengths=W, basis='rlogit')


def _theta(p):
    phys = [{'scatter_length': 10.0 * (i + 1), 'absorption_length': 300.0, 'qe': 0.2}
            for i in range(W)]
    return jnp.asarray(p.theta_from_physical(phys, 0.05, 0.55, 0.25, 0.9))


def _forward(map_fn=None):
    p = _params()
    return CalibrationForward(_stub_sim, _sources(NSRC), p, NS, map_fn=map_fn), p


def test_shape_and_row_ordering():
    """Row g must be (wavelength wl, source s) with g = wl*n_sources + s — wavelength-major.

    A transposed assembly would still return the right shape, so the ordering is asserted by
    identifying each row from the parts of the stub that encode source and wavelength.
    """
    fwd, p = _forward()
    out = np.asarray(fwd(_theta(p), 5000, jnp.ones(NS)))
    assert out.shape == (W * NSRC, NS)
    for wl in range(W):
        for s in range(NSRC):
            row = out[wl * NSRC + s]
            # stub: q = 100*s + scatter_length(wl) + arange(NS) + tiny key term
            assert abs((row[0] - row[0] % 1) - (100 * s + 10.0 * (wl + 1))) < 1.0, \
                f'row {wl*NSRC+s} is not (wl={wl}, s={s})'


def test_key_arithmetic_matches_the_reference_scheme():
    """Singleton: PRNGKey(kb + wl). Grouped source s: PRNGKey(kb + 1000*s + wl).

    Pinned because the whole reproducibility story rests on keys being pure functions of the base
    and the indices — and because a per-seed offset was once missing from exactly this kind of
    stream, leaving an ensemble blind to its own error.
    """
    fwd, _ = _forward()
    single, group = fwd.keys(5000)
    assert single.shape[0] == W and group.shape[:2] == (NSRC - 1, W)
    for wl in range(W):
        np.testing.assert_array_equal(np.asarray(single[wl]),
                                      np.asarray(jax.random.PRNGKey(5000 + wl)))
    for s in range(1, NSRC):
        for wl in range(W):
            np.testing.assert_array_equal(
                np.asarray(group[s - 1][wl]),
                np.asarray(jax.random.PRNGKey(5000 + 1000 * s + wl)))


def test_keys_change_with_the_base():
    """Guard on the guard: the key arithmetic must actually depend on key_base."""
    fwd, _ = _forward()
    a, _ = fwd.keys(5000)
    b, _ = fwd.keys(6000)
    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_average_is_the_mean_of_offset_draws():
    """average() must equal the mean over draws at key_base + 131*b — the reference's offset."""
    fwd, p = _forward()
    th, g = _theta(p), jnp.ones(NS)
    got = np.asarray(fwd.average(th, 5000, g, 3))
    want = np.mean([np.asarray(fwd(th, 5000 + 131 * b, g)) for b in range(3)], axis=0)
    np.testing.assert_allclose(got, want, rtol=0, atol=0)


def test_single_draw_average_is_the_draw():
    fwd, p = _forward()
    th, g = _theta(p), jnp.ones(NS)
    np.testing.assert_allclose(np.asarray(fwd.average(th, 5000, g, 1)),
                               np.asarray(fwd(th, 5000, g)), rtol=0, atol=0)


def test_serial_and_mapped_dispatch_agree():
    """The dispatch strategy is an execution choice and must not change the answer.

    The campaign shards seven isotropic sources across GPUs; off that node the same code takes a
    serial path. If those disagreed, a result would depend on how many devices happened to be
    visible.
    """
    ser, p = _forward(map_fn=None)
    vmapped, _ = _forward(map_fn=lambda fn, ax: jax.vmap(fn, in_axes=ax))
    th, g = _theta(p), jnp.ones(NS)
    np.testing.assert_allclose(np.asarray(ser(th, 5000, g)),
                               np.asarray(vmapped(th, 5000, g)), rtol=0, atol=0)


def test_single_source_needs_no_grouped_dispatch():
    """A one-source setup must work — the grouped branch is empty and must not be entered."""
    p = _params()
    fwd = CalibrationForward(_stub_sim, _sources(1), p, NS)
    out = np.asarray(fwd(_theta(p), 5000, jnp.ones(NS)))
    assert out.shape == (W, NS)


# --------------------------------------------------------------------------------------------
# The estimator's two closed forms. Both are one line of arithmetic, and both are load-bearing:
# the gauge is what makes the gains identifiable at all, and the Neyman weight is the whole reason
# the fixed point sits at truth under a re-drawn Monte-Carlo model. Until now neither had a direct
# test — they were gated only by the slow bit-exact engine pin, which cannot say WHICH property
# broke when it moves.
# --------------------------------------------------------------------------------------------

def _gain_case():
    """Model and observed sums whose ratio is deliberately NOT already gauged."""
    model = jnp.asarray([1.0, 2.0, 4.0, 8.0, 5.0, 2.5])
    ratio = jnp.asarray([0.7, 1.3, 2.0, 0.5, 1.1, 3.0])          # mean(log) != 0, mean != 1
    return model, model * ratio, ratio


def test_profile_gains_recovers_the_ratio_up_to_the_gauge():
    model, observed, ratio = _gain_case()
    k = profile_gains(model, observed, gauge='log')
    r = np.asarray(k) / np.asarray(ratio)
    assert float(np.std(np.log(r))) < 1e-6, 'k must equal sum(Q)/sum(M) up to one overall factor'


def test_log_gauge_sets_mean_log_k_to_zero():
    model, observed, _ = _gain_case()
    k = np.asarray(profile_gains(model, observed, gauge='log'))
    assert abs(float(np.mean(np.log(k)))) < 1e-6


def test_linear_gauge_sets_mean_k_to_one():
    model, observed, _ = _gain_case()
    k = np.asarray(profile_gains(model, observed, gauge='linear'))
    assert abs(float(np.mean(k)) - 1.0) < 1e-6


def test_the_two_gauges_differ_on_a_spread_gain_map():
    """They are different constraints, so on any non-degenerate map they give different k."""
    model, observed, _ = _gain_case()
    a = np.asarray(profile_gains(model, observed, gauge='log'))
    b = np.asarray(profile_gains(model, observed, gauge='linear'))
    assert np.abs(a - b).max() > 1e-3


def test_unknown_gauge_raises():
    model, observed, _ = _gain_case()
    with pytest.raises(ValueError, match='gauge'):
        profile_gains(model, observed, gauge='mean')


def test_neyman_weight_uses_the_data_only():
    """The denominator must not see the model. That is the entire reason this residual was chosen:
    a weight involving a re-drawn Monte-Carlo model makes the residual nonlinear in it, which
    displaces the fixed point permanently rather than merely adding noise."""
    data = jnp.asarray([4.0, 9.0, 16.0, 25.0])
    r1 = neyman_residual(jnp.asarray([4.0, 9.0, 16.0, 25.0]), data, 1e-6)
    np.testing.assert_allclose(np.asarray(r1), 0.0, atol=1e-6)      # zero at the data

    # doubling the model must change the residual by exactly model/sqrt(Q) — linear in the model
    a = np.asarray(neyman_residual(jnp.asarray([1.0, 2.0, 3.0, 4.0]), data, 1e-6))
    b = np.asarray(neyman_residual(jnp.asarray([2.0, 4.0, 6.0, 8.0]), data, 1e-6))
    np.testing.assert_allclose(b - a, np.array([1.0, 2.0, 3.0, 4.0]) / np.sqrt([4, 9, 16, 25]),
                               rtol=1e-6)


def test_neyman_floor_bounds_the_weight_on_an_empty_sensor():
    """1/sqrt(Q) diverges as Q -> 0; the floor is what stops a nearly unlit sensor dominating."""
    data = jnp.asarray([0.0, 1.0])
    r = np.asarray(neyman_residual(jnp.asarray([1.0, 1.0]), data, 0.25))
    assert r[0] == pytest.approx(1.0 / 0.5)          # floored at 0.25 -> sqrt = 0.5
    assert np.isfinite(r).all()
