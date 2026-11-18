"""lucid.fitting.calib — the (source × wavelength) assembly, and the estimator's two functions.

Scope: what the class ADDS on top of the simulator — the keys, the (S, NS) row ordering, draw
averaging, and the equivalence of the serial and mapped dispatch. The physics comes
from the simulator and is not tested here, so these tests use a stub simulator and run in
milliseconds.

The stub is deliberately sensitive to all three inputs the assembly must route correctly — source,
wavelength (through the DetectorParams the parameterisation builds) and key. A stub that ignored
one of them would let a transposed layout or a swapped key pass.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from lucid.fitting.params import CalibrationParams
from lucid.fitting.calib import (ROLE_READOUT, ROLE_TRUTH, CalibrationForward, CalibrationProblem,
                                 neyman_residual, profile_gains)

W, NS, NSRC = 3, 6, 4


# Sources are plain dict pytrees.
def _sources(n):
    return [{'tag': jnp.asarray(float(i))} for i in range(n)]


def _stub_sim(source, dp, key):
    """(charge, time). Depends on source, on the wavelength via dp, and on the key."""
    seed = jax.random.uniform(key, ()) * 1e-3
    per_sensor = jnp.arange(NS, dtype=jnp.float32)
    q = (source['tag'] * 100.0 + dp.scattering.scatter_length + per_sensor) * (1.0 + seed)
    return q, jnp.zeros(NS)


def _key_code(keys):
    """A key's two uint32 words as four 16-bit halves, exact in float32: ``(..., 2) -> (..., 4)``."""
    halves = jnp.stack([keys >> 16, keys & 0xFFFF], axis=-1)
    return halves.reshape(*keys.shape[:-1], 4).astype(jnp.float32)


def _key_sim(source, dp, key):
    """(charge, time) whose charge IS the key."""
    return _key_code(key), jnp.zeros(4)


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
    out = np.asarray(fwd(_theta(p), jax.random.PRNGKey(0), jnp.ones(NS)))
    assert out.shape == (W * NSRC, NS)
    for wl in range(W):
        for s in range(NSRC):
            row = out[wl * NSRC + s]
            # stub: q = 100*s + scatter_length(wl) + arange(NS) + tiny key term
            assert abs((row[0] - row[0] % 1) - (100 * s + 10.0 * (wl + 1))) < 1.0, \
                f'row {wl*NSRC+s} is not (wl={wl}, s={s})'


@pytest.mark.parametrize('n_sources', [1, NSRC])
def test_each_row_draws_on_its_own_key(n_sources):
    """Row ``(wl, s)`` of the forward on ``key`` runs on ``keys(key)[s, wl]``, singleton and
    grouped sources alike: a stub whose charge is its key reads it back."""
    p = _params()
    fwd = CalibrationForward(_key_sim, _sources(n_sources), p, 4)
    key = jax.random.PRNGKey(7)
    out = np.asarray(fwd(_theta(p), key, jnp.ones(4)))
    want = np.asarray(_key_code(fwd.keys(key)))                  # (n_sources, W, 4)
    np.testing.assert_array_equal(out.reshape(W, n_sources, 4), want.transpose(1, 0, 2))


def test_average_is_the_mean_of_folded_draws():
    """average() is the mean over draws ``b`` on ``fold_in(key, b)``."""
    fwd, p = _forward()
    th, g, key = _theta(p), jnp.ones(NS), jax.random.PRNGKey(5)
    got = np.asarray(fwd.average(th, key, g, 3))
    want = np.mean([np.asarray(fwd(th, jax.random.fold_in(key, b), g)) for b in range(3)], axis=0)
    np.testing.assert_allclose(got, want, rtol=0, atol=0)


def test_a_jacobian_draw_is_the_derivative_of_the_forward_on_that_draw():
    """Draw ``h`` of the Jacobian differentiates the weighted forward on ``fold_in(key, h)``, the
    draw :meth:`average` names ``h``: the stub's charge scales with its key, so a Jacobian on any
    other key differs."""
    fwd, p = _forward()
    th, key = _theta(p), jax.random.PRNGKey(9)
    data = jnp.abs(fwd(th, jax.random.PRNGKey(0), jnp.ones(NS)))
    lk = 0.1 * jnp.sin(jnp.arange(NS, dtype=jnp.float32))
    weight = jnp.exp(lk) / jnp.sqrt(jnp.clip(data, 1e-3, None))
    for h in (0, 2):
        got = np.asarray(fwd.jacobian(th, lk, key, data, 1e-3, draws=(h,)))
        want = np.asarray(jax.jacfwd(
            lambda t: fwd(t, jax.random.fold_in(key, h), jnp.ones(NS)) * weight)(th))
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6 * np.abs(want).max())


def test_no_two_draws_of_a_recipe_share_a_key():
    """Every key a small recipe draws on is distinct: 3 seeds x 50 steps x 2 forward draws and 8
    Jacobian draws, x 8 sources x 5 wavelengths, the seeds' gain readouts, and 8 truth draws.

    The integer offsets this replaced (``base + 131*b``, ``+ 13*step``, ``+ 1000*s + wl``) aliased:
    within a seed, (step, source, wavelength) triples shared forward keys, and seed 0's step 348
    was the truth's draw 4.
    """
    n_sources, n_wl, n_steps, n_forward, n_jacobian = 8, 5, 50, 2, 8
    fwd = CalibrationForward(None, _sources(n_sources), CalibrationParams(n_wavelengths=n_wl), 4)
    fold = jax.random.fold_in

    def draws(key, n):
        return [fwd.keys(fold(key, b)) for b in range(n)]

    keys = draws(fold(jax.random.PRNGKey(0), ROLE_TRUTH), 8)
    for seed in range(3):
        prob = CalibrationProblem(fwd, jnp.ones((fwd.S, 4)), seed=seed)
        for step in range(n_steps):
            keys += draws(prob.forward_key(step), n_forward)
            keys += draws(prob.jacobian_key(step), n_jacobian)
        keys.append(fwd.keys(fold(jax.random.PRNGKey(seed), ROLE_READOUT)))
    flat = np.concatenate([np.asarray(k).reshape(-1, 2) for k in keys])
    assert len(flat) == n_sources * n_wl * (8 + 3 * (n_steps * (n_forward + n_jacobian) + 1))
    assert len(np.unique(flat, axis=0)) == len(flat)


def test_serial_and_mapped_dispatch_agree():
    """The dispatch strategy is an execution choice and must not change the answer.

    Multi-device runs shard sources across GPUs; a single device takes the serial path. If those
    disagreed, a result would depend on how many devices happened to be visible.
    """
    ser, p = _forward(map_fn=None)
    vmapped, _ = _forward(map_fn=lambda fn, ax: jax.vmap(fn, in_axes=ax))
    th, g = _theta(p), jnp.ones(NS)
    key = jax.random.PRNGKey(0)
    np.testing.assert_allclose(np.asarray(ser(th, key, g)),
                               np.asarray(vmapped(th, key, g)), rtol=0, atol=0)


def test_single_source_needs_no_grouped_dispatch():
    """A one-source setup must work — the grouped branch is empty and must not be entered."""
    p = _params()
    fwd = CalibrationForward(_stub_sim, _sources(1), p, NS)
    out = np.asarray(fwd(_theta(p), jax.random.PRNGKey(0), jnp.ones(NS)))
    assert out.shape == (W, NS)


# --------------------------------------------------------------------------------------------
# The estimator's two closed forms. Both are one line of arithmetic, and both are load-bearing:
# the gauge is what makes the gains identifiable at all, and the Neyman weight is the whole reason
# the fixed point sits at truth under a re-drawn Monte-Carlo model. They are tested directly here
# because the slow bit-exact engine pin cannot say WHICH property broke when it moves.
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
