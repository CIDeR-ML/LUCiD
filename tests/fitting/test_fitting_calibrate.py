"""The calibration entry points: ``CalibrationForward``, ``closure_data``, ``CalibrationProblem``,
``calibrate``.

These are what a newcomer calls, so they are tested at that level — give a simulator, sources and
a parameterisation, get parameters and a gain map back. A stub simulator stands in for the photon
forward: it reads the ``DetectorParams`` the parameterisation builds and is deterministic, so the
recovery has an exact answer and the assertions can be tight. The physics is held elsewhere:
the forward's golden (``tests/physics/test_forward_golden.py``).

The stub depends on the source, on every fitted field, and on the gains, because a stub that
ignored one of them would let a dropped argument pass unnoticed. Each source carries its OWN
per-sensor response matrix: sources whose patterns differ only by a scalar are degenerate under
profiled gains (the gains absorb the shared pattern and the parameters become unidentifiable,
as with a single source).
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from lucid.fitting import (FieldParams, CalibrationForward, CalibrationProblem,
                           calibrate, closure_data, gauss_newton_tx)

from tests.helpers import calib_toy as toy
from tests.helpers.calib_toy import FIELDS

NS = 24


def _dp_true():
    return toy.dp_true(NS)


def _stub_sim(source, dp, key):
    """Deterministic per-sensor charge. Reads the gains too, so a forward that dropped them would
    show up."""
    return toy.charge(source, dp), jnp.zeros(NS)


def _sources():
    return toy.sources(NS, seed=11, b_spread=0.5, b_scale=2.3)


def _noisy_sim(source, dp, key):
    """As `_stub_sim`, but the key perturbs the charge — so a seed can reach the answer."""
    q, t = _stub_sim(source, dp, key)
    return q * (1.0 + 0.02 * jax.random.normal(key, (NS,))), t


def _setup():
    params = FieldParams(_dp_true(), FIELDS, n_sensors=NS)
    return params, params.theta_from_physical()


GAINS = np.exp(0.2 * np.random.default_rng(5).standard_normal(NS))
GAINS /= np.exp(np.mean(np.log(GAINS)))            # the mean(log k)=0 gauge the fitter reports in
PROB_KW = dict(jacobian_draws=1, n_forward_draws=1)
FIT_KW = dict(steps=120, refresh=4, max_step=0.3, polyak=0, tx=gauss_newton_tx(lam=1e-4, mu=0.02))
PERTURBATION = np.array([0.3, -0.25, 0.2])


def _problem(sim=_stub_sim, gains=GAINS, seed=0, rows=None):
    params, theta_true = _setup()
    fwd = CalibrationForward(sim, _sources(), params, NS)
    data = closure_data(fwd, theta_true, gains, n_draws=1) if rows is None else rows
    return CalibrationProblem(fwd, data, seed=seed, **PROB_KW), params, theta_true


@pytest.fixture(scope='module')
def closure_run():
    """One 120-step closure from a perturbed start, shared by the tests that read it."""
    prob, params, theta_true = _problem()
    return calibrate(prob, theta_true + PERTURBATION, **FIT_KW), params, theta_true


class TestClosure:
    def test_closure_recovers_the_parameters_it_generated(self, closure_run):
        res, params, theta_true = closure_run
        frac = res['real'] / params.to_real(theta_true) - 1
        assert np.abs(frac).max() < 1e-4, frac

    def test_closure_recovers_the_gain_map(self, closure_run):
        ratio = closure_run[0]['gains'] / GAINS
        ratio = ratio / np.exp(np.mean(np.log(ratio)))          # compare up to the gauge
        assert np.std(np.log(ratio)) < 1e-4


class TestClosureData:
    def test_gains_are_baked_into_the_data(self):
        """Unit gains and a spread gain map must give different data, or the truth is not truth."""
        params, theta_true = _setup()
        fwd = CalibrationForward(_stub_sim, _sources(), params, NS)
        a = np.asarray(closure_data(fwd, theta_true, None, n_draws=1))
        b = np.asarray(closure_data(fwd, theta_true, GAINS, n_draws=1))
        assert np.abs(a - b).max() > 1e-3


class TestCalibrate:
    def test_mismatched_data_rows_raise_before_the_fit_runs(self):
        bad = jnp.ones((5, NS))                              # 2 sources, so 2 rows expected
        with pytest.raises(ValueError, match='rows'):
            _problem(rows=bad)

    def test_reports_real_units_and_a_gain_map(self):
        """`real` is to_real of the FITTED vector. Started 0.25 (log) away from truth, so a
        report taken from the start, or a fit that never moved, is off by ~28 %."""
        prob, params, theta_true = _problem()
        res = calibrate(prob, theta_true + 0.25, **FIT_KW)
        np.testing.assert_allclose(res['real'], params.to_real(res['theta']), rtol=1e-6)
        assert res['real'].shape == (params.P,)
        assert res['gains'].shape == (NS,)
        np.testing.assert_allclose(res['real'], np.exp(theta_true), rtol=1e-3)

    def test_history_includes_the_starting_point(self):
        prob, params, theta_true = _problem()
        res = calibrate(prob, theta_true, **{**FIT_KW, 'steps': 7})
        assert res['history'].shape == (8, params.P)
        assert res['loss'].shape == (7,)

    def test_seed_moves_both_key_streams(self):
        """Both key streams (forward and Jacobian) move with the seed. The stub is
        key-independent, so this checks the keys, not the fit; the next test checks the fit.

        Without the seed in the Jacobian's stream, every ensemble member draws the same Jacobian
        noise, so any bias it causes is shared and the ensemble's spread is blind to it.
        """
        a, b = _problem(seed=0)[0], _problem(seed=1)[0]
        assert not np.array_equal(a.jacobian_key(0), b.jacobian_key(0))
        assert not np.array_equal(a.forward_key(0), b.forward_key(0))

    def test_the_problem_methods_are_the_parts_of_grad_metric_loss(self):
        """``residuals``, ``jacobian`` and ``loss`` at one step are exactly what
        ``grad_metric_loss`` assembles, so a solver built on them fits the same estimator; and
        ``step`` selects the forward draw."""
        prob, _, theta_true = _problem(_noisy_sim)
        th = jnp.asarray(theta_true + 0.1, dtype=jnp.float32)
        g, H, loss = prob.grad_metric_loss(th, 3)
        r, J, n = prob.residuals(th, 3), prob.jacobian(th, 3), prob.n_configs
        np.testing.assert_array_equal(np.asarray(g), np.asarray(jnp.einsum('gnp,gn->p', J, r) / n))
        np.testing.assert_array_equal(np.asarray(H),
                                      np.asarray(jnp.einsum('gnp,gnq->pq', J, J) / n))
        assert loss == prob.loss(th, 3)
        assert not np.array_equal(np.asarray(r), np.asarray(prob.residuals(th, 4)))

    def test_the_jacobian_is_the_derivative_of_the_weighted_model(self):
        """CalibrationForward.jacobian against the closed form of the toy.

        The toy charge is ``exp(resp @ theta) * scale`` in the log fields, so
        ``dM/dtheta = M * resp`` and the Neyman Jacobian is ``exp(lk) M resp / sqrt(clip(Q))``.
        Checked with a non-trivial lk and a floor that binds on some sensors. Measured 1e-7
        relative (float32).
        """
        params, theta_true = _setup()
        th = jnp.asarray(theta_true, dtype=jnp.float32)
        srcs = _sources()
        M = np.asarray(CalibrationForward(_stub_sim, srcs, params, NS)(th, jax.random.PRNGKey(0),
                                                                       jnp.ones(NS)), float)
        data = M * np.linspace(0.5, 1.5, NS)[None, :]
        floor = float(np.percentile(data, 20))                 # binds on a fifth of the sensors
        lk = 0.1 * np.sin(np.arange(NS))
        got = np.asarray(CalibrationForward(_stub_sim, srcs, params, NS).jacobian(
            th, jnp.asarray(lk), jax.random.PRNGKey(1), jnp.asarray(data), floor, draws=(0,)),
            float)
        w = np.exp(lk)[None, :] / np.sqrt(np.clip(data, floor, None))
        want = np.stack([(w[g] * M[g])[:, None] * np.asarray(src['resp'], float)
                         for g, src in enumerate(srcs)])
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6 * np.abs(want).max())

    def test_jacobian_draws_are_distinct_draws(self):
        """``jacobian_draws`` averages INDEPENDENT draws. If the draw index did not reach the key,
        every draw would be the same one and the average would reduce no noise at all."""
        params, theta_true = _setup()
        th = jnp.asarray(theta_true, dtype=jnp.float32)
        data = jnp.abs(CalibrationForward(_noisy_sim, _sources(), params, NS)(
            th, jax.random.PRNGKey(5), jnp.ones(NS)))
        fwd = CalibrationForward(_noisy_sim, _sources(), params, NS)
        key = jax.random.PRNGKey(144)
        j0 = np.asarray(fwd.jacobian(th, jnp.zeros(NS), key, data, 1e-3, draws=(0,)))
        j1 = np.asarray(fwd.jacobian(th, jnp.zeros(NS), key, data, 1e-3, draws=(1,)))
        assert np.abs(j0 - j1).max() > 1e-3 * np.abs(j0).max(), 'two draws used the same key'

    def test_seed_changes_a_key_dependent_fit(self):
        """With a forward that actually reads its key, two seeds must give different answers."""
        kw = {**FIT_KW, 'steps': 12}
        (pa, _, theta_true), pb = _problem(_noisy_sim, seed=0), _problem(_noisy_sim, seed=1)[0]
        a = calibrate(pa, theta_true + 0.25, **kw)
        b = calibrate(pb, theta_true + 0.25, **kw)
        assert not np.array_equal(np.asarray(a['history']), np.asarray(b['history'])), \
            'the seed never reached the forward or Jacobian key streams'
