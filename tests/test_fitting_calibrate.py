"""The calibration entry points: ``closure``, ``calibrate``, ``closure_data``.

These are what a newcomer calls, so they are tested at that level — give a simulator, sources and
a parameterisation, get parameters and a gain map back. A stub simulator stands in for the photon
forward: it reads the ``DetectorParams`` the parameterisation builds and is deterministic, so the
recovery has an exact answer and the assertions can be tight. The physics is gated bit-exactly
elsewhere (``tests/reconciliation/``).

The stub depends on the source, on every fitted field, and on the gains, because a stub that
ignored one of them would let a dropped argument pass unnoticed. Each source carries its OWN
per-sensor response matrix, which is not decoration: sources whose patterns differ by a scalar
alone are degenerate under profiled gains — the gains absorb the whole shared pattern and the
parameters become unidentifiable, exactly as a single source is. Source diversity is the lever.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from lucid.fitting import FieldParams, LogParams, calibrate, closure, closure_data
from lucid.fitting.calib import CalibrationForward, CalibrationJacobian

from tests import _calib_toy as toy
from tests._calib_toy import FIELDS

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
    import jax
    q, t = _stub_sim(source, dp, key)
    return q * (1.0 + 0.02 * jax.random.normal(key, (NS,))), t


def _setup():
    params = FieldParams(_dp_true(), FIELDS, n_sensors=NS)
    return params, params.theta_from_physical()


GAINS = np.exp(0.2 * np.random.default_rng(5).standard_normal(NS))
GAINS /= np.exp(np.mean(np.log(GAINS)))            # the mean(log k)=0 gauge the fitter reports in
FIT_KW = dict(steps=120, refresh=4, jacobian_draws=1, n_forward_draws=1,
              max_step=0.3, lam=1e-4, mu=0.02)


class TestClosure:
    def test_closure_recovers_the_parameters_it_generated(self):
        params, theta_true = _setup()
        res = closure(_stub_sim, _sources(), params, theta_true, gains=GAINS,
                      perturbation=0.3, n_truth_draws=1, **FIT_KW)
        assert np.abs(res['frac_error']).max() < 1e-4, res['frac_error']

    def test_closure_recovers_the_gain_map(self):
        params, theta_true = _setup()
        res = closure(_stub_sim, _sources(), params, theta_true, gains=GAINS,
                      perturbation=0.3, n_truth_draws=1, **FIT_KW)
        ratio = res['gains'] / res['gains_truth']
        ratio = ratio / np.exp(np.mean(np.log(ratio)))          # compare up to the gauge
        assert np.std(np.log(ratio)) < 1e-4

    def test_closure_actually_starts_away_from_truth(self):
        """A closure started AT truth tests almost nothing, so the perturbation must bite."""
        params, theta_true = _setup()
        res = closure(_stub_sim, _sources(), params, theta_true, gains=GAINS,
                      perturbation=0.3, n_truth_draws=1, **FIT_KW)
        assert np.abs(res['theta_start'] - res['theta_true']).max() > 0.05

    def test_explicit_perturbation_is_used_as_given(self):
        params, theta_true = _setup()
        pert = np.array([0.2, -0.15, 0.1])
        res = closure(_stub_sim, _sources(), params, theta_true, gains=GAINS,
                      perturbation=pert, n_truth_draws=1, **FIT_KW)
        np.testing.assert_allclose(res['theta_start'] - res['theta_true'], pert, atol=1e-12)

    def test_n_sensors_required_when_gains_are_absent(self):
        params, theta_true = _setup()
        with pytest.raises(ValueError, match='n_sensors'):
            closure(_stub_sim, _sources(), params, theta_true, **FIT_KW)


class TestClosureData:
    def test_generated_data_matches_the_forward_at_truth(self):
        params, theta_true = _setup()
        fwd = CalibrationForward(_stub_sim, _sources(), params, NS)
        data = closure_data(fwd, theta_true, GAINS, n_draws=1, key_base=5000)
        direct = fwd(jnp.asarray(theta_true, dtype=jnp.float32), 5000, jnp.asarray(GAINS))
        np.testing.assert_allclose(np.asarray(data), np.asarray(direct), rtol=0, atol=0)

    def test_gains_are_baked_into_the_data(self):
        """Unit gains and a spread gain map must give different data, or the truth is not truth."""
        params, theta_true = _setup()
        fwd = CalibrationForward(_stub_sim, _sources(), params, NS)
        a = np.asarray(closure_data(fwd, theta_true, None, n_draws=1))
        b = np.asarray(closure_data(fwd, theta_true, GAINS, n_draws=1))
        assert np.abs(a - b).max() > 1e-3

    def test_row_count_is_wavelengths_times_sources(self):
        params, theta_true = _setup()
        fwd = CalibrationForward(_stub_sim, _sources(), params, NS)
        assert closure_data(fwd, theta_true, GAINS, n_draws=1).shape == (params.W * 2, NS)


class TestCalibrate:
    def test_mismatched_data_rows_raise_before_the_fit_runs(self):
        params, theta_true = _setup()
        bad = jnp.ones((5, NS))                              # 2 sources, so 2 rows expected
        with pytest.raises(ValueError, match='rows'):
            calibrate(_stub_sim, _sources(), params, bad, theta_true, steps=1)

    def test_reports_real_units_and_a_gain_map(self):
        params, theta_true = _setup()
        fwd = CalibrationForward(_stub_sim, _sources(), params, NS)
        data = closure_data(fwd, theta_true, GAINS, n_draws=1)
        res = calibrate(_stub_sim, _sources(), params, data, theta_true, **FIT_KW)
        assert res['real'].shape == (params.P,)
        assert res['gains'].shape == (NS,)
        np.testing.assert_allclose(res['real'], np.exp(theta_true), rtol=1e-3)

    def test_history_includes_the_starting_point(self):
        params, theta_true = _setup()
        fwd = CalibrationForward(_stub_sim, _sources(), params, NS)
        data = closure_data(fwd, theta_true, GAINS, n_draws=1)
        res = calibrate(_stub_sim, _sources(), params, data, theta_true,
                        **{**FIT_KW, 'steps': 7})
        assert res['history'].shape == (8, params.P)
        assert res['loss'].shape == (7,)

    def test_seed_moves_both_key_streams(self):
        """Named for what it checks. The stub is key-INDEPENDENT, so the fit is byte-identical
        across seeds here (measured) — what this asserts is that both stream bases move.

        Without a seed term in the JACOBIAN's base, every ensemble member draws the same Jacobian
        noise: anything it displaces biases them all alike and never appears in the spread, so the
        ensemble's own error bar is blind to it. The next test is the one that shows the seed
        actually reaching the answer.
        """
        params, theta_true = _setup()
        fwd = CalibrationForward(_stub_sim, _sources(), params, NS)
        data = closure_data(fwd, theta_true, GAINS, n_draws=1)
        a = calibrate(_stub_sim, _sources(), params, data, theta_true + 0.25, seed=0, **FIT_KW)
        b = calibrate(_stub_sim, _sources(), params, data, theta_true + 0.25, seed=1, **FIT_KW)
        assert a['problem'].jacobian.key0 != b['problem'].jacobian.key0
        assert a['problem'].forward_key0 != b['problem'].forward_key0

    def test_seed_changes_a_key_dependent_fit(self):
        """With a forward that actually reads its key, two seeds must give different answers."""
        params, theta_true = _setup()
        fwd = CalibrationForward(_noisy_sim, _sources(), params, NS)
        data = closure_data(fwd, theta_true, GAINS, n_draws=1)
        kw = {**FIT_KW, 'steps': 12}
        a = calibrate(_noisy_sim, _sources(), params, data, theta_true + 0.25, seed=0, **kw)
        b = calibrate(_noisy_sim, _sources(), params, data, theta_true + 0.25, seed=1, **kw)
        assert not np.array_equal(np.asarray(a['history']), np.asarray(b['history'])), \
            'the seed never reached the forward or Jacobian key streams'

    def test_the_two_dispatch_routes_agree(self):
        """Sources as traced pytrees vs sources as opaque closures — one estimator either way.

        This is what makes `fit` and `calibrate` the same machinery rather than two paths that
        happen to look alike, and until now nothing asserted it: no test anywhere passed
        ``predict=``. The routes compile differently (one program for all sources, versus one per
        source), so they agree only up to float32 constant-folding.
        """
        params, theta_true = _setup()
        srcs = _sources()

        def pred(th, src, wl, g, k):
            return _stub_sim(src, params.to_dp(th, wl, g), k)[0]

        th = jnp.asarray(theta_true, dtype=jnp.float32)
        a = CalibrationForward(_stub_sim, srcs, params, NS)(th, 77, jnp.ones(NS))
        b = CalibrationForward(None, srcs, params, NS, predict=pred)(th, 77, jnp.ones(NS))
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0,
                                   err_msg='the two forward routes disagree')

        data = jnp.abs(a) + 1e-3
        ja = CalibrationJacobian(_stub_sim, srcs, params, NS)(
            th, jnp.zeros(NS), 2, data, 1e-3, draws=(0, 1))
        jb = CalibrationJacobian(None, srcs, params, NS, predict=pred)(
            th, jnp.zeros(NS), 2, data, 1e-3, draws=(0, 1))
        scale = float(jnp.abs(ja).max())
        assert scale > 1e-3, 'the Jacobian is ~zero, so agreement proves nothing'
        assert float(jnp.abs(ja - jb).max()) / scale < 1e-6, 'the two Jacobian routes disagree'


class TestLogParams:
    def test_to_dp_refuses_rather_than_guessing(self):
        with pytest.raises(TypeError, match='LogParams'):
            LogParams(3).to_dp(jnp.zeros(3), 0, jnp.ones(NS))

    def test_to_real_is_exp(self):
        np.testing.assert_allclose(LogParams(3).to_real(np.log([2.0, 3.0, 4.0])),
                                   [2.0, 3.0, 4.0], rtol=1e-12)
