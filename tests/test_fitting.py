"""The calibration fitter on a toy forward: recovery, degeneracy, and the two masks.

A deterministic log-linear charge model ``M = exp(A @ theta)`` stands in for the photon forward,
so the estimator and the linear algebra are exercised without Monte-Carlo noise. That is the point
of the toy: with no noise there is an exact answer, and the fit either finds it to float32
precision or something is wrong. The tolerances below are therefore set at the precision the fit
actually reaches (~1e-7), not at a loose band that a broken fit could also pass.

TWO sources with different per-sensor response matrices share ONE per-PMT gain map. That is what
breaks the global-vs-gain degeneracy: a single source is degenerate because a free per-sensor gain
absorbs any per-sensor pattern, and the shared map across diverse sources over-determines it.
``test_single_source_is_degenerate`` holds that fact down.
"""
import re

import numpy as np
import jax.numpy as jnp
import pytest

from lucid.fitting import fit, crb, SourceModel
from lucid.fitting.schur_gn import make_constrained_schur, ridge_inverse
from lucid.fitting.gn import gauss_newton

NS, NP = 40, 4
_rngA = np.random.default_rng(7)
_A1 = np.linspace(0.2, 1.0, NS)[:, None] * np.cos(np.arange(NP)[None, :] + 1.0)
_A2 = _rngA.standard_normal((NS, NP)) * 0.6        # a structurally different source
THETA_TRUE = np.array([0.3, -0.5, 0.8, -0.2])
_rng = np.random.default_rng(0)
K_TRUE = np.exp(0.15 * _rng.standard_normal(NS))
K_TRUE /= np.exp(np.mean(np.log(K_TRUE)))          # the mean(log k)=0 gauge the fitter uses
START = THETA_TRUE + np.array([0.4, -0.3, 0.5, 0.3])
FIT_KW = dict(steps=200, refresh=5, jacobian_draws=1, max_step=0.3, lam=1e-4, mu=0.02)


def _make_toy_forward(A):
    def fwd(theta, ek, pk):
        return jnp.exp(jnp.asarray(A) @ theta)     # (NS,) mean charge at unit gain; deterministic
    return fwd


def _sources():
    return [SourceModel(_make_toy_forward(_A1)), SourceModel(_make_toy_forward(_A2))]


def _truth_charges():
    return [np.asarray(K_TRUE) * np.array(_make_toy_forward(A)(jnp.asarray(THETA_TRUE), None, None))
            for A in (_A1, _A2)]


def _gauged(k):
    r = np.asarray(k) / K_TRUE
    return r / np.exp(np.mean(np.log(r)))


class TestSchurAndRidge:
    """The CRB path's linear algebra. Not the fitter's — the fit profiles the gains instead."""

    def test_constrained_schur_kills_constant_mode(self):
        Hkk = np.linspace(1.0, 3.0, NS)
        out = make_constrained_schur(Hkk)(np.ones(NS))
        assert abs(float(np.sum(out))) < 1e-9      # the gauged sum-of-log-k mode is projected out

    def test_ridge_inverse_spd(self):
        ev = np.linalg.eigvalsh(ridge_inverse(_A1.T @ _A1, ridge=0.02, mu=0.3))
        assert np.all(ev > 0)


class TestRecovery:
    def test_recovers_toy_globals_and_gains(self):
        """Noise-free data has an exact answer, and the fit reaches it to float32 precision."""
        res = fit(_sources(), _truth_charges(), START, NS, **FIT_KW)
        assert np.abs(res['log_theta'] - THETA_TRUE).max() < 1e-5, res['log_theta']
        assert np.std(np.log(_gauged(res['k']))) < 1e-5
        assert res['loss'][-1] < 1e-10             # the residual is driven to zero, not just small

    def test_single_source_is_degenerate(self):
        """One source: the profiled gains absorb the per-sensor pattern, so theta is not recovered.

        The residual is NOT flat and the fit does not stall — it converges, driving the loss to
        ~1e-14, but to the WRONG theta. The ``mean(log k) = 0`` gauge leaves the overall
        normalisation determined, so there is a well-defined minimum; what a single source cannot
        do is separate the global parameters from a free per-sensor gain. Both halves are asserted,
        because "far from truth" alone is also satisfied by a fit that never moved.
        """
        res = fit(_sources()[:1], _truth_charges()[:1], START, NS,
                  **{**FIT_KW, 'steps': 120, 'lam': 1e-3, 'mu': 0.05})
        assert res['loss'][-1] < 1e-8, 'the fit did not converge, so this shows nothing'
        assert np.abs(res['log_theta'] - START).max() > 0.01, 'the fit never moved'
        assert np.abs(res['log_theta'] - THETA_TRUE).max() > 0.1, 'theta was recovered anyway'

    def test_polyak_returns_averaged_iterate(self):
        res = fit(_sources(), _truth_charges(), START, NS, **{**FIT_KW, 'polyak': 20})
        assert np.abs(res['log_theta'] - THETA_TRUE).max() < 1e-5


class TestFix:
    """``fix`` selects which parameters move. It has a caller: run_campaign.py freezes g."""

    # COUPLED, not diagonal. On a diagonal metric, zeroing the gradient and zeroing the step are
    # the same operation, so a test built on one cannot tell them apart — dropping the gradient
    # mask passes. Production uses `fix` on a coupled 7-parameter metric
    # (scripts/campaign/run_campaign.py freezes g), which is exactly the untestable case.
    _A = np.array([[3.0, 1.2, 0.7], [1.2, 2.0, 0.9], [0.7, 0.9, 4.0]])

    class _Coupled:
        def __init__(self, A):
            self.A = A

        def grad_metric_loss(self, t, s, refresh=True):
            return self.A @ (t - 1.0), self.A, float((t - 1.0) @ self.A @ (t - 1.0))

        def accumulate(self, t, d):
            return t + d

    def test_fix_freezes_a_parameter_at_its_start(self):
        prob = self._Coupled(self._A)
        free = gauss_newton(prob, np.zeros(3), 60, lam=0.0, mu=0.0)
        held = gauss_newton(prob, np.zeros(3), 60, lam=0.0, mu=0.0, fix=(1,))
        np.testing.assert_allclose(free['theta'], np.ones(3), atol=1e-9)
        assert held['theta'][1] == 0.0                       # frozen at its starting value

    def test_fix_solves_the_CONDITIONAL_minimum_not_the_masked_step(self):
        """The free parameters must land where they belong GIVEN the frozen one.

        This is the half a diagonal metric cannot test. With x1 held at 0, the analytic optimum of
        the remaining block is A[free,free]^-1 (A[free,free] @ 1 + A[free,1]*(1 - 0)), which is NOT
        the same as simply zeroing the step's middle component.
        """
        A, free_ix = self._A, [0, 2]
        sub = A[np.ix_(free_ix, free_ix)]
        rhs = sub @ np.ones(2) + A[np.ix_(free_ix, [1])].ravel() * 1.0
        want = np.linalg.solve(sub, rhs)
        assert np.abs(want - 1.0).max() > 0.1, 'the toy is too weakly coupled to discriminate'

        held = gauss_newton(self._Coupled(A), np.zeros(3), 60, lam=0.0, mu=0.0, fix=(1,))
        np.testing.assert_allclose(held['theta'][free_ix], want, atol=1e-8)


class TestRetiredKnobs:
    """Machinery that no longer exists must raise, and must say what replaced it.

    Matching on the knob NAME is not a gate: delete the whole `**retired` mechanism and Python
    raises `TypeError: fit() got an unexpected keyword argument 'eps'` all by itself, which
    contains the name. Measured — all nine passed against that mutation. So each case asserts the
    REPLACEMENT text, which only the curated table can produce.
    """

    @pytest.mark.parametrize('knob,expect', [
        ('eps', 'q_floor'), ('bake_k', 'profiled'), ('kstep_max', 'solved, not stepped'),
        ('lk0', 'initialise'), ('gauge_k', "gauge='log'"), ('ridge', 'lam'),
        ('nb_r', 'INERT'), ('nb_h', 'jacobian_draws'), ('step_max', 'max_step'),
    ])
    def test_retired_keyword_names_its_replacement(self, knob, expect):
        with pytest.raises(TypeError, match=re.escape(expect)):
            fit(_sources(), _truth_charges(), START, NS, **{knob: 1})


class TestFisherCRB:
    def test_crb_positive_and_honesty_factor(self):
        src = SourceModel(_make_toy_forward(_A1))
        c_raw = crb([src], THETA_TRUE, NS, nb_h=1, honesty=1.0)
        c = crb([src], THETA_TRUE, NS, nb_h=1)
        assert np.all(c['sigma'] > 0)
        np.testing.assert_allclose(c['sigma'] / c_raw['sigma'], np.sqrt(12.0), rtol=1e-6)
