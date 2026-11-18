"""The calibration fitter on a toy forward: recovery, degeneracy, the readout, ``fix``, the bound.

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
import numpy as np
import jax.numpy as jnp
import pytest

from lucid.fitting import CalibrationForward, CalibrationProblem, calibrate, crb, gauss_newton_tx
from lucid.fitting.fisher import make_constrained_schur
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
FIT_KW = dict(steps=200, refresh=5, max_step=0.3, polyak=0, tx=gauss_newton_tx(lam=1e-4, mu=0.02))


class _LogLinear:
    """The toy's parameterisation: ``theta`` is handed to the simulator as it is."""
    W, P = 1, NP

    def to_dp(self, theta, wl, gains):
        return theta, gains

    def to_real(self, theta):
        return np.exp(np.asarray(theta))


def _sim(source, dp, key):
    """``N k exp(A theta)``: mean charge, deterministic. The source is ``{'A', 'N'}``."""
    theta, gains = dp
    return source['N'] * gains * jnp.exp(source['A'] @ theta), None


def _model(As=(_A1, _A2), nphot=None):
    nphot = [1.0] * len(As) if nphot is None else nphot
    return CalibrationForward(_sim, [{'A': jnp.asarray(A), 'N': jnp.asarray(N)}
                                     for A, N in zip(As, nphot)], _LogLinear(), NS)


def _truth(As=(_A1, _A2)):
    return np.stack([K_TRUE * np.exp(A @ THETA_TRUE) for A in As])


def _fit(model, data, start=START, **kw):
    return calibrate(CalibrationProblem(model, data, jacobian_draws=1), start, **{**FIT_KW, **kw})


def _gauged(k):
    r = np.asarray(k) / K_TRUE
    return r / np.exp(np.mean(np.log(r)))


class TestSchur:
    """The CRB path's linear algebra. Not the fitter's — the fit profiles the gains instead."""

    def test_constrained_schur_kills_constant_mode(self):
        Hkk = np.linspace(1.0, 3.0, NS)
        out = make_constrained_schur(Hkk)(np.ones(NS))
        assert abs(float(np.sum(out))) < 1e-9      # the gauged sum-of-log-k mode is projected out

    def test_constrained_schur_matrix_branch_is_the_gauged_inverse(self):
        """The matrix branch (the one ``crb`` uses) against the null-space form of the same inverse.

        Minimising x^T D x / 2 - b^T x subject to sum(x) = 0 has the solution P (P^T D P)^-1 P^T b
        for any basis P of the constraint, so the two must agree column by column.
        """
        from scipy.linalg import null_space
        Hkk = np.linspace(1.0, 3.0, NS)
        X = np.random.default_rng(3).standard_normal((NS, 5))
        P = null_space(np.ones((1, NS)))
        want = P @ np.linalg.solve(P.T @ np.diag(Hkk) @ P, P.T @ X)
        np.testing.assert_allclose(make_constrained_schur(Hkk)(X), want, atol=1e-12)


class TestRecovery:
    def test_recovers_toy_globals_and_gains(self):
        """Noise-free data has an exact answer, and the fit reaches it to float32 precision."""
        res = _fit(_model(), _truth())
        assert np.abs(res['theta'] - THETA_TRUE).max() < 1e-5, res['theta']
        assert np.std(np.log(_gauged(res['gains']))) < 1e-5
        assert res['loss'][-1] < 1e-10             # the residual is driven to zero, not just small

    def test_scipy_least_squares_runs_on_the_same_problem(self):
        """Any solver: ``residuals`` and ``jacobian`` at one draw are all scipy needs, and on the
        noise-free toy it reaches the same answer as ``calibrate``."""
        from scipy.optimize import least_squares
        prob = CalibrationProblem(_model(), _truth(), jacobian_draws=1)
        sol = least_squares(lambda t: np.ravel(prob.residuals(t, step=0)), START,
                            jac=lambda t: np.reshape(prob.jacobian(t, step=0), (-1, NP)))
        assert np.abs(sol.x - THETA_TRUE).max() < 1e-4, sol.x

    def test_single_source_is_degenerate(self):
        """One source: the profiled gains absorb the per-sensor pattern, so theta is not recovered.

        The fit still converges (loss ~0) to the WRONG theta rather than sitting flat: the
        ``mean(log k) = 0`` gauge fixes the overall normalisation even though theta and the
        per-sensor gain are not separately identifiable. Convergence and movement are asserted
        too, because "far from truth" alone is also passed by a fit that never moved.
        """
        res = _fit(_model((_A1,)), _truth((_A1,)), steps=120,
                   tx=gauss_newton_tx(lam=1e-3, mu=0.05))
        assert res['loss'][-1] < 1e-8, 'the fit did not converge, so this shows nothing'
        assert np.abs(res['theta'] - START).max() > 0.01, 'the fit never moved'
        assert np.abs(res['theta'] - THETA_TRUE).max() > 0.1, 'theta was recovered anyway'

    def test_polyak_returns_averaged_iterate(self):
        """``polyak=20`` reads out the mean of the last 20 iterates, not the last one.

        ``history`` has the start as row 0, so the last 20 iterates are ``history[-20:]``. A
        shorter run (8 steps, still moving) keeps the mean visibly different from the final iterate.
        """
        res = _fit(_model(), _truth(), polyak=20)
        np.testing.assert_allclose(res['theta'], np.mean(res['history'][-20:], 0), atol=1e-6)
        assert np.abs(res['theta'] - THETA_TRUE).max() < 1e-5
        short = _fit(_model(), _truth(), steps=8, polyak=20)
        hist = np.asarray(short['history'])
        assert hist.shape == (9, NP)
        # window longer than the run: the mean of the 8 iterates, the start excluded
        np.testing.assert_allclose(short['theta'], hist[1:].mean(0), atol=1e-6)
        assert np.abs(hist[1:].mean(0) - hist[-1]).max() > 1e-3


class TestFix:
    """``fix`` selects which parameters move: a masked parameter must not move at all."""

    # COUPLED, not diagonal. On a diagonal metric, zeroing the gradient and zeroing the step are
    # the same operation, so a test built on one cannot tell them apart — dropping the gradient
    # mask passes. Production uses `fix` on a coupled metric, so the toy must be coupled too.
    _A = np.array([[3.0, 1.2, 0.7], [1.2, 2.0, 0.9], [0.7, 0.9, 4.0]])

    class _Coupled:
        def __init__(self, A):
            self.A = A

        def grad_metric_loss(self, t, s, refresh=True):
            return self.A @ (t - 1.0), self.A, float((t - 1.0) @ self.A @ (t - 1.0))

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


class TestFisherCRB:
    """``crb`` against the Poisson Fisher of the toy, written down independently.

    For the log-linear toy the observed charge is Poisson with mean ``mu_si = N_s k_i exp(A_s theta)``,
    so the full information on ``(theta, log k)`` is ``sum_s G_s^T diag(mu_s) G_s`` with
    ``G_s = [A_s | I]`` (``d log mu / d(theta, log k)``) -- textbook Poisson Fisher, no sqrt
    transform, no Schur complement. The gauge ``sum log k = 0`` is imposed by an explicit
    null-space basis of the constraint, and the per-PMT block is marginalised by inverting the whole
    matrix and taking the theta corner. ``crb`` does none of these things the same way (sqrt
    observable with Var = 1/4, rank-1 gauge correction to a diagonal inverse, Schur complement), so
    agreement is a real check. Measured agreement: 5e-8 relative (float32 Jacobian).

    TWO sources and unequal photon counts: with one source the Fisher has rank 1 and every bound
    on it is ~1e4 (the gains absorb everything).
    """
    NPHOT = (3000.0, 800.0)

    def _analytic(self, lk):
        from scipy.linalg import block_diag, null_space
        F = np.zeros((NP + NS, NP + NS))
        for A, N in zip((_A1, _A2), self.NPHOT):
            mu = N * np.exp(lk) * np.exp(A @ THETA_TRUE)
            G = np.hstack([A, np.eye(NS)])
            F += (G * mu[:, None]).T @ G
        T = block_diag(np.eye(NP), null_space(np.ones((1, NS))))   # theta, gauged log k
        cov = np.linalg.inv(T.T @ F @ T)[:NP, :NP]
        return np.linalg.inv(cov), cov

    def test_crb_is_the_gauge_marginalised_poisson_fisher(self):
        lk = np.log(K_TRUE)
        c = crb(_model(), THETA_TRUE, lk_true=lk, nphot=list(self.NPHOT), nb_h=1, honesty=1.0)
        F_want, cov_want = self._analytic(lk)
        # rtol 1e-6 against a measured 5e-8. A factor 2 on F, dropping the Schur term (~10x on
        # some eigenvalues), or dropping the gauge correction (43 % on F) all fail by orders.
        np.testing.assert_allclose(c['fisher'], F_want, rtol=1e-6, atol=1e-6 * np.abs(F_want).max())
        np.testing.assert_allclose(c['cov_raw'], cov_want, rtol=1e-5,
                                   atol=1e-6 * np.abs(cov_want).max())

    def test_honesty_inflates_the_covariance_by_twelve(self):
        """The default bound is the raw one times 12 in covariance (sigma times sqrt 12)."""
        kw = dict(lk_true=np.log(K_TRUE), nphot=list(self.NPHOT), nb_h=1)
        raw = crb(_model(), THETA_TRUE, honesty=1.0, **kw)
        hon = crb(_model(), THETA_TRUE, **kw)
        np.testing.assert_allclose(hon['cov'], 12.0 * raw['cov_raw'], rtol=1e-6)
        np.testing.assert_allclose(hon['sigma'], np.sqrt(12.0 * np.diag(raw['cov_raw'])), rtol=1e-6)

    @pytest.mark.slow
    def test_poisson_closures_scatter_at_the_bound(self):
        """The bound is attained: Poisson-noised closures through ``calibrate`` scatter at ``cov_raw``.

        200 replicas of Poisson(N k exp(A theta)) data, each fitted from the truth by the Neyman
        estimator with profiled gains. That estimator is asymptotically efficient at these counts
        (hundreds to thousands per sensor), so the sample variance of each fitted parameter should
        equal the CRB diagonal. With 200 replicas the variance ratio has a standard error of
        sqrt(2/199) = 0.10; the band [0.7, 1.4] is about 3 of them. Measured ratios: 1.07, 1.13,
        0.95, 0.93. A factor 2 in the Fisher puts the ratio at 2 or 0.5. The Neyman estimator's
        O(1/N) bias was measured at <= 0.14 sigma; the bias bound is 0.35 sigma.
        """
        rng = np.random.default_rng(1)
        model = _model(nphot=self.NPHOT)
        mean = [K_TRUE * N * np.exp(A @ THETA_TRUE) for A, N in zip((_A1, _A2), self.NPHOT)]
        fits = np.array([
            _fit(model, np.stack([rng.poisson(m).astype(np.float32) for m in mean]), THETA_TRUE,
                 steps=30)['theta']
            for _ in range(200)])
        cov = crb(_model(), THETA_TRUE, lk_true=np.log(K_TRUE),
                  nphot=list(self.NPHOT), nb_h=1, honesty=1.0)['cov_raw']
        sig = np.sqrt(np.diag(cov))
        ratio = fits.var(0, ddof=1) / sig ** 2
        assert np.all((ratio > 0.7) & (ratio < 1.4)), ratio
        assert np.all(np.abs(fits.mean(0) - THETA_TRUE) < 0.35 * sig), (fits.mean(0) - THETA_TRUE) / sig
