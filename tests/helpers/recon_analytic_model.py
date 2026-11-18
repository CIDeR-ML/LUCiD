"""An analytic reconstruction problem for testing fit_track's loop without a simulator.

Nonlinear least squares on the nine track parameters, ill-conditioned on purpose, with a
key-dependent data draw -- enough to exercise the damping, the anneal and the metric refresh, with
no detector, no SIREN and no Monte Carlo. Shared by the tests that drive fit_track through the
Gauss-Newton loop.
"""
import numpy as np

# Ill-conditioned on purpose: a flat direction is what the Levenberg floor exists to bound, so a
# well-conditioned problem would not exercise the damping a refactor is most likely to disturb.
_SV = np.array([3.0, 1.7, 0.9, 0.45, 0.2, 0.08, 0.03, 0.011, 0.004])
_THETA_STAR = np.array([1000., 0.5, -0.3, 2.0, 0.1, -0.2, 0.3, 0.15, 5.0])
_U_START = np.array([1.4, 0.3, 0.2, -0.6, 0.1, 1.3, 0.5, -0.2, 0.35]) * 2.5   # SCALED units
_NONLIN = 0.12


def _scale9():
    from lucid.fitting import SCALE9
    return np.asarray(SCALE9, float)


def _design():
    """Fixed (12, 9) design with a controlled singular-value spread, in SCALED units."""
    rng = np.random.default_rng(20260813)
    U, _ = np.linalg.qr(rng.standard_normal((12, 9)))
    V, _ = np.linalg.qr(rng.standard_normal((9, 9)))
    return (U * _SV) @ V.T


class AnalyticModel:
    """Nonlinear least squares with a key-dependent data draw.

    ``r(u, key) = A·u + c·A·u² + ε(key)`` on the scaled offset ``u = (θ−θ*)/SCALE9``. The
    quadratic term stops it converging in one Newton step, so the anneal and the refresh cadence
    matter; ``ε`` makes the ``nkeys`` average and the metric refresh observable.
    """

    def __init__(self):
        self.A = _design()
        self.S = _scale9()
        self.c = _NONLIN

    def _eps(self, key):
        import jax
        return np.asarray(jax.random.normal(key, (self.A.shape[0],)), float) * 0.05

    def _r_and_J(self, th, key):
        u = (np.asarray(th, float) - _THETA_STAR) / self.S
        r = self.A @ u + self.c * (self.A @ (u * u)) + self._eps(key)
        # dr/dθ = (dr/du)·(du/dθ) = [A + 2c·A·diag(u)] / S
        J = (self.A + 2.0 * self.c * (self.A * u[None, :])) / self.S[None, :]
        return r, J

    def grad(self, th, oc, ot, key):
        r, J = self._r_and_J(th, key)
        return J.T @ r

    def fisher_ad(self, th, oc, ot, keys):
        acc = np.zeros((9, 9))
        for k in keys:
            _, J = self._r_and_J(th, k)
            acc += J.T @ J
        return acc / len(keys)



def least_squares_minimum(model, keys):
    """The minimiser of the key-averaged objective, by scipy, independently of the GN loop.

    The Jacobian does not depend on the key, so ``sum_k J^T r_k = 0`` is the normal equation of
    ``|| mean_k r_k ||^2``; ``scipy.optimize.least_squares`` solves that from the truth.
    """
    from scipy.optimize import least_squares
    eps = np.mean([model._eps(k) for k in keys], axis=0)
    A, c = model.A, model.c
    sol = least_squares(lambda u: A @ u + c * (A @ (u * u)) + eps, np.zeros(9),
                        jac=lambda u: A + 2.0 * c * (A * u[None, :]), xtol=1e-15, ftol=1e-15,
                        gtol=1e-15)
    return _THETA_STAR + model.S * sol.x
