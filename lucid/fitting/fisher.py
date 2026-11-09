"""Fisher / Cramér-Rao bound for the calibration parameters, from the model alone.

The bound is a property of the observable, not of an estimator: it needs the model and a point,
no data. The per-PMT (QE/gain) block is marginalised by a gauge-constrained Schur complement, and
the information is the photon-count-scaled Poisson information of the ``√``-transformed charge,
``m = √(k·M + eps)`` (variance 1/4 per sensor, whatever the mean):

    F = 4 · (Σ_g Jᵀ W_g J − (Σ_g Jᵀ W_g Jk) · Minv · (Σ_g Jᵀ W_g Jk)ᵀ),   W_g = Nphot_g

with ``J = ∂m/∂theta`` and ``Jk = ∂m/∂log k = m/2``, summed over every (wavelength, source) row
``g``. The sum is INSIDE: the three blocks are accumulated over every row first and ONE Schur
complement is taken. That is not a detail of implementation -- the per-PMT gains are SHARED
across rows, so marginalising them per row would remove the same nuisance repeatedly and
overstate the information.

The covariance ``F⁻¹`` gives σ on the log-parameters = the FRACTIONAL σ.

⚠️ HONESTY FACTOR. The implicit-capture forward is ~√12 quieter than a real Poisson
shot count, so a toy-MC fit on it looks ~√12 too good. The reported bound multiplies σ
by ``honesty = √12`` (covariance × 12) so we quote the *honest* uncertainty, not the
engine's artificially-low scatter. Pass ``honesty=1.0`` for the raw engine bound.
"""

import numpy as np
import jax
import jax.numpy as jnp

__all__ = ['crb', 'make_constrained_schur']

SQRT12 = float(np.sqrt(12.0))


def make_constrained_schur(Hkk):
    """Return ``Minv(X)`` applying the inverse of the per-PMT block under the gauge
    ``mean(log k)=0`` (a rank-1 correction to the diagonal inverse ``1/Hkk``).

    ``Hkk`` is the (n_sensors,) diagonal of the per-PMT Gauss-Newton block. The gauge
    removes the per-PMT-k ↔ global-amplitude degeneracy (the sum-of-log-k direction).
    """
    Dinv = 1.0 / Hkk
    sD = Dinv.sum()

    def Minv(X):
        if np.ndim(X) == 1:
            return Dinv * X - Dinv * ((Dinv @ X) / sD)
        return Dinv[:, None] * X - Dinv[:, None] * ((Dinv @ X)[None, :] / sD)

    return Minv


def crb(forward, theta, *, lk_true=None, nphot=None, nb_h=3, honesty=SQRT12, ridge_rel=1e-9,
        eps=1e-8):
    """Cramér-Rao bound on the fit parameters at ``theta``.

    Parameters
    ----------
    forward : CalibrationForward
        The model: its simulator, sources and parameterisation (``FieldParams`` or
        ``CalibrationParams``, every wavelength row included).
    theta : array (P,)
        The fit vector the Fisher information is evaluated at (log parameters).
    lk_true : array (n_sensors,) or None
        Truth log per-PMT factor (defaults zeros, k=1).
    nphot : sequence of float or None
        Per-row photon count scaling the Poisson Fisher, one per (wavelength, source) row
        (defaults ones).
    nb_h : int
        Jacobian draws averaged per row.
    honesty : float
        σ inflation for the implicit engine's reduced variance (default √12).

    Returns
    -------
    dict: ``cov`` (P, P) honest covariance, ``sigma`` (fractional, the honest √diag),
    ``fisher`` (the k-marginalised Fisher), ``cov_raw`` (engine bound, no honesty factor).
    """
    sim, params, NS = forward.sim, forward.params, forward.NS
    theta = jnp.asarray(theta, dtype=jnp.float32)
    P = int(theta.shape[0])
    Nphot = np.ones(forward.S) if nphot is None else np.asarray(nphot, float)
    lk = jnp.zeros(NS) if lk_true is None else jnp.asarray(lk_true)

    def m(th, src, wl, lk, key):
        return jnp.sqrt(jnp.exp(lk) * sim(src, params.to_dp(th, wl, jnp.ones(NS)), key)[0] + eps)

    m_jit, jac_jit = jax.jit(m), jax.jit(jax.jacfwd(m))

    def key(b0):
        return jax.random.fold_in(jax.random.PRNGKey(b0), 1)

    Htt = np.zeros((P, P)); Htk = np.zeros((P, NS))
    Hkk = np.zeros(NS) + 1e-12
    for wl in range(forward.W):
        for s, src in enumerate(forward.sources):
            g = wl * forward.n_sources + s
            J = np.zeros((NS, P))
            for h in range(nb_h):
                J += np.asarray(jac_jit(theta, src, wl, lk, key(7_000_000 + 1000 * g + h)))
            J /= nb_h
            Jk = 0.5 * np.asarray(m_jit(theta, src, wl, lk, key(7_000_000 + 1000 * g)))
            w = Nphot[g] * np.ones(NS)
            Htt += (J * w[:, None]).T @ J
            Htk += (J * w[:, None]).T * Jk[None, :]
            Hkk += w * (Jk * Jk)

    Minv = make_constrained_schur(Hkk)
    F = 4.0 * (Htt - Htk @ Minv(Htk.T))
    cov_raw = np.linalg.inv(F + ridge_rel * np.median(np.diag(F)) * np.eye(P))
    cov = cov_raw * (honesty ** 2)
    sigma = np.sqrt(np.clip(np.diag(cov), 0, None))
    return dict(cov=cov, sigma=sigma, fisher=F, cov_raw=cov_raw)
