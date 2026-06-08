"""Fisher / Cramér-Rao bound at the truth for the calibration globals.

Generalised from mie_hunter/fisher_wl2.py. The per-PMT (QE/gain) block is marginalised
by the same gauge-constrained Schur complement as the optimiser, and the global Fisher is
the photon-count-scaled Poisson information

    F = 4 · Σ_sources Nphot_s · (Jᵀ W J − Jᵀ W Jk · Minv · (Jᵀ W Jk)ᵀ)

with ``J = ∂√M/∂theta`` evaluated at the truth. The covariance ``F⁻¹`` gives σ on the
log-parameters = the FRACTIONAL σ.

⚠️ HONESTY FACTOR. The implicit-capture forward is ~√12 quieter than a real Poisson
shot count, so a toy-MC fit on it looks ~√12 too good. The reported bound multiplies σ
by ``honesty = √12`` (covariance × 12) so we quote the *honest* uncertainty, not the
engine's artificially-low scatter. Pass ``honesty=1.0`` for the raw engine bound.
"""

import numpy as np
import jax
import jax.numpy as jnp

from lucid.fitting.gauss_newton import make_constrained_schur, _keys

SQRT12 = float(np.sqrt(12.0))


def crb(sources, theta_true, n_sensors, *, lk_true=None, weights=None,
        nphot=None, nb_h=3, honesty=SQRT12, ridge_rel=1e-9):
    """Cramér-Rao bound on the global parameters at ``theta_true``.

    Parameters
    ----------
    sources : list[SourceModel]
        Forwards (per-sensor mean charge at k=1), as for ``gauss_newton.fit``.
    theta_true : array (n_params,)
        LOG global parameters at truth (Fisher is evaluated here).
    n_sensors : int
    lk_true : array (n_sensors,) or None
        Truth log per-PMT factor (defaults zeros, k=1).
    weights : array (n_sensors,) or None
        Per-sensor weight / lit mask (defaults ones).
    nphot : list[float] or None
        Per-source photon count scaling the Poisson Fisher (defaults ones).
    honesty : float
        σ inflation for the implicit engine's reduced variance (default √12).

    Returns
    -------
    dict: ``cov`` (n_params,n_params) honest covariance, ``sigma`` (fractional, the
    honest √diag), ``fisher`` (the k-marginalised global Fisher), ``cov_raw`` (engine
    bound, no honesty factor).
    """
    S = len(sources)
    n_params = int(np.asarray(theta_true).shape[0])
    W = np.ones(n_sensors) if weights is None else np.asarray(weights, float)
    Nphot = np.ones(S) if nphot is None else np.asarray(nphot, float)
    theta = jnp.asarray(theta_true)
    lk = jnp.zeros(n_sensors) if lk_true is None else jnp.asarray(lk_true)

    Htt = np.zeros((n_params, n_params)); Htk = np.zeros((n_params, n_sensors))
    Hkk = np.zeros(n_sensors) + 1e-12
    for i in range(S):
        Ji = np.zeros((n_sensors, n_params))
        for h in range(nb_h):
            ek, pk = _keys(7_000_000 + 1000 * i + h)
            Ji += sources[i].fd_jacobian(theta, lk, ek, pk)
        Ji /= nb_h
        mi = np.array(sources[i].m(theta, lk, *_keys(7_000_000 + 1000 * i)))
        Jk = 0.5 * mi
        w = Nphot[i] * W
        Htt += (Ji * w[:, None]).T @ Ji
        Htk += (Ji * w[:, None]).T * Jk[None, :]
        Hkk += w * (Jk * Jk)

    Minv = make_constrained_schur(Hkk)
    F = 4.0 * (Htt - Htk @ Minv(Htk.T))
    cov_raw = np.linalg.inv(F + ridge_rel * np.median(np.diag(F)) * np.eye(n_params))
    cov = cov_raw * (honesty ** 2)
    sigma = np.sqrt(np.clip(np.diag(cov), 0, None))
    return dict(cov=cov, sigma=sigma, fisher=F, cov_raw=cov_raw)
