"""Consistent fixed-dataset Gauss-Newton with a constrained per-PMT Schur block.

The unified calibration/reconstruction fitter (generalised from mie_hunter/gn_fast.py,
off its hard-coded 7-vector + toy engine). The model is

    predicted_charge_s = k[s] · M_s(theta)

where ``theta`` are the (log-space) GLOBAL parameters that enter the forward ``M``
(optical lengths, reflection, response, λ-deviation curves, ...) and ``k`` is a
per-PMT multiplicative factor (the QE/gain correction) — a large but DIAGONAL block
that is marginalised analytically by a Schur complement with the gauge ``mean(log k)=0``.

Loss is the τ-less √-MSE residual ``r = √(k·M) − √(truth)`` (Poisson-NLL biases the
optical scales ~1.3%; the √ transform is variance-stabilising and near-unbiased). The
Jacobian ``J = ∂√(k·M)/∂theta`` is recomputed only every ``refresh`` steps and CACHED,
then reused for BOTH the gradient ``Jᵀr`` and the Gauss-Newton Hessian ``JᵀJ`` so they
stay consistently normalised (a single fixed dataset per step — re-drawing fresh keys
each step is a Jensen-bias dead end). Damping = a median-diagonal ridge.

This module is the optimiser; the bridge from DetectorParams ↔ ``theta`` and the
``M_s(theta)`` forwards lives in :mod:`lucid.fitting.problem`.
"""

import numpy as np
import jax
import jax.numpy as jnp

sg = jax.lax.stop_gradient


def sqrt_residual(pred, truth, eps=1e-8):
    """Variance-stabilising √-MSE residual ``√(pred+eps) − √(truth+eps)``."""
    return jnp.sqrt(pred + eps) - jnp.sqrt(truth + eps)


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


def ridge_inverse(H, ridge=0.02, mu=0.3):
    """Damped inverse of the (reduced) global Hessian: median-diagonal ridge +
    additive Levenberg term + positive eigen-floor. Robust to indefiniteness."""
    n = H.shape[0]
    dg = np.clip(np.diag(H), 0, None)
    pos = dg[dg > 1e-30]
    base = np.median(pos) if pos.size else 1.0
    m = mu * base
    A = H + ridge * np.diag(dg) + m * np.eye(n) + 1e-12 * (np.abs(H).max() + 1e-30) * np.eye(n)
    ev, V = np.linalg.eigh(A)
    ev = np.clip(ev, 0.5 * m, None)
    return V @ np.diag(1.0 / ev) @ V.T


def _build_jacobian(predict_list, theta, lk, n_sensors, n_params, key_base, nb_h):
    """Per-source FD Jacobian Ji = ∂√(k·M_i)/∂theta (n_sensors, n_params), averaged over
    ``nb_h`` forward-noise batches. The expensive step — done only on refresh."""
    Js = []
    for i, src in enumerate(predict_list):
        Ji = np.zeros((n_sensors, n_params))
        for h in range(nb_h):
            ek, pk = _keys(key_base + 50000 + 1000 * i + h)
            Ji += src.fd_jacobian(theta, lk, ek, pk)
        Js.append(Ji / nb_h)
    return Js


def _keys(b0):
    return (jax.random.fold_in(jax.random.PRNGKey(b0), 1),
            jax.random.fold_in(jax.random.PRNGKey(b0 + 1), 1))


class SourceModel:
    """Wraps one source's forward ``M(theta)`` (per-sensor mean charge at k=1) into the
    √(k·M) residual ``m`` and a finite-difference Jacobian.

    ``forward(theta, ek, pk) -> (n_sensors,)`` is the mean charge for this source at the
    log-global params ``theta`` (with the per-PMT factor set to 1); ek/pk are
    forward-noise keys (engine + photon).

    The Jacobian is computed by FINITE DIFFERENCES with COMMON RANDOM NUMBERS (same keys
    for the base and perturbed evaluations) — the DiCE ``custom_vjp`` forward does not
    support forward-mode ``jvp``/``jacfwd``, and the expected-value calibration forward is
    deterministic given its keys, so CRN-FD is clean and low-noise.
    """

    def __init__(self, forward, eps=1e-8, fd_step=1e-3):
        self.forward = forward
        self.eps = eps
        self.fd_step = fd_step

        def _m(theta, lk, ek, pk):
            return jnp.sqrt(jnp.exp(lk) * forward(theta, ek, pk) + eps)
        self._m = jax.jit(_m)

    def m(self, theta, lk, ek, pk):
        return self._m(theta, lk, ek, pk)

    def fd_jacobian(self, theta, lk, ek, pk, h=None):
        """(n_sensors, n_params) FD Jacobian ∂m/∂theta with CRN (same ek/pk)."""
        h = self.fd_step if h is None else h
        theta = jnp.asarray(theta)
        base = self.m(theta, lk, ek, pk)
        cols = []
        for d in range(theta.shape[0]):
            pert = self.m(theta.at[d].add(h), lk, ek, pk)
            cols.append(np.array((pert - base) / h))
        return np.stack(cols, axis=1)


def fit(sources, truth_list, theta0, n_sensors, *,
        weights=None, lk0=None, steps=300, refresh=15, ridge=0.02, mu=0.3,
        eps=1e-8, step_max=0.08, kstep_max=0.3, fix=(), seed=0, nb_r=3, nb_h=4,
        gauge_k=True):
    """Run the constrained-Schur Gauss-Newton fit.

    Parameters
    ----------
    sources : list[SourceModel]
        One per calibration source; ``SourceModel.forward(theta, ek, pk)`` is the
        per-sensor mean charge at log-global params ``theta`` and per-PMT factor 1.
    truth_list : list[array (n_sensors,)]
        Observed per-sensor charge for each source.
    theta0 : array (n_params,)
        Initial LOG global params.
    n_sensors : int
    weights : array (n_sensors,) or None
        Per-sensor fit weight (e.g. a lit-PMT mask). Defaults to ones.
    lk0 : array (n_sensors,) or None
        Initial log per-PMT factor (defaults to zeros, i.e. k=1).
    fix : iterable[int]
        Indices of theta to hold fixed (zero gradient/step).

    Returns
    -------
    dict with keys: ``theta`` (real-space global params), ``log_theta``,
    ``k`` (per-PMT factor), ``history`` (per-step theta trajectory).
    """
    S = len(sources)
    n_params = int(np.asarray(theta0).shape[0])
    W = np.ones(n_sensors) if weights is None else np.asarray(weights, float)
    t_sqrt = [jnp.sqrt(jnp.asarray(truth_list[i]) + eps) for i in range(S)]
    lp = jnp.asarray(theta0)
    lkv = jnp.zeros(n_sensors) if lk0 is None else jnp.asarray(lk0)
    fix = list(fix)

    history = np.zeros((steps, n_params))
    Jcache = Htk = Minv = Pinv = None

    for s in range(steps):
        kb = 1000 + 777 * seed + 13 * s
        # residuals on the fixed per-step dataset
        rA, mA = [], []
        for i in range(S):
            mi = sources[i].m(lp, lkv, *_keys(kb + 7000 * i))
            rA.append(np.array(sg(mi - t_sqrt[i])))
            mA.append(np.array(sg(mi)))

        if s % refresh == 0:
            Jcache = _build_jacobian(sources, lp, lkv, n_sensors, n_params,
                                     9_000_000 + 7 * s, nb_h)
            Htt = np.zeros((n_params, n_params)); Htk = np.zeros((n_params, n_sensors))
            Hkk = np.zeros(n_sensors) + 1e-12
            for i in range(S):
                Jk = 0.5 * mA[i]
                Htt += (Jcache[i] * W[:, None]).T @ Jcache[i]
                Htk += (Jcache[i] * W[:, None]).T * Jk[None, :]
                Hkk += W * (Jk * Jk)
            Htt /= S; Htk /= S; Hkk /= S
            Minv = make_constrained_schur(Hkk) if gauge_k else (lambda X: (1.0 / Hkk) * X if np.ndim(X) == 1 else (1.0 / Hkk)[:, None] * X)
            Pinv = ridge_inverse(Htt - Htk @ Minv(Htk.T), ridge=ridge, mu=mu)

        gt = np.zeros(n_params); gk = np.zeros(n_sensors)
        for i in range(S):
            gt += Jcache[i].T @ (W * rA[i])
            gk += W * (0.5 * mA[i]) * rA[i]
        gt /= S; gk /= S
        geff = gt - Htk @ Minv(gk)
        for i in fix:
            geff[i] = 0.0
        dth = -(Pinv @ geff)
        for i in fix:
            dth[i] = 0.0
        dlk = -Minv(gk + Htk.T @ dth)
        lp = lp + jnp.asarray(np.clip(dth, -step_max, step_max))
        lkv = lkv + jnp.asarray(np.clip(dlk, -kstep_max, kstep_max))
        if gauge_k:
            lkv = lkv - jnp.mean(lkv)
        history[s] = np.array(jnp.exp(lp))

    return dict(theta=np.array(jnp.exp(lp)), log_theta=np.array(lp),
                k=np.clip(np.array(jnp.exp(lkv)), 1e-6, None), history=history)
