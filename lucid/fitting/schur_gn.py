"""The Schur-block sqrt-residual family: what the Neyman consolidation did not absorb.

Everything here shares one shape — a ``sqrt``-transformed residual with the per-PMT nuisance
carried as a Schur block, rather than the Neyman residual with the gains profiled that
:mod:`lucid.fitting.calibrate` now runs. That is what the name says, and the name matters: this
file used to be called ``gauss_newton.py``, which was wrong twice over. It has not contained the
Gauss-Newton loop since the consolidation, and worse, importing it SHADOWED
``lucid.fitting.gauss_newton`` — the package's headline export — so
``from lucid.fitting import gauss_newton`` returned this module and could not be called.

The calibration optimiser that used to live here is gone. It ran a sqrt-MSE residual with the
per-PMT gains carried as a free Schur block, and both halves of that were the arm the calibration
campaign rejected — a residual nonlinear in a re-drawn Monte-Carlo model has a permanently
displaced fixed point, and a free per-sensor gain overfits a single noisy draw. Its replacement is
:func:`lucid.fitting.calibrate.fit`, which runs the Neyman residual, profiles the gains in closed
form, and steps with the loop reconstruction uses.

Three things stayed behind, for one reason each:

``SourceModel``          the Fisher/CRB path (:mod:`lucid.fitting.fisher`) needs its
                         ``ad_jacobian``, and that Jacobian is of ``sqrt(k*M)`` — the CRB is a
                         property of the observable, not of the estimator that was retired.
``make_constrained_schur``  the CRB still marginalises the per-PMT block by Schur complement,
                         because it asks a different question than the fit does: the fit profiles
                         the gains at a point, the bound must integrate over them.
``ChargeTimeModel`` /    the joint charge + first-arrival-time fit, which carries a SECOND per-PMT
``fit_charge_time``      block (an additive t0 alongside the multiplicative k) that the shared
                         calibration problem does not yet model. Consolidating it needs the timing
                         residual in :mod:`lucid.fitting.calib` first, so it is deferred rather
                         than duplicated onto a residual that cannot express it.

``ridge_inverse`` remains as the damping those two use; :func:`lucid.fitting.transforms.damped_matrix` is
the one convention for everything that has been consolidated, and differs from it in documented
ways (an eigen-floor here, provably inert whenever the Levenberg term is positive).

``sqrt_residual`` is the fourth, and it is the exception: it has **no caller anywhere in the tree**.
The two models above build their ``sqrt`` transform inline rather than calling it. It is kept, and
still exported, because it names the residual the consolidation replaced — a reader comparing the
two estimators can see the retired one written down in one line instead of reconstructing it from
a diff. That is a documentation reason, not a code one, and it is the only thing holding it here.
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


def _keys(b0):
    return (jax.random.fold_in(jax.random.PRNGKey(b0), 1),
            jax.random.fold_in(jax.random.PRNGKey(b0 + 1), 1))


class SourceModel:
    """Wraps one source's forward ``M(theta)`` (per-sensor mean charge at k=1) into the
    √(k·M) residual ``m`` and a finite-difference Jacobian.

    ``forward(theta, ek, pk) -> (n_sensors,)`` is the mean charge for this source at the
    log-global params ``theta`` (with the per-PMT factor set to 1); ek/pk are
    forward-noise keys (engine + photon).

    The Jacobian ``J = ∂m/∂theta`` (n_sensors, n_params) is computed by FORWARD-MODE
    autodiff (``ad_jacobian``, ``jax.jacfwd``) — P inputs → N_sensors outputs, so
    forward-mode is the efficient mode. AD is unbiased and (for the discrete scatter/mie/g
    channels, whose gradient comes via the DiCE score) ~10× LOWER variance than the
    finite-difference secant, which is noisy through the per-photon decision flips. The
    legacy CRN-FD path (``fd_jacobian``) is retained for cross-checking; it was the default
    only because the old ``custom_vjp`` step blocked ``jacfwd`` (now removed).
    """

    def __init__(self, forward, eps=1e-8, fd_step=1e-3):
        self.forward = forward
        self.eps = eps
        self.fd_step = fd_step

        def _m(theta, lk, ek, pk):
            return jnp.sqrt(jnp.exp(lk) * forward(theta, ek, pk) + eps)
        self._m = jax.jit(_m)
        # Forward-mode Jacobian ∂m/∂theta (argnums=0). Now that the step is custom_vjp-free,
        # jacfwd works; key is a traced arg so different keys reuse the same compiled program.
        self._adjac = jax.jit(jax.jacfwd(_m, argnums=0))

    def m(self, theta, lk, ek, pk):
        return self._m(theta, lk, ek, pk)

    def ad_jacobian(self, theta, lk, ek, pk):
        """(n_sensors, n_params) forward-mode AD Jacobian ∂m/∂theta. Unbiased, low-variance."""
        return np.asarray(self._adjac(jnp.asarray(theta), lk, ek, pk))

    def fd_jacobian(self, theta, lk, ek, pk, h=None):
        """(n_sensors, n_params) FD Jacobian ∂m/∂theta with CRN (same ek/pk). Cross-check only."""
        h = self.fd_step if h is None else h
        theta = jnp.asarray(theta)
        base = self.m(theta, lk, ek, pk)
        cols = []
        for d in range(theta.shape[0]):
            pert = self.m(theta.at[d].add(h), lk, ek, pk)
            cols.append(np.array((pert - base) / h))
        return np.stack(cols, axis=1)


class ChargeTimeModel:
    """One source's joint forward → the √(k·charge) charge residual + the (T+t0) first-
    arrival-time residual, with FD Jacobians of BOTH w.r.t. the global params.

    ``forward(theta, ek, pk) -> (mean_charge, first_arrival_time)`` is evaluated at per-PMT
    factor 1 and per-PMT t0 0 (the per-PMT k and t0 are the two Schur blocks). The time
    residual is additive in t0 (∂/∂t0 = 1), the charge residual multiplicative in k.
    """

    def __init__(self, forward, eps=1e-8, fd_step=1e-3):
        self.forward = forward
        self.eps = eps
        self.fd_step = fd_step

        def _ct(theta, lk, ek, pk):
            c, t = forward(theta, ek, pk)
            return jnp.sqrt(jnp.exp(lk) * c + eps), t
        self._ct = jax.jit(_ct)

    def ct(self, theta, lk, ek, pk):
        return self._ct(theta, lk, ek, pk)

    def fd(self, theta, lk, ek, pk, h=None):
        """Return (Jc, Jt): FD Jacobians (n_sensors, n_params) of √(k·c) and of T, CRN."""
        h = self.fd_step if h is None else h
        theta = jnp.asarray(theta)
        mc0, t0 = self.ct(theta, lk, ek, pk)
        Jc, Jt = [], []
        for d in range(theta.shape[0]):
            mc1, t1 = self.ct(theta.at[d].add(h), lk, ek, pk)
            Jc.append(np.array((mc1 - mc0) / h))
            Jt.append(np.array((t1 - t0) / h))
        return np.stack(Jc, axis=1), np.stack(Jt, axis=1)


def fit_charge_time(sources, truth_charge, truth_time, theta0, n_sensors, *,
                    wc=None, wt=None, lk0=None, t00=None, steps=300, refresh=15,
                    ridge=0.02, mu=0.3, eps=1e-8, step_max=0.08, kstep_max=0.3,
                    t0step_max=2.0, w_time=1.0, fix=(), seed=0, nb_h=4):
    """Joint charge + first-arrival-time Gauss-Newton — the same recipe with a TIME residual.

    Adds the timing observable to the charge fit: the per-PMT QE factor ``k`` (multiplicative
    on charge) and the per-PMT time offset ``t0`` (additive on time) are TWO independent
    diagonal per-PMT Schur blocks (k touches only charge, t0 only time), each gauged to mean
    0 (``mean(log k)=0`` / ``mean(t0)=0``). The global block ``theta`` (optical + tts) is fed
    by BOTH residuals — so the timing term breaks charge-only degeneracies (the 13/30→30/30
    motivation). ``w_time`` weights the time residual vs charge.

    ``sources`` are :class:`ChargeTimeModel`. ``truth_time`` is the observed per-sensor first
    arrival; sensors with ``truth_time<=0`` are unlit and dropped from the time term.
    """
    S = len(sources)
    n_params = int(np.asarray(theta0).shape[0])
    Wc = np.ones(n_sensors) if wc is None else np.asarray(wc, float)
    Wt = [np.asarray(truth_time[i]) > 0 if wt is None else np.asarray(wt[i], float)
          for i in range(S)]
    Wt = [w.astype(float) for w in Wt]
    tc_sqrt = [jnp.sqrt(jnp.asarray(truth_charge[i]) + eps) for i in range(S)]
    tt = [jnp.asarray(truth_time[i]) for i in range(S)]
    lp = jnp.asarray(theta0)
    lkv = jnp.zeros(n_sensors) if lk0 is None else jnp.asarray(lk0)
    t0v = jnp.zeros(n_sensors) if t00 is None else jnp.asarray(t00)
    fix = list(fix)

    history = np.zeros((steps, n_params))
    Jc_c = Jt_c = Htk = Htu = Minv_k = Minv_u = Pinv = None

    for s in range(steps):
        kb = 1000 + 777 * seed + 13 * s
        rcA, mcA, rtA = [], [], []
        for i in range(S):
            mc, T = sources[i].ct(lp, lkv, *_keys(kb + 7000 * i))
            rcA.append(np.array(sg(mc - tc_sqrt[i])))
            mcA.append(np.array(sg(mc)))
            rtA.append(np.array(sg((T + t0v) - tt[i])))           # raw time residual (weight Wt)

        if s % refresh == 0:
            Jc_c, Jt_c = [], []
            for i in range(S):
                Jc_i = np.zeros((n_sensors, n_params)); Jt_i = np.zeros((n_sensors, n_params))
                for h in range(nb_h):
                    ek, pk = _keys(9_000_000 + 7 * s + 1000 * i + h)
                    jc, jt = sources[i].fd(lp, lkv, ek, pk)
                    Jc_i += jc; Jt_i += jt
                Jc_c.append(Jc_i / nb_h); Jt_c.append(Jt_i / nb_h)

            Htt = np.zeros((n_params, n_params))
            Htk = np.zeros((n_params, n_sensors)); Htu = np.zeros((n_params, n_sensors))
            Hkk = np.zeros(n_sensors) + 1e-12; Huu = np.zeros(n_sensors) + 1e-12
            for i in range(S):
                Jk = 0.5 * mcA[i]                                  # ∂√(k·c)/∂log k
                Htt += (Jc_c[i] * Wc[:, None]).T @ Jc_c[i]
                Htt += w_time * (Jt_c[i] * Wt[i][:, None]).T @ Jt_c[i]
                Htk += (Jc_c[i] * Wc[:, None]).T * Jk[None, :]
                Htu += w_time * (Jt_c[i] * Wt[i][:, None]).T       # ∂rt/∂t0 = 1
                Hkk += Wc * (Jk * Jk)
                Huu += w_time * Wt[i]                              # diagonal time block
            Htt /= S; Htk /= S; Htu /= S; Hkk /= S; Huu /= S
            Minv_k = make_constrained_schur(Hkk)
            Minv_u = make_constrained_schur(Huu)
            Pinv = ridge_inverse(Htt - Htk @ Minv_k(Htk.T) - Htu @ Minv_u(Htu.T),
                                 ridge=ridge, mu=mu)

        gt = np.zeros(n_params); gk = np.zeros(n_sensors); gu = np.zeros(n_sensors)
        for i in range(S):
            gt += Jc_c[i].T @ (Wc * rcA[i]) + w_time * Jt_c[i].T @ (Wt[i] * rtA[i])
            gk += Wc * (0.5 * mcA[i]) * rcA[i]
            gu += w_time * Wt[i] * rtA[i]
        gt /= S; gk /= S; gu /= S
        geff = gt - Htk @ Minv_k(gk) - Htu @ Minv_u(gu)
        for i in fix:
            geff[i] = 0.0
        dth = -(Pinv @ geff)
        for i in fix:
            dth[i] = 0.0
        dlk = -Minv_k(gk + Htk.T @ dth)
        dt0 = -Minv_u(gu + Htu.T @ dth)
        lp = lp + jnp.asarray(np.clip(dth, -step_max, step_max))
        lkv = lkv + jnp.asarray(np.clip(dlk, -kstep_max, kstep_max)); lkv = lkv - jnp.mean(lkv)
        t0v = t0v + jnp.asarray(np.clip(dt0, -t0step_max, t0step_max)); t0v = t0v - jnp.mean(t0v)
        history[s] = np.array(jnp.exp(lp))

    return dict(theta=np.array(jnp.exp(lp)), log_theta=np.array(lp),
                k=np.clip(np.array(jnp.exp(lkv)), 1e-6, None),
                t0=np.array(t0v), history=history)
