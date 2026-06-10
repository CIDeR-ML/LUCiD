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
        gauge_k=True, polyak=0, bake_k=False):
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
    polyak : int
        If >0, return the AVERAGE of the last ``polyak`` iterates (theta and k) instead of
        the final one (Polyak/Ruppert iterate-averaging). Stabilises noisy/shot-noise fits;
        default 0 = off (return the final iterate, byte-identical to before).
    bake_k : bool
        If True, replace the free per-PMT Schur block with the CLOSED-FORM pooled estimate
        ``k = ΣQ / ΣM(theta)`` (re-baked each step from the data + current model), and run
        Gauss-Newton on the GLOBALS ONLY. This is the shot-noise-robust per-PMT recipe (the
        free Schur-k overfits a single noisy draw); default False = free Schur-k as before.

    Returns
    -------
    dict with keys: ``theta`` (real-space global params), ``log_theta``,
    ``k`` (per-PMT factor), ``history`` (per-step theta trajectory).
    """
    S = len(sources)
    n_params = int(np.asarray(theta0).shape[0])
    W = np.ones(n_sensors) if weights is None else np.asarray(weights, float)
    t_sqrt = [jnp.sqrt(jnp.asarray(truth_list[i]) + eps) for i in range(S)]
    t_data = [np.asarray(truth_list[i], float) for i in range(S)]
    lp = jnp.asarray(theta0)
    lkv = jnp.zeros(n_sensors) if lk0 is None else jnp.asarray(lk0)
    fix = list(fix)
    _zns = jnp.zeros(n_sensors)

    history = np.zeros((steps, n_params))
    lp_acc = np.zeros(n_params); lk_acc = np.zeros(n_sensors); n_acc = 0
    Jcache = Htk = Minv = Pinv = None

    for s in range(steps):
        kb = 1000 + 777 * seed + 13 * s

        if bake_k:
            # closed-form pooled k = ΣQ / ΣM(theta) (M = model mean charge at k=1), gauged.
            Qsum = np.zeros(n_sensors); Msum = np.zeros(n_sensors) + 1e-12
            for i in range(S):
                Mi = np.array(sg(sources[i].m(lp, _zns, *_keys(kb + 7000 * i)))) ** 2
                Qsum += t_data[i]; Msum += Mi
            lkv = jnp.asarray(np.log(np.clip(Qsum / Msum, 1e-6, None)))
            if gauge_k:
                lkv = lkv - jnp.mean(lkv)

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
            if bake_k:                                   # k is fixed → no Schur reduction
                Minv = None
                Pinv = ridge_inverse(Htt, ridge=ridge, mu=mu)
            else:
                Minv = make_constrained_schur(Hkk) if gauge_k else (lambda X: (1.0 / Hkk) * X if np.ndim(X) == 1 else (1.0 / Hkk)[:, None] * X)
                Pinv = ridge_inverse(Htt - Htk @ Minv(Htk.T), ridge=ridge, mu=mu)

        gt = np.zeros(n_params); gk = np.zeros(n_sensors)
        for i in range(S):
            gt += Jcache[i].T @ (W * rA[i])
            gk += W * (0.5 * mA[i]) * rA[i]
        gt /= S; gk /= S
        geff = gt if bake_k else gt - Htk @ Minv(gk)
        for i in fix:
            geff[i] = 0.0
        dth = -(Pinv @ geff)
        for i in fix:
            dth[i] = 0.0
        lp = lp + jnp.asarray(np.clip(dth, -step_max, step_max))
        if not bake_k:                                   # free Schur-k step
            dlk = -Minv(gk + Htk.T @ dth)
            lkv = lkv + jnp.asarray(np.clip(dlk, -kstep_max, kstep_max))
            if gauge_k:
                lkv = lkv - jnp.mean(lkv)
        history[s] = np.array(jnp.exp(lp))
        if polyak and s >= steps - polyak:               # accumulate the iterate tail
            lp_acc += np.array(lp); lk_acc += np.array(lkv); n_acc += 1

    if polyak and n_acc:
        lp_out = lp_acc / n_acc; lk_out = lk_acc / n_acc
    else:
        lp_out = np.array(lp); lk_out = np.array(lkv)
    return dict(theta=np.exp(lp_out), log_theta=lp_out,
                k=np.clip(np.exp(lk_out), 1e-6, None), history=history)


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
