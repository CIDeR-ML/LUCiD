"""The damped Gauss-Newton loop, written once.

Four copies of this loop existed in the tree, and they had already drifted apart in ways nobody
intended: three different eigen-floor policies for the same damped inverse, and a default damping
in the library that was the value the campaign measured as harmful. All four are gone.

===================================================  ==========================================
was                                                  now
===================================================  ==========================================
``lucid/fitting/gauss_newton.py`` (the ``fit`` loop)  removed; ``lucid.fitting.calibrate.fit``
``lucid/fitting/recon.py`` (``fit_track``)            delegates here, pinned bit-exactly
``analysis/paper/utils/`` (the campaign engine)      superseded by ``utils/calib_run.py``,
                                                      which delegates here; gated bit-exactly
``analysis/paper/utils/pipeline.py`` (projected)      the projector became
                                                      ``recon.ProjectedReconProblem`` and the
                                                      loop is this one; gated at rtol=0,
                                                      atol=0 against a transcription of the
                                                      original
===================================================  ==========================================

Only ``fit_track`` and the projected fit are pinned bit-exactly against pre-delegation references;
``lucid.fitting.calibrate.fit`` deliberately returns different numbers from the loop it replaced,
because its ESTIMATOR changed at the same time.

One damped Gauss-Newton loop remains anywhere in the tree, and it is not a copy of this one:
:func:`lucid.fitting.schur_gn.fit_charge_time` carries TWO per-PMT blocks — a multiplicative gain
and an additive ``t0`` — which the shared calibration problem cannot express, so it is deferred
rather than folded onto a residual that could not hold it.

``analysis/paper/utils/damping.py`` holds the 2-D loss-geometry figure's damped SOLVE, which is not
a loop: one definition shared by that figure's two halves. It declines to call
:func:`damped_matrix` for a documented and measured reason.

The seam
--------
A problem supplies ``grad_metric_loss(theta, step, refresh) -> (g, H, loss)`` and nothing else
about how it got there. That is deliberately the *only* thing the two problems share:

* **calibration** is least squares — it forms a residual and a Jacobian, then assembles
  ``g = Jᵀr`` and ``H = JᵀJ``;
* **reconstruction** is a likelihood — it takes AD of a scalar and builds a Fisher metric
  separately, and has no residual vector at all.

An interface expressed in terms of residuals could not serve both. One expressed as
``(model, residual)`` could not either.

No key ever reaches this loop, and that is a correctness requirement rather than tidiness:
calibration's Jacobian stream *must* be independent of its residual's (sharing them was measured
at 137σ of covariance), while reconstruction deliberately shares keys between its gradient and its
metric. Only the problem can hold both policies, so only the problem holds keys.

Step application belongs to the problem
---------------------------------------
``jax_enable_x64`` is never enabled anywhere in this repo. Calibration therefore accumulates its
iterate in **float32** and reconstruction in **float64** numpy. A shared ``theta += scale*du``
would have to pick one and would silently change the other — and this is not a rounding nit: a
4.4e-07 perturbation at step 0 has been measured growing to 1.6e-02 by step 4 on the calibration
problem. So ``Problem.accumulate`` owns it.
"""
import numpy as np

__all__ = ['damped_matrix', 'gauss_newton']


def damped_matrix(H, *, lam, mu, jitter=0.0, median_floor=1e-12):
    """``A = H + lam·diag(H) + mu·median(diag H)·I + jitter·I``.

    Two damping terms with different jobs, and the distinction matters:

    * **Marquardt** ``lam·diag(H)`` scales with each parameter's own curvature, so it damps
      without freezing the weakly-determined directions;
    * **Levenberg** ``mu·median(diag H)·I`` is isotropic. It costs the soft directions, but it is
      the only guarantee of a non-singular solve: for PSD ``H`` it shifts every eigenvalue by
      exactly ``mu·base``, so ``lambda_min >= mu·base > 0``.

    Because of that shift, an eigen-floor on top is provably inert whenever ``mu > 0`` — and was
    measured never to engage: 0 of 19 directions, margin 3.7-15x. It is therefore not offered
    here. A *relative* eigen-floor is also the recorded cause of an earlier defect in this
    project, where it clipped a low-curvature direction and inflated a quoted uncertainty.

    One convention, chosen rather than parameterised. The call sites this replaces disagreed in
    two details that are invisible until you diff them:

    ===================  =====================  ==============================================
    was                  Marquardt diagonal     median of diag taken over
    ===================  =====================  ==============================================
    calibration          clipped to >= 0        entries strictly > 0        (filtered)
    reconstruction       not clipped            all entries, floored at 1e-12  (clipped)
    ===================  =====================  ==============================================

    Both differences are taken the safer way here: the Marquardt diagonal IS clipped, because a
    negative curvature entry would otherwise *reduce* the damping in that direction; and the
    Levenberg median is FLOORED rather than filtered, so ``base > 0`` holds even when every
    diagonal entry is zero, where filtering would fall back to an arbitrary 1.0.

    The two forms coincide whenever every diagonal entry exceeds ``median_floor``, which is the
    case for both callers today — so unifying them changes nothing measurable, and the pins on
    both sides say so. The point is that the convention is now stated in one place instead of
    differing silently across files.
    """
    n = H.shape[0]
    dg = np.clip(np.diag(H), 0, None)
    base = np.median(np.clip(np.diag(H), median_floor, None))
    A = H + lam * np.diag(dg) + mu * base * np.eye(n)
    if jitter:
        A = A + jitter * np.eye(n)
    return A


def gauss_newton(problem, theta0, steps, *, lam, mu, max_step=None, jitter=0.0,
                 lr=1.0, lr_final=None, scale=None,
                 refresh=1, refresh_final=None, refresh_switch=0.5,
                 readout='final', polyak=0, reject_nonfinite=False, fix=(), on_step=None):
    """Run the damped Gauss-Newton loop.

    Parameters
    ----------
    problem
        Supplies ``grad_metric_loss(theta, step, refresh) -> (g, H, loss)`` and
        ``accumulate(theta, dtheta) -> theta``. On a non-refresh step it may return a metric
        cached from the last refresh — so ``g`` is at the current iterate while ``H`` may be at
        an earlier one. That is intended: the metric is a preconditioner and cannot move the
        fixed point, so a stale one costs convergence rate, not accuracy.
    lr, lr_final
        Step scale, annealed linearly if ``lr_final`` is given. ``lr > 1`` is meaningful when the
        metric systematically under-estimates the curvature, which is the reconstruction case.
    scale
        Coordinate preconditioner: the step is solved as ``S·H·S``, ``S·g`` and applied as
        ``S·du``. ``None`` genuinely SKIPS the multiply — multiplying a float32 metric by a
        float64 ones-vector promotes it, and the damping is then built from a float64 diagonal.
    max_step
        Per-component clip on the scaled step. Units are per-problem: for a track fit ``3.0`` is
        150 MeV on the energy component; for calibration ``0.5`` is +65% in log space.
    refresh, refresh_final, refresh_switch
        Metric cadence. Refresh every ``refresh`` steps, then every ``refresh_final`` after
        ``refresh_switch·steps`` (with a forced refresh at the switch). A fresh metric earns its
        cost in the late precision phase, so spending it only there recovers most of the benefit.
        ``refresh_final=None`` gives a constant cadence, which reduces exactly to
        ``step % refresh == 0``.
    readout
        ``'final'`` last iterate · ``'polyak'`` mean of the last ``polyak`` · ``'ming'`` the
        iterate with the smallest ``‖S·g‖``. The trajectory does not settle — it wanders on the
        Monte-Carlo noise floor — so a single iterate is a draw from a stationary distribution
        rather than an estimate, and ``'polyak'`` is usually the honest choice.
    fix
        Indices of ``theta`` held fixed: their gradient and their step are zeroed, so the solve
        still sees the full coupled metric but the frozen directions cannot move. This is not the
        same as dropping them from the problem — the remaining parameters are then fitted
        CONDITIONAL on the frozen values, which is the point when one direction is known
        independently or too weakly determined to leave free.
    reject_nonfinite
        Evaluate the gradient at the proposed iterate and refuse the step if either is
        non-finite. Large early steps can overshoot into a degenerate region where the next
        gradient blows up; without this, one bad step poisons an averaged readout into NaN.
        Costs nothing when clean — the evaluation is the one the next step would have made anyway.

    Returns ``dict(theta, history, gnorm, loss, n_steps)``. ``history`` and ``gnorm`` INCLUDE the
    starting point, so they have ``steps + 1`` entries; ``loss`` has ``steps``.
    """
    if readout not in ('final', 'polyak', 'ming'):
        raise ValueError(f"readout must be 'final', 'polyak' or 'ming', got {readout!r}")
    theta = theta0
    P = int(np.asarray(theta0).shape[0])
    S = np.ones(P) if scale is None else np.asarray(scale, float)
    fix = np.asarray(list(fix), dtype=int)
    sw = int(refresh_switch * steps)
    since = None                       # None => the metric has never been built

    def _due(step):
        if since is None:
            return True
        r_it = refresh if (refresh_final is None or step < sw) else refresh_final
        return since >= r_it or (refresh_final is not None and step == sw)

    def _norm(gv):
        return float(np.linalg.norm(gv if scale is None else S * gv))

    g, H, loss = problem.grad_metric_loss(theta, 0, refresh=_due(0))
    since = 1
    history = [np.asarray(theta)]
    gnorms = [_norm(np.asarray(g))]
    losses = np.zeros(steps)
    # Seeded from the START's own gradient norm, not from inf: gnorms[0] is recorded and is a
    # legitimate candidate, so seeding at inf silently excluded the one iterate the caller
    # supplied. On a problem whose gradient grows monotonically, 'ming' returned the LAST
    # iterate while argmin(gnorm) was 0.
    best = (gnorms[0], np.asarray(theta))

    for step in range(steps):
        # dtype is preserved deliberately: the metric arrives as float32 and the damping is built
        # from its diagonal, so promoting first shifts median(diag H) in the last bit. The
        # amplification is the documented property of this stack (see the float32 note below); the
        # specific step-by-step figures previously quoted here were not reproducible and are gone.
        g = np.asarray(g)
        H = np.asarray(H)
        losses[step] = float(loss) if loss is not None else np.nan

        Hs = H if scale is None else S[:, None] * H * S[None, :]
        gs = g if scale is None else S * g
        if fix.size:
            gs = np.array(gs); gs[fix] = 0.0
        A = damped_matrix(Hs, lam=lam, mu=mu, jitter=jitter)
        lr_it = lr if lr_final is None else lr + (lr_final - lr) * (step / max(1, steps - 1))
        du = -lr_it * np.linalg.solve(A, gs)
        if max_step is not None:
            du = np.clip(du, -max_step, max_step)
        if fix.size:
            du[fix] = 0.0

        theta_new = problem.accumulate(theta, du if scale is None else S * du)

        nxt = None
        if step + 1 < steps or reject_nonfinite:
            due = _due(step + 1)
            nxt = problem.grad_metric_loss(theta_new, step + 1, refresh=due)
            since = 1 if due else since + 1

        if reject_nonfinite and nxt is not None and not (
                np.isfinite(np.asarray(theta_new)).all() and np.isfinite(np.asarray(nxt[0])).all()):
            pass                                   # keep theta, g, H, loss — the step is refused
        else:
            theta = theta_new
            if nxt is not None:
                g, H, loss = nxt

        gn = _norm(np.asarray(g))
        if gn < best[0]:
            best = (gn, np.asarray(theta))
        history.append(np.asarray(theta))
        gnorms.append(gn)
        if on_step is not None:
            on_step(step, theta, g, H, loss)

    if readout == 'polyak' and polyak:
        # history[1:] first: `history` includes the STARTING point, so a bare [-polyak:]
        # averages the un-stepped start into the answer whenever polyak > steps. That is
        # reachable — a shortened run (--steps 30 against the published polyak of 50) would
        # silently report a number pulled toward its own perturbed start. Identical to
        # [-polyak:] for polyak <= steps, which is every pinned configuration.
        out = np.mean(np.stack(history[1:])[-polyak:], axis=0)
    elif readout == 'ming':
        out = best[1]
    else:
        out = np.asarray(theta)
    return dict(theta=out, history=np.stack(history), gnorm=np.array(gnorms),
                loss=losses, n_steps=steps)
