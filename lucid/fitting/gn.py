"""The damped Gauss-Newton loop shared by calibration and reconstruction.

:func:`lucid.fitting.calibrate.fit` and :func:`lucid.fitting.recon.fit_track` both run this loop.
:func:`lucid.fitting.schur_gn.fit_charge_time` does not: it carries two per-PMT blocks (a
multiplicative gain and an additive ``t0``) that the shared calibration problem cannot express.

The seam
--------
A problem supplies ``grad_metric_loss(theta, step, refresh) -> (g, H, loss)`` and nothing else
about how it got there. That is deliberately the *only* thing the two problems share:

* **calibration** is least squares — it forms a residual and a Jacobian, then assembles
  ``g = Jᵀr`` and ``H = JᵀJ``;
* **reconstruction** is a likelihood — it takes AD of a scalar and builds a Fisher metric
  separately, and has no residual vector at all.

An interface expressed in terms of residuals, or as ``(model, residual)``, could not serve both.

No key ever reaches this loop, and that is a correctness requirement rather than tidiness:
calibration's Jacobian stream *must* be independent of its residual's (sharing them correlates the
two), while reconstruction deliberately shares keys between its gradient and its metric. Only the
problem can hold both policies, so only the problem holds keys.

The loop applies the step and carries the iterate in float32 for both problems.
"""

__all__ = ['gauss_newton', 'gauss_newton_tx']


def gauss_newton(problem, theta0, steps, *, lam, mu, max_step=None, jitter=0.0,
                 lr=1.0, lr_final=None, scale=None,
                 refresh=1, refresh_final=None, refresh_switch=0.5,
                 readout='final', polyak=0, reject_nonfinite=False, fix=(), on_step=None):
    """Run the damped Gauss-Newton loop.

    Parameters
    ----------
    problem
        Supplies ``grad_metric_loss(theta, step, refresh) -> (g, H, loss)``. On a non-refresh step it may return a metric
        cached from the last refresh. For :class:`~lucid.fitting.recon.ReconModel` that means
        ``g`` is at the current iterate while ``H`` may be at an earlier one, which is intended:
        the metric is a preconditioner and cannot move the fixed point, so a stale one costs
        convergence rate, not accuracy.

        That reasoning does NOT extend to calibration, and the difference matters when tuning
        ``refresh``. :class:`~lucid.fitting.calib.CalibrationProblem` forms its gradient as
        ``J^T r`` from the SAME cached ``J``, so on a non-refresh step only ``r`` is at the
        current iterate and the search DIRECTION is stale too, not merely its preconditioner. At
        ``refresh=20`` that is 19 steps in 20. The fixed point still survives (``E[r] = 0`` at
        truth for any ``J``), so this is a convergence-path property and not a bias, but raising
        ``refresh`` costs calibration more than it costs reconstruction.
    lr, lr_final
        Step scale, annealed linearly if ``lr_final`` is given. ``lr > 1`` is meaningful when the
        metric systematically under-estimates the curvature, which is the reconstruction case.
    scale
        Coordinate preconditioner: the step is solved as ``S·H·S``, ``S·g`` and applied as
        ``S·du``. ``None`` genuinely SKIPS the multiply rather than multiplying by ones — a
        no-op scaling would still change the dtype the damping is built from, and "does nothing"
        should mean it.
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
    # This is the damped-Gauss-Newton configuration of the shared driver `minimize`, stepping with
    # `gauss_newton_tx`.
    from lucid.fitting.minimize import minimize as _minimize
    tx = gauss_newton_tx(lam, mu, jitter=jitter, lr=lr, lr_final=lr_final, steps=steps)
    return _minimize(
        problem, theta0, steps, tx,
        needs_metric=True, scale=scale, max_step=max_step,
        refresh=refresh, refresh_final=refresh_final, refresh_switch=refresh_switch,
        readout=readout, polyak=polyak, reject_nonfinite=reject_nonfinite,
        fix=fix, on_step=on_step)


def gauss_newton_tx(lam, mu, *, jitter=0.0, lr=1.0, lr_final=None, steps=1):
    """The step :func:`gauss_newton` runs, as an optax transformation for
    :func:`lucid.fitting.minimize.minimize`: the damped direction, the ``lr -> lr_final`` anneal
    over ``steps`` (read from the driver's iteration), and the descent sign.

    :func:`lucid.fitting.calibrate.calibrate` runs it by default.
    """
    from lucid.fitting.transforms import damped_gauss_newton, annealed_learning_rate
    return damped_gauss_newton(lam, mu, learning_rate=annealed_learning_rate(lr, lr_final, steps),
                               jitter=jitter)
