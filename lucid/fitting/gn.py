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
a loop: one definition shared by that figure's two halves. It is a numpy transcription of
:func:`lucid.fitting.transforms.damped_matrix` rather than a call to it, for a MEASURED reason —
importing anything from ``lucid.fitting`` runs this package's ``__init__``, which costs 9 seconds
and pulls jax into a module that otherwise only reads ``.npz`` files and draws. The conventions are
identical and ``tests/test_paper_damping.py`` holds them in step.

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

__all__ = ['gauss_newton']


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
        cached from the last refresh. For :class:`~lucid.fitting.recon.ReconModel` that means
        ``g`` is at the current iterate while ``H`` may be at an earlier one, which is intended:
        the metric is a preconditioner and cannot move the fixed point, so a stale one costs
        convergence rate, not accuracy.

        That reasoning does NOT extend to calibration, and the difference matters when tuning
        ``refresh``. :class:`~lucid.fitting.calib.CalibrationProblem` forms its gradient as
        ``J^T r`` from the SAME cached ``J``, so on a non-refresh step only ``r`` is at the
        current iterate and the search DIRECTION is stale too, not merely its preconditioner. At
        the published ``REFRESH=20`` that is 19 steps in 20. The fixed point still survives
        (``E[r] = 0`` at truth for any ``J``), so this is a convergence-path property and not a
        bias -- but raising ``refresh`` is not free in the way this paragraph would otherwise
        suggest.
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
    # ONE loop AND ONE STEP. This function is the damped-Gauss-Newton CONFIGURATION of the shared
    # driver, and the step it configures is the optax transformation in `lucid.fitting.transforms`
    # -- there is no second, numpy implementation of the damped solve any more.
    #
    # There was, and deleting it is the point. `exact_damped_gauss_newton` re-did `damped_matrix`
    # and `np.linalg.solve` in float64 so the delegation could be bit-exact against the loop it
    # replaced. That bought pins, and cost a duplicated convention: the Levenberg-base fix earlier
    # in this work had to be written twice and gated by a test whose only job was keeping the two
    # copies in step. A test is how you verify an implementation, not a reason to keep one alive.
    #
    # MEASURED before switching, on the delegation test's own problem: without the anneal the two
    # steps agree to 3.4e-09; with the published `lr 4->1.5` over 150 steps they differ by 1.4e-03
    # in units of SCALE9 -- 0.07 MeV of energy and 0.3 mm of position, against a ~15 cm vertex
    # resolution and a Monte-Carlo loss whose own run-to-run spread is far larger. The JAX arm
    # reached the LOWER final loss (3.8e-11 against 6.3e-10), so this is not a precision
    # concession.
    #
    # The anneal is `scale_by_driver_schedule`, which reads the driver's iteration instead of
    # counting its own steps -- necessary because `minimize` restores optimiser state on a refused
    # step, which would rewind a stateful schedule and repeat an iteration.
    from lucid.fitting.minimize import minimize as _minimize
    from lucid.fitting.transforms import (scale_by_damped_gauss_newton, scale_by_driver_schedule,
                                          annealed_learning_rate)
    import optax as _optax
    tx = _optax.chain(
        scale_by_damped_gauss_newton(lam, mu, jitter=jitter),
        scale_by_driver_schedule(annealed_learning_rate(lr, lr_final, steps)),
        _optax.scale(-1.0),                      # direction -> descent step
    )
    return _minimize(
        problem, theta0, steps, tx,
        needs_metric=True, scale=scale, max_step=max_step,
        refresh=refresh, refresh_final=refresh_final, refresh_switch=refresh_switch,
        readout=readout, polyak=polyak, reject_nonfinite=reject_nonfinite,
        fix=fix, on_step=on_step)


