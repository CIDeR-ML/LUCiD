"""One loop, any optax optimizer — including the damped Gauss-Newton one.

:func:`lucid.fitting.gn.gauss_newton` is this loop configured with the damped Gauss-Newton
transformation. The loop does not know which rule it runs: the step is an
:class:`optax.GradientTransformation`, so the choice of optimizer is an argument to the driver
rather than a property of it::

    minimize(problem, theta0, steps, tx=damped_gauss_newton(lam=0.01, mu=0.1, learning_rate=4.0))
    minimize(problem, theta0, steps, tx=optax.adam(1e-3), needs_metric=False)

With the Gauss-Newton transformation it reproduces ``fit_track``'s reference loop, trajectory and
refused steps included, to the float32 solve (``tests/test_fit_track_matches_main_loop.py``).

What stays in the driver, and why
---------------------------------
Everything that is a property of the TRAJECTORY rather than of the step rule:

* ``max_step`` — a trust region in SCALED units, applied before ``S`` multiplies the step back
  into problem units, so with ``SCALE9[0] = 50`` a ``max_step`` of 3.0 is 150 MeV on energy,
  applied to whatever the transformation returns. ``optax.clip`` clips GRADIENTS, and putting the
  step clip inside a chain makes the chain's order load-bearing and invisible.
* ``fix`` — zeroed in the gradient BEFORE the transformation and in the step AFTER it. Both are
  needed: zeroing only the step gives a masked step, while zeroing the gradient first is what
  makes the remaining parameters fit CONDITIONAL on the frozen values through a coupled metric.
  ``optax.masked`` cannot express this — its masks are per pytree LEAF, ``theta`` is one leaf, and
  a masked-out leaf is passed through UNCHANGED rather than frozen.
* ``polyak`` readout — a property of a trajectory that wanders on a Monte-Carlo noise floor, not
  of the rule that produced it.
* the metric refresh cadence — the problem's policy, which the rule never sees.

What the transformation owns is the step size and any schedule. There is no ``lr`` here on
purpose: optax bakes the learning rate into the transformation, so a driver-side multiplier would
double-apply, and ``optax.adam(1e-3)`` under a driver ``lr=4.0`` would run at 4e-3 while looking
correct.

Two hazards this loop is written around
---------------------------------------
**Nothing jax-typed may reach** ``problem.accumulate``. ``theta_f64 + optax_update_f32`` returns a
jax float32 array, and since the iterate is reassigned each step the corruption is absorbing: one
step and a float64 numpy iterate is gone for the run. ``optax.apply_updates`` does not save you —
with ``jax_enable_x64`` off it casts to float32 too. Every update crosses back through
``np.asarray`` here, gated by
``tests/test_fitting_minimize.py::test_nothing_jax_typed_reaches_accumulate``.

**A rejected step must not poison the optimizer state.** With a stateful rule the transformation
has already folded the bad gradient into its moments by the time ``reject_nonfinite`` decides. The
state is therefore captured before the update and restored on rejection; optax states are
immutable pytrees, so this is a reference copy. Without it, rejection keeps the good iterate while
leaving the velocity that produces the next step contaminated — worse than not rejecting at all.
"""
import numpy as np

__all__ = ['minimize']


def minimize(problem, theta0, steps, tx, *, needs_metric=True, scale=None, max_step=None,
             refresh=1, refresh_final=None, refresh_switch=0.5, readout='final', polyak=0,
             reject_nonfinite=False, fix=(), on_step=None):
    """Run ``steps`` of ``tx`` on ``problem``, starting from ``theta0``.

    Parameters mirror :func:`lucid.fitting.gn.gauss_newton` except that ``lam``/``mu``/``jitter``/
    ``lr``/``lr_final`` are gone — they configure the Gauss-Newton transformation, so they belong
    to ``tx``.

    ``tx``
        Any ``optax.GradientTransformation``. If it needs curvature it must accept ``metric=`` in
        optax's extra arguments (see :func:`lucid.fitting.transforms.damped_gauss_newton`), and
        ``needs_metric`` must be True so the metric is built and passed.
    ``needs_metric``
        False for first-order rules. The metric is then neither passed nor requested — but note
        that a problem may still build one: ``grad_metric_loss`` implementations here cache with
        ``if refresh or self._H is None``, so the first call constructs it regardless. For
        ``CalibrationProblem`` that is not merely a wasted build: its gradient IS ``Jᵀr``, so
        there is no gradient without the Jacobian and a first-order rule saves nothing there.

    Returns ``dict(theta, history, gnorm, loss, n_steps)``, with ``history`` and ``gnorm``
    including the starting point, exactly as ``gauss_newton`` does.
    """
    if readout not in ('final', 'polyak', 'ming'):
        raise ValueError(f"readout must be 'final', 'polyak' or 'ming', got {readout!r}")

    theta = theta0
    P = int(np.asarray(theta0).shape[0])
    S = np.ones(P) if scale is None else np.asarray(scale, float)
    # `fix` is a boolean mask or a list of indices. A mask must not be cast straight to int:
    # [True, False, True] would become indices [1, 0, 1] and silently freeze the wrong
    # parameters. The two cannot collide: integer indices never carry dtype bool.
    fix = np.asarray(list(fix))
    fix = (np.flatnonzero(fix) if fix.dtype == bool else fix.astype(int))
    sw = int(refresh_switch * steps)
    since = None

    def _due(step):
        if since is None:
            return True
        r_it = refresh if (refresh_final is None or step < sw) else refresh_final
        return since >= r_it or (refresh_final is not None and step == sw)

    def _norm(gv):
        return float(np.linalg.norm(gv if scale is None else S * gv))

    # The step INDEX is part of the problem's random-number stream: CalibrationProblem draws its
    # forward at `forward_key0 + stride*step`. The indices used here — 0 for the initial call and
    # `step+1` for the lookahead — must match gauss_newton exactly, or every calibration number
    # moves for a reason no gate on the reconstruction side can see.
    g, H, loss = problem.grad_metric_loss(theta, 0, refresh=_due(0))
    since = 1
    state = tx.init(np.asarray(theta))
    rejected = []
    history = [np.asarray(theta)]
    gnorms = [_norm(np.asarray(g))]
    losses = np.zeros(steps)
    best = (gnorms[0], np.asarray(theta))

    for step in range(steps):
        g = np.asarray(g)
        H = None if H is None else np.asarray(H)
        losses[step] = float(loss) if loss is not None else np.nan

        Hs = None if H is None else (H if scale is None else S[:, None] * H * S[None, :])
        gs = g if scale is None else S * g
        if fix.size:
            gs = np.array(gs)
            gs[fix] = 0.0

        prev_state = state                      # optax states are immutable pytrees
        # `iteration` is the RAW loop index, which advances whether or not the step is accepted.
        # A transformation's own state counter is NOT the same thing: restoring the state on a
        # refused step (below) correctly discards momentum but would also rewind a schedule
        # counter, so the lr anneal is driven by this index to stay on fit_track's schedule.
        extra = ({'metric': Hs, 'iteration': step}
                 if (needs_metric and Hs is not None) else {})
        du, state = tx.update(gs, state, np.asarray(theta), **extra)

        # The jax -> numpy crossing (see the module docstring). `np.asarray` IS the guard: it
        # always returns a numpy array, including from a jax one.
        du = np.asarray(du)

        if max_step is not None:
            du = np.clip(du, -max_step, max_step)
        if fix.size:
            du = np.array(du)
            du[fix] = 0.0

        theta_new = problem.accumulate(theta, du if scale is None else S * du)

        nxt = None
        if step + 1 < steps or reject_nonfinite:
            due = _due(step + 1)
            nxt = problem.grad_metric_loss(theta_new, step + 1, refresh=due)
            since = 1 if due else since + 1

        if reject_nonfinite and nxt is not None and not (
                np.isfinite(np.asarray(theta_new)).all()
                and np.isfinite(np.asarray(nxt[0])).all()):
            # Recorded for diagnostics only; it changes no number.
            rejected.append(step)
            state = prev_state                  # do NOT keep the moments from a refused step
            if due and step + 1 < steps:
                # The look-ahead call refreshed the metric at the REFUSED point, and a problem
                # that caches its metric hands that one back on every later non-refresh step. A
                # non-finite one then refuses every step after it and the fit freezes. Rebuild it
                # at the kept iterate: the cadence and the step index are unchanged, and `g`
                # stays the kept one.
                H = problem.grad_metric_loss(theta, step + 1, refresh=True)[1]
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
        out = np.mean(np.stack(history[1:])[-polyak:], axis=0)
    elif readout == 'ming':
        out = best[1]
    else:
        out = np.asarray(theta)
    return dict(theta=out, history=np.stack(history), gnorm=np.array(gnorms),
                loss=losses, n_steps=steps, n_rejected=len(rejected),
                rejected_steps=tuple(rejected))
