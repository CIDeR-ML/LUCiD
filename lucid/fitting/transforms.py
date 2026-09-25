"""Gauss-Newton as an optax transformation, in JAX.

``lucid.fitting.gn`` configures the shared loop in :mod:`lucid.fitting.minimize`; this module is
the step that loop runs: the damped Gauss-Newton step as an
:class:`optax.GradientTransformationExtraArgs`. The metric arrives per call as ``metric=H`` in
optax's extra arguments, so the step composes with the rest of optax:

* ``optax.chain`` — clipping, weight decay, anything;
* ``optax.inject_hyperparams`` — schedule the Marquardt damping, which ``gn.gauss_newton`` cannot
  (``lam`` is a fixed scalar there);
* ``optax.MultiSteps`` — microbatched gradient accumulation. It jits internally, which is why the
  solve must be ``jnp`` (``np.linalg.solve`` on a traced array raises
  ``TracerArrayConversionError``). It is NOT a photon-budget lever for calibration:
  ``IsotropicSource`` draws directions from a Fibonacci lattice that ignores its key, so running
  the forward at ``N_PH/k`` changes the emission pattern itself and more draws do not average it
  out. The microbatched step is not the full-budget step;
* ``jax.jit`` / ``jax.vmap`` over the step.

Naming follows optax's own convention. :func:`scale_by_damped_gauss_newton` returns the Newton
DIRECTION, and the caller composes ``optax.scale(-lr)`` to turn it into a descent step, exactly as
``scale_by_adam`` is paired with a learning rate. :func:`damped_gauss_newton` is the convenience
pairing of the two.

Precision
---------
The solve is float32, and that is adequate: at ``cond(H) = 6800`` (the 19-parameter calibration
matrix) float32 moves the step direction by ~3e-7 relative, while Marquardt ``lam=0.01``
perturbs it by ~1e-2 by construction. Gated by ``tests/test_float32_is_adequate.py``.

This module does not carry the iterate; step application belongs to the problem (see ``gn.py``).
A float32 iterate in RAW coordinates freezes a parameter at ~1000 (float32 spacing 6.1e-5), so
it is carried in scaled coordinates (:class:`lucid.fitting.scaled.ScaledProblem`).
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

__all__ = ['damped_matrix', 'scale_by_damped_gauss_newton', 'damped_gauss_newton',
           'scale_by_driver_schedule', 'annealed_learning_rate']


def damped_matrix(H, *, lam, mu, jitter=0.0, rel_cutoff=1e-12):
    """``A = H + lam·diag(H) + mu·median(diag H)·I + jitter·I``.

    * the Marquardt diagonal is CLIPPED at zero — a negative curvature entry would otherwise
      *reduce* the damping in the direction that most needs it;
    * the Levenberg median is FILTERED to the entries carrying curvature, above a RELATIVE cutoff.
      A floor is not invariant to adding an unconstrained parameter: each flat direction drags the
      median down, and once more than half the diagonal sits at the floor the median IS the floor,
      so the isotropic damping collapses in exactly the regime it exists for. An absolute cutoff is
      also not scale-invariant, and these fits run in scaled coordinates where the diagonal carries
      different units per parameter.

    Both are asserted in ``tests/test_fitting_transforms.py``. If every diagonal entry is below the
    cutoff (all directions flat), ``A`` is NaN — see the comment on ``base``.
    """
    n = H.shape[0]
    dg = jnp.clip(jnp.diag(H), 0, None)
    cutoff = rel_cutoff * jnp.max(dg)
    # A traced function cannot raise, so the all-flat case returns NaN: `nanmedian` of an all-NaN
    # vector is NaN, which propagates into `A` and the step. Deliberate: an absolute floor would
    # instead silently build a numerically singular `A`.
    base = jnp.nanmedian(jnp.where(dg > cutoff, dg, jnp.nan))
    A = H + lam * jnp.diag(dg) + mu * base * jnp.eye(n, dtype=H.dtype)
    return A + jitter * jnp.eye(n, dtype=H.dtype) if jitter else A


def scale_by_damped_gauss_newton(lam, mu, jitter=0.0, rel_cutoff=1e-12):
    """The damped Newton DIRECTION ``A⁻¹g``, as an optax transformation.

    The metric is supplied per call as ``metric=`` in optax's extra arguments, not held in the
    optimiser state, because it belongs to the problem and its refresh cadence is the caller's
    policy — the same reason ``gn.py`` keeps keys out of the loop.

    Returns the direction, NOT the step: pair it with ``optax.scale(-lr)``. This mirrors
    ``optax.scale_by_adam``, which likewise returns a direction that a learning rate then negates
    and scales. :func:`damped_gauss_newton` is that pairing.

    Stateless by construction. A Gauss-Newton step depends only on the current gradient and
    metric, so there is no moment buffer to corrupt — which is also why it needs no equivalent of
    the state save/restore that a momentum method requires when the driver rejects a step.
    """
    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None, *, metric=None, **extra):
        del params, extra
        if metric is None:
            raise ValueError(
                'scale_by_damped_gauss_newton needs the metric: call '
                'opt.update(g, state, params, metric=H). It is passed per step rather than '
                'held in state because its refresh cadence is the problem\'s policy.')
        A = damped_matrix(jnp.asarray(metric), lam=lam, mu=mu,
                          jitter=jitter, rel_cutoff=rel_cutoff)
        return jnp.linalg.solve(A, jnp.asarray(updates)), state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)




def damped_gauss_newton(lam, mu, learning_rate=1.0, jitter=0.0, rel_cutoff=1e-12):
    """Damped Gauss-Newton as a ready-to-use optax optimiser.

    ``learning_rate`` accepts an optax schedule; to reproduce ``gn.py``'s ``lr``/``lr_final``
    anneal use :func:`annealed_learning_rate` (``transition_steps`` must be ``steps - 1``, not
    ``steps``). ``lam`` and ``mu`` are scalars here; to schedule the DAMPING as well, wrap with ``optax.inject_hyperparams`` — a capability ``gn.gauss_newton`` does not have::

        tx = optax.inject_hyperparams(damped_gauss_newton)(
            lam=optax.linear_schedule(0.1, 0.001, 500), mu=0.1)

    The update returned is the thing ADDED to the parameters, matching optax throughout and
    matching ``gn.py``'s ``du``, so no sign flip is needed anywhere.
    """
    # A SCHEDULE reads the driver's iteration rather than optax's own counter. `minimize` restores
    # the optimiser state on a refused step, which would rewind a counted schedule and spend an
    # iteration's learning rate twice; see `scale_by_driver_schedule`. A constant needs no count.
    if callable(learning_rate):
        return optax.chain(
            scale_by_damped_gauss_newton(lam, mu, jitter=jitter, rel_cutoff=rel_cutoff),
            scale_by_driver_schedule(learning_rate),
            optax.scale(-1.0),
        )
    return optax.chain(
        scale_by_damped_gauss_newton(lam, mu, jitter=jitter, rel_cutoff=rel_cutoff),
        optax.scale_by_learning_rate(learning_rate),
    )


def scale_by_driver_schedule(schedule):
    """Scale by ``schedule(iteration)``, where the iteration comes from the DRIVER, not a counter.

    ``optax.scale_by_schedule`` keeps its own step count in state. That is right for a momentum
    buffer and wrong for a learning-rate anneal the moment a driver can REJECT a step: on a
    rejection :func:`lucid.fitting.minimize.minimize` restores the previous optimiser state, which
    rewinds the counter, so the anneal repeats an iteration it has already spent.

    So the index is READ rather than counted. ``minimize`` passes ``iteration=step`` in optax's
    extra arguments, and this transformation is stateless as a result — there is nothing to
    restore, and nothing that can disagree with the loop about which iteration it is on.

    Falls back to an internal counter when no ``iteration`` is supplied, so it still composes with
    a plain optax driver that knows nothing about this convention.
    """
    def init_fn(params):
        del params
        return {'count': np.zeros((), dtype=np.int64)}

    def update_fn(updates, state, params=None, *, iteration=None, **extra):
        del params, extra
        step = int(state['count']) if iteration is None else int(iteration)
        s = float(schedule(step))
        return jax.tree_util.tree_map(lambda u: u * s, updates), {'count': state['count'] + 1}

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def annealed_learning_rate(lr, lr_final, steps):
    """The ``lr -> lr_final`` linear anneal, as a schedule over the driver's iteration.

    ``transition_steps`` is ``steps - 1``, NOT ``steps``. ``optax.linear_schedule`` divides by
    ``transition_steps`` while the loop this reproduces divides by ``steps - 1``, so the obvious
    ``linear_schedule(4.0, 1.5, steps)`` ends at 1.5167 rather than 1.5.

    The index is CLAMPED to ``steps - 1``. Unclamped, and with ``lr_final`` set, the ratio grows
    without bound and drives the learning rate NEGATIVE within a few iterations past the end,
    turning descent into ascent silently. ``optax.linear_schedule`` already clamps at the end of
    its transition, so this is stated rather than implemented — but it is the reason the schedule
    is built here instead of at each call site.
    """
    if lr_final is None:
        return lambda _step: lr
    return optax.linear_schedule(init_value=lr, end_value=lr_final,
                                 transition_steps=max(1, steps - 1))
