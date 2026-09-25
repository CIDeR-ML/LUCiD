"""Gauss-Newton as an optax transformation, in JAX.

``lucid.fitting.gn`` owns the numpy loop that produced every published number, and it stays the
reference. This module is the other half of the same idea: the damped Gauss-Newton STEP expressed
as an :class:`optax.GradientTransformationExtraArgs`, so that curvature-aware optimisation stops
being a special case the framework works around and becomes a transformation that composes with
everything else optax offers.

Why a transformation rather than another loop
---------------------------------------------
optax's ``update(updates, state, params, **extra_args)`` carries arbitrary extra arguments, which
is exactly the hook a second-order method needs: the metric arrives as ``metric=H`` alongside the
gradient. That one signature buys, for free and without any of it being written here:

* ``optax.chain`` — compose with clipping, weight decay, anything;
* ``optax.inject_hyperparams`` — schedule the MARQUARDT DAMPING, which the numpy loop cannot do
  (``lam`` is a fixed scalar there);
* ``optax.MultiSteps`` — microbatched gradient accumulation. This is the one that does NOT work
  with a numpy implementation: ``MultiSteps`` jits internally, and ``np.linalg.solve`` on a traced
  array raises ``TracerArrayConversionError``. Being in JAX is what unlocks it.

  **It is not, however, a photon-budget lever for the calibration problem, and an earlier version
  of this docstring said it was.** Microbatching there means running the forward at ``N_PH/k``,
  and ``IsotropicSource`` draws its directions from a Fibonacci lattice that IGNORES its key
  (``lucid/sources/calibration_sources.py:264``) — so ``N_PH`` selects the emission pattern
  itself rather than a random realisation. Measured when this was written: total charge is
  conserved to 2e-4, but the
  per-sensor pattern moves 4-10x its own noise floor and MORE DRAWS DO NOT HELP, because every
  draw reuses the same lattice. Peak memory does fall as ~1/k to a fixed floor; the step it
  produces is not the full-budget step. The lever is real for a first-order rule on a
  key-respecting source, and absent here;
* ``jax.jit`` / ``jax.vmap`` over the step.

Naming follows optax's own convention. :func:`scale_by_damped_gauss_newton` returns the Newton
DIRECTION, and the caller composes ``optax.scale(-lr)`` to turn it into a descent step, exactly as
``scale_by_adam`` is paired with a learning rate. :func:`damped_gauss_newton` is the convenience
pairing of the two.

Precision
---------
This computes in float32, and that is ADEQUATE rather than a compromise. There is no float64 path
in the library to fall back to — the numpy one was deleted, not disabled — and nothing here is
waiting for ``jax_enable_x64`` to be turned on.

Measured, and gated by ``tests/test_float32_is_adequate.py``: at ``cond(H) = 6800``, the published
19-parameter calibration conditioning, the step direction moves by 2.8e-07 relative — against a
Marquardt ``lam=0.01`` that perturbs the same direction by ~1e-2 BY CONSTRUCTION. The damping the
method deliberately adds is five orders larger than the arithmetic's error, so a float64 solve
would be computing a number the algorithm then throws away. The same measurement found the
Levenberg median identical on both sides of a float32/float64 boundary.

Two related claims, since precision arguments in this project have twice been made against the
wrong quantity. A float32 ITERATE in RAW coordinates does fail — energy at ~1000 MeV has a float32
spacing of 6.1e-5, so small steps vanish — but in SCALED coordinates it matches float64 exactly,
which is what :class:`lucid.fitting.scaled.ScaledProblem` is for; that is an argument for scaling,
not for double precision. And the "4.4e-07 growing to 1.6e-02 by step 4" figure that once
justified a float64 iterate is a measurement of GPU NON-DETERMINISM at fixed precision — two runs,
same seed, same dtype — which float64 cannot remove, because the gradient still comes from the GPU.

The gate matters more than the number. This claim spent its life as a citation to a script that
asserted nothing and was never run in CI, so it could have stopped being true without anything
noticing. It was re-measured when the test was written and had not drifted.

What this does NOT do is carry the iterate. Step application still belongs to the problem, for the
reason ``gn.py`` gives, and the separate finding that a float32 iterate in RAW coordinates silently
freezes a parameter carried at ~1000 (float32 spacing 6.1e-5) while the same iterate in SCALED
coordinates matches float64 exactly.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

__all__ = ['damped_matrix', 'scale_by_damped_gauss_newton', 'damped_gauss_newton',
           'scale_by_driver_schedule', 'annealed_learning_rate']


def damped_matrix(H, *, lam, mu, jitter=0.0, rel_cutoff=1e-12):
    """``A = H + lam·diag(H) + mu·median(diag H)·I + jitter·I``.

    THE damped matrix. This was once described as a mirror of a numpy twin in ``lucid.fitting.gn``,
    kept in step by a test; the twin is deleted and what were shared conventions are now simply the
    conventions:

    * the Marquardt diagonal is CLIPPED at zero — a negative curvature entry would otherwise
      *reduce* the damping in the direction that most needs it;
    * the Levenberg median is FILTERED to the entries carrying curvature, above a RELATIVE cutoff.
      A floor is not invariant to adding an unconstrained parameter: each flat direction drags the
      median down, and once more than half the diagonal sits at the floor the median IS the floor,
      so the isotropic damping collapses in exactly the regime it exists for. An absolute cutoff is
      also not scale-invariant, and these fits run in scaled coordinates where the diagonal carries
      different units per parameter.

    Both are asserted directly in ``tests/test_fitting_transforms.py`` — on this function, not by
    agreement with another one. A comparison between two implementations can only ever say they
    agree; it can never say either is right.

    The all-flat case cannot raise here because the function is traced, so it returns NaN instead —
    see the comment on ``base``. That is the same event reported the only way this side can report
    it, and it is not a silent fallback.
    """
    n = H.shape[0]
    dg = jnp.clip(jnp.diag(H), 0, None)
    cutoff = rel_cutoff * jnp.max(dg)
    # The numpy version RAISES when nothing survives the cutoff. A traced function cannot, so the
    # all-flat case arrives as NaN instead: `nanmedian` of an all-NaN vector is NaN, which
    # propagates into `A` and then into the step. That is the same event reported the only way
    # this side can report it — loudly and without a fabricated `base`. It is NOT a silent
    # fallback, which is what the previous floor gave (a numerically singular `A` built from
    # 1e-12).
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

    ``learning_rate`` accepts an optax schedule, so the step size can anneal as ``gn.py``'s
    ``lr``/``lr_final`` does -- but the transition length must be ``steps - 1``, NOT ``steps``.
    ``optax.linear_schedule`` divides by ``transition_steps`` while ``gn.py`` divides by
    ``steps-1``, so the obvious ``optax.linear_schedule(4.0, 1.5, steps)`` ends at 1.5167 rather
    than 1.5 -- a 1.1% error on the final learning rate, four orders above the float32 agreement
    this module otherwise argues about. With ``transition_steps=steps-1`` the two agree to
    2.3e-07, which is the float32 figure quoted above. ``lam`` and ``mu`` are scalars here; to schedule the DAMPING as well,
    wrap with ``optax.inject_hyperparams`` — a capability the numpy loop does not have::

        tx = optax.inject_hyperparams(damped_gauss_newton)(
            lam=optax.linear_schedule(0.1, 0.001, 500), mu=0.1)

    The update returned is the thing ADDED to the parameters, matching optax throughout and
    matching ``gn.py``'s ``du``, so no sign flip is needed anywhere.
    """
    return optax.chain(
        scale_by_damped_gauss_newton(lam, mu, jitter=jitter, rel_cutoff=rel_cutoff),
        optax.scale_by_learning_rate(learning_rate),
    )


def scale_by_driver_schedule(schedule):
    """Scale by ``schedule(iteration)``, where the iteration comes from the DRIVER, not a counter.

    ``optax.scale_by_schedule`` keeps its own step count in state. That is right for a momentum
    buffer and wrong for a learning-rate anneal the moment a driver can REJECT a step: on a
    rejection :func:`lucid.fitting.minimize.minimize` restores the previous optimiser state, which
    rewinds the counter, so the anneal repeats an iteration it has already spent. The numpy step
    hit exactly this and was fixed by threading the loop index through; measured on a real event
    the rewind moved a reconstructed energy by 2.41 MeV.

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

    TWO details, both of which have cost real numbers here:

    ``transition_steps`` is ``steps - 1``, NOT ``steps``. ``optax.linear_schedule`` divides by
    ``transition_steps`` while the loop this reproduces divides by ``steps - 1``, so the obvious
    ``linear_schedule(4.0, 1.5, steps)`` ends at 1.5167 rather than 1.5 — a 1.1% error on the
    final learning rate, four orders above the float32 agreement this module otherwise argues
    about.

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
