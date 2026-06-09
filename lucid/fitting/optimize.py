"""The default fitter: the canonical optax loop, params-pytree in → params-pytree out.

``fit(loss, params0)`` is the whole thing — reverse-mode ``grad`` of the loss over the
*entire* params pytree (works through the photon forward's ``custom_vjp``, which defines
the vjp), an optax update, repeat. No flatten, no Schur, no finite differences, no subset
selection: JAX differentiates the whole pytree, so a 10⁴-leaf per-PMT block or a joint
``(DetectorParams, ParticleParams)`` tuple is no harder than a single scalar.

Extension points, all standard idioms:
  - **new observable** → add a term to the loss (:mod:`lucid.fitting.loss`);
  - **new parameter** → add a leaf to the params pytree (autodiff finds it);
  - **freeze / stage** → pass ``optimizer=optax.multi_transform({'train': adam,
    'freeze': optax.set_to_zero()}, labels)`` (note: ``optax.masked`` passes the raw
    gradient *through* for unmasked leaves — use ``set_to_zero`` to actually freeze);
  - **gauge** → pass ``project=`` (e.g. :func:`gauge_mean_log` for ``mean(log k)=0``);
  - **reconstruction / joint** → pass a track or a ``(dp, track)`` tuple as ``params0``.

The Gauss-Newton + Schur solver (:mod:`lucid.fitting.gauss_newton`) stays available as the
opt-in advanced path for score-dominated params or CRB-precision campaigns; it consumes the
same ``loss``/params contract.
"""

import jax
import jax.numpy as jnp
import optax


def fit(loss, params0, *, steps=300, lr=1e-2, optimizer=None, project=None, key=None):
    """Minimise ``loss(params, key)`` over the params pytree with optax.

    Parameters
    ----------
    loss : callable ``loss(params, key) -> scalar``.
    params0 : any pytree (``DetectorParams``, ``ParticleParams``, a tuple of both, …).
    steps : number of optimiser steps.
    lr : learning rate (ignored if ``optimizer`` is given).
    optimizer : an optax ``GradientTransformation`` (default ``optax.adam(lr)``). Use
        ``optax.masked`` here to freeze/stage a subset.
    project : optional ``params -> params`` applied after each update (gauges/constraints).
    key : PRNGKey (folded per step for the stochastic forward).

    Returns
    -------
    The fitted params pytree (same structure as ``params0``).
    """
    opt = optimizer if optimizer is not None else optax.adam(lr)
    key = key if key is not None else jax.random.PRNGKey(0)
    state = opt.init(params0)

    @jax.jit
    def step(params, state, k):
        grads = jax.grad(loss)(params, k)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        if project is not None:
            params = project(params)
        return params, state

    params = params0
    for s in range(steps):
        params, state = step(params, state, jax.random.fold_in(key, s))
    return params


# --- gauges: one-line projections that fix the per-PMT ↔ global degeneracies ----------
# (the first-order analogue of the Gauss-Newton Schur rank-1 gauge correction)

def gauge_mean_log(dp, leaf='qe_corrections'):
    """Fix ``mean(log k)=0`` on a positive per-PMT leaf (k ↔ global-amplitude degeneracy)."""
    k = getattr(dp.per_pmt, leaf)
    return dp._replace(per_pmt=dp.per_pmt._replace(
        **{leaf: k / jnp.exp(jnp.mean(jnp.log(jnp.clip(k, 1e-12, None))))}))


def gauge_mean(dp, leaf='t0'):
    """Fix ``mean(t0)=0`` on a signed per-PMT leaf (clock-offset ↔ emission-time degeneracy)."""
    v = getattr(dp.per_pmt, leaf)
    return dp._replace(per_pmt=dp.per_pmt._replace(**{leaf: v - jnp.mean(v)}))
