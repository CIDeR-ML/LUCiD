"""Composable observation terms + a loss builder — the shared primitive for fitting.

A fit is just ``loss(params, key) -> scalar``. The loss is a sum of per-observable
*terms*, each comparing the forward's per-sensor output ``pred`` to the observed ``obs``.
Both are the raw output of the calibration forward: ``(mean_charge, var_charge, time)`` in
``hit_mode='moments'`` or ``(charge, time)`` in ``'aggregated'``. Terms index positionally,
so ``charge`` / ``first_arrival`` work in either mode and ``charge_var`` needs ``moments``.

This one ``loss`` is the contract every solver and search shares: the default optax
:func:`lucid.fitting.optimize.fit` differentiates it; the grid/cone searches evaluate it;
the advanced Gauss-Newton solver linearises it. New observable = new term; new parameter =
a leaf in the params pytree (autodiff picks it up); reconstruction = pass a track pytree and
write a 3-line ``loss`` directly (``make_loss`` is the calibration convenience).
"""

import jax.numpy as jnp


def sqrt_mse(pred, truth, eps=1e-8):
    """Poisson variance-stabilising residual ``Σ (√pred − √truth)²`` (near-unbiased;
    Poisson-NLL biases the optical scales ~1.3%)."""
    return jnp.sum((jnp.sqrt(pred + eps) - jnp.sqrt(truth + eps)) ** 2)


def charge(pred, obs):
    """Per-sensor charge (mean) residual — aggregated or moments mode (field 0)."""
    return sqrt_mse(pred[0], obs[0])


def charge_var(pred, obs):
    """Per-sensor charge-variance residual — moments mode only (field 1)."""
    return sqrt_mse(pred[1], obs[1])


def first_arrival(pred, obs):
    """Per-sensor first-arrival-time residual over lit sensors (time is the last field).

    ``t0`` enters the time additively *after* the hard min, so this is smooth in the
    per-PMT ``t0`` and global ``tts`` even though the min itself is not differentiable.
    """
    lit = obs[0] > 0
    return jnp.sum(lit * (pred[-1] - obs[-1]) ** 2)


def make_loss(sim, sources, observations, terms):
    """Build ``loss(params, key) = Σ_sources Σ_terms w·term(pred, obs)`` for calibration.

    Parameters
    ----------
    sim : callable ``sim(source, params, key) -> pred`` (the calibration forward).
    sources : list of calibration source objects.
    observations : list of observed forward outputs, one per source (the data).
    terms : list of term callables, or ``(term, weight)`` pairs.
    """
    terms = [t if isinstance(t, tuple) else (t, 1.0) for t in terms]

    def loss(params, key):
        tot = 0.0
        for src, obs in zip(sources, observations):
            pred = sim(src, params, key)
            for term, w in terms:
                tot = tot + w * term(pred, obs)
        return tot

    return loss
