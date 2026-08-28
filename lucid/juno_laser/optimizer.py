"""Ensemble objectives and staged Adam optimization for JUNO laser calibration."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .config import PARAMETER_NAMES
from .data import PerPeObservation
from .losses import per_pe_deviance
from .parameters import unpack_params


LOG = logging.getLogger("lucid.juno_laser.optimizer")


def make_ensemble_forward(forward, sources, hist_shape, *, fit_amp: bool = False):
    """Average fixed transport-key predictions for each source."""
    sources = tuple(sources)
    hist_shape = tuple(hist_shape)

    @jax.jit
    def ensemble(theta_log, keys):
        params = unpack_params(theta_log, fit_amp=fit_amp)
        predictions = []
        for source_index, source in enumerate(sources):
            def accumulate(histogram, key):
                prediction, _ = forward(
                    source, params, jax.random.fold_in(key, source_index))
                return histogram + prediction, None

            first, _ = jax.lax.scan(
                jax.remat(accumulate),
                jnp.zeros(hist_shape, dtype=jnp.float32),
                keys)
            predictions.append(first / keys.shape[0])
        return tuple(predictions)

    return ensemble


def make_per_pe_objective(ensemble_forward,
                          observations: Sequence[PerPeObservation], *,
                          time_masks=None, charge_weight: float = 0.0,
                          fit_amp: bool = False):
    """Build a normalized multi-source binned Poisson objective."""
    observations = tuple(observations)
    obs_counts = tuple(jnp.asarray(item.counts, dtype=jnp.float32)
                       for item in observations)
    n_events = tuple(int(item.n_events) for item in observations)
    if time_masks is None:
        time_masks = tuple(None for _ in observations)
    if len(time_masks) != len(observations):
        raise ValueError("one time mask is required per source")
    masks = tuple(None if mask is None else jnp.asarray(mask, dtype=jnp.float32)
                  for mask in time_masks)

    @jax.jit
    def objective(theta_log, keys):
        predictions = ensemble_forward(theta_log, keys)
        total = jnp.asarray(0.0, dtype=jnp.float32)
        for observed, predicted, events, mask in zip(obs_counts, predictions, n_events, masks):
            total += per_pe_deviance(
                observed, predicted, n_events=events,
                time_mask=mask, charge_weight=charge_weight)
        return total / max(len(observations), 1)

    return objective


def _stage_bounds(fields: Sequence[str], bounds: dict[str, tuple[float, float]]):
    try:
        return (jnp.asarray([np.log(bounds[name][0]) for name in fields], dtype=jnp.float32),
                jnp.asarray([np.log(bounds[name][1]) for name in fields], dtype=jnp.float32))
    except KeyError as exc:
        raise ValueError(f"missing optimizer bounds for {exc.args[0]!r}") from exc


def adam_stage(objective, theta0, fields: Sequence[str], active_fields: Iterable[str],
               *, bounds, keys, steps: int, learning_rate: float,
               grad_clip: float = 1e4, validation_objective=None,
               validation_keys=None, validation_every: int = 0,
               early_stopping_patience: int = 0, min_improvement: float = 1e-4):
    """Run one bounded Adam stage in log space.

    Inactive coordinates are held exactly at their incoming values.  The
    return value includes both train-best and validation-best checkpoints so
    staged callers can choose a held-out checkpoint without re-running forward.
    """
    fields = tuple(fields)
    active_fields = frozenset(active_fields)
    if steps <= 0 or learning_rate <= 0 or grad_clip <= 0:
        raise ValueError("steps, learning_rate and grad_clip must be positive")
    lo, hi = _stage_bounds(fields, bounds)
    active = jnp.asarray([name in active_fields for name in fields])
    if not bool(np.any(np.asarray(active))):
        raise ValueError("stage has no active parameters")
    theta = jnp.clip(jnp.asarray(theta0, dtype=jnp.float32), lo, hi)
    moment1 = jnp.zeros_like(theta)
    moment2 = jnp.zeros_like(theta)
    value_and_grad = jax.jit(jax.value_and_grad(objective))
    history, losses, grad_norms, validation_history = [], [], [], []
    best_loss = np.inf
    best_theta = np.asarray(theta)
    best_validation_loss = np.inf
    best_validation_theta = None
    best_validation_step = 0
    stale_steps = 0

    for step in range(steps):
        value, gradient = value_and_grad(theta, keys)
        loss = float(value)
        gradient = np.asarray(gradient, dtype=np.float64)
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
        gradient *= np.asarray(active, dtype=np.float64)
        norm = float(np.linalg.norm(gradient))
        if norm > grad_clip:
            gradient *= grad_clip / max(norm, 1e-12)
        if np.isfinite(loss) and loss < best_loss - min_improvement:
            best_loss, best_theta, stale_steps = loss, np.asarray(theta), 0
        else:
            stale_steps += 1

        gradient = jnp.asarray(gradient, dtype=theta.dtype)
        moment1 = 0.9 * moment1 + 0.1 * gradient
        moment2 = 0.999 * moment2 + 0.001 * gradient * gradient
        m_hat = moment1 / (1.0 - 0.9 ** (step + 1))
        v_hat = moment2 / (1.0 - 0.999 ** (step + 1))
        proposal = theta - learning_rate * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        theta = jnp.where(active, jnp.clip(proposal, lo, hi), theta)
        history.append(np.asarray(jnp.exp(theta)))
        losses.append(loss)
        grad_norms.append(norm)

        validate = (validation_objective is not None and validation_every > 0 and
                    (step == 0 or (step + 1) % validation_every == 0 or step + 1 == steps))
        if validate:
            validation_loss = float(validation_objective(theta, validation_keys))
            validation_history.append((step + 1, validation_loss))
            if np.isfinite(validation_loss) and validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_validation_theta = np.asarray(theta)
                best_validation_step = step + 1
            LOG.info("stage validation step %d/%d: %.6g", step + 1, steps, validation_loss)
        if step == 0 or (step + 1) % max(1, steps // 5) == 0:
            LOG.info("stage step %d/%d: deviance %.6g, |grad| %.4g",
                     step + 1, steps, loss, norm)
        if early_stopping_patience and stale_steps >= early_stopping_patience:
            LOG.info("stage early stop at step %d", step + 1)
            break

    return {
        "theta": np.asarray(jnp.exp(best_theta)),
        "log_theta": np.asarray(best_theta),
        "last_theta": np.asarray(jnp.exp(theta)),
        "last_log_theta": np.asarray(theta),
        "history": np.asarray(history),
        "loss_history": np.asarray(losses),
        "grad_norm_history": np.asarray(grad_norms),
        "best_loss": float(best_loss),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_step": int(best_validation_step),
        "best_validation_log_theta": best_validation_theta,
        "validation_history": validation_history,
        "active_fields": sorted(active_fields),
    }
