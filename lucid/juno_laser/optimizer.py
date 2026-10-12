"""Ensemble objectives and staged Adam for JUNO laser calibration."""

from __future__ import annotations

import logging
import time
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .parameters import (
    decode_calibration_array,
    optimization_bounds,
    unpack_params,
)
from .source import set_diffuser_angular_shape


LOG = logging.getLogger("lucid.juno_laser.optimizer")


def make_ensemble_forward(
    forward,
    sources,
    hist_shape,
    *,
    fixed_params,
    source_indices=None,
    fit_diffuser_shape=True,
):
    """Average fixed-key predictions using stable absolute source indices.

    ``fit_diffuser_shape=False`` preserves an ideal source exactly and keeps
    the three LSPhase coordinates outside the active computation graph.
    """
    sources = tuple(sources)
    hist_shape = tuple(hist_shape)
    if source_indices is None:
        source_indices = tuple(range(len(sources)))
    source_indices = tuple(int(index) for index in source_indices)
    if len(source_indices) != len(sources):
        raise ValueError("source indices must match the source sequence")
    if len(set(source_indices)) != len(source_indices):
        raise ValueError("source indices must be unique")
    if source_indices and min(source_indices) < 0:
        raise ValueError("source indices must be non-negative")

    stacked_sources = jax.tree_util.tree_map(
        lambda *items: jnp.stack(items), *sources
    )
    stacked_indices = jnp.asarray(source_indices, dtype=jnp.int32)

    @jax.jit
    def ensemble(theta, keys):
        params, shape_coordinates = unpack_params(theta, fixed_params)

        def predict_source(inputs):
            source_index, source = inputs
            varied_source = (
                set_diffuser_angular_shape(source, shape_coordinates)
                if fit_diffuser_shape
                else source
            )

            def accumulate(histogram, key):
                prediction, _ = forward(
                    varied_source,
                    params,
                    jax.random.fold_in(key, source_index),
                )
                return histogram + prediction, None

            total, _ = jax.lax.scan(
                jax.remat(accumulate),
                jnp.zeros(hist_shape, dtype=jnp.float32),
                keys,
            )
            return total / keys.shape[0]

        return jax.lax.map(
            predict_source, (stacked_indices, stacked_sources)
        )

    return ensemble


def select_stage_coordinates(result):
    """Choose held-out, MC-validation or train-best coordinates."""
    heldout = result.get("best_heldout_validation_coordinates")
    mc = result.get("best_mc_validation_coordinates")
    if heldout is not None:
        return np.asarray(heldout)
    if mc is not None:
        return np.asarray(mc)
    return np.asarray(result["coordinates"])


def adam_stage(
    objective,
    theta0,
    fields: Sequence[str],
    active_fields: Iterable[str],
    *,
    bounds,
    train_keys,
    steps: int,
    learning_rate: float,
    grad_clip: float = 1e4,
    mc_validation_objective=None,
    mc_validation_keys=None,
    heldout_validation_objective=None,
    heldout_validation_keys=None,
    validation_every: int = 0,
    early_stopping_patience: int = 0,
    min_improvement: float = 1e-4,
    checkpoint_callback=None,
    checkpoint_every: int = 0,
    value_and_grad_fn=None,
):
    """Run one bounded Adam stage in transformed coordinates.

    Inactive coordinates are held exactly at their incoming values.  Train
    keys remain fixed to give a common-random-number optimization path.
    MC validation uses the same data with independent transport keys, while
    held-out validation may use different events, PMTs or source positions.
    Early-stopping patience counts validation evaluations, not Adam updates.
    """
    fields = tuple(fields)
    active_fields = frozenset(active_fields)
    if steps <= 0 or learning_rate <= 0 or grad_clip <= 0:
        raise ValueError("steps, learning_rate and grad_clip must be positive")
    if min_improvement < 0.0:
        raise ValueError("min_improvement must be non-negative")
    if (
        early_stopping_patience < 0
        or validation_every < 0
        or checkpoint_every < 0
    ):
        raise ValueError("validation controls must be non-negative")
    validations = (
        ("mc", mc_validation_objective, mc_validation_keys),
        (
            "heldout",
            heldout_validation_objective,
            heldout_validation_keys,
        ),
    )
    for name, validation_objective, validation_keys in validations:
        if (validation_objective is None) != (validation_keys is None):
            raise ValueError(
                f"{name} validation objective and keys must be paired"
            )
    stopping_name = (
        "heldout"
        if heldout_validation_objective is not None
        else "mc" if mc_validation_objective is not None else None
    )
    if early_stopping_patience and stopping_name is None:
        raise ValueError("early stopping requires a validation objective")
    if stopping_name is not None and validation_every <= 0:
        raise ValueError("validation_every must be positive with validation")
    missing = [name for name in fields if name not in bounds]
    if missing:
        raise ValueError(f"missing optimizer bounds for {missing[0]!r}")
    lo, hi = optimization_bounds(bounds)
    active = jnp.asarray([name in active_fields for name in fields])
    if not bool(np.any(np.asarray(active))):
        raise ValueError("stage has no active parameters")
    theta = jnp.clip(jnp.asarray(theta0, dtype=jnp.float32), lo, hi)
    moment1 = jnp.zeros_like(theta)
    moment2 = jnp.zeros_like(theta)
    value_and_grad = (
        jax.jit(jax.value_and_grad(objective))
        if value_and_grad_fn is None
        else value_and_grad_fn
    )
    stage_started = time.perf_counter()
    history, coordinate_history = [], []
    losses, grad_norms = [], []
    evaluation_seconds = []
    mc_validation_history, heldout_validation_history = [], []
    active_array = np.asarray(active, dtype=np.float64)

    def evaluate(current, step):
        """Evaluate one parameter state and return its active gradient."""
        evaluation_started = time.perf_counter()
        value, raw_gradient = value_and_grad(current, train_keys)
        loss = float(value)
        gradient = np.asarray(raw_gradient, dtype=np.float64)
        invalid = np.flatnonzero(~np.isfinite(gradient))
        if not np.isfinite(loss):
            raise FloatingPointError(
                f"non-finite objective at optimizer step {step}: {loss}"
            )
        if invalid.size:
            names = ", ".join(fields[index] for index in invalid)
            raise FloatingPointError(
                "non-finite gradient at optimizer step "
                f"{step}: {names}"
            )
        gradient *= active_array
        norm = float(np.linalg.norm(gradient))
        if norm > grad_clip:
            gradient *= grad_clip / max(norm, 1e-12)
        elapsed = time.perf_counter() - evaluation_started
        return (
            loss,
            jnp.asarray(gradient, dtype=current.dtype),
            norm,
            elapsed,
        )

    loss, gradient, norm, elapsed = evaluate(theta, 0)
    history.append(np.asarray(decode_calibration_array(theta)))
    coordinate_history.append(np.asarray(theta))
    losses.append(loss)
    grad_norms.append(norm)
    evaluation_seconds.append(elapsed)
    best_loss = loss
    best_theta = np.asarray(theta)
    best_step = 0
    validation_state = {
        "mc": {
            "best_loss": np.inf,
            "best_coordinates": None,
            "best_step": None,
            "history": mc_validation_history,
        },
        "heldout": {
            "best_loss": np.inf,
            "best_coordinates": None,
            "best_step": None,
            "history": heldout_validation_history,
        },
    }
    patience_reference_loss = np.inf
    stale_validation_evaluations = 0
    validation_evaluations = 0

    def evaluate_validations(current, step):
        """Evaluate configured validation objectives at one checkpoint."""
        nonlocal patience_reference_loss
        nonlocal stale_validation_evaluations
        nonlocal validation_evaluations
        losses_by_name = {}
        for name, validation_objective, validation_keys in validations:
            if validation_objective is None:
                continue
            value = float(validation_objective(current, validation_keys))
            if not np.isfinite(value):
                raise FloatingPointError(
                    f"non-finite {name} validation objective at optimizer "
                    f"step {step}: {value}"
                )
            state = validation_state[name]
            state["history"].append((step, value))
            if value < state["best_loss"]:
                state["best_loss"] = value
                state["best_coordinates"] = np.asarray(current)
                state["best_step"] = step
            losses_by_name[name] = value
        if stopping_name is not None:
            validation_evaluations += 1
            stopping_loss = losses_by_name[stopping_name]
            if stopping_loss < patience_reference_loss - min_improvement:
                patience_reference_loss = stopping_loss
                stale_validation_evaluations = 0
            else:
                stale_validation_evaluations += 1
        return losses_by_name

    if stopping_name is not None:
        evaluate_validations(theta, 0)

    for step in range(1, steps + 1):
        moment1 = 0.9 * moment1 + 0.1 * gradient
        moment2 = 0.999 * moment2 + 0.001 * gradient * gradient
        m_hat = moment1 / (1.0 - 0.9 ** step)
        v_hat = moment2 / (1.0 - 0.999 ** step)
        proposal = theta - learning_rate * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        theta = jnp.where(active, jnp.clip(proposal, lo, hi), theta)
        loss, gradient, norm, elapsed = evaluate(theta, step)
        history.append(np.asarray(decode_calibration_array(theta)))
        coordinate_history.append(np.asarray(theta))
        losses.append(loss)
        grad_norms.append(norm)
        evaluation_seconds.append(elapsed)
        if loss < best_loss:
            best_loss = loss
            best_theta = np.asarray(theta)
            best_step = step
        validate = (
            stopping_name is not None
            and (
                step % validation_every == 0
                or step == steps
            )
        )
        if validate:
            validation_losses = evaluate_validations(theta, step)
            LOG.info("stage validation step %d/%d: %s", step, steps,
                     validation_losses)
        if step == 1 or step % max(1, steps // 5) == 0:
            LOG.info("stage step %d/%d: train loss %.6g, |grad| %.4g",
                     step, steps, loss, norm)
        if (
            checkpoint_callback is not None
            and checkpoint_every > 0
            and (step % checkpoint_every == 0 or step == steps)
        ):
            checkpoint_callback({
                "step": int(step),
                "coordinates": np.asarray(theta),
                "values": np.asarray(decode_calibration_array(theta)),
                "loss": float(loss),
                "gradient": np.asarray(gradient),
                "moment1": np.asarray(moment1),
                "moment2": np.asarray(moment2),
                "best_coordinates": np.asarray(best_theta),
                "best_loss": float(best_loss),
                "best_step": int(best_step),
            })
        if (
            validate
            and early_stopping_patience
            and stale_validation_evaluations >= early_stopping_patience
        ):
            LOG.info(
                "stage early stop at step %d after %d stale %s "
                "validation evaluations",
                step,
                stale_validation_evaluations,
                stopping_name,
            )
            break

    return {
        "values": np.asarray(decode_calibration_array(best_theta)),
        "coordinates": np.asarray(best_theta),
        "last_values": np.asarray(decode_calibration_array(theta)),
        "last_coordinates": np.asarray(theta),
        "history": np.asarray(history),
        "coordinate_history": np.asarray(coordinate_history),
        "loss_history": np.asarray(losses),
        "grad_norm_history": np.asarray(grad_norms),
        "evaluation_seconds": np.asarray(evaluation_seconds),
        "stage_seconds": float(time.perf_counter() - stage_started),
        "best_loss": float(best_loss),
        "best_step": int(best_step),
        "last_loss": float(loss),
        "early_stopping_validation": stopping_name,
        "validation_evaluations": int(validation_evaluations),
        "stale_validation_evaluations": int(stale_validation_evaluations),
        "patience_reference_loss": (
            None
            if not np.isfinite(patience_reference_loss)
            else float(patience_reference_loss)
        ),
        "best_mc_validation_loss": (
            None
            if not np.isfinite(validation_state["mc"]["best_loss"])
            else float(validation_state["mc"]["best_loss"])
        ),
        "best_mc_validation_step": validation_state["mc"]["best_step"],
        "best_mc_validation_coordinates": validation_state["mc"][
            "best_coordinates"
        ],
        "mc_validation_history": mc_validation_history,
        "best_heldout_validation_loss": (
            None
            if not np.isfinite(validation_state["heldout"]["best_loss"])
            else float(validation_state["heldout"]["best_loss"])
        ),
        "best_heldout_validation_step": validation_state["heldout"][
            "best_step"
        ],
        "best_heldout_validation_coordinates": validation_state[
            "heldout"
        ]["best_coordinates"],
        "heldout_validation_history": heldout_validation_history,
        "active_fields": sorted(active_fields),
    }
