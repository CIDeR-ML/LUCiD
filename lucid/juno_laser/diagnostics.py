"""Gradient diagnostics for the minimal JUNO laser closure."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def objective_ad_fd_comparison(
    objective,
    coordinates,
    keys,
    *,
    relative_step: float = 1.0e-3,
    minimum_step: float = 1.0e-4,
    value_and_grad_fn=None,
):
    """Compare automatic and centered finite-difference gradients.

    输入 ~~~ objective: fixed-key scalar objective; coordinates: optimizer
    vector; keys: shared Monte Carlo keys; relative_step/minimum_step:
    finite-difference controls; value_and_grad_fn: optional host evaluator.
    输出 ~~~ Objective, AD/FD gradients, steps, and discrepancy summaries.

    Every perturbation reuses the same transport keys.  Without this common
    random-number condition, finite differences would mostly measure Monte
    Carlo variation instead of the local derivative.
    """
    if relative_step <= 0.0 or minimum_step <= 0.0:
        raise ValueError("finite-difference steps must be positive")
    theta = jnp.asarray(coordinates)
    if theta.ndim != 1:
        raise ValueError("coordinates must be one-dimensional")
    if value_and_grad_fn is None:
        value, ad_gradient = jax.value_and_grad(objective)(theta, keys)
        value.block_until_ready()
    else:
        value, ad_gradient = value_and_grad_fn(theta, keys)
    ad = np.asarray(ad_gradient, dtype=np.float64)
    host_theta = np.asarray(theta, dtype=np.float64)
    steps = np.maximum(np.abs(host_theta) * relative_step, minimum_step)
    plus_values = []
    minus_values = []
    finite_difference = []
    for index, step in enumerate(steps):
        displacement = jnp.zeros_like(theta).at[index].set(
            jnp.asarray(step, dtype=theta.dtype)
        )
        plus = float(objective(theta + displacement, keys))
        minus = float(objective(theta - displacement, keys))
        plus_values.append(plus)
        minus_values.append(minus)
        finite_difference.append((plus - minus) / (2.0 * step))
    fd = np.asarray(finite_difference, dtype=np.float64)
    plus_array = np.asarray(plus_values, dtype=np.float64)
    minus_array = np.asarray(minus_values, dtype=np.float64)
    invalid = np.flatnonzero(
        ~np.isfinite(plus_array) | ~np.isfinite(minus_array)
    )
    if invalid.size:
        details = ", ".join(
            f"index {index}: plus={plus_array[index]}, "
            f"minus={minus_array[index]}"
            for index in invalid
        )
        raise FloatingPointError(
            "AD-FD perturbation left the strict PPP support: " + details
        )
    scale = np.maximum.reduce(
        (np.abs(ad), np.abs(fd), np.full_like(ad, 1.0e-12))
    )
    relative_difference = np.abs(ad - fd) / scale
    if not (
        np.isfinite(float(value))
        and np.all(np.isfinite(ad))
        and np.all(np.isfinite(fd))
    ):
        raise FloatingPointError("AD-FD audit produced non-finite values")
    return {
        "objective": float(value),
        "relative_step": float(relative_step),
        "minimum_step": float(minimum_step),
        "steps": steps,
        "ad_gradient": ad,
        "finite_difference_gradient": fd,
        "absolute_difference": np.abs(ad - fd),
        "relative_difference": relative_difference,
        "plus_objective": plus_array,
        "minus_objective": minus_array,
        "max_relative_difference": float(np.max(relative_difference)),
    }
