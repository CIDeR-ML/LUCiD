"""Differentiable losses for JUNO laser per-PE calibration."""

from __future__ import annotations

import jax.numpy as jnp


def poisson_deviance(observed, expected, *, floor: float = 1e-8):
    """Return ``sum(expected - observed + observed*log(observed/expected))``.

    The zero-observation branch is evaluated without ``0*log(0)`` so the
    result remains finite for empty PMT/time bins.
    """
    observed = jnp.asarray(observed, dtype=jnp.float32)
    expected = jnp.maximum(jnp.asarray(expected, dtype=observed.dtype), floor)
    positive = observed > 0
    log_ratio = jnp.where(
        positive,
        jnp.log(jnp.maximum(observed, 1.0)) - jnp.log(expected),
        0.0,
    )
    return jnp.sum(expected - observed + observed * log_ratio)


def per_pe_deviance(observed, expected, *, n_events: int = 1,
                    time_mask=None, charge_weight: float = 0.0,
                    normalize: bool = True):
    """Poisson deviance for a binned source observation.

    Parameters
    ----------
    observed : array, shape ``(n_sensors, n_bins)``
        Accumulated PE counts over ``n_events`` flashes.
    expected : array, shape ``(n_sensors, n_bins)``
        Per-event expected PE histogram from the forward model.
    n_events : int
        Number of observed flashes; expected counts are scaled by this value.
    time_mask : array or None
        Optional binary PMT/time mask applied to the full-time term.
    charge_weight : float
        Optional weight for a PMT-marginal charge deviance term.
    normalize : bool
        Divide by observed PE count (plus charge contribution) for stable
        learning-rate scale across source positions and event counts.
    """
    if n_events <= 0:
        raise ValueError("n_events must be positive")
    observed = jnp.asarray(observed, dtype=jnp.float32)
    expected = jnp.asarray(expected, dtype=observed.dtype) * float(n_events)
    if observed.shape != expected.shape:
        raise ValueError(f"observed/expected shapes differ: {observed.shape} vs {expected.shape}")
    if time_mask is None:
        mask = jnp.ones_like(observed)
    else:
        mask = jnp.asarray(time_mask, dtype=observed.dtype)
        if mask.shape != observed.shape:
            raise ValueError(f"time_mask shape {mask.shape} != observation shape {observed.shape}")

    full_observed = observed * mask
    full_expected = expected * mask
    total = poisson_deviance(full_observed, full_expected)
    if charge_weight:
        total += float(charge_weight) * poisson_deviance(
            jnp.sum(observed, axis=1), jnp.sum(expected, axis=1))
    if not normalize:
        return total
    denominator = jnp.maximum(
        jnp.sum(full_observed) + float(charge_weight) * jnp.sum(observed), 1.0)
    return total / denominator

