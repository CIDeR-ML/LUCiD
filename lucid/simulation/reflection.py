"""Pluggable reflection models for the differentiable photon transport step.

A reflection model is a function with the signature::

    reflection_fn(direction, normal, hit_sensor, refl_params, lam, key)
        -> (refl_prob, refl_dir, lr_score)

where

- ``direction`` : (3,) incoming photon direction (LIVE — carries the pathwise track gradient)
- ``normal``    : (3,) outward surface normal (the model detaches it where curvature would blow up)
- ``hit_sensor``: bool, sensor (True) vs wall (False)
- ``refl_params``: a model-specific pytree of the fittable reflection parameters
                   (for the scalar model, the two reflection rates)
- ``lam``       : per-photon wavelength (nm); used by wavelength-dependent models, ignored otherwise
- ``key``       : a PRNGKey slot for the reflection's stochastic direction choice

and returns

- ``refl_prob`` : reflection probability (used as ``1 - refl_prob`` in the implicit-capture
                  deposit and ``refl_prob`` in the continuation weight)
- ``refl_dir``  : the post-reflection direction
- ``lr_score``  : a DiCE score increment for any DISCRETE reflection branch — 0.0 for the
                  scalar model; the specular/diffuse-mix log-prob for angular models

The model is chosen at setup and captured in the photon step's closure, so the
``custom_vjp`` step signature stays fixed (``refl_params`` is a single packed pytree
argument) — adding a new model never reshapes it. ``scalar_reflection`` is the default
and reproduces the legacy angle-independent behaviour byte-for-byte.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from lucid.simulation.optics import (
    compute_reflection_direction, sample_cosine_hemisphere,
)

sg = jax.lax.stop_gradient


# ---------------------------------------------------------------------------
# Scalar model (default) — angle/λ-independent rates, hard direction.
# ---------------------------------------------------------------------------

class ScalarReflection(NamedTuple):
    """``refl_params`` for the scalar reflection model.

    Fields
    ------
    wall_rate : jnp.ndarray     scalar reflection probability at walls
    sensor_rate : jnp.ndarray   scalar reflection probability at sensors
    """
    wall_rate: jnp.ndarray
    sensor_rate: jnp.ndarray


def scalar_reflection(direction, normal, hit_sensor, refl_params, lam, key):
    """Angle/λ-independent reflection: scalar wall/sensor rates, hard direction
    (walls diffuse, sensors specular). Byte-identical to the legacy photon step.

    The normal is detached (``sg``) for the reflected direction — the Igehy (1999)
    curvature term through a live normal compounds ~1/r across bounces. ``lam`` is
    unused.
    """
    refl_prob = jnp.where(hit_sensor, refl_params.sensor_rate, refl_params.wall_rate)
    normal_refl = sg(normal)
    specular_dir = compute_reflection_direction(direction, normal_refl)
    diffuse_dir = sample_cosine_hemisphere(-normal_refl, key)
    refl_dir = jnp.where(hit_sensor, specular_dir, diffuse_dir)
    lr_score = jnp.zeros_like(refl_prob)
    return refl_prob, refl_dir, lr_score


# ---------------------------------------------------------------------------
# Model registry — name → (reflection_fn, build_refl_params(detector_params)).
# build_refl_params extracts the model's parameters from a DetectorParams pytree.
# Angular models (Schlick wall / multilayer-Fresnel sensor) are registered in a
# later step; the scalar model is the byte-identical default.
# ---------------------------------------------------------------------------

def _build_scalar_params(detector_params):
    return ScalarReflection(
        wall_rate=detector_params.reflection.wall_reflection_rate,
        sensor_rate=detector_params.reflection.sensor_reflection_rate,
    )


REFLECTION_MODELS = {
    'scalar': (scalar_reflection, _build_scalar_params),
}


def get_reflection_model(name):
    """Return ``(reflection_fn, build_refl_params)`` for a registered model name."""
    if name not in REFLECTION_MODELS:
        raise ValueError(
            f"Unknown reflection model {name!r}; available: {sorted(REFLECTION_MODELS)}")
    return REFLECTION_MODELS[name]
