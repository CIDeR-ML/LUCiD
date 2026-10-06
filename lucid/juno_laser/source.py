"""Source construction and rotation policy for JUNO laser adapters."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from lucid.sources import isotropic_source
from lucid.sources.diffuser_ball import (
    DiffuserBallSource,
    JUNO_LS_PHASE_WEIGHTS,
    diffuser_ball_source,
)


SOURCE_MODELS = ("isotropic", "juno_diffuser_ball")
LSPHASE_SHAPE_NAMES = (
    "lsphase_shape_0",
    "lsphase_shape_1",
    "lsphase_shape_2",
)
LSPHASE_CENTERED_BASIS = jnp.asarray(
    (
        (1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(6.0),
         1.0 / jnp.sqrt(12.0)),
        (-1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(6.0),
         1.0 / jnp.sqrt(12.0)),
        (0.0, -2.0 / jnp.sqrt(6.0), 1.0 / jnp.sqrt(12.0)),
        (0.0, 0.0, -3.0 / jnp.sqrt(12.0)),
    ),
    dtype=jnp.float32,
)
JUNO_LS_PHASE_COEFFICIENTS = jnp.asarray(
    (
        1.091633205450117,
        -0.003381008948690,
        -0.217066500081251,
        0.277844585222388,
    ),
    dtype=jnp.float32,
)
JUNO_LS_PHASE_INTEGRAL = (
    2.0 * JUNO_LS_PHASE_COEFFICIENTS[0]
    + (2.0 / 3.0) * JUNO_LS_PHASE_COEFFICIENTS[2]
)


def _rotation_matrix(key, dtype):
    """Sample a uniform three-dimensional rotation matrix."""
    quaternion = jax.random.normal(key, (4,), dtype=dtype)
    quaternion /= jnp.linalg.norm(quaternion) + 1e-12
    w, x, y, z = quaternion
    return jnp.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=dtype,
    )


def rotate_source_directions(source, directions, key):
    """Rotate isotropic quadrature, but preserve a diffuser's physical axis."""
    if isinstance(source, DiffuserBallSource):
        return directions
    return directions @ _rotation_matrix(key, directions.dtype).T


def set_diffuser_angular_shape(source, shape_coordinates):
    """Apply three identifiable centered-logit LSPhase coordinates."""
    if not isinstance(source, DiffuserBallSource):
        raise TypeError("LSPhase coordinates require a DiffuserBallSource")
    coordinates = jnp.asarray(
        shape_coordinates, dtype=source.angular_logits.dtype
    )
    if coordinates.shape != (3,):
        raise ValueError("LSPhase shape coordinates must have shape (3,)")
    logits = source.angular_logits
    centered = logits - jnp.mean(logits)
    delta = LSPHASE_CENTERED_BASIS @ coordinates
    return source._replace(angular_logits=centered + delta)


def lsphase_polynomial_coefficients(angular_logits):
    """Return four gauge-fixed JUNOSW ``pol3`` coefficients.

    LUCiD samples the opposite cosine sign from ``TF1::GetRandom`` in
    ``GtOpScintTool``.  Odd polynomial coefficients are therefore flipped
    back before reporting the JUNOSW convention.  The integral is fixed to
    the nominal JUNOSW value, eliminating the common-scale zero mode.
    """
    weights = jax.nn.softmax(jnp.asarray(angular_logits))
    w0, w1, w2, w3 = weights
    internal = jnp.stack(
        (
            0.25 * (w0 + 3.0 * w1 + 3.0 * w2 + w3),
            0.75 * (-w0 - w1 + w2 + w3),
            0.75 * (w0 - w1 - w2 + w3),
            0.25 * (-w0 + 3.0 * w1 - 3.0 * w2 + w3),
        )
    )
    junosw = internal * jnp.asarray((1.0, -1.0, 1.0, -1.0))
    return JUNO_LS_PHASE_INTEGRAL * junosw


def build_juno_laser_source(
    source_config: Mapping,
    position,
    *,
    intensity: float,
    wavelength_nm: float,
    model: str | None = None,
):
    """Build a configured point or JUNO diffuser-ball laser source."""
    selected = str(model or source_config.get("model", "isotropic"))
    if selected == "isotropic":
        return isotropic_source(
            position,
            intensity=intensity,
            wavelength=wavelength_nm,
        )
    if selected != "juno_diffuser_ball":
        choices = ", ".join(SOURCE_MODELS)
        raise ValueError(
            f"unknown JUNO laser source model {selected!r}; choose {choices}"
        )
    return diffuser_ball_source(
        position,
        intensity=intensity,
        axis=source_config.get("axis", (0.0, 0.0, 1.0)),
        radius=float(source_config.get("radius_m", 0.010)),
        time_sigma=float(source_config.get("time_sigma_ns", 0.23)),
        angular_weights=source_config.get(
            "angular_weights", JUNO_LS_PHASE_WEIGHTS
        ),
        wavelength=wavelength_nm,
        importance_normalization=source_config.get(
            "importance_normalization", "self_normalized"
        ),
    )
