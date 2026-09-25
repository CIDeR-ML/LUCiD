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
    )
