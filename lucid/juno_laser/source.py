"""Ideal isotropic point-source construction for the minimal closure."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from lucid.sources import isotropic_source


SOURCE_MODELS = ("isotropic",)


def _rotation_matrix(key, dtype):
    """Sample a uniform three-dimensional rotation matrix."""
    quaternion = jax.random.normal(key, (4,), dtype=dtype)
    quaternion /= jnp.linalg.norm(quaternion) + 1e-12
    w, x, y, z = quaternion
    return jnp.asarray(
        (
            (1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)),
            (2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)),
            (2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)),
        ),
        dtype=dtype,
    )


def rotate_source_directions(source, directions, key):
    """Rotate quadrature directions without changing source isotropy."""
    del source
    return directions @ _rotation_matrix(key, directions.dtype).T


def build_juno_laser_source(
    source_config: Mapping,
    position,
    *,
    intensity: float,
    wavelength_nm: float,
    model: str | None = None,
):
    """Build the fixed-position, isotropic, monochromatic point source."""
    selected = str(model or source_config.get("model", "isotropic"))
    if selected != "isotropic":
        raise ValueError("minimal closure only supports an isotropic source")
    return isotropic_source(
        position,
        intensity=intensity,
        wavelength=wavelength_nm,
    )
