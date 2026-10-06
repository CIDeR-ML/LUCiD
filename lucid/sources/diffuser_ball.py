"""Differentiable effective source for the JUNO green diffuser ball.

The model follows the two-stage J26.5.1 ``GtOpScintTool`` convention.  A
surface normal selects the emission point on a 10 mm sphere, then the photon
direction is uniform in solid angle over the local outward hemisphere.  The
non-uniform surface intensity is represented by a positive, normalized cubic
Bernstein density and applied by importance weighting fixed proposal rays.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lucid.utils import generate_orthonormal_basis, normalize


JUNO_LS_PHASE_WEIGHTS = (
    0.28182463805722086,
    0.21706967408470020,
    0.35391735883294184,
    0.14718832902513695,
)
IMPORTANCE_NORMALIZATION_MODES = (
    "analytic",
    "self_normalized",
)
_IMPORTANCE_NORMALIZATION_CODES = {
    name: index for index, name in enumerate(IMPORTANCE_NORMALIZATION_MODES)
}


class SourceBatch(NamedTuple):
    """Photon phase-space samples returned by time-aware sources."""

    directions: jnp.ndarray
    positions: jnp.ndarray
    weights: jnp.ndarray
    times: jnp.ndarray


def bernstein_angular_density(cos_theta, logits):
    """Return a positive density in ``cos(theta)`` normalized on [-1, 1].

    Parameters
    ----------
    cos_theta : array
        Surface-normal cosine relative to the diffuser assembly axis.
    logits : array, shape (4,)
        Unconstrained Bernstein mixture logits.  Softmax removes the common
        scale, leaving three independent angular-shape degrees of freedom.
    """
    x = jnp.clip(0.5 * (cos_theta + 1.0), 0.0, 1.0)
    one_minus_x = 1.0 - x
    basis = jnp.stack(
        (
            one_minus_x**3,
            3.0 * x * one_minus_x**2,
            3.0 * x**2 * one_minus_x,
            x**3,
        ),
        axis=-1,
    )
    mixture = jax.nn.softmax(logits)
    return 2.0 * jnp.sum(basis * mixture, axis=-1)


class DiffuserBallSource(NamedTuple):
    """Finite JUNO diffuser-ball source represented as a JAX pytree."""

    center: jnp.ndarray
    intensity: jnp.ndarray
    axis: jnp.ndarray
    radius: jnp.ndarray
    time_sigma: jnp.ndarray
    angular_logits: jnp.ndarray
    wavelength: object = None
    importance_normalization: object = 1

    def __call__(self, n_photons, key, n_water=1.33):
        """Sample proposal rays and attach differentiable shape weights."""
        del n_water
        dtype = self.center.dtype
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        two_pi = jnp.asarray(2.0 * np.pi, dtype=dtype)
        key_c, key_phi, key_mu, key_local_phi, key_time = jax.random.split(
            key, 5
        )
        cos_theta = jax.random.uniform(
            key_c,
            (n_photons,),
            dtype=dtype,
            minval=-one,
            maxval=one,
        )
        phi = two_pi * jax.random.uniform(
            key_phi, (n_photons,), dtype=dtype
        )
        sin_theta = jnp.sqrt(jnp.maximum(zero, one - cos_theta**2))
        local_normals = jnp.stack(
            (
                sin_theta * jnp.cos(phi),
                sin_theta * jnp.sin(phi),
                cos_theta,
            ),
            axis=1,
        )
        assembly_basis = generate_orthonormal_basis(self.axis).astype(dtype)
        normals = local_normals @ assembly_basis.T
        positions = self.center[None, :] + self.radius * normals

        reference_x = jnp.broadcast_to(
            jnp.array([1.0, 0.0, 0.0], dtype=dtype), normals.shape
        )
        reference_y = jnp.broadcast_to(
            jnp.array([0.0, 1.0, 0.0], dtype=dtype), normals.shape
        )
        references = jnp.where(
            (jnp.abs(normals[:, :1]) > 0.9), reference_y, reference_x
        )
        tangent1 = normalize(jnp.cross(normals, references))
        tangent2 = jnp.cross(normals, tangent1)
        local_cosine = jax.random.uniform(
            key_mu, (n_photons,), dtype=dtype
        )
        local_sine = jnp.sqrt(
            jnp.maximum(zero, one - local_cosine**2)
        )
        local_phi = two_pi * jax.random.uniform(
            key_local_phi, (n_photons,), dtype=dtype
        )
        directions = (
            local_cosine[:, None] * normals
            + local_sine[:, None]
            * (
                jnp.cos(local_phi)[:, None] * tangent1
                + jnp.sin(local_phi)[:, None] * tangent2
            )
        )
        directions = normalize(directions)

        density = bernstein_angular_density(
            cos_theta, self.angular_logits
        )
        importance = 2.0 * density
        sample_normalized = importance / jnp.mean(importance)
        importance = jnp.where(
            jnp.asarray(self.importance_normalization) == 1,
            sample_normalized,
            importance,
        )
        weights = self.intensity * importance / n_photons
        times = self.time_sigma * jax.random.normal(
            key_time, (n_photons,), dtype=dtype
        )
        return SourceBatch(directions, positions, weights, times)


def sample_source(source, n_photons, key):
    """Return a time-aware batch while preserving legacy source callables."""
    sample = source(n_photons, key)
    if isinstance(sample, SourceBatch):
        return sample
    directions, positions, weights = sample
    times = jnp.zeros((n_photons,), dtype=positions.dtype)
    return SourceBatch(directions, positions, weights, times)


def diffuser_ball_source(
    center,
    intensity=1_000_000,
    *,
    axis=(0.0, 0.0, 1.0),
    radius=0.010,
    time_sigma=0.23,
    angular_weights=JUNO_LS_PHASE_WEIGHTS,
    wavelength=532.0,
    importance_normalization="self_normalized",
):
    """Construct the nominal J26.5.1 LSPhase effective diffuser source."""
    center_array = np.asarray(center, dtype=float)
    axis_array = np.asarray(axis, dtype=float)
    weights_array = np.asarray(angular_weights, dtype=float)
    if center_array.shape != (3,):
        raise ValueError("diffuser center must have shape (3,)")
    if axis_array.shape != (3,) or np.linalg.norm(axis_array) <= 0.0:
        raise ValueError("diffuser axis must be a non-zero shape-(3,) vector")
    if weights_array.shape != (4,) or np.any(weights_array <= 0.0):
        raise ValueError("angular_weights must contain four positive values")
    if float(radius) <= 0.0 or float(time_sigma) < 0.0:
        raise ValueError("radius must be positive and time_sigma non-negative")
    if importance_normalization not in _IMPORTANCE_NORMALIZATION_CODES:
        choices = ", ".join(IMPORTANCE_NORMALIZATION_MODES)
        raise ValueError(
            "importance_normalization must be one of " + choices
        )
    weights_array /= weights_array.sum()
    wavelength_value = None
    if wavelength is not None:
        wavelength_value = jnp.asarray(wavelength, dtype=jnp.float32)
    return DiffuserBallSource(
        center=jnp.asarray(center_array, dtype=jnp.float32),
        intensity=jnp.asarray(intensity, dtype=jnp.float32),
        axis=normalize(jnp.asarray(axis_array, dtype=jnp.float32)),
        radius=jnp.asarray(radius, dtype=jnp.float32),
        time_sigma=jnp.asarray(time_sigma, dtype=jnp.float32),
        angular_logits=jnp.log(
            jnp.asarray(weights_array, dtype=jnp.float32)
        ),
        wavelength=wavelength_value,
        importance_normalization=_IMPORTANCE_NORMALIZATION_CODES[
            importance_normalization
        ],
    )
