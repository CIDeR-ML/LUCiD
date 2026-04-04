"""Calibration source ray generators (isotropic, laser, etc.).

Moved from lucid/generate.py and lucid/detector_params.py during Phase 2.2 refactor.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import NamedTuple
from functools import partial
from lucid.utils import normalize, generate_orthonormal_basis


@partial(jax.jit, static_argnums=(2,))
def get_isotropic_rays(source_position, source_intensity, Nphot, key):
    """
    Fibonacci spiral using scan-based recurrence to avoid float32 precision issues.
    """
    # Block configuration
    block_size = 1000
    n_full_blocks = Nphot // block_size
    remainder = Nphot % block_size

    golden_ratio_inverse = jnp.float32(0.6180339887498949)
    block_increment = (block_size * golden_ratio_inverse) % 1.0

    # Compute block offsets using scan
    def scan_fn(carry, x):
        next_offset = (carry + block_increment) % 1.0
        return next_offset, carry

    _, block_offsets = jax.lax.scan(scan_fn, 0.0, None, length=n_full_blocks + (1 if remainder else 0))

    # Generate all points
    all_indices = jnp.arange(Nphot, dtype=jnp.float32)
    block_idx = all_indices // block_size
    local_idx = all_indices % block_size

    # Compute theta using block offsets
    local_t = (local_idx * golden_ratio_inverse) % 1.0
    block_offset = block_offsets[block_idx.astype(jnp.int32)]
    t = (local_t + block_offset) % 1.0
    theta = 2.0 * jnp.pi * t

    # Compute z and r
    z = 1.0 - 2.0 * (all_indices + 0.5) / Nphot
    r = jnp.sqrt(jnp.maximum(0.0, 1.0 - z * z))

    # Convert to Cartesian
    x = jnp.cos(theta) * r
    y = jnp.sin(theta) * r

    ray_vectors = jnp.stack([x, y, z], axis=1)
    ray_origins = jnp.tile(source_position, (Nphot, 1))
    photon_weights = jnp.ones(Nphot) * (source_intensity / Nphot)

    return ray_vectors, ray_origins, photon_weights

@partial(jax.jit, static_argnums=(2,))
def get_isotropic_rays_random(source_position, source_intensity, Nphot, key):
    """
    Generate photons isotropically from a point source using random sampling.
    Uses 3 normal samples + normalization for uniform sphere distribution.
    """
    v = random.normal(key, (Nphot, 3))
    ray_vectors = v / jnp.linalg.norm(v, axis=1, keepdims=True)
    ray_origins = jnp.tile(source_position, (Nphot, 1))
    photon_weights = jnp.ones(Nphot) * (source_intensity / Nphot)

    return ray_vectors, ray_origins, photon_weights


@partial(jax.jit, static_argnums=(3,))
def generate_laser_photons(fiber_position, fiber_direction, source_intensity, n_photons, key, n_water=1.33, fiber_NA=0.22):
    """
    Generate laser photons from a fiber tip with realistic angular distribution.
    Photons are distributed WITHIN the cone, not just on its surface.

    Parameters
    ----------
    fiber_position : jnp.ndarray
        3D position of fiber tip (where photons originate)
    fiber_direction : jnp.ndarray
        3D unit vector of fiber pointing direction
    source_intensity : float
        Total intensity of the laser source
    n_photons : int
        Number of photons to generate
    key : jax.random.PRNGKey
        Random key for JAX
    n_water : float, optional
        Refractive index of water, default 1.33
    fiber_NA : float
        Numerical aperture of the fiber, default 0.22

    Returns
    -------
    ray_vectors : jnp.ndarray
        Array of shape (n_photons, 3) containing photon direction vectors
    ray_origins : jnp.ndarray
        Array of shape (n_photons, 3) containing photon origins (all at fiber tip)
    photon_weights : jnp.ndarray
        Array of shape (n_photons,) containing photon weights (uniform)
    """
    # Normalize fiber direction
    fiber_direction = normalize(fiber_direction)

    # Calculate maximum emission angle in water from numerical aperture
    theta_max = jnp.arcsin(fiber_NA / n_water)

    # Split keys for different random samples
    key1, key2 = jax.random.split(key)

    # Sample angles uniformly in solid angle (not uniform in theta!)
    # This gives the correct sin(theta) weighting for angles WITHIN the cone
    u = jax.random.uniform(key1, (n_photons,))
    theta = jnp.arcsin(jnp.sqrt(u) * jnp.sin(theta_max))

    # Sample azimuthal angles uniformly around fiber axis
    phi = jax.random.uniform(key2, (n_photons,)) * 2 * jnp.pi

    # Generate directions in local fiber coordinate system
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)

    local_x = sin_theta * jnp.cos(phi)
    local_y = sin_theta * jnp.sin(phi)
    local_z = cos_theta  # Along fiber axis

    # Build orthonormal basis with fiber_direction as the z-axis
    basis = generate_orthonormal_basis(fiber_direction)

    # Stack local directions
    local_directions = jnp.stack([local_x, local_y, local_z], axis=1)

    # Transform to global coordinates
    ray_vectors = jnp.einsum('ij,kj->ki', basis, local_directions)

    # All photons originate from the fiber tip position
    ray_origins = jnp.tile(fiber_position[None, :], (n_photons, 1))

    # Uniform weights
    photon_weights = source_intensity * jnp.ones(n_photons) / n_photons

    return ray_vectors, ray_origins, photon_weights


def setup_calibration_generator(source_type='isotropic'):
    """
    Factory function that returns a configured calibration photon generator.

    Parameters
    ----------
    source_type : str
        Type of calibration source: 'isotropic', 'isotropic_random', or 'laser'

    Returns
    -------
    callable
        Generator function with signature:
        (source_origin, source_intensity, Nphot, key) -> (directions, origins, weights)
    """

    if source_type == 'isotropic':
        def generator(source_origin, source_intensity, Nphot, key):
            return get_isotropic_rays(source_origin, source_intensity, Nphot, key)
        return generator

    elif source_type == 'isotropic_random':
        def generator(source_origin, source_intensity, Nphot, key):
            return get_isotropic_rays_random(source_origin, source_intensity, Nphot, key)
        return generator

    elif source_type == 'laser':
        def generator(source_origin, source_intensity, Nphot, key):
            direction = jnp.array([0., 0., -1.])
            return generate_laser_photons(
                source_origin, direction, source_intensity, Nphot, key, fiber_NA=0.22
            )
        return generator

    else:
        raise ValueError(f"Unknown source_type: {source_type}. Available: 'isotropic', 'isotropic_random', 'laser'")


def generate_random_direction(key):
    """
    Generate a random direction uniformly distributed on a unit sphere.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX

    Returns
    -------
    jnp.ndarray
        Normalized 3D vector representing a random direction
    """
    key, subkey = jax.random.split(key)
    # Generate random points on a sphere using the Marsaglia method
    while True:
        # Generate two random numbers between -1 and 1
        u1, u2 = jax.random.uniform(subkey, shape=(2,), minval=-1.0, maxval=1.0)
        s = u1**2 + u2**2
        # Reject if s is outside the unit circle
        if s < 1.0:
            break
        key, subkey = jax.random.split(key)

    # Convert to Cartesian coordinates
    x = 2 * u1 * jnp.sqrt(1 - s)
    y = 2 * u2 * jnp.sqrt(1 - s)
    z = 1 - 2 * s

    # Return normalized vector
    return normalize(jnp.array([x, y, z]))

def generate_random_vertex(key):
    """
    Generate a random vertex within the volume [-1,1]^3.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX

    Returns
    -------
    jnp.ndarray
        3D point within the volume [-1,1]^3
    """
    return jax.random.uniform(key, shape=(3,), minval=-0.1, maxval=0.1)


# ---------------------------------------------------------------------------
# Calibration source types (callable NamedTuples)
# Moved from lucid/detector_params.py during Phase 2.2 refactor.
# ---------------------------------------------------------------------------

class IsotropicSource(NamedTuple):
    """Isotropic point source -- callable JAX pytree.

    Usage: ``source(n_photons, key)`` or ``source(n_photons, key, n_water)``.
    """
    position: jnp.ndarray     # (3,)
    intensity: jnp.ndarray    # scalar

    def __call__(self, n_photons, key, n_water=1.33):
        return get_isotropic_rays(self.position, self.intensity, n_photons, key)


class LaserSource(NamedTuple):
    """Laser fibre source -- callable JAX pytree.

    Usage: ``source(n_photons, key)`` or ``source(n_photons, key, n_water)``.
    """
    position: jnp.ndarray     # (3,)
    intensity: jnp.ndarray    # scalar
    direction: jnp.ndarray    # (3,), default [0, 0, -1]
    fiber_NA: jnp.ndarray     # scalar, default 0.22

    def __call__(self, n_photons, key, n_water=1.33):
        return generate_laser_photons(
            self.position, self.direction, self.intensity,
            n_photons, key, n_water, self.fiber_NA,
        )


# --- Factory helpers with sensible defaults ---

def isotropic_source(position, intensity=1_000_000):
    """Create an IsotropicSource with default intensity."""
    return IsotropicSource(
        position=jnp.asarray(position, dtype=jnp.float32),
        intensity=jnp.asarray(float(intensity), dtype=jnp.float32),
    )


def laser_source(position, intensity=1_000_000, direction=None, fiber_NA=0.22):
    """Create a LaserSource with default direction (downward) and NA."""
    if direction is None:
        direction = [0.0, 0.0, -1.0]
    return LaserSource(
        position=jnp.asarray(position, dtype=jnp.float32),
        intensity=jnp.asarray(float(intensity), dtype=jnp.float32),
        direction=jnp.asarray(direction, dtype=jnp.float32),
        fiber_NA=jnp.asarray(float(fiber_NA), dtype=jnp.float32),
    )
