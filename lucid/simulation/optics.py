"""Pure optics functions: reflection, scattering, local frame."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from lucid.utils import normalize


def compute_reflection_direction(
    incident_dir: jax.Array,  # (3,)
    normal: jax.Array,        # (3,)
) -> jax.Array:               # (3,)
    """Compute specular reflection direction."""
    normal = normalize(normal)
    dot_product = jnp.sum(incident_dir * normal, axis=-1, keepdims=True)
    return normalize(incident_dir - 2 * dot_product * normal)


def sample_cosine_hemisphere(
    normal: jax.Array,   # (3,)
    rng_key: jax.Array,  # PRNGKeyArray
) -> jax.Array:          # (3,)
    """Cosine-weighted hemisphere sampling (reparameterisation trick)."""
    k1, k2 = jax.random.split(rng_key)
    u1 = jax.random.uniform(k1)
    u2 = jax.random.uniform(k2)

    cos_theta = jnp.sqrt(1 - u1)
    sin_theta = jnp.sqrt(u1)
    phi = 2 * jnp.pi * u2

    local_dir = jnp.array([
        sin_theta * jnp.cos(phi),
        sin_theta * jnp.sin(phi),
        cos_theta,
    ])

    frame = create_local_frame(normal)
    return normalize(frame.T @ local_dir)


def create_local_frame(
    z: jax.Array,   # (3,)
) -> jax.Array:     # (3, 3) rotation matrix
    """Create an orthonormal frame with *z* as the z-axis."""
    z = normalize(z)
    t = jnp.where(
        jnp.abs(z[0]) < 0.9,
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0]),
    )
    x = normalize(jnp.cross(t, z))
    y = jnp.cross(z, x)
    return jnp.stack([x, y, z])


def sample_scatter_distance(
    D: jax.Array,         # scalar: distance to surface
    S: jax.Array,         # scalar: scatter length
    rng_key: jax.Array,   # PRNGKeyArray
) -> jax.Array:           # scalar: sampled distance
    """Sample from a truncated exponential (scatter before surface)."""
    u = jax.random.uniform(rng_key)
    prob_term = -jnp.expm1(-D / S)
    return -S * jnp.log1p(-u * prob_term)


def solve_rayleigh_inverse_cdf(
    u: jax.Array,   # scalar uniform [0, 1]
) -> jax.Array:     # scalar cosine [-1, 1]
    """
    Solve the inverse CDF for Rayleigh scattering: P(μ) ∝ (1 + μ²)
    Uses Cardano's formula to solve: μ³ + 3μ - (8u - 4) = 0
    """
    # Transform to standard form: t³ + pt + q = 0 where μ = t
    p = 3.0
    q = -(8.0 * u - 4.0)

    # Cardano's formula
    discriminant = -(4 * p ** 3 + 27 * q ** 2)

    # Three real roots case
    sqrt_disc_pos = jnp.sqrt(jnp.abs(discriminant))
    rho = jnp.sqrt(-p ** 3 / 27)
    theta = jnp.arccos(jnp.clip(-q / (2 * rho), -1, 1))
    mu_three_roots = 2 * jnp.cbrt(rho) * jnp.cos(theta / 3)

    # One real root case
    sqrt_disc_neg = jnp.sqrt(-discriminant)
    A = jnp.cbrt((-q + sqrt_disc_neg / (3 * jnp.sqrt(3))) / 2)
    B = jnp.cbrt((-q - sqrt_disc_neg / (3 * jnp.sqrt(3))) / 2)
    mu_one_root = A + B

    # Select based on discriminant sign
    mu = jnp.where(discriminant >= 0, mu_three_roots, mu_one_root)
    return jnp.clip(mu, -1.0, 1.0)


def compute_scatter_direction(
    incident_dir: jax.Array,  # (3,)
    rng_key: jax.Array,       # PRNGKeyArray
) -> jax.Array:               # (3,)
    """Rayleigh-phase-function scattering direction."""
    k1, k2 = jax.random.split(rng_key)
    u1 = jax.random.uniform(k1)
    u2 = jax.random.uniform(k2)

    cos_theta = solve_rayleigh_inverse_cdf(u1)
    sin_theta = jnp.sqrt(1 - cos_theta ** 2)
    phi = 2 * jnp.pi * u2
    local_dir = normalize(jnp.array([
        sin_theta * jnp.cos(phi),
        sin_theta * jnp.sin(phi),
        cos_theta,
    ]))
    frame = create_local_frame(incident_dir)
    return normalize(frame @ local_dir)


def jax_normalize(
    v: jax.Array,          # (N,) or (3,)
    epsilon: float = 1e-8,
) -> jax.Array:            # same shape as v
    """Normalize a vector with numerical stability using JAX."""
    norm = jnp.linalg.norm(v)
    return jnp.where(norm > epsilon, v / norm, v)


from lucid.utils import jax_rotate_vector  # noqa: F811 — canonical location

