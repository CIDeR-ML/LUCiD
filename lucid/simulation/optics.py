"""Pure optics functions: normalize, reflection, scattering, local frame."""
import jax
import jax.numpy as jnp


def normalize(v, epsilon=1e-6):
    """Normalize a vector (or batch of vectors)."""
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.maximum(norm, epsilon)


def compute_reflection_direction(incident_dir, normal):
    """Compute specular reflection direction."""
    normal = normalize(normal)
    dot_product = jnp.sum(incident_dir * normal, axis=-1, keepdims=True)
    return normalize(incident_dir - 2 * dot_product * normal)


def sample_cosine_hemisphere(normal, rng_key):
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


def create_local_frame(z):
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


def sample_scatter_distance(D, S, rng_key):
    """Sample from a truncated exponential (scatter before surface)."""
    u = jax.random.uniform(rng_key)
    prob_term = -jnp.expm1(-D / S)
    return -S * jnp.log1p(-u * prob_term)


def solve_rayleigh_inverse_cdf(u):
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


def compute_scatter_direction(incident_dir, rng_key):
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


# JAX-compatible rotation helpers (for data mode)
def jax_normalize(v, epsilon=1e-8):
    """Normalize a vector with numerical stability using JAX."""
    norm = jnp.linalg.norm(v)
    return jnp.where(norm > epsilon, v / norm, v)


def jax_rotate_vector(vector, axis, angle):
    """Rotate a vector around an axis by a given angle in radians using JAX."""
    axis = jax_normalize(axis)
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    cross_product = jnp.cross(axis, vector)
    dot_product = jnp.dot(axis, vector) * (1 - cos_angle)
    return cos_angle * vector + sin_angle * cross_product + dot_product * axis

