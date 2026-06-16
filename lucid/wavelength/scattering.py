"""Wavelength-dependent scattering phase functions.

Rayleigh (symmetric) and Henyey-Greenstein / Mie (asymmetric) scattering
direction samplers. These are ported from the ``wavelength_dependency``
branch and available for external use. They are NOT currently called in the
main simulation propagation loop (which uses the Rayleigh sampler from
``lucid.simulation.optics``).
"""
import jax
import jax.numpy as jnp
from lucid.utils import normalize
# create_local_frame / solve_rayleigh_inverse_cdf are imported lazily inside the
# functions below (not at module top) to avoid a circular import: importing
# lucid.simulation.optics runs lucid/simulation/__init__ -> simulator -> photon_step,
# and photon_step imports back from this module. A module-top import here breaks
# `import lucid.wavelength.scattering` when it is the first import in a process.


def compute_rayleigh_scatter_direction(incident_dir, rng_key):
    """Rayleigh phase function scattering: P(cos_theta) ~ (1 + cos^2 theta).

    This is functionally identical to ``lucid.simulation.optics.compute_scatter_direction``
    but kept here for completeness alongside the Mie sampler.

    Parameters
    ----------
    incident_dir : jnp.ndarray
        (3,) current photon direction (unit vector).
    rng_key : jax.random.PRNGKey

    Returns
    -------
    jnp.ndarray
        (3,) scattered direction (unit vector).
    """
    from lucid.simulation.optics import create_local_frame, solve_rayleigh_inverse_cdf
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


def hg_sample_cos_theta(u, g):
    """Sample cos(theta) from the Henyey-Greenstein phase function.

    P(cos_theta) = (1 - g^2) / (2 * (1 + g^2 - 2*g*cos_theta)^(3/2))

    Uses clamping of g to ensure NaN-free gradients for JAX jit/grad/vmap.

    Parameters
    ----------
    u : float
        Uniform random sample in [0, 1].
    g : float
        Asymmetry parameter in (-1, 1).
        g > 0: forward-peaked, g = 0: isotropic, g < 0: backward-peaked.

    Returns
    -------
    float
        Sampled cosine of the scattering angle, in [-1, 1].
    """
    g_safe = jnp.clip(g, 1e-4, 1.0 - 1e-4)
    term = (1.0 - g_safe**2) / (1.0 - g_safe + 2.0 * g_safe * u)
    cos_theta = (1.0 + g_safe**2 - term**2) / (2.0 * g_safe)
    return jnp.clip(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)


def hg_logpdf(mu, g):
    """log of the Henyey-Greenstein phase function at cos(angle) ``mu``.

    log P(μ) = log(1-g²) - log 2 - 1.5·log(1 + g² - 2·g·μ).
    This is the DiCE score density for the Mie scatter angle (carries the ``g``
    gradient). ``g`` clamped for NaN-free grads. (mu is the SAMPLED cosine, so the
    caller passes ``stop_gradient(mu)``.)
    """
    g = jnp.clip(g, 1e-4, 1.0 - 1e-4)
    return jnp.log(1.0 - g**2) - jnp.log(2.0) - 1.5 * jnp.log(1.0 + g**2 - 2.0 * g * mu)


def rayleigh_logpdf(mu):
    """log of the Rayleigh phase function at cos(angle) ``mu``: log(3/8) + log(1+μ²).

    The DiCE score density for the Rayleigh scatter angle (no free parameter; constant
    in the optimised params, but kept for symmetry with ``hg_logpdf`` in the score).
    """
    return jnp.log(3.0 / 8.0) + jnp.log(1.0 + mu**2)


def compute_mie_scatter_direction(incident_dir, rng_key, g=0.95):
    """Mie (Henyey-Greenstein) scattering direction.

    Parameters
    ----------
    incident_dir : jnp.ndarray
        (3,) current photon direction (unit vector).
    rng_key : jax.random.PRNGKey
    g : float
        HG asymmetry parameter. Default 0.95 (strongly forward-peaked,
        consistent with SK water Mie scattering).

    Returns
    -------
    jnp.ndarray
        (3,) scattered direction (unit vector).
    """
    from lucid.simulation.optics import create_local_frame
    k1, k2 = jax.random.split(rng_key)
    u1 = jax.random.uniform(k1)
    u2 = jax.random.uniform(k2)

    cos_theta = hg_sample_cos_theta(u1, g)
    sin_theta = jnp.sqrt(jnp.maximum(1.0 - cos_theta ** 2, 0.0))
    phi = 2 * jnp.pi * u2
    local_dir = normalize(jnp.array([
        sin_theta * jnp.cos(phi),
        sin_theta * jnp.sin(phi),
        cos_theta,
    ]))
    frame = create_local_frame(incident_dir)
    return normalize(frame @ local_dir)
