"""
Parametric cascade (shower) photon emitter for neutrino telescopes.

Generates Cherenkov photons from a point-like electromagnetic or hadronic
cascade without requiring a trained SIREN model. Suitable for testing the
string-telescope propagation pipeline at IceCube energy scales (1 GeV – 10 PeV).

Physics model:
  - Cascade is a point source at the interaction vertex
  - Emission along a short shower length L ≈ X0 * ln(E/Ec)
  - Photon directions drawn from a narrow cone at the Cherenkov angle θ_C
  - Intrinsic angular spread σ ≈ 1/sqrt(E/Ec) (EM) or ~5° (hadronic)
  - Total photon yield ∝ energy
  - Per-photon weight = total_yield / n_sampled

This is approximate — it captures the leading-order geometry (Cherenkov cone
from a short source) and lets the propagation loop handle scattering. Not
suitable for precision physics; suitable for testing propagator integration,
hit patterns, and reconstruction pipelines.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


# Cherenkov yield: ~340 photons per cm per MeV of track in water at 400nm
# Integrated over Cherenkov spectrum (300-600nm): ~24000 photons per cm
# For a shower depositing energy E: total ~ 24000 * L_shower(cm) * (E_deposited / E_shower_total)
# Simplified: ~250 photons per MeV deposited (rough, after integration)
PHOTONS_PER_MEV = 250.0

# Radiation length
X0_WATER = 0.361  # meters (36.1 cm)
X0_ICE = 0.393    # meters

# Critical energy
EC_WATER = 78.6   # MeV (electrons in water)
EC_ICE = 73.0     # MeV (electrons in ice)


def cherenkov_angle(n_medium):
    """Cherenkov angle for a relativistic particle (β ≈ 1) in medium with index n."""
    return jnp.arccos(1.0 / n_medium)


def shower_length(energy_mev, x0=X0_WATER, ec=EC_WATER):
    """Approximate EM shower length in meters.

    L ≈ X0 * ln(E / Ec) for E >> Ec.
    """
    return x0 * jnp.maximum(jnp.log(energy_mev / ec), 1.0)


def intrinsic_spread(energy_mev, ec=EC_WATER, is_hadronic=False):
    """Angular spread of shower particles around shower axis (radians).

    EM: σ ≈ 1/sqrt(E/Ec), typically < 1° at high energy.
    Hadronic: broader, ~5° from pion production kinematics.
    """
    em_spread = 1.0 / jnp.sqrt(jnp.maximum(energy_mev / ec, 1.0))
    hadronic_spread = 0.09  # ~5 degrees
    return jnp.where(is_hadronic, hadronic_spread, em_spread)


def generate_cascade_photons(
    vertex,
    direction,
    energy_mev,
    n_photons,
    key,
    n_medium=1.33,
    x0=X0_WATER,
    ec=EC_WATER,
    is_hadronic=False,
):
    """Generate Cherenkov photons from a cascade.

    Parameters
    ----------
    vertex : (3,)          cascade interaction vertex (meters)
    direction : (3,)       cascade direction (unit vector)
    energy_mev : float     cascade energy in MeV
    n_photons : int        number of photon samples to generate
    key : jax PRNGKey
    n_medium : float       refractive index of the medium
    x0 : float             radiation length (meters)
    ec : float             critical energy (MeV)
    is_hadronic : bool     hadronic cascade (broader angular spread)

    Returns
    -------
    origins : (n_photons, 3)       photon emission positions along shower
    directions : (n_photons, 3)    photon directions (Cherenkov cone)
    weights : (n_photons,)         per-photon intensity weights
    """
    direction = direction / (jnp.linalg.norm(direction) + 1e-30)

    # Total photon yield
    total_photons = PHOTONS_PER_MEV * energy_mev
    per_photon_weight = total_photons / n_photons

    # Shower geometry
    L = shower_length(energy_mev, x0, ec)
    theta_c = cherenkov_angle(n_medium)
    sigma = intrinsic_spread(energy_mev, ec, is_hadronic)

    # Split keys
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Emission positions: uniform along shower axis
    s = jax.random.uniform(k1, (n_photons,)) * L
    origins = vertex[None, :] + s[:, None] * direction[None, :]

    # Photon directions: cone at Cherenkov angle with small gaussian spread
    # Sample polar angle θ around shower axis from N(θ_C, σ)
    theta = theta_c + sigma * jax.random.normal(k2, (n_photons,))
    theta = jnp.clip(theta, 0.0, jnp.pi)

    # Sample azimuthal angle φ uniformly [0, 2π)
    phi = jax.random.uniform(k3, (n_photons,), minval=0.0, maxval=2.0 * jnp.pi)

    # Convert (θ, φ) in shower-local frame to world frame
    directions = _cone_to_world(direction, theta, phi)

    weights = jnp.full(n_photons, per_photon_weight)

    return origins, directions, weights


def _cone_to_world(axis, theta, phi):
    """Convert (θ, φ) in a cone around `axis` to world-frame unit vectors.

    Parameters
    ----------
    axis : (3,)            cone axis (unit vector)
    theta : (n,)           polar angles from axis
    phi : (n,)             azimuthal angles

    Returns
    -------
    dirs : (n, 3)          unit vectors in world frame
    """
    # Build orthonormal frame: axis, u, v
    # Pick a vector not parallel to axis
    ref = jnp.where(jnp.abs(axis[0]) < 0.9,
                    jnp.array([1.0, 0.0, 0.0]),
                    jnp.array([0.0, 1.0, 0.0]))
    u = jnp.cross(axis, ref)
    u = u / (jnp.linalg.norm(u) + 1e-30)
    v = jnp.cross(axis, u)

    # Spherical to Cartesian in local frame, then rotate to world
    sin_t = jnp.sin(theta)
    cos_t = jnp.cos(theta)
    sin_p = jnp.sin(phi)
    cos_p = jnp.cos(phi)

    dirs = (cos_t[:, None] * axis[None, :] +
            sin_t[:, None] * cos_p[:, None] * u[None, :] +
            sin_t[:, None] * sin_p[:, None] * v[None, :])

    norms = jnp.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / (norms + 1e-30)
