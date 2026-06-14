"""
Parametric Cherenkov track (muon) photon emitter for neutrino telescopes.

Generates Cherenkov photons along a straight muon track. Each photon is
emitted at a random position along the track, in a random azimuthal
direction on the Cherenkov cone.

For relativistic muons: dE/dx ≈ 2 MeV/cm in water (minimum ionizing).
Cherenkov yield ≈ 250 photons per MeV deposited (integrated over spectrum).
So photon yield ≈ 500 photons per cm of track, or 50000 per meter.

The track can extend beyond the detector envelope — photons emitted outside
the envelope are still propagated (they may scatter into the detector).
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple

from lucid.sources.cascade import _cone_to_world, cherenkov_angle

PHOTONS_PER_METER = 50000.0
DEDX_MEV_PER_M = 200.0  # ~2 MeV/cm for minimum ionizing muon in water


def track_length_from_energy(energy_mev, dedx=DEDX_MEV_PER_M):
    """Approximate track length for a muon depositing its full energy."""
    return energy_mev / dedx


def generate_track_photons(
    vertex,
    direction,
    track_length,
    n_photons,
    key,
    n_medium=1.33,
):
    """Generate Cherenkov photons along a straight muon track.

    Parameters
    ----------
    vertex : (3,)          track start point (meters)
    direction : (3,)       track direction (unit vector)
    track_length : float   track length (meters)
    n_photons : int        number of photon samples
    key : jax PRNGKey
    n_medium : float       refractive index

    Returns
    -------
    origins : (n_photons, 3)
    directions : (n_photons, 3)
    weights : (n_photons,)
    """
    direction = direction / (jnp.linalg.norm(direction) + 1e-30)

    total_photons = PHOTONS_PER_METER * track_length
    per_photon_weight = total_photons / n_photons

    theta_c = cherenkov_angle(n_medium)

    k1, k2, k3 = jax.random.split(key, 3)

    s = jax.random.uniform(k1, (n_photons,)) * track_length
    origins = vertex[None, :] + s[:, None] * direction[None, :]

    phi = jax.random.uniform(k2, (n_photons,)) * 2.0 * jnp.pi
    theta = jnp.full(n_photons, theta_c)
    directions = _cone_to_world(direction, theta, phi)

    weights = jnp.full(n_photons, per_photon_weight)

    return origins, directions, weights


class TrackSource(NamedTuple):
    """Parametric Cherenkov track source — callable JAX pytree."""
    position: jnp.ndarray       # (3,) track start vertex
    direction: jnp.ndarray      # (3,) track direction
    track_length: jnp.ndarray   # scalar, meters
    intensity: jnp.ndarray      # scalar (total photon yield)
    n_medium: jnp.ndarray       # scalar, refractive index
    wavelength: object = None

    def __call__(self, n_photons, key, n_water=1.33):
        origins, directions, weights = generate_track_photons(
            self.position, self.direction, self.track_length,
            n_photons, key, n_medium=self.n_medium,
        )
        return directions, origins, weights


def track_source(position, direction, energy_mev=None, track_length=None,
                 n_medium=1.33, wavelength=None):
    """Create a TrackSource for neutrino telescope simulations.

    Specify either energy_mev (track length derived from dE/dx) or
    track_length directly.

    Parameters
    ----------
    position : (3,)       track start vertex
    direction : (3,)      track direction (unit vector)
    energy_mev : float    muon energy in MeV (derives track length)
    track_length : float  track length in meters (overrides energy)
    n_medium : float      refractive index
    wavelength : float or None
    """
    if track_length is None and energy_mev is None:
        raise ValueError("specify either energy_mev or track_length")
    if track_length is None:
        track_length = track_length_from_energy(energy_mev)

    total_yield = PHOTONS_PER_METER * track_length
    wl = jnp.asarray(float(wavelength), dtype=jnp.float32) if wavelength is not None else None
    return TrackSource(
        position=jnp.asarray(position, dtype=jnp.float32),
        direction=jnp.asarray(direction, dtype=jnp.float32),
        track_length=jnp.asarray(float(track_length), dtype=jnp.float32),
        intensity=jnp.asarray(float(total_yield), dtype=jnp.float32),
        n_medium=jnp.asarray(float(n_medium), dtype=jnp.float32),
        wavelength=wl,
    )
