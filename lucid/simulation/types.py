"""Pipeline NamedTuples for the simulation loop.

These replace raw dicts, positional tuples, and ad-hoc returns with
structured, named containers. All are registered as JAX pytrees
(NamedTuples are pytrees by default in JAX).
"""
from __future__ import annotations

from typing import NamedTuple, Optional
import jax
import jax.numpy as jnp


class PhotonRays(NamedTuple):
    """Output of ray generation (SIREN or calibration sources).

    ``wavelengths`` is ``None`` in monochromatic mode. When wavelength
    support is active, the simulator populates it after ray generation.
    """
    directions: jax.Array         # (n_photons, 3)
    origins: jax.Array            # (n_photons, 3)
    weights: jax.Array            # (n_photons,)
    wavelengths: Optional[jax.Array] = None  # (n_photons,) nm, or None


class PropagationResult(NamedTuple):
    """Output of the photon propagator (ray-geometry intersection).

    Replaces the dict with keys: sensor_weights, sensor_indices, times,
    positions, normals, inside_sensor.
    """
    sensor_weights: jax.Array     # (max_sensors, n_rays)
    sensor_indices: jax.Array     # (max_sensors, n_rays) int
    times: jax.Array              # (max_sensors, n_rays) meters
    positions: jax.Array          # (n_rays, 3)
    normals: jax.Array            # (n_rays, 3)
    inside_sensor: jax.Array      # (max_sensors, n_rays) bool


class PhotonStepResult(NamedTuple):
    """Output of one photon iteration step (sample or update_factors).

    Replaces the 6-tuple: (new_pos, new_dir, new_time, detect_prob,
    reflection_attenuation, continuing_factor).
    """
    position: jax.Array           # (3,)
    direction: jax.Array          # (3,)
    time: float
    detect_prob: float
    reflection_attenuation: float
    continuing_factor: float


class PhotonState(NamedTuple):
    """Carry state for the jax.lax.scan propagation loop.

    Replaces the 5-tuple: (positions, directions, times, survival, key).
    """
    positions: jax.Array          # (n_rays, 3)
    directions: jax.Array         # (n_rays, 3)
    times: jax.Array              # (n_rays,)
    survival: jax.Array           # (n_rays,)
    key: jax.Array                # PRNGKey
