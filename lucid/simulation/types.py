"""Pipeline NamedTuples for the simulation loop.

These replace raw dicts, positional tuples, and ad-hoc returns with
structured, named containers. All are registered as JAX pytrees
(NamedTuples are pytrees by default in JAX).
"""
from typing import NamedTuple, Optional
import jax.numpy as jnp


class PhotonRays(NamedTuple):
    """Output of ray generation (SIREN or calibration sources).

    ``wavelengths`` is ``None`` in monochromatic mode. When wavelength
    support is active, the simulator populates it after ray generation.
    """
    directions: jnp.ndarray         # (n_photons, 3)
    origins: jnp.ndarray            # (n_photons, 3)
    weights: jnp.ndarray            # (n_photons,)
    wavelengths: Optional[jnp.ndarray] = None  # (n_photons,) nm, or None


class PropagationResult(NamedTuple):
    """Output of the photon propagator (ray-geometry intersection).

    Replaces the dict with keys: sensor_weights, sensor_indices, times,
    positions, normals, inside_sensor.
    """
    sensor_weights: jnp.ndarray     # (max_sensors, n_rays)
    sensor_indices: jnp.ndarray     # (max_sensors, n_rays) int
    times: jnp.ndarray              # (max_sensors, n_rays) meters
    positions: jnp.ndarray          # (n_rays, 3)
    normals: jnp.ndarray            # (n_rays, 3)
    inside_sensor: jnp.ndarray      # (max_sensors, n_rays) bool


class PhotonStepResult(NamedTuple):
    """Output of one photon iteration step (sample or update_factors).

    Replaces the 6-tuple: (new_pos, new_dir, new_time, detect_prob,
    reflection_attenuation, continuing_factor).
    """
    position: jnp.ndarray           # (3,)
    direction: jnp.ndarray          # (3,)
    time: float
    detect_prob: float
    reflection_attenuation: float
    continuing_factor: float


class PhotonState(NamedTuple):
    """Carry state for the jax.lax.scan propagation loop.

    Replaces the 5-tuple: (positions, directions, times, survival, key).
    """
    positions: jnp.ndarray          # (n_rays, 3)
    directions: jnp.ndarray         # (n_rays, 3)
    times: jnp.ndarray              # (n_rays,)
    survival: jnp.ndarray           # (n_rays,)
    key: jnp.ndarray                # PRNGKey
