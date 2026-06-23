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

    7-tuple: (new_pos, new_dir, new_time, detect_prob,
    reflection_attenuation, continuing_factor, logp_increment).

    ``logp_increment`` is the per-photon DiCE score increment for this step
    (``lf + la`` for the differentiable step; ``0.0`` for the sampling step).
    The scan body accumulates it into ``PhotonState.log_p``.
    """
    position: jnp.ndarray           # (3,)
    direction: jnp.ndarray          # (3,)
    time: float
    detect_prob: float
    reflection_attenuation: float
    continuing_factor: float
    logp_increment: float


class PhotonState(NamedTuple):
    """Carry state for the jax.lax.scan propagation loop.

    7-tuple: (positions, directions, times, survival, key, log_p, medium_id).

    ``log_p`` is the per-photon accumulated DiCE score (sum of ``lf + la`` over
    prior steps); the implicit-capture deposit reads the PRE-step ``log_p``.

    ``medium_id`` is the per-photon current medium index for nested two-medium
    detectors (0 = inner, 1 = outer). For single-medium detectors it is an all-zero
    carried array that no computation reads → the forward stays byte-identical.
    Defaults to ``None`` so legacy 6-field positional construction still works
    (the scan body always supplies it explicitly).
    """
    positions: jnp.ndarray          # (n_rays, 3)
    directions: jnp.ndarray         # (n_rays, 3)
    times: jnp.ndarray              # (n_rays,)
    survival: jnp.ndarray           # (n_rays,)
    key: jnp.ndarray                # PRNGKey
    log_p: jnp.ndarray              # (n_rays,)
    medium_id: Optional[jnp.ndarray] = None   # (n_rays,) int, nested only
