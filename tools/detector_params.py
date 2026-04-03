"""
Pytree parameter types for LUCiD / dC simulation framework.

Provides NamedTuple-based parameter containers that work natively with JAX
pytree operations (jax.tree_map, value_and_grad, optax, etc.), replacing
fragile index-based tuple access with named fields.

Types:
    DetectorParams  — detector calibration parameters
    ParticleParams  — particle track parameters
    IsotropicSource — callable isotropic calibration source
    LaserSource     — callable laser calibration source

Helpers:
    normalize / denormalize      — map pytrees to/from [0,1]
    default_bounds               — physical bounds for DetectorParams
    make_optimization_mask       — boolean pytree for optax.masked
    create_default_detector_params / create_default_particle_params
    default_gradient_scales      — per-field gradient scaling
    save_detector_params / load_detector_params
    save_particle_params / load_particle_params
"""

from typing import NamedTuple
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from tools.utils import spherical_to_cartesian
from tools.generate import get_isotropic_rays, generate_laser_photons


# ---------------------------------------------------------------------------
# DetectorParams
# ---------------------------------------------------------------------------

class DetectorParams(NamedTuple):
    """Detector calibration parameters (JAX pytree).

    Fields
    ------
    scatter_length : jnp.ndarray        scalar, symmetric (Rayleigh) scattering MFP, meters
    asym_scatter_length : jnp.ndarray   scalar, asymmetric (Mie) scattering MFP, meters
    mie_g : jnp.ndarray                scalar, HG asymmetry parameter (-1, 1)
    wall_reflection_rate : jnp.ndarray   scalar, [0, 1]
    sensor_reflection_rate : jnp.ndarray scalar, [0, 1]
    absorption_length : jnp.ndarray      scalar, meters
    qe : jnp.ndarray                    scalar, base quantum efficiency [0, 1]
    qe_corrections : jnp.ndarray        shape (num_sensors,), per-sensor QE multipliers
    """
    scatter_length: jnp.ndarray
    asym_scatter_length: jnp.ndarray
    mie_g: jnp.ndarray
    wall_reflection_rate: jnp.ndarray
    sensor_reflection_rate: jnp.ndarray
    absorption_length: jnp.ndarray
    qe: jnp.ndarray
    qe_corrections: jnp.ndarray


# ---------------------------------------------------------------------------
# ParticleParams
# ---------------------------------------------------------------------------

class ParticleParams(NamedTuple):
    """Particle track parameters (JAX pytree).

    Fields
    ------
    energy : jnp.ndarray    scalar, MeV
    position : jnp.ndarray  (3,), meters
    theta : jnp.ndarray     scalar, polar angle radians
    phi : jnp.ndarray       scalar, azimuthal angle radians
    t0 : jnp.ndarray        scalar, vertex time offset (ns)
    """
    energy: jnp.ndarray
    position: jnp.ndarray
    theta: jnp.ndarray
    phi: jnp.ndarray
    t0: jnp.ndarray

    @classmethod
    def from_cartesian(cls, energy, position, direction, t0=0.0):
        """Create from Cartesian direction vector (e.g. data mode)."""
        direction = jnp.asarray(direction, dtype=jnp.float32)
        norm = jnp.linalg.norm(direction)
        theta = jnp.arccos(jnp.clip(direction[2] / norm, -1.0, 1.0))
        phi = jnp.arctan2(direction[1], direction[0])
        return cls(
            energy=jnp.asarray(energy, dtype=jnp.float32),
            position=jnp.asarray(position, dtype=jnp.float32),
            theta=theta,
            phi=phi,
            t0=jnp.asarray(float(t0), dtype=jnp.float32),
        )

    @property
    def direction(self):
        """Cartesian direction unit vector derived from (theta, phi)."""
        return spherical_to_cartesian(self.theta, self.phi)


# ---------------------------------------------------------------------------
# Calibration source types (callable NamedTuples)
# ---------------------------------------------------------------------------

class IsotropicSource(NamedTuple):
    """Isotropic point source — callable JAX pytree.

    Usage: ``source(n_photons, key)`` or ``source(n_photons, key, n_water)``.
    """
    position: jnp.ndarray     # (3,)
    intensity: jnp.ndarray    # scalar

    def __call__(self, n_photons, key, n_water=1.33):
        return get_isotropic_rays(self.position, self.intensity, n_photons, key)


class LaserSource(NamedTuple):
    """Laser fibre source — callable JAX pytree.

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


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

_ARRAY_SENTINEL = "__array__:"


def save_detector_params(params: DetectorParams, filepath: str):
    """Save DetectorParams to JSON + companion .npy files for arrays.

    Scalars are stored directly in JSON.  Arrays whose size > 1 are saved to
    ``<filepath_stem>_<field>.npy`` and referenced in JSON via the sentinel
    ``"__array__:<filename>"`` convention.
    """
    dirpath = os.path.dirname(filepath) or "."
    stem = os.path.splitext(os.path.basename(filepath))[0]
    data = {}
    for field in DetectorParams._fields:
        val = getattr(params, field)
        arr = np.asarray(val)
        if arr.ndim == 0:
            data[field] = float(arr)
        else:
            npy_name = f"{stem}_{field}.npy"
            np.save(os.path.join(dirpath, npy_name), arr)
            data[field] = f"{_ARRAY_SENTINEL}{npy_name}"
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_detector_params(filepath: str, num_sensors: int = None) -> DetectorParams:
    """Load DetectorParams from JSON.

    Supports four value formats per field:
      - ``null``   → ``jnp.asarray(1.0)`` (scalar placeholder)
      - scalar (number)  → ``jnp.asarray(float)``
      - list             → ``jnp.asarray(list)``  (inline array)
      - ``"__array__:filename.npy"`` sentinel → loaded from companion ``.npy``

    If *num_sensors* is given, scalar ``qe_corrections`` are automatically
    expanded to ``jnp.ones(num_sensors) * value``.
    """
    dirpath = os.path.dirname(filepath) or "."
    with open(filepath) as f:
        data = json.load(f)
    kwargs = {}
    for field in DetectorParams._fields:
        val = data.get(field, None)
        # Backward compatibility: old configs without asym_scatter_length/mie_g
        # default to values that effectively disable Mie scattering
        if val is None and field == 'asym_scatter_length':
            kwargs[field] = jnp.asarray(1e6)
        elif val is None and field == 'mie_g':
            kwargs[field] = jnp.asarray(0.95)
        elif val is None:
            kwargs[field] = jnp.asarray(1.0)
        elif isinstance(val, str) and val.startswith(_ARRAY_SENTINEL):
            npy_name = val[len(_ARRAY_SENTINEL):]
            arr = np.load(os.path.join(dirpath, npy_name))
            kwargs[field] = jnp.asarray(arr)
        elif isinstance(val, list):
            kwargs[field] = jnp.asarray(val, dtype=jnp.float32)
        else:
            kwargs[field] = jnp.asarray(float(val))

    # Auto-expand scalar qe_corrections when num_sensors is known
    if num_sensors is not None:
        qe_corr = kwargs['qe_corrections']
        if qe_corr.ndim == 0:
            kwargs['qe_corrections'] = jnp.ones(num_sensors) * qe_corr

    return DetectorParams(**kwargs)


def save_particle_params(params: ParticleParams, filepath: str):
    """Save ParticleParams to JSON."""
    data = {}
    for field in ParticleParams._fields:
        val = np.asarray(getattr(params, field))
        if val.ndim == 0:
            data[field] = float(val)
        else:
            data[field] = val.tolist()
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_particle_params(filepath: str) -> ParticleParams:
    """Load ParticleParams from JSON."""
    with open(filepath) as f:
        data = json.load(f)
    kwargs = {}
    for field in ParticleParams._fields:
        val = data[field]
        if isinstance(val, list):
            kwargs[field] = jnp.asarray(val, dtype=jnp.float32)
        else:
            kwargs[field] = jnp.asarray(float(val), dtype=jnp.float32)
    return ParticleParams(**kwargs)


# ---------------------------------------------------------------------------
# Optimization helpers
# ---------------------------------------------------------------------------

def normalize_params(params, bounds_min, bounds_max):
    """Map a pytree from physical units to [0, 1] using element-wise bounds."""
    return jax.tree.map(
        lambda v, lo, hi: (v - lo) / (hi - lo + 1e-8),
        params, bounds_min, bounds_max,
    )


def denormalize_params(normalized, bounds_min, bounds_max):
    """Map a pytree from [0, 1] back to physical units."""
    return jax.tree.map(
        lambda v, lo, hi: v * (hi - lo) + lo,
        normalized, bounds_min, bounds_max,
    )


def default_bounds(num_sensors: int):
    """Return ``(bounds_min, bounds_max)`` DetectorParams with physical ranges.

    Returns
    -------
    bounds_min, bounds_max : DetectorParams
    """
    bounds_min = DetectorParams(
        scatter_length=jnp.array(0.0),
        asym_scatter_length=jnp.array(0.0),
        mie_g=jnp.array(0.0),
        wall_reflection_rate=jnp.array(0.0),
        sensor_reflection_rate=jnp.array(0.0),
        absorption_length=jnp.array(0.0),
        qe=jnp.array(0.0),
        qe_corrections=jnp.zeros(num_sensors),
    )
    bounds_max = DetectorParams(
        scatter_length=jnp.array(100.0),
        asym_scatter_length=jnp.array(50000.0),
        mie_g=jnp.array(0.999),
        wall_reflection_rate=jnp.array(0.5),
        sensor_reflection_rate=jnp.array(0.4),
        absorption_length=jnp.array(500.0),
        qe=jnp.array(1.0),
        qe_corrections=jnp.full(num_sensors, 2.0),
    )
    return bounds_min, bounds_max


def make_optimization_mask(params, trainable_fields):
    """Create a boolean pytree selecting which fields to train.

    Parameters
    ----------
    params : NamedTuple
        The parameter pytree (DetectorParams or ParticleParams).
    trainable_fields : set[str]
        Field names that should receive gradient updates.

    Returns
    -------
    mask : NamedTuple (same type as *params*)
        Leaves are ``True`` (scalar or array-shaped) where the field is
        trainable, ``False`` otherwise.  Suitable for ``optax.masked``.
    """
    mask_dict = {}
    for field in params._fields:
        val = getattr(params, field)
        if field in trainable_fields:
            mask_dict[field] = jax.tree.map(lambda x: True, val)
        else:
            mask_dict[field] = jax.tree.map(lambda x: False, val)
    return type(params)(**mask_dict)


def create_default_detector_params(num_sensors: int) -> DetectorParams:
    """Sensible initialization defaults for calibration optimization."""
    return DetectorParams(
        scatter_length=jnp.array(50.0),
        asym_scatter_length=jnp.array(10000.0),
        mie_g=jnp.array(0.95),
        wall_reflection_rate=jnp.array(0.2),
        sensor_reflection_rate=jnp.array(0.2),
        absorption_length=jnp.array(150.0),
        qe=jnp.array(0.2),
        qe_corrections=jnp.ones(num_sensors),
    )


def create_default_particle_params() -> ParticleParams:
    """Sensible initialization defaults for track reconstruction."""
    return ParticleParams(
        energy=jnp.array(500.0),
        position=jnp.zeros(3),
        theta=jnp.array(jnp.pi / 2),
        phi=jnp.array(0.0),
        t0=jnp.array(0.0),
    )


def default_gradient_scales(num_sensors: int) -> DetectorParams:
    """Per-field gradient scaling factors.

    Useful for applying different effective learning rates to each parameter
    group::

        scaled_grads = jax.tree.map(lambda g, s: g * s, grads, scales)
    """
    return DetectorParams(
        scatter_length=jnp.array(1.0),
        asym_scatter_length=jnp.array(1.0),
        mie_g=jnp.array(1.0),
        wall_reflection_rate=jnp.array(1.0),
        sensor_reflection_rate=jnp.array(1.0),
        absorption_length=jnp.array(1.0),
        qe=jnp.array(1.0),
        qe_corrections=jnp.full(num_sensors, 0.1),
    )