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

from __future__ import annotations

from typing import Any, NamedTuple
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from lucid.utils import spherical_to_cartesian


class DetectorParams(NamedTuple):
    """Detector calibration parameters (JAX pytree).

    Bulk-optical + sensor fields
    ----------------------------
    scatter_length : jnp.ndarray         scalar, meters
    wall_reflection_rate : jnp.ndarray   scalar, [0, 1]
    sensor_reflection_rate : jnp.ndarray scalar, [0, 1]
    absorption_length : jnp.ndarray      scalar, meters
    qe : jnp.ndarray                     scalar, base quantum efficiency [0, 1]
    qe_corrections : jnp.ndarray         shape (num_sensors,), per-sensor QE multipliers

    Scintillation fields (only meaningful when the closed-over medium has
    ``"scintillation"`` in its ``emission_processes``; left as ``NaN`` for
    non-scintillating detectors and never read in that case)
    -----------------------------------------------------------------------
    S, kB, C : jnp.ndarray               Chou light yield. dL/dx = S * (dE/dx) /
                                          (1 + kB*(dE/dx) + C*(dE/dx)²).
                                          Units: S [ph/MeV], kB [mm/keV],
                                          C [(mm/keV)²]. Set C=0 for Birks-only.
    tau_r, tau_1, tau_2, R_1 : jnp.ndarray  Mixture biexponential timing.
                                          p(t) = R_1 g(t; tau_r, tau_1) +
                                          (1-R_1) g(t; tau_r, tau_2), tau in ns.
    moyal_amp, moyal_loc, moyal_scale : jnp.ndarray
                                          Emission-spectrum shape — used
                                          inside the medium's wavelength
                                          window (medium.scintillation_lambda_*).
    """
    scatter_length: jax.Array
    wall_reflection_rate: jax.Array
    sensor_reflection_rate: jax.Array
    absorption_length: jax.Array
    qe: jax.Array
    qe_corrections: jax.Array
    # Scintillation — light yield (Chou)
    S:  jax.Array = jnp.nan
    kB: jax.Array = jnp.nan
    C:  jax.Array = jnp.nan
    # Scintillation — biexponential timing
    tau_r: jax.Array = jnp.nan
    tau_1: jax.Array = jnp.nan
    tau_2: jax.Array = jnp.nan
    R_1:   jax.Array = jnp.nan
    # Scintillation — Moyal emission spectrum
    moyal_amp:   jax.Array = jnp.nan
    moyal_loc:   jax.Array = jnp.nan
    moyal_scale: jax.Array = jnp.nan


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
    energy: jax.Array
    position: jax.Array
    theta: jax.Array
    phi: jax.Array
    t0: jax.Array

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
# Calibration source types — canonical home is lucid.sources.calibration_sources
# Re-exported here for backwards compatibility.
# ---------------------------------------------------------------------------
from lucid.sources.calibration_sources import (  # noqa: F401, E402
    IsotropicSource, LaserSource,
    isotropic_source, laser_source,
)


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


def _resolve_field(val: Any, config_dir: str) -> jax.Array:
    """Resolve a single physics config field value.

    - ``null`` / missing → ``jnp.asarray(jnp.nan)`` (unresolved placeholder;
      wavelength-projectable scalars are filled in later from curves if
      ``medium_model`` / ``qe_curve`` is referenced)
    - number     → ``jnp.asarray(float)``
    - list       → ``jnp.asarray(list)``
    - string ending in ``.json`` → loaded from file (relative to *config_dir*)
    - ``"__array__:filename.npy"`` → loaded from companion ``.npy``
    """
    if val is None:
        return jnp.asarray(jnp.nan)
    if isinstance(val, str):
        if val.endswith(".json"):
            with open(os.path.join(config_dir, val)) as f:
                arr = json.load(f)
            return jnp.asarray(arr, dtype=jnp.float32)
        if val.startswith(_ARRAY_SENTINEL):
            npy_name = val[len(_ARRAY_SENTINEL):]
            return jnp.asarray(np.load(os.path.join(config_dir, npy_name)))
        raise ValueError(f"Unrecognised string value: {val!r}")
    if isinstance(val, list):
        return jnp.asarray(val, dtype=jnp.float32)
    return jnp.asarray(float(val))


_PROJECTABLE_FIELDS = ("scatter_length", "absorption_length", "qe")


def _project_missing_scalars(kwargs, medium_path, qe_path, ref_wavelength_nm,
                             source_filepath):
    """Fill NaN scalar fields by evaluating wavelength curves at ``ref_wavelength_nm``.

    Modifies *kwargs* in place. Raises if a scalar is missing and no curve is
    available to project from.
    """
    for field in _PROJECTABLE_FIELDS:
        v = kwargs[field]
        if v.ndim != 0 or not bool(jnp.isnan(v)):
            continue

        if field == "qe":
            if qe_path is None:
                raise ValueError(
                    f"Physics config {source_filepath!r} has no scalar 'qe' "
                    f"and no 'qe_curve' to project from at "
                    f"λ={ref_wavelength_nm}nm.")
            from lucid.wavelength.medium import load_qe_curve
            scalar = float(load_qe_curve(qe_path)(ref_wavelength_nm))
        else:
            if medium_path is None:
                raise ValueError(
                    f"Physics config {source_filepath!r} has no scalar "
                    f"{field!r} and no 'medium_model' to project from at "
                    f"λ={ref_wavelength_nm}nm.")
            from lucid.wavelength.medium import make_medium
            wl_grid = jnp.array([ref_wavelength_nm], dtype=jnp.float32)
            m = make_medium("water", wavelength_grid=wl_grid,
                            medium_model_path=medium_path)
            coeff = m.scatter_coeff if field == "scatter_length" else m.absorption_coeff
            scalar = float(1.0 / (coeff[0] + 1e-30))

        kwargs[field] = jnp.asarray(scalar, dtype=jnp.float32)


def load_detector_params(filepath: str, num_sensors: int | None = None,
                         scalar_ref_wavelength: float | None = None) -> DetectorParams:
    """Load DetectorParams from a composable physics config JSON.

    The physics config may contain only the fields relevant to the detector.
    Missing scalar fields for ``scatter_length``, ``absorption_length`` or
    ``qe`` are projected from the referenced wavelength curves at
    ``scalar_ref_wavelength`` (default 400 nm). If neither a scalar value nor
    a curve reference is available, loading fails.

    Field values can be:
      - ``null`` / missing → projected from curves (see above), else error
      - number     → scalar
      - list       → inline array
      - ``"path/to/file.json"`` → loaded from JSON file (relative to config dir)
      - ``"__array__:file.npy"`` → loaded from companion ``.npy`` (legacy)

    The following extra keys are recognised but not stored in DetectorParams:
      - ``medium_model`` — path to medium model JSON
      - ``qe_curve`` — path to PMT QE curve JSON

    If *num_sensors* is given, scalar ``qe_corrections`` are automatically
    expanded to ``jnp.ones(num_sensors) * value``.
    """
    dp, _, _ = load_physics_config(filepath, num_sensors=num_sensors,
                                   scalar_ref_wavelength=scalar_ref_wavelength)
    return dp


def _scintillation_defaults_from_medium(medium_model_path: str | None) -> dict:
    """Pull DetectorParams-bound scintillation values from a material JSON.

    The material file (e.g. ``config/materials/wbls.json``) is the canonical
    home for the scintillator's physical numbers — the physics config inherits
    them so detector configs don't have to copy the WBLS spec verbatim.
    Returns an empty dict for non-scintillating media (no ``scintillation``
    block) or when no material model is referenced.
    """
    if not medium_model_path:
        return {}
    with open(medium_model_path) as f:
        m = json.load(f)
    scint = m.get("scintillation")
    if not scint:
        return {}
    ly  = scint.get("light_yield", {})
    tm  = scint.get("timing",      {})
    sp  = scint.get("spectrum",    {})
    pick = lambda d, k: d[k] if k in d else None
    out = {
        "S":  pick(ly, "S"),  "kB": pick(ly, "kB"), "C": pick(ly, "C"),
        "tau_r": pick(tm, "tau_r"), "tau_1": pick(tm, "tau_1"),
        "tau_2": pick(tm, "tau_2"), "R_1":   pick(tm, "R_1"),
        "moyal_amp":   pick(sp, "moyal_amp"),
        "moyal_loc":   pick(sp, "moyal_loc"),
        "moyal_scale": pick(sp, "moyal_scale"),
    }
    return {k: v for k, v in out.items() if v is not None}


def load_physics_config(filepath: str, num_sensors: int | None = None,
                        scalar_ref_wavelength: float | None = None) -> tuple[DetectorParams, str | None, str | None]:
    """Load a composable physics config — returns DetectorParams plus extras.

    Missing scalar fields (``scatter_length``, ``absorption_length``, ``qe``)
    are projected from the referenced wavelength curves at
    ``scalar_ref_wavelength`` (default 400 nm). If neither a scalar value nor
    a curve reference is available, loading fails with a clear error.

    Returns
    -------
    detector_params : DetectorParams
    medium_model_path : str or None
        Resolved path to the medium model JSON, if present.
    qe_curve_path : str or None
        Resolved path to the PMT QE curve JSON, if present.
    """
    from lucid.wavelength import DEFAULT_WAVELENGTH_NM
    ref_wl = (scalar_ref_wavelength if scalar_ref_wavelength is not None
              else DEFAULT_WAVELENGTH_NM)

    config_dir = os.path.dirname(filepath) or "."
    with open(filepath) as f:
        data = json.load(f)

    medium_model = data.get("medium_model")
    qe_curve = data.get("qe_curve")

    medium_model_path = os.path.join(config_dir, medium_model) if medium_model else None
    qe_curve_path = os.path.join(config_dir, qe_curve) if qe_curve else None

    # If the material JSON carries a `scintillation` block, use it as the
    # default source of the 10 scintillation scalars on DetectorParams.
    # The physics_config can still override any individual field by naming it
    # at the top level. Cherenkov-only materials never enter this branch.
    scint_defaults = _scintillation_defaults_from_medium(medium_model_path)

    kwargs = {}
    for field in DetectorParams._fields:
        val = data.get(field, None)
        if val is None and field in scint_defaults:
            val = scint_defaults[field]
        kwargs[field] = _resolve_field(val, config_dir)

    _project_missing_scalars(kwargs, medium_model_path, qe_curve_path,
                             ref_wl, filepath)

    if num_sensors is not None:
        qe_corr = kwargs['qe_corrections']
        if qe_corr.ndim == 0:
            # NaN qe_corrections means the config omitted it entirely — default
            # to neutral (1.0) rather than poisoning the array.
            fill = jnp.where(jnp.isnan(qe_corr), jnp.float32(1.0), qe_corr)
            kwargs['qe_corrections'] = jnp.ones(num_sensors) * fill

    return DetectorParams(**kwargs), medium_model_path, qe_curve_path


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
        wall_reflection_rate=jnp.array(0.0),
        sensor_reflection_rate=jnp.array(0.0),
        absorption_length=jnp.array(0.0),
        qe=jnp.array(0.0),
        qe_corrections=jnp.zeros(num_sensors),
        # Scintillation — physical lower bounds.
        S=jnp.array(0.0),  kB=jnp.array(0.0), C=jnp.array(0.0),
        tau_r=jnp.array(0.0), tau_1=jnp.array(0.0), tau_2=jnp.array(0.0),
        R_1=jnp.array(0.0),
        moyal_amp=jnp.array(0.0),
        moyal_loc=jnp.array(300.0), moyal_scale=jnp.array(1.0),
    )
    bounds_max = DetectorParams(
        scatter_length=jnp.array(100.0),
        wall_reflection_rate=jnp.array(0.5),
        sensor_reflection_rate=jnp.array(0.4),
        absorption_length=jnp.array(500.0),
        qe=jnp.array(1.0),
        qe_corrections=jnp.full(num_sensors, 2.0),
        # Scintillation — generous upper bounds (LS ~ 10k ph/MeV; tau_2 ~ 100 ns).
        S=jnp.array(1.0e4),  kB=jnp.array(1.0e-3), C=jnp.array(1.0e-7),
        tau_r=jnp.array(5.0), tau_1=jnp.array(10.0), tau_2=jnp.array(100.0),
        R_1=jnp.array(1.0),
        moyal_amp=jnp.array(1000.0),
        moyal_loc=jnp.array(600.0), moyal_scale=jnp.array(100.0),
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
    """Sensible initialization defaults for calibration optimization.

    Scintillation fields default to NaN — they're only read when the closed-over
    medium has ``"scintillation"`` in its ``emission_processes``, and in that
    case the WbLS / scintillator values come from the material JSON via
    ``load_physics_config``.
    """
    return DetectorParams(
        scatter_length=jnp.array(50.0),
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
        wall_reflection_rate=jnp.array(1.0),
        sensor_reflection_rate=jnp.array(1.0),
        absorption_length=jnp.array(1.0),
        qe=jnp.array(1.0),
        qe_corrections=jnp.full(num_sensors, 0.1),
        # Scintillation — uniform scale 1.0 by default.
        S=jnp.array(1.0),  kB=jnp.array(1.0), C=jnp.array(1.0),
        tau_r=jnp.array(1.0), tau_1=jnp.array(1.0), tau_2=jnp.array(1.0),
        R_1=jnp.array(1.0),
        moyal_amp=jnp.array(1.0),
        moyal_loc=jnp.array(1.0), moyal_scale=jnp.array(1.0),
    )
