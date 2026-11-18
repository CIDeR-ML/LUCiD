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
    create_default_detector_params
    load_detector_params
"""

from typing import NamedTuple
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from lucid.utils import cartesian_to_spherical, spherical_to_cartesian
from lucid.wavelength.optical_model import N_CONTROL as _N_CTRL


# ---------------------------------------------------------------------------
# DetectorParams
# ---------------------------------------------------------------------------

class ScatteringParams(NamedTuple):
    """Scattering optical properties (JAX pytree sub-tuple).

    Fields
    ------
    scatter_length : jnp.ndarray        scalar, meters (Rayleigh scatter length)
    mie_scatter_length : jnp.ndarray    scalar, meters
    g : jnp.ndarray                     scalar, Henyey-Greenstein asymmetry [0, 1]
    rayleigh_dev : jnp.ndarray          (n_ctrl,) λ-deviation curve, default ones (≡1)
    mie_dev : jnp.ndarray               (n_ctrl,) λ-deviation curve, default ones (≡1)

    ``rayleigh_dev`` / ``mie_dev`` are the fittable wavelength-deviation curves at
    the control wavelengths (:data:`lucid.wavelength.optical_model.CONTROL_WAVELENGTHS_NM`):
    in wavelength mode the per-photon length is ``L_ref(λ) / interp(λ, control_λ, curve)``.
    An all-ones curve reproduces the pure medium reference (byte-identical). They are
    ignored in monochromatic mode.
    """
    scatter_length: jnp.ndarray
    mie_scatter_length: jnp.ndarray
    g: jnp.ndarray
    rayleigh_dev: jnp.ndarray
    mie_dev: jnp.ndarray


class AbsorptionParams(NamedTuple):
    """Absorption optical properties (JAX pytree sub-tuple).

    Fields
    ------
    absorption_length : jnp.ndarray     scalar, meters
    abs_dev : jnp.ndarray               (n_ctrl,) λ-deviation curve, default ones (≡1)

    ``abs_dev`` is the fittable wavelength-deviation curve; see ``ScatteringParams``.
    """
    absorption_length: jnp.ndarray
    abs_dev: jnp.ndarray


class ReflectionParams(NamedTuple):
    """Reflection optical properties (JAX pytree sub-tuple).

    Fields
    ------
    wall_reflection_rate : jnp.ndarray   scalar, [0, 1] — scalar-model wall rate
    sensor_reflection_rate : jnp.ndarray scalar, [0, 1] — scalar-model sensor rate
    wall_R0 : jnp.ndarray        blacksheet normal-incidence reflectance (Schlick)
    wall_p : jnp.ndarray         blacksheet Schlick angular exponent
    wall_fspec : jnp.ndarray     blacksheet specular fraction (1-fspec diffuse)
    cathode_nr : jnp.ndarray     PMT cathode real refractive index
    cathode_nk : jnp.ndarray     PMT cathode imaginary index (absorption)
    sensor_fspec : jnp.ndarray   PMT specular fraction (1-fspec diffuse)

    The two rates drive every model. ``wall_fspec``/``sensor_fspec`` are the specular
    fractions consumed by the DEFAULT ``scalar_mix`` model (and ``angular``); ``wall_R0/
    wall_p`` and ``cathode_nr/cathode_nk`` parameterise the ``angular`` model (Schlick
    blacksheet + multilayer-Fresnel cathode) and are inert unless ``reflection_model=
    'angular'``. Only the two rates matter for the legacy ``scalar`` model.
    """
    wall_reflection_rate: jnp.ndarray
    sensor_reflection_rate: jnp.ndarray
    wall_R0: jnp.ndarray
    wall_p: jnp.ndarray
    wall_fspec: jnp.ndarray
    cathode_nr: jnp.ndarray
    cathode_nk: jnp.ndarray
    sensor_fspec: jnp.ndarray


class ResponseParams(NamedTuple):
    """Global PMT response properties (JAX pytree sub-tuple).

    Fields
    ------
    qe : jnp.ndarray         scalar, quantum efficiency at the scalar reference
                             wavelength (400 nm) [0, 1], per photon arriving at a
                             PMT (its reflection included, as PMT QE is quoted)
    tts : jnp.ndarray        scalar, transit-time spread, ns (default 0.0)
    qe_dev : jnp.ndarray     (n_ctrl,) QE λ-deviation curve, default ones (≡1)

    ``tts`` is a calibrated PMT-response field, read by the hits; its neutral default (0.0)
    leaves the forward model unchanged. Scalar mode applies ``qe`` to every photon; wavelength
    mode applies ``qe * qe_fn(λ) / qe_fn(400 nm)``, which is ``qe_fn(λ)`` itself when ``qe``
    was projected from the curve. ``qe_dev`` multiplies that per-photon weight in
    wavelength mode (≡1 default; see ``ScatteringParams``). The photon step reflects part
    of the light reaching a sensor and deposits the rest, so a deposit converts at that
    QE / (1 - R0), R0 the sensor reflectance at normal incidence; ``qe * qe_corrections``
    must therefore not exceed ``1 - R0``.
    """
    qe: jnp.ndarray
    tts: jnp.ndarray
    qe_dev: jnp.ndarray


class PerPmtParams(NamedTuple):
    """Per-sensor PMT calibration properties (JAX pytree sub-tuple).

    Fields
    ------
    qe_corrections : jnp.ndarray  shape (num_sensors,), per-sensor QE multipliers
    t0 : jnp.ndarray              shape (num_sensors,), per-sensor time offset, ns (default zeros)

    ``t0`` is a per-PMT first-arrival offset (the TQ-map constant), not yet applied by the
    hits; its default is zero.
    """
    qe_corrections: jnp.ndarray
    t0: jnp.ndarray


class ScintillationParams(NamedTuple):
    """Scintillation emission properties (JAX pytree sub-tuple). Added for the
    refactor-v2 multi-material merge (wbls/ice scintillator support).

    Fields
    ------
    S, kB, C : jnp.ndarray   Chou light yield  dL/dx = S·(dE/dx)/(1+kB·(dE/dx)+C·(dE/dx)²)
                             [S ph/MeV, kB mm/keV, C (mm/keV)²]. LIVE-differentiable.
    tau_rise, tau_fall : jnp.ndarray  hypoexp emission timing (ns). LIVE-differentiable.

    The emission-spectrum shape (``moyal_loc``, ``moyal_scale``) is static medium data, on
    :class:`lucid.wavelength.medium.MediumProperties`.

    All default to 0.0 (NEUTRAL: S=0 → zero scintillation light → forward byte-identical
    for non-scintillating media; finite, NOT NaN, so the generic normalize/bounds tree-walk
    never poisons). Only READ when the closed-over medium has ``"scintillation"`` in its
    ``emission_processes``. Excluded from optimization by default (not in any default
    trainable-field set).
    """
    S: jnp.ndarray = np.float32(0.0)
    kB: jnp.ndarray = np.float32(0.0)
    C: jnp.ndarray = np.float32(0.0)
    tau_rise: jnp.ndarray = np.float32(0.0)
    tau_fall: jnp.ndarray = np.float32(0.0)


class DetectorParams(NamedTuple):
    """Detector calibration parameters (JAX pytree), nested by physics.

    Sub-tuples
    ----------
    scattering   : ScatteringParams   scatter_length, mie_scatter_length, g
    absorption   : AbsorptionParams   absorption_length
    reflection   : ReflectionParams   wall_reflection_rate, sensor_reflection_rate
    response     : ResponseParams     qe, tts, qe_dev
    per_pmt      : PerPmtParams        qe_corrections, t0
    scintillation: ScintillationParams S, kB, C, tau_* (neutral=0 for non-scint media)
    """
    scattering: ScatteringParams
    absorption: AbsorptionParams
    reflection: ReflectionParams
    response: ResponseParams
    per_pmt: PerPmtParams
    scintillation: ScintillationParams

    @classmethod
    def from_flat(cls, *, num_sensors=None, **flat):
        """Build a nested ``DetectorParams`` from FLAT leaf-field values.

        Convenience constructor mirroring the on-disk flat JSON schema: pass
        any subset of the leaf fields (``scatter_length``, ``qe``,
        ``qe_corrections``, ...) and the rest are filled with neutral defaults
        (no-Mie scattering, identity response/per-PMT) so the forward is
        unchanged. The per-sensor ``t0`` defaults to a ``(num_sensors,)`` array;
        ``num_sensors`` is inferred from ``qe_corrections`` when not given.
        """
        ns = num_sensors
        if ns is None and "qe_corrections" in flat:
            qc = jnp.asarray(flat["qe_corrections"])
            ns = int(qc.shape[0]) if qc.ndim >= 1 else 1
        if ns is None:
            ns = 1
        defaults = dict(
            scatter_length=50.0,
            mie_scatter_length=1.0e9,       # no Mie channel, which makes g immaterial
            g=0.0,
            rayleigh_dev=jnp.ones(_N_CTRL),
            mie_dev=jnp.ones(_N_CTRL),
            absorption_length=100.0,
            abs_dev=jnp.ones(_N_CTRL),
            wall_reflection_rate=0.0,
            sensor_reflection_rate=0.0,
            wall_fspec=0.55,                # the SK-lit specular fractions
            sensor_fspec=0.90,
            **_ANGULAR_REFERENCE,
            qe=0.2,
            tts=0.0,
            qe_dev=jnp.ones(_N_CTRL),
            qe_corrections=jnp.ones(ns),
            t0=jnp.zeros(ns),
            S=0.0, kB=0.0, C=0.0, tau_rise=0.0, tau_fall=0.0,    # scintillation: neutral (S=0 → no scint light)
        )
        defaults.update(flat)
        merged = {k: jnp.asarray(v) for k, v in defaults.items()}
        return _nest_flat_kwargs(merged)


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
    t0 : jnp.ndarray        scalar, the track's start time (ns): the track forward offsets
                            every photon's emission time by it
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
        theta, phi = cartesian_to_spherical(direction / jnp.linalg.norm(direction))
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


class JointParams(NamedTuple):
    """Umbrella for jointly-fitted detector + particle parameters — the ``SimParams`` seed (D4).

    A plain NamedTuple of the two existing pytrees, so the generic optimization helpers
    (:func:`normalize_params`, :func:`make_optimization_mask`) descend into it unchanged. A
    future sibling forward (e.g. DTRAX track params) extends this by adding a field — nothing
    else changes. Optimizing ``(detector, particle)`` jointly is self-calibration.
    """
    detector: DetectorParams
    particle: ParticleParams


# ---------------------------------------------------------------------------
# Calibration source types — canonical home is lucid.sources.calibration_sources
# Re-exported here for backwards compatibility.
# ---------------------------------------------------------------------------
from lucid.sources.calibration_sources import (  # noqa: F401, E402
    IsotropicSource, LaserSource,
    isotropic_source, laser_source,
)


# ---------------------------------------------------------------------------
# Flat <-> nested mapping.
#
# The on-disk JSON format keeps FLAT physics-config keys (scatter_length, qe,
# qe_corrections, ...). In-memory the pytree is nested by physics. These helpers
# translate between the two representations.
# ---------------------------------------------------------------------------

# leaf field name -> (sub-tuple attribute on DetectorParams, sub-tuple class)
_SUBTUPLES = (
    ("scattering", ScatteringParams),
    ("absorption", AbsorptionParams),
    ("reflection", ReflectionParams),
    ("response", ResponseParams),
    ("per_pmt", PerPmtParams),
    ("scintillation", ScintillationParams),     # appended LAST — preserves the existing leaf order
)

# Ordered list of every leaf (flat) field name across all sub-tuples.
_FLAT_FIELDS = tuple(
    field for _, cls in _SUBTUPLES for field in cls._fields
)


def _nest_flat_kwargs(flat: dict) -> DetectorParams:
    """Build a nested ``DetectorParams`` from a flat ``{leaf_field: value}`` dict.

    Tolerant of OMITTED fields for sub-tuples whose NamedTuple supplies defaults
    (currently ``ScintillationParams``, all 0.0) — so callers that predate the scint
    sub-tuple (hardcoded flat dicts) build unchanged. Sub-tuples without field defaults
    (the original five) still require every leaf (a missing one raises, as before).
    """
    subs = {}
    for attr, cls in _SUBTUPLES:
        subs[attr] = cls(**{f: flat[f] for f in cls._fields if f in flat})
    return DetectorParams(**subs)


def _flatten_detector_params(params: DetectorParams) -> dict:
    """Walk a nested ``DetectorParams`` to a flat ``{leaf_field: value}`` dict."""
    flat = {}
    for attr, cls in _SUBTUPLES:
        sub = getattr(params, attr)
        for f in cls._fields:
            flat[f] = getattr(sub, f)
    return flat


# ---------------------------------------------------------------------------
# Physics config loading
# ---------------------------------------------------------------------------

# Keys a physics config may carry besides the leaf fields (and ``_``-prefixed comments).
# ``digitizer`` and ``trigger`` are production's readout blocks (lucid.production.dataprod.run_job).
_CONFIG_KEYS = ("medium_model", "qe_curve", "scalar_mode", "angular_reflection", "digitizer",
                "trigger")

# The optical lengths only scalar mode (wavelength_mode=False) reads: wavelength mode takes
# them from the medium's curves. A config states them under "scalar_mode", which says so.
_SCALAR_MODE_FIELDS = ("scatter_length", "absorption_length", "mie_scatter_length")

# Leaves every config states: nothing to project them from and no neutral value.
_REQUIRED_FIELDS = ("wall_reflection_rate", "sensor_reflection_rate",
                    "wall_fspec", "sensor_fspec", "qe_corrections")

# Leaves projected from the medium at the scalar reference wavelength when a config gives a
# medium_model and omits them; required otherwise. ``qe`` likewise from the qe_curve.
_MEDIUM_FIELDS = ("scatter_length", "absorption_length", "mie_scatter_length", "g")

# Leaves only reflection_model='angular' reads (Schlick blacksheet wall, Fresnel cathode). A
# config states them in an "angular_reflection" block, all four, which says which model reads
# them; a setup that selects the angular model with a physics config requires the block.
# Without it they hold these reference values (blacksheet near-diffuse with a low R0; bialkali
# cathode n = 2.8 + 1.5i), which no other model reads.
_ANGULAR_REFERENCE = {"wall_R0": 0.05, "wall_p": 1.0, "cathode_nr": 2.8, "cathode_nk": 1.5}

# Identity values: the forward is unchanged by them (no TTS smearing, no per-PMT offset, no λ
# deviation, no scintillation light).
_IDENTITY = {"tts": 0.0, "t0": 0.0,
             "S": 0.0, "kB": 0.0, "C": 0.0, "tau_rise": 0.0, "tau_fall": 0.0}
_DEV_CURVE_FIELDS = ("rayleigh_dev", "mie_dev", "abs_dev", "qe_dev")


def _check_keys(keys, allowed, where):
    """Raise on a key outside ``allowed`` (``_``-prefixed comments pass), naming the closest one."""
    import difflib
    for key in keys:
        if key.startswith("_") or key in allowed:
            continue
        close = difflib.get_close_matches(key, allowed, n=1)
        hint = f"; did you mean {close[0]!r}?" if close else ""
        raise ValueError(f"{where}: unknown key {key!r}{hint}")


def _resolve_field(val, config_dir):
    """A physics config field value: a number, an inline list, or ``"file.json"`` (relative to
    *config_dir*) holding a list."""
    if isinstance(val, str):
        if val.endswith(".json"):
            with open(os.path.join(config_dir, val)) as f:
                arr = json.load(f)
            return jnp.asarray(arr, dtype=jnp.float32)
        raise ValueError(f"Unrecognised string value: {val!r}")
    if isinstance(val, list):
        return jnp.asarray(val, dtype=jnp.float32)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"not a number, list or file: {val!r}")
    return jnp.asarray(float(val))


def _scintillation_defaults_from_medium(medium):
    """Pull DetectorParams-bound scintillation values from a loaded material.

    The material file (e.g. ``config/materials/wbls.json``) is the canonical home
    for the scintillator's physical numbers — the physics config inherits them so
    detector configs don't have to copy the WbLS spec verbatim. Returns an empty
    dict for non-scintillating media (no ``scintillation`` block, e.g. water) or
    when no material model is referenced — so water stays neutral (S=0).
    """
    if medium is None:
        return {}
    scint = medium.model.get("scintillation")
    if not scint:
        return {}
    ly = scint.get("light_yield", {})
    tm = scint.get("timing", {})
    pick = lambda d, k: d[k] if k in d else None
    out = {
        "S": pick(ly, "S"), "kB": pick(ly, "kB"), "C": pick(ly, "C"),
        "tau_rise": pick(tm, "tau_rise"), "tau_fall": pick(tm, "tau_fall"),
    }
    return {k: v for k, v in out.items() if v is not None}


def _project_missing_scalars(kwargs, medium, qe_curve, source_filepath):
    """Fill the omitted :data:`_MEDIUM_FIELDS` and ``qe`` from the medium and the QE curve at the
    scalar reference wavelength; raise for one with nothing to project from.

    Modifies *kwargs* in place. ``medium`` (loaded, or None) is evaluated once, on the one-point
    grid ``[λ_ref]``.
    """
    from lucid.wavelength import DEFAULT_WAVELENGTH_NM as ref_wl
    if "qe" not in kwargs:
        if qe_curve is None:
            raise ValueError(f"Physics config {source_filepath!r} has no scalar 'qe' and no "
                             f"'qe_curve' to project from at λ={ref_wl}nm.")
        kwargs["qe"] = jnp.asarray(float(qe_curve(ref_wl)), dtype=jnp.float32)

    missing = [f for f in _MEDIUM_FIELDS if f not in kwargs]
    if not missing:
        return
    if medium is None:
        raise ValueError(f"Physics config {source_filepath!r} has no {missing} and no "
                         f"'medium_model' to project from at λ={ref_wl}nm.")
    from lucid.wavelength.medium import medium_on_grid
    m = medium_on_grid(medium, jnp.array([ref_wl], dtype=jnp.float32))
    projected = {
        "scatter_length": lambda: float(1.0 / (m.scatter_coeff[0] + 1e-30)),
        "absorption_length": lambda: float(1.0 / (m.absorption_coeff[0] + 1e-30)),
        "mie_scatter_length": lambda: float(1.0 / (m.mie_scatter_coeff[0] + 1e-30)),
        "g": lambda: float(m.mie_asymmetry),
    }
    for field in missing:
        kwargs[field] = jnp.asarray(projected[field](), dtype=jnp.float32)


def load_detector_params(filepath: str, num_sensors: int = None) -> DetectorParams:
    """The DetectorParams of :func:`load_physics_config`."""
    dp, _, _ = load_physics_config(filepath, num_sensors=num_sensors)
    return dp


def load_physics_config(filepath: str, num_sensors: int = None, reflection_model: str = None):
    """Load a physics config JSON: its DetectorParams, medium and QE curve.

    Keys are the DetectorParams leaf names, ``medium_model`` and ``qe_curve`` (paths relative
    to the config), ``scalar_mode``, ``angular_reflection``, production's ``digitizer`` /
    ``trigger`` blocks and ``_``-prefixed comments; any other key raises. A value is a number,
    an inline list or a ``"file.json"`` holding a list.

    - ``wall_reflection_rate``, ``sensor_reflection_rate``, ``wall_fspec``, ``sensor_fspec`` and
      ``qe_corrections`` are required.
    - ``scatter_length``, ``absorption_length`` and ``mie_scatter_length`` are read only in
      scalar mode (``wavelength_mode=False``; wavelength mode evaluates the medium's curves), so
      a config states them inside a ``"scalar_mode"`` block. Omitted, they and ``g`` are the
      medium's values at the scalar reference wavelength (400 nm), and ``qe`` the QE curve's;
      with nothing to project from, loading raises.
    - ``wall_R0``, ``wall_p``, ``cathode_nr`` and ``cathode_nk`` are read only by
      ``reflection_model='angular'``, so a config states all four inside an
      ``"angular_reflection"`` block. ``reflection_model='angular'`` requires the block; without
      it the four hold reference values no other model reads.
    - The other leaves take identity values when omitted; a scintillating medium supplies
      ``S``, ``kB``, ``C``, ``tau_rise`` and ``tau_fall``.

    If *num_sensors* is given, scalar per-PMT leaves (``qe_corrections``, ``t0``)
    are expanded to ``(num_sensors,)``, and a per-PMT file of another length raises.

    Returns
    -------
    detector_params : DetectorParams
    medium : MediumProperties or None
        The ``medium_model`` material, loaded (:func:`lucid.wavelength.medium.load_medium`).
    qe_curve : QECurve or None
        The ``qe_curve``, loaded (:func:`lucid.wavelength.medium.load_qe_curve`).

    Each referenced file is read once.
    """
    config_dir = os.path.dirname(filepath) or "."
    with open(filepath) as f:
        data = json.load(f)
    leaf_keys = tuple(f for f in _FLAT_FIELDS
                      if f not in _SCALAR_MODE_FIELDS and f not in _ANGULAR_REFERENCE)
    for key in _SCALAR_MODE_FIELDS:
        if key in data:
            raise ValueError(
                f"physics config {filepath!r}: {key!r} is read only in scalar mode "
                "(wavelength_mode=False); wavelength mode takes it from the medium. State it "
                'under "scalar_mode".')
    for key in _ANGULAR_REFERENCE:
        if key in data:
            raise ValueError(
                f"physics config {filepath!r}: {key!r} is read only by "
                "reflection_model='angular'. State it under \"angular_reflection\".")
    _check_keys(data, leaf_keys + _CONFIG_KEYS, f"physics config {filepath!r}")
    scalar_mode = data.get("scalar_mode", {})
    _check_keys(scalar_mode, _SCALAR_MODE_FIELDS, f"physics config {filepath!r} 'scalar_mode'")
    angular = data.get("angular_reflection")
    if angular is not None:
        _check_keys(angular, tuple(_ANGULAR_REFERENCE),
                    f"physics config {filepath!r} 'angular_reflection'")
        missing = [f for f in _ANGULAR_REFERENCE if f not in angular]
        if missing:
            raise ValueError(f"physics config {filepath!r} 'angular_reflection' does not state "
                             f"{missing}")
    elif reflection_model == 'angular':
        raise ValueError(
            f"reflection_model='angular' reads {list(_ANGULAR_REFERENCE)}; physics config "
            f"{filepath!r} has no \"angular_reflection\" block stating them.")
    missing = [f for f in _REQUIRED_FIELDS if f not in data]
    if missing:
        raise ValueError(f"physics config {filepath!r} does not state {missing}")

    from lucid.wavelength.medium import load_medium, load_qe_curve
    medium_model, qe_curve = data.get("medium_model"), data.get("qe_curve")
    medium = (load_medium(medium_model_path=os.path.join(config_dir, medium_model))
              if medium_model is not None else None)
    qe_curve = load_qe_curve(os.path.join(config_dir, qe_curve)) if qe_curve is not None else None

    given = {k: v for k, v in data.items() if k in leaf_keys}
    for block in (scalar_mode, angular or {}):
        given.update({k: v for k, v in block.items() if not k.startswith("_")})
    kwargs = {}
    for field, val in given.items():
        try:
            kwargs[field] = _resolve_field(val, config_dir)
        except ValueError as e:
            raise ValueError(f"physics config {filepath!r}, {field!r}: {e}") from None

    _project_missing_scalars(kwargs, medium, qe_curve, filepath)
    # Scintillation scalars: a scintillating material JSON (wbls) supplies the physical
    # S/kB/C/tau values for any the config leaves out; otherwise the identity (no light).
    for field, val in _scintillation_defaults_from_medium(medium).items():
        kwargs.setdefault(field, jnp.asarray(val, dtype=jnp.float32))
    for field, val in {**_IDENTITY, **_ANGULAR_REFERENCE}.items():
        kwargs.setdefault(field, jnp.asarray(val, dtype=jnp.float32))
    for field in _DEV_CURVE_FIELDS:
        kwargs.setdefault(field, jnp.ones(_N_CTRL, dtype=jnp.float32))

    if num_sensors is not None:
        qe_corr = kwargs['qe_corrections']
        if qe_corr.ndim == 0:
            kwargs['qe_corrections'] = jnp.ones(num_sensors) * qe_corr
        elif qe_corr.shape[0] != num_sensors:
            raise ValueError(f"qe_corrections has {qe_corr.shape[0]} entries for {num_sensors} "
                             f"sensors ({filepath})")
        if kwargs['t0'].ndim == 0:
            kwargs['t0'] = jnp.full(num_sensors, kwargs['t0'], dtype=jnp.float32)

    return _nest_flat_kwargs(kwargs), medium, qe_curve


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
    bounds_min = _nest_flat_kwargs(dict(
        scatter_length=jnp.array(0.0),
        mie_scatter_length=jnp.array(0.0),
        g=jnp.array(0.0),
        rayleigh_dev=jnp.full(_N_CTRL, 0.1),
        mie_dev=jnp.full(_N_CTRL, 0.1),
        absorption_length=jnp.array(0.0),
        abs_dev=jnp.full(_N_CTRL, 0.1),
        wall_reflection_rate=jnp.array(0.0),
        sensor_reflection_rate=jnp.array(0.0),
        wall_R0=jnp.array(0.0), wall_p=jnp.array(0.5), wall_fspec=jnp.array(0.0),
        cathode_nr=jnp.array(1.5), cathode_nk=jnp.array(0.0), sensor_fspec=jnp.array(0.0),
        qe=jnp.array(0.0),
        tts=jnp.array(0.0),
        qe_dev=jnp.full(_N_CTRL, 0.1),
        qe_corrections=jnp.zeros(num_sensors),
        t0=jnp.full(num_sensors, -10.0),
        S=jnp.array(0.0), kB=jnp.array(0.0), C=jnp.array(0.0),       # scintillation (inert unless fit)
        tau_rise=jnp.array(0.0), tau_fall=jnp.array(0.0),
    ))
    bounds_max = _nest_flat_kwargs(dict(
        scatter_length=jnp.array(100.0),
        mie_scatter_length=jnp.array(100.0),
        g=jnp.array(1.0),
        rayleigh_dev=jnp.full(_N_CTRL, 10.0),
        mie_dev=jnp.full(_N_CTRL, 10.0),
        absorption_length=jnp.array(500.0),
        abs_dev=jnp.full(_N_CTRL, 10.0),
        wall_reflection_rate=jnp.array(0.5),
        sensor_reflection_rate=jnp.array(0.4),
        wall_R0=jnp.array(0.5), wall_p=jnp.array(12.0), wall_fspec=jnp.array(1.0),
        cathode_nr=jnp.array(4.0), cathode_nk=jnp.array(4.0), sensor_fspec=jnp.array(1.0),
        qe=jnp.array(1.0),
        tts=jnp.array(5.0),
        qe_dev=jnp.full(_N_CTRL, 10.0),
        qe_corrections=jnp.full(num_sensors, 2.0),
        t0=jnp.full(num_sensors, 10.0),
        S=jnp.array(5000.0), kB=jnp.array(1e-3), C=jnp.array(1e-6), # scintillation physical ranges
        tau_rise=jnp.array(10.0), tau_fall=jnp.array(50.0),
    ))
    return bounds_min, bounds_max


def particle_bounds(detector_r, detector_h, *, energy_min=0.0, energy_max=5000.0,
                    t0_range=50.0):
    """Return ``(bounds_min, bounds_max)`` :class:`ParticleParams` with physical ranges.

    Position is bounded by the cylinder (``±r`` in x/y, ``±h/2`` in z); ``theta∈[0,π]``,
    ``phi∈[-π,π]``, ``energy∈[energy_min, energy_max]`` MeV, ``t0∈[±t0_range]`` ns. Mirrors
    :func:`default_bounds` for the particle pytree so the same normalize/denormalize helpers
    apply (and :func:`joint_bounds` stacks the two for a self-calibration fit).
    """
    r = float(detector_r); h2 = float(detector_h) / 2.0; pi = float(jnp.pi)
    bmin = ParticleParams(energy=jnp.asarray(energy_min), position=jnp.array([-r, -r, -h2]),
                          theta=jnp.asarray(0.0), phi=jnp.asarray(-pi), t0=jnp.asarray(-t0_range))
    bmax = ParticleParams(energy=jnp.asarray(energy_max), position=jnp.array([r, r, h2]),
                          theta=jnp.asarray(pi), phi=jnp.asarray(pi), t0=jnp.asarray(t0_range))
    return bmin, bmax


def joint_bounds(num_sensors, detector_r, detector_h, **particle_kw):
    """``(bounds_min, bounds_max)`` :class:`JointParams` = detector :func:`default_bounds`
    + :func:`particle_bounds` — the bounds for a joint detector+track (self-calibration) fit."""
    dmin, dmax = default_bounds(num_sensors)
    pmin, pmax = particle_bounds(detector_r, detector_h, **particle_kw)
    return JointParams(dmin, pmin), JointParams(dmax, pmax)


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
    def build(node):
        # Recurse into nested NamedTuples, keying the decision on the LEAF field
        # name. For DetectorParams this descends scattering/absorption/... into
        # their leaf fields (scatter_length, t0, tts, ...).
        fields = getattr(type(node), "_fields", None)
        if fields is None:
            # Reached a leaf array — caller decided trainability already.
            raise RuntimeError("build() expects a NamedTuple node")
        out = {}
        for field in fields:
            val = getattr(node, field)
            sub_fields = getattr(type(val), "_fields", None)
            if sub_fields is not None:
                # Nested sub-tuple: recurse (decision happens on its leaves).
                out[field] = build(val)
            else:
                flag = field in trainable_fields
                out[field] = jax.tree.map(lambda x: flag, val)
        return type(node)(**out)

    return build(params)


def create_default_detector_params(num_sensors: int) -> DetectorParams:
    """Sensible initialization defaults for calibration optimization."""
    return _nest_flat_kwargs(dict(
        scatter_length=jnp.array(50.0),
        mie_scatter_length=jnp.array(50.0),
        g=jnp.array(0.85),
        rayleigh_dev=jnp.ones(_N_CTRL),
        mie_dev=jnp.ones(_N_CTRL),
        absorption_length=jnp.array(150.0),
        abs_dev=jnp.ones(_N_CTRL),
        wall_reflection_rate=jnp.array(0.2),
        sensor_reflection_rate=jnp.array(0.2),
        wall_fspec=jnp.array(0.55),
        sensor_fspec=jnp.array(0.90),
        **{k: jnp.array(v) for k, v in _ANGULAR_REFERENCE.items()},
        qe=jnp.array(0.2),
        tts=jnp.array(0.0),
        qe_dev=jnp.ones(_N_CTRL),
        qe_corrections=jnp.ones(num_sensors),
        t0=jnp.zeros(num_sensors),
    ))
