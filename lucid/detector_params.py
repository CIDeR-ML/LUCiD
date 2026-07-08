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

from lucid.utils import spherical_to_cartesian
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

    The first two drive the default ``scalar`` reflection model. The remaining six
    parameterise the ``angular`` model (Schlick blacksheet + multilayer-Fresnel
    cathode); they are inert unless ``reflection_model='angular'``.
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
    qe : jnp.ndarray         scalar, base quantum efficiency [0, 1]
    spe_width : jnp.ndarray  scalar, single-photoelectron charge resolution (default 0.0)
    tts : jnp.ndarray        scalar, transit-time spread, ns (default 0.0)
    qe_dev : jnp.ndarray     (n_ctrl,) QE λ-deviation curve, default ones (≡1)

    ``spe_width`` and ``tts`` are calibrated PMT-response fields. Their neutral
    defaults (0.0) leave the current forward model unchanged; wiring them into
    ``make_hits`` is a later step. ``qe_dev`` multiplies the per-photon QE weight
    ``qe_fn(λ)`` in wavelength mode (≡1 default; see ``ScatteringParams``).
    """
    qe: jnp.ndarray
    spe_width: jnp.ndarray
    tts: jnp.ndarray
    qe_dev: jnp.ndarray


class PerPmtParams(NamedTuple):
    """Per-sensor PMT calibration properties (JAX pytree sub-tuple).

    Fields
    ------
    qe_corrections : jnp.ndarray  shape (num_sensors,), per-sensor QE multipliers
    gain : jnp.ndarray            shape (num_sensors,), per-sensor gain (default ones)
    t0 : jnp.ndarray              shape (num_sensors,), per-sensor time offset, ns (default zeros)
    walk : jnp.ndarray            shape (num_sensors,), per-sensor time-walk (default zeros)

    ``gain`` (charge moments) and ``t0`` (first-arrival offset, the TQ-map constant) are
    live calibration parameters. ``walk`` is the **electronics time-walk** (the charge-
    AMPLITUDE-dependent threshold-crossing shift) — a distinct effect from the TTS
    occupancy order-statistic that already lives in ``make_hits_moments``. It has **no
    mechanism in the first-arrival engine** (which has no pulse shape / discriminator),
    so it is INERT here and reserved for a future waveform-based timing calibration; do
    not expect it to be fit by the first-arrival path. Neutral defaults leave the forward
    unchanged.
    """
    qe_corrections: jnp.ndarray
    gain: jnp.ndarray
    t0: jnp.ndarray
    walk: jnp.ndarray


class ScintillationParams(NamedTuple):
    """Scintillation emission properties (JAX pytree sub-tuple). Added for the
    refactor-v2 multi-material merge (wbls/ice scintillator support).

    Fields
    ------
    S, kB, C : jnp.ndarray   Chou light yield  dL/dx = S·(dE/dx)/(1+kB·(dE/dx)+C·(dE/dx)²)
                             [S ph/MeV, kB mm/keV, C (mm/keV)²]. LIVE-differentiable.
    tau_rise, tau_fall : jnp.ndarray  hypoexp emission timing (ns). LIVE-differentiable.
    moyal_amp, moyal_loc, moyal_scale : jnp.ndarray  emission-spectrum shape. Currently
                             BAKED from the material JSON at setup (closed into the Moyal
                             inverse-CDF), so dormant on the pytree — carried for a future
                             reparametrized-sampler calibration; moyal_amp is unused.

    All default to 0.0 (NEUTRAL: S=0 → zero scintillation light → forward byte-identical
    for non-scintillating media; finite, NOT NaN, so the generic normalize/bounds tree-walk
    never poisons). Only READ when the closed-over medium has ``"scintillation"`` in its
    ``emission_processes``. Excluded from optimization by default (not in any default
    trainable-field set).
    """
    S: jnp.ndarray = jnp.asarray(0.0, jnp.float32)
    kB: jnp.ndarray = jnp.asarray(0.0, jnp.float32)
    C: jnp.ndarray = jnp.asarray(0.0, jnp.float32)
    tau_rise: jnp.ndarray = jnp.asarray(0.0, jnp.float32)
    tau_fall: jnp.ndarray = jnp.asarray(0.0, jnp.float32)
    moyal_amp: jnp.ndarray = jnp.asarray(0.0, jnp.float32)
    moyal_loc: jnp.ndarray = jnp.asarray(0.0, jnp.float32)
    moyal_scale: jnp.ndarray = jnp.asarray(0.0, jnp.float32)


class DetectorParams(NamedTuple):
    """Detector calibration parameters (JAX pytree), nested by physics.

    Sub-tuples
    ----------
    scattering   : ScatteringParams   scatter_length, mie_scatter_length, g
    absorption   : AbsorptionParams   absorption_length
    reflection   : ReflectionParams   wall_reflection_rate, sensor_reflection_rate
    response     : ResponseParams     qe, spe_width, tts
    per_pmt      : PerPmtParams        qe_corrections, gain, t0, walk
    scintillation: ScintillationParams S, kB, C, tau_*, moyal_* (appended LAST so the
                   existing 23-leaf flatten order is preserved; neutral=0 for non-scint media)
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
        unchanged. Per-sensor placeholder fields (``gain``/``t0``/``walk``)
        default to ``(num_sensors,)`` arrays; ``num_sensors`` is inferred from
        ``qe_corrections`` when not given.
        """
        ns = num_sensors
        if ns is None and "qe_corrections" in flat:
            qc = jnp.asarray(flat["qe_corrections"])
            ns = int(qc.shape[0]) if qc.ndim >= 1 else 1
        if ns is None:
            ns = 1
        defaults = dict(
            scatter_length=50.0,
            mie_scatter_length=_MIE_DEFAULTS["mie_scatter_length"],
            g=_MIE_DEFAULTS["g"],
            rayleigh_dev=jnp.ones(_N_CTRL),
            mie_dev=jnp.ones(_N_CTRL),
            absorption_length=100.0,
            abs_dev=jnp.ones(_N_CTRL),
            wall_reflection_rate=0.0,
            sensor_reflection_rate=0.0,
            wall_R0=_ANGULAR_REFL_DEFAULTS["wall_R0"],
            wall_p=_ANGULAR_REFL_DEFAULTS["wall_p"],
            wall_fspec=_ANGULAR_REFL_DEFAULTS["wall_fspec"],
            cathode_nr=_ANGULAR_REFL_DEFAULTS["cathode_nr"],
            cathode_nk=_ANGULAR_REFL_DEFAULTS["cathode_nk"],
            sensor_fspec=_ANGULAR_REFL_DEFAULTS["sensor_fspec"],
            qe=0.2,
            spe_width=0.0,
            tts=0.0,
            qe_dev=jnp.ones(_N_CTRL),
            qe_corrections=jnp.ones(ns),
            gain=jnp.ones(ns),
            t0=jnp.zeros(ns),
            walk=jnp.zeros(ns),
            S=0.0, kB=0.0, C=0.0, tau_rise=0.0, tau_fall=0.0,    # scintillation: neutral (S=0 → no scint light)
            moyal_amp=0.0, moyal_loc=0.0, moyal_scale=0.0,
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
# File I/O
# ---------------------------------------------------------------------------

_ARRAY_SENTINEL = "__array__:"


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

# leaf field name -> sub-tuple attribute name (e.g. 'scatter_length' -> 'scattering')
_FIELD_TO_SUBTUPLE = {
    field: attr for attr, cls in _SUBTUPLES for field in cls._fields
}


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


def save_detector_params(params: DetectorParams, filepath: str):
    """Save DetectorParams to JSON + companion .npy files for arrays.

    Scalars are stored directly in JSON.  Arrays whose size > 1 are saved to
    ``<filepath_stem>_<field>.npy`` and referenced in JSON via the sentinel
    ``"__array__:<filename>"`` convention.
    """
    dirpath = os.path.dirname(filepath) or "."
    stem = os.path.splitext(os.path.basename(filepath))[0]
    data = {}
    flat = _flatten_detector_params(params)
    for field in _FLAT_FIELDS:
        val = flat[field]
        arr = np.asarray(val)
        if arr.ndim == 0:
            data[field] = float(arr)
        else:
            npy_name = f"{stem}_{field}.npy"
            np.save(os.path.join(dirpath, npy_name), arr)
            data[field] = f"{_ARRAY_SENTINEL}{npy_name}"
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def _resolve_field(val, config_dir):
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


# Essential scalars — must come from a value or be projectable from a curve, else error.
_PROJECTABLE_FIELDS = ("scatter_length", "absorption_length", "qe")

# Mie-engine scalars (``mie_scatter_length``, ``g``) were added later; existing configs
# omit them. Project from the medium when one is referenced (its ``mie_scatter_coeff`` /
# ``mie_asymmetry``); otherwise fall back to a "no Mie" default — never raise and never
# leave them NaN (a NaN ``g`` poisons the expected-value photon step: effective_ratio =
# ... (1-g)/L_M → NaN → zero charge + NaN gradients). ``mie_scatter_length`` huge ⇒
# 1/L_M ≈ 0 (no Mie channel), which makes ``g`` immaterial.
_MIE_DEFAULTS = {"mie_scatter_length": 1.0e9, "g": 0.0}

# New calibrated-response / per-PMT placeholder fields (added with the nested
# refactor). Existing configs omit them entirely → loaded NaN → filled with a
# neutral default so the forward is unchanged. Per-sensor fields (gain/t0/walk)
# are kept scalar here and broadcast to (num_sensors,) by the caller.
_NEUTRAL_DEFAULTS = {
    "spe_width": 0.0,
    "tts": 0.0,
    "gain": 1.0,
    "t0": 0.0,
    "walk": 0.0,
}

# Wavelength λ-deviation curves (one value per control wavelength). Existing configs
# omit them → loaded NaN scalar → filled with an all-ones curve (≡ pure medium
# reference, byte-identical). A config may supply a curve as an inline list instead.
_DEV_CURVE_FIELDS = ("rayleigh_dev", "mie_dev", "abs_dev", "qe_dev")

# Scintillation scalars (refactor-v2 wbls/ice merge). Configs without a 'scintillation'
# block omit them → loaded NaN scalar → filled 0.0 (NEUTRAL: S=0 → no scint light; finite
# so normalize/bounds never poison). A scintillating material JSON supplies real values.
_SCINT_DEFAULTS = {
    "S": 0.0, "kB": 0.0, "C": 0.0, "tau_rise": 0.0, "tau_fall": 0.0,
    "moyal_amp": 0.0, "moyal_loc": 0.0, "moyal_scale": 0.0,
}


def _scintillation_defaults_from_medium(medium_model_path):
    """Pull DetectorParams-bound scintillation values from a material JSON.

    The material file (e.g. ``config/materials/wbls.json``) is the canonical home
    for the scintillator's physical numbers — the physics config inherits them so
    detector configs don't have to copy the WbLS spec verbatim. Returns an empty
    dict for non-scintillating media (no ``scintillation`` block, e.g. water) or
    when no material model is referenced — so water stays neutral (S=0).
    """
    if not medium_model_path:
        return {}
    with open(medium_model_path) as f:
        m = json.load(f)
    scint = m.get("scintillation")
    if not scint:
        return {}
    ly = scint.get("light_yield", {})
    tm = scint.get("timing", {})
    sp = scint.get("spectrum", {})
    pick = lambda d, k: d[k] if k in d else None
    out = {
        "S": pick(ly, "S"), "kB": pick(ly, "kB"), "C": pick(ly, "C"),
        "tau_rise": pick(tm, "tau_rise"), "tau_fall": pick(tm, "tau_fall"),
        "moyal_amp": pick(sp, "moyal_amp"),
        "moyal_loc": pick(sp, "moyal_loc"),
        "moyal_scale": pick(sp, "moyal_scale"),
    }
    return {k: v for k, v in out.items() if v is not None}

# Angular-reflection-model scalars (Schlick blacksheet + multilayer-Fresnel cathode).
# Inert unless reflection_model='angular'; configs omit them → filled with physical
# defaults. Bialkali cathode ~ n_r=2.8, n_k=1.5; blacksheet near-diffuse low-R0.
_ANGULAR_REFL_DEFAULTS = {
    "wall_R0": 0.05,
    "wall_p": 1.0,
    "wall_fspec": 0.0,
    "cathode_nr": 2.8,
    "cathode_nk": 1.5,
    "sensor_fspec": 0.0,
}


def _project_missing_scalars(kwargs, medium_path, qe_path, ref_wavelength_nm,
                             source_filepath):
    """Fill NaN scalar fields by evaluating wavelength curves at ``ref_wavelength_nm``.

    Modifies *kwargs* in place. Raises if an *essential* scalar is missing and no curve
    is available to project from. Mie scalars fall back to a no-Mie default instead.
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

    # Mie scalars: project from the medium if present, else default to "no Mie".
    for field, default in _MIE_DEFAULTS.items():
        v = kwargs[field]
        if v.ndim != 0 or not bool(jnp.isnan(v)):
            continue
        if medium_path is not None:
            from lucid.wavelength.medium import make_medium
            wl_grid = jnp.array([ref_wavelength_nm], dtype=jnp.float32)
            m = make_medium("water", wavelength_grid=wl_grid,
                            medium_model_path=medium_path)
            scalar = (float(m.mie_asymmetry) if field == "g"
                      else float(1.0 / (m.mie_scatter_coeff[0] + 1e-30)))
        else:
            scalar = default
        kwargs[field] = jnp.asarray(scalar, dtype=jnp.float32)

    # New response / per-PMT placeholder scalars: neutral defaults when absent.
    for field, default in _NEUTRAL_DEFAULTS.items():
        v = kwargs[field]
        if v.ndim == 0 and bool(jnp.isnan(v)):
            kwargs[field] = jnp.asarray(default, dtype=jnp.float32)

    # λ-deviation curves: an all-ones curve (≡ pure reference) when absent.
    for field in _DEV_CURVE_FIELDS:
        v = kwargs[field]
        if v.ndim == 0 and bool(jnp.isnan(v)):
            kwargs[field] = jnp.ones(_N_CTRL, dtype=jnp.float32)

    # Angular/mixture-reflection scalars: physical defaults when absent. Inert for the
    # 'scalar' model; CONSUMED by 'angular' and 'scalar_mix' — note the fspec defaults are
    # 0.0 (= fully diffuse under scalar_mix), so set wall_fspec/sensor_fspec explicitly in
    # the physics config when selecting reflection_model='scalar_mix'.
    for field, default in _ANGULAR_REFL_DEFAULTS.items():
        v = kwargs[field]
        if v.ndim == 0 and bool(jnp.isnan(v)):
            kwargs[field] = jnp.asarray(default, dtype=jnp.float32)

    # Scintillation scalars: neutral 0.0 when absent (non-scintillating media never read them).
    for field, default in _SCINT_DEFAULTS.items():
        v = kwargs[field]
        if v.ndim == 0 and bool(jnp.isnan(v)):
            kwargs[field] = jnp.asarray(default, dtype=jnp.float32)


def load_detector_params(filepath: str, num_sensors: int = None,
                         scalar_ref_wavelength: float = None) -> DetectorParams:
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


def load_physics_config(filepath: str, num_sensors: int = None,
                        scalar_ref_wavelength: float = None):
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

    # Read FLAT JSON keys for every leaf field. New response/per_pmt fields
    # (spe_width, tts, gain, t0, walk) are typically absent from existing configs;
    # _resolve_field returns NaN, which _project_missing_scalars fills with the
    # neutral default below.
    kwargs = {}
    for field in _FLAT_FIELDS:
        val = data.get(field, None)
        kwargs[field] = _resolve_field(val, config_dir)

    # Scintillation scalars: a scintillating material JSON (wbls) supplies the
    # physical S/kB/C/tau/moyal values; inherit them for any scint field the
    # config left absent (NaN). Water's material has no scintillation block →
    # this is a no-op and _project_missing_scalars fills the neutral 0.0.
    scint_defaults = _scintillation_defaults_from_medium(medium_model_path)
    for field, val in scint_defaults.items():
        v = kwargs.get(field)
        if v is not None and v.ndim == 0 and bool(jnp.isnan(v)):
            kwargs[field] = jnp.asarray(val, dtype=jnp.float32)

    _project_missing_scalars(kwargs, medium_model_path, qe_curve_path,
                             ref_wl, filepath)

    if num_sensors is not None:
        qe_corr = kwargs['qe_corrections']
        if qe_corr.ndim == 0:
            # NaN qe_corrections means the config omitted it entirely — default
            # to neutral (1.0) rather than poisoning the array.
            fill = jnp.where(jnp.isnan(qe_corr), jnp.float32(1.0), qe_corr)
            kwargs['qe_corrections'] = jnp.ones(num_sensors) * fill
        # Per-sensor placeholders (gain/t0/walk): broadcast scalar neutral
        # defaults to (num_sensors,) so the in-memory pytree is well-shaped.
        for field, neutral in (('gain', 1.0), ('t0', 0.0), ('walk', 0.0)):
            v = kwargs[field]
            if v.ndim == 0:
                kwargs[field] = jnp.full(num_sensors, neutral, dtype=jnp.float32)

    return _nest_flat_kwargs(kwargs), medium_model_path, qe_curve_path


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
        spe_width=jnp.array(0.0),
        tts=jnp.array(0.0),
        qe_dev=jnp.full(_N_CTRL, 0.1),
        qe_corrections=jnp.zeros(num_sensors),
        gain=jnp.full(num_sensors, 0.3),
        t0=jnp.full(num_sensors, -10.0),
        walk=jnp.full(num_sensors, -5.0),
        S=jnp.array(0.0), kB=jnp.array(0.0), C=jnp.array(0.0),       # scintillation (inert unless fit)
        tau_rise=jnp.array(0.0), tau_fall=jnp.array(0.0),
        moyal_amp=jnp.array(0.0), moyal_loc=jnp.array(0.0), moyal_scale=jnp.array(0.0),
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
        spe_width=jnp.array(1.0),
        tts=jnp.array(5.0),
        qe_dev=jnp.full(_N_CTRL, 10.0),
        qe_corrections=jnp.full(num_sensors, 2.0),
        gain=jnp.full(num_sensors, 3.0),
        t0=jnp.full(num_sensors, 10.0),
        walk=jnp.full(num_sensors, 5.0),
        S=jnp.array(5000.0), kB=jnp.array(1e-3), C=jnp.array(1e-6), # scintillation physical ranges
        tau_rise=jnp.array(10.0), tau_fall=jnp.array(50.0),
        moyal_amp=jnp.array(10.0), moyal_loc=jnp.array(600.0), moyal_scale=jnp.array(100.0),
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
        # their leaf fields (scatter_length, gain, tts, ...).
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
        wall_R0=jnp.array(_ANGULAR_REFL_DEFAULTS["wall_R0"]),
        wall_p=jnp.array(_ANGULAR_REFL_DEFAULTS["wall_p"]),
        wall_fspec=jnp.array(_ANGULAR_REFL_DEFAULTS["wall_fspec"]),
        cathode_nr=jnp.array(_ANGULAR_REFL_DEFAULTS["cathode_nr"]),
        cathode_nk=jnp.array(_ANGULAR_REFL_DEFAULTS["cathode_nk"]),
        sensor_fspec=jnp.array(_ANGULAR_REFL_DEFAULTS["sensor_fspec"]),
        qe=jnp.array(0.2),
        spe_width=jnp.array(0.0),
        tts=jnp.array(0.0),
        qe_dev=jnp.ones(_N_CTRL),
        qe_corrections=jnp.ones(num_sensors),
        gain=jnp.ones(num_sensors),
        t0=jnp.zeros(num_sensors),
        walk=jnp.zeros(num_sensors),
    ))


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
    return _nest_flat_kwargs(dict(
        scatter_length=jnp.array(1.0),
        mie_scatter_length=jnp.array(1.0),
        g=jnp.array(1.0),
        rayleigh_dev=jnp.ones(_N_CTRL),
        mie_dev=jnp.ones(_N_CTRL),
        absorption_length=jnp.array(1.0),
        abs_dev=jnp.ones(_N_CTRL),
        wall_reflection_rate=jnp.array(1.0),
        sensor_reflection_rate=jnp.array(1.0),
        wall_R0=jnp.array(1.0), wall_p=jnp.array(1.0), wall_fspec=jnp.array(1.0),
        cathode_nr=jnp.array(1.0), cathode_nk=jnp.array(1.0), sensor_fspec=jnp.array(1.0),
        qe=jnp.array(1.0),
        spe_width=jnp.array(1.0),
        tts=jnp.array(1.0),
        qe_dev=jnp.ones(_N_CTRL),
        qe_corrections=jnp.full(num_sensors, 0.1),
        gain=jnp.full(num_sensors, 0.1),
        t0=jnp.full(num_sensors, 0.1),
        walk=jnp.full(num_sensors, 0.1),
    ))
