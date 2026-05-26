"""Medium optical properties (material physics, NOT detector hardware).

MediumProperties holds the physical properties of the detection medium
(water, WbLS, ice): refractive index, wavelength-dependent scattering and
absorption coefficients, and — for scintillating media — the light-yield
model, emission timing, and emission spectrum. QE is a detector property
and lives elsewhere.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np


class MediumProperties(NamedTuple):
    """Optical properties of the detection medium.

    Scalar fields are always populated. Wavelength-dependent arrays are
    ``None`` when running in monochromatic mode and populated when
    wavelength support is active.

    Scintillation fields (``S``, ``kB``, ``C``, ``tau_*``, ``R_1``,
    ``moyal_*``) are differentiable ``jax.Array`` scalars — only read when
    ``"scintillation"`` is in ``emission_processes``. For non-scintillating
    media they default to NaN to fail loud if accidentally used.
    """
    material: str
    refractive_index: float
    speed_of_light: float                              # m/ns in this medium

    # --- Emission-process toggle ------------------------------------------
    # Tuple of process names enabled for this medium. Order is fixed: the
    # simulator concatenates rays in this order when both processes run.
    emission_processes: tuple = ("cherenkov",)

    # --- Wavelength-dependent arrays (None = monochromatic mode) ---------
    wavelength_grid: Optional[jax.Array] = None      # (N,) nm
    scatter_coeff: Optional[jax.Array] = None         # (N,) symmetric Rayleigh 1/m
    absorption_coeff: Optional[jax.Array] = None      # (N,) 1/m
    refractive_index_curve: Optional[jax.Array] = None  # (N,) n(lambda)
    mie_scatter_coeff: Optional[jax.Array] = None     # (N,) asymmetric Mie 1/m
    mie_asymmetry: Optional[float] = None               # HG g parameter

    # --- Scintillation: Chou light-yield model (3 differentiable params) -
    # Chou: dL/dx = S * (dE/dx) / [1 + kB*(dE/dx) + C*(dE/dx)²]. Birks-only
    # is recovered by setting C=0. Units: S [ph/MeV], kB [mm/keV], C [(mm/keV)²].
    S:  jax.Array = jnp.nan
    kB: jax.Array = jnp.nan
    C:  jax.Array = jnp.nan

    # --- Scintillation: biexponential timing (4 differentiable params) ----
    # p(t) = R_1 * g(t; tau_r, tau_1) + (1-R_1) * g(t; tau_r, tau_2),
    # g(t; tau_r, tau_d) = (exp(-t/tau_d) - exp(-t/tau_r)) / (tau_d - tau_r).
    # Units: tau_* in ns; R_1 dimensionless ∈ [0,1].
    tau_r: jax.Array = jnp.nan
    tau_1: jax.Array = jnp.nan
    tau_2: jax.Array = jnp.nan
    R_1:   jax.Array = jnp.nan

    # --- Scintillation: emission spectrum (3 differentiable Moyal params)-
    # p(λ) = amp * moyal_pdf(λ; loc, scale), evaluated on the window
    # [scintillation_lambda_min, scintillation_lambda_max] nm; outside that
    # window the ray weight is forced to zero.
    moyal_amp:   jax.Array = jnp.nan
    moyal_loc:   jax.Array = jnp.nan
    moyal_scale: jax.Array = jnp.nan

    # --- Scintillation: static knobs (not differentiable) -----------------
    scintillation_lambda_min: float = 340.0
    scintillation_lambda_max: float = 550.0
    # When BOTH cherenkov and scintillation are enabled, this fraction of
    # the user's `n_photons` budget goes to Cherenkov; the rest to
    # scintillation (floored to a multiple of 5 internally for the
    # 5-time-twin scheme).
    cherenkov_fraction: float = 0.5


# --- JAX pytree registration ------------------------------------------------
# MediumProperties carries both differentiable jax.Array scalars (the
# scintillation gradient targets) and static metadata (the string `material`,
# the `emission_processes` tuple, the lambda window, the cherenkov split).
# JAX's default NamedTuple-as-pytree treatment puts every field on the leaf
# side, which fails the moment a string field hits a jit'd function.
#
# Registering it explicitly lets us route the metadata through aux_data
# (static) and the rest through children (traced) — gradients flow through
# the array fields, the strings/tuples stay Python-level constants.

_MEDIUM_STATIC_FIELDS = (
    'material',
    'emission_processes',
    'scintillation_lambda_min',
    'scintillation_lambda_max',
    'cherenkov_fraction',
    'mie_asymmetry',          # Python float, not differentiable
)
_MEDIUM_DYNAMIC_FIELDS = tuple(
    f for f in MediumProperties._fields if f not in _MEDIUM_STATIC_FIELDS
)


def _medium_tree_flatten(m: MediumProperties):
    children = tuple(getattr(m, f) for f in _MEDIUM_DYNAMIC_FIELDS)
    aux_data = tuple(getattr(m, f) for f in _MEDIUM_STATIC_FIELDS)
    return children, aux_data


def _medium_tree_unflatten(aux_data, children) -> MediumProperties:
    kwargs = dict(zip(_MEDIUM_STATIC_FIELDS, aux_data))
    kwargs.update(zip(_MEDIUM_DYNAMIC_FIELDS, children))
    return MediumProperties(**kwargs)


jax.tree_util.register_pytree_node(
    MediumProperties, _medium_tree_flatten, _medium_tree_unflatten,
)


# ---------------------------------------------------------------------------
# QE curve loading
# ---------------------------------------------------------------------------

def load_qe_curve(json_path: str) -> Callable[[jax.Array], jax.Array]:
    """Load PMT QE curve and return a JAX-compatible interpolation function.

    Parameters
    ----------
    json_path : str
        Path to a JSON file with keys ``wavelengths_nm`` and ``qe_percent``.

    Returns
    -------
    callable
        ``qe_fn(wavelength_nm) -> qe_fraction`` (0 to 1).
        Returns 0 outside the tabulated range.
    """
    with open(json_path) as f:
        data = json.load(f)

    wavelengths = np.array(data["wavelengths_nm"])
    qe_pct = np.array(data["qe_percent"])

    sort_idx = np.argsort(wavelengths)
    wl_knots = jnp.array(wavelengths[sort_idx])
    qe_knots = jnp.array(qe_pct[sort_idx] / 100.0)  # percent -> fraction
    wl_min = float(wl_knots[0])
    wl_max = float(wl_knots[-1])

    @jax.jit
    def get_qe(wavelength):
        qe = jnp.interp(wavelength, wl_knots, qe_knots)
        in_range = (wavelength >= wl_min) & (wavelength <= wl_max)
        return jnp.where(in_range, qe, 0.0)

    return get_qe


def qe_curve_bounds(json_path: str) -> tuple[float, float]:
    """Return the (min, max) wavelength knots from a QE-curve JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    wavelengths = np.array(data["wavelengths_nm"])
    return float(wavelengths.min()), float(wavelengths.max())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _load_medium_json(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


# Default lookup: config/materials/<material>.json relative to repo root.
_MATERIALS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config", "materials")


def _scintillation_kwargs_from(data: dict) -> dict:
    """Extract scintillation kwargs from a material JSON, or return defaults.

    A material is non-scintillating if it lacks a ``"scintillation"`` block
    or that block does not include ``"scintillation"`` in
    ``emission_processes``. In that case all 10 differentiable scalars stay
    NaN (sentinel) — code paths that read them assert via
    ``"scintillation" in medium.emission_processes`` first.
    """
    scint = data.get("scintillation")
    if scint is None:
        return {"emission_processes": ("cherenkov",)}

    ly      = scint["light_yield"]      # {"S", "kB", "C"}
    timing  = scint["timing"]           # {"tau_r", "tau_1", "tau_2", "R_1"}
    spec    = scint["spectrum"]         # {"moyal_amp", "moyal_loc", "moyal_scale", "lambda_min", "lambda_max"}
    split   = scint.get("photon_split", {"cherenkov_fraction": 0.5})

    f32 = lambda v: jnp.asarray(float(v), dtype=jnp.float32)

    procs = tuple(scint.get("emission_processes", ("cherenkov", "scintillation")))
    return {
        "emission_processes":         procs,
        "S":  f32(ly["S"]),  "kB": f32(ly["kB"]), "C": f32(ly["C"]),
        "tau_r": f32(timing["tau_r"]), "tau_1": f32(timing["tau_1"]),
        "tau_2": f32(timing["tau_2"]), "R_1":   f32(timing["R_1"]),
        "moyal_amp":   f32(spec["moyal_amp"]),
        "moyal_loc":   f32(spec["moyal_loc"]),
        "moyal_scale": f32(spec["moyal_scale"]),
        "scintillation_lambda_min": float(spec["lambda_min"]),
        "scintillation_lambda_max": float(spec["lambda_max"]),
        "cherenkov_fraction":       float(split["cherenkov_fraction"]),
    }


def make_medium(material: str = "water",
                wavelength_grid: jax.Array | None = None,
                medium_model_path: str | None = None) -> MediumProperties:
    """Build a MediumProperties from a material name or model file.

    Parameters
    ----------
    material : str
        Material name. Used to locate ``config/materials/<material>.json``
        when ``medium_model_path`` is not given.
    wavelength_grid : jnp.ndarray, optional
        If provided, wavelength-dependent arrays are computed on this grid.
        If ``None``, only scalar properties are populated (monochromatic mode).
    medium_model_path : str, optional
        Explicit path to the medium model JSON file. If ``None``, falls back
        to ``config/materials/<material>.json``.

    Returns
    -------
    MediumProperties
    """
    if medium_model_path is not None:
        data = _load_medium_json(medium_model_path)
    else:
        path = os.path.join(_MATERIALS_DIR, f"{material}.json")
        if not os.path.exists(path):
            raise ValueError(f"No material config at {path!r}. Add a "
                             f"config/materials/{material}.json or pass an "
                             f"explicit medium_model_path.")
        data = _load_medium_json(path)

    n = data["refractive_index"]
    c_vac = data["speed_of_light_vacuum_m_per_ns"]
    c_medium = c_vac / n

    scint_kwargs = _scintillation_kwargs_from(data)

    if wavelength_grid is None:
        return MediumProperties(
            material=material,
            refractive_index=n,
            speed_of_light=c_medium,
            **scint_kwargs,
        )

    wl = jnp.asarray(wavelength_grid, dtype=jnp.float32)

    # Symmetric (Rayleigh) scattering coefficient
    sc = data["scattering"]["symmetric"]
    P4, P5 = sc["P4"], sc["P5"]
    alpha_sym = P4 / (wl ** 4) * (1.0 + P5 / (wl ** 2))

    # Asymmetric (Mie) scattering coefficient
    ac = data["scattering"]["asymmetric"]
    P6, P7, P8 = ac["P6"], ac["P7"], ac["P8"]
    alpha_asym = P6 * (1.0 + P7 / (wl ** 4) * (wl - P8) ** 2)
    g = ac["default_asymmetry_g"]

    # Absorption coefficient
    ab = data["absorption"]
    P0, P1, P2, P3 = ab["P0"], ab["P1"], ab["P2"], ab["P3"]
    pf_wl = jnp.array(ab["pope_fry_wavelengths_nm"], dtype=jnp.float32)
    pf_abs = jnp.array(ab["pope_fry_absorption_m_inv"], dtype=jnp.float32)

    C_powerlaw = P0 * P2 * (wl / 500.0) ** P3
    C_popefry = jnp.interp(wl, pf_wl, pf_abs)
    C_abs = jnp.where(wl <= 464.0, C_powerlaw, C_popefry)
    alpha_abs = P0 * P1 / (wl ** 4) + C_abs

    return MediumProperties(
        material=material,
        refractive_index=n,
        speed_of_light=c_medium,
        wavelength_grid=wl,
        scatter_coeff=alpha_sym,
        absorption_coeff=alpha_abs,
        refractive_index_curve=jnp.full_like(wl, n),  # constant n for now
        mie_scatter_coeff=alpha_asym,
        mie_asymmetry=g,
        **scint_kwargs,
    )


# ---------------------------------------------------------------------------
# Effective property derivation
# ---------------------------------------------------------------------------

def compute_effective_properties(
    detector_params: Any, medium: MediumProperties,
    wavelengths: jax.Array | None = None,
    qe_curve: Callable[[jax.Array], jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Derive per-photon effective scatter/absorption/QE from calibration
    scalars, medium corrections, and optional wavelength arrays.

    For monochromatic mode (``wavelengths is None``), the returned values
    are simply the scalar ``DetectorParams`` fields (corrections = 1.0).

    Parameters
    ----------
    detector_params : DetectorParams
        Calibration scalars (scatter_length, absorption_length, qe).
    medium : MediumProperties
        Medium optical data (may or may not have wavelength arrays).
    wavelengths : jnp.ndarray, optional
        Per-photon wavelengths in nm, shape ``(n_photons,)``.
    qe_curve : callable, optional
        ``qe_curve(wavelength) -> qe_fraction``.

    Returns
    -------
    eff_scatter : float or jnp.ndarray
        Effective scattering length (m).
    eff_absorption : float or jnp.ndarray
        Effective absorption length (m).
    eff_qe : float or jnp.ndarray
        Effective quantum efficiency.
    """
    if wavelengths is None:
        # Monochromatic: scalar passthrough
        return (detector_params.scatter_length,
                detector_params.absorption_length,
                detector_params.qe)

    # Wavelength-active: per-photon corrections
    # Reference wavelength (fixed at Cherenkov peak, not grid midpoint)
    ref_wl = 400.0  # nm

    wl_grid = medium.wavelength_grid
    scatter_at_wl = jnp.interp(wavelengths, wl_grid, medium.scatter_coeff)
    scatter_at_ref = jnp.interp(ref_wl, wl_grid, medium.scatter_coeff)
    scatter_correction = scatter_at_ref / (scatter_at_wl + 1e-30)

    abs_at_wl = jnp.interp(wavelengths, wl_grid, medium.absorption_coeff)
    abs_at_ref = jnp.interp(ref_wl, wl_grid, medium.absorption_coeff)
    abs_correction = abs_at_ref / (abs_at_wl + 1e-30)

    eff_scatter = detector_params.scatter_length * scatter_correction
    eff_absorption = detector_params.absorption_length * abs_correction

    if qe_curve is not None:
        eff_qe = detector_params.qe * qe_curve(wavelengths)
    else:
        eff_qe = detector_params.qe

    return eff_scatter, eff_absorption, eff_qe
