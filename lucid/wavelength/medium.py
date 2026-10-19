"""Medium optical properties (material physics, NOT detector hardware).

MediumProperties holds the physical properties of the detection medium
(water, ice): refractive index, wavelength-dependent scattering and
absorption coefficients. QE is a detector property and lives elsewhere.
"""
import json
import os
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np


class MediumProperties(NamedTuple):
    """Optical properties of the detection medium.

    Scalar fields are always populated.  Wavelength-dependent arrays are
    ``None`` when running in monochromatic mode and populated when
    wavelength support is active.
    """
    material: str
    refractive_index: float
    speed_of_light: float                              # m/ns in this medium

    # Wavelength-dependent arrays (None = monochromatic mode)
    wavelength_grid: Optional[jnp.ndarray] = None      # (N,) nm
    scatter_coeff: Optional[jnp.ndarray] = None         # (N,) symmetric Rayleigh 1/m
    absorption_coeff: Optional[jnp.ndarray] = None      # (N,) 1/m
    refractive_index_curve: Optional[jnp.ndarray] = None  # (N,) n(lambda)
    mie_scatter_coeff: Optional[jnp.ndarray] = None     # (N,) asymmetric Mie 1/m
    mie_asymmetry: Optional[float] = None               # HG g parameter

    # --- Emission dispatch (static, non-differentiable; the multi-material merge) ---
    # APPENDED LAST so existing positional MediumProperties(...) construction is unchanged.
    # Differentiable scintillation values (S/kB/C/tau/moyal) live on DetectorParams; these are
    # the setup-time knobs the simulator reads to pick Cherenkov vs scintillation surrogates.
    emission_processes: tuple = ("cherenkov",)          # ("cherenkov",) | ("cherenkov","scintillation")
    scintillation_lambda_min: float = 340.0             # Moyal sampling window (nm)
    scintillation_lambda_max: float = 550.0
    cherenkov_fraction: float = 0.5                     # n_photons split when both processes run

    # Dispersion is setup-time/static; the curve is a diagnostic sample of the
    # analytic model used by ``group_velocity_m_per_ns``.
    dispersion_model: str = "constant"
    group_refractive_index_curve: Optional[jnp.ndarray] = None


# ---------------------------------------------------------------------------
# Refractive-index dispersion and group velocity
# ---------------------------------------------------------------------------

# SK-like water dispersion at 13 C and 1.01325 bar. The phase index n
# determines the group index n_g = n - lambda * dn/dlambda used for optical
# time-of-flight updates.
_SK4_TEMP_C = 13.0
_SK4_PRESSURE_BAR = 1.01325
_SK4_TEMP_REF_C = 19.993
_SK4_PRESSURE_REF_BAR = 1.01325
_SK4_WLA_UM2 = 0.018085
_SK4_A = (5.743534e-3, 1.769238, -2.797222e-2, 8.715348e-3, -1.413942e-3)
_SK4_B = (-8.454823e-5, -2.787742e-5, 2.608176e-6,
          -2.050671e-6, 1.019989e-6, -2.611919e-6,
          8.194989e-9, -8.10707e-9, 4.877274e-8)
_SK4_C = (8.419632e-6, 1.941681e-5, -7.762524e-8,
          4.371257e-8, 7.089664e-9, -2.240384e-8)


def sk4_water_refractive_index(wavelength_nm):
    """SK-like phase refractive index for water at 13 C and 1.01325 bar.

    Uses only JAX operations, so wavelength derivatives work in forward
    and reverse mode.
    """
    wl = jnp.asarray(wavelength_nm) / 1000.0  # nm -> micrometres
    a1, a2, a3, a4, a5 = _SK4_A
    b1, b2, b3, b4, b5, b6, b7, b8, b9 = _SK4_B
    c1, c2, c3, c4, c5, c6 = _SK4_C
    dt = _SK4_TEMP_C - _SK4_TEMP_REF_C
    dp = _SK4_PRESSURE_BAR - _SK4_PRESSURE_REF_BAR
    wl2 = wl * wl
    wl4 = wl2 * wl2
    wl6 = wl4 * wl2
    base = jnp.sqrt(
        a1 / (wl2 - _SK4_WLA_UM2) + a2 + a3 * wl2 + a4 * wl4 + a5 * wl6)
    return (
        base
        + (b1 + b2 * wl2 + b3 * wl4) * dt
        + (b4 + b5 * wl2 + b6 * wl4) * dt**2
        + (b7 + b8 * wl2 + b9 * wl4) * dt**3
        + (c1 + c2 * wl2 + (c3 + c4 * wl2) * _SK4_TEMP_C) * dp
        + (c5 + c6 * wl2) * dp**2
    )


def sk4_water_refractive_index_derivative(wavelength_nm):
    """Analytic derivative of the SK-like water phase index, ``dn/dlambda`` per nm."""
    wl = jnp.asarray(wavelength_nm) / 1000.0
    a1, _, a3, a4, a5 = _SK4_A
    _, b2, b3, _, b5, b6, _, b8, b9 = _SK4_B
    _, c2, _, c4, _, c6 = _SK4_C
    dt = _SK4_TEMP_C - _SK4_TEMP_REF_C
    dp = _SK4_PRESSURE_BAR - _SK4_PRESSURE_REF_BAR
    wl2 = wl * wl
    wl3 = wl2 * wl
    wl4 = wl2 * wl2
    wl5 = wl4 * wl
    root = jnp.sqrt(
        _SK4_A[0] / (wl2 - _SK4_WLA_UM2)
        + _SK4_A[1] + _SK4_A[2] * wl2 + _SK4_A[3] * wl4
        + _SK4_A[4] * wl4 * wl2)
    derivative_per_um = (
        (-2.0 * a1 * wl / (wl2 - _SK4_WLA_UM2)**2
         + 2.0 * a3 * wl + 4.0 * a4 * wl3 + 6.0 * a5 * wl5)
        / (2.0 * root)
        + (2.0 * b2 * wl + 4.0 * b3 * wl3) * dt
        + (2.0 * b5 * wl + 4.0 * b6 * wl3) * dt**2
        + (2.0 * b8 * wl + 4.0 * b9 * wl3) * dt**3
        + (2.0 * c2 * wl + 2.0 * c4 * wl * _SK4_TEMP_C) * dp
        + 2.0 * c6 * wl * dp**2
    )
    return derivative_per_um / 1000.0


def sk4_water_group_refractive_index(wavelength_nm):
    """SK-like water group index: ``n_g = n - lambda * dn/dlambda``."""
    wavelength_nm = jnp.asarray(wavelength_nm)
    return (sk4_water_refractive_index(wavelength_nm)
            - wavelength_nm * sk4_water_refractive_index_derivative(wavelength_nm))


def group_velocity_m_per_ns(wavelength_nm, medium):
    """Return differentiable group velocity for each wavelength."""
    wavelength_nm = jnp.asarray(wavelength_nm)
    if medium.dispersion_model == "sk4_water":
        c_vacuum = medium.speed_of_light * medium.refractive_index
        return c_vacuum / sk4_water_group_refractive_index(wavelength_nm)
    if medium.dispersion_model == "constant":
        return jnp.full_like(wavelength_nm, medium.speed_of_light)
    raise ValueError(f"Unknown dispersion model {medium.dispersion_model!r}")


# ---------------------------------------------------------------------------
# QE curve loading
# ---------------------------------------------------------------------------

def load_qe_curve(json_path):
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


def qe_curve_bounds(json_path):
    """Return the (min, max) wavelength knots from a QE-curve JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    wavelengths = np.array(data["wavelengths_nm"])
    return float(wavelengths.min()), float(wavelengths.max())


def qe_curve_peak(json_path):
    """Return the largest QE (fraction) of a QE-curve JSON file: the curve's maximum,
    since ``load_qe_curve`` interpolates linearly between knots."""
    with open(json_path) as f:
        data = json.load(f)
    return float(np.max(data["qe_percent"])) / 100.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _load_medium_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


# Legacy default path (used when no explicit path is given)
_LEGACY_WATER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "materials", "water.json")
# Default material lookup dir: config/materials/<material>.json (multi-material merge).
_MATERIALS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config", "materials")


def _scintillation_kwargs_from(data: dict) -> dict:
    """Extract the STATIC scintillation dispatch knobs from a material JSON.

    The differentiable values (light yield S/kB/C, timing tau, spectrum moyal) live on
    DetectorParams and are sourced via ``load_physics_config``; this returns only what the
    simulator needs at setup to pick surrogates + the Moyal window. A material is
    non-scintillating if it lacks a ``"scintillation"`` block (→ Cherenkov-only defaults).
    """
    scint = data.get("scintillation")
    if scint is None:
        return {"emission_processes": ("cherenkov",)}
    procs = tuple(scint.get("emission_processes", ("cherenkov", "scintillation")))
    spec = scint.get("spectrum", {})
    split = scint.get("photon_split", {"cherenkov_fraction": 0.5})
    return {
        "emission_processes": procs,
        "scintillation_lambda_min": float(spec.get("lambda_min", 340.0)),
        "scintillation_lambda_max": float(spec.get("lambda_max", 550.0)),
        "cherenkov_fraction": float(split["cherenkov_fraction"]),
    }


def make_medium(material: str = "water",
                wavelength_grid: Optional[jnp.ndarray] = None,
                medium_model_path: str = None) -> MediumProperties:
    """Build a MediumProperties from a material name or model file.

    Parameters
    ----------
    material : str
        Material name. Used to locate ``config/materials/<material>.json`` when no explicit
        ``medium_model_path`` is given (e.g. ``"water"``, ``"ice"``, ``"wbls"``).
    wavelength_grid : jnp.ndarray, optional
        If provided, wavelength-dependent arrays are computed on this grid.
        If ``None``, only scalar properties are populated (monochromatic mode).
    medium_model_path : str, optional
        Explicit path to the medium model JSON file. If ``None``, resolves
        ``config/materials/<material>.json`` (water falls back to the legacy bundled path).

    Returns
    -------
    MediumProperties
    """
    if medium_model_path is not None:
        data = _load_medium_json(medium_model_path)
    else:
        path = os.path.join(_MATERIALS_DIR, f"{material}.json")
        if not os.path.exists(path):
            if material == "water":
                path = _LEGACY_WATER_PATH
            else:
                raise ValueError(f"No material config at {path!r}. Add a "
                                 f"config/materials/{material}.json or pass medium_model_path.")
        data = _load_medium_json(path)

    scint_kwargs = _scintillation_kwargs_from(data)     # static emission dispatch
    n = data["refractive_index"]
    c_vac = data["speed_of_light_vacuum_m_per_ns"]
    c_medium = c_vac / n
    dispersion_model = data.get("dispersion", {}).get("model", "constant")

    if wavelength_grid is None:
        return MediumProperties(
            material=material,
            refractive_index=n,
            speed_of_light=c_medium,
            dispersion_model=dispersion_model,
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

    # Eq.(15) power-law fit (blue) spliced onto the Pope&Fry data (red) at 464 nm.
    # The two pieces differ by ~12% at the seam, so a hard switch leaves a kink; blend
    # them with a C2 smootherstep over a +-12 nm window. (The blend WEIGHT is C2; the
    # result inherits the piecewise-linear knot kinks of the interpolated Pope&Fry
    # table itself — tiny, but not strictly C2.)
    C_powerlaw = P0 * P2 * (wl / 500.0) ** P3
    C_popefry = jnp.interp(wl, pf_wl, pf_abs)
    _bw = 12.0
    _t = jnp.clip((wl - (464.0 - _bw)) / (2.0 * _bw), 0.0, 1.0)
    _s = _t * _t * _t * (_t * (_t * 6.0 - 15.0) + 10.0)   # smootherstep (C2)
    C = (1.0 - _s) * C_powerlaw + _s * C_popefry
    alpha_abs = P0 * P1 / (wl ** 4) + C

    if dispersion_model == "sk4_water":
        refractive_index_curve = sk4_water_refractive_index(wl)
        group_refractive_index_curve = sk4_water_group_refractive_index(wl)
    elif dispersion_model == "constant":
        refractive_index_curve = jnp.full_like(wl, n)
        group_refractive_index_curve = jnp.full_like(wl, n)
    else:
        raise ValueError(f"Unknown dispersion model {dispersion_model!r}")

    return MediumProperties(
        material=material,
        refractive_index=n,
        speed_of_light=c_medium,
        wavelength_grid=wl,
        scatter_coeff=alpha_sym,
        absorption_coeff=alpha_abs,
        refractive_index_curve=refractive_index_curve,
        mie_scatter_coeff=alpha_asym,
        mie_asymmetry=g,
        dispersion_model=dispersion_model,
        group_refractive_index_curve=group_refractive_index_curve,
        **scint_kwargs,
    )


# ---------------------------------------------------------------------------
# Effective property derivation
# ---------------------------------------------------------------------------

def compute_effective_properties(detector_params, medium, wavelengths=None,
                                 qe_curve=None):
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
        return (detector_params.scattering.scatter_length,
                detector_params.scattering.mie_scatter_length,
                detector_params.absorption.absorption_length,
                detector_params.response.qe)

    # Wavelength-active: per-photon corrections
    # Reference wavelength (fixed at Cherenkov peak, not grid midpoint)
    ref_wl = 400.0  # nm

    wl_grid = medium.wavelength_grid
    scatter_at_wl = jnp.interp(wavelengths, wl_grid, medium.scatter_coeff)
    mie_scatter_at_wl = jnp.interp(wavelengths, wl_grid, medium.mie_scatter_coeff)
    scatter_at_ref = jnp.interp(ref_wl, wl_grid, medium.scatter_coeff)
    mie_scatter_at_ref = jnp.interp(ref_wl, wl_grid, medium.mie_scatter_coeff)
    scatter_correction = scatter_at_ref / (scatter_at_wl + 1e-30)
    mie_scatter_correction = mie_scatter_at_ref / (mie_scatter_at_wl + 1e-30)


    abs_at_wl = jnp.interp(wavelengths, wl_grid, medium.absorption_coeff)
    abs_at_ref = jnp.interp(ref_wl, wl_grid, medium.absorption_coeff)
    abs_correction = abs_at_ref / (abs_at_wl + 1e-30)

    eff_scatter = detector_params.scattering.scatter_length * scatter_correction
    eff_absorption = detector_params.absorption.absorption_length * abs_correction
    eff_mie_scatter = detector_params.scattering.mie_scatter_length * mie_scatter_correction

    if qe_curve is not None:
        # response.qe is the QE at ref_wl; the curve supplies the shape around it,
        # as the scattering and absorption corrections above do.
        eff_qe = detector_params.response.qe * qe_curve(wavelengths) / qe_curve(ref_wl)
    else:
        eff_qe = detector_params.response.qe

    return eff_scatter, eff_mie_scatter, eff_absorption, eff_qe
