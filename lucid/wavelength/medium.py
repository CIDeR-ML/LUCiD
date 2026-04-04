"""Medium optical properties (material physics, NOT detector hardware).

MediumProperties holds the physical properties of the detection medium
(water, ice): refractive index, wavelength-dependent scattering and
absorption coefficients. QE is a detector property and lives elsewhere.
"""
import json
import os
from typing import NamedTuple, Optional

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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_water_json():
    path = os.path.join(_DATA_DIR, "water.json")
    with open(path, "r") as f:
        return json.load(f)


def make_medium(material: str = "water",
                wavelength_grid: Optional[jnp.ndarray] = None) -> MediumProperties:
    """Build a MediumProperties from a material name.

    Parameters
    ----------
    material : str
        Currently only ``"water"`` is supported.
    wavelength_grid : jnp.ndarray, optional
        If provided, wavelength-dependent arrays are computed on this grid.
        If ``None``, only scalar properties are populated (monochromatic mode).

    Returns
    -------
    MediumProperties
    """
    if material != "water":
        raise ValueError(f"Unknown material '{material}'. Supported: 'water'.")

    data = _load_water_json()
    n = data["refractive_index"]
    c_vac = data["speed_of_light_vacuum_m_per_ns"]
    c_medium = c_vac / n

    if wavelength_grid is None:
        return MediumProperties(
            material=material,
            refractive_index=n,
            speed_of_light=c_medium,
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
    C = jnp.where(wl <= 464.0, C_powerlaw, C_popefry)
    alpha_abs = P0 * P1 / (wl ** 4) + C

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
        return (detector_params.scatter_length,
                detector_params.absorption_length,
                detector_params.qe)

    # Wavelength-active: per-photon corrections
    # Reference wavelength index (use medium's grid midpoint)
    ref_wl = 400.0  # nm — typical Cherenkov peak

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
