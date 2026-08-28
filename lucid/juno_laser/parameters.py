"""Optical parameter definitions used by JUNO laser calibration."""

from __future__ import annotations

from typing import NamedTuple
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .config import JunoLaserConfig, PARAMETER_NAMES


class CalibParams(NamedTuple):
    """Traceable six-parameter optical bundle (plus optional global amplitude)."""

    n_ls: jnp.ndarray
    scat_ls: jnp.ndarray
    abs_ls: jnp.ndarray
    n_water: jnp.ndarray
    scat_water: jnp.ndarray
    abs_water: jnp.ndarray
    amp: jnp.ndarray


def unpack_params(theta_log, *, fit_amp: bool = False) -> CalibParams:
    """Map log-space parameters to a JAX pytree."""
    values = {name: jnp.exp(theta_log[i]) for i, name in enumerate(PARAMETER_NAMES)}
    values["amp"] = jnp.exp(theta_log[-1]) if fit_amp else jnp.asarray(1.0, dtype=jnp.float32)
    return CalibParams(**values)


def medium_lengths_at(material_paths: dict[str, Path], wavelength_nm: float):
    """Evaluate scalar Rayleigh, Mie and absorption lengths from material JSONs."""
    result = {}
    wl = float(wavelength_nm)
    for name, path in material_paths.items():
        with Path(path).open() as stream:
            data = json.load(stream)
        symmetric = data["scattering"]["symmetric"]
        alpha_sym = symmetric["P4"] / wl ** 4 * (1 + symmetric["P5"] / wl ** 2)
        asymmetric = data["scattering"]["asymmetric"]
        alpha_asym = asymmetric["P6"] * (
            1 + asymmetric["P7"] / wl ** 4 * (wl - asymmetric["P8"]) ** 2)
        absorption = data["absorption"]
        table_wl = np.asarray(absorption["pope_fry_wavelengths_nm"], dtype=float)
        table_alpha = np.asarray(absorption["pope_fry_absorption_m_inv"], dtype=float)
        if wl <= 464:
            alpha_continuum = absorption["P0"] * absorption["P2"] * (
                wl / 500.0) ** absorption["P3"]
        else:
            alpha_continuum = float(np.interp(wl, table_wl, table_alpha))
        alpha_abs = absorption["P0"] * absorption["P1"] / wl ** 4 + alpha_continuum
        result[name] = {
            "scat": 1.0 / alpha_sym,
            "mie": 1.0 / max(alpha_asym, 1e-30),
            "abs": 1.0 / max(alpha_abs, 1e-30),
            "n": float(data["refractive_index"]),
        }
    return result


def truth_from_geometry(config: JunoLaserConfig, det_geom):
    """Extract the v42 truth vector from geometry, physics and material files."""
    with config.physics_path.open() as stream:
        physics = json.load(stream)
    lengths = medium_lengths_at(config.material_paths, config.wavelength_nm)
    return {
        "n_ls": float(det_geom.medium.refractive_index),
        "scat_ls": float(physics["scatter_length"]),
        "abs_ls": float(physics["absorption_length"]),
        "n_water": float(det_geom.medium_outer.refractive_index),
        "scat_water": float(lengths["outer"]["scat"]),
        "abs_water": float(lengths["outer"]["abs"]),
    }

