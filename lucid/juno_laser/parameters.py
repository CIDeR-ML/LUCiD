"""Optical parameter definitions used by JUNO laser calibration."""

from __future__ import annotations

from typing import NamedTuple
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .config import (
    JunoLaserConfig,
    PARAMETER_NAMES,
    POSITIVE_PARAMETER_NAMES,
)
from .source import LSPHASE_SHAPE_NAMES


class CalibParams(NamedTuple):
    """Traceable fixed optical baseline with two replaceable LS lengths."""

    n_ls: jnp.ndarray
    scat_ls: jnp.ndarray
    abs_ls: jnp.ndarray
    n_water: jnp.ndarray
    scat_water: jnp.ndarray
    abs_water: jnp.ndarray
    amp: jnp.ndarray


def decode_calibration_vector(theta):
    """Decode optimization coordinates into physical calibration values."""
    theta = jnp.asarray(theta, dtype=jnp.float32)
    if theta.shape != (len(PARAMETER_NAMES),):
        raise ValueError(
            "calibration vector must have shape "
            f"({len(PARAMETER_NAMES)},), got {theta.shape}"
        )
    return {
        name: (
            jnp.exp(theta[index])
            if name in POSITIVE_PARAMETER_NAMES
            else theta[index]
        )
        for index, name in enumerate(PARAMETER_NAMES)
    }


def encode_calibration_values(values):
    """Encode physical calibration values in optimizer coordinates."""
    encoded = []
    for name in PARAMETER_NAMES:
        if name not in values:
            raise ValueError(f"missing calibration value {name!r}")
        value = float(values[name])
        if name in POSITIVE_PARAMETER_NAMES:
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            value = np.log(value)
        encoded.append(value)
    return jnp.asarray(encoded, dtype=jnp.float32)


def decode_calibration_array(theta):
    """Return decoded values ordered like :data:`PARAMETER_NAMES`."""
    values = decode_calibration_vector(theta)
    return jnp.stack(tuple(values[name] for name in PARAMETER_NAMES))


def optimization_bounds(bounds):
    """Transform physical parameter bounds into optimizer coordinates."""
    lower = []
    upper = []
    for name in PARAMETER_NAMES:
        lo, hi = bounds[name]
        if name in POSITIVE_PARAMETER_NAMES:
            lo, hi = np.log(lo), np.log(hi)
        lower.append(lo)
        upper.append(hi)
    return (
        jnp.asarray(lower, dtype=jnp.float32),
        jnp.asarray(upper, dtype=jnp.float32),
    )


def fixed_params_from_geometry(config, det_geom):
    """Build the fixed optical baseline used around the five fit values."""
    truth = truth_from_geometry(config, det_geom)
    dtype = jnp.float32
    return CalibParams(
        n_ls=jnp.asarray(truth["n_ls"], dtype=dtype),
        scat_ls=jnp.asarray(truth["scat_ls"], dtype=dtype),
        abs_ls=jnp.asarray(truth["abs_ls"], dtype=dtype),
        n_water=jnp.asarray(truth["n_water"], dtype=dtype),
        scat_water=jnp.asarray(truth["scat_water"], dtype=dtype),
        abs_water=jnp.asarray(truth["abs_water"], dtype=dtype),
        amp=jnp.asarray(1.0, dtype=dtype),
    )


def unpack_params(theta, fixed_params):
    """Insert fitted LS lengths into a fixed optical parameter bundle."""
    values = decode_calibration_vector(theta)
    params = fixed_params._replace(
        scat_ls=values["scat_ls"],
        abs_ls=values["abs_ls"],
    )
    shape_coordinates = jnp.stack(
        tuple(values[name] for name in LSPHASE_SHAPE_NAMES)
    )
    return params, shape_coordinates


def medium_lengths_at(material_paths: dict[str, Path], wavelength_nm: float):
    """Evaluate scalar scattering and absorption lengths from material JSON."""
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
        table_wl = np.asarray(
            absorption["pope_fry_wavelengths_nm"], dtype=float
        )
        table_alpha = np.asarray(
            absorption["pope_fry_absorption_m_inv"], dtype=float
        )
        if wl <= 464:
            alpha_continuum = absorption["P0"] * absorption["P2"] * (
                wl / 500.0) ** absorption["P3"]
        else:
            alpha_continuum = float(np.interp(wl, table_wl, table_alpha))
        alpha_abs = (
            absorption["P0"] * absorption["P1"] / wl ** 4
            + alpha_continuum
        )
        result[name] = {
            "scat": 1.0 / alpha_sym,
            "mie": 1.0 / max(alpha_asym, 1e-30),
            "abs": 1.0 / max(alpha_abs, 1e-30),
            "n": float(data["refractive_index"]),
        }
    return result


def truth_from_geometry(config: JunoLaserConfig, det_geom):
    """Extract the reference truth vector from geometry and material files."""
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
        **{name: 0.0 for name in LSPHASE_SHAPE_NAMES},
    }
