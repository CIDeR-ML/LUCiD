"""Photosensor angular response epsilon(cos eta) — the ``angRespAll_<r>`` input.

fiTQun factorises the geometric term of the predicted charge as
``J = Omega(R) * T(R) * epsilon(cos eta)`` (``fiTQun.cc::J``), where ``Omega``
is the *unprojected* solid angle of the sensor face and ``epsilon`` carries
everything that depends on the angle of incidence — the projection of the face
onto the line of sight and whatever the photocathode does to light arriving
off-normal.

The reference measurement (``Utilities/angular/angularResponsePlotter.cc``)
isolates it by taking direct photons only and keeping just those whose source
sits in a thin spherical shell of radius ``r`` about the sensor: at fixed ``R``
both ``Omega(R)`` and ``T(R)`` are constant, so the ``cos eta`` spectrum of
detected photons *is* ``epsilon`` up to normalisation, which is fixed by
setting the normal-incidence bin to 1.

This module is the same reduction against LUCiD photons. Two things the caller
owns:

* **Direct light only.** The reference uses a WCSim fork that kills scattered
  and reflected photons; the LUCiD equivalent is to run the source scan with
  scattering and reflection off in the physics config.
* **Containment.** A shell is only usable if it lies entirely inside the
  detector, or the missing solid angle biases the spectrum. :func:`measure`
  enforces this the same way the reference does, from the detector extent.

The output is the histogram; the ``TPolyFunc`` fit that turns it into
``angResp_<config>.root`` stays with ``Utilities/angular/fit_cos.C``, which
owns the piecewise-polynomial form fiTQun expects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import binning, rootio


def cos_eta(source_pos: np.ndarray, sensor_pos: np.ndarray,
            sensor_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(R, cos eta)`` for photons travelling from source to sensor.

    ``sensor_dir`` is the inward-facing sensor axis, so a photon arriving
    head-on gives ``cos eta = 1``. All three arrays are per photon.
    """
    rel = np.asarray(source_pos, dtype=np.float64) - np.asarray(sensor_pos, dtype=np.float64)
    R = np.linalg.norm(rel, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.einsum("ij,ij->i", rel, np.asarray(sensor_dir, dtype=np.float64))
        c = np.where(R > 0, c / np.where(R > 0, R, 1.0), 1.0)
    return R, np.clip(c, -1.0, 1.0)


def measure(source_pos: np.ndarray, sensor_pos: np.ndarray, sensor_dir: np.ndarray,
            *, shell_r_cm: float, shell_dr_cm: float,
            det_radius_cm: float, det_halflength_cm: float,
            weights: Optional[np.ndarray] = None,
            n_bins: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Histogram ``cos eta`` for detected direct photons in one spherical shell.

    Returns ``(edges, counts, sumw2)``, unnormalised — merging several scans
    means adding the counts, so normalisation is left to :func:`normalise`.
    """
    R, c = cos_eta(source_pos, sensor_pos, sensor_dir)
    w = np.ones_like(R) if weights is None else np.asarray(weights, dtype=np.float64)

    in_shell = (R >= shell_r_cm - shell_dr_cm) & (R < shell_r_cm + shell_dr_cm)

    # The shell must be fully inside the detector in at least one of the two
    # directions that can clip it -- a barrel sensor is safe if the shell
    # clears the end caps, an end-cap sensor if it clears the barrel wall.
    reach = shell_r_cm + shell_dr_cm
    sensor_pos = np.asarray(sensor_pos, dtype=np.float64)
    clear_z = np.abs(sensor_pos[:, 2]) + reach <= det_halflength_cm
    clear_r = np.hypot(sensor_pos[:, 0], sensor_pos[:, 1]) + reach <= det_radius_cm
    if np.any(clear_z & clear_r):
        raise ValueError(
            "some sensors sit clear of both the barrel wall and the end caps, "
            "which no sensor of a cylindrical detector can -- check that the "
            "detector extent passed here matches the geometry the photons came from")
    contained = clear_z | clear_r

    keep = in_shell & contained
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(c[keep], bins=edges, weights=w[keep])
    sumw2, _ = np.histogram(c[keep], bins=edges, weights=w[keep] ** 2)
    return edges, counts, sumw2


def normalise(counts: np.ndarray, sumw2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale so normal incidence (the top bin) is 1, as ``fit_cos.C`` expects."""
    ref = counts[-1]
    if ref <= 0:
        raise ValueError("no photons at normal incidence; cannot normalise the response")
    return counts / ref, sumw2 / (ref * ref)


def write_angular_response(path, edges, counts, sumw2, *, shell_r_cm: float) -> Path:
    """Write the histogram under the name ``fit_cos.C`` looks up."""
    values, errs = normalise(counts, sumw2)
    name = f"angRespAll_{int(round(shell_r_cm))}"
    rootio.write(path, {name: rootio.th1(
        name, edges, values, sumw2=errs,
        title="Angular response function", xtitle="cos#eta")})
    return Path(path)
