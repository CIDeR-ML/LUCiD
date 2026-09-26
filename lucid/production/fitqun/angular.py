"""Photosensor angular response epsilon(cos eta) — the ``angRespAll_<r>`` input.

fiTQun factorises the geometric term of the predicted charge as
``J = Omega(R) * T(R) * epsilon(cos eta)`` (``fiTQun.cc::J``), where ``Omega``
is the *unprojected* solid angle of the sensor face and ``epsilon`` carries
everything that depends on the angle of incidence — the projection of the face
onto the line of sight and whatever the photocathode does to light arriving
off-normal.

The reference measurement (``Utilities/angular/angularResponsePlotter_v1.C``)
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

from . import rootio


def cos_eta(emission_pos: np.ndarray, sensor_pos: np.ndarray,
            sensor_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(R, cos eta)`` for photons travelling from emission point to sensor.

    ``emission_pos`` is the **photon's emission point**, not the event vertex.
    The reference is explicit about this -- it fills from ``oppos_`` and has the
    ``srcpos_`` version commented out -- because the shell of fixed radius about
    the sensor is only meaningful for the point the light actually left from.

    ``sensor_dir`` is the inward-facing sensor axis, so a photon arriving
    head-on gives ``cos eta = 1``. All arrays are per photon, in **cm**.
    """
    rel = np.asarray(emission_pos, dtype=np.float64) - np.asarray(sensor_pos, dtype=np.float64)
    R = np.linalg.norm(rel, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.einsum("ij,ij->i", rel, np.asarray(sensor_dir, dtype=np.float64))
        c = np.where(R > 0, c / np.where(R > 0, R, 1.0), 1.0)
    return R, np.clip(c, -1.0, 1.0)


def measure(emission_pos: np.ndarray, sensor_pos: np.ndarray, sensor_dir: np.ndarray,
            *, shell_r_cm: float, shell_dr_cm: float = 50.0,
            det_radius_cm: float, det_halfheight_cm: float,
            weights: Optional[np.ndarray] = None,
            n_bins: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Histogram ``cos eta`` for detected direct photons in one spherical shell.

    ``shell_dr_cm`` defaults to the reference's 50 cm (a superseded variant of
    the plotter used 25; the macros that produced the shipped tune used 50).
    ``n_bins`` is 25, the value ``fit_cos.C`` was tuned against -- confirmed by
    the shipped ``angResp`` TF1 carrying ``fNpfits = 25``.

    Returns ``(edges, counts, sumw2)``, unnormalised — merging several scans
    means adding the counts, so normalisation is left to :func:`normalise`.

    The containment cut below keeps a shell only around sensors clear of the
    surface that would clip it, so each radius is measured at its own subset of
    sensors and carries its own statistics. That does not change *what* is
    measured: epsilon is a property of the sensor, so the radii should agree
    within errors. A radius that comes out empty or truncated means the sources
    never reached it -- which is a bug in the sample, not a property of the
    method. ``fit_cos.C`` fits a single hardcoded ``angRespAll_100``, so the
    100 cm shell in particular has to be populated; that is what requires the
    sources to fill the volume out to the sensor faces (see
    :func:`lucid.production.fitqun.isotropic_sample.translate_uniform`).

    Entries are per photon with unit weight, as the reference fills them
    (``totalPe`` is hardcoded to 1 there). Photons from one event that land on
    one sensor share a cos(eta) exactly, so they are one correlated lump and
    sqrt(N) understates the error -- measured at 3.6-8.5x on the 1e8-electron
    sample. That is a property of the method, not of this implementation.
    """
    R, c = cos_eta(emission_pos, sensor_pos, sensor_dir)
    w = np.ones_like(R) if weights is None else np.asarray(weights, dtype=np.float64)

    in_shell = (R >= shell_r_cm - shell_dr_cm) & (R < shell_r_cm + shell_dr_cm)

    # The shell must be fully inside the detector in at least one of the two
    # directions that can clip it -- a barrel sensor is safe if the shell
    # clears the end caps, an end-cap sensor if it clears the barrel wall.
    reach = shell_r_cm + shell_dr_cm
    sensor_pos = np.asarray(sensor_pos, dtype=np.float64)
    clear_z = np.abs(sensor_pos[:, 2]) + reach <= det_halfheight_cm
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
    # TH1F, not TH1D: fit_cos.C reads this with GetObject(..., TH1F*), which
    # type-checks and leaves the pointer null on a mismatch.
    rootio.write(path, {name: rootio.th1(
        name, edges, values, sumw2=errs, dtype=np.float32,
        title="Angular response function", xtitle="cos#eta")})
    return Path(path)
