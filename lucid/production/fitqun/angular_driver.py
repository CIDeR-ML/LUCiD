"""Drive the angular-response measurement over the 3 MeV electron sample.

The reduction lives in :mod:`lucid.production.fitqun.angular`; this is the part
that produces its inputs. It takes the shared isotropic sample (see
:mod:`lucid.production.fitqun.isotropic_sample`), propagates it through LUCiD,
and hands the reduction the four per-photon quantities it needs: the emission
point, the sensor that was hit, that sensor's position and its inward axis.

Two selections carry over from the reference and are applied here rather than
left to the caller:

* **Direct light only.** ``angularResponsePlotter.cc`` skips every photon with
  ``isct != 0``. LUCiD now reports the same thing per photon, so the cut is
  ``~deviated`` -- no second production, no physics-config gymnastics.
* **The shell.** Only photons whose emission point lies in a thin spherical
  shell about the sensor are kept, so ``Omega(R)`` and ``T(R)`` are common to
  every entry and the cos(eta) spectrum is the angular response alone.

Statistics: the reference reuses the ~1e8-event indirect-light sample, and the
shell cut keeps a small fraction of it, so this is not a cheap measurement. Run
it on the same files the scattering table uses rather than generating a second
sample.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from . import angular


def sensor_axes(sensor_positions: np.ndarray, *, det_radius_cm: float,
                det_halfheight_cm: float, tol_cm: float = 1.0) -> np.ndarray:
    """Inward-facing unit axis for each sensor of a cylinder.

    A sensor on an end cap points along -sign(z); one on the barrel points at
    the axis. Which surface a sensor is on is decided by how close it sits to
    each, so a geometry whose sensors are inset still classifies correctly.
    """
    pos = np.asarray(sensor_positions, dtype=np.float64)
    rho = np.hypot(pos[:, 0], pos[:, 1])
    on_cap = (det_halfheight_cm - np.abs(pos[:, 2])) < (det_radius_cm - rho) - tol_cm

    axes = np.zeros_like(pos)
    axes[on_cap, 2] = -np.sign(pos[on_cap, 2])
    barrel = ~on_cap
    safe_rho = np.where(rho[barrel] > 0, rho[barrel], 1.0)
    axes[barrel, 0] = -pos[barrel, 0] / safe_rho
    axes[barrel, 1] = -pos[barrel, 1] / safe_rho
    return axes


def accumulate(chunks: Iterable[dict], sensor_positions: np.ndarray, *,
               shell_r_cm: float, det_radius_cm: float, det_halfheight_cm: float,
               shell_dr_cm: float = 50.0, n_bins: int = 25,
               direct_only: bool = True):
    """Histogram cos(eta) over a stream of propagated photon chunks.

    Each chunk is a dict with ``emission_pos`` (n, 3) in cm, ``sensor_id`` (n,),
    ``detected`` (n,) and optionally ``deviated`` (n,). Returns
    ``(edges, counts, sumw2)``, ready for :func:`angular.write_angular_response`.
    """
    axes = sensor_axes(sensor_positions, det_radius_cm=det_radius_cm,
                       det_halfheight_cm=det_halfheight_cm)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts = np.zeros(n_bins)
    sumw2 = np.zeros(n_bins)

    for chunk in chunks:
        # np.asarray on a JAX output can be read-only, so build a new array
        # rather than masking in place.
        keep = np.array(chunk["detected"], dtype=bool)
        if direct_only:
            dev = chunk.get("deviated")
            if dev is None:
                raise ValueError(
                    "direct_only needs the per-photon 'deviated' flag; the "
                    "sample predates it or was written without it")
            keep = keep & ~np.asarray(dev, dtype=bool)
        if not keep.any():
            continue

        sid = np.asarray(chunk["sensor_id"])[keep]
        emission = np.asarray(chunk["emission_pos"])[keep]
        _, c, s2 = angular.measure(
            emission, sensor_positions[sid], axes[sid],
            shell_r_cm=shell_r_cm, shell_dr_cm=shell_dr_cm,
            det_radius_cm=det_radius_cm, det_halfheight_cm=det_halfheight_cm,
            n_bins=n_bins)
        counts += c
        sumw2 += s2

    return edges, counts, sumw2


def run(shotgun_files: Iterable[str], sensor_positions: np.ndarray, *,
        output, shell_r_cm: float, det_radius_cm: float,
        det_halfheight_cm: float, **kwargs) -> Path:
    """End to end: shotgun outputs in, ``angRespAll_<r>.root`` out."""
    from lucid.production.photon_shotgun.io import load_shotgun_per_photon

    def _chunks():
        for path in shotgun_files:
            d = load_shotgun_per_photon(str(path))
            src = d.get("source")
            if src is None:
                raise ValueError(f"{path}: no source block; cannot recover "
                                 "the photon emission points")
            yield {
                "emission_pos": np.asarray(src.origins).reshape(-1, 3),
                "sensor_id": np.asarray(d["sensor_id"]).reshape(-1),
                "detected": np.asarray(d["detected"]).reshape(-1),
                "deviated": (None if d.get("deviated") is None
                             else np.asarray(d["deviated"]).reshape(-1)),
            }

    edges, counts, sumw2 = accumulate(
        _chunks(), sensor_positions, shell_r_cm=shell_r_cm,
        det_radius_cm=det_radius_cm, det_halfheight_cm=det_halfheight_cm,
        **kwargs)
    return angular.write_angular_response(output, edges, counts, sumw2,
                                          shell_r_cm=shell_r_cm)
