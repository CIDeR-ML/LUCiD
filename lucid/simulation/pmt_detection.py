"""Differentiable local-incidence PMT detection response.

The geometric PMT model decides whether a ray bundle intersects the active
photocathode.  This module supplies the separate conditional probability that
an intercepted photon produces a photoelectron.  A response is represented as
a relative factor normalized to one at normal incidence; the ordinary QE and
per-PMT QE corrections remain in the sensor-response stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class TabulatedPmtDetectionResponse:
    """One-dimensional response versus local incidence cosine.

    ``relative_efficiency`` may exceed one.  Sensor-response code constrains
    the final Bernoulli probability, rather than clipping this physical ratio.
    Linear interpolation is continuous and differentiable almost everywhere
    with respect to the incidence cosine and table values.
    """

    incidence_cosine: jnp.ndarray
    relative_efficiency: jnp.ndarray
    wavelength_nm: float
    metadata: dict

    def factor(self, incidence_cosine):
        mu = jnp.clip(jnp.asarray(incidence_cosine), 0.0, 1.0)
        return jnp.interp(
            mu, self.incidence_cosine, self.relative_efficiency)


def _validated_arrays(incidence_cosine, relative_efficiency):
    mu = np.asarray(incidence_cosine, dtype=np.float64)
    response = np.asarray(relative_efficiency, dtype=np.float64)
    if mu.ndim != 1 or response.ndim != 1 or mu.shape != response.shape:
        raise ValueError(
            "incidence_cosine and relative_efficiency must be equal-length "
            "one-dimensional arrays")
    if len(mu) < 2:
        raise ValueError("PMT response table needs at least two points")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(response)):
        raise ValueError("PMT response table values must be finite")
    if np.any((mu < 0.0) | (mu > 1.0)):
        raise ValueError("incidence_cosine values must lie in [0, 1]")
    if np.any(response < 0.0):
        raise ValueError("relative_efficiency values must be non-negative")
    order = np.argsort(mu)
    mu = mu[order]
    response = response[order]
    if np.any(np.diff(mu) <= 0.0):
        raise ValueError("incidence_cosine values must be unique")
    return mu, response


def make_tabulated_pmt_detection_response(
    incidence_cosine,
    relative_efficiency,
    *,
    wavelength_nm,
    metadata=None,
):
    """Construct a validated response directly from arrays."""
    mu, response = _validated_arrays(
        incidence_cosine, relative_efficiency)
    wavelength_nm = float(wavelength_nm)
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0.0:
        raise ValueError("wavelength_nm must be finite and positive")
    return TabulatedPmtDetectionResponse(
        incidence_cosine=jnp.asarray(mu),
        relative_efficiency=jnp.asarray(response),
        wavelength_nm=wavelength_nm,
        metadata=dict(metadata or {}),
    )


def load_pmt_detection_response(path):
    """Load a portable ``.npz`` local-incidence response artifact."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PMT detection response does not exist: {path}")
    if path.suffix != ".npz":
        raise ValueError("PMT detection response artifacts must use .npz")
    with np.load(path, allow_pickle=False) as artifact:
        required = {
            "incidence_cosine", "relative_efficiency", "wavelength_nm",
        }
        missing = required.difference(artifact.files)
        if missing:
            raise ValueError(
                f"PMT response artifact {path} is missing {sorted(missing)}")
        metadata = {}
        if "metadata_json" in artifact:
            metadata = json.loads(str(artifact["metadata_json"]))
        return make_tabulated_pmt_detection_response(
            artifact["incidence_cosine"],
            artifact["relative_efficiency"],
            wavelength_nm=float(artifact["wavelength_nm"]),
            metadata=metadata,
        )


__all__ = [
    "TabulatedPmtDetectionResponse",
    "load_pmt_detection_response",
    "make_tabulated_pmt_detection_response",
]
