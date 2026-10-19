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
    relative_efficiency_s: jnp.ndarray | None = None
    relative_efficiency_p: jnp.ndarray | None = None

    def factor(self, incidence_cosine):
        mu = jnp.clip(jnp.asarray(incidence_cosine), 0.0, 1.0)
        return jnp.interp(
            mu, self.incidence_cosine, self.relative_efficiency)

    def polarized_factor(self, incidence_cosine, direction, polarization, normal):
        """Return the s/p-weighted response for a linearly polarized photon."""
        if self.relative_efficiency_s is None or self.relative_efficiency_p is None:
            return self.factor(incidence_cosine)
        mu = jnp.clip(jnp.asarray(incidence_cosine), 0.0, 1.0)
        s_response = jnp.interp(
            mu, self.incidence_cosine, self.relative_efficiency_s)
        p_response = jnp.interp(
            mu, self.incidence_cosine, self.relative_efficiency_p)

        # s is perpendicular to the incidence plane.  At normal incidence the
        # plane is undefined, but the two table columns coincide; use equal
        # weights there to keep the calculation finite.
        ray_direction = (
            direction if direction.ndim == normal.ndim
            else direction[None, :, :])
        s_axis_raw = jnp.cross(ray_direction, normal)
        s_norm2 = jnp.sum(s_axis_raw * s_axis_raw, axis=-1)
        s_axis = s_axis_raw / jnp.sqrt(s_norm2[..., None] + 1e-12)
        s_weight = jnp.sum(polarization[None, :, :] * s_axis, axis=-1) ** 2
        s_weight = jnp.where(s_norm2 > 1e-10, jnp.clip(s_weight, 0.0, 1.0), 0.5)
        # SK-IV's legacy SGREFPMT branch assigns physical p polarization to
        # RFPMSG ipol=2 (the s table column), and physical s to ipol=1 (the p
        # column).  SK-V later fixed this convention.  Preserve the SK-IV swap
        # here because this response is explicitly the SK4 comparison model.
        return s_weight * p_response + (1.0 - s_weight) * s_response


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
    relative_efficiency_s=None,
    relative_efficiency_p=None,
):
    """Construct a validated response directly from arrays."""
    mu, response = _validated_arrays(
        incidence_cosine, relative_efficiency)
    wavelength_nm = float(wavelength_nm)
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0.0:
        raise ValueError("wavelength_nm must be finite and positive")
    response_s = response_p = None
    if relative_efficiency_s is not None or relative_efficiency_p is not None:
        if relative_efficiency_s is None or relative_efficiency_p is None:
            raise ValueError("both s and p response arrays must be supplied")
        mu_s, response_s = _validated_arrays(
            incidence_cosine, relative_efficiency_s)
        mu_p, response_p = _validated_arrays(
            incidence_cosine, relative_efficiency_p)
        if not np.array_equal(mu, mu_s) or not np.array_equal(mu, mu_p):
            raise ValueError("s/p response grids must match incidence_cosine")
    return TabulatedPmtDetectionResponse(
        incidence_cosine=jnp.asarray(mu),
        relative_efficiency=jnp.asarray(response),
        wavelength_nm=wavelength_nm,
        metadata=dict(metadata or {}),
        relative_efficiency_s=(None if response_s is None else jnp.asarray(response_s)),
        relative_efficiency_p=(None if response_p is None else jnp.asarray(response_p)),
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
        response_s = response_p = None
        if "absorptance_s" in artifact and "absorptance_p" in artifact:
            a_s = np.asarray(artifact["absorptance_s"], dtype=np.float64)
            a_p = np.asarray(artifact["absorptance_p"], dtype=np.float64)
            normal_absorptance = 0.5 * (a_s[-1] + a_p[-1])
            response_s = a_s / normal_absorptance
            response_p = a_p / normal_absorptance
        return make_tabulated_pmt_detection_response(
            artifact["incidence_cosine"],
            artifact["relative_efficiency"],
            wavelength_nm=float(artifact["wavelength_nm"]),
            metadata=metadata,
            relative_efficiency_s=response_s,
            relative_efficiency_p=response_p,
        )


__all__ = [
    "TabulatedPmtDetectionResponse",
    "load_pmt_detection_response",
    "make_tabulated_pmt_detection_response",
]
