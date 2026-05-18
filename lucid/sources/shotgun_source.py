"""Shotgun source: user-specified per-photon origin/direction/wavelength.

ShotgunSource conforms to the calibration-source callable contract
``source(n_photons, key) -> (directions, origins, intensities)`` with an optional
``wavelength`` attribute (scalar nm, per-photon array, or None for Cherenkov
sampling). This lets it plug into ``setup_event_simulator`` with
``is_calibration=True`` using the existing propagation path.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Union

import jax
import jax.numpy as jnp

from lucid.utils import normalize
from lucid.wavelength.spectrum import sample_cherenkov_wavelengths


class ShotgunSource(NamedTuple):
    """Hand-specified photon batch — callable JAX pytree.

    Fields
    ------
    origins : jnp.ndarray
        Per-photon origins in meters, shape ``(n_photons, 3)``.
    directions : jnp.ndarray
        Per-photon directions (normalized), shape ``(n_photons, 3)``.
    intensities : jnp.ndarray
        Per-photon weights, shape ``(n_photons,)``.
    wavelength : jnp.ndarray or None
        Either ``None`` (sample Cherenkov at call time), a scalar (broadcast to
        all photons), or a ``(n_photons,)`` per-photon array.

    Contract
    --------
    ``source(n_photons, key)`` returns ``(directions, origins, intensities)``,
    matching ``IsotropicSource`` / ``LaserSource``. Wavelengths are read from
    ``source.wavelength`` by ``setup_event_simulator`` for optical lookups.
    """
    origins: jax.Array
    directions: jax.Array
    intensities: jax.Array
    wavelength: Optional[jax.Array] = None

    def __call__(self, n_photons, key, n_water=1.33):
        return self.directions, self.origins, self.intensities


def shotgun_source(
    origins,
    directions,
    *,
    n_photons: Optional[int] = None,
    wavelength: Union[float, jax.Array, str, None] = None,
    intensity: Union[float, jax.Array] = 1.0,
):
    """Build a ShotgunSource with flexible broadcasting.

    Parameters
    ----------
    origins : array-like
        Shape ``(3,)`` (broadcast) or ``(n_photons, 3)``.
    directions : array-like
        Shape ``(3,)`` (broadcast) or ``(n_photons, 3)``. Normalized internally.
    n_photons : int, optional
        Required when both origins and directions are shape ``(3,)``.
    wavelength : float, array, "cherenkov", or None
        * ``float`` — monochromatic (stored as scalar; simulator broadcasts).
        * array shape ``(n_photons,)`` — per-photon wavelengths in nm.
        * ``"cherenkov"`` or ``None`` — sample from Cherenkov spectrum at
          simulator call.
    intensity : float or array
        Scalar broadcast to ``(n_photons,)``, or per-photon array.
    """
    origins = jnp.asarray(origins, dtype=jnp.float32)
    directions = jnp.asarray(directions, dtype=jnp.float32)

    if origins.ndim == 2:
        n_from_origins = origins.shape[0]
    elif origins.ndim == 1 and origins.shape[0] == 3:
        n_from_origins = None
    else:
        raise ValueError(f"origins must be shape (3,) or (n, 3); got {origins.shape}")

    if directions.ndim == 2:
        n_from_dirs = directions.shape[0]
    elif directions.ndim == 1 and directions.shape[0] == 3:
        n_from_dirs = None
    else:
        raise ValueError(f"directions must be shape (3,) or (n, 3); got {directions.shape}")

    candidates = [x for x in (n_from_origins, n_from_dirs, n_photons) if x is not None]
    if not candidates:
        raise ValueError(
            "Cannot infer n_photons: pass per-photon origins/directions or set n_photons."
        )
    if len(set(candidates)) > 1:
        raise ValueError(
            f"Inconsistent n_photons across origins/directions/n_photons: {candidates}"
        )
    n = candidates[0]

    if origins.ndim == 1:
        origins = jnp.broadcast_to(origins, (n, 3))
    if directions.ndim == 1:
        directions = jnp.broadcast_to(directions, (n, 3))
    directions = normalize(directions)

    intensity = jnp.asarray(intensity, dtype=jnp.float32)
    if intensity.ndim == 0:
        intensities = jnp.full((n,), intensity)
    else:
        if intensity.shape != (n,):
            raise ValueError(f"intensity must be scalar or shape ({n},); got {intensity.shape}")
        intensities = intensity

    if wavelength is None or (isinstance(wavelength, str) and wavelength.lower() == "cherenkov"):
        wl = None
    else:
        wl_arr = jnp.asarray(wavelength, dtype=jnp.float32)
        if wl_arr.ndim == 0:
            wl = wl_arr
        elif wl_arr.shape == (n,):
            wl = wl_arr
        else:
            raise ValueError(
                f"wavelength must be scalar, shape ({n},), 'cherenkov', or None; got {wl_arr.shape}"
            )

    return ShotgunSource(
        origins=origins,
        directions=directions,
        intensities=intensities,
        wavelength=wl,
    )


def stack_shotgun_sources(sources):
    """Stack a list of ShotgunSource into a batched one with leading case axis.

    All sources must share ``n_photons`` and wavelength mode (all scalar, all
    array-shape, or all None).
    """
    if not sources:
        raise ValueError("sources cannot be empty")
    n0 = sources[0].origins.shape[0]
    for s in sources:
        if s.origins.shape[0] != n0:
            raise ValueError("all sources must share n_photons")

    wl_none = [s.wavelength is None for s in sources]
    if any(wl_none) and not all(wl_none):
        raise ValueError("all sources must have wavelengths set or none set "
                         "(mixed Cherenkov + fixed not supported)")

    origins = jnp.stack([s.origins for s in sources], axis=0)
    directions = jnp.stack([s.directions for s in sources], axis=0)
    intensities = jnp.stack([s.intensities for s in sources], axis=0)
    if all(wl_none):
        wl = None
    else:
        wl_shapes = {tuple(s.wavelength.shape) for s in sources}
        if len(wl_shapes) > 1:
            raise ValueError(f"inconsistent wavelength shapes across sources: {wl_shapes}")
        wl = jnp.stack([jnp.asarray(s.wavelength) for s in sources], axis=0)
    return ShotgunSource(origins=origins, directions=directions,
                         intensities=intensities, wavelength=wl)
