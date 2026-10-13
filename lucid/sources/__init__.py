from typing import NamedTuple

import jax.numpy as jnp


class SourceBatch(NamedTuple):
    """Time-aware photon samples used by calibration forward models."""

    directions: jnp.ndarray
    positions: jnp.ndarray
    weights: jnp.ndarray
    times: jnp.ndarray


def sample_source(source, n_photons, key):
    """Adapt a legacy source callable to a time-aware source batch."""
    directions, positions, weights = source(n_photons, key)
    times = jnp.zeros((n_photons,), dtype=positions.dtype)
    return SourceBatch(directions, positions, weights, times)


__all__ = [
    "make_cherenkov_surrogate_fn",
    "make_scintillation_surrogate_fn",
    "evaluate_siren_lhs",
    "latin_hypercube_2d",
    "predict_t0",
    "predict_t0_wrapper",
    "IsotropicSource",
    "LaserSource",
    "isotropic_source",
    "laser_source",
    "get_isotropic_rays",
    "generate_laser_photons",
    "setup_calibration_generator",
    "SourceBatch",
    "sample_source",
    "ShotgunSource",
    "shotgun_source",
    "stack_shotgun_sources",
]

from lucid.sources.siren_rays import (
    make_cherenkov_surrogate_fn,
    make_scintillation_surrogate_fn,
    evaluate_siren_lhs,
    latin_hypercube_2d,
    predict_t0,
    predict_t0_wrapper,
)
from lucid.sources.calibration_sources import (
    IsotropicSource, LaserSource,
    isotropic_source, laser_source,
    get_isotropic_rays,
    generate_laser_photons,
    setup_calibration_generator,
)
from lucid.sources.shotgun_source import (
    ShotgunSource,
    shotgun_source,
    stack_shotgun_sources,
)
