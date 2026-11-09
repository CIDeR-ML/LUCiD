"""Photon sources and event input.

The SIREN track emitters (``siren_rays``, exported here with the calibration and shotgun
sources), the scintillation photons (``scintillation_photons``), the cascade source
(``cascade``), the PhotonSim ROOT reader (``root_reader``), the data-event helpers
(``placement``, ``data_event``, ``interaction``) and the per-event key derivation
(``seed_utils``).
"""
__all__ = [
    "make_cherenkov_surrogate_fn",
    "make_scintillation_surrogate_fn",
    "latin_hypercube_2d",
    "predict_t0",
    "IsotropicSource",
    "IsotropicSourceRandom",
    "LaserSource",
    "isotropic_source",
    "isotropic_source_random",
    "laser_source",
    "get_isotropic_rays",
    "generate_laser_photons",
    "ShotgunSource",
    "shotgun_source",
    "stack_shotgun_sources",
]

from lucid.sources.siren_rays import (
    make_cherenkov_surrogate_fn,
    make_scintillation_surrogate_fn,
    latin_hypercube_2d,
    predict_t0,
)
from lucid.sources.calibration_sources import (
    IsotropicSource, IsotropicSourceRandom, LaserSource,
    isotropic_source, isotropic_source_random, laser_source,
    get_isotropic_rays,
    generate_laser_photons,
)
from lucid.sources.shotgun_source import (
    ShotgunSource,
    shotgun_source,
    stack_shotgun_sources,
)
