"""The 3 MeV electron sample that feeds the angular response and the scattering table.

Both tables come from the *same* MC production in the reference chain. Ryo's
description of it:

    We generate 3 MeV electrons at uniformly distributed positions in the
    detector and with uniformly distributed directions. From the truth
    information in the simulation, we select only the indirect light and create
    a six-dimensional table. [...] For the angular response we reuse the
    simulation samples generated for the indirect light table.

So this module produces one photon-level sample and hands it to both
reductions. The pieces that make that possible:

* PhotonSim fires the electrons and returns every Cherenkov photon's emission
  point and direction -- ``oppos``/``opdir`` in the reference's vocabulary. A
  3 MeV electron ranges about 1.5 cm in water, so those points sit essentially
  at the vertex, but they are taken from the simulation rather than assumed.
* LUCiD propagates that photon list through the real detector via
  :func:`lucid.sources.shotgun_source.shotgun_source`, which accepts per-photon
  origins and directions.
* The propagation now reports, per detected photon, whether it scattered or
  reflected on the way (``per_photon/deviated``). That is the reference's
  ``isct`` flag, and it is what splits direct from indirect light in a single
  pass -- the reference never runs two productions.

Statistics: the reference quotes ~1e8 events, which is the dominant cost of
the whole tuning exercise and scales with detector volume. :func:`plan` sizes a
run from a target photon count so a reduced first pass can validate the chain
before committing to the full one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# The reference's source particle. Its range in water (~1.5 cm) is what makes
# the emission effectively point-like, which is what the shell selection in the
# angular response and the source binning in the scattering table both assume.
SOURCE_PDG = 11
SOURCE_KINETIC_MEV = 3.0


@dataclass
class SamplePlan:
    """How many events, split over how many jobs, for a target photon count."""
    n_events: int
    n_jobs: int
    events_per_job: int
    label: str

    def __str__(self) -> str:
        return (f"{self.label}: {self.n_events:,} electrons over {self.n_jobs} "
                f"job(s) x {self.events_per_job:,}")


def plan(n_events: int, events_per_job: int = 200_000,
         label: str = "isotropic_e3MeV") -> SamplePlan:
    """Split a requested event count into jobs.

    The reference runs ~1e8 events; a reduced pass of 1e5-1e6 is enough to
    exercise the whole chain and see whether the tables fill sensibly, which is
    worth doing before spending the full production.
    """
    if n_events <= 0:
        raise ValueError("n_events must be positive")
    n_jobs = max(1, -(-n_events // events_per_job))
    return SamplePlan(n_events=n_events, n_jobs=n_jobs,
                      events_per_job=min(n_events, events_per_job), label=label)


def photonsim_macro(*, output_path, n_events: int, seed: int) -> str:
    """PhotonSim macro for one job of the sample.

    The gun sits at the origin with an isotropic direction. The *position* half
    of "uniformly distributed positions" is not done here: PhotonSim has no
    volume-sampling command, and LUCiD already places each event uniformly in
    the detector through its ``apply_translation`` option -- the same mechanism
    WAND production uses. :func:`translate_uniform` applies it to the photon
    list when this sample is driven directly.

    Unlike the Cherenkov-profile scan this keeps the full physics: there is no
    single track to isolate, and the light wanted here is whatever a 3 MeV
    electron actually produces.
    """
    return "\n".join([
        f"# Isotropic {SOURCE_KINETIC_MEV:g} MeV electron sample "
        "(angular response + scattering tables)",
        f"/output/filename {output_path}",
        "/run/initialize",
        "",
        "/photon/storeIndividual true",
        "/photon/streamPhotonsChunked true",
        "",
        "/gun/particle e-",
        f"/gun/energy {SOURCE_KINETIC_MEV:g} MeV",
        "/gun/position 0 0 0 cm",
        "/gun/randomDirection true",
        "",
        f"/random/setSeeds {seed} {seed + 1}",
        "",
        f"/run/beamOn {n_events}",
    ]) + "\n"


def translate_uniform(origins: np.ndarray, rng, *, det_radius_cm: float,
                      det_halfheight_cm: float,
                      fiducial_fraction: float = 1.0) -> np.ndarray:
    """Shift a photon list to a uniformly sampled vertex in the cylinder.

    Uniform in volume means uniform in r^2, not in r -- sampling r linearly
    would pile events toward the axis and bias every table built from them.
    """
    r = det_radius_cm * fiducial_fraction
    hz = det_halfheight_cm * fiducial_fraction
    rho = r * np.sqrt(rng.random())
    phi = 2.0 * np.pi * rng.random()
    shift = np.array([rho * np.cos(phi), rho * np.sin(phi),
                      hz * (2.0 * rng.random() - 1.0)], dtype=np.float32)
    return origins + shift


def load_photons(photonsim_path, step_size: str = "200 MB"):
    """Stream ``(origins, directions)`` in cm from a PhotonSim file.

    These are the per-photon emission points and directions the shotgun needs,
    and the emission point is what the angular response measures its shell
    radius from -- the reference is explicit that this is ``oppos``, the photon
    origin, not the event vertex.
    """
    import uproot

    pos_branches = ["PhotonPosX", "PhotonPosY", "PhotonPosZ"]
    dir_branches = ["PhotonDirX", "PhotonDirY", "PhotonDirZ"]
    with uproot.open(photonsim_path) as f:
        raw = f["OpticalPhotonsRaw"]
        for chunk in raw.iterate(pos_branches + dir_branches,
                                 step_size=step_size, library="np"):
            origins = np.stack(
                [np.concatenate(chunk[b]) for b in pos_branches], axis=1) * 0.1
            directions = np.stack(
                [np.concatenate(chunk[b]) for b in dir_branches], axis=1)
            yield origins.astype(np.float32), directions.astype(np.float32)
