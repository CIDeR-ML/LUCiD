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
  reflected on the way (``per_photon/indirect``). That is the reference's
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

    Multiple scattering is inactivated, as ``scattab_nuPRISM_mPMT.mac`` does.
    That is a choice of the reference tune, not an optimisation: it keeps the
    3 MeV electron's path straight so the emission stays point-like, which is
    what the source binning of both tables assumes. Decays and the rest of the
    physics stay on -- the light wanted here is whatever the electron produces.
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
        # The reference inactivates multiple scattering for this sample.
        "/process/inactivate msc",
        "",
        f"/random/setSeeds {seed} {seed + 1}",
        "",
        f"/run/beamOn {n_events}",
    ]) + "\n"


def translate_uniform(origins: np.ndarray, rng, *, det_radius_m: float,
                      det_halfheight_m: float, pmt_radius_m: float,
                      event_id: Optional[np.ndarray] = None) -> np.ndarray:
    """Shift photons to uniformly sampled vertices in the cylinder.

    ``event_id`` gives each photon's event, and every event gets its **own**
    vertex. Passing it is effectively mandatory for the tables built on this
    sample: the spherical-shell construction behind the angular response is
    only valid when sources are uniform in the volume around each PMT, because
    that is what makes the source distribution flat in cos(eta). Shifting a
    whole chunk by one draw collapses thousands of events onto a single vertex
    and the measured spectrum then reflects those few positions, not the
    photosensor. Omitting it keeps the old single-shift behaviour, which is
    only meaningful when the array really is one event.

    The volume is the one both tables are binned on, so there is no fiducial
    margin to choose: ``scatTableLooper.C`` sets ``rmax = cylRadius - tuberadius``
    and ``zmax = tubezpos - tuberadius``, filling the cylinder right up to the
    sensor faces. Holding sources further in is not the conservative choice it
    looks like -- it empties the small angular-response shells, and ``fit_cos.C``
    fits the 100 cm one.

    Uniform in volume means uniform in r^2, not in r -- sampling r linearly
    would pile events toward the axis. Dimensions are in meters, matching
    :func:`load_photons` and the propagation it feeds.
    """
    r = det_radius_m - pmt_radius_m
    hz = det_halfheight_m - pmt_radius_m
    if not (r > 0.0 and hz > 0.0):
        raise ValueError(
            f"a {pmt_radius_m} m sensor leaves no source volume in a "
            f"{det_radius_m} x {det_halfheight_m} m half-cylinder")

    def _draw(n):
        rho = r * np.sqrt(rng.random(n))
        phi = 2.0 * np.pi * rng.random(n)
        return np.stack([rho * np.cos(phi), rho * np.sin(phi),
                         hz * (2.0 * rng.random(n) - 1.0)], axis=1).astype(np.float32)

    if event_id is None:
        return origins + _draw(1)[0]

    event_id = np.asarray(event_id)
    uniq, inverse = np.unique(event_id, return_inverse=True)
    return origins + _draw(uniq.size)[inverse]


def load_photons(photonsim_path, step_size: str = "200 MB"):
    """Stream ``(origins, directions, event_id)`` in METERS from a PhotonSim file.

    Meters, not fiTQun's cm: these photons are fed to
    :func:`lucid.sources.shotgun_source.shotgun_source`, and every LUCiD
    propagation input is in meters. The reductions convert to cm themselves,
    at the point where fiTQun's binning is applied.

    The event id travels with the photons because the vertex has to be drawn
    per event (see :func:`translate_uniform`); flattening it away is what
    silently reduces a large sample to a handful of source positions.

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
        for chunk in raw.iterate(pos_branches + dir_branches + ["EventID"],
                                 step_size=step_size, library="np"):
            per_entry = [len(v) for v in chunk["PhotonPosX"]]
            event_id = np.repeat(np.asarray(chunk["EventID"]), per_entry)
            origins = np.stack(
                [np.concatenate(chunk[b]) for b in pos_branches], axis=1) * 0.001
            directions = np.stack(
                [np.concatenate(chunk[b]) for b in dir_branches], axis=1)
            yield (origins.astype(np.float32), directions.astype(np.float32),
                   event_id)
