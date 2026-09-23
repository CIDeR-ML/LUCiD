"""PhotonSim macros for the Cherenkov-profile scan.

The profile fiTQun wants is the prompt light of one particle ranging out in
water, so the reference tuning macros switch off the processes that would add
light from something else — decay (and the Michel electron that follows), muon
capture, and for pions the hadronic interactions that turn the track into a
shower.

**Those processes are named here, never numbered.** ``/particle/process/
inactivate`` takes an index into a per-particle, per-physics-list process
table, and the indices are not portable: the reference WCSim macros disable
``7`` and ``8`` for ``mu-``, which in WCSim's list are decay and capture but in
PhotonSim's list are ``muMinusCaptureAtRest`` and **``Cerenkov``**. Copying the
numbers across therefore silently deletes the muon's own Cherenkov light —
leaving only delta-ray light, a plausible-looking profile with the wrong angle
and a sixth of the yield. ``/process/inactivate <name>`` has no such trap.

The momentum knob is ``/gun/momentumAmp``, which G4's own particle-gun
messenger provides and which is the same command WCSim's tuning macros use, so
the momentum grids transfer unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .particles import PDG_NAMES

# Processes switched off per particle, by name. Cherenkov and the EM processes
# that shape the track (ionisation, multiple scattering, brems) all stay on --
# they are the profile.
INACTIVE_PROCESSES = {
    11: (),                                         # e-: nothing to suppress
    13: ("Decay", "muMinusCaptureAtRest"),
    211: ("Decay", "hadElastic", "pi+Inelastic"),
}


def profile_macro(*, pdg: int, momentum_mev: float, output_path, n_events: int,
                  seed: Optional[int] = None,
                  inactivate: Optional[Sequence[str]] = None) -> str:
    """The macro for one (particle, momentum) cell of the Cherenkov-profile scan."""
    if pdg not in PDG_NAMES:
        raise ValueError(f"no fiTQun hypothesis for PDG {pdg}")
    processes = INACTIVE_PROCESSES[pdg] if inactivate is None else tuple(inactivate)

    lines = [
        f"# Cherenkov profile: {PDG_NAMES[pdg]} at {momentum_mev:g} MeV/c",
        f"/output/filename {output_path}",
        "/run/initialize",
        "",
        "/photon/storeIndividual true",
        "/photon/streamPhotonsChunked true",
        "",
    ]
    if processes:
        lines.append("# By name: process indices differ between physics lists.")
        lines += [f"/process/inactivate {name}" for name in processes]
        lines.append("")
    lines += [
        f"/gun/particle {PDG_NAMES[pdg]}",
        f"/gun/momentumAmp {momentum_mev:g} MeV",
        "/gun/position 0 0 0 cm",
        "/gun/direction 0 0 1",
        "/gun/randomDirection false",
        "",
    ]
    if seed is not None:
        lines.append(f"/random/setSeeds {seed} {seed + 1}")
        lines.append("")
    lines.append(f"/run/beamOn {n_events}")
    return "\n".join(lines) + "\n"


def write_profile_macro(path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile_macro(**kwargs))
    return path
