"""Generate a Geant4 macro for PhotonSim from a dataprod JSON config.

The macro drives a single PhotonSim invocation: set the output ROOT
filename, initialize, configure photon/edep storage, optionally disable
decays, build the primary list with uniform-energy-range or monoenergetic
primaries, request random directions, and beamOn.

Shape of config dict (matches JSON in lucid/production/configs/):

    {
        "config_number": int,
        "name": str,                    # human label, e.g. "single mu-"
        "material": "water",
        "energy_distribution": "uniform" | "monoenergetic",
        "store_individual_photons": bool,
        "disable_decays": bool,
        "particles": [
            {"type": "mu-", "energy_min_MeV": ..., "energy_max_MeV": ...},
            ...
        ],
        ...
    }

For monoenergetic with an energy scan, the caller passes
`override_energy_MeV` to fix the per-run energy; we emit
`/gun/addPrimary <type> <E> MeV` for each primary.
"""
from __future__ import annotations

from typing import Optional, Tuple


def generate_macro(
    config: dict,
    output_root_file: str,
    n_events: int,
    override_energy_MeV: Optional[float] = None,
    genie_rootracker: Optional[str] = None,
    photonsim_seeds: Optional[Tuple[int, int]] = None,
) -> str:
    """Return the Geant4 macro text for one PhotonSim job.

    Parameters
    ----------
    config : dict
        Parsed dataprod JSON config.
    output_root_file : str
        Value for `/output/filename` (typically `output_job_NNNNNN.root`).
    n_events : int
        Argument to `/run/beamOn`.
    override_energy_MeV : float, optional
        If set, force monoenergetic primaries at this energy regardless of
        config's `energy_distribution`. Used by the S3DF wrapper's energy-scan
        fan-out. If unset, behavior follows `config['energy_distribution']`.
    genie_rootracker : str, optional
        Path to a GENIE rooTracker ROOT file (from `gntpc -f rootracker`).
        When set, emit `/gun/genieInput` / `/gun/genieIsotropic` instead of
        `/gun/addPrimary*` — PhotonSim injects final-state particles from the
        rootracker entries. Requires config['primary_source'] == 'genie'.
    photonsim_seeds : (int, int), optional
        Two positive integers for `/random/setSeeds`. When set, emit the
        command immediately after `/run/initialize` so Geant4's CLHEP
        engine is explicitly seeded. Without this, two identically-invoked
        PhotonSim jobs can produce identical output because CLHEP falls
        back to a fixed internal default.
    """
    if genie_rootracker is not None:
        return _generate_genie_macro(
            config=config,
            output_root_file=output_root_file,
            n_events=n_events,
            rootracker_path=genie_rootracker,
            photonsim_seeds=photonsim_seeds,
        )
    config_name = config.get("name", "")
    config_number = config.get("config_number", -1)
    material = config.get("material", "")
    particles = config["particles"]
    n_particles = len(particles)
    store_individual = bool(config.get("store_individual_photons", False))
    disable_decays = bool(config.get("disable_decays", False))
    energy_dist = (
        "monoenergetic" if override_energy_MeV is not None
        else config.get("energy_distribution", "uniform")
    )

    lines: list[str] = []

    # Header
    header_kind = "Monoenergetic" if energy_dist == "monoenergetic" else "Unified Multi-Primary Workflow"
    lines.append(f"# PhotonSim macro ({header_kind})")
    lines.append(f"# Configuration: {config_name}")
    lines.append(f"# Config Number: {config_number:06d}")
    lines.append(f"# Material: {material}")
    lines.append(f"# Particles: {n_particles}")
    if override_energy_MeV is not None:
        lines.append(f"# Energy: {override_energy_MeV:g} MeV")
    lines.append("")
    lines.append("# Set output filename before initialization")
    lines.append(f"/output/filename {output_root_file}")
    lines.append("")
    lines.append("/run/initialize")
    lines.append("")

    if photonsim_seeds is not None:
        s1, s2 = photonsim_seeds
        lines.append(f"/random/setSeeds {int(s1)} {int(s2)}")
        lines.append("")

    # Photon / edep storage
    if store_individual:
        lines.append("# ENABLE individual photon/edep storage for event-by-event analysis")
        lines.append("/photon/storeIndividual true")
        lines.append("/edep/storeIndividual false")
    else:
        lines.append("# DISABLE individual photon/edep storage to save space")
        lines.append("/photon/storeIndividual false")
        lines.append("/edep/storeIndividual false")

    # Decays
    if disable_decays:
        lines.append("")
        lines.append("# Disable decay processes (for lookup table generation)")
        lines.append("/particle/select mu-")
        lines.append("/particle/process/inactivate 1")
        lines.append("/particle/process/inactivate 7")
        lines.append("/particle/select mu+")
        lines.append("/particle/process/inactivate 1")
        lines.append("/particle/select pi+")
        lines.append("/particle/process/inactivate 1")
        lines.append("/particle/select pi-")
        lines.append("/particle/process/inactivate 1")
    else:
        lines.append("")
        lines.append("# Decay processes ENABLED (for data production)")

    # Primaries
    lines.append("")
    lines.append("# Clear any existing primaries and set up new ones")
    lines.append("/gun/clearPrimaries")

    for p in particles:
        ptype = p["type"]
        if energy_dist == "uniform":
            emin = p.get("energy_min_MeV", config.get("energy_min_MeV"))
            emax = p.get("energy_max_MeV", config.get("energy_max_MeV"))
            if emin is None or emax is None:
                raise ValueError(
                    f"Particle {ptype!r} missing energy_min_MeV / energy_max_MeV "
                    f"(and config-level fallbacks also absent)"
                )
            lines.append(f"/gun/addPrimaryWithEnergyRange {ptype} {emin:g} {emax:g} MeV")
        else:
            if override_energy_MeV is not None:
                energy = override_energy_MeV
            else:
                energy = p.get("energy_MeV", config.get("energy_MeV"))
                if energy is None:
                    raise ValueError(
                        f"Monoenergetic particle {ptype!r} missing energy_MeV "
                        f"(and config-level fallback also absent)"
                    )
            lines.append(f"/gun/addPrimary {ptype} {energy:g} MeV")

    lines.append("")
    lines.append("# Use random directions for all primaries")
    lines.append("/gun/randomDirection true")
    lines.append("")
    lines.append(f"# Run {n_events} events")
    lines.append(f"/run/beamOn {n_events}")

    # Trailing newline matches `cat >> HEREDOC` output: one `\n` after the last line.
    return "\n".join(lines) + "\n"


def _generate_genie_macro(
    *,
    config: dict,
    output_root_file: str,
    n_events: int,
    rootracker_path: str,
    photonsim_seeds: Optional[Tuple[int, int]] = None,
) -> str:
    """Emit a Geant4 macro that reads primaries from a GENIE rootracker file."""
    config_name = config.get("name", "")
    config_number = config.get("config_number", -1)
    material = config.get("material", "")
    g = config.get("genie", {})
    probe = g.get("probe_pdg", "?")
    emin = g.get("energy_min_GeV", "?")
    emax = g.get("energy_max_GeV", "?")
    isotropic = str(g.get("direction", "isotropic")).lower() == "isotropic"
    store_individual = bool(config.get("store_individual_photons", False))
    disable_decays = bool(config.get("disable_decays", False))

    lines: list[str] = []
    lines.append("# PhotonSim macro (GENIE primary source)")
    lines.append(f"# Configuration: {config_name}")
    lines.append(f"# Config Number: {config_number:06d}")
    lines.append(f"# Material: {material}")
    lines.append(f"# Probe PDG: {probe}, E range [{emin}, {emax}] GeV, "
                 f"direction: {'isotropic' if isotropic else 'GENIE beam axis'}")
    lines.append("")
    lines.append("# Set output filename before initialization")
    lines.append(f"/output/filename {output_root_file}")
    lines.append("")
    lines.append("/run/initialize")
    lines.append("")

    if photonsim_seeds is not None:
        s1, s2 = photonsim_seeds
        lines.append(f"/random/setSeeds {int(s1)} {int(s2)}")
        lines.append("")

    if store_individual:
        lines.append("/photon/storeIndividual true")
        lines.append("/edep/storeIndividual false")
    else:
        lines.append("/photon/storeIndividual false")
        lines.append("/edep/storeIndividual false")

    if disable_decays:
        lines.append("")
        lines.append("# Decays disabled (unusual for GENIE events; included for parity)")
        lines.append("/particle/select mu-")
        lines.append("/particle/process/inactivate 1")
        lines.append("/particle/process/inactivate 7")
        lines.append("/particle/select mu+")
        lines.append("/particle/process/inactivate 1")
        lines.append("/particle/select pi+")
        lines.append("/particle/process/inactivate 1")
        lines.append("/particle/select pi-")
        lines.append("/particle/process/inactivate 1")
    else:
        lines.append("")
        lines.append("# Decays ENABLED (standard for GENIE final-state propagation)")

    lines.append("")
    lines.append("# Load final-state particles from GENIE rootracker file.")
    lines.append("/gun/clearPrimaries")
    lines.append(f"/gun/genieInput {rootracker_path}")
    lines.append(f"/gun/genieIsotropic {'true' if isotropic else 'false'}")
    lines.append("")
    lines.append(f"# Run {n_events} events")
    lines.append(f"/run/beamOn {n_events}")

    return "\n".join(lines) + "\n"
