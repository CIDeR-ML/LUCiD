"""Run GENIE v3 (gevgen + gntpc) to produce a rooTracker file for PhotonSim.

Called from run_job.py before the PhotonSim step when
`config['primary_source'] == 'genie'`. Produces a file
`<output_dir>/gntp_job_<NNNNNN>.gtrac.root` that is fed to PhotonSim via
the `/gun/genieInput` macro command.

The two required environment variables are:

    GENIE_PREFIX     — install root, e.g.
                       /cvmfs/larsoft.opensciencegrid.org/spack-fnal-v1.1.0/
                       spack_env/opt/spack/linux-x86_64_v2/
                       genie-3.06.02-<hash>
    GENIE_XSEC_FILE  — path to cross-section spline XML
                       (e.g. gxspl-FNALsmall.xml for the tune of interest)

The environment must also have gevgen and gntpc resolvable — either by
placing $GENIE_PREFIX/bin first on PATH, or by running this inside a
container where GENIE and its dependencies are already on the library
path (spack-activated).

Water target is expressed as a two-nucleus mix by mass fraction:
16O (0.888) + 1H (0.112).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

WATER_TARGET = "1000080160[0.888],1000010010[0.112]"


class GenieError(RuntimeError):
    """Raised when GENIE invocation or the rootracker conversion fails."""


def _resolve_binary(name: str) -> str:
    """Find `name` on PATH, falling back to $GENIE_PREFIX/bin/<name>."""
    from_path = shutil.which(name)
    if from_path:
        return from_path
    prefix = os.environ.get("GENIE_PREFIX")
    if prefix:
        candidate = Path(prefix) / "bin" / name
        if candidate.is_file():
            return str(candidate)
    raise GenieError(
        f"Cannot locate {name!r}: not on PATH and $GENIE_PREFIX/bin/{name} not found. "
        f"Ensure GENIE v3 is loaded (spack-activated or $GENIE_PREFIX set)."
    )


def _target_spec(config_genie: dict) -> str:
    tgt = config_genie.get("target", "water")
    if tgt == "water":
        return WATER_TARGET
    if isinstance(tgt, (int, str)):
        return str(tgt)
    raise GenieError(f"Unrecognized GENIE target spec: {tgt!r}")


def run_genie(
    *,
    config: dict,
    output_dir: Path,
    job_id: int,
    n_events: int,
    seed: Optional[int] = None,
) -> Path:
    """Run gevgen + gntpc to produce a rootracker file. Returns the path.

    Seeding: if `seed` is None, derive a per-job seed from job_id so reruns
    are reproducible; otherwise use the caller-provided value directly.
    """
    if "genie" not in config:
        raise GenieError("config missing 'genie' block (primary_source=genie requires it)")
    g = config["genie"]
    probe = int(g["probe_pdg"])
    emin = float(g["energy_min_GeV"])
    emax = float(g["energy_max_GeV"])
    tune = str(g.get("tune", "G18_02a_00_000"))
    evg_list = str(g.get("event_generator_list", "CC+NC"))
    target = _target_spec(g)

    xsec = os.environ.get("GENIE_XSEC_FILE")
    if not xsec or not Path(xsec).is_file():
        raise GenieError(
            f"GENIE_XSEC_FILE unset or missing: {xsec!r}. "
            f"Point to a cross-section spline XML matching --tune={tune}."
        )

    gevgen = _resolve_binary("gevgen")
    gntpc  = _resolve_binary("gntpc")

    if seed is None:
        # Deterministic per-job seed. job_id is 1-based; avoid seed==0.
        seed = 100000 + job_id

    job_padded = f"{job_id:06d}"
    # gevgen v3's -o is the literal output filename (no suffix appended).
    ghep_file  = output_dir / f"gntp_job_{job_padded}.ghep.root"
    gtrac_file = output_dir / f"gntp_job_{job_padded}.gtrac.root"

    # gevgen runs in output_dir because it leaves auxiliary files
    # (input-flux.root, .status) and chatty output there; keeping it
    # job-local avoids cross-job contention.
    gevgen_cmd = [
        gevgen,
        "-n", str(n_events),
        "-e", f"{emin},{emax}",
        "-f", "1",  # flat flux across the energy range
        "-p", str(probe),
        "-t", target,
        "--cross-sections", xsec,
        "--tune", tune,
        "--seed", str(seed),
        "-o", str(ghep_file),
    ]
    # Allow opt-in to a specific event-generator-list; by default use GENIE's
    # own default (Tune's bundled generator list). "CC+NC" is not a valid
    # generator-list name in v3 — the Tune already covers both.
    if evg_list and evg_list.lower() not in ("", "default", "cc+nc"):
        gevgen_cmd += ["--event-generator-list", evg_list]

    print("=== GENIE: running gevgen ===", flush=True)
    print("    " + " ".join(gevgen_cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(gevgen_cmd, cwd=str(output_dir))
    if rc != 0:
        raise GenieError(f"gevgen failed (exit {rc})")
    print(f"    gevgen ok ({time.time() - t0:.1f}s)")

    if not ghep_file.is_file():
        raise GenieError(f"gevgen did not produce {ghep_file}")

    gntpc_cmd = [gntpc, "-i", str(ghep_file), "-f", "rootracker", "-o", str(gtrac_file)]
    print("=== GENIE: running gntpc → rootracker ===", flush=True)
    print("    " + " ".join(gntpc_cmd), flush=True)
    t1 = time.time()
    rc = subprocess.call(gntpc_cmd, cwd=str(output_dir))
    if rc != 0:
        raise GenieError(f"gntpc failed (exit {rc})")
    if not gtrac_file.is_file():
        raise GenieError(f"gntpc did not produce {gtrac_file}")
    print(f"    gntpc ok ({time.time() - t1:.1f}s)")

    # Remove the intermediate ghep file unless the user wants to keep it.
    if not config.get("keep_genie_ghep", False):
        try:
            ghep_file.unlink()
        except OSError:
            pass

    return gtrac_file
