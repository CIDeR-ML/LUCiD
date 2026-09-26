#!/usr/bin/env python3
"""Cluster fan-out for the fiTQun Cherenkov-profile scan.

One job per (particle, momentum) cell. Each job generates its PhotonSim macro,
runs it, reduces the photon list to a profile cell, and deletes the ROOT file
straight away — a high-momentum cell is hundreds of MB of raw photons but a
few tens of kB once reduced, so nothing but the reduction is kept.

Merge and build the table once the jobs finish:

    python -m lucid.production.fitqun cprofile build <out>/13/*/cell.npz \\
        --pdg 13 -o CProf_13_WCSim.root

The other four tuning stages do not fan out like this: the charge PDF runs in
one process (``python -m lucid.production.fitqun chargepdf scan``), and the
angular, time and scattering tables are reductions over LUCiD simulation
output produced by the normal production path.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).parent
LUCID_ROOT = SCRIPT_DIR.resolve().parents[3]
if str(LUCID_ROOT) not in sys.path:
    sys.path.insert(0, str(LUCID_ROOT))

from lucid.production.cluster_common import htcondor  # noqa: E402, F401  registers adapter
from lucid.production.cluster_common.cluster import get_adapter  # noqa: E402
from lucid.production.cluster_common.user_paths import load_user_paths  # noqa: E402
from lucid.production.fitqun import macros  # noqa: E402

JOBS_DIR = SCRIPT_DIR.parent
USER_PATHS_DEFAULT = JOBS_DIR / "user_paths.sh"


def cell_command(*, cell_dir: Path, pdg: int, momentum: float) -> str:
    """The one-liner a cell job runs inside the container.

    The macro is written host-side at submit time, so this stays free of
    nested quoting: HTCondor's submit ``arguments`` is itself a quoted string,
    and an embedded ``python -c "..."`` would need its double quotes doubled.
    The ``&&`` chain also means the ROOT file is deleted only once the reduced
    cell exists, so a failure leaves the raw output to debug with.
    """
    macro = cell_dir / "photonsim.mac"
    root = cell_dir / "photonsim.root"
    cell = cell_dir / "cell.npz"
    # PHOTONSIM_BIN is inlined by the HTCondor adapter when a dev checkout is
    # configured; fall back to the image's binary otherwise.
    run = f"${{PHOTONSIM_BIN:-/opt/PhotonSim/build/PhotonSim}} {macro}"
    reduce_ = (f"python -m lucid.production.fitqun cprofile accumulate {root} "
               f"--pdg {pdg} --momentum {momentum:g} -o {cell}")
    return f"{run} && {reduce_} && rm -f {root}"


def events_for(momentum_mev: float, schedule: dict) -> int:
    """Events per cell, from a ``{max_momentum: n_events}`` ladder.

    Photon yield grows roughly linearly with track length, so a fixed event
    count would over-sample the high-momentum cells (which are also the slow,
    disk-hungry ones) and under-sample the cheap low-momentum ones.
    """
    for cut, n in sorted((float(k), v) for k, v in schedule.items()):
        if momentum_mev <= cut:
            return int(n)
    return int(schedule[max(schedule, key=lambda k: float(k))])


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", type=Path, required=True,
                   help="scan config JSON (see configs/)")
    p.add_argument("-s", "--submit", action="store_true",
                   help="submit (default: write the submit files only)")
    p.add_argument("-t", "--test", action="store_true",
                   help="only the first cell")
    p.add_argument("-o", "--output-base", type=Path, default=None,
                   help="output root (default: <OUTPUT_BASE_PATH>/fitqun_cprofile)")
    p.add_argument("-P", "--partition", type=str, default="",
                   help="SLURM partition / HTCondor JobFlavour override")
    p.add_argument("--user-paths", type=Path, default=USER_PATHS_DEFAULT)
    p.add_argument("--no-skip-existing", action="store_true",
                   help="re-run cells that already have cell.npz")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = json.loads(args.config.read_text())
    env = load_user_paths(args.user_paths)
    adapter = get_adapter(env)

    out_base = args.output_base or Path(env["OUTPUT_BASE_PATH"]) / "fitqun_cprofile"
    schedule = cfg.get("events_schedule", {"1e9": cfg.get("n_events_per_job", 200)})
    momenta_cfg = cfg.get("momentum_list_MeV")

    submitted = 0
    for pdg in cfg["pdgs"]:
        if momenta_cfg is not None:
            momenta = [float(m) for m in momenta_cfg]
        else:
            # Deferred: reading the reference grid needs numpy, which the submit
            # host may not have. A config naming its own grid never touches it.
            # The grid is per particle -- each starts at its own Cherenkov
            # threshold, so this has to be resolved inside the PDG loop.
            from lucid.production.fitqun import binning
            momenta = [float(m) for m in binning.cprofile_momenta(int(pdg))]
        for momentum in momenta:
            cell_dir = out_base / str(pdg) / f"{momentum:g}MeV"
            if (cell_dir / "cell.npz").exists() and not args.no_skip_existing:
                continue
            cell_dir.mkdir(parents=True, exist_ok=True)

            macros.write_profile_macro(
                cell_dir / "photonsim.mac", pdg=int(pdg), momentum_mev=float(momentum),
                output_path=cell_dir / "photonsim.root",
                n_events=events_for(momentum, schedule),
                seed=int(cfg.get("seed_base", 0)) + int(momentum))
            command = cell_command(cell_dir=cell_dir, pdg=int(pdg),
                                   momentum=float(momentum))
            body = adapter.render_command_job(
                command=command, cell_dir=cell_dir,
                job_name=f"{cfg['name']}_{pdg}_{momentum:g}",
                log_stem="cprofile", partition=args.partition,
                # PhotonSim writes the raw photon list before it is reduced;
                # high-momentum cells need real scratch space.
                request_disk_mb=int(cfg.get("request_disk_mb", 16384)))
            sub = cell_dir / f"cprofile.{adapter.submit_extension}"
            sub.write_text(body)
            sub.chmod(0o755)

            if args.submit:
                subprocess.run([adapter.submit_cmd, str(sub)], check=True)
            submitted += 1
            if args.test:
                print(f"test mode: stopping after {sub}")
                return 0

    verb = "submitted" if args.submit else "prepared"
    print(f"{verb} {submitted} cells under {out_base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
