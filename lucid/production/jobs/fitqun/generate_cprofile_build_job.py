#!/usr/bin/env python3
"""Submit the Cherenkov-profile merge as a batch job.

``cprofile build`` reads every cell of one PDG and writes the single
``CProf_<pdg>_WCSim.root`` that ``fiTQun_shared::LoadProfiles`` reads. With the
reference's grid that is 551 cells per particle, each holding the I_n integrals
on a 401 x 201 x 2200 axis set, so the merge is minutes of solid CPU and tens
of GB of transient memory -- a batch job, not something to run on a login node.

One job covers all three PDGs in sequence rather than three jobs, because the
merge is IO-bound on EOS and running them concurrently only contends.

    ./generate_cprofile_build_job.py -s
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
JOBS_DIR = SCRIPT_DIR.parent
USER_PATHS_DEFAULT = JOBS_DIR / "user_paths.sh"
LUCID_ROOT = SCRIPT_DIR.resolve().parents[3]
if str(LUCID_ROOT) not in sys.path:
    sys.path.insert(0, str(LUCID_ROOT))

from lucid.production.cluster_common import htcondor  # noqa: E402, F401  registers adapter
from lucid.production.cluster_common import nersc  # noqa: E402, F401  registers adapter
from lucid.production.cluster_common.cluster import get_adapter  # noqa: E402
from lucid.production.cluster_common.user_paths import load_user_paths  # noqa: E402

PDGS = (11, 13, 211)


def build_command(cell_root: Path) -> str:
    """Merge each PDG's cells, chained so a failure stops the rest."""
    steps = []
    for pdg in PDGS:
        steps.append(
            f"python -m lucid.production.fitqun cprofile build "
            f"{cell_root}/{pdg}/*/cell.npz --pdg {pdg} "
            f"-o {cell_root}/CProf_{pdg}_WCSim.root")
    return " && ".join(steps)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-s", "--submit", action="store_true")
    p.add_argument("-o", "--output-base", type=Path, default=None,
                   help="cell root (default: <OUTPUT_BASE_PATH>/fitqun_full/cprofile)")
    p.add_argument("-P", "--partition", type=str, default="")
    p.add_argument("--request-memory-mb", type=int, default=32768)
    p.add_argument("--user-paths", type=Path, default=USER_PATHS_DEFAULT)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    env = load_user_paths(args.user_paths)
    adapter = get_adapter(env)

    cell_root = args.output_base or Path(env["OUTPUT_BASE_PATH"]) / "fitqun_full" / "cprofile"
    partition = (args.partition or env.get("SLURM_PARTITION")
                 or env.get("CONDOR_JOB_FLAVOUR", ""))
    if not partition:
        raise SystemExit("no partition/flavour: pass -P or set it in user_paths.sh")

    body = adapter.render_command_job(
        command=build_command(cell_root), cell_dir=cell_root,
        job_name="fitqun_cprofile_build", log_stem="cprofile_build",
        partition=partition, request_disk_mb=8192)
    # The merge holds a full I_n table in memory; the adapter's default is the
    # per-event production figure and is not enough here.
    body = body.replace("request_memory            = 12000",
                        f"request_memory            = {args.request_memory_mb}")
    body = body.replace("request_memory          = 12000",
                        f"request_memory          = {args.request_memory_mb}")

    sub = cell_root / f"cprofile_build.{adapter.submit_extension}"
    sub.write_text(body)
    sub.chmod(0o755)
    print(f"[PREPARED] {sub}")

    if args.submit:
        subprocess.run([adapter.submit_cmd, str(sub)], check=True)
        print(f"submitted {sub}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
