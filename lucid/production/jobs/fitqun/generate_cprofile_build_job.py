#!/usr/bin/env python3
"""Submit the Cherenkov-profile merge as a batch job.

``cprofile build`` reads every cell of one PDG and writes the single
``CProf_<pdg>_WCSim.root`` that ``fiTQun_shared::LoadProfiles`` reads. The
reference's grid is per particle -- 659 cells for e-, 551 for mu-, 532 for pi+ --
each holding the I_n integrals on a 401 x 201 axis set, so the merge is minutes
of solid CPU and tens of GB of transient memory: a batch job, not something to
run on a login node.

ONE JOB PER PDG. They were chained into a single job on the assumption that the
merge is IO-bound on EOS and concurrency would only contend -- but the jobs land
on different worker nodes reading disjoint cell sets, so any contention is
server-side, and the measured cost of chaining is concrete: the merge was the
whole tail of its stage at 4.2 h, against 23.5 min for the slowest of the 1778
scan jobs feeding it. One job per PDG puts the stage's critical path at the
slowest single PDG instead of the sum of all three.

``--pdgs`` selects which to build. A tune only needs the hypotheses it fits, and
the cells for the others stay on disk, so they can be built later without
re-running the scan:

    ./generate_cprofile_build_job.py --pdgs 13 -s     # muon tune only
    ./generate_cprofile_build_job.py -s               # all three, in parallel
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


def build_command(cell_root: Path, pdg: int) -> str:
    """Merge one PDG's cells into the file ``LoadProfiles`` reads."""
    return (f"python -m lucid.production.fitqun cprofile build "
            f"{cell_root}/{pdg}/*/cell.npz --pdg {pdg} "
            f"-o {cell_root}/CProf_{pdg}_WCSim.root")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-s", "--submit", action="store_true")
    p.add_argument("-o", "--output-base", type=Path, default=None,
                   help="cell root (default: <OUTPUT_BASE_PATH>/fitqun_full/cprofile)")
    p.add_argument("--pdgs", type=str, default=",".join(str(p) for p in PDGS),
                   help="comma list of PDGs to build, one job each "
                        "(default: all three)")
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

    pdgs = [int(x) for x in args.pdgs.split(",") if x.strip()]
    unknown = [p for p in pdgs if p not in PDGS]
    if unknown:
        raise SystemExit(f"no cells are scanned for PDG {unknown}; known: {list(PDGS)}")

    for pdg in pdgs:
        # The merge holds a full I_n table in memory; the adapter's default is
        # the per-event production figure and is not enough here.
        body = adapter.render_command_job(
            command=build_command(cell_root, pdg), cell_dir=cell_root,
            job_name=f"fitqun_cprofile_build_{pdg}",
            log_stem=f"cprofile_build_{pdg}",
            partition=partition, request_disk_mb=8192,
            request_memory_mb=args.request_memory_mb)

        sub = cell_root / f"cprofile_build_{pdg}.{adapter.submit_extension}"
        sub.parent.mkdir(parents=True, exist_ok=True)
        sub.write_text(body)
        sub.chmod(0o755)
        print(f"[PREPARED] {sub}")

        if args.submit:
            subprocess.run([adapter.submit_cmd, str(sub)], check=True)
            print(f"submitted {sub}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
