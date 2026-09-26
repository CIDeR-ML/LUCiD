#!/usr/bin/env python3
"""Submit the electron-bomb shard merge as a batch job.

``sample build`` sums every shard into the indirect-light tables and the
angular-response histograms. The scattering tables are densified to do it --
a barrel table is 35 x 16 x 35 x 16 x 16 x 16 = 80.3M cells and each cap 36.7M,
held for the scattered and direct sets at once, so the merge sits on ~2.5 GB
before the ratio step's temporaries. That is a batch job, not something to run
on a login node.

    ./generate_sample_build_job.py -s
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


def build_command(sample_root: Path, out_dir: Path) -> str:
    return (f"python -m lucid.production.fitqun sample build "
            f"{sample_root}/job_*/shard.npz -o {out_dir}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-s", "--submit", action="store_true")
    p.add_argument("-o", "--output-base", type=Path, default=None,
                   help="shard root (default: <OUTPUT_BASE_PATH>/fitqun_sample)")
    p.add_argument("-P", "--partition", type=str, default="")
    p.add_argument("--request-memory-mb", type=int, default=16384)
    p.add_argument("--user-paths", type=Path, default=USER_PATHS_DEFAULT)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    env = load_user_paths(args.user_paths)
    adapter = get_adapter(env)

    sample_root = args.output_base or Path(env["OUTPUT_BASE_PATH"]) / "fitqun_sample"
    partition = (args.partition or env.get("SLURM_PARTITION")
                 or env.get("CONDOR_JOB_FLAVOUR", ""))
    if not partition:
        raise SystemExit("no partition/flavour: pass -P or set it in user_paths.sh")

    out_dir = sample_root / "built"
    body = adapter.render_command_job(
        command=build_command(sample_root, out_dir), cell_dir=sample_root,
        job_name="fitqun_sample_build", log_stem="sample_build",
        partition=partition, request_disk_mb=8192,
        request_memory_mb=args.request_memory_mb)

    sub = sample_root / f"sample_build.{adapter.submit_extension}"
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
