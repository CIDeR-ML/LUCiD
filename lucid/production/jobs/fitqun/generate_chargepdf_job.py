#!/usr/bin/env python3
"""Submit the charge-PDF scan as a batch job.

The scan is one process -- no geometry, no propagation, just LUCiD's digitizer
handed Poisson(mu) photoelectrons per PMT for each of the reference's 202 mu
points -- so unlike the Cherenkov profile it does not fan out over cells. It
still belongs in the batch system: the full grid is around three quarters of
an hour of solid CPU, which is not something to run on a login node.

Points are seeded ``seed_base + i``, so the scan is reproducible and a re-run
overwrites each file with identical content. There is no skip-existing: the
stage is cheap enough to redo whole, and a partial directory from an
interrupted run is then repaired by simply running it again.

    ./generate_chargepdf_job.py -c configs/chargepdf_sk.json -s
"""
from __future__ import annotations

import argparse
import json
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


def scan_command(*, out_dir: Path, n_pmt: int, n_events: int, model: str,
                 seed: int) -> str:
    """The one-liner the job runs inside the container."""
    return (f"python -m lucid.production.fitqun chargepdf scan "
            f"-o {out_dir} --n-pmt {n_pmt} --n-events {n_events} "
            f"--model {model} --seed {seed}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", type=Path, required=True)
    p.add_argument("-s", "--submit", action="store_true")
    p.add_argument("-t", "--test", action="store_true",
                   help="prepare only, do not submit")
    p.add_argument("-o", "--output-base", type=Path, default=None)
    p.add_argument("-P", "--partition", type=str, default="")
    p.add_argument("--user-paths", type=Path, default=USER_PATHS_DEFAULT)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = json.loads(args.config.read_text())
    env = load_user_paths(args.user_paths)
    adapter = get_adapter(env)

    out_dir = args.output_base or Path(env["OUTPUT_BASE_PATH"]) / "fitqun_full" / "chargepdf"
    partition = (args.partition or env.get("SLURM_PARTITION")
                 or env.get("CONDOR_JOB_FLAVOUR", ""))
    if not partition:
        raise SystemExit("no partition/flavour: pass -P or set it in user_paths.sh")
    out_dir.mkdir(parents=True, exist_ok=True)

    body = adapter.render_command_job(
        command=scan_command(out_dir=out_dir, n_pmt=int(cfg["n_pmt"]),
                             n_events=int(cfg["n_events"]),
                             model=cfg.get("model", "ski"),
                             seed=int(cfg.get("seed_base", 1))),
        cell_dir=out_dir, job_name=cfg["name"], log_stem="chargepdf",
        partition=partition,
        # Output is 202 small histogram files; the scan holds nothing large.
        request_disk_mb=int(cfg.get("request_disk_mb", 2048)))

    sub = out_dir / f"chargepdf.{adapter.submit_extension}"
    sub.write_text(body)
    sub.chmod(0o755)
    # Stable marker for the bash shim's host-side submission pass.
    print(f"[PREPARED] {sub}")

    if args.submit and not args.test:
        subprocess.run([adapter.submit_cmd, str(sub)], check=True)
        print(f"submitted {sub}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
