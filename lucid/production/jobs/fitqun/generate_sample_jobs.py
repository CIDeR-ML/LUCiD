#!/usr/bin/env python3
"""Fan out the isotropic sample behind the angular-response and indirect-light tables.

The reference builds both tables from one production of 3 MeV electrons spread
uniformly through the detector, so this stage produces that sample once and
reduces it for both.

**No PhotonSim.** The reference fires electrons because WCSim can only make
light that way; what the tables actually bin is the photon's emission point and
direction, and the shotgun samples both directly -- uniform in the volume,
isotropic in direction. Every axis of both tables sees the same distribution,
and a 3 MeV electron's ~1.5 cm range is negligible against ``zs`` bins ~1 m
wide. Dropping the electron stage removes a Geant4 pass and the terabytes of
intermediate ROOT that went with it.

Each job is a chain, the way the Cherenkov-profile cells are:

    photon shotgun -> reduce to a shard -> delete the propagated photons

The reduce happens *inside* the job on purpose. The propagated photon list for
the full sample is terabytes; reduced to sparse counts it is a few gigabytes,
and only the reduced form has to survive until the merge. Chaining with ``&&``
also means the shard's input is deleted only once the shard exists, so a failed
job leaves it behind to debug with.

Run it like its sibling:

    ./generate_sample_jobs.py -c configs/sample_e3mev_test.json -t -s
    python -m lucid.production.fitqun sample build <out>/job_*/shard.npz -o <out>
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


def job_command(*, job_dir: Path, geometry: Path, detector: Path, physics: Path,
                n_cases: int, n_photons: int, seed: int, shells,
                position_fraction: float) -> str:
    """The one-liner a shard job runs inside the container."""
    shotgun = job_dir / "shotgun.h5"
    shard = job_dir / "shard.npz"
    shell_args = " ".join(f"{float(r):g}" for r in shells)

    propagate = (
        f"python -m lucid.production.photon_shotgun.run "
        f"--detector {detector} --physics-config {physics} "
        f"--n-cases {n_cases} --n-photons {n_photons} "
        f"--position-mode uniform --position-fraction {position_fraction:g} "
        f"--direction-mode isotropic "
        f"--output-mode per_photon --save-source --seed {seed} -o {shotgun}")
    reduce_ = (f"python -m lucid.production.fitqun sample accumulate {shotgun} "
               f"--geometry {geometry} --shells {shell_args} -o {shard}")
    cleanup = f"rm -f {shotgun}"
    return " && ".join([propagate, reduce_, cleanup])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", type=Path, required=True)
    p.add_argument("-s", "--submit", action="store_true")
    p.add_argument("-t", "--test", action="store_true",
                   help="prepare (and submit) only the first job")
    p.add_argument("-o", "--output-base", type=Path, default=None)
    p.add_argument("-P", "--partition", type=str, default="")
    p.add_argument("-N", "--n-jobs", type=int, default=None,
                   help="override the config's job count for this invocation")
    p.add_argument("--user-paths", type=Path, default=USER_PATHS_DEFAULT)
    p.add_argument("--no-skip-existing", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = json.loads(args.config.read_text())
    env = load_user_paths(args.user_paths)
    adapter = get_adapter(env)

    out_base = args.output_base or Path(env["OUTPUT_BASE_PATH"]) / "fitqun_sample"
    partition = (args.partition or env.get("SLURM_PARTITION")
                 or env.get("CONDOR_JOB_FLAVOUR", ""))
    if not partition:
        raise SystemExit("no partition/flavour: pass -P or set it in user_paths.sh")

    n_jobs = int(args.n_jobs or cfg["n_jobs"])
    shells = cfg.get("shell_radii_cm", [100.0, 200.0, 400.0, 800.0, 1200.0])
    geometry = Path(cfg["geometry"])
    detector = Path(cfg["detector_config"])
    physics = Path(cfg["physics_config"])
    for path in (geometry, detector, physics):
        if not path.is_absolute():
            raise SystemExit(f"{path}: config paths must be absolute (the job "
                             "runs from an unspecified cwd inside the container)")

    submitted = 0
    for job_id in range(1, n_jobs + 1):
        job_dir = out_base / f"job_{job_id:06d}"
        if (job_dir / "shard.npz").exists() and not args.no_skip_existing:
            continue
        job_dir.mkdir(parents=True, exist_ok=True)

        body = adapter.render_command_job(
            command=job_command(
                job_dir=job_dir, geometry=geometry, detector=detector,
                physics=physics, n_cases=int(cfg["cases_per_job"]),
                n_photons=int(cfg["photons_per_case"]),
                seed=int(cfg.get("seed_base", 0)) + job_id, shells=shells,
                position_fraction=float(cfg.get("position_fraction", 0.9))),
            cell_dir=job_dir, job_name=f"{cfg['name']}_{job_id:06d}",
            log_stem=f"job_{job_id:06d}", partition=partition,
            # Only the propagated photon list needs scratch; it is deleted as
            # soon as the shard exists.
            request_disk_mb=int(cfg.get("request_disk_mb", 8192)))

        sub = job_dir / f"sample.{adapter.submit_extension}"
        sub.write_text(body)
        sub.chmod(0o755)
        # Stable marker for the bash shim's host-side submission pass.
        print(f"[PREPARED] {sub}")

        if args.submit:
            subprocess.run([adapter.submit_cmd, str(sub)], check=True)
        submitted += 1
        if args.test:
            print(f"test mode: stopping after {sub}")
            return 0

    verb = "submitted" if args.submit else "prepared"
    print(f"{verb} {submitted} jobs under {out_base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
