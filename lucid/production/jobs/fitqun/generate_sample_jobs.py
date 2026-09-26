#!/usr/bin/env python3
"""Fan out the isotropic sample behind the angular-response and indirect-light tables.

The reference builds both tables from one production of 3 MeV electrons spread
uniformly through the detector, so this stage produces that sample once and
reduces it for both.

This follows ``scattab_nuPRISM_mPMT.mac``: a 3 MeV electron bomb, one electron
per event, isotropic, with multiple scattering inactivated. The tables are
generated the way the reference generates them.

Letting the shotgun sample its own photons instead is *not* equivalent. It
emits one direction per case, so a case lands on a single sensor and yields one
(source, sensor) geometry, whereas an electron emits a Cherenkov cone across a
ring of sensors. The marginal distributions agree; the correlation structure
does not, and the angular response is built out of the correlation structure. A
6.4e10-photon run done that way gave an angular response 3-10x noisier than
Poisson.

Each job is a chain, the way the Cherenkov-profile cells are:

    PhotonSim -> propagate and reduce in one pass -> delete the ROOT

Propagation and reduction are one step because the per-photon arrays for a job
are several GB and nothing needs them afterwards. Writing them out only to read
them back cost that much I/O per job and, with a few hundred jobs doing it at
once, failed outright: EOS returned ``errno 121`` mid-write for ~4% of a
250-job run. Now the only thing a job writes is its shard.

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
                n_photons: int, seed: int, shells) -> str:
    """The one-liner a shard job runs inside the container.

    The macro is written host-side at submit time, so this stays free of nested
    quoting. The && chain deletes the ROOT and the propagated photons only once
    the shard exists, leaving both to debug with when a step fails.
    """
    macro = job_dir / "photonsim.mac"
    root = job_dir / "photonsim.root"
    shard = job_dir / "shard.npz"
    shell_args = " ".join(f"{float(r):g}" for r in shells)

    # PHOTONSIM_BIN is inlined by the HTCondor adapter when a dev checkout is
    # configured; fall back to the image's binary otherwise.
    run = f"${{PHOTONSIM_BIN:-/opt/PhotonSim/build/PhotonSim}} {macro}"
    reduce_ = (f"python -m lucid.production.fitqun sample run {root} "
               f"--geometry {geometry} --detector-config {detector} "
               f"--physics-config {physics} --n-photons {n_photons} "
               f"--shells {shell_args} --seed {seed} -o {shard}")
    cleanup = f"rm -f {root}"
    return " && ".join([run, reduce_, cleanup])


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
    shells = cfg.get("shell_radii_cm")
    if shells is None:
        from lucid.production.fitqun import binning
        shells = list(binning.ANGRESP_SHELL_RADII_CM)
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

        # Deferred: the macro builder needs only the numpy-free half of the
        # package, so a bare submit host is still enough.
        from lucid.production.fitqun import isotropic_sample as iso
        (job_dir / "photonsim.mac").write_text(iso.photonsim_macro(
            output_path=job_dir / "photonsim.root",
            n_events=int(cfg["events_per_job"]),
            seed=int(cfg.get("seed_base", 0)) + job_id))

        body = adapter.render_command_job(
            command=job_command(
                job_dir=job_dir, geometry=geometry, detector=detector,
                physics=physics, n_photons=int(cfg["photons_per_group"]),
                seed=int(cfg.get("seed_base", 0)) + job_id, shells=shells),
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
