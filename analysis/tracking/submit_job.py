#!/usr/bin/env python3
"""Submit tracking-study configs to SLURM on S3DF — one GPU job per config.

    # single config
    python analysis/tracking/submit_job.py --config configs/nrays/muon/config_00.json \
        --output /sdf/data/neutrino/cjesus/CIDER/LUCiD_tracking/muon/nrays --submit

    # every config in a directory (one job each)
    python analysis/tracking/submit_job.py --config configs/nrays/muon \
        --output /sdf/data/neutrino/cjesus/CIDER/LUCiD_tracking/muon/nrays --submit

Each job runs ``run_study.py`` inside the LUCiD container on a GPU node (default partition
``turing``). Per QUICKSTART_S3DF §"Running the JAX stack on a GPU", the baked jaxlib is
CPU-only, so the CUDA build is layered on via ``APPTAINERENV_PYTHONUSERBASE`` + ``--nv``; the
checkout's own ``lucid`` wins because run_study.py inserts the repo root first on sys.path.

Without ``--submit`` the sbatch scripts are written (to ``<output>/slurm/``) but not queued.
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKING_DIR = REPO_ROOT / 'analysis' / 'tracking'
DEFAULT_IMAGE = os.environ.get('LUCID_IMAGE_PATH',
                               '/sdf/data/neutrino/cjesus/software/images/lucid.sif')
DEFAULT_ENV_BASE = '/sdf/data/neutrino/cjesus/python_envs/lucid'
BINDS = '/sdf,/fs,/sdf/scratch,/lscratch,/cvmfs'


def sbatch_text(cfg_path, out_dir, log_dir, a):
    """SLURM batch script for one config."""
    run_script = (TRACKING_DIR / a.run_script).resolve()
    job = a.job_name or f"trk_{cfg_path.stem}"
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log = log_dir / f'{job}_{stamp}.log'
    return f"""#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --output={log}
#SBATCH --error={log}
#SBATCH --partition={a.partition}
#SBATCH --account={a.account}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={a.cpus}
#SBATCH --gpus={a.gpus}
#SBATCH --mem={a.mem}
#SBATCH --time={a.time}

set -euo pipefail
echo "=== job $SLURM_JOB_ID on $SLURM_NODELIST  $(date) ==="
nvidia-smi -L || true

APPTAINERENV_PYTHONUSERBASE={a.env_base} \\
APPTAINERENV_PYTHONPATH="" \\
apptainer exec --nv -B {BINDS} \\
    {a.image} \\
    /opt/conda/bin/python3 {run_script} \\
        --config {cfg_path} \\
        --output {out_dir}

echo "=== done  $(date) ==="
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='config JSON, or a dir of *.json (one job each)')
    ap.add_argument('--output', required=True, help='output dir for the HDF5 results')
    ap.add_argument('--job-name', default=None)
    ap.add_argument('--run-script', default='run_study.py',
                    help="script in analysis/tracking/ to run (run_study.py | seed_study.py)")
    ap.add_argument('--partition', default='turing', help='GPU partition (default turing)')
    ap.add_argument('--account', default='mli:cider-ml')
    ap.add_argument('--image', default=DEFAULT_IMAGE)
    ap.add_argument('--env-base', default=DEFAULT_ENV_BASE,
                    help='PYTHONUSERBASE layered in for the CUDA jax wheels')
    ap.add_argument('--gpus', default='1')
    ap.add_argument('--cpus', default='4')
    ap.add_argument('--mem', default='39936')
    ap.add_argument('--time', default='23:00:00')
    ap.add_argument('--submit', action='store_true', help='actually queue the jobs')
    args = ap.parse_args()

    cfg_arg = Path(args.config)
    configs = sorted(cfg_arg.glob('*.json')) if cfg_arg.is_dir() else [cfg_arg]
    if not configs:
        print(f"No config JSON found at {cfg_arg}", file=sys.stderr); return 1

    out_dir = Path(args.output).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / 'logs'; log_dir.mkdir(exist_ok=True)
    slurm_dir = out_dir / 'slurm'; slurm_dir.mkdir(exist_ok=True)

    print(f"image     : {args.image}")
    print(f"partition : {args.partition}   env-base: {args.env_base}")
    print(f"output    : {out_dir}")
    print(f"configs   : {len(configs)}\n")

    for cfg in configs:
        script = slurm_dir / f'{cfg.stem}.sh'
        script.write_text(sbatch_text(cfg.resolve(), out_dir, log_dir, args))
        script.chmod(0o755)
        if args.submit:
            subprocess.run(['sbatch', str(script)], check=True)
            print(f"  submitted {cfg.name}")
        else:
            print(f"  wrote {script}")

    if not args.submit:
        print(f"\nDry run. Re-run with --submit to queue, or: for f in {slurm_dir}/*.sh; do sbatch $f; done")
    return 0


if __name__ == '__main__':
    sys.exit(main())
