#!/usr/bin/ python3
"""
Submit a SLURM job to run track optimization on S3DF.

This script generates a SLURM batch script and submits it to the queue.
Each job will process all configurations listed in the config file sequentially.

Usage:
    python submit_job.py --config CONFIG_FILE --output OUTPUT_DIR [--job-name JOB_NAME]

Arguments:
    --config: Path to JSON file containing list of optimization configs (with full paths)
    --output: Directory where optimization results will be saved
    --job-name: Name for the SLURM job (optional, default: track_opt)
    --submit: Actually submit the job (default: False, just generate script)
"""

import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Submit SLURM job for track optimization'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to JSON file containing list of optimization configs'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Directory where optimization results will be saved'
    )
    parser.add_argument(
        '--job-name',
        type=str,
        default='track_opt',
        help='Name for the SLURM job (default: track_opt)'
    )
    parser.add_argument(
        '--submit',
        action='store_true',
        help='Actually submit the job (default: just generate script)'
    )
    parser.add_argument(
        '--partition',
        type=str,
        default='ampere',
        help='SLURM partition (default: ampere)'
    )
    parser.add_argument(
        '--account',
        type=str,
        default='mli:cider-ml',
        help='SLURM account (default: mli:cider-ml)'
    )
    return parser.parse_args()


def generate_slurm_script(config_file, output_dir, job_name, project_root, partition='ampere', account='mli:cider-ml'):
    """Generate SLURM batch script content."""

    # Convert paths to absolute
    config_path = Path(config_file).resolve()
    output_path = Path(output_dir).resolve()
    run_script = project_root / 's3df_jobs' / 'run_track_optimization.py'

    # Create log directory
    log_dir = project_root / 's3df_jobs' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{job_name}_{timestamp}.log'

    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_file}
#SBATCH --error={log_file}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=39936
#SBATCH --time=23:00:00

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="
echo ""

# Change to project directory
cd {project_root}
echo "Working directory: $(pwd)"
echo ""

# Setup Singularity environment
echo "Setting up Singularity environment..."
export SINGULARITY_IMAGE_PATH=/sdf/group/neutrino/images/develop.sif
function spython() {{
    singularity exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${{SINGULARITY_IMAGE_PATH}} python "$@"
}}
echo "Singularity image: $SINGULARITY_IMAGE_PATH"
echo ""

# Print GPU information from inside container
echo "GPU Information (from inside container):"
singularity exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${{SINGULARITY_IMAGE_PATH}} nvidia-smi
echo ""

# Run the optimization script
echo "Starting track optimization..."
spython -u {run_script} \\
    --config {config_path} \\
    --output {output_path}

exit_code=$?

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
"""

    return slurm_script, log_file


def main():
    """Main execution function."""
    args = parse_args()

    # Get project root (parent of s3df_jobs directory)
    project_root = Path(__file__).resolve().parent.parent

    # Debug output
    print(f"Script location: {Path(__file__).resolve()}")
    print(f"Project root: {project_root}\n")

    # Verify config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        return 1

    # Generate SLURM script
    slurm_script, log_file = generate_slurm_script(
        args.config,
        args.output,
        args.job_name,
        project_root,
        args.partition,
        args.account
    )

    # Save SLURM script
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    script_dir = project_root / 's3df_jobs' / 'generated_scripts'
    script_dir.mkdir(parents=True, exist_ok=True)

    script_file = script_dir / f'{args.job_name}_{timestamp}.sh'
    with open(script_file, 'w') as f:
        f.write(slurm_script)

    # Make script executable
    script_file.chmod(0o755)

    print(f"{'='*80}")
    print(f"SLURM Job Script Generated")
    print(f"{'='*80}")
    print(f"Script: {script_file}")
    print(f"Config: {Path(args.config).resolve()}")
    print(f"Output: {Path(args.output).resolve()}")
    print(f"Log file: {log_file}")
    print(f"{'='*80}")

    # Submit job if requested
    if args.submit:
        print("\nSubmitting job to SLURM...")
        try:
            subprocess.run(['sbatch', str(script_file)], check=True)
            print(f"\n✓ Job submitted successfully!")
            print(f"\nMonitor with: squeue -u $USER")
            print(f"View log: tail -f {log_file}")

        except subprocess.CalledProcessError as e:
            print(f"✗ Error submitting job: {e}")
            return 1
        except FileNotFoundError:
            print("✗ Error: 'sbatch' command not found. Are you on a SLURM system?")
            return 1
    else:
        print("\nTo submit this job, run:")
        print(f"  sbatch {script_file}")
        print("\nOr re-run this script with --submit flag:")
        print(f"  python3 {Path(__file__).name} --config {args.config} --output {args.output} --submit")

    return 0


if __name__ == '__main__':
    exit(main())
