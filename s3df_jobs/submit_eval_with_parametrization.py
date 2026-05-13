#!/usr/bin/env python3
"""
Submit a SLURM job to evaluate reconstruction with learned tau_vtx parametrization.

This script evaluates reconstruction performance using:
    tau_vtx = a * Nrays + b * E + c

where tau_vtx is computed dynamically during optimization based on the
current reconstructed energy (with stop_gradient).

Grid: 9 combinations (3 Nrays x 3 Energy)
Events: 50 per combination = 450 total

Usage:
    python submit_eval_with_parametrization.py --output OUTPUT_DIR [--n-events N] [--submit]
"""

import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Submit SLURM job for evaluation with learned tau_vtx parametrization'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Directory where results will be saved'
    )
    parser.add_argument(
        '--n-events',
        type=int,
        default=50,
        help='Number of events per parameter combination (default: 50)'
    )
    parser.add_argument(
        '--job-name',
        type=str,
        default='eval_tau_param',
        help='Name for the SLURM job (default: eval_tau_param)'
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
    parser.add_argument(
        '--time',
        type=str,
        default='08:00:00',
        help='Job time limit HH:MM:SS (default: 08:00:00)'
    )
    return parser.parse_args()


def generate_slurm_script(output_dir, n_events, job_name, project_root,
                          partition='ampere', account='mli:cider-ml', time_limit='08:00:00'):
    """Generate SLURM batch script content."""

    output_path = Path(output_dir).resolve()
    run_script = project_root / 's3df_jobs' / 'run_eval_with_parametrization.py'

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
#SBATCH --time={time_limit}

# Print job information
echo "=========================================="
echo "EVALUATION WITH LEARNED TAU_VTX PARAMETRIZATION"
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
export SINGULARITY_IMAGE_PATH=/sdf/data/neutrino/cjesus/software/images/lucid.sif
function spython() {{
    singularity exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${{SINGULARITY_IMAGE_PATH}} python "$@"
}}
echo "Singularity image: $SINGULARITY_IMAGE_PATH"
echo ""

# Print GPU information from inside container
echo "GPU Information (from inside container):"
singularity exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${{SINGULARITY_IMAGE_PATH}} nvidia-smi
echo ""

# Print evaluation parameters
echo "=========================================="
echo "Evaluation Parameters:"
echo "  Output directory: {output_path}"
echo "  Events per combination: {n_events}"
echo ""
echo "  Parametrization: tau_vtx = a*Nrays + b*E + c"
echo "    a = 1.092557e-06"
echo "    b = 2.578522e-04"
echo "    c = -0.0442"
echo ""
echo "  Grid:"
echo "    Nrays: [50k, 150k, 250k]"
echo "    Energy: [500, 1000, 1500] MeV"
echo "  Total combinations: 9"
echo "  Total events: {9 * n_events}"
echo "=========================================="
echo ""

# Run the evaluation script
echo "Starting evaluation with learned tau_vtx parametrization..."
spython -u {run_script} \\
    --output {output_path} \\
    --n-events {n_events}

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

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate SLURM script
    slurm_script, log_file = generate_slurm_script(
        args.output,
        args.n_events,
        args.job_name,
        project_root,
        args.partition,
        args.account,
        args.time
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
    print(f"EVALUATION WITH LEARNED TAU_VTX PARAMETRIZATION - SLURM Job Script Generated")
    print(f"{'='*80}")
    print(f"Script: {script_file}")
    print(f"Output: {Path(args.output).resolve()}")
    print(f"Log file: {log_file}")
    print(f"Events per combination: {args.n_events}")
    print(f"Time limit: {args.time}")
    print()
    print("Parametrization: tau_vtx = 1.093e-6 * Nrays + 2.579e-4 * E - 0.044")
    print()
    print("Evaluation grid:")
    print("  Nrays: [50k, 150k, 250k]")
    print("  Energy: [500, 1000, 1500] MeV")
    print(f"  Total: 9 combinations x {args.n_events} events = {9 * args.n_events} events")
    print(f"{'='*80}")

    # Submit job if requested
    if args.submit:
        print("\nSubmitting job to SLURM...")
        try:
            result = subprocess.run(['sbatch', str(script_file)], check=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"\n{result.stdout.decode().strip()}")
            print(f"\nMonitor with: squeue -u $USER")
            print(f"View log: tail -f {log_file}")

        except subprocess.CalledProcessError as e:
            print(f"Error submitting job: {e}")
            if e.stderr:
                print(f"STDERR: {e.stderr}")
            return 1
        except FileNotFoundError:
            print("Error: 'sbatch' command not found. Are you on a SLURM system?")
            return 1
    else:
        print("\nTo submit this job, run:")
        print(f"  sbatch {script_file}")
        print("\nOr re-run this script with --submit flag:")
        print(f"  python3 {Path(__file__).name} --output {args.output} --submit")

    return 0


if __name__ == '__main__':
    exit(main())
