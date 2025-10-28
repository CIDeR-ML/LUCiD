#!/usr/bin/env python
"""
Run track optimization for multiple configurations sequentially.

This script is designed to run as a SLURM job on S3DF, processing multiple
optimization configurations sequentially on a single GPU.

Usage:
    python run_track_optimization.py --config CONFIG_FILE --output OUTPUT_DIR

Arguments:
    --config: Path to JSON file containing list of optimization configs (with full paths)
    --output: Directory where optimization results will be saved
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run track optimization for multiple configurations'
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
    return parser.parse_args()


def load_config_list(config_file):
    """Load the list of configuration files to process."""
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_path, 'r') as f:
        config_list = json.load(f)

    # Validate that all config files exist
    for config_info in config_list:
        if 'config_path' not in config_info:
            raise ValueError(f"Config entry missing 'config_path' field: {config_info}")

        config_file_path = Path(config_info['config_path'])
        if not config_file_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_info['config_path']}")

    return config_list


def run_optimization(config_file, optimization_script, output_dir, name=None):
    """Run the optimization script for a single configuration."""
    cmd = ['python', '-u', str(optimization_script), str(config_file), '--output', str(output_dir)]

    # Add name argument if provided
    if name:
        cmd.extend(['--name', name])

    # Run the optimization
    subprocess.run(cmd, check=True)


def main():
    """Main execution function."""
    args = parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    optimization_script = project_root / 'tools' / 'optimization' / 'single_track_optimization.py'
    output_dir = Path(args.output)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*80}")
    print(f"Track Optimization Job")
    print(f"{'='*80}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config file: {args.config}")
    print(f"Output directory: {output_dir}")
    print(f"Optimization script: {optimization_script}")
    print(f"{'='*80}\n")

    # Verify optimization script exists
    if not optimization_script.exists():
        print(f"Error: Optimization script not found: {optimization_script}")
        sys.exit(1)

    # Load configuration list
    try:
        config_list = load_config_list(args.config)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

    print(f"Loaded {len(config_list)} configurations to process\n")

    # Track results
    successful = 0
    failed = 0

    # Process each configuration
    for i, config_info in enumerate(tqdm(config_list, desc="Running experiments")):
        config_file = config_info['config_path']
        config_name = config_info.get('name', Path(config_file).stem)

        print(f"\n{'='*80}")
        print(f"Configuration {i+1}/{len(config_list)}: {config_name}")
        print(f"Config file: {config_file}")
        print(f"{'='*80}")

        try:
            run_optimization(config_file, optimization_script, output_dir, name=config_info.get('name'))
            print(f"✓ Completed: {config_name}")
            successful += 1

        except subprocess.CalledProcessError as e:
            print(f"✗ Error running configuration {config_name}: {e}")
            failed += 1

        except Exception as e:
            print(f"✗ Unexpected error for configuration {config_name}: {e}")
            failed += 1

    # Print summary
    print(f"\n{'='*80}")
    print(f"Job Summary")
    print(f"{'='*80}")
    print(f"Total configurations: {len(config_list)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
