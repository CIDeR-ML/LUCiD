#!/usr/bin/env python3
"""
Timing Report Generator for PhotonSim Jobs

Parses SLURM job output files to extract per-event LUCiD processing times
and generates statistics and histograms.

Usage:
    # Analyze specific config directory
    python timing_report.py --config-dir /path/to/config_000001

    # Analyze all configs in base directory
    python timing_report.py --all --base-dir /path/to/water/uniform_energy

    # Save histogram to file
    python timing_report.py --all --base-dir /path/to/output --output timing_report.png
"""

import argparse
import re
import os
import glob
import numpy as np
from pathlib import Path


def parse_job_output(filepath):
    """
    Parse a job output file and extract per-event processing times.

    Looks for lines like: "Event total time: X.XXs"

    Returns:
        list of floats: Processing times in seconds
    """
    times = []
    # Match "Event total time: X.XXs" format from LUCiD output
    pattern = re.compile(r'Event total time: ([\d.]+)s')

    try:
        with open(filepath, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    times.append(float(match.group(1)))
    except Exception as e:
        print(f"  Warning: Could not read {filepath}: {e}")

    return times


def get_config_name(config_dir):
    """
    Try to get the config name from the README.md in the config directory.
    """
    readme_path = os.path.join(config_dir, 'README.md')
    if os.path.exists(readme_path):
        try:
            with open(readme_path, 'r') as f:
                for line in f:
                    if '**Configuration Name**:' in line:
                        return line.split(':')[-1].strip()
        except:
            pass
    return os.path.basename(config_dir)


def analyze_config(config_dir, use_latest_only=True):
    """
    Analyze job output files in a config directory.

    Args:
        config_dir: Path to config directory
        use_latest_only: If True, only use the most recent output file per job ID

    Returns:
        dict with config name, times, and statistics
    """
    config_name = get_config_name(config_dir)

    # Find all job output files (format: job_XXXXXX-SLURM_ID.out)
    output_files = glob.glob(os.path.join(config_dir, 'job_*-*.out'))

    if not output_files:
        return None

    if use_latest_only:
        # Group by job ID and keep only the most recent (highest SLURM job ID)
        job_files = {}
        for f in output_files:
            basename = os.path.basename(f)
            # Extract job ID (e.g., "000001" from "job_000001-16025048.out")
            match = re.match(r'job_(\d+)-(\d+)\.out', basename)
            if match:
                job_id = match.group(1)
                slurm_id = int(match.group(2))
                if job_id not in job_files or slurm_id > job_files[job_id][1]:
                    job_files[job_id] = (f, slurm_id)

        output_files = [f for f, _ in job_files.values()]

    all_times = []
    n_jobs = 0

    for output_file in sorted(output_files):
        times = parse_job_output(output_file)
        if times:
            all_times.extend(times)
            n_jobs += 1

    if not all_times:
        return None

    times_array = np.array(all_times)

    return {
        'config_dir': config_dir,
        'config_id': os.path.basename(config_dir),
        'config_name': config_name,
        'n_jobs': n_jobs,
        'n_events': len(all_times),
        'times': times_array,
        'min': np.min(times_array),
        'max': np.max(times_array),
        'mean': np.mean(times_array),
        'median': np.median(times_array),
        'std': np.std(times_array),
        'p25': np.percentile(times_array, 25),
        'p75': np.percentile(times_array, 75),
        'p95': np.percentile(times_array, 95),
    }


def print_statistics(result):
    """Print statistics for a config analysis result."""
    print(f"\n{'='*60}")
    print(f"Config: {result['config_id']}")
    print(f"Name: {result['config_name']}")
    print(f"{'='*60}")
    print(f"Jobs analyzed: {result['n_jobs']}")
    print(f"Events analyzed: {result['n_events']}")
    print(f"\nTime distribution (seconds):")
    print(f"  Min:      {result['min']:.3f}")
    print(f"  25th %:   {result['p25']:.3f}")
    print(f"  Median:   {result['median']:.3f}")
    print(f"  Mean:     {result['mean']:.3f}")
    print(f"  75th %:   {result['p75']:.3f}")
    print(f"  95th %:   {result['p95']:.3f}")
    print(f"  Max:      {result['max']:.3f}")
    print(f"  Std Dev:  {result['std']:.3f}")

    # ASCII histogram
    print_ascii_histogram(result['times'])


def print_ascii_histogram(times, bins=10, width=40):
    """Print an ASCII histogram of the times."""
    counts, edges = np.histogram(times, bins=bins)
    max_count = max(counts)

    print(f"\nHistogram:")
    for i, count in enumerate(counts):
        bar_len = int(width * count / max_count) if max_count > 0 else 0
        bar = '█' * bar_len
        print(f"  {edges[i]:6.2f}-{edges[i+1]:6.2f}s: {bar} ({count})")


def plot_histograms(results, output_file=None):
    """
    Generate matplotlib histograms for all configs.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("\nWarning: matplotlib not available, skipping plot generation")
        return

    n_configs = len(results)
    if n_configs == 0:
        return

    # Use 3x3 grid for 9 configs
    if n_configs <= 4:
        rows, cols = 2, 2
    elif n_configs <= 6:
        rows, cols = 2, 3
    elif n_configs <= 9:
        rows, cols = 3, 3
    else:
        rows = (n_configs + 2) // 3
        cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 4*rows))
    axes = axes.flatten() if n_configs > 1 else [axes]

    # Color palette for different configs
    colors = plt.cm.tab10.colors

    for i, result in enumerate(results):
        ax = axes[i]
        times = result['times']
        color = colors[i % len(colors)]

        # Create histogram with auto-scaled bins
        # Use range parameter to set bin edges, which ensures proper bin distribution
        p99 = np.percentile(times, 99)
        n, bins, patches = ax.hist(times, bins=15, color=color,
                                    edgecolor='white', alpha=0.8, linewidth=0.5)

        # Color the bars
        for patch in patches:
            patch.set_facecolor(color)

        # Add vertical lines for mean and median
        ax.axvline(result['mean'], color='darkred', linestyle='--', linewidth=1.5,
                   label=f"Mean: {result['mean']:.2f}s")
        ax.axvline(result['median'], color='darkgreen', linestyle='-', linewidth=1.5,
                   label=f"Median: {result['median']:.2f}s")

        # Labels and title
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)

        # Shorter title
        config_num = result['config_id'].replace('config_', '#')
        short_name = result['config_name'][:25]
        ax.set_title(f"{config_num}: {short_name}", fontsize=10, fontweight='bold')

        ax.legend(fontsize=7, loc='upper right')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Add stats box
        stats_text = f"n={result['n_events']}\nmin={result['min']:.2f}s\nmax={result['max']:.2f}s"
        ax.text(0.98, 0.72, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

        # x-axis limits match histogram range
        # ax.set_xlim(0, p99)

    # Hide unused subplots
    for i in range(n_configs, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('LUCiD Event Processing Time Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nHistogram saved to: {output_file}")
    else:
        # Save to default file
        default_output = 'timing_report.png'
        plt.savefig(default_output, dpi=150, bbox_inches='tight')
        print(f"\nHistogram saved to: {default_output}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate timing report from PhotonSim job outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--config-dir', type=str,
                        help='Analyze a specific config directory')
    parser.add_argument('--all', action='store_true',
                        help='Analyze all configs in base directory')
    parser.add_argument('--base-dir', type=str,
                        help='Base directory containing config_XXXXXX folders')
    parser.add_argument('--output', type=str,
                        help='Output file for histogram (PNG)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plot generation, only print statistics')

    args = parser.parse_args()

    results = []

    if args.config_dir:
        # Analyze single config
        if not os.path.isdir(args.config_dir):
            print(f"Error: Directory not found: {args.config_dir}")
            return 1

        result = analyze_config(args.config_dir)
        if result:
            results.append(result)
        else:
            print(f"No timing data found in {args.config_dir}")
            return 1

    elif args.all and args.base_dir:
        # Analyze all configs in base directory
        if not os.path.isdir(args.base_dir):
            print(f"Error: Directory not found: {args.base_dir}")
            return 1

        # Find all config directories
        config_dirs = sorted(glob.glob(os.path.join(args.base_dir, 'config_*')))

        if not config_dirs:
            print(f"No config_* directories found in {args.base_dir}")
            return 1

        print(f"Found {len(config_dirs)} config directories")

        for config_dir in config_dirs:
            result = analyze_config(config_dir)
            if result:
                results.append(result)

    else:
        parser.print_help()
        print("\nError: Please specify --config-dir or --all with --base-dir")
        return 1

    if not results:
        print("No timing data found")
        return 1

    # Print statistics for all results
    for result in results:
        print_statistics(result)

    # Print summary table
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY TABLE")
        print(f"{'='*80}")
        print(f"{'Config':<20} {'Events':>8} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print("-" * 80)
        for r in results:
            print(f"{r['config_id']:<20} {r['n_events']:>8} {r['mean']:>8.2f} {r['median']:>8.2f} {r['std']:>8.2f} {r['min']:>8.2f} {r['max']:>8.2f}")

    # Generate plots
    if not args.no_plot:
        plot_histograms(results, args.output)

    return 0


if __name__ == '__main__':
    exit(main())
