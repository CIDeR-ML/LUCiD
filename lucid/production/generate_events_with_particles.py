#!/usr/bin/env python3
"""
Production script to generate events from PhotonSim ROOT files using particle-based workflow.
This script uses particle-based processing where photons are classified by genealogy.

Uses VMAP-optimized processing with jax.vmap for 5-10x speedup through vectorized particle processing.

Usage:
    python generate_events_with_particles.py --root-file path/to/file.root --config config.json --output output_dir/
"""

import argparse
import sys
import os

import jax.numpy as jnp

from lucid.sources.event_io import generate_events_from_photonsim_particles
from lucid.simulation import setup_event_simulator
from lucid.detector_params import DetectorParams
from lucid.utils import base_dir_path

def main():
    parser = argparse.ArgumentParser(
        description='Generate events from PhotonSim ROOT files using particle-based workflow',
        epilog='Example: python generate_events_with_particles.py --root-file test.root --config SK_geom_config.json --output events/'
    )
    parser.add_argument(
        '--root-file',
        type=str,
        required=True,
        help='Path to PhotonSim ROOT file with particle-based data'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for generated events'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=base_dir_path()+'config/SK_geom_config.json',
        help='Path to detector geometry configuration (default: '+base_dir_path()+'config/SK_geom_config.json)'
    )
    parser.add_argument(
        '--n-events',
        type=int,
        default=None,
        help='Number of events to generate (default: all entries in ROOT file)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for parallel processing (default: 100)'
    )
    parser.add_argument(
        '--merged-filename',
        type=str,
        default='merged_events.h5',
        help='Name of the merged output file (default: merged_events.h5)'
    )
    parser.add_argument(
        '--master-seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: random based on time)'
    )
    parser.add_argument(
        '--apply-smearing',
        action='store_true',
        help='Apply smearing to Q_true and T_true to get Q_reco and T_reco'
    )
    parser.add_argument(
        '--apply-rotation',
        action='store_true',
        help='Apply random rotation per primary to all photons and tracks'
    )
    parser.add_argument(
        '--apply-translation',
        action='store_true',
        help='Apply random translation per event to all photons and tracks'
    )
    parser.add_argument(
        '--include-track-segments',
        action='store_true',
        help='Include meaningful track and segment data in output (track-level G4 information)'
    )
    parser.add_argument(
        '--include-voxels',
        action='store_true',
        help='Include voxelized photon position data in output'
    )
    args = parser.parse_args()

    # Verify ROOT file exists
    if not os.path.exists(args.root_file):
        print(f"Error: ROOT file not found: {args.root_file}")
        sys.exit(1)

    # Verify config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Setup event simulator
    print(f"\nSetting up event simulator")
    print(f"Using configuration file: {args.config}")
    simulate_event = setup_event_simulator(
        args.config,
        0,  # The number of photons is irrelevant in data mode as it is decided based on the input file.
        K=5,
        is_data=True,
        temperature=0.0,
        apply_smearing=False  # Do NOT smear per-particle; smearing is applied after summing PE_per_particle
    )

    # Define sensor parameters (same as generate_events.py)
    sensor_params = DetectorParams(
        scatter_length=jnp.array(50.0),
        wall_reflection_rate=jnp.array(0.2),
        sensor_reflection_rate=jnp.array(0.0),
        absorption_length=jnp.array(50.0),
        qe=jnp.array(1.0),
        qe_corrections=jnp.array(1.0),
    )

    # Generate events
    print(f"\nGenerating events using particle-based workflow:")
    print(f"  ROOT file: {args.root_file}")
    print(f"  Output directory: {args.output}")
    print(f"  Detector config: {args.config}")
    if args.n_events:
        print(f"  Number of events: {args.n_events}")
    if args.master_seed:
        print(f"  Master seed: {args.master_seed}")
    print(f"  Apply smearing: {args.apply_smearing}")
    print(f"  Apply rotation: {args.apply_rotation}")
    print(f"  Apply translation: {args.apply_translation}")
    print(f"  Include track segments: {args.include_track_segments}")
    print(f"  Include voxels: {args.include_voxels}")

    result = generate_events_from_photonsim_particles(
        event_simulator=simulate_event,
        root_file_path=args.root_file,
        sensor_params=sensor_params,
        output_dir=args.output,
        n_events=args.n_events,
        batch_size=args.batch_size,
        master_seed=args.master_seed,
        apply_smearing=args.apply_smearing,
        apply_rotation=args.apply_rotation,
        apply_translation=args.apply_translation,
        detector_config_path=args.config,
        merge_output=True,
        merged_filename=args.merged_filename,
        include_track_segments=args.include_track_segments,
        include_voxels=args.include_voxels
    )

    print(f"\nOutput file: {result}")


if __name__ == '__main__':
    main()
