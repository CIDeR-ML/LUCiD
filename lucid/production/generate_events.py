#!/usr/bin/env python3
"""
Production script to generate events from PhotonSim ROOT files.
Supports multi-particle events where all particles share a common vertex.

Usage:
    python generate_events.py --particle mu-:muon.root --particle pi-:pion.root --output events/

Each event will contain all specified particle types with:
    - Shared vertex position
    - Independent track directions
    - Random sampling from respective ROOT files
"""

import argparse
import sys
import os
import random

import jax
import jax.numpy as jnp

from lucid.sources.event_io import generate_events_from_photonsim
from lucid.simulation import setup_event_simulator
from lucid.detector_params import DetectorParams, load_physics_config

from lucid.utils import base_dir_path

def main():
    parser = argparse.ArgumentParser(
        description='Generate events from PhotonSim ROOT files with multiple particle types per event',
        epilog='Example: python generate_events.py --particle mu-:muon.root --particle pi-:pion.root --output events/'
    )
    parser.add_argument(
        '--particle',
        type=str,
        action='append',
        required=True,
        help='Particle specification in format "type:root_file_path" (can be specified multiple times). '
             'Example: --particle mu-:muon.root --particle pi-:pion.root'
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
        default=base_dir_path()+'config/SK_like_geom_config.json',
        help='Path to detector geometry configuration (default: '+base_dir_path()+'config/SK_like_geom_config.json'
    )
    parser.add_argument(
        '--n-events',
        type=int,
        default=None,
        help='Number of events to generate (default: minimum number of entries across all ROOT files)'
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
        '--physics-config',
        type=str,
        default=base_dir_path()+'config/SK_like_physics_config.json',
        help='Path to physics configuration (medium model, QE curve, detector params). '
             'Default: '+base_dir_path()+'config/SK_like_physics_config.json'
    )
    args = parser.parse_args()

    # Parse particle specifications into dictionary
    particles_dict = {}
    print("\nParsing particle specifications:")
    for particle_spec in args.particle:
        try:
            particle_type, root_file_path = particle_spec.split(':', 1)
            if particle_type in particles_dict:
                print(f"Warning: Duplicate particle type '{particle_type}', using latest specification")
            particles_dict[particle_type] = root_file_path
            print(f"  - {particle_type}: {root_file_path}")
        except ValueError:
            print(f"Error: Invalid particle specification '{particle_spec}'")
            print(f"Expected format: 'particle_type:root_file_path'")
            sys.exit(1)

    if not particles_dict:
        print("Error: No valid particle specifications provided")
        sys.exit(1)

    # Load detector parameters from physics config
    print(f"\nLoading physics configuration: {args.physics_config}")
    sensor_params, _, _ = load_physics_config(args.physics_config)
    print(f"  Wall reflection rate: {float(sensor_params.wall_reflection_rate):.3f}")
    print(f"  Sensor reflection rate: {float(sensor_params.sensor_reflection_rate):.3f}")
    print(f"  QE (scalar): {float(sensor_params.qe):.3f}")

    # Setup event simulator with physics config for wavelength-dependent QE and medium model
    print(f"\nSetting up event simulator")
    print(f"Using geometry configuration: {args.config}")
    simulate_event = setup_event_simulator(
        args.config,
        0, # The number of photons is irrelevant in data mode as it is decided based on the input file.
        K=5,
        is_data=True,
        temperature=0.0,
        physics_config=args.physics_config,
    )

    # Generate events
    print(f"\nGenerating events:")
    print(f"  Output directory: {args.output}")
    print(f"  Particles per event: {len(particles_dict)}")
    print(f"  Particle types: {', '.join(particles_dict.keys())}")
    if args.n_events:
        print(f"  Number of events: {args.n_events}")
    if args.master_seed:
        print(f"  Master seed: {args.master_seed}")

    result = generate_events_from_photonsim(
        event_simulator=simulate_event,
        particles_dict=particles_dict,
        sensor_params=sensor_params,
        output_dir=args.output,
        n_events=args.n_events,
        batch_size=args.batch_size,
        master_seed=args.master_seed,
        merge_output=True,
        merged_filename=args.merged_filename
    )

    print(f"\nOutput file: {result}")


if __name__ == '__main__':
    main()
