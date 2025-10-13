#!/usr/bin/env python3
"""
Production script to generate events from PhotonSim ROOT files.
"""

import argparse
import sys
import os
import random

# Add parent directory to path for imports
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_path)
sys.path.append(main_path+'/../')

import jax
import jax.numpy as jnp

from generate import generate_events_from_photonsim
from simulation import setup_event_simulator

from utils import base_dir_path

def main():
    parser = argparse.ArgumentParser(
        description='Generate events from PhotonSim ROOT file'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input ROOT file'
    )
    parser.add_argument(
        '--particle-type',
        type=str,
        required=True,
        help='Particle type (e.g., mu-, mu+, e-, e+, pi-, pi+, pi0)'
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
        help='Path to detector geometry configuration (default: '+base_dir_path()+'config/SK_geom_config.json'
    )
    parser.add_argument(
        '--n-events',
        type=int,
        default=None,
        help='Number of events to process (default: all events in file)'
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
    args = parser.parse_args()


    # Setup event simulator
    print(f"Setting up event simulator for particle type: {args.particle_type}")
    print(f"Using configuration file: {args.config}")
    simulate_event = setup_event_simulator(
        args.config,
        0, # The number of photons is irrelevant in data mode as it is decided based on the input file.
        K=5,
        is_data=True,
        temperature=0.0
    )

    # Define sensor parameters
    sensor_params = (
        jnp.array(50.0),      # scatter_length
        jnp.array(0.2),       # reflection_rate
        jnp.array(50.0),   # absorption_length
        jnp.array(0.001)       # this parameter is depreceated
    )

    # Generate events
    print(f"\nGenerating events from: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Particle type: {args.particle_type}")

    result = generate_events_from_photonsim(
        event_simulator=simulate_event,
        root_file_path=args.input,
        sensor_params=sensor_params,
        output_dir=args.output,
        n_events=args.n_events,
        batch_size=args.batch_size,
        particle_type=args.particle_type,
        merge_output=True,
        merged_filename=args.merged_filename
    )

    print(f"Output file: {result}")


if __name__ == '__main__':
    main()
