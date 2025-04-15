#!/usr/bin/env python3

import argparse
from tools.propagate import create_photon_propagator
from tools.geometry import generate_detector
from tools.utils import load_single_event, save_single_event, print_params, generate_random_params
from tools.losses import compute_loss
from tools.simulation import setup_event_simulator

import jax
import jax.numpy as jnp
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Simulate photon events in a detector')
    parser.add_argument('--config', type=str, default='config/cyl_geom_config.json',
                        help='Path to the detector configuration JSON file')
    parser.add_argument('--output', type=str, default='events/test_event_data.h5',
                        help='Output filename for the simulated event data')
    parser.add_argument('--nphot', type=int, default=1_000_000,
                        help='Number of photons to simulate')
    parser.add_argument('--temp', type=float, default=100.0,
                        help='Temperature parameter')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--random-params', action='store_true',
                        help='Use random parameters instead of default ones')
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup detector
    detector = generate_detector(args.config)
    detector_points = jnp.array(detector.all_points)
    detector_radius = detector.S_radius
    NUM_DETECTORS = len(detector_points)

    # Setup simulation
    simulate_event = setup_event_simulator(args.config, args.nphot, args.temp)

    # Generate key for random number generation
    key = jax.random.PRNGKey(args.seed)

    if args.random_params:
        current_time = int(time.time() * 1e9)

        # Create a PRNGKey using the current time
        key = jax.random.PRNGKey(current_time)
        true_params = generate_random_params(key)
    else:
        # Default parameters: cone on the wall
        true_params = (
            jnp.array(30.0),  # opening angle
            jnp.array([0.5, 0.0, -0.5]),  # position
            jnp.array([1.0, -1.0, 0.2]),  # direction
            jnp.array(5000.0)  # intensity
        )

    # Run simulation
    single_event_data = simulate_event(true_params, key)

    # Save results
    save_single_event(single_event_data, true_params, filename=args.output)
    print(f"Event data saved to: {args.output}")


if __name__ == "__main__":
    main()