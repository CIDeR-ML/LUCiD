#!/usr/bin/env python3
"""
Single Ring Track Optimization Script (Adam Optimizer)

Five-stage optimization pipeline:
1. Energy estimation from photon count
2. Hierarchical grid search for position + t0
3. Hierarchical cone-based direction search
4. Energy scan optimization
5. Adam optimization (position + direction + t0 + energy)

Usage:
    python -m lucid.optimization.run <config_file_path>

Example:
    python -m lucid.optimization.run ../../config/single_track_optimization_config.json
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import json
import time

import jax
import jax.numpy as jnp
import math
import numpy as np
from functools import partial
import pickle
from tqdm import tqdm
from jax import grad, jit, vmap, value_and_grad
import uproot
from jax import jit
import optax  # JAX optimization library

from lucid.geometry import generate_detector
from lucid.utils import load_single_event, save_single_event, generate_random_params, print_particle_params
from lucid.utils import load_range_params, check_track_endpoint_in_detector
from lucid.simulation import setup_event_simulator
from lucid.sources.event_io import read_photon_data_from_photonsim

from lucid.optimization.utils.functions import estimate_muon_energy_from_photon_count, cone_points
from lucid.optimization.utils.functions import hierarchical_direction_search_cone, energy_scan_optimization
from lucid.optimization.utils.functions import cartesian_to_spherical, spherical_to_cartesian
from lucid.optimization.utils.functions import performance_summary

from lucid.optimization.grid_search import load_optimization_config, print_optimization_parameters
from lucid.optimization.grid_search import get_detector_bounds, hierarchical_position_grid_search
from lucid.detector_params import ParticleParams

from lucid.optimization.pipeline import (
    create_combined_loss_function,
    run_complete_optimization_adam,
    generate_event_data,
)

from lucid.losses import (
    energy_loss,
    counts_loss,
    origin_time_loss,
    cone_time_loss,
    smooth_pinball,
    first_arrival_nll,
    poisson_nll,
    get_optimal_tau_vtx,
    TAU_VTX_PARAM_A,
    TAU_VTX_PARAM_B,
    TAU_VTX_PARAM_C,
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Single Ring Track Optimization with Adam Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python single_track_optimization.py ../../config/single_track_optimization_config.json
        """
    )
    parser.add_argument(
        'config_file',
        type=str,
        help='Path to optimization configuration JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results (default: project_root/output)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Name for the output file (default: timestamp-based name)'
    )
    return parser.parse_args()


def load_config(config_path):
    """Load and validate configuration file"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_optimization_config(str(config_path))

    # Add default values for new parameters if not present
    if 'optimization_params' not in config:
        config['optimization_params'] = {}

    optimization_params = config['optimization_params']
    optimization_params.setdefault('damping_factor', 0.998)

    # Add Adam optimizer parameters if not present
    if 'adam_optimizer' not in config:
        config['adam_optimizer'] = {}

    adam_params = config['adam_optimizer']
    adam_params.setdefault('learning_rate', 0.2)
    adam_params.setdefault('b1', 0.9)
    adam_params.setdefault('b2', 0.999)
    adam_params.setdefault('eps', 1e-8)

    # Add fixed_vertex config if not present (default: disabled)
    if 'fixed_vertex' not in config:
        config['fixed_vertex'] = {'enabled': False, 'position': [0.0, 0.0, 0.0]}

    return config


def main():
    """Main execution function"""

    # Parse arguments
    args = parse_arguments()

    # Load configuration
    print(f"Loading configuration from: {args.config_file}")
    config = load_config(args.config_file)

    # Extract basic configuration
    default_json_filename = config['basic_config']['default_json_filename']
    data_dir = config['basic_config']['data_dir']
    TEMPERATURE = config['basic_config']['temperature']
    N_EVENTS = config['basic_config']['n_events']
    K = config['basic_config']['k']
    Nphot = config['basic_config']['nphot']
    C_MEDIUM = config['basic_config']['c_medium']

    # Extract optimization parameters
    VERTEX_WEIGHT_SCALE = config['optimization_weights']['vertex_weight_scale']
    COUNTS_WEIGHT_SCALE = config['optimization_weights']['counts_weight_scale']

    qe = config['detector_params']['qe']

    # Extract optimization parameters
    damping_factor = config['optimization_params']['damping_factor']

    # Extract Adam parameters
    adam_lr = config['adam_optimizer']['learning_rate']
    adam_b1 = config['adam_optimizer']['b1']
    adam_b2 = config['adam_optimizer']['b2']
    adam_eps = config['adam_optimizer']['eps']

    MAX_ITERATIONS = config['gradient_descent']['max_iterations']
    verbosity = config.get('verbosity', {}).get('level', 2)
    store_true_data = config.get('storage', {}).get('store_true_data', True)

    # Setup detector
    print(f"Setting up detector from: {default_json_filename}")
    detector = generate_detector(default_json_filename)
    detector_points = jnp.array(detector.all_points)
    detector_radius = detector.S_radius
    NUM_DETECTORS = len(detector_points)

    # Load range parametrization for track endpoint validation
    # TODO: Extract particle and medium from config instead of hardcoding
    range_params = load_range_params('muon', 'water')
    print(f"Loaded range parametrization: {range_params['description']}")

    # Get detector bounds
    detector_bounds = get_detector_bounds(detector)
    DETECTOR_R = detector_bounds['r'] if detector_bounds['type'] == 'cylinder' else None
    DETECTOR_H = detector_bounds['H'] if detector_bounds['type'] == 'cylinder' else None

    # Setup simulators
    PHYSICS_CONFIG = config['basic_config'].get('physics_config', None)
    print("Setting up simulators...")
    prediction_simulator = setup_event_simulator(
        default_json_filename, Nphot, TEMPERATURE, max_sensors_per_cell=4, K=K, is_data=False,
        physics_config=PHYSICS_CONFIG, default_detector_params=True
    )

    data_simulator = setup_event_simulator(
        default_json_filename, Nphot, temperature=0.0, K=12, is_data=True, is_calibration=False,
        physics_config=PHYSICS_CONFIG, default_detector_params=True
    )

    # Print parameters
    print_optimization_parameters(config, detector_bounds['r'], detector_bounds.get('H', 0), NUM_DETECTORS)

    # Print optimization parameters
    print("\n" + "=" * 80)
    print("Additional Optimization Parameters:")
    print("=" * 80)
    print(f"Quantum efficiency:      {qe}")
    print(f"Damping factor:          {damping_factor}")

    # Print Adam parameters
    print("\n" + "=" * 80)
    print("Adam Optimizer Parameters:")
    print("=" * 80)
    print(f"Learning rate:           {adam_lr}")
    print(f"Beta1:                   {adam_b1}")
    print(f"Beta2:                   {adam_b2}")
    print(f"Epsilon:                 {adam_eps}")

    # Check data directory
    print(f"\nData directory: {data_dir}")
    import glob
    root_files = glob.glob(os.path.join(data_dir, "*.root"))
    print(f"Found {len(root_files)} .root files in directory")

    # Create combined gradient function
    combined_grad_fn = create_combined_loss_function(
        VERTEX_WEIGHT_SCALE, COUNTS_WEIGHT_SCALE,
        prediction_simulator, C_MEDIUM, nrays=Nphot, num_detectors=NUM_DETECTORS
    )

    # Warm-up: Pre-compile JIT functions
    print("\nPre-compiling JIT functions...")
    main_key = jax.random.PRNGKey(42)
    warmup_key = jax.random.split(main_key, 1)[0]

    # Generate warmup event with endpoint validation
    warmup_attempts = 0
    warmup_endpoint_valid = False
    while not warmup_endpoint_valid and warmup_attempts < 10:
        warmup_attempts += 1
        warmup_event_data = generate_event_data(
            0, warmup_key, data_dir, data_simulator, detector_bounds, fraction=0.9,
            fixed_vertex_config=config.get('fixed_vertex')
        )
        warmup_endpoint_valid = check_track_endpoint_in_detector(
            warmup_event_data['true_position'],
            warmup_event_data['true_direction'],
            warmup_event_data['true_energy'],
            range_params, detector_bounds, fraction=0.9
        )
        if not warmup_endpoint_valid:
            warmup_key, _ = jax.random.split(warmup_key)

    if not warmup_endpoint_valid:
        print("ERROR: Could not generate valid warmup event. Exiting.")
        sys.exit(1)

    hit_mask = warmup_event_data['hit_counts'] > -999
    hit_detector_positions = detector_points[hit_mask]
    observed_times = warmup_event_data['hit_times'][hit_mask]
    observed_counts = warmup_event_data['hit_counts'][hit_mask]

    _ = run_complete_optimization_adam(
        initial_t0=0.0,
        hit_detector_positions=hit_detector_positions,
        observed_times=observed_times,
        observed_counts=observed_counts,
        true_data=warmup_event_data['true_data'],
        true_energy=warmup_event_data['true_energy'],
        true_position=warmup_event_data['true_position'],
        true_direction=warmup_event_data['true_direction'],
        TRUE_T0=warmup_event_data['TRUE_T0'],
        config=config,
        detector_bounds=detector_bounds,
        prediction_simulator=prediction_simulator,
        combined_grad_fn=combined_grad_fn,
        qe=qe
    )
    print("JIT compilation complete.\n")

    # Process events
    print(f"Starting optimization for {N_EVENTS} events...")
    if verbosity >= 2:
        print(f"Storage mode: store_true_data = {store_true_data}")
        print("=" * 80)

    # Storage for results
    all_event_results = []

    # Performance tracking
    energy_guess_errors = []
    grid_position_errors = []
    grid_t0_errors = []
    cone_direction_errors = []
    energy_scan_improvements = []
    final_position_errors = []
    final_direction_errors = []
    final_t0_errors = []
    final_energy_errors = []
    final_position_differences = []  # 3D: [dx, dy, dz]
    final_direction_differences = []  # 2D: [dtheta, dphi]
    final_t0_differences = []
    final_energy_differences = []
    final_combined_losses = []
    final_vertex_losses = []
    final_counts_losses = []
    final_energy_losses = []
    final_direction_losses = []
    convergence_rates = []

    # Generate random keys
    event_keys = jax.random.split(main_key, N_EVENTS)

    # Process each event
    progress_bar = tqdm(range(N_EVENTS), desc="Processing events") if verbosity == 0 else range(N_EVENTS)
    for event_idx in progress_bar:
        event_start_time = time.time()
        try:
            if verbosity >= 2:
                print(f"\n--- Processing Event {event_idx} ---")

            # Generate event data with endpoint validation
            # Try up to 10 times to generate an event with endpoint within detector
            event_generation_attempts = 0
            max_attempts = 10
            event_key = event_keys[event_idx]
            endpoint_valid = False

            while not endpoint_valid and event_generation_attempts < max_attempts:
                event_generation_attempts += 1

                # Generate event data
                event_data = generate_event_data(
                    event_idx, event_key, data_dir,
                    data_simulator, detector_bounds, fraction=0.9,
                    fixed_vertex_config=config.get('fixed_vertex')
                )

                # Extract event parameters
                true_position = event_data['true_position']
                true_direction = event_data['true_direction']
                true_energy = event_data['true_energy']

                # Check if track endpoint is within detector bounds
                endpoint_valid = check_track_endpoint_in_detector(
                    true_position, true_direction, true_energy,
                    range_params, detector_bounds, fraction=0.9
                )

                if not endpoint_valid:
                    if event_generation_attempts == 5:
                        print(f"  WARNING: Event {event_idx} - 5 failed attempts to generate event with endpoint in detector")

                    # Generate new random key for next attempt
                    event_key, _ = jax.random.split(event_key)

            # Check if we exceeded max attempts
            if not endpoint_valid:
                print(f"  ERROR: Event {event_idx} - Failed to generate event with endpoint in detector after {max_attempts} attempts")
                print(f"  Last attempt: position={true_position}, energy={true_energy:.1f} MeV")
                print(f"  Exiting program.")
                sys.exit(1)

            if verbosity >= 2 and event_generation_attempts > 1:
                print(f"  Generated valid event after {event_generation_attempts} attempts")

            # Continue with already extracted parameters
            TRUE_T0 = event_data['TRUE_T0']
            true_data = event_data['true_data']

            # Get hit information
            hit_mask = event_data['hit_counts'] > -999
            hit_detector_positions = detector_points[hit_mask]
            observed_times = event_data['hit_times'][hit_mask]
            observed_counts = event_data['hit_counts'][hit_mask]

            # Convert true direction to spherical coordinates
            true_theta, true_phi = cartesian_to_spherical(true_direction)

            initial_t0 = 0.0

            if verbosity >= 2:
                print(f"  True position: {true_position}")
                print(f"  True direction: {true_direction}")
                print(f"  True energy: {true_energy:.1f}")
                print(f"  True t0: {TRUE_T0:.3f}")
                print(f"  Initial t0 guess: {initial_t0:.3f}")

            # Run optimization
            if verbosity >= 2:
                print("  Running Adam optimization...")
            results = run_complete_optimization_adam(
                initial_t0=initial_t0,
                hit_detector_positions=hit_detector_positions,
                observed_times=observed_times,
                observed_counts=observed_counts,
                true_data=true_data,
                true_energy=true_energy,
                true_position=true_position,
                true_direction=true_direction,
                TRUE_T0=TRUE_T0,
                config=config,
                detector_bounds=detector_bounds,
                prediction_simulator=prediction_simulator,
                combined_grad_fn=combined_grad_fn,
                qe=qe
            )

            if results is None:
                if verbosity >= 1:
                    print(f"  ERROR: Optimization failed for event {event_idx}")
                continue

            # Calculate improvements
            energy_guess_error = results['energy_estimation']['energy_guess_error']
            grid_position_error = results['grid_position_search']['position_error']
            grid_t0_error = results['grid_position_search']['t0_error']

            cone_direction = results['cone_direction_search']['best_direction']
            cone_cos_angle = np.clip(np.dot(cone_direction, true_direction), -1.0, 1.0)
            cone_direction_error = np.degrees(np.arccos(cone_cos_angle))

            energy_scan_improvement = results['energy_scan_search']['energy_improvement']

            # Create event data for storage
            event_data_to_store = {
                'event_idx': event_data['event_idx'],
                'entry_idx': event_data['entry_idx'],
                'true_energy': event_data['true_energy'],
                'true_position': event_data['true_position'],
                'true_direction': event_data['true_direction'],
                'TRUE_T0': event_data['TRUE_T0'],
                'hit_detector_positions': hit_detector_positions,
                'observed_times': observed_times,
                'observed_counts': observed_counts,
                'n_hits': int(jnp.sum(hit_mask))
            }

            if store_true_data:
                event_data_to_store['true_data'] = event_data['true_data']

            # Calculate event timing
            event_end_time = time.time()
            total_event_time = event_end_time - event_start_time

            # Store results
            event_result = {
                'event_data': event_data_to_store,
                'initial_t0': initial_t0,
                'true_theta': float(true_theta),
                'true_phi': float(true_phi),
                'energy_guess_error': energy_guess_error,
                'grid_position_error': grid_position_error,
                'grid_t0_error': grid_t0_error,
                'cone_direction_error': cone_direction_error,
                'energy_scan_improvement': energy_scan_improvement,
                'total_event_time': total_event_time,
                'adam_optimization_time': results['adam_optimization_time'],
                'optimization_results': results
            }
            all_event_results.append(event_result)

            # Track metrics
            energy_guess_errors.append(energy_guess_error)
            grid_position_errors.append(grid_position_error)
            grid_t0_errors.append(grid_t0_error)
            cone_direction_errors.append(cone_direction_error)
            energy_scan_improvements.append(energy_scan_improvement)
            final_position_errors.append(results['final_position_error'])
            final_direction_errors.append(results['final_direction_error'])
            final_t0_errors.append(results['final_t0_error'])
            final_energy_errors.append(results['final_energy_error'])
            final_position_differences.append(results['final_position_difference'])
            final_direction_differences.append(results['final_direction_difference'])
            final_t0_differences.append(results['final_t0_difference'])
            final_energy_differences.append(results['final_energy_difference'])
            final_combined_losses.append(results['final_combined_loss'])
            final_vertex_losses.append(results['final_vertex_loss'])
            final_counts_losses.append(results['final_counts_loss'])
            final_energy_losses.append(results['final_energy_loss'])
            final_direction_losses.append(results['final_direction_loss'])
            convergence_rates.append(1.0 if results['converged'] else 0.0)

            # Print results
            if verbosity == 1:
                print(f"Event {event_idx}: pos_err={results['final_position_error']*100:.1f}cm, "
                      f"dir_err={results['final_direction_error']:.2f}\u00b0, t0_err={results['final_t0_error']:.3f}, "
                      f"E_err={results['final_energy_error']:.1f}")
            elif verbosity >= 2:
                print(f"  Energy guess error: {energy_guess_error:.1f}")
                print(f"  Grid search - Position error: {grid_position_error*100:.1f}cm, t0 error: {grid_t0_error:.3f}")
                print(f"  Cone direction error: {cone_direction_error:.2f}\u00b0")
                print(f"  Energy scan improvement: {energy_scan_improvement:.1f}")
                print(f"  Final errors - Position: {results['final_position_error']:.3f}m, "
                      f"Direction: {results['final_direction_error']:.2f}\u00b0, t0: {results['final_t0_error']:.3f}, "
                      f"Energy: {results['final_energy_error']:.1f}")
                print(f"  Converged: {results['converged']}, Iterations: {results['total_iterations']}")

        except Exception as e:
            if verbosity >= 1:
                print(f"Error processing event {event_idx}: {e}")
            if verbosity >= 2:
                import traceback
                traceback.print_exc()
            continue

    if verbosity >= 1:
        print(f"\nCompleted processing {len(all_event_results)} events successfully")

    # Performance summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION PERFORMANCE SUMMARY")
    print("=" * 80)

    performance_summary(
        energy_guess_errors,
        grid_position_errors,
        cone_direction_errors,
        energy_scan_improvements,
        final_position_errors,
        final_direction_errors,
        final_t0_errors,
        final_energy_errors,
        final_combined_losses,
        final_vertex_losses,
        final_counts_losses,
        final_energy_losses,
        convergence_rates,
    )

    # Print t0 grid search statistics
    print("\n" + "=" * 80)
    print("4D Grid Search t0 Performance:")
    print("=" * 80)
    grid_t0_errors_arr = np.array(grid_t0_errors)
    print(f"Mean t0 error:   {np.mean(grid_t0_errors_arr):.4f}")
    print(f"Median t0 error: {np.median(grid_t0_errors_arr):.4f}")
    print(f"Std t0 error:    {np.std(grid_t0_errors_arr):.4f}")
    print(f"Min t0 error:    {np.min(grid_t0_errors_arr):.4f}")
    print(f"Max t0 error:    {np.max(grid_t0_errors_arr):.4f}")

    # Timing summary
    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    total_event_times = [event['total_event_time'] for event in all_event_results]
    adam_times = [event['adam_optimization_time'] for event in all_event_results]

    print(f"Total Event Processing Time:")
    print(f"  Mean:   {np.mean(total_event_times):.2f}s")
    print(f"  Median: {np.median(total_event_times):.2f}s")
    print(f"  Min:    {np.min(total_event_times):.2f}s")
    print(f"  Max:    {np.max(total_event_times):.2f}s")
    print(f"  Total:  {np.sum(total_event_times):.2f}s")

    print(f"\nAdam Optimization Time:")
    print(f"  Mean:   {np.mean(adam_times):.2f}s")
    print(f"  Median: {np.median(adam_times):.2f}s")
    print(f"  Min:    {np.min(adam_times):.2f}s")
    print(f"  Max:    {np.max(adam_times):.2f}s")
    print(f"  Total:  {np.sum(adam_times):.2f}s")

    print(f"\nAdam optimization as % of total event time: {100 * np.sum(adam_times) / np.sum(total_event_times):.1f}%")

    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert arrays
    energy_guess_errors = np.array(energy_guess_errors)
    grid_position_errors = np.array(grid_position_errors)
    grid_t0_errors = np.array(grid_t0_errors)
    cone_direction_errors = np.array(cone_direction_errors)
    energy_scan_improvements = np.array(energy_scan_improvements)
    final_position_errors = np.array(final_position_errors)
    final_direction_errors = np.array(final_direction_errors)
    final_t0_errors = np.array(final_t0_errors)
    final_energy_errors = np.array(final_energy_errors)
    final_position_differences = np.array(final_position_differences)  # Shape: (N_events, 3)
    final_direction_differences = np.array(final_direction_differences)  # Shape: (N_events, 2)
    final_t0_differences = np.array(final_t0_differences)
    final_energy_differences = np.array(final_energy_differences)
    final_combined_losses = np.array(final_combined_losses)
    final_vertex_losses = np.array(final_vertex_losses)
    final_counts_losses = np.array(final_counts_losses)
    final_energy_losses = np.array(final_energy_losses)
    final_direction_losses = np.array(final_direction_losses)
    convergence_rates = np.array(convergence_rates)

    # Prepare results summary with config
    results_summary = {
        'timestamp': timestamp,
        'config_file': args.config_file,
        'config': {
            'optimizer': 'Adam',
            'nphot': Nphot,
            'adam_learning_rate': adam_lr,
            'adam_b1': adam_b1,
            'adam_b2': adam_b2,
            'adam_eps': adam_eps,
            'n_events': N_EVENTS,
            'temperature': TEMPERATURE,
            'vertex_weight_scale': VERTEX_WEIGHT_SCALE,
            'counts_weight_scale': COUNTS_WEIGHT_SCALE,
            'pos_n_div': config['position_grid_search']['pos_n_div'],
            'pos_levels': config['position_grid_search']['pos_levels'],
            'pos_fraction': config['position_grid_search']['pos_fraction'],
            'pos_min_L': config['position_grid_search']['pos_min_L'],
            't0_n_div': config['position_grid_search']['t0_n_div'],
            't0_min': config['position_grid_search']['t0_min'],
            't0_max': config['position_grid_search']['t0_max'],
            'cone_levels': config['cone_direction_search']['cone_levels'],
            'cone_initial_div': config['cone_direction_search']['cone_initial_div'],
            'cone_max_angle_deg': config['cone_direction_search']['cone_max_angle_deg'],
            'cone_reduction': config['cone_direction_search']['cone_reduction'],
            'energy_delta': config['energy_optimization']['energy_delta'],
            'energy_scan_steps': config['energy_optimization']['energy_scan_steps'],
            'qe': qe,
            'damping_factor': damping_factor,
            'detector_r': float(DETECTOR_R) if DETECTOR_R is not None else None,
            'detector_h': float(DETECTOR_H) if DETECTOR_H is not None else None,
            'detector_bounds': detector_bounds,
            'data_dir': data_dir,
            'detector_file': default_json_filename
        },
        'raw_data': {
            'energy_guess_errors': energy_guess_errors.tolist(),
            'grid_position_errors': grid_position_errors.tolist(),
            'grid_t0_errors': grid_t0_errors.tolist(),
            'cone_direction_errors': cone_direction_errors.tolist(),
            'energy_scan_improvements': energy_scan_improvements.tolist(),
            'final_position_errors': final_position_errors.tolist(),
            'final_direction_errors': final_direction_errors.tolist(),
            'final_t0_errors': final_t0_errors.tolist(),
            'final_energy_errors': final_energy_errors.tolist(),
            'final_position_differences': final_position_differences.tolist(),  # (N_events, 3)
            'final_direction_differences': final_direction_differences.tolist(),  # (N_events, 2)
            'final_t0_differences': final_t0_differences.tolist(),
            'final_energy_differences': final_energy_differences.tolist(),
            'final_combined_losses': final_combined_losses.tolist(),
            'final_vertex_losses': final_vertex_losses.tolist(),
            'final_counts_losses': final_counts_losses.tolist(),
            'final_energy_losses': final_energy_losses.tolist(),
            'final_direction_losses': final_direction_losses.tolist(),
            'convergence_rates': convergence_rates.tolist()
        },
        'all_event_results': all_event_results
    }

    # Save results
    if args.name:
        output_file = output_dir / f'{args.name}.pkl'
    else:
        output_file = output_dir / f'single_track_optimization_adam_{timestamp}_{N_EVENTS}_events.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results_summary, f)

    print(f"\n{'=' * 80}")
    print("Results saved:")
    print(f"  {output_file}")
    print(f"{'=' * 80}")

    print("\nOptimization complete!")


if __name__ == '__main__':
    main()
