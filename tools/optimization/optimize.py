#!/usr/bin/env python3
"""
LUCiD Parameter Optimization - Master Script

This script provides a unified interface for all optimization methods in LUCiD:
- Numerical optimization (adaptive search)
- Gradient-based optimization (autodiff)
- Hybrid optimization (numerical + gradient)

Usage examples:
    # Numerical optimization (like old adaptive_search.py)
    python -m tools.optimization.optimize --mode numerical -i 40 -p 30 -n 10 -q
    
    # Gradient-based optimization
    python -m tools.optimization.optimize --mode gradient --n-gradient 100 --energy-lr 0.5
    
    # Hybrid optimization
    python -m tools.optimization.optimize --mode hybrid -i 20 --n-gradient 50
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
import argparse
import time
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.utils import base_dir_path, generate_random_params, spherical_to_cartesian
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.optimization.algorithms import adaptive_search, hybrid_optimization
from tools.optimization.utils import (
    create_event_visualization, print_summary_statistics,
    create_convergence_plots, create_summary_plots,
    create_hybrid_convergence_plot
)


def get_detector_bounds(detector):
    """Extract detector bounds based on detector type."""
    detector_type = detector.__class__.__name__
    
    if 'Cylinder' in detector_type:
        return {
            'type': 'cylinder',
            'r': detector.r,
            'H': detector.H
        }
    elif 'Sphere' in detector_type:
        return {
            'type': 'sphere',
            'r': detector.r
        }
    elif 'Box' in detector_type:
        return {
            'type': 'box',
            'x': detector.L,
            'y': detector.W,
            'z': detector.H
        }
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


def generate_random_event_params(key, detector_bounds):
    """Generate random event parameters based on detector geometry."""
    if detector_bounds['type'] == 'cylinder':
        r_vert = jax.random.uniform(key, shape=(), minval=0, maxval=detector_bounds['r'] * 0.8)
        key, _ = jax.random.split(key)
        theta = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
        key, _ = jax.random.split(key)
        z_vert = jax.random.uniform(key, shape=(), minval=-detector_bounds['H']/2 * 0.8, 
                                   maxval=detector_bounds['H']/2 * 0.8)
        position = jnp.array([r_vert * jnp.cos(theta), r_vert * jnp.sin(theta), z_vert])
        
    elif detector_bounds['type'] == 'sphere':
        u = jax.random.uniform(key, shape=())
        key, _ = jax.random.split(key)
        cos_theta = jax.random.uniform(key, shape=(), minval=-1, maxval=1)
        key, _ = jax.random.split(key)
        phi = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
        
        r = detector_bounds['r'] * 0.8 * jnp.cbrt(u)
        sin_theta = jnp.sqrt(1 - cos_theta**2)
        position = jnp.array([r * sin_theta * jnp.cos(phi), 
                             r * sin_theta * jnp.sin(phi), 
                             r * cos_theta])
        
    elif detector_bounds['type'] == 'box':
        position = jax.random.uniform(key, shape=(3,), 
                                    minval=jnp.array([-detector_bounds['x']/2, 
                                                     -detector_bounds['y']/2, 
                                                     -detector_bounds['z']/2]) * 0.8,
                                    maxval=jnp.array([detector_bounds['x']/2, 
                                                     detector_bounds['y']/2, 
                                                     detector_bounds['z']/2]) * 0.8)
    
    # Random direction
    key, _ = jax.random.split(key)
    phi = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
    key, _ = jax.random.split(key)
    cos_theta = jax.random.uniform(key, shape=(), minval=-1, maxval=1)
    sin_theta = jnp.sqrt(1 - cos_theta**2)
    direction = jnp.array([sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), cos_theta])
    
    # Random energy
    key, _ = jax.random.split(key)
    energy = jax.random.uniform(key, shape=(), minval=250.0, maxval=900.0)
    
    return position, direction, energy


def run_optimization(
    config_file=None,
    detector_type='Cylinder',
    n_events=1,
    mode='numerical',
    n_iterations=40,
    population_size=20,
    n_gradient_iterations=50,
    n_photons=1_000_000,
    K=2,
    # Gradient optimization parameters
    energy_lr=1.0,
    spatial_lr=0.1,
    energy_scale=0.01,
    position_scale=0.1,
    direction_scale=0.1,
    patience=20,
    # Adaptive scaling parameters
    auto_scale=True,
    target_energy_update_mev=10.0,
    target_position_update_fraction=0.05,
    target_direction_update_degrees=5.0,
    # Other parameters
    save_event_plots=False,
    verbose=True,
    gradient_debug=False,
    random_seed=1234567
):
    """
    Main optimization function that supports all modes.
    
    Parameters
    ----------
    config_file : str
        Path to detector configuration file
    detector_type : str
        Type of detector: 'Cylinder', 'Sphere', or 'Box'
    n_events : int
        Number of events to process
    mode : str
        Optimization mode: 'numerical', 'gradient', or 'hybrid'
    n_iterations : int
        Number of numerical iterations
    population_size : int
        Population size for numerical optimization
    n_gradient_iterations : int
        Number of gradient iterations
    n_photons : int
        Number of photons to simulate
    K : int
        Number of nearest neighbors for sensor mapping
    energy_lr : float
        Learning rate for energy parameter
    spatial_lr : float
        Learning rate for spatial parameters
    energy_scale : float
        Scale factor for energy gradients
    position_scale : float
        Scale factor for position gradients
    direction_scale : float
        Scale factor for direction gradients
    patience : int
        Patience for learning rate reduction
    save_event_plots : bool
        Whether to save individual event plots
    verbose : bool
        Whether to show detailed output
    random_seed : int
        Random seed for reproducibility
    """
    
    # Configuration
    if config_file is None:
        config_file = base_dir_path() + 'config/IWCD_geom_config.json'
    
    # Create output directory
    figures_dir = os.path.join(base_dir_path(), 'output', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Initialize random key
    key = jax.random.PRNGKey(random_seed)
    
    # Setup detector
    if verbose or n_events == 1:
        print(f"Loading detector configuration...")
        print(f"Detector type: {detector_type}")
        print(f"Optimization mode: {mode}")
        if mode in ['gradient', 'hybrid']:
            print(f"Gradient parameters: energy_lr={energy_lr}, spatial_lr={spatial_lr}")
            print(f"Gradient scales: energy={energy_scale}, position={position_scale}, direction={direction_scale}")
    
    detector = generate_detector(config_file)
    sensor_positions = jnp.array(detector.all_points)
    n_sensors = len(sensor_positions)
    
    if verbose or n_events == 1:
        print(f"Detector has {n_sensors} sensors")
    
    # Get detector bounds
    detector_bounds = get_detector_bounds(detector)
    
    # Setup simulation parameters
    sensor_params = (
        jnp.array(50.0),    # scatter_length
        jnp.array(0.1),     # reflection_rate
        jnp.array(100.0),   # absorption_length
        jnp.array(0.001)    # gumbel_softmax_temperature
    )
    
    # Setup event simulator
    if verbose or n_events == 1:
        print("Setting up event simulator...")
    
    simulate_event = setup_event_simulator(
        json_filename=config_file,
        max_sensors_per_cell=4,
        n_photons=n_photons,
        temperature=0.0,
        K=K,
        detector_type=detector_type
    )
    
    # Initialize statistics collection
    results = []
    event_histories = []
    total_search_time = 0.0
    
    # Gradient optimization kwargs
    gradient_kwargs = {
        'energy_lr': energy_lr,
        'spatial_lr': spatial_lr,
        'energy_scale': energy_scale,
        'position_scale': position_scale,
        'direction_scale': direction_scale,
        'patience': patience,
        'patience_factor': 0.5,
        'tau': 0.01,
        'lambda_time': 1.0,
        'lambda_intensity': 1.0,
        'auto_scale': auto_scale,
        'target_energy_update_mev': target_energy_update_mev,
        'target_position_update_fraction': target_position_update_fraction,
        'target_direction_update_degrees': target_direction_update_degrees,
        'gradient_debug': gradient_debug,
        'gradient_verbose': gradient_debug  # Enable gradient verbosity when debug is requested
    }
    
    if verbose:
        print(f"\nProcessing {n_events} events...")
    else:
        print(f"\nProcessing {n_events} events...")
    
    # Progress bar for multi-event processing
    event_iterator = range(n_events)
    if not verbose and n_events > 1:
        event_iterator = tqdm(event_iterator, desc="Events", unit="event")
    
    for event_idx in event_iterator:
        if verbose:
            print(f"\n{'='*80}")
            print(f"EVENT {event_idx + 1}/{n_events}")
            print(f"{'='*80}")
        
        # Generate TRUE event
        key, subkey = jax.random.split(key)
        
        if detector_type in ['Cylinder', 'Sphere', 'Box']:
            # Use detector-aware random generation
            true_position, true_direction, true_energy = generate_random_event_params(subkey, detector_bounds)
        else:
            # Use generic random generation
            true_energy, true_position, true_direction_angles = generate_random_params(subkey)
            true_direction = spherical_to_cartesian(true_direction_angles[0], true_direction_angles[1])
        
        # Convert direction to spherical angles
        true_theta = jnp.arccos(jnp.clip(true_direction[2], -1.0, 1.0))
        true_phi = jnp.arctan2(true_direction[1], true_direction[0])
        true_direction_angles = jnp.array([true_theta, true_phi])
        
        # Simulate true event
        true_particle_params = (true_energy, true_position, true_direction_angles)
        true_charges, true_times = simulate_event(true_particle_params, sensor_params, subkey)
        
        if verbose or n_events == 1:
            print(f"True event parameters:")
            print(f"  Position: {true_position}")
            print(f"  Direction: {true_direction}")
            print(f"  Energy: {true_energy:.1f} MeV")
            print(f"  Active sensors: {jnp.sum(true_charges > 0)}")
        
        # Run optimization based on mode
        if verbose or n_events == 1:
            print(f"\nStarting {mode} optimization...")
        
        search_start_time = datetime.now()
        
        # Determine optimization parameters based on mode
        if mode == 'numerical':
            optimization_type = 'numerical'
            gradient_iterations = 0
        elif mode == 'gradient':
            optimization_type = 'gradient'
            gradient_iterations = n_gradient_iterations
        elif mode == 'hybrid':
            optimization_type = 'hybrid'
            gradient_iterations = n_gradient_iterations
        else:
            raise ValueError(f"Unknown optimization mode: {mode}")
        
        # Track history for multi-event runs to enable convergence plots
        track_history = n_events > 1
        
        # Use a different seed for optimization to avoid correlation with event generation
        optimization_seed = random_seed + 10000 + event_idx * 1000
        
        search_result = adaptive_search(
            true_charges, true_times, simulate_event, sensor_params, sensor_positions,
            detector_bounds, true_position, true_direction, true_energy,
            n_iterations=n_iterations, 
            population_size=population_size, 
            elite_fraction=0.2,
            verbose=verbose and n_events == 1, 
            track_history=track_history,
            optimization_type=optimization_type,
            gradient_iterations=gradient_iterations,
            gradient_kwargs=gradient_kwargs,
            random_seed=optimization_seed
        )
        
        # Unpack results based on whether history was tracked
        if track_history:
            best_match, final_population, event_history = search_result
            event_histories.append(event_history)
        else:
            best_match, final_population = search_result
            
        search_duration = (datetime.now() - search_start_time).total_seconds()
        total_search_time += search_duration
        
        if verbose or n_events == 1:
            print(f"Optimization completed in {search_duration:.1f} seconds")
        
        # Calculate errors and store results
        if best_match:
            position_error = float(jnp.linalg.norm(best_match['position'] - true_position))
            direction_error = float(jnp.arccos(jnp.clip(jnp.abs(jnp.dot(best_match['direction'], true_direction)), 0, 1)))
            direction_error_deg = np.degrees(direction_error)
            energy_error = float(jnp.abs(best_match['energy'] - true_energy))
            energy_error_percent = (energy_error / float(true_energy)) * 100
            
            # Store results
            result = {
                'event_idx': event_idx,
                'true_position': np.array(true_position),
                'true_direction': np.array(true_direction),
                'true_energy': float(true_energy),
                'fitted_position': np.array(best_match['position']),
                'fitted_direction': np.array(best_match['direction']),
                'fitted_energy': float(best_match['energy']),
                'position_error': position_error,
                'direction_error_deg': direction_error_deg,
                'energy_error': energy_error,
                'energy_error_percent': energy_error_percent,
                'final_loss': best_match['loss'],
                'search_time': search_duration,
                'success': True,
                'optimization_mode': mode
            }
            
            if verbose or n_events == 1:
                print(f"\nEvent {event_idx + 1} Results:")
                print(f"  Position error: {position_error:.3f} m")
                print(f"  Direction error: {direction_error_deg:.1f}°")
                print(f"  Energy error: {energy_error:.1f} MeV ({energy_error_percent:.1f}%)")
                print(f"  Final loss: {best_match['loss']:.6f}")
            
            # Create visualization based on flags
            if save_event_plots:
                create_event_visualization(
                    true_position, true_direction, true_energy,
                    best_match, true_charges, true_times, 
                    sensor_positions, detector_bounds,
                    position_error, direction_error_deg, energy_error, energy_error_percent,
                    event_idx, figures_dir, verbose, config_file
                )
        else:
            result = {
                'event_idx': event_idx,
                'success': False,
                'search_time': search_duration,
                'optimization_mode': mode
            }
            print(f"Event {event_idx + 1}: Optimization failed!")
        
        results.append(result)
    
    # Print summary statistics and create plots
    if n_events > 1:
        print_summary_statistics(results, total_search_time)
        create_summary_plots(results, figures_dir, config_file=config_file)
        
        # Create convergence plots if we have event histories
        if event_histories:
            # Check if we have gradient history (hybrid optimization)
            if event_histories[0] and 'gradient_loss' in event_histories[0]:
                create_hybrid_convergence_plot(event_histories, figures_dir, config_file=config_file)
            else:
                create_convergence_plots(event_histories, figures_dir, config_file=config_file)
    
    return results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='LUCiD parameter optimization with multiple methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Numerical optimization (like old adaptive_search.py)
    python -m tools.optimization.optimize --mode numerical -i 40 -p 30 -n 10 -q
    
    # Gradient-based optimization 
    python -m tools.optimization.optimize --mode gradient --n-gradient 100 --energy-lr 0.5
    
    # Hybrid optimization
    python -m tools.optimization.optimize --mode hybrid -i 20 --n-gradient 50
    
    # Save event plots
    python -m tools.optimization.optimize --mode hybrid --save-event-plots -n 5
        """
    )
    
    # Basic parameters
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to detector configuration file')
    parser.add_argument('--detector', '-d', type=str, default='Cylinder',
                        choices=['Cylinder', 'Sphere', 'Box'],
                        help='Detector geometry type')
    parser.add_argument('--events', '-n', type=int, default=1,
                        help='Number of events to process (default: 1)')
    parser.add_argument('--mode', '-m', type=str, default='numerical',
                        choices=['numerical', 'gradient', 'hybrid'],
                        help='Optimization mode (default: numerical)')
    
    # Numerical optimization parameters
    parser.add_argument('--iterations', '-i', type=int, default=40,
                        help='Number of numerical iterations (default: 40)')
    parser.add_argument('--population', '-p', type=int, default=20,
                        help='Population size for numerical optimization (default: 20)')
    
    # Gradient optimization parameters
    parser.add_argument('--n-gradient', type=int, default=50,
                        help='Number of gradient iterations (default: 50)')
    parser.add_argument('--energy-lr', type=float, default=1.0,
                        help='Learning rate for energy (default: 1.0)')
    parser.add_argument('--spatial-lr', type=float, default=0.1,
                        help='Learning rate for spatial parameters (default: 0.1)')
    parser.add_argument('--energy-scale', type=float, default=0.01,
                        help='Scale factor for energy gradients (default: 0.01)')
    parser.add_argument('--position-scale', type=float, default=0.1,
                        help='Scale factor for position gradients (default: 0.1)')
    parser.add_argument('--direction-scale', type=float, default=0.1,
                        help='Scale factor for direction gradients (default: 0.1)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for learning rate reduction (default: 20)')
    
    # Adaptive scaling parameters
    parser.add_argument('--no-auto-scale', action='store_true',
                        help='Disable automatic gradient scale calculation')
    parser.add_argument('--target-energy-update', type=float, default=10.0,
                        help='Target first energy update in MeV (default: 10.0)')
    parser.add_argument('--target-position-update', type=float, default=0.05,
                        help='Target first position update as fraction of detector scale (default: 0.05)')
    parser.add_argument('--target-direction-update', type=float, default=5.0,
                        help='Target first direction update in degrees (default: 5.0)')
    
    # Simulation parameters
    parser.add_argument('--photons', type=int, default=1_000_000,
                        help='Number of photons to simulate (default: 1000000)')
    parser.add_argument('--K', type=int, default=2,
                        help='Number of nearest neighbors for sensor mapping (default: 2)')
    
    # Output and verbosity
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed progress during iterations')
    parser.add_argument('--gradient-debug', action='store_true',
                        help='Show gradient debugging info every iteration (instead of every 10)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress iteration details (opposite of verbose)')
    parser.add_argument('--save-event-plots', action='store_true',
                        help='Save visualization plots for individual events')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=1234567,
                        help='Random seed (default: 1234567)')
    
    args = parser.parse_args()
    
    # Determine verbosity (quiet takes precedence)
    verbose = True
    if args.quiet:
        verbose = False
    elif args.verbose:
        verbose = True
    
    # Run optimization
    results = run_optimization(
        config_file=args.config,
        detector_type=args.detector,
        n_events=args.events,
        mode=args.mode,
        n_iterations=args.iterations,
        population_size=args.population,
        n_gradient_iterations=args.n_gradient,
        n_photons=args.photons,
        K=args.K,
        energy_lr=args.energy_lr,
        spatial_lr=args.spatial_lr,
        energy_scale=args.energy_scale,
        position_scale=args.position_scale,
        direction_scale=args.direction_scale,
        patience=args.patience,
        auto_scale=not args.no_auto_scale,
        target_energy_update_mev=args.target_energy_update,
        target_position_update_fraction=args.target_position_update,
        target_direction_update_degrees=args.target_direction_update,
        save_event_plots=args.save_event_plots,
        verbose=verbose,
        gradient_debug=args.gradient_debug,
        random_seed=args.seed
    )
    
    return results


if __name__ == "__main__":
    main()