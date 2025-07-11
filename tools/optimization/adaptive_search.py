 #!/usr/bin/env python3
"""
Main script for adaptive search optimization using LUCiD.
Supports multiple detector geometries: Cylinder, Sphere, and Box.
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
from datetime import datetime
import argparse
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.utils import base_dir_path, spherical_to_cartesian
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.optimization.optimize import adaptive_search
from tools.optimization.visualization import (
    create_event_visualization, print_summary_statistics,
    create_convergence_plots, create_summary_plots
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
        # Assuming the box detector has attributes for dimensions
        return {
            'type': 'box',
            'x': detector.x_size,
            'y': detector.y_size,
            'z': detector.z_size
        }
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


def generate_random_event_params(key, detector_bounds):
    """Generate random event parameters based on detector geometry."""
    if detector_bounds['type'] == 'cylinder':
        # Random vertex position
        r_vert = jax.random.uniform(key, shape=(), minval=0, maxval=detector_bounds['r'] * 0.8)
        key, _ = jax.random.split(key)
        theta = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
        key, _ = jax.random.split(key)
        z_vert = jax.random.uniform(key, shape=(), minval=-detector_bounds['H']/2 * 0.8, 
                                   maxval=detector_bounds['H']/2 * 0.8)
        position = jnp.array([r_vert * jnp.cos(theta), r_vert * jnp.sin(theta), z_vert])
        
    elif detector_bounds['type'] == 'sphere':
        # Random position in sphere (uniform volume sampling)
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
        # Random position in box
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


def main(config_file=None, detector_type='Cylinder', verbose=True, n_events=1, 
         save_event_plots=False, n_iterations=40, population_size=20):
    """
    Run adaptive search for best parameters on N events.
    
    Parameters:
    -----------
    config_file : str
        Path to detector configuration file
    detector_type : str
        Type of detector: 'Cylinder', 'Sphere', or 'Box'
    verbose : bool
        Whether to show detailed output
    n_events : int
        Number of events to process
    save_event_plots : bool
        Whether to save individual event visualizations
    n_iterations : int
        Number of iterations for adaptive search
    population_size : int
        Population size for adaptive search
    """
    
    # Configuration
    random_seed = 43
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
        n_photons=1_000_000,
        temperature=0.0,
        K=2,
        detector_type=detector_type
    )
    
    # Initialize statistics collection
    results = []
    event_histories = []
    total_search_time = 0.0
    
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
        true_position, true_direction, true_energy = generate_random_event_params(subkey, detector_bounds)
        
        # Convert direction to spherical angles
        true_theta = jnp.arccos(true_direction[2])
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
        
        # Run adaptive search
        if verbose or n_events == 1:
            print(f"\nStarting adaptive search...")
        search_start_time = datetime.now()
        
        # Track history for multi-event runs to enable convergence plots
        track_history = n_events > 1
        search_result = adaptive_search(
            true_charges, true_times, simulate_event, sensor_params, sensor_positions,
            detector_bounds, true_position, true_direction, true_energy,
            n_iterations=n_iterations, population_size=population_size, elite_fraction=0.2, 
            verbose=verbose and n_events == 1, track_history=track_history
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
            print(f"Adaptive search completed in {search_duration:.1f} seconds")
        
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
                'success': True
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
                    event_idx, figures_dir, verbose
                )
        else:
            result = {
                'event_idx': event_idx,
                'success': False,
                'search_time': search_duration
            }
            print(f"Event {event_idx + 1}: Adaptive search failed!")
        
        results.append(result)
    
    # Print summary statistics and create plots
    if n_events > 1:
        print_summary_statistics(results, total_search_time)
        create_summary_plots(results, figures_dir)
        
        # Create convergence plots if we have event histories
        if event_histories:
            create_convergence_plots(event_histories, figures_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adaptive search for LUCiD reconstruction')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to detector configuration file')
    parser.add_argument('--detector', '-d', type=str, default='Cylinder',
                        choices=['Cylinder', 'Sphere', 'Box'],
                        help='Detector geometry type')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Show detailed progress during iterations')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress iteration details (opposite of verbose)')
    parser.add_argument('--events', '-n', type=int, default=1,
                        help='Number of events to process (default: 1)')
    parser.add_argument('--save-event-plots', action='store_true',
                        help='Save visualization plots for individual events')
    parser.add_argument('--iterations', '-i', type=int, default=40,
                        help='Number of iterations for adaptive search (default: 40)')
    parser.add_argument('--population', '-p', type=int, default=20,
                        help='Population size for adaptive search (default: 20)')
    
    args = parser.parse_args()
    
    # Determine verbosity (quiet takes precedence)
    verbose = True
    if args.quiet:
        verbose = False
    elif args.verbose:
        verbose = True
    
    main(config_file=args.config, detector_type=args.detector, verbose=verbose, 
         n_events=args.events, save_event_plots=args.save_event_plots,
         n_iterations=args.iterations, population_size=args.population)