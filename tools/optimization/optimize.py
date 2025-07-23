#!/usr/bin/env python3
"""
LUCiD Parameter Optimization - Master Script
This script provides a unified interface for track optimization in LUCiD

Usage examples:
    python -m tools.optimization.optimize -i 20 -p 20 --n-gradient 250 -n 2 \
      -c "config/HK_geom_config.json" -d "Cylinder" --energy-scale 20 --position-scale 3.0 \
      --direction-scale 0.025 --photons 100_000 --patience 100 --K 6
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
import argparse
import time
import pickle
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.utils import base_dir_path, generate_random_params, spherical_to_cartesian
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.optimization.algorithms import create_initial_population, create_initial_population_database, evolve_population, gradient_optimization
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


def generate_random_event_params(key, detector_bounds, fraction=0.7):
    """Generate random event parameters based on detector geometry."""
    if detector_bounds['type'] == 'cylinder':
        r_vert = jax.random.uniform(key, shape=(), minval=0, maxval=detector_bounds['r'] * fraction)
        key, _ = jax.random.split(key)
        theta = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
        key, _ = jax.random.split(key)
        z_vert = jax.random.uniform(key, shape=(), minval=-detector_bounds['H']/2 * fraction, 
                                   maxval=detector_bounds['H']/2 * fraction)
        position = jnp.array([r_vert * jnp.cos(theta), r_vert * jnp.sin(theta), z_vert])
        
    elif detector_bounds['type'] == 'sphere':
        u = jax.random.uniform(key, shape=())
        key, _ = jax.random.split(key)
        cos_theta = jax.random.uniform(key, shape=(), minval=-1, maxval=1)
        key, _ = jax.random.split(key)
        phi = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
        
        r = detector_bounds['r'] * fraction * jnp.cbrt(u)
        sin_theta = jnp.sqrt(1 - cos_theta**2)
        position = jnp.array([r * sin_theta * jnp.cos(phi), 
                             r * sin_theta * jnp.sin(phi), 
                             r * cos_theta])
        
    elif detector_bounds['type'] == 'box':
        position = jax.random.uniform(key, shape=(3,), 
                                    minval=jnp.array([-detector_bounds['x']/2, 
                                                     -detector_bounds['y']/2, 
                                                     -detector_bounds['z']/2]) * fraction,
                                    maxval=jnp.array([detector_bounds['x']/2, 
                                                     detector_bounds['y']/2, 
                                                     detector_bounds['z']/2]) * fraction)
    
    # Random direction
    key, _ = jax.random.split(key)
    phi = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
    key, _ = jax.random.split(key)
    cos_theta = jax.random.uniform(key, shape=(), minval=-1, maxval=1)
    sin_theta = jnp.sqrt(1 - cos_theta**2)
    direction = jnp.array([sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), cos_theta])
    
    # Random energy
    key, _ = jax.random.split(key)
    energy = jax.random.uniform(key, shape=(), minval=250.0, maxval=850.0)
    
    return position, direction, energy


def optimization_engine(true_charges, true_times, simulate_event, sensor_params, sensor_positions,
                   detector_bounds, true_position, true_direction, true_energy,
                   numerical_iterations, population_size, elite_fraction,
                   seed, verbose, use_track_history,
                   gradient_iterations, gradient_kwargs,
                   loss_function, numerical_debug, crossover_rate=0.3,
                   events_cache=None):
    """
    Some description

    """
    key = jax.random.PRNGKey(seed)
    
    # Initialize history tracking
    history = {
        'best_loss': [],
        'best_energy': [],
        'best_position': [],
        'best_direction': [],
        'position_error': [],
        'direction_error': [],
        'energy_error': [],
        'population_size': population_size  # Store for evaluation counting
    } if use_track_history else None
    
    # Choose initialization method based on numerical_iterations
    start_num_search = time.time()
    
    if numerical_iterations == 0:
        # Use N=3 averaging for direct gradient optimization
        if verbose:
            print("Using initial guess from database (N=3 averaging)...")
        population = create_initial_population_database(
            None, None, None, 
            true_charges, true_times, population_size, N=3, verbose=verbose, events_cache=events_cache
        )
    else:
        # Use single best match for hybrid optimization
        if verbose:
            print("Using initial guess from database (single best match)...")
        population = create_initial_population_database(
            None, None, None,
            true_charges, true_times, population_size, N=1, verbose=verbose, events_cache=events_cache
        )
    
    if numerical_debug:
        print(f"  True parameters for reference:")
        print(f"    Position: [{true_position[0]:.3f}, {true_position[1]:.3f}, {true_position[2]:.3f}]")
        print(f"    Direction: [{true_direction[0]:.3f}, {true_direction[1]:.3f}, {true_direction[2]:.3f}]")
        print(f"    Energy: {true_energy:.1f} MeV")

    true_values = true_charges, true_times, true_energy, true_position, true_direction
    n_elite = int(population_size * elite_fraction)
    
    if numerical_iterations == 0:
        # Skip genetic algorithm - use initial guess directly for gradient optimization
        best_overall = population[0].copy()  # First element is the best initial guess
        best_overall_loss = best_overall['loss']
        
        if verbose:
            print(f"Skipping numerical search (iterations=0)")
            print(f"Using initial guess directly:")
            print(f"  Position: [{best_overall['position'][0]:.3f}, {best_overall['position'][1]:.3f}, {best_overall['position'][2]:.3f}]")
            print(f"  Direction: [{best_overall['direction'][0]:.3f}, {best_overall['direction'][1]:.3f}, {best_overall['direction'][2]:.3f}]")
            print(f"  Energy: {best_overall['energy']:.1f} MeV")
            print(f"  Initial loss: {best_overall_loss:.6f}")
    else:
        # Run genetic algorithm
        population, best_overall, best_overall_loss, history = evolve_population(key, numerical_iterations, population, n_elite, loss_function, \
            detector_bounds, true_values, history, verbose, numerical_debug, crossover_rate)

    if verbose:
        print(f"   Numerical search took: {time.time() - start_num_search:.6f} seconds")

    # Apply gradient optimization if requested
    if gradient_iterations > 0:
        if verbose:
            print(f"\nApplying gradient-based optimization...")
            print(f"  Gradient iterations: {gradient_iterations}")
            print(f"  Gradient kwargs: {gradient_kwargs}")
        
        # Convert best result to parameter format
        theta = jnp.arccos(best_overall['direction'][2])
        phi = jnp.arctan2(best_overall['direction'][1], best_overall['direction'][0])
        initial_params = (
            best_overall['energy'],
            best_overall['position'],
            jnp.array([theta, phi])
        )
        
        # Run grad optimization
        best_params, grad_history = gradient_optimization(
            initial_params, true_charges, true_times,
            simulate_event, sensor_params, sensor_positions,
            detector_bounds, true_position, true_direction, true_energy,
            gradient_iterations, gradient_kwargs, key, verbose)

        # Convert back to result format
        energy, position, direction_angles = best_params
        theta, phi = direction_angles
        direction = spherical_to_cartesian(theta, phi)
        
        best_overall = {
            'position': position,
            'direction': direction,
            'energy': energy,
            'loss': grad_history['loss'][-1]
        }
        
        # Update history if tracking
        if use_track_history:
            history['gradient_loss'] = grad_history['loss']
            history['gradient_energy'] = grad_history['energy']
            history['gradient_position'] = grad_history['position']
            history['gradient_direction'] = grad_history['direction']
            # Store error histories from gradient optimization
            history['gradient_position_error'] = grad_history['position_error']
            history['gradient_direction_error'] = grad_history['direction_error']
            history['gradient_energy_error'] = grad_history['energy_error']
            # Store the actual best loss from gradient optimization
            history['gradient_best_loss'] = min(grad_history['loss'])
    
    if use_track_history:
        return best_overall, population, history
    else:
        return best_overall, population



def run_optimization(
    config_file,
    detector_type,
    n_events,
    numerical_iterations,
    population_size,
    n_gradient_iterations,
    n_photons,
    K,
    energy_lr,
    spatial_lr,
    energy_scale,
    position_scale,
    direction_scale,
    patience,
    save_event_plots,
    verbose,
    gradient_debug,
    numerical_debug,
    seed,
    color_by,
    event_seeds,
    crossover_rate=0.3
):
    """
    Main optimization function
    """
    
    # Configuration
    if config_file is None:
        config_file = base_dir_path() + 'config/IWCD_geom_config.json'
    
    # Create output directory
    figures_dir = os.path.join(base_dir_path(), 'output', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Initialize random key
    key = jax.random.PRNGKey(seed)
    
    # Setup detector
    if verbose or n_events == 1:
        print(f"Loading detector configuration...")
        print(f"Detector type: {detector_type}")
        print(f"Gradient parameters: energy_lr={energy_lr}, spatial_lr={spatial_lr}")
        print(f"Gradient scales: energy={energy_scale}, position={position_scale}, direction={direction_scale}")
    
    detector = generate_detector(config_file)
    sensor_positions = jnp.array(detector.all_points)
    n_sensors = len(sensor_positions)
    
    if verbose or n_events == 1:
        print(f"Detector has {n_sensors} sensors")
    
    # Get detector bounds
    detector_bounds = get_detector_bounds(detector)
    
    # Load event cache once for all events
    if verbose or n_events == 1:
        print("Loading event cache for initial guess system...")
    
    from tools.optimization.event_cache import load_event_cache
    events_cache = load_event_cache(config_file, detector_type, K, verbose=(verbose or n_events == 1))
    
    if verbose or n_events == 1:
        print(f"Loaded cache with {len(events_cache['metadata'])} events")
    
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
        temperature=0.05,
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
        'gradient_debug': gradient_debug,
        'gradient_verbose': gradient_debug
    }
    
    # Determine which events to process
    if event_seeds is not None:
        event_iterator = event_seeds
        actual_n_events = len(event_seeds)
        if verbose:
            print(f"\nProcessing {actual_n_events} specific events: {event_seeds}")
        else:
            print(f"\nProcessing {actual_n_events} specific events: {event_seeds}")
    else:
        event_iterator = range(n_events)
        actual_n_events = n_events
        if verbose:
            print(f"\nProcessing {actual_n_events} events...")
        else:
            print(f"\nProcessing {actual_n_events} events...")
    
    # Progress bar for multi-event processing
    if not verbose and actual_n_events > 1:
        event_iterator = tqdm(event_iterator, desc="Events", unit="event")
    
    for event_idx in event_iterator:
        if verbose:
            print(f"\n{'='*80}")
            if event_seeds is not None:
                print(f"EVENT INDEX {event_idx} (#{event_seeds.index(event_idx) + 1}/{actual_n_events})")
            else:
                print(f"EVENT {event_idx + 1}/{actual_n_events}")
            print(f"{'='*80}")
        
        # Generate TRUE event with deterministic seed based on event index
        # This ensures that event_idx=3 always generates the same event
        event_key = jax.random.PRNGKey(seed + event_idx * 1000)        
        true_position, true_direction, true_energy = generate_random_event_params(event_key, detector_bounds)

        # Convert direction to spherical angles
        true_theta = jnp.arccos(jnp.clip(true_direction[2], -1.0, 1.0))
        true_phi = jnp.arctan2(true_direction[1], true_direction[0])
        true_direction_angles = jnp.array([true_theta, true_phi])
        
        # Simulate true event
        true_particle_params = (true_energy, true_position, true_direction_angles)
        true_charges, true_times = simulate_event(true_particle_params, sensor_params, event_key)
        
        if verbose or n_events == 1:
            print(f"True event parameters:")
            print(f"  Position: {true_position}")
            print(f"  Direction: {true_direction}")
            print(f"  Energy: {true_energy:.1f} MeV")
            print(f"  Active sensors: {jnp.sum(true_charges > 0)}")
        
        if verbose or n_events == 1:
            print(f"\nStarting optimization...")
        
        search_start_time = datetime.now()
        
        # Track history for multi-event runs to enable convergence plots
        use_track_history = n_events > 1
        
        # Use a different seed for optimization to avoid correlation with event generation
        # But keep it deterministic based on event index
        optimization_seed = seed + 1000000 + event_idx * 1000
        
        # Use the combined loss function from losses.py
        from tools.optimization.losses import combined_loss_fn
        
        search_result = optimization_engine(
            true_charges, true_times, simulate_event, sensor_params, sensor_positions,
            detector_bounds, true_position, true_direction, true_energy,
            numerical_iterations=numerical_iterations, 
            population_size=population_size, 
            elite_fraction=0.2,
            verbose=verbose and n_events == 1, 
            use_track_history=use_track_history,
            gradient_iterations=n_gradient_iterations,
            gradient_kwargs=gradient_kwargs,
            seed=optimization_seed,
            loss_function=lambda params, true_charges, true_times, event_key: sum(combined_loss_fn(
                params, true_charges, true_times, 
                simulate_event, sensor_params, sensor_positions, event_key,
                tau=gradient_kwargs['tau'], 
                lambda_time=gradient_kwargs['lambda_time']
            )),
            numerical_debug=numerical_debug,
            crossover_rate=crossover_rate,
            events_cache=events_cache
        )
        
        # Unpack results based on whether history was tracked
        if use_track_history:
            best_match, final_population, event_history = search_result
            event_histories.append(event_history)
        else:
            best_match, final_population = search_result
            
        search_duration = (datetime.now() - search_start_time).total_seconds()
        total_search_time += search_duration
        
        if verbose or n_events == 1:
            print(f"Optimization completed in {search_duration:.1f} seconds")
        
        # Calculate errors and store results
        position_error = float(jnp.linalg.norm(best_match['position'] - true_position))
        direction_error = float(jnp.arccos(jnp.clip(jnp.dot(best_match['direction'], true_direction), -1, 1)))
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
                event_idx, figures_dir, verbose, config_file, color_by
            )
        
        results.append(result)
    
    # Print summary statistics and create plots
    if actual_n_events > 1:
        print_summary_statistics(results, total_search_time)
        create_summary_plots(results, figures_dir, config_file=config_file)
        
        # Create convergence plots if we have event histories
        if event_histories:
            # Save convergence data for later plotting
            save_convergence_data(event_histories, results, figures_dir, config_file)
            # # Check if we have gradient history (hybrid optimization)
            if event_histories[0] and 'gradient_loss' in event_histories[0]:
                create_hybrid_convergence_plot(event_histories, figures_dir, config_file=config_file)
            else:
                create_convergence_plots(event_histories, figures_dir, config_file=config_file)
    
    return results


def save_convergence_data(event_histories, results, figures_dir, config_file):
    """
    Save convergence plot data to pickle file for later plotting.
    
    Parameters:
    -----------
    event_histories : list
        List of optimization histories for each event
    results : list
        List of optimization results for each event
    figures_dir : str
        Directory to save the data file
    config_file : str
        Path to detector configuration file
    """
    # Extract detector name from config file
    detector_name = "Unknown"
    if config_file:
        config_basename = os.path.basename(config_file)
        if '_geom_config.json' in config_basename:
            detector_name = config_basename.replace('_geom_config.json', '')
        elif '.json' in config_basename:
            detector_name = config_basename.replace('.json', '')
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data to save
    convergence_data = {
        'detector_name': detector_name,
        'timestamp': timestamp,
        'config_file': config_file,
        'event_histories': event_histories,
        'results': results,
        'metadata': {
            'n_events': len(results),
            'successful_events': len([r for r in results if r.get('success', False)]),
            'has_gradient_data': event_histories[0] and 'gradient_loss' in event_histories[0] if event_histories else False
        }
    }
    
    # Convert JAX arrays to numpy for serialization
    def convert_jax_to_numpy(obj):
        """Recursively convert JAX arrays to numpy arrays for pickle serialization."""
        if isinstance(obj, jnp.ndarray):
            return np.array(obj)
        elif isinstance(obj, dict):
            return {key: convert_jax_to_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_jax_to_numpy(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_jax_to_numpy(item) for item in obj)
        else:
            return obj
    
    convergence_data = convert_jax_to_numpy(convergence_data)
    
    # Save to pickle file
    filename = f'{detector_name}_convergence_data_{timestamp}.pkl'
    filepath = os.path.join(figures_dir, filename)
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(convergence_data, f)
        print(f"Convergence data saved to {filepath}")
        
        # Also save a human-readable summary
        summary_filename = f'{detector_name}_convergence_summary_{timestamp}.txt'
        summary_filepath = os.path.join(figures_dir, summary_filename)
        
        with open(summary_filepath, 'w') as f:
            f.write(f"LUCiD Optimization Convergence Data Summary\n")
            f.write(f"==========================================\n\n")
            f.write(f"Detector: {detector_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Config File: {config_file}\n")
            f.write(f"Number of Events: {convergence_data['metadata']['n_events']}\n")
            f.write(f"Successful Events: {convergence_data['metadata']['successful_events']}\n")
            f.write(f"Has Gradient Data: {convergence_data['metadata']['has_gradient_data']}\n\n")
            
            f.write(f"Data File: {filename}\n\n")
            
            f.write(f"To load and plot this data later:\n")
            f.write(f"```python\n")
            f.write(f"import pickle\n")
            f.write(f"import matplotlib.pyplot as plt\n\n")
            f.write(f"# Load data\n")
            f.write(f"with open('{filename}', 'rb') as f:\n")
            f.write(f"    data = pickle.load(f)\n\n")
            f.write(f"event_histories = data['event_histories']\n")
            f.write(f"results = data['results']\n\n")
            f.write(f"# Plot convergence (example for position error)\n")
            f.write(f"for i, history in enumerate(event_histories):\n")
            f.write(f"    plt.plot(history['position_error'], alpha=0.7, label=f'Event {{i+1}}')\n")
            f.write(f"plt.xlabel('Iteration')\n")
            f.write(f"plt.ylabel('Position Error (m)')\n")
            f.write(f"plt.title('Position Error Convergence')\n")
            f.write(f"plt.legend()\n")
            f.write(f"plt.show()\n")
            f.write(f"```\n")
        
        print(f"Convergence summary saved to {summary_filepath}")
        
    except Exception as e:
        print(f"Warning: Could not save convergence data: {e}")


def main():

    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='LUCiD parameter optimization with multiple methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Numerical optimization (like old adaptive_search.py)
    python -m tools.optimization.optimize -i 40 -p 30 -n 10 -q
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
    
    # Numerical optimization parameters
    parser.add_argument('--iterations', '-i', type=int, default=30,
                        help='Number of numerical iterations (default: 30)')
    parser.add_argument('--population', '-p', type=int, default=20,
                        help='Population size for numerical optimization (default: 20)')
    parser.add_argument('--crossover-rate', type=float, default=0.3,
                        help='Crossover rate for genetic algorithm (0.0-1.0, default: 0.3)')
    
    # Gradient optimization parameters
    parser.add_argument('--n-gradient', type=int, default=1000,
                        help='Number of gradient iterations (default: 1000)')
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
    parser.add_argument('--patience', type=int, default=100,
                        help='Patience for learning rate reduction (default: 100)')

    # Simulation parameters
    parser.add_argument('--photons', type=int, default=500_000,
                        help='Number of photons to simulate (default: 500000)')
    parser.add_argument('--K', type=int, default=6,
                        help='Number of nearest neighbors for sensor mapping (default: 2)')
    
    # Output and verbosity
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed progress during iterations')
    parser.add_argument('--gradient-debug', action='store_true',
                        help='Show gradient debugging info every iteration (instead of every 10)')
    parser.add_argument('--numerical-debug', action='store_true',
                        help='Show detailed debugging info for numerical optimization')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress iteration details (opposite of verbose)')
    parser.add_argument('--save-event-plots', action='store_true',
                        help='Save visualization plots for individual events')
    parser.add_argument('--color-by', type=str, default='time',
                        choices=['time', 'charge'],
                        help='Color sensor hits by time or charge (default: time)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=1234567,
                        help='Random seed (default: 1234567)')
    parser.add_argument('--event-seeds', type=str, default=None,
                        help='Comma-separated list of event indices to investigate (e.g., "3,9,11")')
    
    args = parser.parse_args()
    
    # Determine verbosity (quiet takes precedence)
    verbose = True
    if args.quiet:
        verbose = False
    elif args.verbose:
        verbose = True
    
    # Parse event seeds if provided
    event_seeds = None
    if args.event_seeds:
        try:
            event_seeds = [int(x.strip()) for x in args.event_seeds.split(',')]
            print(f"Will investigate specific events: {event_seeds}")
        except ValueError:
            print("Error: --event-seeds must be comma-separated integers (e.g., '3,9,11')")
            return None
    
    print(f"Running optimization")
    # Run optimization
    results = run_optimization(
        config_file=args.config,
        detector_type=args.detector,
        n_events=args.events,
        numerical_iterations=args.iterations,
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
        save_event_plots=args.save_event_plots,
        verbose=verbose,
        gradient_debug=args.gradient_debug,
        numerical_debug=args.numerical_debug,
        seed=args.seed,
        color_by=args.color_by,
        event_seeds=event_seeds,
        crossover_rate=args.crossover_rate
    )
    
    return results


if __name__ == "__main__":
    main()