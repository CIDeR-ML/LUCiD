#!/usr/bin/env python3
"""
BO-LEAP Optimization for LUCiD - Simplified Interface

This script merges the initial guess system from optimize.py with the BO-LEAP 
optimization algorithm from bo_leap.py, using WC_loss for a unified optimization.

Supports both data-like events and prediction events (closure tests).
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

from tools.utils import base_dir_path, spherical_to_cartesian
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.generate import read_photon_data_from_photonsim
from tools.optimization.algorithms import create_initial_population_database
from tools.optimization.bo_leap.bo_leap import setup_and_run_bo_leap
from tools.optimization.utils import (
    create_event_visualization, print_summary_statistics,
    create_convergence_plots, create_summary_plots
)
from tools.losses import WC_loss


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


def generate_random_event_params(key, detector_bounds, fraction=0.6):
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
    energy = 500.0  # Fixed energy for now
    
    return position, direction, energy


def create_bo_leap_loss_function(true_charges, true_times, simulate_event, sensor_params, 
                                sensor_positions, lambda_poisson=1.0, lambda_time=1.0):
    """Create loss function compatible with BO-LEAP optimization."""
    
    def loss_fn(params):
        """
        Loss function that takes a parameter array and returns scalar loss.
        
        params format: [energy, x, y, z, theta, phi]
        """
        energy = params[0]
        position = params[1:4]
        theta = params[4]
        phi = params[5]
        
        # Convert spherical to cartesian direction
        direction = spherical_to_cartesian(theta, phi)
        direction_angles = jnp.array([theta, phi])
        
        # Create particle params for simulator
        particle_params = (energy, position, direction_angles)
        
        # Generate key for simulation (using a fixed seed for reproducibility)
        sim_key = jax.random.PRNGKey(42)
        
        # Simulate event
        sim_charges, sim_times = simulate_event(particle_params, sensor_params, sim_key)
        
        # Calculate WC_loss
        loss = WC_loss(
            sensor_points=sensor_positions,
            true_charge=true_charges,
            true_time=true_times,
            simulated_charge=sim_charges,
            simulated_time=sim_times,
            lambda_poisson=lambda_poisson,
            lambda_time=lambda_time
        )
        
        return loss
    
    return loss_fn


def get_initial_guess_from_cache(true_charges, true_times, events_cache, detector_bounds, verbose=False):
    """Get initial guess using the database system from optimize.py."""
    
    # Use the existing database system for initial guess
    population = create_initial_population_database(
        None, None, None,  # These are not used when events_cache is provided
        true_charges, true_times, population_size=1, N=1, verbose=verbose, 
        events_cache=events_cache
    )
    
    # Extract the best guess
    best_guess = population[0]
    
    # Convert to BO-LEAP parameter format: [energy, x, y, z, theta, phi]
    position = best_guess['position']
    direction = best_guess['direction']
    energy = best_guess['energy']
    
    # Convert direction to spherical angles
    theta = jnp.arccos(jnp.clip(direction[2], -1.0, 1.0))
    phi = jnp.arctan2(direction[1], direction[0])
    
    initial_params = jnp.array([energy, position[0], position[1], position[2], theta, phi])
    
    if verbose:
        print(f"Initial guess from cache:")
        print(f"  Energy: {energy:.1f} MeV")
        print(f"  Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] m")
        print(f"  Direction: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")
        print(f"  Initial loss: {best_guess['loss']:.6f}")
    
    return initial_params


def setup_parameter_bounds(detector_bounds):
    """Setup parameter bounds for BO-LEAP optimization."""
    
    # Energy bounds (MeV)
    energy_min, energy_max = 100.0, 1000.0
    
    # Position bounds based on detector geometry
    if detector_bounds['type'] == 'cylinder':
        pos_min = jnp.array([-detector_bounds['r'], -detector_bounds['r'], -detector_bounds['H']/2])
        pos_max = jnp.array([detector_bounds['r'], detector_bounds['r'], detector_bounds['H']/2])
    elif detector_bounds['type'] == 'sphere':
        r = detector_bounds['r']
        pos_min = jnp.array([-r, -r, -r])
        pos_max = jnp.array([r, r, r])
    elif detector_bounds['type'] == 'box':
        pos_min = jnp.array([-detector_bounds['x']/2, -detector_bounds['y']/2, -detector_bounds['z']/2])
        pos_max = jnp.array([detector_bounds['x']/2, detector_bounds['y']/2, detector_bounds['z']/2])
    
    # Direction bounds (spherical angles)
    theta_min, theta_max = 0.0, jnp.pi  # polar angle
    phi_min, phi_max = 0.0, 2*jnp.pi    # azimuthal angle
    
    # Combine all bounds: [energy, x, y, z, theta, phi]
    lower_bounds = jnp.array([energy_min, pos_min[0], pos_min[1], pos_min[2], theta_min, phi_min])
    upper_bounds = jnp.array([energy_max, pos_max[0], pos_max[1], pos_max[2], theta_max, phi_max])
    
    # Cyclic mask for angular parameters (phi is cyclic)
    cyclic_mask = jnp.array([False, False, False, False, False, True])
    
    return lower_bounds, upper_bounds, cyclic_mask


def run_bo_leap_optimization(
    config_file,
    detector_type,
    n_events,
    bo_leap_iterations,
    cache_dir=None,
    data_mode=False,
    data_file='data/water/muon/50_data_like_events.root',
    n_photons=500_000,
    K=6,
    lambda_poisson=1.0,
    lambda_time=1.0,
    verbose=False,
    seed=1234567,
    save_event_plots=False,
    color_by='charge'
):
    """
    Main BO-LEAP optimization function.
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
    if verbose:
        print(f"Loading detector configuration...")
        print(f"Detector type: {detector_type}")
    
    detector = generate_detector(config_file)
    sensor_positions = jnp.array(detector.all_points)
    n_sensors = len(sensor_positions)
    
    if verbose:
        print(f"Detector has {n_sensors} sensors")
    
    # Get detector bounds
    detector_bounds = get_detector_bounds(detector)
    
    # Load event cache for initial guess system
    if verbose:
        print("Loading event cache for initial guess system...")
    
    from tools.optimization.event_cache import load_event_cache
    events_cache = load_event_cache(config_file, detector_type, K, verbose=verbose, cache_dir=cache_dir)
    
    if verbose:
        print(f"Loaded cache with {len(events_cache['metadata'])} events")
    
    # Setup simulation parameters
    sensor_params = (
        jnp.array(100.0),    # scatter_length
        jnp.array(0.1),      # reflection_rate
        jnp.array(100.0),    # absorption_length
        jnp.array(0.001)     # gumbel_softmax_temperature
    )
    
    # Setup event simulator
    if verbose:
        print("Setting up event simulator...")
    
    simulate_event = setup_event_simulator(
        json_filename=config_file,
        max_sensors_per_cell=4,
        n_photons=n_photons,
        temperature=0.05,
        K=K,
        detector_type=detector_type
    )
    
    # Setup parameter bounds for BO-LEAP
    lower_bounds, upper_bounds, cyclic_mask = setup_parameter_bounds(detector_bounds)
    
    # Initialize results storage
    results = []
    optimization_histories = []  # Store BO-LEAP histories for convergence plots
    total_optimization_time = 0.0
    
    # Progress bar for multi-event processing
    if not verbose and n_events > 1:
        event_iterator = tqdm(range(n_events), desc="Events", unit="event")
    else:
        event_iterator = range(n_events)
    
    for event_idx in event_iterator:
        if verbose:
            print(f"\n{'='*80}")
            print(f"EVENT {event_idx + 1}/{n_events}")
            print(f"{'='*80}")
        
        # Generate TRUE event - either prediction-like or data-like
        event_key = jax.random.PRNGKey(seed + event_idx * 1000)
        
        if data_mode:
            # Data-like events from ROOT file
            if verbose:
                print(f"Loading data-like event from ROOT file: {data_file}")
            
            # Setup data simulator with is_data=True and temperature=0.0
            data_simulate_event = setup_event_simulator(
                json_filename=config_file,
                max_sensors_per_cell=4,
                n_photons=n_photons,
                temperature=0.0,  # Zero temperature for data mode
                K=K,
                detector_type=detector_type,
                is_data=True  # Use data simulator
            )
            
            # Read ROOT file to get number of entries dynamically
            import uproot
            with uproot.open(data_file) as file:
                tree = file['OpticalPhotons']
                n_entries = tree.num_entries
            
            if verbose:
                print(f"ROOT file has {n_entries} entries")
            
            # Generate random entry index based on event_idx for reproducibility
            entry_key = jax.random.PRNGKey(seed + event_idx * 1000 + 500)
            entry_idx = int(jax.random.randint(entry_key, shape=(), minval=0, maxval=n_entries))
            
            if verbose:
                print(f"Loading entry {entry_idx} from ROOT file")
            
            # Load photon data from ROOT file
            photon_data = read_photon_data_from_photonsim(data_file, entry_idx)
            
            # Add the missing 'N' field that the data simulator expects
            photon_data['N'] = len(photon_data['photon_origins'])
            
            # Generate true event using data simulator
            # For data simulator, we need particle_params, detector_params, key, photon_data
            # Generate track parameters that we want to simulate
            true_position, true_direction, _ = generate_random_event_params(event_key, detector_bounds)
            true_energy = photon_data['energy']  # Energy from ROOT file
            
            # Create particle_params tuple for data simulator
            true_particle_params = (true_energy, true_position, true_direction)
            
            # Call data simulator with correct argument order
            true_charges, true_times = data_simulate_event(true_particle_params, sensor_params, event_key, photon_data)
            
        else:
            # Prediction-like events (original closure test)
            true_position, true_direction, true_energy = generate_random_event_params(event_key, detector_bounds)
            
            # Convert direction to spherical angles
            true_theta = jnp.arccos(jnp.clip(true_direction[2], -1.0, 1.0))
            true_phi = jnp.arctan2(true_direction[1], true_direction[0])
            true_direction_angles = jnp.array([true_theta, true_phi])
            
            # Simulate true event using standard simulator
            true_particle_params = (true_energy, true_position, true_direction_angles)
            true_charges, true_times = simulate_event(true_particle_params, sensor_params, event_key)
        
        if verbose:
            print(f"True event parameters:")
            print(f"  Position: {true_position}")
            print(f"  Direction: {true_direction}")
            print(f"  Energy: {true_energy:.1f} MeV")
            print(f"  Active sensors: {jnp.sum(true_charges > 0)}")
        
        # Get initial guess from cache
        if verbose:
            print(f"\nGetting initial guess from cache...")
        
        initial_guess = get_initial_guess_from_cache(
            true_charges, true_times, events_cache, detector_bounds, verbose=verbose
        )
        
        # Create loss function for BO-LEAP
        loss_fn = create_bo_leap_loss_function(
            true_charges, true_times, simulate_event, sensor_params, 
            sensor_positions, lambda_poisson, lambda_time
        )
        
        if verbose:
            print(f"\nStarting BO-LEAP optimization...")
            print(f"  Iterations: {bo_leap_iterations}")
        
        optimization_start_time = time.time()
        
        # Run BO-LEAP optimization
        opt_key = jax.random.PRNGKey(seed + 1000000 + event_idx * 1000)
        
        best_params, best_loss, history = setup_and_run_bo_leap(
            loss_fn_original=loss_fn,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            initial_guess=initial_guess,
            cyclic_mask=cyclic_mask,
            n_iterations=bo_leap_iterations,
            local_steps=20,  # Fixed local steps
            K=10,           # Population size
            J=10,           # Gradient steps
            M=100,          # GP subset size
            alpha=0.1,      # Gradient step size
            key=opt_key
        )
        
        optimization_duration = time.time() - optimization_start_time
        total_optimization_time += optimization_duration
        
        # Store optimization history for convergence plots
        optimization_histories.append(history)
        
        if verbose:
            print(f"Optimization completed in {optimization_duration:.1f} seconds")
        
        # Extract results
        fitted_energy = float(best_params[0])
        fitted_position = np.array(best_params[1:4])
        fitted_theta = float(best_params[4])
        fitted_phi = float(best_params[5])
        fitted_direction = np.array(spherical_to_cartesian(fitted_theta, fitted_phi))
        
        # Calculate errors
        position_error = float(jnp.linalg.norm(fitted_position - true_position))
        direction_error = float(jnp.arccos(jnp.clip(jnp.dot(fitted_direction, true_direction), -1, 1)))
        direction_error_deg = np.degrees(direction_error)
        energy_error = float(jnp.abs(fitted_energy - true_energy))
        energy_error_percent = (energy_error / float(true_energy)) * 100
        
        # Store results
        result = {
            'event_idx': event_idx,
            'true_position': np.array(true_position),
            'true_direction': np.array(true_direction),
            'true_energy': float(true_energy),
            'fitted_position': fitted_position,
            'fitted_direction': fitted_direction,
            'fitted_energy': fitted_energy,
            'position_error': position_error,
            'direction_error_deg': direction_error_deg,
            'energy_error': energy_error,
            'energy_error_percent': energy_error_percent,
            'final_loss': float(best_loss),
            'optimization_time': optimization_duration,
            'n_evaluations': history['n_evaluations'],
            'success': True
        }
        
        if verbose:
            print(f"\nEvent {event_idx + 1} Results:")
            print(f"  Position error: {position_error:.3f} m")
            print(f"  Direction error: {direction_error_deg:.1f}°")
            print(f"  Energy error: {energy_error:.1f} MeV ({energy_error_percent:.1f}%)")
            print(f"  Final loss: {best_loss:.6f}")
            print(f"  Evaluations: {history['n_evaluations']}")
        
        # Create event visualization if requested
        if save_event_plots:
            # Create a best_match dict in the format expected by create_event_visualization
            best_match = {
                'position': fitted_position,
                'direction': fitted_direction,
                'energy': fitted_energy,
                'loss': float(best_loss)
            }
            
            create_event_visualization(
                true_position, true_direction, true_energy,
                best_match, true_charges, true_times, 
                sensor_positions, detector_bounds,
                position_error, direction_error_deg, energy_error, energy_error_percent,
                event_idx, figures_dir, verbose, config_file, color_by
            )
        
        results.append(result)
    
    # Print summary statistics and create plots
    if n_events > 1:
        try:
            print_summary_statistics(results, total_optimization_time)
        except Exception as e:
            print(f"Error in print_summary_statistics: {e}")
            # Fallback to basic summary
            print(f"\n{'='*80}")
            print(f"SUMMARY STATISTICS ({n_events} events)")
            print(f"{'='*80}")
            
            position_errors = [r['position_error'] for r in results]
            direction_errors = [r['direction_error_deg'] for r in results]
            energy_errors = [r['energy_error_percent'] for r in results]
            final_losses = [r['final_loss'] for r in results]
            optimization_times = [r['optimization_time'] for r in results]
            n_evals = [r['n_evaluations'] for r in results]
            
            print(f"Position Error (m):")
            print(f"  Mean: {np.mean(position_errors):.3f} ± {np.std(position_errors):.3f}")
            print(f"  Median: {np.median(position_errors):.3f}")
            print(f"  Range: [{np.min(position_errors):.3f}, {np.max(position_errors):.3f}]")
            
            print(f"\nDirection Error (degrees):")
            print(f"  Mean: {np.mean(direction_errors):.1f} ± {np.std(direction_errors):.1f}")
            print(f"  Median: {np.median(direction_errors):.1f}")
            print(f"  Range: [{np.min(direction_errors):.1f}, {np.max(direction_errors):.1f}]")
            
            print(f"\nEnergy Error (%):")
            print(f"  Mean: {np.mean(energy_errors):.1f} ± {np.std(energy_errors):.1f}")
            print(f"  Median: {np.median(energy_errors):.1f}")
            print(f"  Range: [{np.min(energy_errors):.1f}, {np.max(energy_errors):.1f}]")
            
            print(f"\nFinal Loss:")
            print(f"  Mean: {np.mean(final_losses):.2e}")
            print(f"  Median: {np.median(final_losses):.2e}")
            print(f"  Range: [{np.min(final_losses):.2e}, {np.max(final_losses):.2e}]")
            
            print(f"\nFunction Evaluations:")
            print(f"  Mean: {np.mean(n_evals):.0f} ± {np.std(n_evals):.0f}")
            print(f"  Total: {np.sum(n_evals)}")
            
            print(f"\nTiming:")
            print(f"  Total optimization time: {total_optimization_time:.1f} seconds")
            print(f"  Average time per event: {np.mean(optimization_times):.1f} seconds")
            print(f"  Success rate: 100.0%")
        
        try:
            create_summary_plots(results, figures_dir, config_file=config_file)
            print(f"Summary plots saved to {figures_dir}")
        except Exception as e:
            print(f"Error creating summary plots: {e}")
        
        try:
            # Create BO-LEAP convergence plots
            create_bo_leap_convergence_plots(optimization_histories, figures_dir, config_file=config_file)
        except Exception as e:
            print(f"Error creating convergence plots: {e}")
    
    return results


def create_bo_leap_convergence_plots(optimization_histories, figures_dir, config_file=None):
    """Create convergence plots specific to BO-LEAP optimization."""
    import matplotlib.pyplot as plt
    
    # Extract detector name from config file for plot titles
    detector_name = "Unknown"
    if config_file:
        config_basename = os.path.basename(config_file)
        if '_geom_config.json' in config_basename:
            detector_name = config_basename.replace('_geom_config.json', '')
        elif '.json' in config_basename:
            detector_name = config_basename.replace('.json', '')
    
    n_events = len(optimization_histories)
    
    # Create convergence plot showing best loss vs evaluation number
    plt.figure(figsize=(12, 8))
    
    for event_idx, history in enumerate(optimization_histories):
        # Extract loss values from BO-LEAP history
        all_losses = history['all_y']
        
        # Compute cumulative best (running minimum)
        cumulative_best = np.minimum.accumulate(all_losses)
        
        # Plot convergence
        plt.subplot(2, 2, 1)
        plt.plot(cumulative_best, alpha=0.7, label=f'Event {event_idx+1}')
    
    plt.subplot(2, 2, 1)
    plt.xlabel('Function Evaluations')
    plt.ylabel('Best Loss')
    plt.title(f'BO-LEAP Convergence - {detector_name}')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    if n_events <= 10:
        plt.legend()
    
    # Plot distribution of final losses
    final_losses = [history['all_y'][np.argmin(history['all_y'])] for history in optimization_histories]
    
    plt.subplot(2, 2, 2)
    plt.hist(final_losses, bins=min(20, n_events//2), alpha=0.7, edgecolor='black')
    plt.xlabel('Final Loss')
    plt.ylabel('Frequency')
    plt.title('Distribution of Final Losses')
    plt.grid(True, alpha=0.3)
    
    # Plot number of evaluations per event
    n_evaluations = [history['n_evaluations'] for history in optimization_histories]
    
    plt.subplot(2, 2, 3)
    plt.bar(range(1, n_events+1), n_evaluations, alpha=0.7)
    plt.xlabel('Event Number')
    plt.ylabel('Function Evaluations')
    plt.title('Evaluations per Event')
    plt.grid(True, alpha=0.3)
    
    # Plot average convergence across all events
    plt.subplot(2, 2, 4)
    
    # Find maximum number of evaluations to align curves
    max_evals = max(len(history['all_y']) for history in optimization_histories)
    
    # Create interpolated convergence curves
    eval_points = np.linspace(1, max_evals, 200)
    convergence_curves = []
    
    for history in optimization_histories:
        all_losses = history['all_y']
        cumulative_best = np.minimum.accumulate(all_losses)
        # Interpolate to common evaluation points
        eval_indices = np.arange(1, len(cumulative_best) + 1)
        interpolated = np.interp(eval_points, eval_indices, cumulative_best)
        convergence_curves.append(interpolated)
    
    # Compute statistics
    convergence_curves = np.array(convergence_curves)
    mean_convergence = np.mean(convergence_curves, axis=0)
    std_convergence = np.std(convergence_curves, axis=0)
    
    plt.plot(eval_points, mean_convergence, 'b-', linewidth=2, label='Mean')
    plt.fill_between(eval_points, 
                     mean_convergence - std_convergence,
                     mean_convergence + std_convergence,
                     alpha=0.3, label='±1 std')
    
    plt.xlabel('Function Evaluations')
    plt.ylabel('Best Loss')
    plt.title('Average Convergence')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'{detector_name}_bo_leap_convergence_{timestamp}.png'
    plot_path = os.path.join(figures_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"BO-LEAP convergence plots saved to {plot_path}")
    plt.close()


def main():
    """Main function with simplified command-line interface."""
    parser = argparse.ArgumentParser(
        description='BO-LEAP optimization for LUCiD with simplified interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Prediction events (closure test) with 10 BO-LEAP iterations
    python -m tools.optimization.bo_leap_optimize --iterations 10 -n 5
    
    # Data-like events with custom cache directory
    python -m tools.optimization.bo_leap_optimize --data-mode --iterations 15 --cache-dir /path/to/cache
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
    
    # BO-LEAP parameters
    parser.add_argument('--iterations', '-i', type=int, default=10,
                        help='Number of BO-LEAP iterations (default: 10)')
    
    # Data mode parameters
    parser.add_argument('--data-mode', action='store_true',
                        help='Use data-like events from ROOT file instead of prediction-like events')
    parser.add_argument('--data-file', type=str, default='data/water/muon/50_data_like_events.root',
                        help='Path to ROOT file containing data-like events')
    
    # Cache parameters
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='External directory path for event cache storage')
    
    # Simulation parameters
    parser.add_argument('--photons', type=int, default=500_000,
                        help='Number of photons to simulate (default: 500000)')
    parser.add_argument('--K', type=int, default=6,
                        help='Number of nearest neighbors for sensor mapping (default: 6)')
    
    # Loss function parameters
    parser.add_argument('--lambda-poisson', type=float, default=1.0,
                        help='Weight for Poisson loss component (default: 1.0)')
    parser.add_argument('--lambda-time', type=float, default=1.0,
                        help='Weight for time loss component (default: 1.0)')
    
    # Visualization parameters
    parser.add_argument('--save-event-plots', action='store_true',
                        help='Save 3D visualization plots for individual events')
    parser.add_argument('--color-by', type=str, default='charge',
                        choices=['time', 'charge'],
                        help='Color sensor hits by time or charge (default: charge)')
    
    # Other parameters
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed progress information')
    parser.add_argument('--seed', type=int, default=1234567,
                        help='Random seed (default: 1234567)')
    
    args = parser.parse_args()
    
    print(f"Running BO-LEAP optimization...")
    if args.data_mode:
        print(f"Mode: Data-like events from {args.data_file}")
    else:
        print(f"Mode: Prediction events (closure test)")
    print(f"BO-LEAP iterations: {args.iterations}")
    print(f"Events to process: {args.events}")
    
    # Run optimization
    results = run_bo_leap_optimization(
        config_file=args.config,
        detector_type=args.detector,
        n_events=args.events,
        bo_leap_iterations=args.iterations,
        cache_dir=args.cache_dir,
        data_mode=args.data_mode,
        data_file=args.data_file,
        n_photons=args.photons,
        K=args.K,
        lambda_poisson=args.lambda_poisson,
        lambda_time=args.lambda_time,
        verbose=args.verbose,
        seed=args.seed,
        save_event_plots=args.save_event_plots,
        color_by=args.color_by
    )
    
    return results


if __name__ == "__main__":
    main()