#!/usr/bin/env python3
"""
Script to generate N events, store them, and benchmark loss calculation.
This script:
1. Generates N events with random parameters
2. Stores them (charges and times)
3. Generates a new event and calculates loss to all stored events
4. Times the loss calculation
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
import time
import pickle
import argparse
from datetime import datetime
from jax import jit, vmap
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.utils import base_dir_path, spherical_to_cartesian
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.optimization.optimize import get_detector_bounds, generate_random_event_params


def generate_and_store_events(n_events, config_file, detector_type, n_photons, K, seed=42):
    """
    Generate N events and return their charges and times.
    
    Args:
        n_events: Number of events to generate
        config_file: Path to detector configuration
        detector_type: Type of detector (Cylinder, Sphere, Box)
        n_photons: Number of photons to simulate
        K: Number of nearest neighbors for sensor mapping
        seed: Random seed
    
    Returns:
        events_data: List of dicts containing charges, times, and true parameters
        detector_bounds: Detector boundary information
        simulate_event: Simulation function
        sensor_params: Sensor parameters
        sensor_positions: Sensor positions
    """
    print(f"Generating {n_events} events...")
    
    # Initialize
    key = jax.random.PRNGKey(seed)
    
    # Setup detector
    detector = generate_detector(config_file)
    sensor_positions = jnp.array(detector.all_points)
    detector_bounds = get_detector_bounds(detector)
    
    # Setup simulation parameters
    sensor_params = (
        jnp.array(50.0),    # scatter_length
        jnp.array(0.1),     # reflection_rate
        jnp.array(100.0),   # absorption_length
        jnp.array(0.001)    # gumbel_softmax_temperature
    )
    
    # Setup event simulator
    simulate_event = setup_event_simulator(
        json_filename=config_file,
        max_sensors_per_cell=4,
        n_photons=n_photons,
        temperature=0.05,
        K=K,
        detector_type=detector_type
    )
    
    # Generate events
    events_data = []
    generation_start = time.time()
    
    # Pre-allocate arrays for charges and times
    n_sensors_estimate = len(sensor_positions)
    all_charges = []
    all_times = []
    all_metadata = []
    
    for i in range(n_events):
        # Generate random event parameters
        key, subkey = jax.random.split(key)
        event_key = jax.random.PRNGKey(seed + i * 1000)
        
        position, direction, energy = generate_random_event_params(event_key, detector_bounds, fraction=0.9)
        
        # Convert to simulation format
        theta = jnp.arccos(jnp.clip(direction[2], -1.0, 1.0))
        phi = jnp.arctan2(direction[1], direction[0])
        direction_angles = jnp.array([theta, phi])
        
        # Simulate event
        particle_params = (energy, position, direction_angles)
        charges, times = simulate_event(particle_params, sensor_params, event_key)
        
        # Store charges and times separately for efficient stacking
        all_charges.append(charges)
        all_times.append(times)
        
        # Store metadata
        all_metadata.append({
            'true_position': position,
            'true_direction': direction,
            'true_energy': energy,
            'event_index': i
        })
        
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - generation_start
            rate = (i + 1) / elapsed
            print(f"  Generated {i + 1}/{n_events} events ({rate:.1f} events/sec)")
    
    # Stack all charges and times once
    print("  Stacking event data...")
    stack_start = time.time()
    all_charges_stacked = jnp.stack(all_charges)
    all_times_stacked = jnp.stack(all_times)
    stack_time = time.time() - stack_start
    
    # Create events_data with pre-stacked arrays
    events_data = {
        'all_charges': all_charges_stacked,
        'all_times': all_times_stacked,
        'metadata': all_metadata
    }
    
    total_gen_time = time.time() - generation_start
    print(f"Generated {n_events} events successfully in {total_gen_time:.1f} seconds ({n_events/total_gen_time:.1f} events/sec)")
    print(f"  Stacking took {stack_time:.3f} seconds")
    
    return events_data, detector_bounds, simulate_event, sensor_params, sensor_positions

@jit
def calculate_loss_to_event(simulated_charges, true_charges, simulated_time, true_time):
    """
    Calculate loss between simulated and true charges and times.
    Loss = sum(|simulated_charge - true_charge|) + sum(|simulated_time - true_time|)
    
    Args:
        simulated_charges: Charges from simulated event
        true_charges: Charges from true event
        simulated_time: Times from simulated event
        true_time: Times from true event
    
    Returns:
        loss: Scalar loss value
    """

    eps = 1e-8
    threshold = 1e-8
    
    # Compute mean times for active locations
    true_active_mask = true_charges > threshold
    sim_active_mask = simulated_charges > threshold

    true_mean_time = jnp.sum(true_time * true_active_mask) / (jnp.sum(true_active_mask) + eps)
    sim_mean_time = jnp.sum(simulated_time * sim_active_mask) / (jnp.sum(sim_active_mask) + eps)

    true_time_centered = jnp.where(true_active_mask, true_time - true_mean_time, 0.0)
    sim_time_centered = jnp.where(sim_active_mask, simulated_time - sim_mean_time, 0.0)

    L_delta_charge = jnp.sum((jnp.abs(simulated_charges - true_charges)))
    L_delta_time = jnp.sum((jnp.abs(sim_time_centered - true_time_centered)))

    return L_delta_charge*L_delta_time


def benchmark_loss_calculation(events_data, detector_bounds, simulate_event, sensor_params, seed=12345, verbose=True):
    """
    Generate a new event and calculate loss to all stored events.
    Time the loss calculation.
    
    Args:
        events_data: List of stored events
        detector_bounds: Detector boundary information
        simulate_event: Simulation function
        sensor_params: Sensor parameters
        seed: Random seed for new event
        verbose: Whether to print detailed output
    
    Returns:
        losses: Array of losses to each event
        timing_info: Dict with timing information
        new_event_params: Dict with new event parameters
    """
    timing_breakdown = {}
    
    if verbose:
        print("\nGenerating new event for loss calculation...")
    
    # Time event generation
    gen_start = time.time()
    
    # Generate new event
    key = jax.random.PRNGKey(seed)
    new_position, new_direction, new_energy = generate_random_event_params(key, detector_bounds, fraction=0.7)
    
    # Convert to simulation format
    theta = jnp.arccos(jnp.clip(new_direction[2], -1.0, 1.0))
    phi = jnp.arctan2(new_direction[1], new_direction[0])
    direction_angles = jnp.array([theta, phi])
    
    gen_time = time.time() - gen_start
    timing_breakdown['param_generation'] = gen_time
    
    # Time event simulation
    sim_start = time.time()
    particle_params = (new_energy, new_position, direction_angles)
    new_charges, new_times = simulate_event(particle_params, sensor_params, key)
    sim_time = time.time() - sim_start
    timing_breakdown['simulation'] = sim_time
    
    if verbose:
        print(f"New event parameters:")
        print(f"  Position: {new_position}")
        print(f"  Direction: {new_direction}")
        print(f"  Energy: {new_energy:.1f} MeV")
        print(f"  Active sensors: {jnp.sum(new_charges > 0)}")
    
    # Get pre-stacked charges and times (no loading needed!)
    if verbose:
        print(f"\nUsing pre-stacked charges and times from {len(events_data['metadata'])} events...")
    
    load_start = time.time()
    # Check if we have the old format or new format
    if isinstance(events_data, dict) and 'all_charges' in events_data:
        # New format - already stacked
        all_true_charges = events_data['all_charges']
        all_true_times = events_data['all_times']
    else:
        # Old format - need to stack (for backward compatibility)
        all_true_charges = jnp.stack([event['charges'] for event in events_data])
        all_true_times = jnp.stack([event['times'] for event in events_data])
    load_time = time.time() - load_start
    timing_breakdown['data_loading'] = load_time
    
    # Time loss calculation
    if verbose:
        print("Calculating losses...")
    
    # Create a vectorized version of the loss function
    # vmap over the second and fourth arguments (true_charges, true_times), keeping new data fixed
    calculate_losses_vmap = vmap(lambda true_charges, true_times: calculate_loss_to_event(new_charges, true_charges, new_times, true_times))
    
    # Warm-up JIT compilation
    warmup_start = time.time()
    _ = calculate_losses_vmap(all_true_charges[:1], all_true_times[:1])
    warmup_time = time.time() - warmup_start
    timing_breakdown['jit_warmup'] = warmup_time
    
    # Time the actual calculation
    start_time = time.time()
    
    # Calculate all losses at once using vmap
    losses = calculate_losses_vmap(all_true_charges, all_true_times)
    
    end_time = time.time()
    calculation_time = end_time - start_time
    timing_breakdown['loss_calculation'] = calculation_time
    
    # Calculate timing statistics
    timing_info = {
        'total_time': calculation_time,
        'time_per_event': calculation_time / len(events_data),
        'events_per_second': len(events_data) / calculation_time if calculation_time > 0 else float('inf'),
        'n_events': len(events_data),
        'breakdown': timing_breakdown
    }
    
    if verbose:
        print(f"\nTiming Results:")
        print(f"  Parameter generation: {timing_breakdown['param_generation']*1000:.3f} ms")
        print(f"  Event simulation: {timing_breakdown['simulation']*1000:.3f} ms")
        print(f"  Data loading (stacking): {timing_breakdown['data_loading']*1000:.3f} ms")
        print(f"  JIT warmup: {timing_breakdown['jit_warmup']*1000:.3f} ms")
        print(f"  Loss calculation: {calculation_time:.6f} seconds")
        print(f"  Time per event: {timing_info['time_per_event']*1000:.3f} ms")
        print(f"  Events per second: {timing_info['events_per_second']:.1f}")
        
        # Find best matching events
        best_indices = jnp.argsort(losses)[:5]
        print(f"\nBest matching events (by loss):")
        for i, idx in enumerate(best_indices):
            # Handle both old and new formats
            if isinstance(events_data, dict) and 'metadata' in events_data:
                event = events_data['metadata'][idx]
            else:
                event = events_data[idx]
            loss = losses[idx]
            
            # Calculate errors
            position_error = float(jnp.linalg.norm(new_position - event['true_position']))
            direction_error = float(jnp.arccos(jnp.clip(jnp.dot(new_direction, event['true_direction']), -1, 1)))
            direction_error_deg = np.degrees(direction_error)
            energy_error = float(jnp.abs(new_energy - event['true_energy']))
            energy_error_percent = (energy_error / float(event['true_energy'])) * 100
            
            print(f"\n  Event {event['event_index']} (rank {i+1}):")
            print(f"    Loss: {loss:.3f}")
            print(f"    Position error: {position_error:.3f} m")
            print(f"    Direction error: {direction_error_deg:.1f}°")
            print(f"    Energy error: {energy_error:.1f} MeV ({energy_error_percent:.1f}%)")
    
    # Store new event parameters for return
    new_event_params = {
        'position': new_position,
        'direction': new_direction,
        'energy': new_energy,
        'charges': new_charges,
        'times': new_times
    }
    
    return losses, timing_info, new_event_params


def analyze_best_events_reconstruction(events_data, losses, new_event_params, max_n=100, verbose=False):
    """
    Analyze how well we can reconstruct the true parameters by averaging
    the N best matching events.
    
    Args:
        events_data: List of all events
        losses: Array of losses for each event
        new_event_params: Parameters of the new event
        max_n: Maximum number of best events to consider
        verbose: Whether to print detailed output
        
    Returns:
        analysis_results: Dict with error metrics for different N values
    """
    if verbose:
        print(f"\nAnalyzing reconstruction accuracy using best matching events...")
    
    # Get indices sorted by loss
    sorted_indices = jnp.argsort(losses)
    
    # True parameters
    true_position = new_event_params['position']
    true_direction = new_event_params['direction']
    true_energy = new_event_params['energy']
    
    # Analyze for different numbers of best events
    n_values = [1, 2, 3, 5, 10, 20, 30, 50, 100]
    # Get the actual number of events - should be based on losses array length
    n_events = len(losses)
    n_values = [n for n in n_values if n <= n_events and n <= max_n]
    
    if verbose:
        print(f"\nAnalyzing with n_values: {n_values} (n_events={n_events}, max_n={max_n})")
    
    analysis_results = {
        'n_values': n_values,
        'position_errors': [],
        'direction_errors': [],
        'energy_errors': [],
        'mean_losses': [],
        'reconstructed_positions': [],
        'reconstructed_directions': [],
        'reconstructed_energies': []
    }
    
    if verbose:
        print(f"\nReconstruction accuracy using N best events:")
        print(f"{'N':>5} | {'Pos Error (m)':>13} | {'Dir Error (°)':>13} | {'Energy Error (MeV)':>18} | {'Mean Loss':>10}")
        print("-" * 75)
    
    for n in n_values:
        # Get the N best events
        best_indices = sorted_indices[:n]
        
        # Handle both old and new formats
        if isinstance(events_data, dict) and 'metadata' in events_data:
            best_events = [events_data['metadata'][idx] for idx in best_indices]
        else:
            best_events = [events_data[idx] for idx in best_indices]
        
        # Average their parameters
        avg_position = jnp.mean(jnp.array([event['true_position'] for event in best_events]), axis=0)
        avg_energy = jnp.mean(jnp.array([event['true_energy'] for event in best_events]))
        
        # For direction, we need to be careful about averaging unit vectors
        directions = jnp.array([event['true_direction'] for event in best_events])
        avg_direction = jnp.mean(directions, axis=0)
        avg_direction = avg_direction / jnp.linalg.norm(avg_direction)  # Renormalize
        
        # Calculate errors
        position_error = float(jnp.linalg.norm(avg_position - true_position))
        direction_error = float(jnp.arccos(jnp.clip(jnp.dot(avg_direction, true_direction), -1, 1)))
        direction_error_deg = np.degrees(direction_error)
        energy_error = float(jnp.abs(avg_energy - true_energy))
        mean_loss = float(jnp.mean(losses[best_indices]))
        
        # Store results
        analysis_results['position_errors'].append(position_error)
        analysis_results['direction_errors'].append(direction_error_deg)
        analysis_results['energy_errors'].append(energy_error)
        analysis_results['mean_losses'].append(mean_loss)
        analysis_results['reconstructed_positions'].append(avg_position)
        analysis_results['reconstructed_directions'].append(avg_direction)
        analysis_results['reconstructed_energies'].append(avg_energy)
        
        if verbose:
            print(f"{n:>5} | {position_error:>13.3f} | {direction_error_deg:>13.1f} | {energy_error:>18.1f} | {mean_loss:>10.3f}")
    
    # Find optimal N
    position_errors_array = jnp.array(analysis_results['position_errors'])
    best_n_idx = jnp.argmin(position_errors_array)
    best_n = n_values[best_n_idx]
    
    if verbose:
        print(f"\nOptimal N for position reconstruction: {best_n} (error = {position_errors_array[best_n_idx]:.3f} m)")
    
    return analysis_results


def plot_error_histograms(aggregated_results, output_dir, n_test_events):
    """
    Plot histograms of reconstruction errors for different N values.
    Also identify and report outliers.
    
    Args:
        aggregated_results: Dict with error data for different N values
        output_dir: Directory to save plots
        n_test_events: Number of test events
    """
    # Create figure with subplots for different N values
    selected_n_values = [1, 2, 3, 5, 10]  # Select a subset for visualization
    fig, axes = plt.subplots(3, len(selected_n_values), figsize=(20, 12))
    fig.suptitle(f'Reconstruction Error Distributions for {n_test_events} Test Events', fontsize=16)
    
    error_types = ['position_errors', 'direction_errors', 'energy_errors']
    error_labels = ['Position Error (m)', 'Direction Error (°)', 'Energy Error (MeV)']
    
    outlier_info = {}
    
    for i, (error_type, label) in enumerate(zip(error_types, error_labels)):
        for j, n in enumerate(selected_n_values):
            ax = axes[i, j]
            
            if n in aggregated_results[error_type]:
                errors = np.array(aggregated_results[error_type][n])
                
                # Calculate statistics
                mean_err = np.mean(errors)
                std_err = np.std(errors)
                median_err = np.median(errors)
                
                # Identify outliers (values beyond 3 standard deviations)
                outlier_threshold = mean_err + 3 * std_err
                outliers = errors[errors > outlier_threshold]
                n_outliers = len(outliers)
                
                # Store outlier info
                if n not in outlier_info:
                    outlier_info[n] = {}
                outlier_info[n][error_type] = {
                    'n_outliers': n_outliers,
                    'outlier_values': outliers,
                    'threshold': outlier_threshold
                }
                
                # Create histogram
                ax.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
                ax.axvline(mean_err, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_err:.3f}')
                ax.axvline(median_err, color='green', linestyle='--', linewidth=2, label=f'Median: {median_err:.3f}')
                
                # Mark outlier threshold
                if n_outliers > 0:
                    ax.axvline(outlier_threshold, color='orange', linestyle=':', linewidth=2, 
                              label=f'3σ threshold ({n_outliers} outliers)')
                
                ax.set_xlabel(label)
                ax.set_ylabel('Count')
                ax.set_title(f'N = {n}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'error_distributions_{n_test_events}tests.png'
    plot_filepath = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nError distribution plots saved to: {plot_filepath}")
    
    # Report outliers
    print("\nOutlier Analysis (values beyond 3σ):")
    print("-" * 60)
    
    for n in selected_n_values:
        if n in outlier_info:
            print(f"\nN = {n}:")
            for error_type, label in zip(error_types, error_labels):
                info = outlier_info[n][error_type]
                if info['n_outliers'] > 0:
                    print(f"  {label}: {info['n_outliers']} outliers (threshold: {info['threshold']:.3f})")
                    print(f"    Outlier values: {info['outlier_values']}")
    
    # Create a box plot for better outlier visualization
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle(f'Error Distributions Box Plots for {n_test_events} Test Events', fontsize=16)
    
    for i, (error_type, label) in enumerate(zip(error_types, error_labels)):
        ax = axes2[i]
        
        # Prepare data for box plot
        data_to_plot = []
        labels_to_plot = []
        
        for n in selected_n_values:
            if n in aggregated_results[error_type]:
                data_to_plot.append(aggregated_results[error_type][n])
                labels_to_plot.append(f'N={n}')
        
        # Create box plot
        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
        
        # Customize box plot
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_ylabel(label)
        ax.set_title(f'{label} Distribution')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save box plot
    boxplot_filename = f'error_boxplots_{n_test_events}tests.png'
    boxplot_filepath = os.path.join(output_dir, boxplot_filename)
    plt.savefig(boxplot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Error box plots saved to: {boxplot_filepath}")
    
    return outlier_info


def save_results(events_data, losses, timing_info, analysis_results, output_dir):
    """Save results to file for later analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"event_loss_benchmark.pkl"
    filepath = os.path.join(output_dir, filename)
    
    results = {
        'n_events': len(events_data),
        'losses': np.array(losses),
        'timing_info': timing_info,
        'analysis_results': analysis_results,
        'timestamp': timestamp
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {filepath}")


def load_or_generate_events(cache_file, n_events, config_file, detector_type, n_photons, K, seed):
    """
    Load events from cache if available, otherwise generate new ones.
    
    Args:
        cache_file: Path to cache file
        Other args: Same as generate_and_store_events
    
    Returns:
        Same as generate_and_store_events
    """
    # Check if cache exists and is valid
    if os.path.exists(cache_file):
        print(f"Loading existing events from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify cache validity
            if (cache_data['n_events'] == n_events and 
                cache_data['config_file'] == config_file and
                cache_data['detector_type'] == detector_type and
                cache_data['n_photons'] == n_photons and
                cache_data['K'] == K and
                cache_data['seed'] == seed):
                
                # Check if it's the new format
                if 'events_data' in cache_data and isinstance(cache_data['events_data'], dict) and 'metadata' in cache_data['events_data']:
                    print(f"Loaded {len(cache_data['events_data']['metadata'])} events from cache (new format)")
                else:
                    # Old format
                    print(f"Loaded {len(cache_data['events_data'])} events from cache")
                
                # Recreate non-pickleable objects
                detector = generate_detector(config_file)
                sensor_positions = jnp.array(detector.all_points)
                detector_bounds = get_detector_bounds(detector)
                
                sensor_params = (
                    jnp.array(50.0),    # scatter_length
                    jnp.array(0.1),     # reflection_rate
                    jnp.array(100.0),   # absorption_length
                    jnp.array(0.001)    # gumbel_softmax_temperature
                )
                
                simulate_event = setup_event_simulator(
                    json_filename=config_file,
                    max_sensors_per_cell=4,
                    n_photons=n_photons,
                    temperature=0.05,
                    K=K,
                    detector_type=detector_type
                )
                
                return (cache_data['events_data'], 
                        detector_bounds,
                        simulate_event,
                        sensor_params,
                        sensor_positions)
            else:
                print("Cache configuration mismatch, regenerating events...")
        except Exception as e:
            print(f"Error loading cache: {e}, regenerating events...")
    
    # Generate new events
    result = generate_and_store_events(n_events, config_file, detector_type, n_photons, K, seed)
    events_data, detector_bounds, simulate_event, sensor_params, sensor_positions = result
    
    # Save only the pickleable data to cache
    print(f"Saving events to cache: {cache_file}")
    cache_data = {
        'events_data': events_data,
        'n_events': n_events,
        'config_file': config_file,
        'detector_type': detector_type,
        'n_photons': n_photons,
        'K': K,
        'seed': seed
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    return result


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(
        description='Initial Guess Benchmark - Analyze event reconstruction using nearest neighbor approach',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings (quiet mode)
    python -m tools.optimization.initial_guess_benchmark
    
    # Run with verbose output
    python -m tools.optimization.initial_guess_benchmark -v
    
    # Run with very verbose output (show all timing details)
    python -m tools.optimization.initial_guess_benchmark -vv
    
    # Quick test with fewer events
    python -m tools.optimization.initial_guess_benchmark -n 1000 -t 10
    
    # Skip plotting
    python -m tools.optimization.initial_guess_benchmark --no-plots
        """
    )
    
    # Verbosity control
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Increase verbosity (use -v for verbose, -vv for very verbose)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Quiet mode - minimal output')
    
    # Configuration parameters
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='Path to detector configuration file')
    parser.add_argument('-d', '--detector', type=str, default='Cylinder',
                        choices=['Cylinder', 'Sphere', 'Box'],
                        help='Detector type (default: Cylinder)')
    parser.add_argument('-n', '--n-events', type=int, default=50000,
                        help='Number of events to generate for database (default: 50000)')
    parser.add_argument('-t', '--n-test-events', type=int, default=100,
                        help='Number of test events to evaluate (default: 100)')
    parser.add_argument('--photons', type=int, default=1_000_000,
                        help='Number of photons per event (default: 1_000_000)')
    parser.add_argument('-K', type=int, default=6,
                        help='Number of nearest neighbors for sensor mapping (default: 6)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # Output control
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating histogram plots')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (default: output/benchmarks)')
    
    args = parser.parse_args()
    
    # Set verbosity level
    if args.quiet:
        verbosity = 0
    else:
        verbosity = args.verbose
    
    # Configuration
    config_file = args.config or (base_dir_path() + 'config/HK_geom_config.json')
    detector_type = args.detector
    n_events = args.n_events
    n_test_events = args.n_test_events
    n_photons = args.photons
    K = args.K
    seed = args.seed
    
    # Create output directory
    output_dir = args.output_dir or os.path.join(base_dir_path(), 'output', 'benchmarks')
    os.makedirs(output_dir, exist_ok=True)
    
    # Define cache file path
    cache_file = os.path.join(output_dir, f'events_cache_{detector_type}_{n_events}_{n_photons}_{K}_{seed}.pkl')
    
    if verbosity >= 1:
        print(f"Initial Guess Benchmark")
        print(f"======================")
        print(f"Configuration:")
        print(f"  Detector: {detector_type}")
        print(f"  Number of events: {n_events}")
        print(f"  Photons per event: {n_photons}")
        print(f"  K nearest neighbors: {K}")
        print(f"  Test events: {n_test_events}")
        print(f"  Verbosity level: {verbosity}")
        print()
    
    # Load or generate events
    events_data, detector_bounds, simulate_event, sensor_params, sensor_positions = load_or_generate_events(
        cache_file, n_events, config_file, detector_type, n_photons, K, seed
    )
    
    if verbosity >= 1:
        print(f"\nTesting reconstruction on {n_test_events} events...")
    
    # Initialize aggregated results
    n_values = [1, 2, 3, 5, 10, 20, 30, 50, 100]
    aggregated_results = {
        'n_values': n_values,
        'position_errors': {n: [] for n in n_values},
        'direction_errors': {n: [] for n in n_values},
        'energy_errors': {n: [] for n in n_values},
        'timing_info': [],
        'event_data': []  # Store event data for outlier analysis with N=5
    }
    
    # Test multiple events
    for test_idx in range(n_test_events):
        if verbosity >= 1:
            print(f"\rProcessing test event {test_idx + 1}/{n_test_events}", end='', flush=True)
        
        # Use different seed for each test event
        test_seed = seed + 100000 + test_idx
        
        # Benchmark loss calculation for this test event
        # Use verbose=True only if verbosity >= 2
        losses, timing_info, new_event_params = benchmark_loss_calculation(
            events_data, detector_bounds, simulate_event, sensor_params, 
            seed=test_seed, verbose=(verbosity >= 2)
        )
        
        aggregated_results['timing_info'].append(timing_info['time_per_event'])
        
        # Analyze reconstruction accuracy for this event
        analysis_results = analyze_best_events_reconstruction(
            events_data, losses, new_event_params, max_n=100, verbose=(verbosity >= 2)
        )
        
        # Aggregate results
        for i, n in enumerate(analysis_results['n_values']):
            if n in aggregated_results['position_errors']:
                aggregated_results['position_errors'][n].append(analysis_results['position_errors'][i])
                aggregated_results['direction_errors'][n].append(analysis_results['direction_errors'][i])
                aggregated_results['energy_errors'][n].append(analysis_results['energy_errors'][i])
        
        # Store event data for N=5 outlier analysis
        if 5 in analysis_results['n_values']:
            n5_idx = analysis_results['n_values'].index(5)
            aggregated_results['event_data'].append({
                'test_idx': test_idx,
                'true_position': np.array(new_event_params['position']),
                'true_direction': np.array(new_event_params['direction']), 
                'true_energy': float(new_event_params['energy']),
                'reco_position': np.array(analysis_results['reconstructed_positions'][n5_idx]),
                'reco_direction': np.array(analysis_results['reconstructed_directions'][n5_idx]),
                'reco_energy': float(analysis_results['reconstructed_energies'][n5_idx]),
                'position_error': analysis_results['position_errors'][n5_idx],
                'direction_error': analysis_results['direction_errors'][n5_idx],
                'energy_error': analysis_results['energy_errors'][n5_idx]
            })
    
    if verbosity >= 1:
        print()  # New line after progress indicator
    
    # Always show final results
    print(f"\n{'='*80}")
    print(f"AGGREGATED RESULTS OVER {n_test_events} TEST EVENTS")
    print(f"{'='*80}")
    
    print(f"\nTiming Performance:")
    mean_time = np.mean(aggregated_results['timing_info']) * 1000
    std_time = np.std(aggregated_results['timing_info']) * 1000
    print(f"  Mean time per event comparison: {mean_time:.3f} ± {std_time:.3f} ms")
    print(f"  Mean events per second: {1000/mean_time:.1f}")
    
    print(f"\nReconstruction Accuracy (Mean ± Std):")
    print(f"{'N':>5} | {'Pos Error (m)':>20} | {'Dir Error (°)':>20} | {'Energy Error (MeV)':>20}")
    print("-" * 90)
    
    best_n_for_position = None
    best_position_error = float('inf')
    
    for n in n_values:
        if n in aggregated_results['position_errors'] and aggregated_results['position_errors'][n]:
            pos_errors = aggregated_results['position_errors'][n]
            dir_errors = aggregated_results['direction_errors'][n]
            energy_errors = aggregated_results['energy_errors'][n]
            
            mean_pos = np.mean(pos_errors)
            std_pos = np.std(pos_errors)
            mean_dir = np.mean(dir_errors)
            std_dir = np.std(dir_errors)
            mean_energy = np.mean(energy_errors)
            std_energy = np.std(energy_errors)
            
            print(f"{n:>5} | {mean_pos:>8.3f} ± {std_pos:<8.3f} | {mean_dir:>8.1f} ± {std_dir:<8.1f} | {mean_energy:>8.1f} ± {std_energy:<8.1f}")
            
            if mean_pos < best_position_error:
                best_position_error = mean_pos
                best_n_for_position = n
    
    print(f"\nOptimal N for position reconstruction: {best_n_for_position} (mean error = {best_position_error:.3f} m)")
    
    # Analyze outliers for N=5
    if 5 in aggregated_results['direction_errors'] and aggregated_results['event_data']:
        print(f"\nAnalyzing outliers for N=5 reconstruction...")
        
        # Calculate outlier threshold (3 standard deviations) using direction errors
        dir_errors = np.array(aggregated_results['direction_errors'][5])
        mean_dir_error = np.mean(dir_errors)
        std_dir_error = np.std(dir_errors)
        outlier_threshold = mean_dir_error + 3 * std_dir_error
        
        # Find outlier events
        outlier_events = []
        for event_data in aggregated_results['event_data']:
            if event_data['direction_error'] > outlier_threshold:
                outlier_events.append(event_data)
        
        print(f"Found {len(outlier_events)} outlier events (direction error > {outlier_threshold:.1f}°)")
        print(f"Outlier threshold: mean + 3σ = {mean_dir_error:.1f}° + 3×{std_dir_error:.1f}° = {outlier_threshold:.1f}°")
        
        # Print details for each outlier
        if outlier_events:
            print(f"\nOutlier Event Details (N=5 reconstruction):")
            print("=" * 100)
            for i, event in enumerate(outlier_events):
                print(f"\nEvent {event['test_idx']} (outlier #{i+1}):")
                print(f"  Position Error: {event['position_error']:.3f} m")
                print(f"  Direction Error: {event['direction_error']:.1f}°")
                print(f"  Energy Error: {event['energy_error']:.1f} MeV")
                print(f"  True Parameters:")
                print(f"    Position: [{event['true_position'][0]:.3f}, {event['true_position'][1]:.3f}, {event['true_position'][2]:.3f}] m")
                print(f"    Direction: [{event['true_direction'][0]:.3f}, {event['true_direction'][1]:.3f}, {event['true_direction'][2]:.3f}]")
                print(f"    Energy: {event['true_energy']:.1f} MeV")
                print(f"  Reconstructed Parameters (N=5 average):")
                print(f"    Position: [{event['reco_position'][0]:.3f}, {event['reco_position'][1]:.3f}, {event['reco_position'][2]:.3f}] m")
                print(f"    Direction: [{event['reco_direction'][0]:.3f}, {event['reco_direction'][1]:.3f}, {event['reco_direction'][2]:.3f}]")
                print(f"    Energy: {event['reco_energy']:.1f} MeV")
    else:
        print('NO OUTLIER DATA')
    # Plot error histograms and identify outliers (unless disabled)
    if not args.no_plots:
        outlier_info = plot_error_histograms(aggregated_results, output_dir, n_test_events)
    else:
        outlier_info = None
    
    # Save aggregated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"event_loss_benchmark_aggregated_{n_test_events}tests.pkl"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump({
            'n_test_events': n_test_events,
            'aggregated_results': aggregated_results,
            'outlier_info': outlier_info,
            'timestamp': timestamp
        }, f)
    
    print(f"\nAggregated results saved to: {filepath}")


if __name__ == "__main__":
    main()