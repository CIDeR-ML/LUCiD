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
from datetime import datetime
from jax import jit
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
    for i in range(n_events):
        # Generate random event parameters
        key, subkey = jax.random.split(key)
        event_key = jax.random.PRNGKey(seed + i * 1000)
        
        position, direction, energy = generate_random_event_params(event_key, detector_bounds)
        
        # Convert to simulation format
        theta = jnp.arccos(jnp.clip(direction[2], -1.0, 1.0))
        phi = jnp.arctan2(direction[1], direction[0])
        direction_angles = jnp.array([theta, phi])
        
        # Simulate event
        particle_params = (energy, position, direction_angles)
        charges, times = simulate_event(particle_params, sensor_params, event_key)
        
        # Store event data
        events_data.append({
            'charges': charges,
            'times': times,
            'true_position': position,
            'true_direction': direction,
            'true_energy': energy,
            'event_index': i
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{n_events} events")
    
    print(f"Generated {n_events} events successfully")
    
    return events_data, detector_bounds, simulate_event, sensor_params, sensor_positions

@jit
def calculate_loss_to_event(simulated_charges, true_charges):
    """
    Calculate loss between simulated and true charges.
    Loss = sum(|simulated_charge - true_charge|)
    
    Args:
        simulated_charges: Charges from simulated event
        true_charges: Charges from true event
    
    Returns:
        loss: Scalar loss value
    """
    return jnp.sum(jnp.abs(simulated_charges - true_charges))


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
    if verbose:
        print("\nGenerating new event for loss calculation...")
    
    # Generate new event
    key = jax.random.PRNGKey(seed)
    new_position, new_direction, new_energy = generate_random_event_params(key, detector_bounds)
    
    # Convert to simulation format
    theta = jnp.arccos(jnp.clip(new_direction[2], -1.0, 1.0))
    phi = jnp.arctan2(new_direction[1], new_direction[0])
    direction_angles = jnp.array([theta, phi])
    
    # Simulate new event
    particle_params = (new_energy, new_position, direction_angles)
    new_charges, new_times = simulate_event(particle_params, sensor_params, key)
    
    if verbose:
        print(f"New event parameters:")
        print(f"  Position: {new_position}")
        print(f"  Direction: {new_direction}")
        print(f"  Energy: {new_energy:.1f} MeV")
        print(f"  Active sensors: {jnp.sum(new_charges > 0)}")
    
    # Load all event charges at once
    if verbose:
        print(f"\nLoading charges from {len(events_data)} events...")
    all_true_charges = jnp.stack([event['charges'] for event in events_data])
    
    # Time loss calculation
    if verbose:
        print("Calculating losses...")
    
    # Warm-up JIT compilation
    _ = calculate_loss_to_event(new_charges, all_true_charges[0])
    
    # Time the actual calculation
    start_time = time.time()
    
    # Calculate loss to each event
    losses = []
    for i in range(len(events_data)):
        loss = calculate_loss_to_event(new_charges, all_true_charges[i])
        losses.append(loss)
    
    end_time = time.time()
    calculation_time = end_time - start_time
    
    losses = jnp.array(losses)
    
    # Calculate timing statistics
    timing_info = {
        'total_time': calculation_time,
        'time_per_event': calculation_time / len(events_data),
        'events_per_second': len(events_data) / calculation_time,
        'n_events': len(events_data)
    }
    
    if verbose:
        print(f"\nTiming Results:")
        print(f"  Total calculation time: {calculation_time:.6f} seconds")
        print(f"  Time per event: {timing_info['time_per_event']*1000:.3f} ms")
        print(f"  Events per second: {timing_info['events_per_second']:.1f}")
        
        # Find best matching events
        best_indices = jnp.argsort(losses)[:5]
        print(f"\nBest matching events (by loss):")
        for i, idx in enumerate(best_indices):
            event = events_data[idx]
            loss = losses[idx]
            
            # Calculate errors
            position_error = float(jnp.linalg.norm(new_position - event['true_position']))
            direction_error = float(jnp.arccos(jnp.clip(jnp.abs(jnp.dot(new_direction, event['true_direction'])), 0, 1)))
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
    n_values = [n for n in n_values if n <= len(events_data) and n <= max_n]
    
    analysis_results = {
        'n_values': n_values,
        'position_errors': [],
        'direction_errors': [],
        'energy_errors': [],
        'mean_losses': []
    }
    
    if verbose:
        print(f"\nReconstruction accuracy using N best events:")
        print(f"{'N':>5} | {'Pos Error (m)':>13} | {'Dir Error (°)':>13} | {'Energy Error (MeV)':>18} | {'Mean Loss':>10}")
        print("-" * 75)
    
    for n in n_values:
        # Get the N best events
        best_indices = sorted_indices[:n]
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
        direction_error = float(jnp.arccos(jnp.clip(jnp.abs(jnp.dot(avg_direction, true_direction)), 0, 1)))
        direction_error_deg = np.degrees(direction_error)
        energy_error = float(jnp.abs(avg_energy - true_energy))
        mean_loss = float(jnp.mean(losses[best_indices]))
        
        # Store results
        analysis_results['position_errors'].append(position_error)
        analysis_results['direction_errors'].append(direction_error_deg)
        analysis_results['energy_errors'].append(energy_error)
        analysis_results['mean_losses'].append(mean_loss)
        
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
    selected_n_values = [1, 5, 10, 20, 50]  # Select a subset for visualization
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
    plot_filename = f'error_distributions_{n_test_events}tests_{timestamp}.png'
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
    boxplot_filename = f'error_boxplots_{n_test_events}tests_{timestamp}.png'
    boxplot_filepath = os.path.join(output_dir, boxplot_filename)
    plt.savefig(boxplot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Error box plots saved to: {boxplot_filepath}")
    
    return outlier_info


def save_results(events_data, losses, timing_info, analysis_results, output_dir):
    """Save results to file for later analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"event_loss_benchmark_{timestamp}.pkl"
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
    # Configuration
    config_file = base_dir_path() + 'config/IWCD_geom_config.json'
    detector_type = 'Cylinder'
    n_events = 10000  # Number of events to generate and store
    n_test_events = 100  # Number of test events to evaluate
    n_photons = 100_000
    K = 6
    seed = 42
    
    # Create output directory
    output_dir = os.path.join(base_dir_path(), 'output', 'benchmarks')
    os.makedirs(output_dir, exist_ok=True)
    
    # Define cache file path
    cache_file = os.path.join(output_dir, f'events_cache_{detector_type}_{n_events}_{n_photons}_{K}_{seed}.pkl')
    
    print(f"Event Loss Benchmark")
    print(f"===================")
    print(f"Configuration:")
    print(f"  Detector: {detector_type}")
    print(f"  Number of events: {n_events}")
    print(f"  Photons per event: {n_photons}")
    print(f"  K nearest neighbors: {K}")
    print()
    
    # Load or generate events
    events_data, detector_bounds, simulate_event, sensor_params, sensor_positions = load_or_generate_events(
        cache_file, n_events, config_file, detector_type, n_photons, K, seed
    )
    
    print(f"\nTesting reconstruction on {n_test_events} events...")
    
    # Initialize aggregated results
    n_values = [1, 2, 3, 5, 10, 20, 30, 50, 100]
    aggregated_results = {
        'n_values': n_values,
        'position_errors': {n: [] for n in n_values},
        'direction_errors': {n: [] for n in n_values},
        'energy_errors': {n: [] for n in n_values},
        'timing_info': []
    }
    
    # Test multiple events
    for test_idx in range(n_test_events):
        print(f"\rProcessing test event {test_idx + 1}/{n_test_events}", end='', flush=True)
        
        # Use different seed for each test event
        test_seed = seed + 100000 + test_idx
        
        # Benchmark loss calculation for this test event
        losses, timing_info, new_event_params = benchmark_loss_calculation(
            events_data, detector_bounds, simulate_event, sensor_params, seed=test_seed, verbose=False
        )
        
        aggregated_results['timing_info'].append(timing_info['time_per_event'])
        
        # Analyze reconstruction accuracy for this event
        analysis_results = analyze_best_events_reconstruction(
            events_data, losses, new_event_params, max_n=100, verbose=False
        )
        
        # Aggregate results
        for i, n in enumerate(analysis_results['n_values']):
            if n in aggregated_results['position_errors']:
                aggregated_results['position_errors'][n].append(analysis_results['position_errors'][i])
                aggregated_results['direction_errors'][n].append(analysis_results['direction_errors'][i])
                aggregated_results['energy_errors'][n].append(analysis_results['energy_errors'][i])
    
    print()  # New line after progress indicator
    
    # Calculate and display aggregated statistics
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
    
    # Plot error histograms and identify outliers
    outlier_info = plot_error_histograms(aggregated_results, output_dir, n_test_events)
    
    # Save aggregated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"event_loss_benchmark_aggregated_{n_test_events}tests_{timestamp}.pkl"
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