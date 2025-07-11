#!/usr/bin/env python3
"""
Adaptive search for best parameters using LUCiD loss landscape.
Uses a population-based approach that concentrates search around low-loss regions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# Add parent directories to path to access tools
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.utils import base_dir_path, spherical_to_cartesian
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.losses import compute_softmin_loss

# Import cone-cylinder utilities  
from cone_cylinder_utils import get_cherenkov_angle
from visualize_cone_cylinder_intersection import compute_cone_cylinder_intersection


def create_cylinder_surface(radius, height, center=(0, 0, 0), n_points=20):
    """Create a transparent cylinder surface for detector boundaries."""
    theta = np.linspace(0, 2*np.pi, n_points)
    z = np.linspace(-height/2, height/2, n_points)
    theta_mesh, z_mesh = np.meshgrid(theta, z)
    
    x_mesh = center[0] + radius * np.cos(theta_mesh)
    y_mesh = center[1] + radius * np.sin(theta_mesh)
    z_mesh = center[2] + z_mesh
    
    return x_mesh, y_mesh, z_mesh


def sample_around_point(key, center_position, center_direction, center_energy, 
                       position_std=0.5, direction_std=0.1, energy_std=50.0,
                       detector_bounds=None):
    """
    Sample new parameters around a center point with Gaussian noise.
    """
    # Sample position
    key, subkey = jax.random.split(key)
    position_noise = jax.random.normal(subkey, shape=(3,)) * position_std
    new_position = center_position + position_noise
    
    # Ensure position is within bounds
    if detector_bounds:
        r = jnp.sqrt(new_position[0]**2 + new_position[1]**2)
        if r > detector_bounds['r'] * 0.8:
            new_position = new_position * (detector_bounds['r'] * 0.8 / r)
        new_position = jnp.clip(new_position, 
                               jnp.array([-detector_bounds['r'], -detector_bounds['r'], -detector_bounds['H']/2]) * 0.8,
                               jnp.array([detector_bounds['r'], detector_bounds['r'], detector_bounds['H']/2]) * 0.8)
    
    # Sample direction (add noise and renormalize)
    key, subkey = jax.random.split(key)
    direction_noise = jax.random.normal(subkey, shape=(3,)) * direction_std
    new_direction = center_direction + direction_noise
    new_direction = new_direction / jnp.linalg.norm(new_direction)
    
    # Sample energy
    key, subkey = jax.random.split(key)
    energy_noise = jax.random.normal(subkey) * energy_std
    new_energy = jnp.clip(center_energy + energy_noise, 250.0, 900.0)
    
    return new_position, new_direction, new_energy


def adaptive_search(true_charges, true_times, simulate_event, sensor_params, sensor_positions,
                   detector_bounds, true_position, true_direction, true_energy,
                   n_iterations=20, population_size=50, elite_fraction=0.2,
                   random_seed=42, verbose=True, track_history=False):
    """
    Adaptive search algorithm that focuses on promising regions.
    
    Parameters:
    -----------
    n_iterations : int
        Number of iterations
    population_size : int
        Number of candidates per iteration
    elite_fraction : float
        Fraction of best candidates to keep for next generation
    verbose : bool
        Whether to print detailed progress during iterations
    track_history : bool
        Whether to track parameter evolution history
    """
    key = jax.random.PRNGKey(random_seed)
    n_elite = int(population_size * elite_fraction)
    
    # Initialize history tracking
    history = {
        'best_loss': [],
        'best_energy': [],
        'best_position': [],
        'best_direction': [],
        'position_error': [],
        'direction_error': [],
        'energy_error': []
    } if track_history else None
    
    # Initialize with random population
    if verbose:
        print("Initializing random population...")
    population = []
    
    for i in range(population_size):
        key, subkey = jax.random.split(key)
        
        # Random position
        r_vert = jax.random.uniform(subkey, shape=(), minval=0, maxval=detector_bounds['r'] * 0.8)
        subkey, _ = jax.random.split(subkey)
        theta = jax.random.uniform(subkey, shape=(), minval=0, maxval=2*jnp.pi)
        subkey, _ = jax.random.split(subkey)
        z_vert = jax.random.uniform(subkey, shape=(), minval=-detector_bounds['H']/2 * 0.8, 
                                   maxval=detector_bounds['H']/2 * 0.8)
        position = jnp.array([r_vert * jnp.cos(theta), r_vert * jnp.sin(theta), z_vert])
        
        # Random direction
        subkey, _ = jax.random.split(subkey)
        phi = jax.random.uniform(subkey, shape=(), minval=0, maxval=2*jnp.pi)
        subkey, _ = jax.random.split(subkey)
        cos_theta = jax.random.uniform(subkey, shape=(), minval=-1, maxval=1)
        sin_theta = jnp.sqrt(1 - cos_theta**2)
        direction = jnp.array([sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), cos_theta])
        
        # Random energy
        subkey, _ = jax.random.split(subkey)
        energy = jax.random.uniform(subkey, shape=(), minval=250.0, maxval=900.0)
        
        population.append({
            'position': position,
            'direction': direction,
            'energy': energy,
            'loss': float('inf')
        })
    
    # Evolution loop
    best_overall = None
    best_overall_loss = float('inf')
    
    for iteration in range(n_iterations):
        if verbose:
            print(f"\nIteration {iteration + 1}/{n_iterations}")
        
        # Evaluate population
        for i, candidate in enumerate(population):
            # Convert to simulation format
            theta = jnp.arccos(candidate['direction'][2])
            phi = jnp.arctan2(candidate['direction'][1], candidate['direction'][0])
            direction_angles = jnp.array([theta, phi])
            
            particle_params = (candidate['energy'], candidate['position'], direction_angles)
            test_charges, test_times = simulate_event(particle_params, sensor_params, key)
            
            # Calculate loss
            loss = compute_softmin_loss(
                sensor_positions,
                true_charges,
                true_times,
                test_charges,
                test_times,
                tau=0.01,
                lambda_time=1.0,
                lambda_intensity=1.0
            )
            
            candidate['loss'] = float(loss)
            key, _ = jax.random.split(key)
        
        # Sort by loss
        population.sort(key=lambda x: x['loss'])
        
        # Update best overall
        if population[0]['loss'] < best_overall_loss:
            best_overall = population[0].copy()
            best_overall_loss = population[0]['loss']
        
        # Calculate errors for best candidate this iteration (for both verbose and history tracking)
        best_pos_error = float(jnp.linalg.norm(population[0]['position'] - true_position))
        best_dir_error = float(jnp.arccos(jnp.clip(jnp.abs(jnp.dot(population[0]['direction'], true_direction)), 0, 1)))
        best_dir_error_deg = np.degrees(best_dir_error)
        best_energy_error = float(jnp.abs(population[0]['energy'] - true_energy))
        
        # Track history if requested
        if track_history and history is not None:
            history['best_loss'].append(population[0]['loss'])
            history['best_energy'].append(population[0]['energy'])
            history['best_position'].append(np.array(population[0]['position']))
            history['best_direction'].append(np.array(population[0]['direction']))
            history['position_error'].append(best_pos_error)
            history['direction_error'].append(best_dir_error_deg)
            history['energy_error'].append(best_energy_error)
        
        if verbose:
            print(f"  Best loss this iteration: {population[0]['loss']:.6f}")
            print(f"  Best overall loss: {best_overall_loss:.6f}")
            print(f"  Population loss range: [{population[0]['loss']:.6f}, {population[-1]['loss']:.6f}]")
            print(f"  Best candidate errors: Pos={best_pos_error:.3f}m, Dir={best_dir_error_deg:.1f}°, Energy={best_energy_error:.1f}MeV")
        
        # Create next generation
        if iteration < n_iterations - 1:  # Don't create new generation on last iteration
            new_population = []
            
            # Keep elite
            elite = population[:n_elite]
            new_population.extend([e.copy() for e in elite])
            
            # Generate new candidates around elite
            remaining_slots = population_size - n_elite
            
            # Adaptive standard deviations (decrease over iterations)
            progress = (iteration + 1) / n_iterations
            position_std = 0.5 * (1 - 0.7 * progress)  # Start at 0.5, end at 0.15
            direction_std = 0.2 * (1 - 0.7 * progress)  # Start at 0.2, end at 0.06
            energy_std = 100.0 * (1 - 0.7 * progress)  # Start at 100, end at 30
            
            for i in range(remaining_slots):
                # Select parent from elite (with bias towards better ones)
                parent_idx = min(int(abs(jax.random.normal(key)) * n_elite / 3), n_elite - 1)
                parent = elite[parent_idx]
                key, _ = jax.random.split(key)
                
                # Generate offspring
                new_position, new_direction, new_energy = sample_around_point(
                    key, parent['position'], parent['direction'], parent['energy'],
                    position_std, direction_std, energy_std, detector_bounds
                )
                
                new_population.append({
                    'position': new_position,
                    'direction': new_direction,
                    'energy': new_energy,
                    'loss': float('inf')
                })
                
                key, _ = jax.random.split(key)
            
            population = new_population
    
    if track_history:
        return best_overall, population, history
    else:
        return best_overall, population


def create_event_visualization(true_position, true_direction, true_energy, best_match, 
                              true_charges, true_times, sensor_positions, detector_bounds,
                              position_error, direction_error_deg, energy_error, energy_error_percent,
                              event_idx=0, figures_dir=None, verbose=True):
    """Create visualization for a single event."""
    # Filter hits with charge > 30
    min_charge = 30.0
    significant_mask = true_charges > min_charge
    hit_positions = sensor_positions[significant_mask]
    hit_charges_vis = true_charges[significant_mask]
    hit_times_vis = true_times[significant_mask]
    
    if verbose:
        print(f"Creating visualization for event {event_idx + 1}...")
        print(f"Visualizing {len(hit_positions)} hits with charge > {min_charge}")
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot detector hits (color by time, size by charge)
    scatter = ax.scatter(hit_positions[:, 0], hit_positions[:, 1], hit_positions[:, 2], 
                        c=hit_times_vis, s=hit_charges_vis*2, cmap='plasma', alpha=0.6,
                        label='Detector Hits (colored by time)')
    
    # Plot true track
    t_vals = np.linspace(0, 8, 100)
    true_track_points = true_position[:, np.newaxis] + t_vals[np.newaxis, :] * true_direction[:, np.newaxis]
    ax.plot(true_track_points[0], true_track_points[1], true_track_points[2], 
            'blue', linewidth=5, label='True Track', zorder=30)
    ax.scatter(true_position[0], true_position[1], true_position[2], 
               c='blue', s=400, marker='*', edgecolors='black', linewidth=3, 
               label='True Origin', zorder=35)
    
    # Plot best fitted track
    fitted_track_points = best_match['position'][:, np.newaxis] + t_vals[np.newaxis, :] * best_match['direction'][:, np.newaxis]
    ax.plot(fitted_track_points[0], fitted_track_points[1], fitted_track_points[2], 
            'red', linewidth=5, label='Best Fitted Track', zorder=30)
    ax.scatter(best_match['position'][0], best_match['position'][1], best_match['position'][2], 
               c='red', s=400, marker='*', edgecolors='black', linewidth=3, 
               label='Best Fitted Origin', zorder=35)
    
    # Add Cherenkov cone intersections
    cherenkov_angle = get_cherenkov_angle(1.33)  # Water refractive index
    
    # True cone intersection
    true_intersection = compute_cone_cylinder_intersection(
        true_position, true_direction, cherenkov_angle,
        detector_bounds['r'], detector_bounds['H']
    )
    if len(true_intersection) > 0:
        ax.plot(true_intersection[:, 0], true_intersection[:, 1], true_intersection[:, 2],
               'cyan', linewidth=3, alpha=0.8, label='True Cherenkov Ring', zorder=25)
    
    # Fitted cone intersection
    fitted_intersection = compute_cone_cylinder_intersection(
        best_match['position'], best_match['direction'], cherenkov_angle,
        detector_bounds['r'], detector_bounds['H']
    )
    if len(fitted_intersection) > 0:
        ax.plot(fitted_intersection[:, 0], fitted_intersection[:, 1], fitted_intersection[:, 2],
               'orange', linewidth=3, alpha=0.8, label='Fitted Cherenkov Ring', zorder=25)
    
    # Add cylinder boundaries
    x_cyl, y_cyl, z_cyl = create_cylinder_surface(detector_bounds['r'], detector_bounds['H'])
    ax.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.15, color='gray')
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'Adaptive Search Results - Event {event_idx + 1}\n' +
                f'Energy Error: {energy_error:.1f} MeV, ' +
                f'Position Error: {position_error:.2f} m, ' +
                f'Direction Error: {direction_error_deg:.1f}°',
                fontsize=14)
    
    # Add colorbar and legend
    plt.colorbar(scatter, ax=ax, label='Hit Time', shrink=0.6)
    ax.legend(loc='upper right', fontsize=10)
    
    # Set axis limits
    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    ax.set_zlim([-5, 5])
    
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    
    # Save figure
    output_file = f'adaptive_search_event_{event_idx + 1:03d}.png'
    if figures_dir:
        output_file = os.path.join(figures_dir, output_file)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"Visualization saved to {output_file}")
    plt.close()


def print_summary_statistics(results, total_search_time):
    """Print summary statistics for multiple events."""
    successful_results = [r for r in results if r['success']]
    n_successful = len(successful_results)
    n_total = len(results)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS ({n_successful}/{n_total} successful)")
    print(f"{'='*80}")
    
    if n_successful > 0:
        position_errors = [r['position_error'] for r in successful_results]
        direction_errors = [r['direction_error_deg'] for r in successful_results]
        energy_errors = [r['energy_error_percent'] for r in successful_results]
        final_losses = [r['final_loss'] for r in successful_results]
        
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
    
    print(f"\nTiming:")
    print(f"  Total search time: {total_search_time:.1f} seconds")
    print(f"  Average time per event: {total_search_time/n_total:.1f} seconds")
    print(f"  Success rate: {n_successful/n_total*100:.1f}%")


def create_convergence_plots(event_histories, figures_dir=None, show_individual=True, show_statistics=True):
    """
    Create comprehensive visualization of multi-event optimization convergence.
    Shows parameter errors evolution during iterations.
    """
    if not event_histories:
        print("No convergence histories to plot.")
        return
    
    N_events = len(event_histories)
    n_iterations = len(event_histories[0]['position_error'])
    
    # Create figure with subplots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Energy Error
    if show_individual:
        for i in range(N_events):
            ax1.plot(event_histories[i]['energy_error'], alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_energy_errors = np.array([h['energy_error'] for h in event_histories])
        mean_energy_error = np.mean(all_energy_errors, axis=0)
        std_energy_error = np.std(all_energy_errors, axis=0)
        median_energy_error = np.median(all_energy_errors, axis=0)
        
        iterations = range(n_iterations)
        ax1.plot(iterations, mean_energy_error, 'r-', linewidth=2, label=f'Mean (N={N_events})')
        ax1.fill_between(iterations, mean_energy_error - std_energy_error, 
                        mean_energy_error + std_energy_error, alpha=0.2, color='red', label='±1σ')
        ax1.plot(iterations, median_energy_error, 'g--', linewidth=2, label='Median')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Energy Error (MeV)')
    ax1.set_title('Energy Error Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Position Error
    if show_individual:
        for i in range(N_events):
            ax2.plot(event_histories[i]['position_error'], alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_position_errors = np.array([h['position_error'] for h in event_histories])
        mean_position_error = np.mean(all_position_errors, axis=0)
        std_position_error = np.std(all_position_errors, axis=0)
        median_position_error = np.median(all_position_errors, axis=0)
        
        ax2.plot(iterations, mean_position_error, 'r-', linewidth=2, label='Mean')
        ax2.fill_between(iterations, mean_position_error - std_position_error, 
                        mean_position_error + std_position_error, alpha=0.2, color='red')
        ax2.plot(iterations, median_position_error, 'g--', linewidth=2, label='Median')
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Position Error (m)')
    ax2.set_title('Position Error Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Direction Error
    if show_individual:
        for i in range(N_events):
            ax3.plot(event_histories[i]['direction_error'], alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_direction_errors = np.array([h['direction_error'] for h in event_histories])
        mean_direction_error = np.mean(all_direction_errors, axis=0)
        std_direction_error = np.std(all_direction_errors, axis=0)
        median_direction_error = np.median(all_direction_errors, axis=0)
        
        ax3.plot(iterations, mean_direction_error, 'r-', linewidth=2, label='Mean')
        ax3.fill_between(iterations, mean_direction_error - std_direction_error, 
                        mean_direction_error + std_direction_error, alpha=0.2, color='red')
        ax3.plot(iterations, median_direction_error, 'g--', linewidth=2, label='Median')
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Direction Error (degrees)')
    ax3.set_title('Direction Error Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Best Loss Evolution
    if show_individual:
        for i in range(N_events):
            ax4.plot(event_histories[i]['best_loss'], alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_losses = np.array([h['best_loss'] for h in event_histories])
        mean_loss = np.mean(all_losses, axis=0)
        std_loss = np.std(all_losses, axis=0)
        median_loss = np.median(all_losses, axis=0)
        
        ax4.plot(iterations, mean_loss, 'r-', linewidth=2, label='Mean')
        ax4.fill_between(iterations, mean_loss - std_loss, 
                        mean_loss + std_loss, alpha=0.2, color='red')
        ax4.plot(iterations, median_loss, 'g--', linewidth=2, label='Median')
    
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Best Loss')
    ax4.set_title('Loss Function Convergence')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Energy Evolution
    if show_individual:
        for i in range(N_events):
            ax5.plot(event_histories[i]['best_energy'], alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_energies = np.array([h['best_energy'] for h in event_histories])
        mean_energy = np.mean(all_energies, axis=0)
        std_energy = np.std(all_energies, axis=0)
        median_energy = np.median(all_energies, axis=0)
        
        ax5.plot(iterations, mean_energy, 'r-', linewidth=2, label='Mean')
        ax5.fill_between(iterations, mean_energy - std_energy, 
                        mean_energy + std_energy, alpha=0.2, color='red')
        ax5.plot(iterations, median_energy, 'g--', linewidth=2, label='Median')
    
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Best Energy (MeV)')
    ax5.set_title('Energy Parameter Evolution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Combined Error Metric
    if show_individual:
        for i in range(N_events):
            # Normalized combined error (position/5 + direction/90 + energy/500)
            combined_error = [
                p/5.0 + d/90.0 + e/500.0 
                for p, d, e in zip(event_histories[i]['position_error'], 
                                 event_histories[i]['direction_error'],
                                 event_histories[i]['energy_error'])
            ]
            ax6.plot(combined_error, alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_combined_errors = []
        for i in range(N_events):
            combined_error = [
                p/5.0 + d/90.0 + e/500.0 
                for p, d, e in zip(event_histories[i]['position_error'], 
                                 event_histories[i]['direction_error'],
                                 event_histories[i]['energy_error'])
            ]
            all_combined_errors.append(combined_error)
        
        combined_error_array = np.array(all_combined_errors)
        mean_combined = np.mean(combined_error_array, axis=0)
        std_combined = np.std(combined_error_array, axis=0)
        median_combined = np.median(combined_error_array, axis=0)
        
        ax6.plot(iterations, mean_combined, 'r-', linewidth=2, label='Mean')
        ax6.fill_between(iterations, mean_combined - std_combined, 
                        mean_combined + std_combined, alpha=0.2, color='red')
        ax6.plot(iterations, median_combined, 'g--', linewidth=2, label='Median')
    
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Normalized Combined Error')
    ax6.set_title('Combined Error Convergence')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'adaptive_search_convergence.png'
    if figures_dir:
        output_file = os.path.join(figures_dir, output_file)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Convergence plots saved to {output_file}")
    plt.close()


def create_summary_plots(results, figures_dir=None):
    """Create summary histogram plots for multiple events."""
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) == 0:
        print("No successful results to plot.")
        return
    
    # Extract data
    position_errors = [r['position_error'] for r in successful_results]
    direction_errors = [r['direction_error_deg'] for r in successful_results]
    energy_errors = [r['energy_error_percent'] for r in successful_results]
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Position error histogram
    ax1.hist(position_errors, bins=max(5, len(position_errors)//3), alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Position Error (m)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Position Error Distribution\n(μ={np.mean(position_errors):.3f}±{np.std(position_errors):.3f}m)')
    ax1.grid(True, alpha=0.3)
    
    # Direction error histogram
    ax2.hist(direction_errors, bins=max(5, len(direction_errors)//3), alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Direction Error (degrees)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Direction Error Distribution\n(μ={np.mean(direction_errors):.1f}±{np.std(direction_errors):.1f}°)')
    ax2.grid(True, alpha=0.3)
    
    # Energy error histogram
    ax3.hist(energy_errors, bins=max(5, len(energy_errors)//3), alpha=0.7, color='red', edgecolor='black')
    ax3.set_xlabel('Energy Error (%)')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Energy Error Distribution\n(μ={np.mean(energy_errors):.1f}±{np.std(energy_errors):.1f}%)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'adaptive_search_summary.png'
    if figures_dir:
        output_file = os.path.join(figures_dir, output_file)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSummary plots saved to {output_file}")
    plt.close()


def main(verbose=True, n_events=1, save_event_plots=False):
    """Run adaptive search for best parameters on N events."""
    
    # Configuration
    random_seed = 43
    config_file = base_dir_path() + 'config/IWCD_geom_config.json'
    
    # Create figures directory
    figures_dir = os.path.join(base_dir_path(), 'scripts', 'single_ring_initial_guess_development', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Initialize random key
    key = jax.random.PRNGKey(random_seed)
    
    # Setup detector
    if verbose or n_events == 1:
        print(f"Loading detector configuration...")
    detector = generate_detector(config_file)
    sensor_positions = jnp.array(detector.all_points)
    n_sensors = len(sensor_positions)
    if verbose or n_events == 1:
        print(f"Detector has {n_sensors} sensors")
    
    detector_bounds = {
        'r': detector.r,
        'H': detector.H
    }
    
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
        detector_type='Cylinder'
    )
    
    # Initialize statistics collection
    results = []
    event_histories = []
    total_search_time = 0.0
    
    if verbose:
        print(f"\nProcessing {n_events} events...")
    else:
        print(f"\nProcessing {n_events} events...")
    
    for event_idx in range(n_events):
        if verbose:
            print(f"\n{'='*80}")
            print(f"EVENT {event_idx + 1}/{n_events}")
            print(f"{'='*80}")
        elif n_events > 1:
            # Progress bar for quiet mode
            progress = (event_idx + 1) / n_events
            bar_length = 50
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f'\rProgress: |{bar}| {event_idx + 1}/{n_events} ({progress*100:.1f}%)', end='', flush=True)
        
        # Generate TRUE event
        key, subkey = jax.random.split(key)
        
        # Random vertex position (within detector bounds)
        r_vert = jax.random.uniform(subkey, shape=(), minval=0, maxval=detector.r * 0.8)
        subkey, _ = jax.random.split(subkey)
        theta = jax.random.uniform(subkey, shape=(), minval=0, maxval=2*jnp.pi)
        subkey, _ = jax.random.split(subkey)
        z_vert = jax.random.uniform(subkey, shape=(), minval=-detector.H/2 * 0.8, maxval=detector.H/2 * 0.8)
        true_position = jnp.array([r_vert * jnp.cos(theta), r_vert * jnp.sin(theta), z_vert])
        
        # Random direction
        subkey, _ = jax.random.split(subkey)
        phi = jax.random.uniform(subkey, shape=(), minval=0, maxval=2*jnp.pi)
        subkey, _ = jax.random.split(subkey)
        cos_theta = jax.random.uniform(subkey, shape=(), minval=-1, maxval=1)
        sin_theta = jnp.sqrt(1 - cos_theta**2)
        true_direction = jnp.array([sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), cos_theta])
        
        # Random energy
        subkey, _ = jax.random.split(subkey)
        true_energy = jax.random.uniform(subkey, shape=(), minval=250.0, maxval=900.0)
        
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
            n_iterations=40, population_size=20, elite_fraction=0.2, 
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
    
    # Add newline after progress bar in quiet mode
    if not verbose and n_events > 1:
        print()  # New line after progress bar
    
    # Print summary statistics and create plots
    if n_events > 1:
        print_summary_statistics(results, total_search_time)
        create_summary_plots(results, figures_dir)
        
        # Create convergence plots if we have event histories
        if event_histories:
            create_convergence_plots(event_histories, figures_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adaptive search for LUCiD reconstruction')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Show detailed progress during iterations')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress iteration details (opposite of verbose)')
    parser.add_argument('--events', '-n', type=int, default=1,
                        help='Number of events to process (default: 1)')
    parser.add_argument('--save-event-plots', action='store_true',
                        help='Save visualization plots for individual events')
    
    args = parser.parse_args()
    
    # Determine verbosity (quiet takes precedence)
    verbose = True
    if args.quiet:
        verbose = False
    elif args.verbose:
        verbose = True
    
    main(verbose=verbose, n_events=args.events, save_event_plots=args.save_event_plots)