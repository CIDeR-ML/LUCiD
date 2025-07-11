"""
Visualization functions for LUCiD optimization results.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from .utils import (
    create_cylinder_surface, create_sphere_surface, create_box_surface,
    compute_cone_cylinder_intersection, compute_cone_sphere_intersection, 
    compute_cone_box_intersection, get_cherenkov_angle
)


def create_event_visualization(true_position, true_direction, true_energy, best_match, 
                              true_charges, true_times, sensor_positions, detector_bounds,
                              position_error, direction_error_deg, energy_error, energy_error_percent,
                              event_idx=0, figures_dir=None, verbose=True, config_file=None):
    """Create visualization for a single event."""
    # Extract detector name from config file path
    detector_name = "Unknown"
    if config_file:
        # Extract detector name from config filename
        # e.g., "config/IWCD_geom_config.json" -> "IWCD"
        config_basename = os.path.basename(config_file)
        if '_geom_config.json' in config_basename:
            detector_name = config_basename.replace('_geom_config.json', '')
        elif '.json' in config_basename:
            detector_name = config_basename.replace('.json', '')
    
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
    
    # Detector-specific visualization
    if detector_bounds['type'] == 'cylinder':
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
        
    elif detector_bounds['type'] == 'sphere':
        # True cone intersection
        true_intersection = compute_cone_sphere_intersection(
            true_position, true_direction, cherenkov_angle, detector_bounds['r']
        )
        if len(true_intersection) > 0:
            ax.plot(true_intersection[:, 0], true_intersection[:, 1], true_intersection[:, 2],
                   'cyan', linewidth=3, alpha=0.8, label='True Cherenkov Ring', zorder=25)
        
        # Fitted cone intersection
        fitted_intersection = compute_cone_sphere_intersection(
            best_match['position'], best_match['direction'], cherenkov_angle, detector_bounds['r']
        )
        if len(fitted_intersection) > 0:
            ax.plot(fitted_intersection[:, 0], fitted_intersection[:, 1], fitted_intersection[:, 2],
                   'orange', linewidth=3, alpha=0.8, label='Fitted Cherenkov Ring', zorder=25)
        
        # Add sphere boundaries
        x_sph, y_sph, z_sph = create_sphere_surface(detector_bounds['r'])
        ax.plot_surface(x_sph, y_sph, z_sph, alpha=0.15, color='gray')
        
    elif detector_bounds['type'] == 'box':
        # True cone intersection
        true_intersection = compute_cone_box_intersection(
            true_position, true_direction, cherenkov_angle,
            detector_bounds['x'], detector_bounds['y'], detector_bounds['z']
        )
        if len(true_intersection) > 0:
            # For box, we might get multiple segments
            for segment in true_intersection:
                ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                       'cyan', linewidth=3, alpha=0.8, zorder=25)
        
        # Fitted cone intersection
        fitted_intersection = compute_cone_box_intersection(
            best_match['position'], best_match['direction'], cherenkov_angle,
            detector_bounds['x'], detector_bounds['y'], detector_bounds['z']
        )
        if len(fitted_intersection) > 0:
            for segment in fitted_intersection:
                ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                       'orange', linewidth=3, alpha=0.8, zorder=25)
        
        # Add box boundaries
        vertices, edges = create_box_surface(detector_bounds['x'], detector_bounds['y'], detector_bounds['z'])
        for edge in edges:
            ax.plot3D(*vertices[edge].T, 'gray', alpha=0.3)
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'{detector_name} Detector - Event {event_idx + 1}\n' +
                f'Energy Error: {energy_error:.1f} MeV, ' +
                f'Position Error: {position_error:.2f} m, ' +
                f'Direction Error: {direction_error_deg:.1f}°',
                fontsize=14)
    
    # Add colorbar and legend
    plt.colorbar(scatter, ax=ax, label='Hit Time', shrink=0.6)
    ax.legend(loc='upper right', fontsize=10)
    
    # Set axis limits based on detector type with equal aspect ratio
    if detector_bounds['type'] == 'cylinder':
        max_extent = max(detector_bounds['r'], detector_bounds['H']/2) * 1.2
        ax.set_xlim([-max_extent, max_extent])
        ax.set_ylim([-max_extent, max_extent])
        ax.set_zlim([-max_extent, max_extent])
    elif detector_bounds['type'] == 'sphere':
        max_extent = detector_bounds['r'] * 1.2
        ax.set_xlim([-max_extent, max_extent])
        ax.set_ylim([-max_extent, max_extent])
        ax.set_zlim([-max_extent, max_extent])
    elif detector_bounds['type'] == 'box':
        max_extent = max(detector_bounds['x']/2, detector_bounds['y']/2, detector_bounds['z']/2) * 1.2
        ax.set_xlim([-max_extent, max_extent])
        ax.set_ylim([-max_extent, max_extent])
        ax.set_zlim([-max_extent, max_extent])
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    
    # Save figure
    output_file = f'{detector_name}_adaptive_search_event_{event_idx + 1:03d}.png'
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


def create_convergence_plots(event_histories, figures_dir=None, show_individual=True, show_statistics=True, config_file=None):
    """
    Create comprehensive visualization of multi-event optimization convergence.
    Shows parameter errors evolution during iterations.
    """
    # Extract detector name from config file path
    detector_name = "Unknown"
    if config_file:
        config_basename = os.path.basename(config_file)
        if '_geom_config.json' in config_basename:
            detector_name = config_basename.replace('_geom_config.json', '')
        elif '.json' in config_basename:
            detector_name = config_basename.replace('.json', '')
    
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
    output_file = f'{detector_name}_adaptive_search_convergence.png'
    if figures_dir:
        output_file = os.path.join(figures_dir, output_file)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Convergence plots saved to {output_file}")
    plt.close()


def create_summary_plots(results, figures_dir=None, config_file=None):
    """Create summary histogram plots for multiple events."""
    # Extract detector name from config file path
    detector_name = "Unknown"
    if config_file:
        config_basename = os.path.basename(config_file)
        if '_geom_config.json' in config_basename:
            detector_name = config_basename.replace('_geom_config.json', '')
        elif '.json' in config_basename:
            detector_name = config_basename.replace('.json', '')
    
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
    output_file = f'{detector_name}_adaptive_search_summary.png'
    if figures_dir:
        output_file = os.path.join(figures_dir, output_file)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSummary plots saved to {output_file}")
    plt.close()