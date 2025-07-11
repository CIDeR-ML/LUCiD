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
                   random_seed=42, verbose=True):
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
    """
    key = jax.random.PRNGKey(random_seed)
    n_elite = int(population_size * elite_fraction)
    
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
        
        if verbose:
            # Calculate errors for best candidate this iteration
            best_pos_error = float(jnp.linalg.norm(population[0]['position'] - true_position))
            best_dir_error = float(jnp.arccos(jnp.clip(jnp.abs(jnp.dot(population[0]['direction'], true_direction)), 0, 1)))
            best_dir_error_deg = np.degrees(best_dir_error)
            best_energy_error = float(jnp.abs(population[0]['energy'] - true_energy))
            
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
    
    return best_overall, population


def main(verbose=True):
    """Run adaptive search for best parameters."""
    
    # Configuration
    random_seed = 43
    config_file = base_dir_path() + 'config/IWCD_geom_config.json'
    
    # Initialize random key
    key = jax.random.PRNGKey(random_seed)
    
    # Setup detector
    print(f"Loading detector configuration...")
    detector = generate_detector(config_file)
    sensor_positions = jnp.array(detector.all_points)
    n_sensors = len(sensor_positions)
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
    print("Setting up event simulator...")
    simulate_event = setup_event_simulator(
        json_filename=config_file,
        max_sensors_per_cell=4,
        n_photons=1_000_000,
        temperature=0.0,
        K=2,
        detector_type='Cylinder'
    )
    
    # Generate TRUE event
    print("\nGenerating TRUE event...")
    key, subkey = jax.random.split(key)
    
    # Random vertex position
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
    
    print(f"True event parameters:")
    print(f"  Position: {true_position}")
    print(f"  Direction: {true_direction}")
    print(f"  Energy: {true_energy:.1f} MeV")
    print(f"  Active sensors: {jnp.sum(true_charges > 0)}")
    
    # Run adaptive search
    print(f"\nStarting adaptive search...")
    search_start_time = datetime.now()
    best_match, final_population = adaptive_search(
        true_charges, true_times, simulate_event, sensor_params, sensor_positions,
        detector_bounds, true_position, true_direction, true_energy,
        n_iterations=20, population_size=20, elite_fraction=0.2, verbose=verbose
    )
    search_duration = (datetime.now() - search_start_time).total_seconds()
    print(f"Adaptive search completed in {search_duration:.1f} seconds")
    
    # Calculate errors
    if best_match:
        position_error = float(jnp.linalg.norm(best_match['position'] - true_position))
        direction_error = float(jnp.arccos(jnp.clip(jnp.abs(jnp.dot(best_match['direction'], true_direction)), 0, 1)))
        direction_error_deg = np.degrees(direction_error)
        energy_error = float(jnp.abs(best_match['energy'] - true_energy))
        energy_error_percent = (energy_error / float(true_energy)) * 100
        
        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"Best match found with loss: {best_match['loss']:.6f}")
        print(f"\nBest match parameters:")
        print(f"  Position: {best_match['position']}")
        print(f"  Direction: {best_match['direction']}")
        print(f"  Energy: {best_match['energy']:.1f} MeV")
        print(f"\nErrors:")
        print(f"  Position error: {position_error:.3f} m")
        print(f"  Direction error: {direction_error_deg:.1f}°")
        print(f"  Energy error: {energy_error:.1f} MeV ({energy_error_percent:.1f}%)")
        
        # Test with exact parameters
        print(f"\nTesting with EXACT true parameters...")
        true_loss = compute_softmin_loss(
            sensor_positions,
            true_charges,
            true_times,
            true_charges,
            true_times,
            tau=0.01,
            lambda_time=1.0,
            lambda_intensity=1.0
        )
        print(f"Loss at true parameters (should be 0): {float(true_loss):.6f}")
        
        # Show top 5 from final population
        print(f"\nTop 5 candidates from final population:")
        for i in range(min(5, len(final_population))):
            print(f"  {i+1}. Loss: {final_population[i]['loss']:.6f}")
        
        # Create visualization
        print(f"\nCreating visualization...")
        
        # Filter hits with charge > 30
        min_charge = 30.0
        significant_mask = true_charges > min_charge
        hit_positions = sensor_positions[significant_mask]
        hit_charges_vis = true_charges[significant_mask]
        hit_times_vis = true_times[significant_mask]
        
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
        ax.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.05, color='gray')
        
        # Set labels and title
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(f'Adaptive Search Results\n' +
                    f'E Err: {energy_error:.1f}MeV ({energy_error_percent:.1f}%), ' +
                    f'Pos Err: {position_error:.2f}m, Dir Err: {direction_error_deg:.1f}°\n' +
                    f'Best Loss: {best_match["loss"]:.6f}',
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'adaptive_search_result_{timestamp}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adaptive search for LUCiD reconstruction')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Show detailed progress during iterations')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress iteration details (opposite of verbose)')
    
    args = parser.parse_args()
    
    # Determine verbosity (quiet takes precedence)
    verbose = True
    if args.quiet:
        verbose = False
    elif args.verbose:
        verbose = True
    
    main(verbose=verbose)