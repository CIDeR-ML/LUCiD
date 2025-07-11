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

# Add parent directories to path to access tools
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.utils import base_dir_path, spherical_to_cartesian
from tools.geometry import generate_detector
from tools.simulation import setup_event_simulator
from tools.losses import compute_softmin_loss


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
                   detector_bounds, n_iterations=20, population_size=50, elite_fraction=0.2,
                   random_seed=42):
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
    """
    key = jax.random.PRNGKey(random_seed)
    n_elite = int(population_size * elite_fraction)
    
    # Initialize with random population
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
        
        print(f"  Best loss this iteration: {population[0]['loss']:.6f}")
        print(f"  Best overall loss: {best_overall_loss:.6f}")
        print(f"  Population loss range: [{population[0]['loss']:.6f}, {population[-1]['loss']:.6f}]")
        
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


def main():
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
        temperature=0.2,
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
    best_match, final_population = adaptive_search(
        true_charges, true_times, simulate_event, sensor_params, sensor_positions,
        detector_bounds, n_iterations=100, population_size=20, elite_fraction=0.2
    )
    
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


if __name__ == "__main__":
    main()