"""
Optimization algorithms for LUCiD parameter reconstruction.
Contains both numerical and gradient-based optimization methods.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from datetime import datetime
from functools import partial
from jax import jit, value_and_grad

from .losses import energy_loss_fn, spatial_loss_fn, combined_loss_fn, compute_gradients
from ..utils import spherical_to_cartesian
import time


def apply_detector_bounds(position, energy, detector_bounds):
    """
    Apply detector bounds to position and energy parameters.
    
    Args:
        position: JAX array of shape (3,) representing position
        energy: Scalar energy value
        detector_bounds: Dictionary with detector type and dimensions
        
    Returns:
        Tuple of (bounded_position, bounded_energy)
    """
    # Apply energy bounds
    bounded_energy = jnp.clip(energy, 250.0, 950.0)
    
    # Apply position bounds based on detector type
    if detector_bounds is None:
        return position, bounded_energy
        
    if detector_bounds['type'] == 'cylinder':
        # For cylinder: check radial distance and height
        r = jnp.sqrt(position[0]**2 + position[1]**2)
        r_max = detector_bounds['r'] * 0.9
        
        # Scale down if outside radial bounds
        if r > r_max:
            position = position * (r_max / r)
        
        # Clip height
        position = jnp.clip(position, 
                           jnp.array([-detector_bounds['r'], -detector_bounds['r'], -detector_bounds['H']/2]) * 0.9,
                           jnp.array([detector_bounds['r'], detector_bounds['r'], detector_bounds['H']/2]) * 0.9)
                           
    elif detector_bounds['type'] == 'sphere':
        # For sphere: check radial distance
        r = jnp.linalg.norm(position)
        r_max = detector_bounds['r'] * 0.9
        
        # Scale down if outside bounds
        if r > r_max:
            position = position * (r_max / r)
            
    elif detector_bounds['type'] == 'box':
        # For box: clip each dimension
        position = jnp.clip(position,
                           jnp.array([-detector_bounds['x']/2, -detector_bounds['y']/2, -detector_bounds['z']/2]) * 0.9,
                           jnp.array([detector_bounds['x']/2, detector_bounds['y']/2, detector_bounds['z']/2]) * 0.9)
    
    return position, bounded_energy

def sample_around_point(key, center_position, center_direction, center_energy, 
                       position_std=0.5, direction_std=10.0, energy_std=50.0,
                       detector_bounds=None):
    """
    Sample new parameters around a center point with Gaussian noise.

    """
    # Sample position
    key, subkey = jax.random.split(key)
    position_noise = jax.random.normal(subkey, shape=(3,)) * position_std
    new_position = center_position + position_noise
    
    # Sample direction using angle-based approach (like gradient optimization)
    key, subkey = jax.random.split(key)
    
    # Convert current direction to spherical angles
    current_theta = jnp.arccos(jnp.clip(center_direction[2], -1.0, 1.0))
    current_phi = jnp.arctan2(center_direction[1], center_direction[0])
    
    # Add angular noise (in radians) - use direction_std as degrees and convert
    angular_std_rad = direction_std * jnp.pi / 180.0  # Convert from degrees to radians
    theta_noise = jax.random.normal(subkey, shape=()) * angular_std_rad #/2 # divide by two because theta range is smaller (0 to pi)
    key, subkey = jax.random.split(key)
    phi_noise = jax.random.normal(subkey, shape=()) * angular_std_rad
    
    # Apply angular noise
    new_theta = current_theta + theta_noise
    new_phi = current_phi + phi_noise
    
    # Convert back to Cartesian direction vector
    sin_theta = jnp.sin(new_theta)
    cos_theta = jnp.cos(new_theta)
    sin_phi = jnp.sin(new_phi)
    cos_phi = jnp.cos(new_phi)
    
    new_direction = jnp.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])
    
    # Sample energy
    key, subkey = jax.random.split(key)
    energy_noise = jax.random.normal(subkey) * energy_std
    new_energy = center_energy + energy_noise
    
    # Apply bounds to position and energy
    new_position, new_energy = apply_detector_bounds(new_position, new_energy, detector_bounds)
    
    return new_position, new_direction, new_energy


def crossover_candidates(parent1, parent2, key, crossover_type='uniform'):
    """
    Perform crossover between two parent candidates to create offspring.
    
    Args:
        parent1, parent2: Parent candidates with 'position', 'direction', 'energy'
        key: JAX random key
        crossover_type: Type of crossover ('uniform', 'intermediate', 'single_point')
    
    Returns:
        Two offspring candidates
    """
    if crossover_type == 'uniform':
        # Uniform crossover - each parameter has 50% chance from each parent
        key, subkey = jax.random.split(key)
        position_mask = jax.random.bernoulli(subkey, p=0.5, shape=(3,))
        offspring1_position = jnp.where(position_mask, parent1['position'], parent2['position'])
        offspring2_position = jnp.where(position_mask, parent2['position'], parent1['position'])
        
        key, subkey = jax.random.split(key)
        energy_mask = jax.random.bernoulli(subkey, p=0.5)
        offspring1_energy = jnp.where(energy_mask, parent1['energy'], parent2['energy'])
        offspring2_energy = jnp.where(energy_mask, parent2['energy'], parent1['energy'])
        
        # For direction, use spherical linear interpolation (slerp) or just pick one
        key, subkey = jax.random.split(key)
        direction_mask = jax.random.bernoulli(subkey, p=0.5)
        offspring1_direction = jnp.where(direction_mask, parent1['direction'], parent2['direction'])
        offspring2_direction = jnp.where(direction_mask, parent2['direction'], parent1['direction'])
        
    elif crossover_type == 'intermediate':
        # Intermediate crossover - offspring are between parents
        key, subkey = jax.random.split(key)
        alpha = jax.random.uniform(subkey, shape=(3,), minval=0.0, maxval=1.0)
        offspring1_position = alpha * parent1['position'] + (1 - alpha) * parent2['position']
        offspring2_position = alpha * parent2['position'] + (1 - alpha) * parent1['position']
        
        key, subkey = jax.random.split(key)
        alpha_energy = jax.random.uniform(subkey)
        offspring1_energy = alpha_energy * parent1['energy'] + (1 - alpha_energy) * parent2['energy']
        offspring2_energy = alpha_energy * parent2['energy'] + (1 - alpha_energy) * parent1['energy']
        
        # For direction, use spherical linear interpolation
        key, subkey = jax.random.split(key)
        alpha_dir = jax.random.uniform(subkey)
        offspring1_direction = spherical_lerp(parent1['direction'], parent2['direction'], alpha_dir)
        offspring2_direction = spherical_lerp(parent2['direction'], parent1['direction'], alpha_dir)
        
    elif crossover_type == 'single_point':
        # Single point crossover
        key, subkey = jax.random.split(key)
        cut_point = jax.random.randint(subkey, shape=(), minval=0, maxval=3)
        
        # For simplicity, treat as: [energy, position_x, position_y, position_z, direction]
        # and do crossover at the cut point
        if cut_point == 0:
            # Cut before energy - swap everything
            offspring1 = parent2.copy()
            offspring2 = parent1.copy()
            return offspring1, offspring2
        elif cut_point == 1:
            # Cut after energy
            offspring1_energy = parent1['energy']
            offspring2_energy = parent2['energy']
            offspring1_position = parent2['position']
            offspring2_position = parent1['position']
            offspring1_direction = parent2['direction']
            offspring2_direction = parent1['direction']
        else:
            # Cut within position
            offspring1_energy = parent1['energy']
            offspring2_energy = parent2['energy']
            offspring1_position = jnp.concatenate([parent1['position'][:cut_point-1], parent2['position'][cut_point-1:]])
            offspring2_position = jnp.concatenate([parent2['position'][:cut_point-1], parent1['position'][cut_point-1:]])
            offspring1_direction = parent2['direction']
            offspring2_direction = parent1['direction']
    
    # Normalize directions
    offspring1_direction = offspring1_direction / jnp.linalg.norm(offspring1_direction)
    offspring2_direction = offspring2_direction / jnp.linalg.norm(offspring2_direction)
    
    offspring1 = {
        'position': offspring1_position,
        'direction': offspring1_direction,
        'energy': offspring1_energy,
        'loss': float('inf')
    }
    
    offspring2 = {
        'position': offspring2_position,
        'direction': offspring2_direction,
        'energy': offspring2_energy,
        'loss': float('inf')
    }
    
    return offspring1, offspring2


def spherical_lerp(dir1, dir2, alpha):
    """
    Spherical linear interpolation between two direction vectors.
    """
    # Compute angle between vectors
    dot = jnp.clip(jnp.dot(dir1, dir2), -1.0, 1.0)
    theta = jnp.arccos(dot)
    
    # If vectors are nearly parallel, just return linear interpolation
    sin_theta = jnp.sin(theta)
    
    # Use conditional to handle near-parallel case
    def slerp_calculation():
        a = jnp.sin((1 - alpha) * theta) / sin_theta
        b = jnp.sin(alpha * theta) / sin_theta
        return a * dir1 + b * dir2
    
    def linear_calculation():
        return (1 - alpha) * dir1 + alpha * dir2
    
    result = jax.lax.cond(
        sin_theta > 1e-6,
        slerp_calculation,
        linear_calculation
    )
    
    return result / jnp.linalg.norm(result)


def create_initial_population_database(config_file, detector_type, K, true_charges, true_times, population_size, N=3, verbose=False, events_cache=None):
    """
    Create initial population using pre-generated event database.
    
    When numerical_iterations > 0, use N=3 averaging to find best candidates.
    Otherwise, use single best match.
    
    Args:
        config_file: Detector configuration file
        detector_type: Type of detector
        K: Number of nearest neighbors
        true_charges: True event charges
        true_times: True event times
        population_size: Size of population to return
        N: Number of best matches to average (use 3 when numerical_iterations=0)
        verbose: Print debug info
        
    Returns:
        list: Population of best candidates
    """
    from .event_cache import load_event_cache
    from .losses import initial_guess_loss
    from jax import vmap
    
    # Use pre-loaded cache if provided, otherwise load it
    if events_cache is not None:
        events_data = events_cache
        if verbose:
            print(f"Using pre-loaded cache")
    else:
        events_data = load_event_cache(config_file, detector_type, K, verbose=True)
    
    # Pre-stacked arrays for efficient computation
    all_charges_stacked = events_data['all_charges']
    all_times_stacked = events_data['all_times']
    metadata = events_data['metadata']
    
    if verbose:
        print(f"Using database with {len(metadata)} events for initial guess")
    
    # Vectorized loss calculation
    # We want to vmap over the cached events (axis 0 of stacked arrays)
    calculate_losses_vmap = vmap(
        lambda cached_charges, cached_times: initial_guess_loss(cached_charges, true_charges, cached_times, true_times),
        in_axes=(0, 0)
    )
    
    # Calculate losses
    losses = calculate_losses_vmap(all_charges_stacked, all_times_stacked)
    
    # Get indices of N best matches
    best_indices = jnp.argsort(losses)[:N]
    
    if N > 1:
        # Average the N best candidates
        best_positions = jnp.array([metadata[int(idx)]['position'] for idx in best_indices])
        best_directions = jnp.array([metadata[int(idx)]['direction'] for idx in best_indices])
        best_energies = jnp.array([metadata[int(idx)]['energy'] for idx in best_indices])
        
        avg_position = jnp.mean(best_positions, axis=0)
        avg_direction = jnp.mean(best_directions, axis=0)
        avg_direction = avg_direction / jnp.linalg.norm(avg_direction)  # Normalize
        avg_energy = jnp.mean(best_energies)
        
        if verbose:
            print(f"Averaged top {N} matches:")
            print(f"  Position: [{avg_position[0]:.3f}, {avg_position[1]:.3f}, {avg_position[2]:.3f}]")
            print(f"  Direction: [{avg_direction[0]:.3f}, {avg_direction[1]:.3f}, {avg_direction[2]:.3f}]")
            print(f"  Energy: {avg_energy:.1f} MeV")
            print(f"  Average loss: {jnp.mean(losses[best_indices]):.6f}")
    else:
        # Use single best match
        best_idx = int(best_indices[0])
        avg_position = jnp.array(metadata[best_idx]['position'])
        avg_direction = jnp.array(metadata[best_idx]['direction'])
        avg_energy = metadata[best_idx]['energy']
        
        if verbose:
            print(f"Using single best match (index {best_idx}):")
            print(f"  Position: [{avg_position[0]:.3f}, {avg_position[1]:.3f}, {avg_position[2]:.3f}]")
            print(f"  Direction: [{avg_direction[0]:.3f}, {avg_direction[1]:.3f}, {avg_direction[2]:.3f}]")
            print(f"  Energy: {avg_energy:.1f} MeV")
            print(f"  Loss: {losses[best_idx]:.6f}")
    
    # Create population around the best candidate(s)
    population = []
    
    # Add the best candidate itself
    best_candidate = {
        'position': avg_position,
        'direction': avg_direction,
        'energy': avg_energy,
        'loss': float(jnp.mean(losses[best_indices]))
    }
    population.append(best_candidate)
    
    # Fill remaining population slots with variations around the best candidate
    # Use the existing sample_around_point function for consistency
    key = jax.random.PRNGKey(12345)
    
    for _ in range(population_size - 1):
        key, subkey = jax.random.split(key)
        
        # Sample around the best candidate with small variations
        new_position, new_direction, new_energy = sample_around_point(
            subkey, avg_position, avg_direction, avg_energy,
            position_std=0.1,    # Small variations
            direction_std=5.0,   # Small angular variations (degrees)
            energy_std=25.0,     # Small energy variations
            detector_bounds=events_data['config']['detector_bounds']
        )
        
        candidate = {
            'position': new_position,
            'direction': new_direction,
            'energy': new_energy,
            'loss': float('inf')  # Will be evaluated in evolve_population
        }
        population.append(candidate)
    
    return population


def create_initial_population(key, detector_bounds, population_size, loss_function, true_charges, true_times):
    """
    Legacy function for backwards compatibility.
    This generates a random population as before.
    """
    population = []
    for i in range(population_size*10):
        key, subkey = jax.random.split(key)
        
        # Generate random parameters based on detector type
        if detector_bounds['type'] == 'cylinder':
            # Random position for cylinder
            r_vert = jax.random.uniform(subkey, shape=(), minval=0, maxval=detector_bounds['r'] * 0.8)
            subkey, _ = jax.random.split(subkey)
            theta = jax.random.uniform(subkey, shape=(), minval=0, maxval=2*jnp.pi)
            subkey, _ = jax.random.split(subkey)
            z_vert = jax.random.uniform(subkey, shape=(), minval=-detector_bounds['H']/2 * 0.8, 
                                       maxval=detector_bounds['H']/2 * 0.8)
            position = jnp.array([r_vert * jnp.cos(theta), r_vert * jnp.sin(theta), z_vert])
        elif detector_bounds['type'] == 'sphere':
            # Random position for sphere (uniform in volume)
            subkey, _ = jax.random.split(subkey)
            u = jax.random.uniform(subkey, shape=())
            subkey, _ = jax.random.split(subkey)
            cos_theta = jax.random.uniform(subkey, shape=(), minval=-1, maxval=1)
            subkey, _ = jax.random.split(subkey)
            phi = jax.random.uniform(subkey, shape=(), minval=0, maxval=2*jnp.pi)
            
            r = detector_bounds['r'] * 0.8 * jnp.cbrt(u)  # cbrt for uniform volume sampling
            sin_theta = jnp.sqrt(1 - cos_theta**2)
            position = jnp.array([r * sin_theta * jnp.cos(phi), 
                                 r * sin_theta * jnp.sin(phi), 
                                 r * cos_theta])
        elif detector_bounds['type'] == 'box':
            # Random position for box
            subkey, _ = jax.random.split(subkey)
            position = jax.random.uniform(subkey, shape=(3,), 
                                        minval=jnp.array([-detector_bounds['x']/2, -detector_bounds['y']/2, -detector_bounds['z']/2]) * 0.8,
                                        maxval=jnp.array([detector_bounds['x']/2, detector_bounds['y']/2, detector_bounds['z']/2]) * 0.8)
        
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
            'energy': 800.,
        })

        theta = jnp.arccos(population[-1]['direction'][2])
        phi = jnp.arctan2(population[-1]['direction'][1], population[-1]['direction'][0])
        direction_angles = jnp.array([theta, phi])
        particle_params = (population[-1]['energy'], population[-1]['position'], direction_angles)
        loss = loss_function(particle_params, true_charges, true_times, key)
        #print(particle_params, loss)
        population[-1]['loss'] = float(loss)

    population.sort(key=lambda x: x['loss'])
    print('BEST LOSS: ', population[0])

    return population[:population_size]

def evolve_population(key, iterations, population, n_elite, loss_function, detector_bounds, true_values, history, verbose, numerical_debug, crossover_rate, mutation_after_crossover=0.5):

    best_overall = None
    best_overall_loss = float('inf')
    (true_charges, true_times, true_energy, true_position, true_direction) = true_values
    
    for iteration in range(iterations):
        if verbose:
            print(f"\nIteration {iteration + 1}/{iterations}")
        
        # Evaluate population
        for i, candidate in enumerate(population):
            # Convert to simulation format
            theta = jnp.arccos(candidate['direction'][2])
            phi = jnp.arctan2(candidate['direction'][1], candidate['direction'][0])
            direction_angles = jnp.array([theta, phi])
            particle_params = (candidate['energy'], candidate['position'], direction_angles)
            loss = loss_function(particle_params, true_charges, true_times, key)
            candidate['loss'] = float(loss)
            candidate['ranking_score'] = candidate['loss']

            key, _ = jax.random.split(key)
            

        # Sort by ranking score (which uses weighted loss for combined loss type)
        population.sort(key=lambda x: x['ranking_score'])
        
        # Update best overall (using ranking score for comparison)
        if not best_overall:
            best_overall = population[0].copy()
            best_overall_loss = population[0]['loss']
        elif population[0]['ranking_score'] < best_overall.get('ranking_score', float('inf')):
            best_overall = population[0].copy()
            best_overall_loss = population[0]['loss']

            if numerical_debug:
                print(f"    New Best Loss: {best_overall_loss:.3f}")
                print(f"    New Best Position: [{population[0]['position'][0]:.3f}, {population[0]['position'][1]:.3f}, {population[0]['position'][2]:.3f}]")
                print(f"    New Best Energy: {population[0]['energy']:.1f} MeV")
                print(f"    New Best Direction: [{population[0]['direction'][0]:.3f}, {population[0]['direction'][1]:.3f}, {population[0]['direction'][2]:.3f}]")
                print("\n")

        # Calculate errors for best candidate this iteration (for both verbose and history tracking)
        best_pos_error = float(jnp.linalg.norm(population[0]['position'] - true_position))
        best_dir_error = float(jnp.arccos(jnp.clip(jnp.dot(population[0]['direction'], true_direction), -1, 1)))
        best_dir_error_deg = np.degrees(best_dir_error)
        best_energy_error = float(jnp.abs(population[0]['energy'] - true_energy))
        
        if history is not None:
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
        if iteration < iterations - 1:  # Don't create new generation on last iteration
            new_population = []
            
            # Keep elite
            elite = population[:n_elite]
            new_population.extend([e.copy() for e in elite])
            
            # Generate new candidates around elite
            population_size = len(population)
            remaining_slots = population_size - n_elite
            
            # Adaptive standard deviations (decrease over iterations)
            progress = (iteration + 1) / iterations
            
            # Scale position std based on detector size
            if detector_bounds['type'] == 'cylinder':
                # Use fraction of detector radius and height
                detector_scale = max(detector_bounds['r'], detector_bounds['H']/2)
            elif detector_bounds['type'] == 'sphere':
                # Use fraction of detector radius
                detector_scale = detector_bounds['r']
            elif detector_bounds['type'] == 'box':
                # Use fraction of largest dimension
                detector_scale = max(detector_bounds['x']/2, detector_bounds['y']/2, detector_bounds['z']/2)
            
            position_std = detector_scale * 0.1
            direction_std = 10.0 * (1 - 0.7 * progress)
            energy_std = 50.0

            i = 0
            while i < remaining_slots:
                key, subkey = jax.random.split(key)
                use_crossover = jax.random.uniform(subkey) < crossover_rate
                
                if use_crossover and i + 1 < remaining_slots:
                    # Crossover: select two parents and create two offspring
                    key, subkey = jax.random.split(key)
                    parent1_idx = min(int(abs(jax.random.normal(subkey)) * n_elite / 3), n_elite - 1)
                    key, subkey = jax.random.split(key)
                    parent2_idx = min(int(abs(jax.random.normal(subkey)) * n_elite / 3), n_elite - 1)
                    
                    parent1 = elite[parent1_idx]
                    parent2 = elite[parent2_idx]
                    
                    # Perform crossover
                    key, subkey = jax.random.split(key)
                    offspring1, offspring2 = crossover_candidates(parent1, parent2, subkey, crossover_type='intermediate')
                    
                    # Apply mutation to crossover offspring with reduced strength
                    mutation_strength = mutation_after_crossover  # Use parameter for mutation strength
                    for offspring in [offspring1, offspring2]:
                        key, subkey = jax.random.split(key)
                        # Apply small mutation
                        mutated_position, mutated_direction, mutated_energy = sample_around_point(
                            subkey, offspring['position'], offspring['direction'], offspring['energy'],
                            position_std * mutation_strength, direction_std * mutation_strength, energy_std * mutation_strength, 
                            detector_bounds
                        )
                        offspring['position'] = mutated_position
                        offspring['direction'] = mutated_direction
                        offspring['energy'] = 800.#mutated_energy
                    
                    new_population.extend([offspring1, offspring2])
                    i += 2
                else:
                    # Mutation: select one parent and create one offspring
                    key, subkey = jax.random.split(key)
                    parent_idx = min(int(abs(jax.random.normal(subkey)) * n_elite / 3), n_elite - 1)
                    parent = elite[parent_idx]
                    
                    # Generate offspring through mutation
                    key, subkey = jax.random.split(key)
                    new_position, new_direction, new_energy = sample_around_point(
                        subkey, parent['position'], parent['direction'], parent['energy'],
                        position_std, direction_std, energy_std, detector_bounds
                    )
                    
                    new_population.append({
                        'position': new_position,
                        'direction': new_direction,
                        'energy': 800.,#new_energy,
                        'loss': float('inf')
                    })
                    
                    i += 1
            
            population = new_population

    return population, best_overall, best_overall_loss, history


# Create JIT-compiled loss functions that work with pre-simulated data
@jit
def _energy_loss_jit(simulated_charges, true_charges):
    """JIT-compiled energy loss function."""
    from .losses import _compute_energy_loss
    return _compute_energy_loss(simulated_charges, true_charges)

@jit  
def _spatial_loss_jit(simulated_charges, simulated_times, true_charges, true_times, sensor_positions, tau, lambda_time):
    """JIT-compiled spatial loss function."""
    from .losses import _compute_spatial_loss
    return _compute_spatial_loss(simulated_charges, simulated_times, true_charges, true_times, sensor_positions, tau, lambda_time)

# JIT functions will be compiled on first use


def create_gradient_optimizer(simulate_event, sensor_positions, sensor_params,
                            tau, lambda_time):
    """
    Create JIT-compiled gradient functions and optimizers for gradient-based optimization.
    """
    
    # Create functions that compute value and gradient w.r.t. parameters
    def energy_loss_fn(params, true_charges, true_times, event_key):
        # Simulate once
        simulated_charges, simulated_times = simulate_event(params, sensor_params, event_key)
        # Compute loss using JIT-compiled function
        return _energy_loss_jit(simulated_charges, true_charges)
    
    def spatial_loss_fn(params, true_charges, true_times, event_key):
        # Simulate once
        simulated_charges, simulated_times = simulate_event(params, sensor_params, event_key)
        # Compute loss using JIT-compiled function
        return _spatial_loss_jit(simulated_charges, simulated_times, true_charges, true_times, sensor_positions, tau, lambda_time)
    
    # Create value_and_grad functions w.r.t. parameters
    energy_grad_fn = jax.value_and_grad(energy_loss_fn)
    spatial_grad_fn = jax.value_and_grad(spatial_loss_fn)
    
    return energy_grad_fn, spatial_grad_fn


def gradient_step(params, opt_state, true_charges, true_times, key,
                        energy_grad_fn, spatial_grad_fn, optimizer,
                        energy_lr_multiplier, spatial_lr_multiplier,
                        energy_scale, position_scale, direction_scale,
                        detector_bounds):
    """
    Perform a single gradient descent step using pre-compiled gradient functions.
    """
    # Compute gradients using pre-compiled gradient functions
    energy_grad, spatial_grad, energy_loss_val, spatial_loss_val = compute_gradients(
        params, true_charges, true_times, energy_grad_fn, spatial_grad_fn, key
    )

    # Combine losses for total loss
    total_loss = energy_loss_val + spatial_loss_val
    
    # Extract individual gradients
    energy_grad_energy, energy_grad_position, energy_grad_direction = energy_grad
    spatial_grad_energy, spatial_grad_position, spatial_grad_direction = spatial_grad
    
    # Combine scaled gradients
    grads = (energy_grad_energy, spatial_grad_position, spatial_grad_direction)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    
    # Extract individual updates
    energy_update, position_update, direction_update = updates
    
    position_update  = position_update    
    direction_update = direction_update
    position_update_mag = jnp.linalg.norm(position_update)
    direction_update_mag = jnp.linalg.norm(direction_update)

    # If the gradients are too big they are not good as individual components get clipped to 1 (this is a quick fix):
    if position_update_mag > 1.5: 
        direction_update *= 1e-2
        position_update *= 1e-2

    energy_update_mag = jnp.abs(energy_update)
    energy_update = energy_update*energy_scale*energy_lr_multiplier   
    position_update = position_update*position_scale*spatial_lr_multiplier
    direction_update = direction_update*direction_scale*spatial_lr_multiplier

    # Apply clipped updates
    scaled_updates = (energy_update, position_update, direction_update)
    new_params = optax.apply_updates(params, scaled_updates)
    
    # Ensure direction remains normalized
    new_energy, new_position, new_direction_angles = new_params
    new_theta, new_phi = new_direction_angles
    new_direction = spherical_to_cartesian(new_theta, new_phi)
    new_direction = new_direction / jnp.linalg.norm(new_direction)
    new_theta = jnp.arccos(jnp.clip(new_direction[2], -1.0, 1.0))
    new_phi = jnp.arctan2(new_direction[1], new_direction[0])
    new_direction_angles = jnp.array([new_theta, new_phi])
    
    # Apply detector bounds to position and energy
    new_position, new_energy = apply_detector_bounds(new_position, new_energy, detector_bounds)
    
    new_params = (new_energy, new_position, new_direction_angles)
    
    grad_info = {
        'energy_loss': energy_loss_val,
        'spatial_loss': spatial_loss_val,
    }
    
    return new_params, new_opt_state, float(total_loss), grad_info

def gradient_optimization(
    initial_params, true_charges, true_times,
    simulate_event, sensor_params, sensor_positions,
    detector_bounds, true_position, true_direction, true_energy,
    n_gradient, gradient_kwargs, key, verbose):

    if verbose:
        print(f"\nGradient-based optimization ({n_gradient} iterations)")

    # Create simple optimizer for gradient approach
    optimizer = optax.chain(
        optax.adam(learning_rate=1.0)  # Will be scaled by learning rate multipliers and scale factors
    )
    
    # Create pre-compiled gradient functions
    energy_grad_fn, spatial_grad_fn = create_gradient_optimizer(
        simulate_event, sensor_positions, sensor_params,
        tau=gradient_kwargs['tau'],
        lambda_time=gradient_kwargs['lambda_time']
    )
    
    # Run gradient optimization
    key, subkey = jax.random.split(key)
       
    params = initial_params
    opt_state = optimizer.init(params)

    # Best tracking
    best_loss = float('inf')
    best_params = params
    patience_counter = 0
    
    # History tracking
    history = {
        'loss': [],
        'energy_loss': [],
        'spatial_loss': [],
        'energy': [],
        'position': [],
        'direction': [],
        'energy_lr': [],
        'spatial_lr': [],
        'position_error': [],
        'direction_error': [],
        'energy_error': []
    }

    energy_lr = gradient_kwargs['energy_lr']
    spatial_lr = gradient_kwargs['spatial_lr']
    patience = gradient_kwargs['patience']
    patience_factor = gradient_kwargs['patience_factor']

    for i in range(n_gradient):    
        start = time.time()
        # Gradient step using pre-compiled gradient functions
        params, opt_state, loss, grad_info = gradient_step(
            params, opt_state, true_charges, true_times, subkey,
            energy_grad_fn, spatial_grad_fn, optimizer,
            energy_lr, spatial_lr,
            gradient_kwargs['energy_scale'], gradient_kwargs['position_scale'], gradient_kwargs['direction_scale'],
            detector_bounds
        )

        # Track history
        energy, position, direction_angles = params
        history['loss'].append(loss)
        history['energy_loss'].append(grad_info['energy_loss'])
        history['spatial_loss'].append(grad_info['spatial_loss'])
        history['energy'].append(float(energy))
        history['position'].append(jnp.array(position))
        history['direction'].append(spherical_to_cartesian(direction_angles[0], direction_angles[1]))
        history['energy_lr'].append(energy_lr)
        history['spatial_lr'].append(spatial_lr)
        
        direction = spherical_to_cartesian(direction_angles[0], direction_angles[1])
        pos_error = float(jnp.linalg.norm(position - true_position))
        dir_error = float(jnp.arccos(jnp.clip(jnp.dot(direction, true_direction), -1, 1)))
        dir_error_deg = dir_error * 180.0 / jnp.pi
        energy_error = float(jnp.abs(energy - true_energy))
        
        history['position_error'].append(pos_error)
        history['direction_error'].append(dir_error_deg)
        history['energy_error'].append(energy_error)
        
        # Look only at the energy loss to decrease LR for spatial part
        if grad_info['energy_loss'] < best_loss:
            best_loss = grad_info['energy_loss']
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Reduce learning rate if patience exceeded
        if patience_counter >= patience:
            spatial_lr *= patience_factor
            patience_counter = 0

        # let's reduce the energy lr every 50 iterations.
        if i>0 and i%50==0:
            energy_lr *= patience_factor

            if verbose:
                print(f"  Iteration {i+1}: Reducing learning rates to {energy_lr:.6f}, {spatial_lr:.6f}")

        best_params = params # let's take the last iterations as the solution (loss isn't at minimum because of our energy+spatial loss approach)

    return best_params, history