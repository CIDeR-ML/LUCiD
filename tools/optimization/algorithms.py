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
    
    # Ensure position is within bounds
    if detector_bounds:
        if detector_bounds['type'] == 'cylinder':
            r = jnp.sqrt(new_position[0]**2 + new_position[1]**2)
            if r > detector_bounds['r'] * 0.8:
                new_position = new_position * (detector_bounds['r'] * 0.8 / r)
            new_position = jnp.clip(new_position, 
                                   jnp.array([-detector_bounds['r'], -detector_bounds['r'], -detector_bounds['H']/2]) * 0.8,
                                   jnp.array([detector_bounds['r'], detector_bounds['r'], detector_bounds['H']/2]) * 0.8)
        elif detector_bounds['type'] == 'sphere':
            r = jnp.linalg.norm(new_position)
            if r > detector_bounds['r'] * 0.8:
                new_position = new_position * (detector_bounds['r'] * 0.8 / r)
        elif detector_bounds['type'] == 'box':
            new_position = jnp.clip(new_position,
                                   jnp.array([-detector_bounds['x']/2, -detector_bounds['y']/2, -detector_bounds['z']/2]) * 0.8,
                                   jnp.array([detector_bounds['x']/2, detector_bounds['y']/2, detector_bounds['z']/2]) * 0.8)
    
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
    new_energy = jnp.clip(center_energy + energy_noise, 250.0, 900.0)
    
    return new_position, new_direction, new_energy


def create_initial_population(key, detector_bounds, population_size):
    
    population = []
    for i in range(population_size*5):
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
            'energy': energy,
            'loss': float('inf')
        })

        population.sort(key=lambda x: x['loss'])

    return population[:population_size]

def evolve_population(key, iterations, population, n_elite, loss_function, detector_bounds, true_values, history, verbose, numerical_debug):

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
            
            if numerical_debug:
                print(f"    Position: [{candidate['position'][0]:.3f}, {candidate['position'][1]:.3f}, {candidate['position'][2]:.3f}]")
                print(f"    Energy: {candidate['energy']:.1f} MeV")
                print(f"    Direction: [{candidate['direction'][0]:.3f}, {candidate['direction'][1]:.3f}, {candidate['direction'][2]:.3f}]")
        
        # Sort by ranking score (which uses weighted loss for combined loss type)
        population.sort(key=lambda x: x['ranking_score'])
        
        # Update best overall (using ranking score for comparison)
        if not best_overall:
            best_overall = population[0].copy()
            best_overall_loss = population[0]['loss']
        elif population[0]['ranking_score'] < best_overall.get('ranking_score', float('inf')):
            best_overall = population[0].copy()
            best_overall_loss = population[0]['loss']

        # Calculate errors for best candidate this iteration (for both verbose and history tracking)
        best_pos_error = float(jnp.linalg.norm(population[0]['position'] - true_position))
        best_dir_error = float(jnp.arccos(jnp.clip(jnp.abs(jnp.dot(population[0]['direction'], true_direction)), 0, 1)))
        best_dir_error_deg = np.degrees(best_dir_error)
        best_energy_error = float(jnp.abs(population[0]['energy'] - true_energy))
        
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
            direction_std = 45.0 * (1 - 0.7 * progress)
            energy_std = 50.0

            if numerical_debug:
                print(f"    Adaptive params: position_std={position_std:.3f}, direction_std={direction_std:.3f}, energy_std={energy_std:.1f}")
            
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
    
    # Clip energy to valid range
    new_energy = jnp.clip(new_energy, 250.0, 900.0)
    
    # Apply detector bounds to position (same as in adaptive search)
    if detector_bounds:
        if detector_bounds['type'] == 'cylinder':
            r = jnp.sqrt(new_position[0]**2 + new_position[1]**2)
            if r > detector_bounds['r'] * 0.8:
                new_position = new_position * (detector_bounds['r'] * 0.8 / r)
            new_position = jnp.clip(new_position, 
                                   jnp.array([-detector_bounds['r'], -detector_bounds['r'], -detector_bounds['H']/2]) * 0.8,
                                   jnp.array([detector_bounds['r'], detector_bounds['r'], detector_bounds['H']/2]) * 0.8)
        elif detector_bounds['type'] == 'sphere':
            r = jnp.linalg.norm(new_position)
            if r > detector_bounds['r'] * 0.8:
                new_position = new_position * (detector_bounds['r'] * 0.8 / r)
        elif detector_bounds['type'] == 'box':
            new_position = jnp.clip(new_position,
                                   jnp.array([-detector_bounds['x']/2, -detector_bounds['y']/2, -detector_bounds['z']/2]) * 0.8,
                                   jnp.array([detector_bounds['x']/2, detector_bounds['y']/2, detector_bounds['z']/2]) * 0.8)
    
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
        dir_error = float(jnp.arccos(jnp.clip(jnp.abs(jnp.dot(direction, true_direction)), 0, 1)))
        dir_error_deg = dir_error * 180.0 / jnp.pi
        energy_error = float(jnp.abs(energy - true_energy))
        
        history['position_error'].append(pos_error)
        history['direction_error'].append(dir_error_deg)
        history['energy_error'].append(energy_error)
        
        # Check for improvement
        if loss < best_loss:
            best_loss = loss
            best_params = params
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Reduce learning rate if patience exceeded
        if patience_counter >= patience:
            energy_lr *= patience_factor
            spatial_lr *= patience_factor
            patience_counter = 0
            
            if verbose:
                print(f"  Iteration {i+1}: Reducing learning rates to {energy_lr:.6f}, {spatial_lr:.6f}")

    return best_params, history