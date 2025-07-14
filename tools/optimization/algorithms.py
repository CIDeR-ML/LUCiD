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

from ..losses import compute_softmin_loss
from ..utils import spherical_to_cartesian


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
                   random_seed=42, verbose=True, track_history=False,
                   optimization_type='numerical', gradient_iterations=0, gradient_kwargs=None):
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
    optimization_type : str
        Type of optimization: 'numerical', 'gradient', or 'hybrid'
    gradient_iterations : int
        Number of gradient iterations (for 'gradient' or 'hybrid' types)
    gradient_kwargs : dict
        Keyword arguments for gradient optimization
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
        'energy_error': [],
        'population_size': population_size  # Store for evaluation counting
    } if track_history else None
    
    # Initialize with random population
    if verbose:
        print("Initializing random population...")
    population = []
    
    for i in range(population_size):
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
            
            position_std = detector_scale * 0.1 * (1 - 0.7 * progress)
            direction_std = 0.2 * (1 - 0.7 * progress)
            energy_std = 50.0
            
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
    
    # Apply gradient optimization if requested
    if optimization_type in ['gradient', 'hybrid'] and gradient_iterations > 0:
        
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
        
        # Set up gradient kwargs with defaults
        if gradient_kwargs is None:
            gradient_kwargs = {}
        
        # Run hybrid optimization
        n_numerical = 0 if optimization_type == 'gradient' else n_iterations
        best_params, grad_history = hybrid_optimization(
            initial_params, true_charges, true_times,
            simulate_event, sensor_params, sensor_positions,
            detector_bounds, true_position, true_direction, true_energy,
            n_numerical=0,  # We already did numerical optimization
            n_gradient=gradient_iterations,
            gradient_kwargs=gradient_kwargs,
            key=key,
            verbose=verbose,
            auto_scale=gradient_kwargs.get('auto_scale', True)  # Enable auto-scaling by default
        )
        
        # Convert back to result format
        energy, position, direction_angles = best_params
        theta, phi = direction_angles
        direction = spherical_to_cartesian(theta, phi)
        
        # Recalculate final loss
        test_charges, test_times = simulate_event(best_params, sensor_params, key)
        final_loss = compute_softmin_loss(
            sensor_positions,
            true_charges,
            true_times,
            test_charges,
            test_times,
            tau=gradient_kwargs.get('tau', 0.01),
            lambda_time=gradient_kwargs.get('lambda_time', 1.0),
            lambda_intensity=gradient_kwargs.get('lambda_intensity', 1.0)
        )
        
        # Get the best loss from gradient optimization history
        best_gradient_loss = min(grad_history['gradient']['loss']) if 'gradient' in grad_history else float(final_loss)
        
        best_overall = {
            'position': position,
            'direction': direction,
            'energy': energy,
            'loss': best_gradient_loss  # Use the actual best loss from optimization
        }
        
        # Update history if tracking
        if track_history and history is not None and 'gradient' in grad_history:
            history['gradient_loss'] = grad_history['gradient']['loss']
            history['gradient_energy'] = grad_history['gradient']['energy']
            history['gradient_position'] = grad_history['gradient']['position']
            history['gradient_direction'] = grad_history['gradient']['direction']
            # Store error histories from gradient optimization
            history['gradient_position_error'] = grad_history['gradient']['position_error']
            history['gradient_direction_error'] = grad_history['gradient']['direction_error']
            history['gradient_energy_error'] = grad_history['gradient']['energy_error']
            # Store the actual best loss from gradient optimization
            history['gradient_best_loss'] = min(grad_history['gradient']['loss'])
    
    if track_history:
        return best_overall, population, history
    else:
        return best_overall, population


# ===================================================================
# GRADIENT-BASED OPTIMIZATION ALGORITHMS
# ===================================================================

def create_gradient_optimizer(simulate_event, sensor_positions, sensor_params,
                            energy_lr=1.0, spatial_lr=1.0, 
                            energy_scale=1.0, position_scale=1.0, direction_scale=1.0,
                            tau=0.01, lambda_time=1.0, lambda_intensity=1.0):
    """
    Create gradient-based optimizer using compute_softmin_loss.
    
    Parameters:
    -----------
    simulate_event : function
        Event simulation function
    sensor_positions : jnp.ndarray
        Array of sensor positions
    sensor_params : tuple
        Sensor parameters for simulation
    energy_lr : float
        Learning rate for energy parameter
    spatial_lr : float
        Learning rate for spatial parameters (position and direction)
    energy_scale : float
        Scale factor for energy gradient updates
    position_scale : float
        Scale factor for position gradient updates
    direction_scale : float
        Scale factor for direction gradient updates
    tau : float
        Temperature parameter for softmin loss
    lambda_time : float
        Time loss weight
    lambda_intensity : float
        Intensity loss weight
    
    Returns:
    -------
    loss_fn : function
        Loss function
    grad_fn : function
        Gradient function
    optimizer : optax optimizer
        Combined optimizer for all parameters
    """
    
    def loss_fn(params, true_charges, true_times, key):
        """Compute loss for given parameters."""
        energy, position, direction_angles = params
        
        # Simulate event
        test_charges, test_times = simulate_event(params, sensor_params, key)
        
        # Compute softmin loss
        loss = compute_softmin_loss(
            sensor_positions,
            true_charges,
            true_times,
            test_charges,
            test_times,
            tau=tau,
            lambda_time=lambda_time,
            lambda_intensity=0.#lambda_intensity
        )
        
        return loss
    
    # Create gradient function
    grad_fn = jax.grad(loss_fn, argnums=0)
    
    # Create optimizer with different learning rates for different parameters
    # Using OptaxWrapper to handle parameter-specific learning rates
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping for stability
        optax.adam(learning_rate=1.0)    # Base learning rate of 1.0, will be scaled
    )
    
    return loss_fn, grad_fn, optimizer


def gradient_step(params, opt_state, true_charges, true_times, key,
                 loss_fn, grad_fn, optimizer,
                 energy_lr_multiplier=1.0, spatial_lr_multiplier=1.0,
                 energy_scale=1.0, position_scale=1.0, direction_scale=1.0,
                 max_energy_update=15.0, max_position_update=0.15, max_direction_update_deg=5.0,
                 detector_bounds=None):
    """
    Perform a single gradient descent step with parameter-specific scaling.
    
    Parameters:
    -----------
    params : tuple
        Current parameters (energy, position, direction_angles)
    opt_state : optax state
        Optimizer state
    true_charges : jnp.ndarray
        True charge measurements
    true_times : jnp.ndarray
        True time measurements
    key : PRNGKey
        Random key for simulation
    loss_fn : function
        Loss function
    grad_fn : function
        Gradient function
    optimizer : optax optimizer
        Optimizer
    energy_lr_multiplier : float
        Current learning rate multiplier for energy
    spatial_lr_multiplier : float
        Current learning rate multiplier for spatial parameters
    energy_scale : float
        Scale factor for energy updates
    position_scale : float
        Scale factor for position updates
    direction_scale : float
        Scale factor for direction updates
    max_energy_update : float
        Maximum allowed energy update in MeV
    max_position_update : float
        Maximum allowed position update in meters
    max_direction_update_deg : float
        Maximum allowed direction update in degrees
    detector_bounds : dict
        Detector boundary information for position clipping
    
    Returns:
    -------
    new_params : tuple
        Updated parameters
    new_opt_state : optax state
        Updated optimizer state
    loss : float
        Current loss value
    grads : tuple
        Computed gradients
    """
    # Compute loss and gradients
    loss = loss_fn(params, true_charges, true_times, key)
    grads = grad_fn(params, true_charges, true_times, key)
    
    # Extract individual gradients
    energy_grad, position_grad, direction_grad = grads
    
    # Apply parameter-specific scaling
    scaled_energy_grad = energy_grad * energy_scale * energy_lr_multiplier
    scaled_position_grad = position_grad * position_scale * spatial_lr_multiplier
    scaled_direction_grad = direction_grad * direction_scale * spatial_lr_multiplier
    
    # Combine scaled gradients
    scaled_grads = (scaled_energy_grad, scaled_position_grad, scaled_direction_grad)
    
    # Update parameters
    updates, new_opt_state = optimizer.update(scaled_grads, opt_state, params)
    
    # Extract individual updates
    energy_update, position_update, direction_update = updates
    old_energy, old_position, old_direction_angles = params
    
    # Clip actual parameter updates to maximum allowed changes
    # Energy update clipping
    energy_update_mag = jnp.abs(energy_update)
    if energy_update_mag > max_energy_update:
        energy_update = energy_update * (max_energy_update / energy_update_mag)
    
    # Position update clipping
    position_update_mag = jnp.linalg.norm(position_update)
    if position_update_mag > max_position_update:
        position_update = position_update * (max_position_update / position_update_mag)
    
    # Direction update clipping (convert to radians)
    max_direction_update_rad = max_direction_update_deg * jnp.pi / 180.0
    direction_update_mag = jnp.linalg.norm(direction_update)
    if direction_update_mag > max_direction_update_rad:
        direction_update = direction_update * (max_direction_update_rad / direction_update_mag)
    
    # Apply clipped updates
    clipped_updates = (energy_update, position_update, direction_update)
    new_params = optax.apply_updates(params, clipped_updates)
    
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
    
    return new_params, new_opt_state, float(loss), grads


def gradient_optimization_with_patience(
    initial_params, true_charges, true_times, 
    loss_fn, grad_fn, optimizer,
    n_iterations=100, patience=20, patience_factor=0.5,
    energy_lr=1.0, spatial_lr=1.0,
    energy_scale=1.0, position_scale=1.0, direction_scale=1.0,
    max_energy_update=15.0, max_position_update=0.15, max_direction_update_deg=5.0,
    detector_bounds=None, key=None, verbose=False, gradient_debug=False,
    true_position=None, true_direction=None, true_energy=None):
    """
    Run gradient-based optimization with patience-based learning rate reduction.
    
    Parameters:
    -----------
    initial_params : tuple
        Initial parameters (energy, position, direction_angles)
    true_charges : jnp.ndarray
        True charge measurements
    true_times : jnp.ndarray
        True time measurements
    loss_fn : function
        Loss function
    grad_fn : function
        Gradient function
    optimizer : optax optimizer
        Optimizer
    n_iterations : int
        Maximum number of iterations
    patience : int
        Number of iterations to wait before reducing learning rate
    patience_factor : float
        Factor to reduce learning rate by
    energy_lr : float
        Initial learning rate for energy
    spatial_lr : float
        Initial learning rate for spatial parameters
    energy_scale : float
        Scale factor for energy updates
    position_scale : float
        Scale factor for position updates
    direction_scale : float
        Scale factor for direction updates
    key : PRNGKey
        Random key for simulation
    verbose : bool
        Whether to print progress
    
    Returns:
    -------
    best_params : tuple
        Best parameters found
    history : dict
        Optimization history
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Initialize
    params = initial_params
    opt_state = optimizer.init(params)
    
    # Learning rate multipliers
    energy_lr_multiplier = energy_lr
    spatial_lr_multiplier = spatial_lr
    
    # Best tracking
    best_loss = float('inf')
    best_params = params
    patience_counter = 0
    
    # History tracking
    history = {
        'loss': [],
        'energy': [],
        'position': [],
        'direction': [],
        'energy_lr': [],
        'spatial_lr': [],
        'position_error': [],
        'direction_error': [],
        'energy_error': []
    }
    
    # Print initial parameters if gradient debug is enabled
    if gradient_debug:
        energy, position, direction_angles = initial_params
        direction = spherical_to_cartesian(direction_angles[0], direction_angles[1])
        print(f"  Initial gradient parameters:")
        print(f"    Energy: {float(energy):.1f} MeV")
        print(f"    Position: [{float(position[0]):.3f}, {float(position[1]):.3f}, {float(position[2]):.3f}] m")
        print(f"    Direction: [{float(direction[0]):.3f}, {float(direction[1]):.3f}, {float(direction[2]):.3f}]")
        
        if true_position is not None and true_direction is not None and true_energy is not None:
            # Calculate initial errors
            initial_pos_error = float(jnp.linalg.norm(position - true_position))
            initial_dir_error = float(jnp.arccos(jnp.clip(jnp.abs(jnp.dot(direction, true_direction)), 0, 1)))
            initial_dir_error_deg = initial_dir_error * 180.0 / jnp.pi
            initial_energy_error = float(jnp.abs(energy - true_energy))
            print(f"    Initial errors: Pos={initial_pos_error:.3f}m, Dir={initial_dir_error_deg:.1f}°, Energy={initial_energy_error:.1f}MeV")
    
    for i in range(n_iterations):
        key, subkey = jax.random.split(key)
        
        # Gradient step
        params, opt_state, loss, grads = gradient_step(
            params, opt_state, true_charges, true_times, subkey,
            loss_fn, grad_fn, optimizer,
            energy_lr_multiplier, spatial_lr_multiplier,
            energy_scale, position_scale, direction_scale,
            max_energy_update, max_position_update, max_direction_update_deg,
            detector_bounds
        )
        
        # Extract gradients for debugging
        energy_grad, position_grad, direction_grad = grads
        
        # Track history
        energy, position, direction_angles = params
        history['loss'].append(loss)
        history['energy'].append(float(energy))
        history['position'].append(jnp.array(position))
        history['direction'].append(spherical_to_cartesian(direction_angles[0], direction_angles[1]))
        history['energy_lr'].append(energy_lr_multiplier)
        history['spatial_lr'].append(spatial_lr_multiplier)
        
        # Track errors if true values are provided
        if true_position is not None and true_direction is not None and true_energy is not None:
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
            energy_lr_multiplier *= patience_factor
            spatial_lr_multiplier *= patience_factor
            patience_counter = 0
            
            if verbose:
                print(f"  Iteration {i+1}: Reducing learning rates to {energy_lr_multiplier:.6f}, {spatial_lr_multiplier:.6f}")
        
        # Determine when to show debugging info
        show_debug = (verbose and (gradient_debug or i % 10 == 0 or i == n_iterations - 1)) or gradient_debug
        
        if show_debug:
            # Calculate gradient and update magnitudes for debugging
            energy_grad_mag = float(jnp.linalg.norm(energy_grad))
            position_grad_mag = float(jnp.linalg.norm(position_grad))
            direction_grad_mag = float(jnp.linalg.norm(direction_grad))
            
            # Calculate scaled update magnitudes
            scaled_energy_update_mag = float(jnp.linalg.norm(energy_grad * energy_scale * energy_lr_multiplier))
            scaled_position_update_mag = float(jnp.linalg.norm(position_grad * position_scale * spatial_lr_multiplier))
            scaled_direction_update_mag = float(jnp.linalg.norm(direction_grad * direction_scale * spatial_lr_multiplier))
            
            # print(f"  Iteration {i+1}/{n_iterations}: Loss = {loss:.6f}, Best = {best_loss:.6f}")
            # print(f"    Gradients:  Energy={energy_grad_mag:.2e}, Position={position_grad_mag:.2e}, Direction={direction_grad_mag:.2e}")
            # print(f"    Updates:    Energy={scaled_energy_update_mag:.2e}, Position={scaled_position_update_mag:.2e}, Direction={scaled_direction_update_mag:.2e}")
            # print(f"    LR Mult:    Energy={energy_lr_multiplier:.4f}, Spatial={spatial_lr_multiplier:.4f}")
            
            # Show current parameter values and errors if gradient_debug is enabled
            if gradient_debug:
                current_energy, current_position, current_direction_angles = params
                current_direction = spherical_to_cartesian(current_direction_angles[0], current_direction_angles[1])
                # print(f"    Current params: Energy={float(current_energy):.1f}MeV, Position=[{float(current_position[0]):.3f}, {float(current_position[1]):.3f}, {float(current_position[2]):.3f}]m")
                # print(f"                    Direction=[{float(current_direction[0]):.3f}, {float(current_direction[1]):.3f}, {float(current_direction[2]):.3f}]")
                
                if true_position is not None and true_direction is not None and true_energy is not None:
                    # Calculate current errors
                    current_pos_error = float(jnp.linalg.norm(current_position - true_position))
                    current_dir_error = float(jnp.arccos(jnp.clip(jnp.abs(jnp.dot(current_direction, true_direction)), 0, 1)))
                    current_dir_error_deg = current_dir_error * 180.0 / jnp.pi
                    current_energy_error = float(jnp.abs(current_energy - true_energy))
                    print(f"    Current errors: Pos={current_pos_error:.3f}m, Dir={current_dir_error_deg:.1f}°, Energy={current_energy_error:.1f}MeV")
    
    return best_params, history


def calculate_adaptive_scales(initial_params, true_charges, true_times,
                            simulate_event, sensor_params, sensor_positions,
                            detector_bounds, energy_lr=1.0, spatial_lr=0.1,
                            target_energy_update_mev=10.0,
                            target_position_update_fraction=0.05,
                            target_direction_update_degrees=5.0,
                            key=None, verbose=False):
    """
    Calculate appropriate gradient scales based on initial gradients and desired update sizes.
    
    Parameters:
    -----------
    target_energy_update_mev : float
        Desired first energy update in MeV when energy_scale=1.0
    target_position_update_fraction : float
        Desired first position update as fraction of detector scale when position_scale=1.0
    target_direction_update_degrees : float
        Desired first direction update in degrees when direction_scale=1.0
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Create a temporary optimizer to compute initial gradients
    loss_fn, grad_fn, _ = create_gradient_optimizer(
        simulate_event, sensor_positions, sensor_params,
        energy_lr=energy_lr, spatial_lr=spatial_lr,
        energy_scale=1.0, position_scale=1.0, direction_scale=1.0,  # Use 1.0 for initial calculation
        tau=0.01, lambda_time=1.0, lambda_intensity=1.0
    )
    
    # Compute initial gradients
    initial_grads = grad_fn(initial_params, true_charges, true_times, key)
    energy_grad, position_grad, direction_grad = initial_grads
    
    # Calculate gradient magnitudes
    energy_grad_mag = float(jnp.linalg.norm(energy_grad))
    position_grad_mag = float(jnp.linalg.norm(position_grad))
    direction_grad_mag = float(jnp.linalg.norm(direction_grad))
    
    # Calculate detector scale (same way as in algorithms.py adaptive search)
    if detector_bounds['type'] == 'cylinder':
        detector_scale = max(detector_bounds['r'], detector_bounds['H']/2)
    elif detector_bounds['type'] == 'sphere':
        detector_scale = detector_bounds['r']
    elif detector_bounds['type'] == 'box':
        detector_scale = max(detector_bounds['x']/2, detector_bounds['y']/2, detector_bounds['z']/2)
    else:
        detector_scale = 1.0  # fallback
    
    # Calculate scales based on desired update sizes
    # energy_update = energy_grad * energy_scale * energy_lr
    # We want: target_energy_update_mev = energy_grad_mag * energy_scale * energy_lr
    energy_scale = target_energy_update_mev / (energy_grad_mag * energy_lr) if energy_grad_mag > 0 else 0.01
    
    # position_update = position_grad * position_scale * spatial_lr  
    # We want: target_position_update_fraction * detector_scale = position_grad_mag * position_scale * spatial_lr
    target_position_update = target_position_update_fraction * detector_scale
    position_scale = target_position_update / (position_grad_mag * spatial_lr) if position_grad_mag > 0 else 0.1
    
    # direction_update = direction_grad * direction_scale * spatial_lr
    # We want: target_direction_update_degrees * π/180 = direction_grad_mag * direction_scale * spatial_lr
    target_direction_update_rad = target_direction_update_degrees * jnp.pi / 180.0
    direction_scale = target_direction_update_rad / (direction_grad_mag * spatial_lr) if direction_grad_mag > 0 else 0.1
    
    if verbose:
        print(f"  Adaptive scale calculation:")
        print(f"    Initial gradients: Energy={energy_grad_mag:.2e}, Position={position_grad_mag:.2e}, Direction={direction_grad_mag:.2e}")
        print(f"    Detector scale: {detector_scale:.2f}")
        print(f"    Target updates: Energy={target_energy_update_mev:.1f}MeV, Position={target_position_update:.3f}m ({target_position_update_fraction*100:.1f}%), Direction={target_direction_update_degrees:.1f}°")
        print(f"    Calculated scales: Energy={energy_scale:.6f}, Position={position_scale:.6f}, Direction={direction_scale:.6f}")
    
    return energy_scale, position_scale, direction_scale


def hybrid_optimization(
    initial_params, true_charges, true_times,
    simulate_event, sensor_params, sensor_positions,
    detector_bounds, true_position, true_direction, true_energy,
    n_numerical=10, n_gradient=50,
    numerical_kwargs=None, gradient_kwargs=None,
    key=None, verbose=False, auto_scale=True):
    """
    Hybrid optimization combining numerical (adaptive search) and gradient-based methods.
    
    Parameters:
    -----------
    initial_params : tuple or None
        Initial parameters. If None, will use numerical search to find initial guess
    true_charges : jnp.ndarray
        True charge measurements
    true_times : jnp.ndarray
        True time measurements
    simulate_event : function
        Event simulation function
    sensor_params : tuple
        Sensor parameters
    sensor_positions : jnp.ndarray
        Sensor positions
    detector_bounds : dict
        Detector boundary information
    true_position : jnp.ndarray
        True position (for error calculation)
    true_direction : jnp.ndarray
        True direction (for error calculation)
    true_energy : float
        True energy (for error calculation)
    n_numerical : int
        Number of numerical optimization iterations
    n_gradient : int
        Number of gradient optimization iterations
    numerical_kwargs : dict
        Keyword arguments for numerical optimization
    gradient_kwargs : dict
        Keyword arguments for gradient optimization
    key : PRNGKey
        Random key
    verbose : bool
        Whether to print progress
    
    Returns:
    -------
    best_params : tuple
        Best parameters found
    full_history : dict
        Combined optimization history
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Default kwargs
    if numerical_kwargs is None:
        numerical_kwargs = {
            'population_size': 20,
            'elite_fraction': 0.2
        }
    if gradient_kwargs is None:
        gradient_kwargs = {
            'energy_lr': 1.0,
            'spatial_lr': 0.1,
            'energy_scale': 0.01,
            'position_scale': 0.1,
            'direction_scale': 0.1,
            'patience': 20,
            'patience_factor': 0.5
        }
    
    # Phase 1: Numerical optimization (if needed)
    if n_numerical > 0:
        if verbose:
            print(f"\nPhase 1: Numerical optimization ({n_numerical} iterations)")
        
        key, subkey = jax.random.split(key)
        best_numerical, _, numerical_history = adaptive_search(
            true_charges, true_times, simulate_event, sensor_params, sensor_positions,
            detector_bounds, true_position, true_direction, true_energy,
            n_iterations=n_numerical,
            random_seed=int(subkey[0]),
            verbose=verbose,
            track_history=True,
            **numerical_kwargs
        )
        
        # Extract parameters from best match
        if best_numerical:
            theta = jnp.arccos(best_numerical['direction'][2])
            phi = jnp.arctan2(best_numerical['direction'][1], best_numerical['direction'][0])
            initial_params = (
                best_numerical['energy'],
                best_numerical['position'],
                jnp.array([theta, phi])
            )
        else:
            raise RuntimeError("Numerical optimization failed to find initial parameters")
    
    # Phase 2: Gradient-based optimization
    if n_gradient > 0:
        if verbose:
            print(f"\nPhase 2: Gradient-based optimization ({n_gradient} iterations)")
            if n_numerical > 0:
                print(f"  Starting from numerical result: Loss = {best_numerical['loss']:.6f}")
        
        # Calculate adaptive scales if requested
        if auto_scale:
            if verbose:
                print(f"  Calculating adaptive gradient scales...")
            
            energy_scale, position_scale, direction_scale = calculate_adaptive_scales(
                initial_params, true_charges, true_times,
                simulate_event, sensor_params, sensor_positions, detector_bounds,
                energy_lr=gradient_kwargs.get('energy_lr', 1.0),
                spatial_lr=gradient_kwargs.get('spatial_lr', 0.1),
                target_energy_update_mev=gradient_kwargs.get('target_energy_update_mev', 10.0),
                target_position_update_fraction=gradient_kwargs.get('target_position_update_fraction', 0.05),
                target_direction_update_degrees=gradient_kwargs.get('target_direction_update_degrees', 5.0),
                key=key, verbose=verbose
            )
            
            # Update gradient_kwargs with calculated scales
            gradient_kwargs = gradient_kwargs.copy()  # Don't modify original
            gradient_kwargs['energy_scale'] = energy_scale
            gradient_kwargs['position_scale'] = position_scale
            gradient_kwargs['direction_scale'] = direction_scale
        
        # Create gradient optimizer
        loss_fn, grad_fn, optimizer = create_gradient_optimizer(
            simulate_event, sensor_positions, sensor_params,
            energy_lr=gradient_kwargs.get('energy_lr', 1.0),
            spatial_lr=gradient_kwargs.get('spatial_lr', 0.1),
            energy_scale=gradient_kwargs.get('energy_scale', 0.01),
            position_scale=gradient_kwargs.get('position_scale', 0.1),
            direction_scale=gradient_kwargs.get('direction_scale', 0.1),
            tau=gradient_kwargs.get('tau', 0.01),
            lambda_time=gradient_kwargs.get('lambda_time', 1.0),
            lambda_intensity=gradient_kwargs.get('lambda_intensity', 1.0)
        )
        
        # Run gradient optimization
        key, subkey = jax.random.split(key)
        
        # Extract parameters that are specific to gradient_optimization_with_patience
        patience_kwargs = {
            'n_iterations': n_gradient,
            'patience': gradient_kwargs.get('patience', 20),
            'patience_factor': gradient_kwargs.get('patience_factor', 0.5),
            'energy_lr': gradient_kwargs.get('energy_lr', 1.0),
            'spatial_lr': gradient_kwargs.get('spatial_lr', 0.1),
            'energy_scale': gradient_kwargs.get('energy_scale', 0.01),
            'position_scale': gradient_kwargs.get('position_scale', 0.1),
            'direction_scale': gradient_kwargs.get('direction_scale', 0.1),
            'max_energy_update': gradient_kwargs.get('target_energy_update_mev', 15.0),
            'max_position_update': gradient_kwargs.get('target_position_update_fraction', 0.05) * (
                max(detector_bounds['r'], detector_bounds['H']/2) if detector_bounds and detector_bounds.get('type') == 'cylinder'
                else detector_bounds['r'] if detector_bounds and detector_bounds.get('type') == 'sphere'
                else max(detector_bounds['x']/2, detector_bounds['y']/2, detector_bounds['z']/2) if detector_bounds and detector_bounds.get('type') == 'box'
                else 5.0
            ),
            'max_direction_update_deg': gradient_kwargs.get('target_direction_update_degrees', 5.0),
            'detector_bounds': detector_bounds,
            'key': subkey,
            'verbose': gradient_kwargs.get('gradient_verbose', verbose),
            'gradient_debug': gradient_kwargs.get('gradient_debug', False),
            'true_position': true_position,
            'true_direction': true_direction,
            'true_energy': true_energy
        }
        
        best_params, gradient_history = gradient_optimization_with_patience(
            initial_params, true_charges, true_times,
            loss_fn, grad_fn, optimizer,
            **patience_kwargs
        )
    else:
        best_params = initial_params
        gradient_history = None
    
    # Combine histories
    full_history = {}
    if n_numerical > 0 and numerical_history:
        full_history['numerical'] = numerical_history
    if n_gradient > 0 and gradient_history:
        full_history['gradient'] = gradient_history
    
    return best_params, full_history