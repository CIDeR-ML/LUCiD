from tools.utils import generate_random_point_inside_cylinder
from functools import partial
from tools.geometry import generate_detector

import jax.numpy as jnp
from jax import jit
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import optax

import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np

from tools.utils import generate_random_params

from tools.simulation import setup_event_simulator

def create_multi_objective_optimizer(
    simulate_event,
    detector_points,
    detector_params,
    energy_lr,
    spatial_lr,
    lambda_time,
    tau=0.01
):
    """Create optimizers for different parameter groups."""
    
    # Create separate loss functions within this scope
    def energy_loss_fn(params, true_event_data, event_key):
        """Energy-focused loss function."""

        true_charge, true_time = true_event_data
        
        simulated_data = simulate_event(params, detector_params, event_key)
        simulated_charge, _ = simulated_data
        
        total_true_charge = jnp.sum(true_charge)
        total_sim_charge = jnp.sum(simulated_charge)
        
        eps = 1e-8
        intensity_loss = jnp.abs(jnp.log(total_sim_charge / (total_true_charge + eps)))
        return intensity_loss
    
    def spatial_loss_fn(params, true_event_data, event_key):
        """Spatial/temporal-focused loss function."""

        true_charge, true_time = true_event_data
        
        simulated_data = simulate_event(params, detector_params, event_key)
        simulated_charge, simulated_time = simulated_data
        
        # Use the spatial loss computation from your original softmin loss
        eps = 1e-8
        threshold = 1e-8
        
        # Compute mean times for active locations
        true_active_mask = true_charge > threshold
        sim_active_mask = simulated_charge > threshold

        true_mean_time = jnp.sum(true_time * true_active_mask) / (jnp.sum(true_active_mask) + eps)
        sim_mean_time = jnp.sum(simulated_time * sim_active_mask) / (jnp.sum(sim_active_mask) + eps)

        true_time_centered = jnp.where(true_active_mask, true_time - true_mean_time, 0.0)
        sim_time_centered = jnp.where(sim_active_mask, simulated_time - sim_mean_time, 0.0)

        # Distance matrix and soft assignments
        N = detector_points.shape[0]
        dist = jnp.linalg.norm(
            detector_points[:, None, :] - detector_points[None, :, :], axis=-1
        )

        # Soft assignments Sim -> True
        logits_s2t = -dist / tau
        w_s2t = jax.nn.softmax(logits_s2t, axis=1)
        
        Q_sim_per_true = w_s2t.T @ simulated_charge
        qt_sim_per_true = w_s2t.T @ (simulated_charge * sim_time_centered)
        avg_sim_time_per_true = qt_sim_per_true / (Q_sim_per_true + eps)
        
        L_charge_s2t = jnp.sum(jnp.abs(Q_sim_per_true - true_charge))
        L_time_s2t = jnp.sum(jnp.abs(avg_sim_time_per_true - true_time_centered) * Q_sim_per_true)
        
        # Soft assignments True -> Sim
        logits_t2s = -dist.T / tau
        w_t2s = jax.nn.softmax(logits_t2s, axis=1)
        
        Q_true_per_sim = w_t2s.T @ true_charge
        qt_true_per_sim = w_t2s.T @ (true_charge * true_time_centered)
        avg_true_time_per_sim = qt_true_per_sim / (Q_true_per_sim + eps)
        
        L_charge_t2s = jnp.sum(jnp.abs(Q_true_per_sim - simulated_charge))
        L_time_t2s = jnp.sum(jnp.abs(avg_true_time_per_sim - sim_time_centered) * Q_true_per_sim)
        
        L_charge = L_charge_s2t + L_charge_t2s
        L_time = (L_time_s2t + L_time_t2s) * lambda_time
        
        return L_charge + L_time
    
    # JIT compile gradient functions
    energy_grad_fn = jit(value_and_grad(energy_loss_fn))
    spatial_grad_fn = jit(value_and_grad(spatial_loss_fn))
    
    # Create optimizers
    energy_optimizer = optax.adam(energy_lr)
    spatial_optimizer = optax.adam(spatial_lr)
    
    return energy_grad_fn, spatial_grad_fn, energy_optimizer, spatial_optimizer

def run_multi_objective_optimization(
    params,
    energy_grad_fn,
    spatial_grad_fn,
    energy_optimizer,
    spatial_optimizer,
    event_key,
    true_event_data,
    n_iterations,
    position_scale,
    patience = 100
):
    """Run optimization with separate objectives."""
    
    # Initialize optimizer states
    energy_opt_state = energy_optimizer.init(params)
    spatial_opt_state = spatial_optimizer.init(params)
    
    # Store losses separately
    loss_history = {
        'total': [],
        'energy': [],
        'spatial': []
    }
    param_history = {
        'energy': [],
        'position_x': [],
        'position_y': [], 
        'position_z': [],
        'theta': [],
        'phi': []
    }
    
    # For learning rate reduction with shared patience
    best_energy_loss = float('inf')
    best_spatial_loss = float('inf')
    energy_patience_counter = 0
    spatial_patience_counter = 0
    energy_lr_multiplier = 1.0
    spatial_lr_multiplier = 1.0
    
    for i in range(n_iterations):
        # Compute gradients for each objective
        energy_loss, energy_grads = energy_grad_fn(params, true_event_data, event_key)
        spatial_loss, spatial_grads = spatial_grad_fn(params, true_event_data, event_key)
        
        total_loss = energy_loss + spatial_loss
        loss_history['total'].append(float(total_loss))
        loss_history['energy'].append(float(energy_loss))
        loss_history['spatial'].append(float(spatial_loss))
        
        # Store parameters
        energy, position, direction_angles = params
        param_history['energy'].append(float(energy))
        param_history['position_x'].append(float(position[0]))
        param_history['position_y'].append(float(position[1]))
        param_history['position_z'].append(float(position[2]))
        param_history['theta'].append(float(direction_angles[0]))
        param_history['phi'].append(float(direction_angles[1]))
        
        # Check if energy loss improved
        if energy_loss < best_energy_loss:
            best_energy_loss = energy_loss
            energy_patience_counter = 0
        else:
            energy_patience_counter += 1

        # Check if spatial loss improved
        if spatial_loss < best_spatial_loss:
            best_spatial_loss = spatial_loss
            spatial_patience_counter = 0
        else:
            spatial_patience_counter += 1
        
        # Update energy parameter
        energy_updates, energy_opt_state = energy_optimizer.update(
            energy_grads, energy_opt_state
        )
        
        # Update spatial parameters  
        spatial_updates, spatial_opt_state = spatial_optimizer.update(
            spatial_grads, spatial_opt_state
        )
        
        # Reduce learning rate if patience exceeded
        if energy_patience_counter >= patience/4:
            energy_lr_multiplier *= 0.5
            energy_patience_counter = 0
            #print(f"Reducing energy learning rate to {energy_lr_multiplier} of original")

        if spatial_patience_counter >= patience:
            spatial_lr_multiplier *= 0.5
            spatial_patience_counter = 0
            #print(f"Reducing spatial learning rate to {spatial_lr_multiplier} of original")
        
        # Combine updates - energy gets energy updates, spatial gets spatial updates
        energy_update, _, _ = energy_updates
        _, position_update, direction_update = spatial_updates
        
        # Scale updates based on learning rate multipliers
        energy_update = jax.tree.map(lambda x: x * energy_lr_multiplier, energy_update)
        position_update = jax.tree.map(lambda x: x * spatial_lr_multiplier, position_update)
        direction_update = jax.tree.map(lambda x: x * spatial_lr_multiplier, direction_update)
        
        # # let's not update the spatial parameters until we have a good guess for the energy
        w = 1.
        # if i<20:
        #     w = 0.

        combined_updates = (energy_update, position_update*position_scale*w, direction_update*w)
        params = optax.apply_updates(params, combined_updates)
    
    return params, loss_history, param_history


@partial(jax.jit, static_argnums=(0, 1))
def generate_random_initial_guess(spatial_grad_fn, energy_grad_fn, true_event_data, event_key, detector_points):
    """
    Generate random initial parameters within reasonable bounds.
    
    Parameters:
        spatial_grad_fn: Spatial gradient function (not used, kept for consistency)
        energy_grad_fn: Energy gradient function (not used, kept for consistency)
        true_event_data: True event data (not used, kept for consistency)
        event_key: JAX random key
        detector_points: Array of detector positions to infer cylinder dimensions
    
    Returns:
        (energy, position, angles) tuple with random initial values
    """
    # Split key for different random components
    key_energy, key_position, key_theta, key_phi = jax.random.split(event_key, 4)
    
    # Random energy between 200 and 800 MeV
    energy_guess = jax.random.uniform(key_energy, shape=(), minval=200., maxval=800.)
    
    # Infer cylinder dimensions from detector points
    # Calculate radius as maximum distance from origin in xy-plane
    xy_distances = jnp.sqrt(detector_points[:, 0]**2 + detector_points[:, 1]**2)
    max_radius = jnp.max(xy_distances)
    
    # Calculate height as z-range
    z_min = jnp.min(detector_points[:, 2])
    z_max = jnp.max(detector_points[:, 2])
    height = z_max - z_min
    
    # Use 80% of the inferred dimensions
    r_80 = 0.8 * max_radius
    h_80 = 0.8 * height
    
    # Note: Print statements won't work inside JIT-compiled functions
    # The cylinder dimensions will be printed from the main function if needed
    
    # Random position inside cylinder with 80% of detector dimensions
    position_guess = generate_random_point_inside_cylinder(key_position, h=h_80, r=r_80)
    
    # Random theta between 0 and pi
    theta_guess = jax.random.uniform(key_theta, shape=(), minval=0., maxval=jnp.pi)
    
    # Random phi between 0 and 2*pi
    phi_guess = jax.random.uniform(key_phi, shape=(), minval=0., maxval=2*jnp.pi)
    
    angles_guess = jnp.array([theta_guess, phi_guess])
    
    return (energy_guess, position_guess, angles_guess)

@partial(jax.jit, static_argnums=(0, 1))
def grid_scan_initial_guess_vectorized(spatial_grad_fn, energy_grad_fn, true_event_data, event_key):
    """
    Vectorized 8x8 grid scan - much faster for JIT compilation.
    """
    energy_guess = 500.
    position_guess = jnp.array([0.,0.,0.])
    
    # Create 8x8 grid for theta (0 to pi) and phi (0 to 2*pi)
    theta_grid = jnp.linspace(0, jnp.pi, 8)
    phi_grid = jnp.linspace(0, 2*jnp.pi, 8)
    
    # Create all combinations at once
    theta_mesh, phi_mesh = jnp.meshgrid(theta_grid, phi_grid)
    all_angles = jnp.stack([theta_mesh.flatten(), phi_mesh.flatten()], axis=1)  # Shape: (64, 2)
    
    # Vectorized evaluation of all angle combinations
    def eval_single_angle(angles):
        params = (energy_guess, position_guess, angles)
        loss_val, _ = spatial_grad_fn(params, true_event_data, event_key)
        return loss_val
    
    # Use vmap to evaluate all angles at once
    all_losses = jax.vmap(eval_single_angle)(all_angles)
    
    # Find best angles
    best_idx = jnp.argmin(all_losses)
    best_angles = all_angles[best_idx]
    
    # Energy search (also vectorized)
    energies = jnp.linspace(300, 900, 15)
    
    def eval_single_energy(energy):
        params = (energy, position_guess, best_angles)
        loss_val, _ = energy_grad_fn(params, true_event_data, event_key)
        return loss_val
    
    energy_losses = jax.vmap(eval_single_energy)(energies)
    best_energy_idx = jnp.argmin(energy_losses)
    best_energy = energies[best_energy_idx]

    return (best_energy, position_guess, best_angles)

@partial(jax.jit, static_argnums=(0, 1))
def grid_scan_initial_guess(spatial_grad_fn, energy_grad_fn, true_event_data, event_key):
    """
    Perform 8x8 grid scan on theta and phi angles to find best initial guess.
    """
    energy_guess = 500.
    position_guess = jnp.array([0.,0.,0.])
    
    # Create 8x8 grid for theta (0 to pi) and phi (0 to 2*pi)
    theta_grid = jnp.linspace(0, jnp.pi, 8)
    phi_grid = jnp.linspace(0, 2*jnp.pi, 8)
    
    # Initialize with JAX arrays instead of Python floats
    best_loss = jnp.array(jnp.inf)
    best_angles = jnp.array([0., 0.])
    
    # Vectorized approach for angle grid search
    for theta in theta_grid:
        for phi in phi_grid:
            angles_candidate = jnp.array([theta, phi])
            params_candidate = (energy_guess, position_guess, angles_candidate)
            
            # Evaluate spatial loss using spatial_grad_fn
            loss_val, _ = spatial_grad_fn(params_candidate, true_event_data, event_key)
            
            # Use jnp.where for conditional updates instead of if statements
            is_better = loss_val < best_loss
            best_loss = jnp.where(is_better, loss_val, best_loss)
            best_angles = jnp.where(is_better, angles_candidate, best_angles)

    # Reset for energy search
    best_loss = jnp.array(jnp.inf)
    best_energy = jnp.array(500.)

    energies = jnp.linspace(300, 900, 15)
    for energy in energies:
        params_candidate = (energy, position_guess, best_angles)
        loss_val, _ = energy_grad_fn(params_candidate, true_event_data, event_key)
        
        # Use jnp.where for conditional updates
        is_better = loss_val < best_loss
        best_loss = jnp.where(is_better, loss_val, best_loss)
        best_energy = jnp.where(is_better, energy, best_energy)

    return (best_energy, position_guess, best_angles)
    
@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def run_single_optimization_step(params, energy_grad_fn, spatial_grad_fn, energy_optimizer, spatial_optimizer, 
                                true_event_data, event_key, position_scale):
    """
    JIT-compatible single optimization step.
    Returns: (new_params, energy_loss, spatial_loss, total_loss)
    """
    # Energy optimization step
    energy_loss, energy_grads = energy_grad_fn(params, true_event_data, event_key)
    energy_updates, energy_opt_state = energy_optimizer.update(energy_grads, None, params)
    
    # Apply energy updates (only to energy parameter)
    new_energy = params[0] + energy_updates[0]
    params_after_energy = (new_energy, params[1], params[2])
    
    # Spatial optimization step
    spatial_loss, spatial_grads = spatial_grad_fn(params_after_energy, true_event_data, event_key)
    spatial_updates, spatial_opt_state = spatial_optimizer.update(spatial_grads, None, params_after_energy)
    
    # Apply spatial updates (position and angles)
    new_position = params_after_energy[1] + spatial_updates[1] * position_scale
    new_angles = params_after_energy[2] + spatial_updates[2]
    
    new_params = (params_after_energy[0], new_position, new_angles)
    total_loss = energy_loss + spatial_loss
    
    return new_params, energy_loss, spatial_loss, total_loss

def run_multi_event_optimization(
    N_events=10,
    json_filename='../config/IWCD_geom_config.json',
    Nphot=1_000_000,
    K=2,
    loss_function='multi_objective',
    lambda_time=1e3,
    energy_lr=1.0,     
    spatial_lr=1.0,
    position_scale=1.0,
    n_iterations=200,
    patience=100,
    base_seed=None,
    verbose=True,
    initial_guess_method='grid_scan'
):
    """
    Main optimization loop - NOT JIT compiled due to Python control flow and I/O.
    
    Parameters
    ----------
    initial_guess_method : str, optional
        Method for generating initial parameter guesses. Options are:
        - 'grid_scan': Use grid scan to find best initial angles and energy (default)
        - 'random': Generate random initial parameters within bounds
    """
    
    # Setup code (same as before)
    detector = generate_detector(json_filename)
    detector_points = jnp.array(detector.all_points)
    
    if verbose:
        print(f"Running parameter optimization on {N_events} events...")
        print(f"Configuration: {loss_function} loss, {n_iterations} iterations each")
        print(f"Initial guess method: {initial_guess_method}")
        
        if initial_guess_method == 'random':
            # Calculate and display detector dimensions for random initialization
            xy_distances = jnp.sqrt(detector_points[:, 0]**2 + detector_points[:, 1]**2)
            max_radius = jnp.max(xy_distances)
            z_min = jnp.min(detector_points[:, 2])
            z_max = jnp.max(detector_points[:, 2])
            height = z_max - z_min
            print(f"Detector dimensions: radius={float(max_radius):.2f}m, height={float(height):.2f}m")
            print(f"Random initialization will use 80% of these dimensions: radius={float(0.8*max_radius):.2f}m, height={float(0.8*height):.2f}m")
    
    simulate_event = setup_event_simulator(json_filename, Nphot, temperature=0.05, K=K, is_calibration=False)
    
    if base_seed is None:
        base_seed = int(time.time())
    
    detector_params = (
        jnp.array(4.),
        jnp.array(0.2),
        jnp.array(6.),
        jnp.array(0.001)
    )
    
    all_results = {
        'loss_histories': [],
        'param_histories': [],
        'true_params': [],
        'initial_guesses': [],
        'final_params': [],
        'final_errors': [],
        'seeds': []
    }

    energy_grad_fn, spatial_grad_fn, energy_optimizer, spatial_optimizer = create_multi_objective_optimizer(
        simulate_event=simulate_event,
        detector_points=detector_points,
        detector_params=detector_params,
        energy_lr=energy_lr,
        spatial_lr=spatial_lr,
        lambda_time=lambda_time,
        tau=0.01
    )
    
    # Process each event
    for event_idx in tqdm(range(N_events), desc="Processing events"):
        event_seed = base_seed + event_idx
        all_results['seeds'].append(event_seed)
        
        key = jax.random.PRNGKey(event_seed)
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        
        # Generate true parameters and simulate event
        true_energy, true_position, true_direction_angles = generate_random_params(subkey1)
        true_params = (true_energy, true_position, true_direction_angles)
        all_results['true_params'].append(true_params)
        
        true_event_data = jax.lax.stop_gradient(simulate_event(true_params, detector_params, subkey1))
        
        # Generate initial guess using the selected method
        if initial_guess_method == 'grid_scan':
            initial_params = grid_scan_initial_guess_vectorized(spatial_grad_fn, energy_grad_fn, true_event_data, subkey2)
        elif initial_guess_method == 'random':
            initial_params = generate_random_initial_guess(spatial_grad_fn, energy_grad_fn, true_event_data, subkey3, detector_points)
        else:
            raise ValueError(f"Unknown initial_guess_method: {initial_guess_method}. Must be 'grid_scan' or 'random'.")
        
        if verbose:
            # Convert to Python types for printing (outside JIT context)
            init_energy, init_pos, init_angles = initial_params
            true_energy, true_pos, true_angles = true_params
            
            print(f"\n--- Event {event_idx} ---")
            print(f"Initial guess method: {initial_guess_method}")
            
            # True parameters
            print(f"True parameters:")
            print(f"  Energy: {float(true_energy):.2f} MeV")
            print(f"  Position: [{float(true_pos[0]):.3f}, {float(true_pos[1]):.3f}, {float(true_pos[2]):.3f}] m")
            print(f"  Angles: θ={float(true_angles[0]):.3f} rad ({float(true_angles[0])*180/jnp.pi:.1f}°), φ={float(true_angles[1]):.3f} rad ({float(true_angles[1])*180/jnp.pi:.1f}°)")
            
            # Initial guess
            print(f"Initial guess:")
            print(f"  Energy: {float(init_energy):.2f} MeV (error: {abs(float(init_energy) - float(true_energy)):.2f} MeV)")
            print(f"  Position: [{float(init_pos[0]):.3f}, {float(init_pos[1]):.3f}, {float(init_pos[2]):.3f}] m (distance: {float(jnp.linalg.norm(init_pos - true_pos)):.3f} m)")
            print(f"  Angles: θ={float(init_angles[0]):.3f} rad ({float(init_angles[0])*180/jnp.pi:.1f}°), φ={float(init_angles[1]):.3f} rad ({float(init_angles[1])*180/jnp.pi:.1f}°)")
            
            # Calculate angle opening
            true_dir = jnp.array([
                jnp.sin(true_angles[0]) * jnp.cos(true_angles[1]),
                jnp.sin(true_angles[0]) * jnp.sin(true_angles[1]),
                jnp.cos(true_angles[0])
            ])
            init_dir = jnp.array([
                jnp.sin(init_angles[0]) * jnp.cos(init_angles[1]),
                jnp.sin(init_angles[0]) * jnp.sin(init_angles[1]),
                jnp.cos(init_angles[0])
            ])
            cos_angle = jnp.clip(jnp.dot(true_dir, init_dir), -1.0, 1.0)
            angle_opening = jnp.arccos(cos_angle)
            print(f"  Direction angle opening: {float(angle_opening):.3f} rad ({float(angle_opening)*180/jnp.pi:.1f}°)")

        all_results['initial_guesses'].append(initial_params)
        
        # Run optimization (JIT-compatible version)
        params, loss_history, param_history = run_multi_objective_optimization_jit_compatible(
            params=initial_params,
            energy_grad_fn=energy_grad_fn,
            spatial_grad_fn=spatial_grad_fn,
            energy_optimizer=energy_optimizer,
            spatial_optimizer=spatial_optimizer,
            true_event_data=true_event_data,
            event_key=subkey1,
            n_iterations=n_iterations,
            patience=patience,
            position_scale=position_scale
        )
        
        # Store results
        all_results['loss_histories'].append(loss_history)
        all_results['param_histories'].append(param_history)
        all_results['final_params'].append(params)
        
        # Calculate errors (convert to Python types for storage)
        true_energy, true_position, true_angles = true_params
        final_energy, final_position, final_angles = params
        
        energy_error = float(jnp.abs(final_energy - true_energy) / true_energy)
        position_error = float(jnp.linalg.norm(final_position - true_position) / jnp.linalg.norm(true_position))
        angle_error_theta = float(jnp.abs(final_angles[0] - true_angles[0]) / jnp.maximum(true_angles[0], 1.0))
        angle_error_phi = float(jnp.mod(jnp.abs(final_angles[1] - true_angles[1]) / jnp.maximum(true_angles[1], 1.0), 2 * jnp.pi))
        
        event_errors = {
            'energy': energy_error,
            'position': position_error,
            'theta': angle_error_theta,
            'phi': angle_error_phi
        }
        all_results['final_errors'].append(event_errors)
    
    if verbose:
        print(f"\nCompleted optimization for all {N_events} events!")
    
    return all_results


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 7, 8, 9))
def run_multi_objective_optimization_jit_compatible(
    params, energy_grad_fn, spatial_grad_fn, energy_optimizer, spatial_optimizer,
    true_event_data, event_key, n_iterations=200, patience=100, position_scale=1.0):
    """
    JIT-compatible version following the original logic with separate patience counters
    and learning rate multipliers.
    """
    
    # Initialize optimizer states
    energy_opt_state = energy_optimizer.init(params)
    spatial_opt_state = spatial_optimizer.init(params)
    
    # Pre-allocate arrays for storing history
    loss_history = {
        'energy': jnp.zeros(n_iterations),
        'spatial': jnp.zeros(n_iterations),
        'total': jnp.zeros(n_iterations)
    }
    
    # Store parameter history as separate arrays
    energy_history = jnp.zeros(n_iterations)
    position_x_history = jnp.zeros(n_iterations)
    position_y_history = jnp.zeros(n_iterations)
    position_z_history = jnp.zeros(n_iterations)
    theta_history = jnp.zeros(n_iterations)
    phi_history = jnp.zeros(n_iterations)
    
    # Initial state includes all tracking variables from original
    init_state = (
        params,
        energy_opt_state,
        spatial_opt_state,
        loss_history,
        energy_history,
        position_x_history,
        position_y_history, 
        position_z_history,
        theta_history,
        phi_history,
        jnp.inf,  # best_energy_loss
        jnp.inf,  # best_spatial_loss
        0,        # energy_patience_counter
        0,        # spatial_patience_counter
        1.0,      # energy_lr_multiplier
        1.0       # spatial_lr_multiplier
    )
    
    def step_fn(i, state):
        (current_params, energy_opt_state, spatial_opt_state, loss_hist, 
         energy_hist, pos_x_hist, pos_y_hist, pos_z_hist, theta_hist, phi_hist,
         best_energy_loss, best_spatial_loss, energy_patience_counter, spatial_patience_counter,
         energy_lr_multiplier, spatial_lr_multiplier) = state
        
        # Compute gradients for each objective
        energy_loss, energy_grads = energy_grad_fn(current_params, true_event_data, event_key)
        spatial_loss, spatial_grads = spatial_grad_fn(current_params, true_event_data, event_key)
        total_loss = energy_loss + spatial_loss
        
        # Update loss history
        new_loss_hist = {
            'energy': loss_hist['energy'].at[i].set(energy_loss),
            'spatial': loss_hist['spatial'].at[i].set(spatial_loss),
            'total': loss_hist['total'].at[i].set(total_loss)
        }
        
        # Store parameters
        energy, position, direction_angles = current_params
        new_energy_hist = energy_hist.at[i].set(energy)
        new_pos_x_hist = pos_x_hist.at[i].set(position[0])
        new_pos_y_hist = pos_y_hist.at[i].set(position[1])
        new_pos_z_hist = pos_z_hist.at[i].set(position[2])
        new_theta_hist = theta_hist.at[i].set(direction_angles[0])
        new_phi_hist = phi_hist.at[i].set(direction_angles[1])
        
        # Check if energy loss improved
        energy_improved = energy_loss < best_energy_loss
        new_best_energy_loss = jnp.where(energy_improved, energy_loss, best_energy_loss)
        new_energy_patience_counter = jnp.where(energy_improved, 0, energy_patience_counter + 1)
        
        # Check if spatial loss improved  
        spatial_improved = spatial_loss < best_spatial_loss
        new_best_spatial_loss = jnp.where(spatial_improved, spatial_loss, best_spatial_loss)
        new_spatial_patience_counter = jnp.where(spatial_improved, 0, spatial_patience_counter + 1)
        
        # Update energy parameter
        energy_updates, new_energy_opt_state = energy_optimizer.update(
            energy_grads, energy_opt_state
        )
        
        # Update spatial parameters
        spatial_updates, new_spatial_opt_state = spatial_optimizer.update(
            spatial_grads, spatial_opt_state  
        )
        
        # Reduce learning rate if patience exceeded (following original logic)
        energy_patience_exceeded = new_energy_patience_counter >= (patience // 4)  # patience/4
        new_energy_lr_multiplier = jnp.where(energy_patience_exceeded, 
                                            energy_lr_multiplier * 0.5, 
                                            energy_lr_multiplier)
        # Reset patience counter when reducing learning rate
        new_energy_patience_counter = jnp.where(energy_patience_exceeded, 0, new_energy_patience_counter)
        
        spatial_patience_exceeded = new_spatial_patience_counter >= patience
        new_spatial_lr_multiplier = jnp.where(spatial_patience_exceeded,
                                             spatial_lr_multiplier * 0.5,
                                             spatial_lr_multiplier)
        # Reset patience counter when reducing learning rate  
        new_spatial_patience_counter = jnp.where(spatial_patience_exceeded, 0, new_spatial_patience_counter)
        
        # Combine updates - energy gets energy updates, spatial gets spatial updates
        energy_update, _, _ = energy_updates
        _, position_update, direction_update = spatial_updates
        
        # Scale updates based on learning rate multipliers
        scaled_energy_update = jax.tree.map(lambda x: x * new_energy_lr_multiplier, energy_update)
        scaled_position_update = jax.tree.map(lambda x: x * new_spatial_lr_multiplier, position_update)
        scaled_direction_update = jax.tree.map(lambda x: x * new_spatial_lr_multiplier, direction_update)
        
        # Conditional spatial updates (following original logic)
        w = 1.0  # You can modify this logic as needed
        # w = jnp.where(i < 20, 0.0, 1.0)  # Uncomment for conditional updates
        
        combined_updates = (
            scaled_energy_update, 
            scaled_position_update * position_scale * w, 
            scaled_direction_update * w
        )
        new_params = optax.apply_updates(current_params, combined_updates)
        
        return (new_params, new_energy_opt_state, new_spatial_opt_state, new_loss_hist,
                new_energy_hist, new_pos_x_hist, new_pos_y_hist, new_pos_z_hist, 
                new_theta_hist, new_phi_hist, new_best_energy_loss, new_best_spatial_loss,
                new_energy_patience_counter, new_spatial_patience_counter, 
                new_energy_lr_multiplier, new_spatial_lr_multiplier)
    
    # Run the full loop
    final_state = jax.lax.fori_loop(0, n_iterations, step_fn, init_state)
    
    (final_params, _, _, final_loss_hist, final_energy_hist, final_pos_x_hist, 
     final_pos_y_hist, final_pos_z_hist, final_theta_hist, final_phi_hist, 
     _, _, _, _, _, _) = final_state
    
    # Reconstruct param_history in same format as original
    param_history = {
        'energy': final_energy_hist,
        'position_x': final_pos_x_hist,
        'position_y': final_pos_y_hist,
        'position_z': final_pos_z_hist,
        'theta': final_theta_hist,
        'phi': final_phi_hist
    }
    
    return final_params, final_loss_hist, param_history

def filter_inf_results(results):
    """
    Filter out events that have 'inf' values in their loss histories.
    Detect whether infinities appear in energy loss, spatial loss, or both.
    
    Args:
        results: Dictionary containing optimization results with keys:
                'loss_histories', 'param_histories', 'true_params', 
                'initial_guesses', 'final_params', 'final_errors', 'seeds'
    
    Returns:
        new_results: Dictionary with same structure but inf events removed
    """
    
    # Find events without inf values in loss histories
    valid_event_indices = []
    
    # Track infinity occurrences
    energy_inf_count = 0
    spatial_inf_count = 0
    both_inf_count = 0
    total_inf_count = 0
    
    for i, loss_history in enumerate(results['loss_histories']):
        # Check for inf in each component
        has_inf_energy = np.any(np.isinf(loss_history['energy']))
        has_inf_spatial = np.any(np.isinf(loss_history['spatial']))
        has_inf_total = np.any(np.isinf(loss_history['total']))
        
        # Count which type of infinity occurred
        if has_inf_energy and has_inf_spatial:
            both_inf_count += 1
        elif has_inf_energy:
            energy_inf_count += 1
        elif has_inf_spatial:
            spatial_inf_count += 1
            
        # Event is valid only if it has no infinities in any component
        if not (has_inf_energy or has_inf_spatial or has_inf_total):
            valid_event_indices.append(i)
        else:
            total_inf_count += 1
    
    # Create new results dictionary with only valid events
    new_results = {
        'loss_histories': [results['loss_histories'][i] for i in valid_event_indices],
        'param_histories': [results['param_histories'][i] for i in valid_event_indices],
        'true_params': [results['true_params'][i] for i in valid_event_indices],
        'initial_guesses': [results['initial_guesses'][i] for i in valid_event_indices],
        'final_params': [results['final_params'][i] for i in valid_event_indices],
        'final_errors': [results['final_errors'][i] for i in valid_event_indices],
        'seeds': [results['seeds'][i] for i in valid_event_indices]
    }
    
    print(f"Filtered out {total_inf_count} events with inf values.")
    print(f"Infinity breakdown: {energy_inf_count} energy-only, {spatial_inf_count} spatial-only, {both_inf_count} both components")
    print(f"Remaining events: {len(valid_event_indices)}")
    
    return new_results

def angular_distance(angle1, angle2):
    """
    Calculate the minimum angular distance between two angles in radians.
    Accounts for 2π periodicity.
    
    Args:
        angle1, angle2: Angles in radians
        
    Returns:
        Minimum angular distance (always between 0 and π)
    """
    # Calculate the absolute difference
    diff = abs(angle1 - angle2) % (2 * np.pi)
    # Return the minimum of diff and 2π - diff
    return min(diff, 2 * np.pi - diff)

def angular_distance_jax(angle1, angle2):
    """JAX version of angular_distance for use in optimization."""
    diff = jnp.abs(angle1 - angle2) % (2 * jnp.pi)
    return jnp.minimum(diff, 2 * jnp.pi - diff)

# Updated error calculation in run_multi_event_optimization
def calculate_phi_error(final_phi, true_phi, true_phi_value):
    """
    Calculate phi error accounting for 2π periodicity.
    Returns relative error as a fraction.
    """
    angular_diff = angular_distance_jax(final_phi, true_phi)
    # For relative error, use angular difference divided by π (maximum possible angular distance)
    # or you could use a different normalization if preferred
    relative_error = angular_diff / jnp.pi
    return relative_error

def calculate_parameter_error(param_value, true_param, param_name):
    """
    Calculate error for a parameter, handling angular parameters properly.
    
    Args:
        param_value: Current parameter value
        true_param: True parameter value
        param_name: Name of the parameter
        
    Returns:
        Absolute error (angular distance for phi, regular difference for others)
    """
    if param_name == 'phi':
        return angular_distance(param_value, true_param)
    else:
        return abs(param_value - true_param)

def get_true_param_value(true_params, param_name):
    """Helper function to extract parameter values from true_params tuple."""
    true_energy, true_position, true_angles = true_params
    
    if param_name == 'energy':
        return float(true_energy)
    elif param_name == 'position_x':
        return float(true_position[0])
    elif param_name == 'position_y':
        return float(true_position[1])
    elif param_name == 'position_z':
        return float(true_position[2])
    elif param_name == 'theta':
        return float(true_angles[0])
    elif param_name == 'phi':
        return float(true_angles[1])
    else:
        raise ValueError(f"Unknown parameter name: {param_name}")

def plot_multi_event_convergence(results, save_path=None, show_individual=True, show_statistics=True, figsize=(9, 6)):
    """
    Create comprehensive visualization of multi-event optimization convergence.
    Shows parameter errors in a 3x3 grid layout.
    """
    
    N_events = len(results['loss_histories'])
    n_iterations = len(results['loss_histories'][0]['total'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Top row: Energy absolute error, Euclidean distance, Angle opening
    
    # Plot 1: Energy absolute error
    ax1 = plt.subplot(3, 3, 1)
    
    if show_individual:
        for i in range(N_events):
            param_hist = results['param_histories'][i]['energy']
            true_param = get_true_param_value(results['true_params'][i], 'energy')
            
            abs_error = [calculate_parameter_error(p, true_param, 'energy') for p in param_hist]
            plt.plot(abs_error, alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_abs_errors = []
        for i in range(N_events):
            param_hist = results['param_histories'][i]['energy']
            true_param = get_true_param_value(results['true_params'][i], 'energy')
            
            abs_error = [calculate_parameter_error(p, true_param, 'energy') for p in param_hist]
            all_abs_errors.append(abs_error)
        
        abs_error_array = np.array(all_abs_errors)
        mean_abs_error = np.mean(abs_error_array, axis=0)
        std_abs_error = np.std(abs_error_array, axis=0)
        median_abs_error = np.median(abs_error_array, axis=0)
        
        iterations = range(n_iterations)
        plt.plot(iterations, mean_abs_error, 'r-', linewidth=2, label=f'Mean (N={N_events})')
        plt.fill_between(iterations, mean_abs_error - std_abs_error, 
                       mean_abs_error + std_abs_error, alpha=0.2, color='red', label='±1σ')
        plt.plot(iterations, median_abs_error, 'g--', linewidth=2, label='Median')
    
    plt.xlabel('Iteration')
    plt.ylabel('Absolute Error')
    plt.title('Energy Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Euclidean distance
    ax2 = plt.subplot(3, 3, 2)
    
    if show_individual:
        for i in range(N_events):
            param_hist = results['param_histories'][i]
            true_pos = results['true_params'][i][1]
            
            pos_distances = []
            for j in range(n_iterations):
                reconstructed_pos = np.array([
                    param_hist['position_x'][j],
                    param_hist['position_y'][j],
                    param_hist['position_z'][j]
                ])
                distance = np.linalg.norm(reconstructed_pos - np.array(true_pos))
                pos_distances.append(distance)
            
            plt.plot(pos_distances, alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_pos_distances = []
        for i in range(N_events):
            param_hist = results['param_histories'][i]
            true_pos = results['true_params'][i][1]
            
            pos_distances = []
            for j in range(n_iterations):
                reconstructed_pos = np.array([
                    param_hist['position_x'][j],
                    param_hist['position_y'][j],
                    param_hist['position_z'][j]
                ])
                distance = np.linalg.norm(reconstructed_pos - np.array(true_pos))
                pos_distances.append(distance)
            
            all_pos_distances.append(pos_distances)
        
        pos_distance_array = np.array(all_pos_distances)
        mean_pos_distance = np.mean(pos_distance_array, axis=0)
        std_pos_distance = np.std(pos_distance_array, axis=0)
        median_pos_distance = np.median(pos_distance_array, axis=0)
        
        iterations = range(n_iterations)
        plt.plot(iterations, mean_pos_distance, 'r-', linewidth=2, label='Mean')
        plt.fill_between(iterations, mean_pos_distance - std_pos_distance, 
                       mean_pos_distance + std_pos_distance, alpha=0.2, color='red')
        plt.plot(iterations, median_pos_distance, 'g--', linewidth=2, label='Median')
    
    plt.xlabel('Iteration')
    plt.ylabel('Euclidean Distance')
    plt.title('Position Distance Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Angle opening from true track direction
    ax3 = plt.subplot(3, 3, 3)
    
    if show_individual:
        for i in range(N_events):
            param_hist = results['param_histories'][i]
            true_theta = get_true_param_value(results['true_params'][i], 'theta')
            true_phi = get_true_param_value(results['true_params'][i], 'phi')
            
            # Calculate true direction vector
            true_dir = np.array([
                np.sin(true_theta) * np.cos(true_phi),
                np.sin(true_theta) * np.sin(true_phi),
                np.cos(true_theta)
            ])
            
            angle_openings = []
            for j in range(n_iterations):
                # Calculate reconstructed direction vector
                recon_theta = param_hist['theta'][j]
                recon_phi = param_hist['phi'][j]
                recon_dir = np.array([
                    np.sin(recon_theta) * np.cos(recon_phi),
                    np.sin(recon_theta) * np.sin(recon_phi),
                    np.cos(recon_theta)
                ])
                
                # Calculate angle between vectors
                cos_angle = np.clip(np.dot(true_dir, recon_dir), -1.0, 1.0)
                angle_opening = np.arccos(cos_angle)
                angle_openings.append(angle_opening)
            
            plt.plot(angle_openings, alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_angle_openings = []
        for i in range(N_events):
            param_hist = results['param_histories'][i]
            true_theta = get_true_param_value(results['true_params'][i], 'theta')
            true_phi = get_true_param_value(results['true_params'][i], 'phi')
            
            # Calculate true direction vector
            true_dir = np.array([
                np.sin(true_theta) * np.cos(true_phi),
                np.sin(true_theta) * np.sin(true_phi),
                np.cos(true_theta)
            ])
            
            angle_openings = []
            for j in range(n_iterations):
                # Calculate reconstructed direction vector
                recon_theta = param_hist['theta'][j]
                recon_phi = param_hist['phi'][j]
                recon_dir = np.array([
                    np.sin(recon_theta) * np.cos(recon_phi),
                    np.sin(recon_theta) * np.sin(recon_phi),
                    np.cos(recon_theta)
                ])
                
                # Calculate angle between vectors
                cos_angle = np.clip(np.dot(true_dir, recon_dir), -1.0, 1.0)
                angle_opening = np.arccos(cos_angle)
                angle_openings.append(angle_opening)
            
            all_angle_openings.append(angle_openings)
        
        angle_opening_array = np.array(all_angle_openings)
        mean_angle_opening = np.mean(angle_opening_array, axis=0)
        std_angle_opening = np.std(angle_opening_array, axis=0)
        median_angle_opening = np.median(angle_opening_array, axis=0)
        
        iterations = range(n_iterations)
        plt.plot(iterations, mean_angle_opening, 'r-', linewidth=2, label='Mean')
        plt.fill_between(iterations, mean_angle_opening - std_angle_opening, 
                       mean_angle_opening + std_angle_opening, alpha=0.2, color='red')
        plt.plot(iterations, median_angle_opening, 'g--', linewidth=2, label='Median')
    
    plt.xlabel('Iteration')
    plt.ylabel('Angle Opening (rad)')
    plt.title('Track Direction Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Mid row: X, Y, Z absolute errors
    
    # Plot 4: X absolute error
    ax4 = plt.subplot(3, 3, 4)
    
    if show_individual:
        for i in range(N_events):
            param_hist = results['param_histories'][i]['position_x']
            true_param = get_true_param_value(results['true_params'][i], 'position_x')
            
            abs_error = [calculate_parameter_error(p, true_param, 'position_x') for p in param_hist]
            plt.plot(abs_error, alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_abs_errors = []
        for i in range(N_events):
            param_hist = results['param_histories'][i]['position_x']
            true_param = get_true_param_value(results['true_params'][i], 'position_x')
            
            abs_error = [calculate_parameter_error(p, true_param, 'position_x') for p in param_hist]
            all_abs_errors.append(abs_error)
        
        abs_error_array = np.array(all_abs_errors)
        mean_abs_error = np.mean(abs_error_array, axis=0)
        std_abs_error = np.std(abs_error_array, axis=0)
        median_abs_error = np.median(abs_error_array, axis=0)
        
        iterations = range(n_iterations)
        plt.plot(iterations, mean_abs_error, 'r-', linewidth=2, label='Mean')
        plt.fill_between(iterations, mean_abs_error - std_abs_error, 
                       mean_abs_error + std_abs_error, alpha=0.2, color='red')
        plt.plot(iterations, median_abs_error, 'g--', linewidth=2, label='Median')
    
    plt.xlabel('Iteration')
    plt.ylabel('Absolute Error')
    plt.title('Position X Convergence')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Y absolute error
    ax5 = plt.subplot(3, 3, 5)
    
    if show_individual:
        for i in range(N_events):
            param_hist = results['param_histories'][i]['position_y']
            true_param = get_true_param_value(results['true_params'][i], 'position_y')
            
            abs_error = [calculate_parameter_error(p, true_param, 'position_y') for p in param_hist]
            plt.plot(abs_error, alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_abs_errors = []
        for i in range(N_events):
            param_hist = results['param_histories'][i]['position_y']
            true_param = get_true_param_value(results['true_params'][i], 'position_y')
            
            abs_error = [calculate_parameter_error(p, true_param, 'position_y') for p in param_hist]
            all_abs_errors.append(abs_error)
        
        abs_error_array = np.array(all_abs_errors)
        mean_abs_error = np.mean(abs_error_array, axis=0)
        std_abs_error = np.std(abs_error_array, axis=0)
        median_abs_error = np.median(abs_error_array, axis=0)
        
        iterations = range(n_iterations)
        plt.plot(iterations, mean_abs_error, 'r-', linewidth=2, label='Mean')
        plt.fill_between(iterations, mean_abs_error - std_abs_error, 
                       mean_abs_error + std_abs_error, alpha=0.2, color='red')
        plt.plot(iterations, median_abs_error, 'g--', linewidth=2, label='Median')
    
    plt.xlabel('Iteration')
    plt.ylabel('Absolute Error')
    plt.title('Position Y Convergence')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Z absolute error
    ax6 = plt.subplot(3, 3, 6)
    
    if show_individual:
        for i in range(N_events):
            param_hist = results['param_histories'][i]['position_z']
            true_param = get_true_param_value(results['true_params'][i], 'position_z')
            
            abs_error = [calculate_parameter_error(p, true_param, 'position_z') for p in param_hist]
            plt.plot(abs_error, alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_abs_errors = []
        for i in range(N_events):
            param_hist = results['param_histories'][i]['position_z']
            true_param = get_true_param_value(results['true_params'][i], 'position_z')
            
            abs_error = [calculate_parameter_error(p, true_param, 'position_z') for p in param_hist]
            all_abs_errors.append(abs_error)
        
        abs_error_array = np.array(all_abs_errors)
        mean_abs_error = np.mean(abs_error_array, axis=0)
        std_abs_error = np.std(abs_error_array, axis=0)
        median_abs_error = np.median(abs_error_array, axis=0)
        
        iterations = range(n_iterations)
        plt.plot(iterations, mean_abs_error, 'r-', linewidth=2, label='Mean')
        plt.fill_between(iterations, mean_abs_error - std_abs_error, 
                       mean_abs_error + std_abs_error, alpha=0.2, color='red')
        plt.plot(iterations, median_abs_error, 'g--', linewidth=2, label='Median')
    
    plt.xlabel('Iteration')
    plt.ylabel('Absolute Error')
    plt.title('Position Z Convergence')
    plt.grid(True, alpha=0.3)
    
    # Bottom row: Theta and Phi errors
    
    # Plot 7: Theta error
    ax7 = plt.subplot(3, 3, 7)
    
    if show_individual:
        for i in range(N_events):
            param_hist = results['param_histories'][i]['theta']
            true_param = get_true_param_value(results['true_params'][i], 'theta')
            
            abs_error = [calculate_parameter_error(p, true_param, 'theta') for p in param_hist]
            plt.plot(abs_error, alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_abs_errors = []
        for i in range(N_events):
            param_hist = results['param_histories'][i]['theta']
            true_param = get_true_param_value(results['true_params'][i], 'theta')
            
            abs_error = [calculate_parameter_error(p, true_param, 'theta') for p in param_hist]
            all_abs_errors.append(abs_error)
        
        abs_error_array = np.array(all_abs_errors)
        mean_abs_error = np.mean(abs_error_array, axis=0)
        std_abs_error = np.std(abs_error_array, axis=0)
        median_abs_error = np.median(abs_error_array, axis=0)
        
        iterations = range(n_iterations)
        plt.plot(iterations, mean_abs_error, 'r-', linewidth=2, label='Mean')
        plt.fill_between(iterations, mean_abs_error - std_abs_error, 
                       mean_abs_error + std_abs_error, alpha=0.2, color='red')
        plt.plot(iterations, median_abs_error, 'g--', linewidth=2, label='Median')
    
    plt.xlabel('Iteration')
    plt.ylabel('Absolute Error')
    plt.title('θ Convergence')
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Phi error
    ax8 = plt.subplot(3, 3, 8)
    
    if show_individual:
        for i in range(N_events):
            param_hist = results['param_histories'][i]['phi']
            true_param = get_true_param_value(results['true_params'][i], 'phi')
            
            abs_error = [calculate_parameter_error(p, true_param, 'phi') for p in param_hist]
            plt.plot(abs_error, alpha=0.3, color='blue', linewidth=0.5)
    
    if show_statistics:
        all_abs_errors = []
        for i in range(N_events):
            param_hist = results['param_histories'][i]['phi']
            true_param = get_true_param_value(results['true_params'][i], 'phi')
            
            abs_error = [calculate_parameter_error(p, true_param, 'phi') for p in param_hist]
            all_abs_errors.append(abs_error)
        
        abs_error_array = np.array(all_abs_errors)
        mean_abs_error = np.mean(abs_error_array, axis=0)
        std_abs_error = np.std(abs_error_array, axis=0)
        median_abs_error = np.median(abs_error_array, axis=0)
        
        iterations = range(n_iterations)
        plt.plot(iterations, mean_abs_error, 'r-', linewidth=2, label='Mean')
        plt.fill_between(iterations, mean_abs_error - std_abs_error, 
                       mean_abs_error + std_abs_error, alpha=0.2, color='red')
        plt.plot(iterations, median_abs_error, 'g--', linewidth=2, label='Median')
    
    plt.xlabel('Iteration')
    plt.ylabel('Angular Distance (rad)')
    plt.title('φ Convergence (Angular Distance)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


import numpy as np
import matplotlib.pyplot as plt


def plot_single_event_convergence(event_idx, results, save_path=None, figsize=(12, 9)):
    """
    Create comprehensive visualization of single-event optimization convergence.
    Shows parameter convergence for one specific event in a 3x3 grid.
    
    Args:
        event_idx (int): Index of the event to plot (0-based)
        results (dict): Results dictionary from multi-event optimization
        save_path (str, optional): Path to save the plot
        figsize (tuple): Figure size (width, height)
    """
    
    # Validate event index
    N_events = len(results['loss_histories'])
    if event_idx >= N_events or event_idx < 0:
        raise ValueError(f"Event index {event_idx} out of range. Available events: 0 to {N_events-1}")
    
    # Extract data for the specific event
    param_hist = results['param_histories'][event_idx]
    true_params = results['true_params'][event_idx]
    
    n_iterations = len(param_hist['energy'])
    iterations = range(n_iterations)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f'Event {event_idx} Parameter Convergence', y=0.98)
    
    # Top row: Energy, Euclidean distance, Angle opening
    
    # Plot 1: Energy convergence
    ax1 = plt.subplot(3, 3, 1)
    true_energy = get_true_param_value(true_params, 'energy')
    energy_errors = [abs(e - true_energy) for e in param_hist['energy']]
    
    plt.plot(iterations, energy_errors, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Absolute Error')
    plt.title('Energy Convergence')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Position Distance (Euclidean)
    ax2 = plt.subplot(3, 3, 2)
    true_pos = np.array(true_params[1])
    pos_distances = []
    for j in range(n_iterations):
        reconstructed_pos = np.array([
            param_hist['position_x'][j],
            param_hist['position_y'][j],
            param_hist['position_z'][j]
        ])
        distance = np.linalg.norm(reconstructed_pos - true_pos)
        pos_distances.append(distance)
    
    plt.plot(iterations, pos_distances, 'g-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Euclidean Distance')
    plt.title('Position Distance from Truth')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Angle opening from true track direction
    ax3 = plt.subplot(3, 3, 3)
    true_theta = get_true_param_value(true_params, 'theta')
    true_phi = get_true_param_value(true_params, 'phi')
    
    # Calculate true direction vector
    true_dir = np.array([
        np.sin(true_theta) * np.cos(true_phi),
        np.sin(true_theta) * np.sin(true_phi),
        np.cos(true_theta)
    ])
    
    angle_openings = []
    for j in range(n_iterations):
        # Calculate reconstructed direction vector
        recon_theta = param_hist['theta'][j]
        recon_phi = param_hist['phi'][j]
        recon_dir = np.array([
            np.sin(recon_theta) * np.cos(recon_phi),
            np.sin(recon_theta) * np.sin(recon_phi),
            np.cos(recon_theta)
        ])
        
        # Calculate angle between vectors
        cos_angle = np.clip(np.dot(true_dir, recon_dir), -1.0, 1.0)
        angle_opening = np.arccos(cos_angle)
        angle_openings.append(angle_opening)
    
    plt.plot(iterations, angle_openings, 'm-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Angle Opening (rad)')
    plt.title('Track Direction Convergence')
    plt.grid(True, alpha=0.3)
    
    # Mid row: X, Y, Z absolute errors
    
    # Plot 4: X position convergence
    ax4 = plt.subplot(3, 3, 4)
    true_x = get_true_param_value(true_params, 'position_x')
    x_errors = [abs(x - true_x) for x in param_hist['position_x']]
    
    plt.plot(iterations, x_errors, 'r-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Absolute Error')
    plt.title('Position X Convergence')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Y position convergence
    ax5 = plt.subplot(3, 3, 5)
    true_y = get_true_param_value(true_params, 'position_y')
    y_errors = [abs(y - true_y) for y in param_hist['position_y']]
    
    plt.plot(iterations, y_errors, 'orange', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Absolute Error')
    plt.title('Position Y Convergence')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Z position convergence
    ax6 = plt.subplot(3, 3, 6)
    true_z = get_true_param_value(true_params, 'position_z')
    z_errors = [abs(z - true_z) for z in param_hist['position_z']]
    
    plt.plot(iterations, z_errors, 'purple', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Absolute Error')
    plt.title('Position Z Convergence')
    plt.grid(True, alpha=0.3)
    
    # Bottom row: Theta, Phi, Summary
    
    # Plot 7: Theta convergence
    ax7 = plt.subplot(3, 3, 7)
    theta_errors = [calculate_parameter_error(theta, true_theta, 'theta') for theta in param_hist['theta']]
    
    plt.plot(iterations, theta_errors, 'c-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Absolute Error')
    plt.title('θ Convergence')
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Phi convergence
    ax8 = plt.subplot(3, 3, 8)
    phi_errors = [calculate_parameter_error(phi, true_phi, 'phi') for phi in param_hist['phi']]
    
    plt.plot(iterations, phi_errors, 'brown', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Angular Distance (rad)')
    plt.title('φ Convergence (Angular Distance)')
    plt.grid(True, alpha=0.3)
    
    # Plot 9: Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')  # Turn off axes
    
    # Calculate final errors for top row quantities only
    final_energy_error = energy_errors[-1]
    final_pos_error = pos_distances[-1]
    final_angle_opening = angle_openings[-1]
    
    # Create summary text for top row only
    summary_text = f"""Final Results Summary:

Top Row Convergence:
Energy Error: {final_energy_error:.4f}
Position Distance: {final_pos_error:.4f}
Angle Opening: {final_angle_opening:.4f} rad

True Parameters:
Energy: {true_energy:.4f}
Position: ({true_params[1][0]:.2f}, {true_params[1][1]:.2f}, {true_params[1][2]:.2f})
θ: {true_theta:.4f} rad
φ: {true_phi:.4f} rad"""
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Single event plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary to console as well
    print(f"\n=== Event {event_idx} Summary ===")
    print(f"Final Parameter Errors (Top Row):")
    print(f"  Energy: {final_energy_error:.4f}")
    print(f"  Position Distance: {final_pos_error:.4f}")
    print(f"  Angle Opening: {final_angle_opening:.4f} rad")

def plot_single_event_comparison(event_indices, results, save_path=None, figsize=(14, 8)):
    """
    Compare convergence of multiple single events side by side.
    
    Args:
        event_indices (list): List of event indices to compare
        results (dict): Results dictionary from multi-event optimization
        save_path (str, optional): Path to save the plot
        figsize (tuple): Figure size (width, height)
    """
    
    n_events = len(event_indices)
    fig, axes = plt.subplots(2, n_events, figsize=figsize)
    if n_events == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(f'Event Comparison: {event_indices}')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, event_idx in enumerate(event_indices):
        # Validate event index
        N_events = len(results['loss_histories'])
        if event_idx >= N_events or event_idx < 0:
            print(f"Warning: Event index {event_idx} out of range. Skipping.")
            continue
            
        loss_hist = results['loss_histories'][event_idx]
        param_hist = results['param_histories'][event_idx]
        true_params = results['true_params'][event_idx]
        
        n_iterations = len(loss_hist['total'])
        iterations = range(n_iterations)
        
        # Top row: Loss convergence
        ax_top = axes[0, i]
        ax_top.plot(iterations, loss_hist['total'], color=colors[i % len(colors)], 
                   linewidth=2, label='Total')
        ax_top.plot(iterations, loss_hist['energy'], color=colors[i % len(colors)], 
                   linewidth=1, linestyle='--', alpha=0.7, label='Energy')
        ax_top.plot(iterations, loss_hist['spatial'], color=colors[i % len(colors)], 
                   linewidth=1, linestyle=':', alpha=0.7, label='Spatial')
        ax_top.set_yscale('log')
        ax_top.set_xlabel('Iteration')
        ax_top.set_ylabel('Loss')
        ax_top.set_title(f'Event {event_idx} - Loss')
        ax_top.legend()
        ax_top.grid(True, alpha=0.3)
        
        # Bottom row: Position distance
        ax_bottom = axes[1, i]
        true_pos = np.array(true_params[1])
        pos_distances = []
        for j in range(n_iterations):
            reconstructed_pos = np.array([
                param_hist['position_x'][j],
                param_hist['position_y'][j],
                param_hist['position_z'][j]
            ])
            distance = np.linalg.norm(reconstructed_pos - true_pos)
            pos_distances.append(distance)
        
        ax_bottom.plot(iterations, pos_distances, color=colors[i % len(colors)], linewidth=2)
        ax_bottom.set_xlabel('Iteration')
        ax_bottom.set_ylabel('Position Distance')
        ax_bottom.set_title(f'Event {event_idx} - Position Error')
        ax_bottom.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Event comparison plot saved to: {save_path}")
    
    plt.show()



@jit
def get_initial_guess(
        charges: jnp.ndarray,
        detector_points: jnp.ndarray,
        energy_scaling_factor: float = 0.0213,
        energy_scale_intercept: float = 213.11,
        key: jnp.ndarray = None,
        offset: float = 0.5
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate initial parameter guesses from detector hits.
    Assumes photons originate from center and form a cone to the detector wall.
    Only considers nonzero charge hits for calculations.

    Parameters
    ----------
    charges : jnp.ndarray
        Array of charge values for detector hits
    detector_points : jnp.ndarray
        Array of detector positions where hits occurred
    intensity_scale : float
        Scale factor for intensity estimation
    key : jnp.ndarray
        Key for random initial position generation
    offset : float
        Offset from the wall used for random initial position generation

    Returns
    -------
    Tuple containing:
        opening_angle : jnp.ndarray (scalar)
        position : jnp.ndarray (3,)
        direction : jnp.ndarray (3,)
    """
    if key is not None:
        position = generate_random_point_inside_cylinder(key, offset=offset)
    else:
        position = jnp.zeros(3)

    # Get average direction by weighting detector positions by charge
    total_charge = jnp.sum(charges)
    charge_weights = charges / (total_charge + 1e-8)

    vectors_to_hits = detector_points - position
    vectors_to_hits = vectors_to_hits / (jnp.linalg.norm(vectors_to_hits, axis=1, keepdims=True) + 1e-8)

    # Get direction from vectors
    direction = jnp.sum(vectors_to_hits * charge_weights[:, None], axis=0)
    direction = direction / (jnp.linalg.norm(direction) + 1e-8)

    total_energy = total_charge * energy_scaling_factor + energy_scale_intercept
    total_energy = jnp.maximum(total_energy, 180.0)  # Ensure minimum energy is 180.0

    return (
        total_energy,
        position,
        direction
    )


def plot_simple_event_convergence(event_idx, results, save_path=None, figsize=(15, 4)):
    """
    Create simple visualization of single-event optimization convergence.
    Shows only the three key convergence metrics in a horizontal layout.
    
    Args:
        event_idx (int): Index of the event to plot (0-based)
        results (dict): Results dictionary from multi-event optimization
        save_path (str, optional): Path to save the plot
        figsize (tuple): Figure size (width, height)
    """
    
    # Validate event index
    N_events = len(results['loss_histories'])
    if event_idx >= N_events or event_idx < 0:
        raise ValueError(f"Event index {event_idx} out of range. Available events: 0 to {N_events-1}")
    
    # Extract data for the specific event
    param_hist = results['param_histories'][event_idx]
    true_params = results['true_params'][event_idx]
    
    n_iterations = len(param_hist['energy'])
    iterations = range(n_iterations)
    
    # Create figure with 1x3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    #fig.suptitle(f'Event {event_idx} - Key Convergence Metrics', fontsize=14, y=1.02)
    
    # Plot 1: Energy convergence
    true_energy = get_true_param_value(true_params, 'energy')
    energy_errors = [abs(e - true_energy) for e in param_hist['energy']]
    
    ax1.plot(iterations, energy_errors, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Energy Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Position Distance (Euclidean)
    true_pos = np.array(true_params[1])
    pos_distances = []
    for j in range(n_iterations):
        reconstructed_pos = np.array([
            param_hist['position_x'][j],
            param_hist['position_y'][j],
            param_hist['position_z'][j]
        ])
        distance = np.linalg.norm(reconstructed_pos - true_pos)
        pos_distances.append(distance)
    
    ax2.plot(iterations, pos_distances, 'g-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Euclidean Distance')
    ax2.set_title('Position Distance from Truth')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Angle opening from true track direction
    true_theta = get_true_param_value(true_params, 'theta')
    true_phi = get_true_param_value(true_params, 'phi')
    
    # Calculate true direction vector
    true_dir = np.array([
        np.sin(true_theta) * np.cos(true_phi),
        np.sin(true_theta) * np.sin(true_phi),
        np.cos(true_theta)
    ])
    
    angle_openings = []
    for j in range(n_iterations):
        # Calculate reconstructed direction vector
        recon_theta = param_hist['theta'][j]
        recon_phi = param_hist['phi'][j]
        recon_dir = np.array([
            np.sin(recon_theta) * np.cos(recon_phi),
            np.sin(recon_theta) * np.sin(recon_phi),
            np.cos(recon_theta)
        ])
        
        # Calculate angle between vectors
        cos_angle = np.clip(np.dot(true_dir, recon_dir), -1.0, 1.0)
        angle_opening = np.arccos(cos_angle)
        angle_openings.append(angle_opening)
    
    ax3.plot(iterations, angle_openings, 'm-', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Angle Opening (rad)')
    ax3.set_title('Track Direction Convergence')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Simple convergence plot saved to: {save_path}")
    
    plt.show()
    
    # Calculate and print final errors
    final_energy_error = energy_errors[-1]
    final_pos_error = pos_distances[-1]
    final_angle_opening = angle_openings[-1]
    
    print(f"\n=== Event {event_idx} Final Results ===")
    print(f"Energy Error: {final_energy_error:.4f}")
    print(f"Position Distance: {final_pos_error:.4f}")
    print(f"Angle Opening: {final_angle_opening:.4f} rad ({np.degrees(final_angle_opening):.2f}°)")
    
    return final_energy_error, final_pos_error, final_angle_opening


def plot_simple_multi_event_convergence(results, save_path=None, show_individual=True, 
                                       show_statistics=True, show_histograms=False, figsize=None):
    """
    Create simple visualization of multi-event optimization convergence.
    Shows only the three key convergence metrics with optional histograms.
    
    Args:
        results (dict): Results dictionary from multi-event optimization
        save_path (str, optional): Path to save the plot
        show_individual (bool): Show individual event traces
        show_statistics (bool): Show mean/median/std statistics
        show_histograms (bool): Show histograms of final iteration values
        figsize (tuple, optional): Figure size (width, height). Auto-calculated if None.
    """
    
    N_events = len(results['loss_histories'])
    n_iterations = len(results['loss_histories'][0]['total'])
    
    # Determine layout and figure size
    if show_histograms:
        nrows, ncols = 2, 3
        if figsize is None:
            figsize = (15, 8)
    else:
        nrows, ncols = 1, 3
        if figsize is None:
            figsize = (15, 4)
    
    # Create figure with subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    fig.suptitle(f'Multi-Event Convergence Summary (N={N_events} events)', fontsize=14, y=0.98)
    
    # Convergence plots (top row)
    if show_histograms:
        convergence_axes = axes[0]  # First row of 2D array
    else:
        convergence_axes = axes     # 1D array of 3 axes
    
    # Plot 1: Energy absolute error
    ax1 = convergence_axes[0]
    all_energy_errors = []
    
    if show_individual:
        for i in range(N_events):
            param_hist = results['param_histories'][i]['energy']
            true_param = get_true_param_value(results['true_params'][i], 'energy')
            
            abs_error = [calculate_parameter_error(p, true_param, 'energy') for p in param_hist]
            all_energy_errors.append(abs_error)
            ax1.plot(abs_error, alpha=0.3, color='blue', linewidth=0.5)
    else:
        for i in range(N_events):
            param_hist = results['param_histories'][i]['energy']
            true_param = get_true_param_value(results['true_params'][i], 'energy')
            abs_error = [calculate_parameter_error(p, true_param, 'energy') for p in param_hist]
            all_energy_errors.append(abs_error)
    
    if show_statistics:
        abs_error_array = np.array(all_energy_errors)
        mean_abs_error = np.mean(abs_error_array, axis=0)
        std_abs_error = np.std(abs_error_array, axis=0)
        median_abs_error = np.median(abs_error_array, axis=0)
        
        iterations = range(n_iterations)
        ax1.plot(iterations, mean_abs_error, 'r-', linewidth=2, label=f'Mean (N={N_events})')
        ax1.fill_between(iterations, mean_abs_error - std_abs_error, 
                        mean_abs_error + std_abs_error, alpha=0.2, color='red', label='±1σ')
        ax1.plot(iterations, median_abs_error, 'g--', linewidth=2, label='Median')
        ax1.legend()
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Energy Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Euclidean distance
    ax2 = convergence_axes[1]
    all_pos_distances = []
    
    if show_individual:
        for i in range(N_events):
            param_hist = results['param_histories'][i]
            true_pos = results['true_params'][i][1]
            
            pos_distances = []
            for j in range(n_iterations):
                reconstructed_pos = np.array([
                    param_hist['position_x'][j],
                    param_hist['position_y'][j],
                    param_hist['position_z'][j]
                ])
                distance = np.linalg.norm(reconstructed_pos - np.array(true_pos))
                pos_distances.append(distance)
            
            all_pos_distances.append(pos_distances)
            ax2.plot(pos_distances, alpha=0.3, color='blue', linewidth=0.5)
    else:
        for i in range(N_events):
            param_hist = results['param_histories'][i]
            true_pos = results['true_params'][i][1]
            
            pos_distances = []
            for j in range(n_iterations):
                reconstructed_pos = np.array([
                    param_hist['position_x'][j],
                    param_hist['position_y'][j],
                    param_hist['position_z'][j]
                ])
                distance = np.linalg.norm(reconstructed_pos - np.array(true_pos))
                pos_distances.append(distance)
            all_pos_distances.append(pos_distances)
    
    if show_statistics:
        pos_distance_array = np.array(all_pos_distances)
        mean_pos_distance = np.mean(pos_distance_array, axis=0)
        std_pos_distance = np.std(pos_distance_array, axis=0)
        median_pos_distance = np.median(pos_distance_array, axis=0)
        
        iterations = range(n_iterations)
        ax2.plot(iterations, mean_pos_distance, 'r-', linewidth=2, label='Mean')
        ax2.fill_between(iterations, mean_pos_distance - std_pos_distance, 
                        mean_pos_distance + std_pos_distance, alpha=0.2, color='red')
        ax2.plot(iterations, median_pos_distance, 'g--', linewidth=2, label='Median')
        ax2.legend()
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Euclidean Distance')
    ax2.set_title('Position Distance Convergence')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Angle opening from true track direction
    ax3 = convergence_axes[2]
    all_angle_openings = []
    
    if show_individual:
        for i in range(N_events):
            param_hist = results['param_histories'][i]
            true_theta = get_true_param_value(results['true_params'][i], 'theta')
            true_phi = get_true_param_value(results['true_params'][i], 'phi')
            
            # Calculate true direction vector
            true_dir = np.array([
                np.sin(true_theta) * np.cos(true_phi),
                np.sin(true_theta) * np.sin(true_phi),
                np.cos(true_theta)
            ])
            
            angle_openings = []
            for j in range(n_iterations):
                # Calculate reconstructed direction vector
                recon_theta = param_hist['theta'][j]
                recon_phi = param_hist['phi'][j]
                recon_dir = np.array([
                    np.sin(recon_theta) * np.cos(recon_phi),
                    np.sin(recon_theta) * np.sin(recon_phi),
                    np.cos(recon_theta)
                ])
                
                # Calculate angle between vectors
                cos_angle = np.clip(np.dot(true_dir, recon_dir), -1.0, 1.0)
                angle_opening = np.arccos(cos_angle)
                angle_openings.append(angle_opening)
            
            all_angle_openings.append(angle_openings)
            ax3.plot(angle_openings, alpha=0.3, color='blue', linewidth=0.5)
    else:
        for i in range(N_events):
            param_hist = results['param_histories'][i]
            true_theta = get_true_param_value(results['true_params'][i], 'theta')
            true_phi = get_true_param_value(results['true_params'][i], 'phi')
            
            # Calculate true direction vector
            true_dir = np.array([
                np.sin(true_theta) * np.cos(true_phi),
                np.sin(true_theta) * np.sin(true_phi),
                np.cos(true_theta)
            ])
            
            angle_openings = []
            for j in range(n_iterations):
                # Calculate reconstructed direction vector
                recon_theta = param_hist['theta'][j]
                recon_phi = param_hist['phi'][j]
                recon_dir = np.array([
                    np.sin(recon_theta) * np.cos(recon_phi),
                    np.sin(recon_theta) * np.sin(recon_phi),
                    np.cos(recon_theta)
                ])
                
                # Calculate angle between vectors
                cos_angle = np.clip(np.dot(true_dir, recon_dir), -1.0, 1.0)
                angle_opening = np.arccos(cos_angle)
                angle_openings.append(angle_opening)
            
            all_angle_openings.append(angle_openings)
    
    if show_statistics:
        angle_opening_array = np.array(all_angle_openings)
        mean_angle_opening = np.mean(angle_opening_array, axis=0)
        std_angle_opening = np.std(angle_opening_array, axis=0)
        median_angle_opening = np.median(angle_opening_array, axis=0)
        
        iterations = range(n_iterations)
        ax3.plot(iterations, mean_angle_opening, 'r-', linewidth=2, label='Mean')
        ax3.fill_between(iterations, mean_angle_opening - std_angle_opening, 
                        mean_angle_opening + std_angle_opening, alpha=0.2, color='red')
        ax3.plot(iterations, median_angle_opening, 'g--', linewidth=2, label='Median')
        ax3.legend()
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Angle Opening (rad)')
    ax3.set_title('Track Direction Convergence')
    ax3.grid(True, alpha=0.3)
    
    # Histograms (bottom row) - only if requested
    if show_histograms:
        # Extract final iteration values
        final_energy_errors = [errors[-1] for errors in all_energy_errors]
        final_pos_distances = [distances[-1] for distances in all_pos_distances]
        final_angle_openings = [openings[-1] for openings in all_angle_openings]
        
        # Histogram 1: Final energy errors
        hist_ax1 = axes[1][0]
        hist_ax1.hist(final_energy_errors, bins=min(15, N_events//2), alpha=0.7, color='blue', edgecolor='black')
        hist_ax1.axvline(np.mean(final_energy_errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_energy_errors):.3f}')
        hist_ax1.axvline(np.median(final_energy_errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(final_energy_errors):.3f}')
        hist_ax1.set_xlabel('Final Energy Error')
        hist_ax1.set_ylabel('Count')
        hist_ax1.set_title('Final Energy Error Distribution')
        hist_ax1.legend()
        hist_ax1.grid(True, alpha=0.3)
        
        # Histogram 2: Final position distances
        hist_ax2 = axes[1][1]
        hist_ax2.hist(final_pos_distances, bins=min(15, N_events//2), alpha=0.7, color='green', edgecolor='black')
        hist_ax2.axvline(np.mean(final_pos_distances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_pos_distances):.3f}')
        hist_ax2.axvline(np.median(final_pos_distances), color='darkgreen', linestyle='--', linewidth=2, label=f'Median: {np.median(final_pos_distances):.3f}')
        hist_ax2.set_xlabel('Final Position Distance')
        hist_ax2.set_ylabel('Count')
        hist_ax2.set_title('Final Position Distance Distribution')
        hist_ax2.legend()
        hist_ax2.grid(True, alpha=0.3)
        
        # Histogram 3: Final angle openings
        hist_ax3 = axes[1][2]
        hist_ax3.hist(final_angle_openings, bins=min(15, N_events//2), alpha=0.7, color='magenta', edgecolor='black')
        hist_ax3.axvline(np.mean(final_angle_openings), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_angle_openings):.3f}')
        hist_ax3.axvline(np.median(final_angle_openings), color='purple', linestyle='--', linewidth=2, label=f'Median: {np.median(final_angle_openings):.3f}')
        hist_ax3.set_xlabel('Final Angle Opening (rad)')
        hist_ax3.set_ylabel('Count')
        hist_ax3.set_title('Final Angle Opening Distribution')
        hist_ax3.legend()
        hist_ax3.grid(True, alpha=0.3)
        
        # Print final statistics
        print(f"\n=== Final Iteration Statistics (N={N_events} events) ===")
        print(f"Energy Error - Mean: {np.mean(final_energy_errors):.4f}, Std: {np.std(final_energy_errors):.4f}")
        print(f"Position Distance - Mean: {np.mean(final_pos_distances):.4f}, Std: {np.std(final_pos_distances):.4f}")
        print(f"Angle Opening - Mean: {np.mean(final_angle_openings):.4f} rad ({np.degrees(np.mean(final_angle_openings)):.2f}°), Std: {np.std(final_angle_openings):.4f} rad")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Return final values for further analysis
    if show_histograms:
        return {
            'final_energy_errors': final_energy_errors,
            'final_pos_distances': final_pos_distances, 
            'final_angle_openings': final_angle_openings
        }
    else:
        return None

def get_error_at_iteration(results, ID, iteration):

    def get_opening_angle(true_theta, true_phi, recon_theta, recon_phi):
        true_dir = np.array([
            np.sin(true_theta) * np.cos(true_phi),
            np.sin(true_theta) * np.sin(true_phi),
            np.cos(true_theta)
        ])
        recon_dir = np.array([
            np.sin(recon_theta) * np.cos(recon_phi),
            np.sin(recon_theta) * np.sin(recon_phi),
            np.cos(recon_theta)
        ])

        cos_angle = np.clip(np.dot(true_dir, recon_dir), -1.0, 1.0)
        angle_opening = np.arccos(cos_angle)

        return angle_opening

    best_spatial_it = np.argmin(results['loss_histories'][ID]['spatial'])

    final_E_err = results['final_errors'][ID]['energy']
    final_dist_err = results['final_errors'][ID]['position']
    final_phi_err = results['final_errors'][ID]['theta']
    final_theta_err = results['final_errors'][ID]['phi']

    E_reco_final = results['param_histories'][ID]['energy'][iteration]
    E_true = results['true_params'][ID][0]
    E_err_final = abs(E_reco_final-E_true)

    P_reco =  np.array([results['param_histories'][ID]['position_x'][iteration], results['param_histories'][ID]['position_y'][iteration], results['param_histories'][ID]['position_z'][iteration]])
    P_true = results['true_params'][ID][1]
    P_err_final = np.linalg.norm(P_reco - np.array(P_true))

    theta_true = results['true_params'][ID][2][0]
    phi_true = results['true_params'][ID][2][1]

    theta_reco = results['param_histories'][ID]['theta'][iteration]
    phi_reco = results['param_histories'][ID]['phi'][iteration]

    A_err_final = get_opening_angle(theta_true, phi_true, theta_reco, phi_reco)

    return E_err_final, P_err_final, A_err_final


