"""
Loss functions for optimization in LUCiD.

This module contains specialized loss functions for gradient-based optimization:
- energy_loss_fn: Focuses on energy parameter optimization using intensity matching
- spatial_loss_fn: Focuses on position and direction optimization using spatial-temporal matching
- combined_loss_fn: Combined loss for numerical optimization consistency
"""

import jax
import jax.numpy as jnp
from jax import jit


def _compute_energy_loss(simulated_charge, true_charge):
    """
    Core energy loss computation using intensity matching.
    
    Parameters:
    -----------
    simulated_charge : jnp.ndarray
        Simulated charge values
    true_charge : jnp.ndarray
        True charge values
        
    Returns:
    --------
    float
        Energy loss value
    """
    total_true_charge = jnp.sum(true_charge)
    total_sim_charge = jnp.sum(simulated_charge)
    eps = 1e-8
    return jnp.abs(jnp.log(total_sim_charge / (total_true_charge + eps)))


def _compute_spatial_loss(simulated_charge, simulated_time, true_charge, true_time, 
                         sensor_positions, tau=0.01, lambda_time=1.0, 
                         include_time=True, scale_factor=1e4):
    """
    Core spatial loss computation using soft assignments.
    
    Parameters:
    -----------
    simulated_charge : jnp.ndarray
        Simulated charge values
    simulated_time : jnp.ndarray
        Simulated time values
    true_charge : jnp.ndarray
        True charge values
    true_time : jnp.ndarray
        True time values
    sensor_positions : jnp.ndarray
        Array of sensor positions
    tau : float
        Temperature parameter for softmax assignments
    lambda_time : float
        Weight for time loss component
    include_time : bool
        Whether to include time loss component
    scale_factor : float
        Scale factor for final loss value
        
    Returns:
    --------
    float
        Spatial loss value
    """
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
    N = sensor_positions.shape[0]
    dist = jnp.linalg.norm(
        sensor_positions[:, None, :] - sensor_positions[None, :, :], axis=-1
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
    
    if include_time:
        return (L_charge + L_time) / scale_factor
    else:
        return L_charge / scale_factor


def energy_loss_fn(params, true_event_data, simulate_event, sensor_params, sensor_positions, event_key):
    """
    Energy-focused loss function.
    
    Uses intensity matching between true and simulated total charge to optimize energy parameter.
    
    Parameters:
    -----------
    params : tuple
        (energy, position, direction_angles)
    true_event_data : tuple
        (true_charge, true_time)
    simulate_event : function
        Event simulation function
    sensor_params : tuple
        Sensor parameters for simulation
    sensor_positions : jnp.ndarray
        Sensor positions (not used in this loss but kept for consistency)
    event_key : PRNGKey
        Random key for simulation
        
    Returns:
    --------
    float
        Energy loss value
    """
    true_charge, true_time = true_event_data
    
    simulated_data = simulate_event(params, sensor_params, event_key)
    simulated_charge, _ = simulated_data
    
    return _compute_energy_loss(simulated_charge, true_charge)


def spatial_loss_fn(params, true_event_data, simulate_event, sensor_params, sensor_positions, event_key, tau=0.01, lambda_time=1.0):
    """
    Spatial/temporal-focused loss function.
    
    Uses spatial-temporal matching via soft assignments to optimize position and direction parameters.
    
    Parameters:
    -----------
    params : tuple
        (energy, position, direction_angles)
    true_event_data : tuple
        (true_charge, true_time)
    simulate_event : function
        Event simulation function
    sensor_params : tuple
        Sensor parameters for simulation
    sensor_positions : jnp.ndarray
        Array of sensor positions
    event_key : PRNGKey
        Random key for simulation
    tau : float
        Temperature parameter for softmax assignments
    lambda_time : float
        Weight for time loss component
        
    Returns:
    --------
    float
        Spatial loss value
    """
    true_charge, true_time = true_event_data
    
    simulated_data = simulate_event(params, sensor_params, event_key)
    simulated_charge, simulated_time = simulated_data
    
    # Use the same scaling and tau settings as the previous version
    # Note: Previous version used hardcoded tau=0.001 and included time loss
    return _compute_spatial_loss(
        simulated_charge, simulated_time, true_charge, true_time, 
        sensor_positions, tau=0.001, lambda_time=lambda_time, 
        include_time=True, scale_factor=1e4
    )


def combined_loss_fn(params, true_event_data, simulate_event, sensor_params, sensor_positions, event_key, tau=0.01, lambda_time=1.0):
    """
    Combined loss function for numerical optimization.
    
    Combines energy and spatial losses to provide consistent loss definition 
    between numerical and gradient optimization phases.
    
    Parameters:
    -----------
    params : tuple
        (energy, position, direction_angles)
    true_event_data : tuple
        (true_charge, true_time)
    simulate_event : function
        Event simulation function
    sensor_params : tuple
        Sensor parameters for simulation
    sensor_positions : jnp.ndarray
        Array of sensor positions
    event_key : PRNGKey
        Random key for simulation
    tau : float
        Temperature parameter for softmax assignments
    lambda_time : float
        Weight for time loss component
        
    Returns:
    --------
    float
        Combined loss value (energy_loss + spatial_loss)
    """
    true_charge, true_time = true_event_data
    
    simulated_data = simulate_event(params, sensor_params, event_key)
    simulated_charge, simulated_time = simulated_data
    
    energy_loss = _compute_energy_loss(simulated_charge, true_charge)
    spatial_loss = _compute_spatial_loss(
        simulated_charge, simulated_time, true_charge, true_time, 
        sensor_positions, tau=tau, lambda_time=lambda_time, 
        include_time=True, scale_factor=1e4
    )
    
    return energy_loss + spatial_loss


def create_optimization_loss_functions(simulate_event, sensor_positions, sensor_params, tau=0.01, lambda_time=1.0):
    """
    Create loss functions with simulation parameters baked in.
    
    Parameters:
    -----------
    simulate_event : function
        Event simulation function
    sensor_positions : jnp.ndarray
        Array of sensor positions
    sensor_params : tuple
        Sensor parameters for simulation
    tau : float
        Temperature parameter for softmax assignments
    lambda_time : float
        Weight for time loss component
        
    Returns:
    --------
    tuple
        (energy_loss_func, spatial_loss_func, combined_loss_func) with baked-in parameters
    """
    
    def energy_loss_func(params, true_event_data, event_key):
        return energy_loss_fn(params, true_event_data, simulate_event, sensor_params, sensor_positions, event_key)
    
    def spatial_loss_func(params, true_event_data, event_key):
        return spatial_loss_fn(params, true_event_data, simulate_event, sensor_params, sensor_positions, event_key, tau, lambda_time)
    
    def combined_loss_func(params, true_charges, true_times, event_key):
        true_event_data = (true_charges, true_times)
        return combined_loss_fn(params, true_event_data, simulate_event, sensor_params, sensor_positions, event_key, tau, lambda_time)
    
    return energy_loss_func, spatial_loss_func, combined_loss_func


def compute_shared_gradients(params, true_charges, true_times, simulate_event, sensor_params, sensor_positions, event_key, tau=0.01, lambda_time=1.0):
    """
    Compute gradients for both energy and spatial losses with a single simulation.
    
    This approach creates a combined loss function that runs simulate_event once and computes both losses,
    then uses JAX to get gradients with respect to both loss components.
    
    Parameters:
    -----------
    params : tuple
        Current parameters (energy, position, direction_angles)
    true_charges : jnp.ndarray
        True charge measurements
    true_times : jnp.ndarray
        True time measurements
    simulate_event : function
        Event simulation function
    sensor_params : tuple
        Sensor parameters for simulation
    sensor_positions : jnp.ndarray
        Array of sensor positions
    event_key : PRNGKey
        Random key for simulation
    tau : float
        Temperature parameter for softmax assignments
    lambda_time : float
        Weight for time loss component
        
    Returns:
    --------
    energy_grad : tuple
        Gradient for energy-focused loss
    spatial_grad : tuple
        Gradient for spatial-focused loss
    energy_loss_val : float
        Energy loss value
    spatial_loss_val : float
        Spatial loss value
    """
    
    def compute_both_losses(params):
        """Compute both energy and spatial losses with a single simulation."""
        # Single simulation call
        simulated_charges, simulated_times = simulate_event(params, sensor_params, event_key)
        
        # Compute both losses using the shared core functions
        energy_loss = _compute_energy_loss(simulated_charges, true_charges)
        spatial_loss = _compute_spatial_loss(
            simulated_charges, simulated_times, true_charges, true_times, 
            sensor_positions, tau=tau, lambda_time=lambda_time, 
            include_time=True, scale_factor=1e4
        )
        
        return energy_loss, spatial_loss
    
    # Use JAX's jacobian computation for vector-valued function
    def vector_loss(params):
        """Return both losses as a vector for jacobian computation."""
        energy_loss, spatial_loss = compute_both_losses(params)
        return jnp.array([energy_loss, spatial_loss])
    
    # Compute loss values
    loss_values = vector_loss(params)
    
    # Compute jacobian (gradients for each output component)
    jacobian_fn = jax.jacrev(vector_loss)  # or jax.jacfwd for forward-mode
    jacobian = jacobian_fn(params)
    
    # Extract individual components
    energy_loss_val = loss_values[0]
    spatial_loss_val = loss_values[1]
    
    # Extract gradients for each loss component
    # jacobian has shape (2, *param_shape) where first dim corresponds to [energy_loss, spatial_loss]
    energy_grad = jax.tree.map(lambda x: x[0], jacobian)  # Gradient w.r.t. energy loss
    spatial_grad = jax.tree.map(lambda x: x[1], jacobian)  # Gradient w.r.t. spatial loss
    
    # Check for NaN values and replace with zeros if needed (to prevent complete optimization failure)
    def replace_nan_with_zero(x, component_name=""):
        nan_mask = jnp.isnan(x)
        nan_count = jnp.sum(nan_mask)
        total_elements = x.size
        nan_fraction = nan_count / total_elements
        
        # Print warning if NaNs are found
        if nan_count > 0:
            print(f"⚠️  NaN replacement in {component_name}: {nan_count}/{total_elements} elements ({nan_fraction:.3f} fraction)")
        
        return jnp.where(nan_mask, 0.0, x)
    
    # Apply NaN replacement with component identification
    def replace_nans_in_tree(tree, tree_name):
        def replace_with_name(x, path=""):
            component_name = f"{tree_name}"
            if path:
                component_name += f".{path}"
            return replace_nan_with_zero(x, component_name)
        
        # For our gradient tree structure: (energy_grad, position_grad, direction_grad)
        if isinstance(tree, tuple) and len(tree) == 3:
            energy_grad_clean = replace_nan_with_zero(tree[0], f"{tree_name}.energy")
            position_grad_clean = jax.tree.map(
                lambda x: replace_nan_with_zero(x, f"{tree_name}.position"), tree[1]
            )
            direction_grad_clean = jax.tree.map(
                lambda x: replace_nan_with_zero(x, f"{tree_name}.direction"), tree[2]
            )
            return (energy_grad_clean, position_grad_clean, direction_grad_clean)
        else:
            return jax.tree.map(replace_with_name, tree)
    
    energy_grad = replace_nans_in_tree(energy_grad, "energy_grad")
    spatial_grad = replace_nans_in_tree(spatial_grad, "spatial_grad")
    
    return energy_grad, spatial_grad, float(energy_loss_val), float(spatial_loss_val)