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
from jax.scipy import special



@jit
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

    return jnp.abs(jnp.log(total_sim_charge / (total_true_charge + eps)))*0.1

@jit
def _compute_spatial_loss(simulated_charge, simulated_time, true_charge, true_time, 
                         sensor_positions, tau, lambda_time, scale_factor=1e4):
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
    scale_factor : float
        Scale factor for final loss value
        
    Returns:
    --------
    float
        Spatial loss value
    """

    # Distance matrix and soft assignments
    N = sensor_positions.shape[0]
    dist = jnp.linalg.norm(
        sensor_positions[:, None, :] - sensor_positions[None, :, :], axis=-1
    )

    # Soft assignments Sim -> True
    logits_s2t = -dist / tau
    w_s2t = jax.nn.softmax(logits_s2t, axis=1)
    Q_sim_per_true = w_s2t.T @ simulated_charge

    # Soft assignments True -> Sim
    logits_t2s = -dist.T / tau
    w_t2s = jax.nn.softmax(logits_t2s, axis=1)
    Q_true_per_sim = w_t2s.T @ true_charge

    L_charge = jnp.sum(jnp.abs(Q_sim_per_true - true_charge)/(true_charge+1.))#*0.05
    
    T_sim_per_true = w_s2t.T @ simulated_time
   
    L_time = jnp.sum(jnp.abs(T_sim_per_true - true_time)*(true_charge)**2)/50

    jax.debug.print("L_charge: {}", L_charge)
    jax.debug.print("L_time: {}", L_time)

    return (L_charge + L_time) / scale_factor

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
        sensor_positions, tau=0.001, lambda_time=lambda_time)


def combined_loss_fn(params, true_charges, true_times, simulate_event, sensor_params, sensor_positions, event_key, tau=0.01, lambda_time=1.0):
    """
    Combined loss function for optimization.
    
    Computes both energy and spatial losses with a single simulation call.
    
    Parameters:
    -----------
    params : tuple
        (energy, position, direction_angles)
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
    tuple
        (energy_loss, spatial_loss)
    """
    true_event_data = (true_charges, true_times)
    true_charge, true_time = true_event_data
    
    simulated_data = simulate_event(params, sensor_params, event_key)
    simulated_charge, simulated_time = simulated_data
    
    energy_loss = _compute_energy_loss(simulated_charge, true_charge)
    spatial_loss = _compute_spatial_loss(
        simulated_charge, simulated_time, true_charge, true_time, 
        sensor_positions, tau=tau, lambda_time=lambda_time, 
        scale_factor=1e4
    )
    
    return energy_loss, spatial_loss


def compute_gradients(params, true_charges, true_times, energy_grad_fn, spatial_grad_fn, event_key):
    """
    Compute gradients for both energy and spatial losses using pre-compiled gradient functions.
    
    This approach uses JIT-compiled gradient functions to efficiently compute gradients
    for both energy and spatial losses without re-jitting.
    
    Parameters:
    -----------
    params : tuple
        Current parameters (energy, position, direction_angles)
    true_charges : jnp.ndarray
        True charge measurements
    true_times : jnp.ndarray
        True time measurements
    energy_grad_fn : function
        Pre-compiled JIT gradient function for energy loss
    spatial_grad_fn : function
        Pre-compiled JIT gradient function for spatial loss
    event_key : PRNGKey
        Random key for simulation
        
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
    
    # Compute energy loss and gradient
    energy_loss_val, energy_grad = energy_grad_fn(params, true_charges, true_times, event_key)
    
    # Compute spatial loss and gradient
    spatial_loss_val, spatial_grad = spatial_grad_fn(params, true_charges, true_times, event_key)
    
    # Efficient NaN safety - replace NaNs with zeros to prevent optimization failure
    # This is JIT-friendly and fast
    
    # Check for NaNs first (for optional debugging)
    energy_has_nan = jax.tree_util.tree_reduce(
        lambda acc, x: acc | jnp.any(jnp.isnan(x)), energy_grad, False
    )
    spatial_has_nan = jax.tree_util.tree_reduce(
        lambda acc, x: acc | jnp.any(jnp.isnan(x)), spatial_grad, False
    )
    
    # Replace NaNs with zeros
    energy_grad = jax.tree_map(lambda x: jnp.where(jnp.isnan(x), 0.0, x), energy_grad)
    spatial_grad = jax.tree_map(lambda x: jnp.where(jnp.isnan(x), 0.0, x), spatial_grad)
    
    # # Optional: Use jax.debug.print for minimal NaN reporting (can be disabled for performance)
    # jax.debug.print("NaN detected - Energy: {energy_nan}, Spatial: {spatial_nan}", 
    #                 energy_nan=energy_has_nan, spatial_nan=spatial_has_nan)

    # jax.debug.print("Energy Grad: {energy_grad}, Spatial Grad: {spatial_grad}", 
    #                 energy_grad=energy_grad, spatial_grad=spatial_grad)
    
    return energy_grad, spatial_grad, float(energy_loss_val), float(spatial_loss_val)


@jit
def initial_guess_loss(simulated_charges, true_charges, simulated_times, true_times):
    """
    Calculate loss between simulated and true charges and times for initial guess selection.
    This is used to find best matching events in the pre-generated database.
    
    Loss = sum(|simulated_charge - true_charge|) + sum(|simulated_time_centered - true_time_centered|)
    
    Args:
        simulated_charges: Charges from simulated event
        true_charges: Charges from true event
        simulated_times: Times from simulated event
        true_times: Times from true event
    
    Returns:
        loss: Scalar loss value combining charge and time differences
    """
    eps = 1e-8
    threshold = 1e-8
    
    # Compute mean times for active locations
    true_active_mask = true_charges > threshold
    sim_active_mask = simulated_charges > threshold

    true_mean_time = jnp.sum(true_times * true_active_mask) / (jnp.sum(true_active_mask) + eps)
    sim_mean_time = jnp.sum(simulated_times * sim_active_mask) / (jnp.sum(sim_active_mask) + eps)

    true_time_centered = jnp.where(true_active_mask, true_times - true_mean_time, 0.0)
    sim_time_centered = jnp.where(sim_active_mask, simulated_times - sim_mean_time, 0.0)

    L_delta_charge = jnp.sum(jnp.abs(simulated_charges - true_charges))
    L_delta_time = jnp.sum(jnp.abs(sim_time_centered - true_time_centered))

    return L_delta_charge #+ L_delta_time