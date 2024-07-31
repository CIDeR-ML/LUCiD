

from functools import partial
import jax.numpy as jnp
import jax
from tools.generate import *

@partial(jax.jit, static_argnums=(7, 9))
def combined_loss_function(true_indices, true_times, cone_opening, track_origin, track_direction, detector_points, detector_radius, Nphot, key, use_time_loss):
    return smooth_combined_loss_function(true_indices, true_times, cone_opening, track_origin, track_direction, detector_points, detector_radius, Nphot, key)

def softmin(x, alpha=1.0):
    exp_x = jnp.exp(-alpha * x)
    return jnp.sum(x * exp_x) / jnp.sum(exp_x)

@partial(jax.jit, static_argnums=(7,))
def smooth_combined_loss_function(true_indices, true_times, cone_opening, track_origin, track_direction, detector_points, detector_radius, Nphot, key):
    simulated_points, closest_detector_indices, photon_times = differentiable_toy_mc_simulator(
        cone_opening, track_origin, track_direction, detector_points, detector_radius, Nphot, key
    )
    
    true_hit_positions = detector_points[true_indices]
    
    # Compute distances from each simulated point to all true hit positions
    distances = jnp.linalg.norm(simulated_points[:, None, :] - true_hit_positions[None, :, :], axis=-1)
    
    # Compute time differences for each simulated photon to each true hit
    time_differences = jnp.abs(photon_times[:, None] - true_times[None, :])
    
    # Compute weights using a more robust method
    max_distance = jnp.max(distances, axis=1, keepdims=True)
    normalized_distances = distances / max_distance
    weights = jnp.exp(-normalized_distances / 0.1)  # 0.1 is a softness parameter, adjust as needed
    weights = weights / jnp.sum(weights, axis=1, keepdims=True)
    
    # Compute weighted average of time differences
    weighted_time_differences = jnp.sum(weights * time_differences, axis=1)
    
    # Compute spatial component using minimum distances
    min_distances = jnp.min(distances, axis=1)
    
    # Normalize and combine the components
    avg_time_diff = jnp.mean(weighted_time_differences)
    avg_min_dist = jnp.mean(min_distances)
    
    return 3*avg_time_diff+avg_min_dist