

from functools import partial
import jax.numpy as jnp
import jax
from tools.generate import *

def softmin(x, alpha=1.0):
    exp_x = jnp.exp(-alpha * x)
    return jnp.sum(x * exp_x) / jnp.sum(exp_x)


@partial(jax.jit, static_argnums=(9,))
def smooth_combined_loss_function(true_indices, true_times, reflection_prob, cone_opening, track_origin, track_direction, detector_points, detector_radius, detector_height, Nphot, key):
    simulated_points, closest_detector_indices, photon_times, _, ray_weights = differentiable_photon_pmt_distance(
        reflection_prob, cone_opening, track_origin, track_direction, detector_points, detector_radius, detector_height, Nphot, key
    )
    
    true_hit_positions = detector_points[true_indices]
    
    # Compute distances and time differences in a single vectorized operation
    diff = simulated_points[:, None, :] - true_hit_positions[None, :, :]
    distances = jnp.linalg.norm(diff, axis=-1)
    time_differences = jnp.abs(photon_times[:, None] - true_times[None, :])
    
    # Compute weights using a more efficient method
    normalized_distances = distances / jnp.max(distances, axis=1, keepdims=True)
    weights = jax.nn.softmax(-normalized_distances / 0.1, axis=1)  # Using softmax for normalization
    
    # Compute weighted average of time differences
    weighted_time_differences = jnp.sum(weights * time_differences, axis=1)
    
    # Compute spatial component using minimum distances
    min_distances = jnp.min(distances, axis=1)
    
    # Combine the components
    return 3 * jnp.mean(weighted_time_differences) + jnp.mean(min_distances)


