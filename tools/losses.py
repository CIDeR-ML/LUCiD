

from functools import partial
import jax.numpy as jnp
import jax
from tools.generate import *

def softmin(x, alpha=1.0):
    exp_x = jnp.exp(-alpha * x)
    return jnp.sum(x * exp_x) / jnp.sum(exp_x)


# @partial(jax.jit, static_argnums=(13,))
# def smooth_combined_loss_function(true_indices, true_times, reflection_prob, cone_opening, track_origin, track_direction, \
#     photon_norm, att_L, trk_L, scatt_L, detector_points, detector_radius, detector_height, Nphot, key):    
#     final_closest_points, final_closest_detector_indices, final_photon_times, same_detector_count, closest_points, closest_detector_indices, photon_times, ray_weights = differentiable_photon_pmt_distance(
#         reflection_prob, cone_opening, track_origin, track_direction, detector_points, detector_radius, detector_height, Nphot, key
#     )

#     reflection_mask = ray_weights > 0.5
#     selected_points = jnp.where(reflection_mask[:, None], final_closest_points, closest_points)
#     selected_photon_times = jnp.where(reflection_mask, final_photon_times, photon_times)
#     selected_detector_indices = jnp.where(reflection_mask, final_closest_detector_indices, closest_detector_indices)
#     hit_flag = jnp.linalg.norm(selected_points - detector_points[selected_detector_indices], axis=1) < detector_radius

#     valid_indices = selected_detector_indices[hit_flag]
#     idx, idx_inverse = jnp.unique(valid_indices, return_inverse=True)
#     cts = jnp.bincount(idx_inverse)
    


#     true_hit_positions = detector_points[true_indices]
    
#     # Compute distances and time differences in a single vectorized operation
#     diff = selected_points[:, None, :] - true_hit_positions[None, :, :]
#     distances = jnp.linalg.norm(diff, axis=-1)
#     time_differences = jnp.abs(selected_photon_times[:, None] - true_times[None, :])
    
#     # Compute weights using a more efficient method
#     normalized_distances = distances / jnp.max(distances, axis=1, keepdims=True)
#     weights = jax.nn.softmax(-normalized_distances / 0.1, axis=1)  # Using softmax for normalization
    
#     # Compute weighted average of time differences
#     weighted_time_differences = jnp.sum(weights * time_differences, axis=1)
    
#     # Compute spatial component using minimum distances
#     min_distances = jnp.min(distances, axis=1)
    
#     # Combine the components
#     return 3 * jnp.mean(weighted_time_differences) + jnp.mean(min_distances)







@partial(jax.jit, static_argnums=(14,))
def smooth_combined_loss_function(true_indices, true_hits, true_times, reflection_prob, cone_opening, track_origin, track_direction, \
    photon_norm, att_L, trk_L, scatt_L, detector_points, detector_radius, detector_height, Nphot, key):        
    
    final_closest_points, final_closest_detector_indices, final_photon_times, same_detector_count, closest_points, closest_detector_indices, photon_times, ray_weights = differentiable_photon_pmt_distance(
        reflection_prob, cone_opening, track_origin, track_direction, detector_points, detector_radius, detector_height, Nphot, key
    )

    reflection_mask = ray_weights > 0.5
    selected_points = jnp.where(reflection_mask[:, None], final_closest_points, closest_points)
    selected_photon_times = jnp.where(reflection_mask, final_photon_times, photon_times)
    selected_detector_indices = jnp.where(reflection_mask, final_closest_detector_indices, closest_detector_indices)
    hit_flag = jnp.linalg.norm(selected_points - detector_points[selected_detector_indices], axis=1) < detector_radius

    valid_indices = jnp.where(hit_flag, selected_detector_indices, -1)

    max_index = len(detector_points)
    reco_hits = jax.ops.segment_sum(jnp.ones_like(valid_indices), valid_indices, num_segments=max_index+1)[:max_index]
    true_hit_positions = detector_points[true_indices]
    
    # # Compute distances and time differences in a single vectorized operation
    # diff = selected_points[:, None, :] - true_hit_positions[None, :, :]
    # distances = jnp.linalg.norm(diff, axis=-1)
    # time_differences = jnp.abs(selected_photon_times[:, None] - true_times[None, :])
    
    # # Compute weights using a more efficient method
    # normalized_distances = distances / jnp.max(distances, axis=1, keepdims=True)
    # weights = jax.nn.softmax(-normalized_distances / 0.1, axis=1)  # Using softmax for normalization
    
    # # Compute weighted average of time differences
    # weighted_time_differences = jnp.sum(weights * time_differences, axis=1)
    
    # # Compute spatial component using minimum distances
    # min_distances = jnp.min(distances, axis=1)
    
    # # Combine the components
    # return 3 * jnp.mean(weighted_time_differences) + jnp.mean(min_distances) + (jnp.sum(reco_hits)*photon_norm-jnp.sum(true_hits))**2*jnp.mean((reco_hits*photon_norm - true_hits) ** 2)

    

    #### ---- not reflected -----
    A_diff = closest_points[:, None, :] - true_hit_positions[None, :, :]
    A_distances = jnp.linalg.norm(A_diff, axis=-1)
    A_time_differences = jnp.abs(photon_times[:, None] - true_times[None, :])
    
    # Compute weights using a more efficient method
    A_normalized_distances = A_distances / jnp.max(A_distances, axis=1, keepdims=True)
    A_weights = jax.nn.softmax(-A_normalized_distances / 0.1, axis=1)  # Using softmax for normalization
    
    # Compute weighted average of time differences
    A_weighted_time_differences = jnp.sum(A_weights *A_time_differences, axis=1)
    
    # Compute spatial component using minimum distances
    A_min_distances = jnp.min(A_distances, axis=1)

    #A_loss = 3 * jnp.mean(A_weighted_time_differences* (1-ray_weights))#+ jnp.mean(A_min_distances* (1-ray_weights[:, None]))
    #A_loss = jnp.mean(A_min_distances* (1-ray_weights))

    A_loss = 3 * jnp.mean(A_weighted_time_differences)+jnp.mean(A_min_distances* (1-ray_weights))


    #### ---- not reflected -----
    B_diff = final_closest_points[:, None, :] - true_hit_positions[None, :, :]
    B_distances = jnp.linalg.norm(B_diff, axis=-1)
    B_time_differences = jnp.abs(final_photon_times[:, None] - true_times[None, :])
    
    # Compute weights using a more efficient method
    B_normalized_distances = B_distances / jnp.max(B_distances, axis=1, keepdims=True)
    B_weights = jax.nn.softmax(-B_normalized_distances / 0.1, axis=1)  # Using softmax for normalization
    
    # Compute weighted average of time differences
    B_weighted_time_differences = jnp.sum(B_weights *B_time_differences, axis=1)
    
    # Compute spatial component using minimum distances
    B_min_distances = jnp.min(B_distances, axis=1)

    #B_loss = 3 * jnp.mean(B_weighted_time_differences*ray_weights) #+ jnp.mean((ray_weights) * B_min_distances)
    #B_loss = jnp.mean(B_min_distances* (ray_weights))

    B_loss = 3 * jnp.mean(B_weighted_time_differences)+jnp.mean(B_min_distances* (ray_weights))


    Q_loss = (jnp.sum(reco_hits)*photon_norm-jnp.sum(true_hits))**2*jnp.mean((reco_hits*photon_norm - true_hits) ** 2)

    return A_loss+B_loss+Q_loss

    












    #return jnp.mean((reco_hits*photon_norm - true_hits) ** 2) #(jnp.sum(reco_hits)*photon_norm/Nphot-jnp.sum(true_hits))**2

    #return (jnp.sum(reco_hits)*photon_norm-jnp.sum(true_hits))**2*jnp.mean((reco_hits*photon_norm - true_hits) ** 2)


    #return jnp.mean((reco_hits*photon_norm/Nphot - true_hits) ** 2)



    # reflection_mask = ray_weights > 0.5
    # selected_points = jnp.where(reflection_mask[:, None], final_closest_points, closest_points)
    # selected_detector_indices = jnp.where(reflection_mask, final_closest_detector_indices, closest_detector_indices)
    # selected_photon_times = jnp.where(reflection_mask, final_photon_times, photon_times)

    # hit_flag = jnp.linalg.norm(selected_points - detector_points[selected_detector_indices], axis=1) < detector_radius




    # # NOTE: ray_weights similar to 1 means reflection happens

    # true_hit_positions = detector_points[true_indices]

    # #### ---- not reflected -----
    # A_diff = closest_points[:, None, :] - true_hit_positions[None, :, :]
    # A_distances = jnp.linalg.norm(A_diff, axis=-1)
    # A_time_differences = jnp.abs(photon_times[:, None] - true_times[None, :])
    
    # # Compute weights using a more efficient method
    # A_normalized_distances = A_distances / jnp.max(A_distances, axis=1, keepdims=True)
    # A_weights = jax.nn.softmax(-A_normalized_distances / 0.1, axis=1)  # Using softmax for normalization
    
    # # Compute weighted average of time differences
    # A_weighted_time_differences = jnp.sum(A_weights *A_time_differences, axis=1)
    
    # # Compute spatial component using minimum distances
    # A_min_distances = jnp.min(A_distances, axis=1)

    # #A_loss = 3 * jnp.mean(A_weighted_time_differences* (1-ray_weights))#+ jnp.mean(A_min_distances* (1-ray_weights[:, None]))
    # #A_loss = jnp.mean(A_min_distances* (1-ray_weights))

    # A_loss = 3 * jnp.mean(A_weighted_time_differences)+jnp.mean(A_min_distances* (1-ray_weights))


    # #### ---- not reflected -----
    # B_diff = final_closest_points[:, None, :] - true_hit_positions[None, :, :]
    # B_distances = jnp.linalg.norm(B_diff, axis=-1)
    # B_time_differences = jnp.abs(final_photon_times[:, None] - true_times[None, :])
    
    # # Compute weights using a more efficient method
    # B_normalized_distances = B_distances / jnp.max(B_distances, axis=1, keepdims=True)
    # B_weights = jax.nn.softmax(-B_normalized_distances / 0.1, axis=1)  # Using softmax for normalization
    
    # # Compute weighted average of time differences
    # B_weighted_time_differences = jnp.sum(B_weights *B_time_differences, axis=1)
    
    # # Compute spatial component using minimum distances
    # B_min_distances = jnp.min(B_distances, axis=1)

    # #B_loss = 3 * jnp.mean(B_weighted_time_differences*ray_weights) #+ jnp.mean((ray_weights) * B_min_distances)
    # #B_loss = jnp.mean(B_min_distances* (ray_weights))

    # B_loss = 3 * jnp.mean(B_weighted_time_differences)+jnp.mean(B_min_distances* (ray_weights))



    #return A_loss+B_loss


