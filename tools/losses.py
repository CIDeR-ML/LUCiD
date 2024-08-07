

from functools import partial
import jax.numpy as jnp
import jax
from tools.generate import *
from jax import lax

def softmin(x, alpha=1.0):
    exp_x = jnp.exp(-alpha * x)
    return jnp.sum(x * exp_x) / jnp.sum(exp_x)




@partial(jax.jit, static_argnums=(14,))
def smooth_combined_loss_function(true_indices, true_hits, true_times, reflection_prob, cone_opening, track_origin, track_direction, \
    photon_norm, att_L, trk_L, scatt_L, detector_points, detector_radius, detector_height, Nphot, key):        
    
    final_closest_points, final_closest_detector_indices, final_photon_times, same_detector_count, closest_points, closest_detector_indices, photon_times, ray_weights = differentiable_photon_pmt_distance(
        reflection_prob, cone_opening, track_origin, track_direction, detector_points, detector_radius, detector_height, Nphot, key
    )

    reflection_mask = ray_weights > 0.5
    selected_points = jnp.where(reflection_mask[:, None], final_closest_points, closest_points)
    selected_photon_times = jnp.where(reflection_mask, final_photon_times, photon_times)

    true_hit_positions = detector_points[true_indices]

    diff = selected_points[:, None, :] - true_hit_positions[None, :, :]
    distances = jnp.linalg.norm(diff, axis=-1)
    time_differences = jnp.abs(selected_photon_times[:, None] - true_times[None, :])
    
    # Compute weights using a more efficient method
    normalized_distances = distances / jnp.max(distances, axis=1, keepdims=True)
    weights = jax.nn.softmax(-normalized_distances / 0.1, axis=1)  # Using softmax for normalization
    
    # Compute weighted average of time differences
    weighted_time_differences = jnp.sum(weights *time_differences, axis=1)
    
    # Compute spatial component using minimum distances
    min_distances = jnp.min(distances, axis=1)

    loss = 3 * jnp.mean(weighted_time_differences)+jnp.mean(min_distances)

    hit_flag_direct = jnp.linalg.norm(closest_points - detector_points[closest_detector_indices], axis=1) < 0.04
    hit_flag_reflec = jnp.linalg.norm(final_closest_points - detector_points[final_closest_detector_indices], axis=1) < 0.04

    indices_direct = jnp.where(hit_flag_direct, closest_detector_indices, -1)
    indices_reflec = jnp.where(hit_flag_reflec, final_closest_detector_indices, -1)
    
    max_index = len(detector_points)
    reco_hits_direct = jax.ops.segment_sum((1-reflection_prob), indices_direct, num_segments=max_index+1)[:max_index]
    reco_hits_reflec = jax.ops.segment_sum(reflection_prob, indices_reflec, num_segments=max_index+1)[:max_index]
    
    # Create a mask for non-zero elements
    mask_direct = reco_hits_direct > 0
    mask_reflec = reco_hits_reflec > 0
    
    # Use the masks to filter true_hits
    filtered_true_hits_direct = jnp.where(mask_direct, true_hits, 0.0)
    filtered_true_hits_reflec = jnp.where(mask_reflec, true_hits, 0.0)
    

    # print('there are Nphot: ', Nphot)
    # print('there are N PMTs: ', len(true_hits))
    # print('there are N true hits: ', np.sum(true_hits>0))
    # print('total Q in true hits: ', np.sum(true_hits))

    # print('There are N direct reco hits: ', np.sum(hit_flag_direct))
    # print('There are N direct reco hits: ', np.sum(reco_hits_reflec))

    # print('total Q in direct reco hits: ',  np.sum(reco_hits_direct))
    # print('total Q in reflec reco hits:  ', np.sum(reco_hits_reflec))

    # print('total true hits in direct reco hit indices: ', np.sum(filtered_true_hits_direct))
    # print('total true hits in reflec reco hit indices: ', np.sum(filtered_true_hits_reflec))

    Q_sym = (jnp.sum(reco_hits_direct)*lax.stop_gradient(photon_norm) -jnp.sum(filtered_true_hits_direct))**2 + (jnp.sum(reco_hits_reflec)*lax.stop_gradient(photon_norm) -jnp.sum(filtered_true_hits_reflec))**2

    selected_detector_indices = jnp.where(reflection_mask, final_closest_detector_indices, closest_detector_indices)
    hit_flag = jnp.linalg.norm(selected_points - detector_points[selected_detector_indices], axis=1) < 0.04 ## need to pass another argument here to be sensor radius!
    valid_indices = jnp.where(hit_flag, selected_detector_indices, -1)
    max_index = len(detector_points)
    reco_hits = jax.ops.segment_sum(jnp.ones_like(valid_indices), valid_indices, num_segments=max_index+1)[:max_index]
    Q_tot = (jnp.sum(reco_hits)*photon_norm-jnp.sum(true_hits))**2*jnp.mean((reco_hits*photon_norm - true_hits) ** 2)
    # Q_loss = (jnp.sum(reco_hits)*photon_norm-jnp.sum(true_hits))**2

    return (3 * jnp.mean(weighted_time_differences) + jnp.mean(min_distances))+0.1*Q_sym/(Nphot**2)+Q_tot/(Nphot**2)#+0.1*Q_sym/(Nphot**2)






































# # @partial(jax.jit, static_argnums=(13,))
# # def smooth_combined_loss_function(true_indices, true_times, reflection_prob, cone_opening, track_origin, track_direction, \
# #     photon_norm, att_L, trk_L, scatt_L, detector_points, detector_radius, detector_height, Nphot, key):    
# #     final_closest_points, final_closest_detector_indices, final_photon_times, same_detector_count, closest_points, closest_detector_indices, photon_times, ray_weights = differentiable_photon_pmt_distance(
# #         reflection_prob, cone_opening, track_origin, track_direction, detector_points, detector_radius, detector_height, Nphot, key
# #     )

# #     reflection_mask = ray_weights > 0.5
# #     selected_points = jnp.where(reflection_mask[:, None], final_closest_points, closest_points)
# #     selected_photon_times = jnp.where(reflection_mask, final_photon_times, photon_times)
# #     selected_detector_indices = jnp.where(reflection_mask, final_closest_detector_indices, closest_detector_indices)
# #     hit_flag = jnp.linalg.norm(selected_points - detector_points[selected_detector_indices], axis=1) < 0.04

# #     valid_indices = selected_detector_indices[hit_flag]
# #     idx, idx_inverse = jnp.unique(valid_indices, return_inverse=True)
# #     cts = jnp.bincount(idx_inverse)
    


# #     true_hit_positions = detector_points[true_indices]
    
# #     # Compute distances and time differences in a single vectorized operation
# #     diff = selected_points[:, None, :] - true_hit_positions[None, :, :]
# #     distances = jnp.linalg.norm(diff, axis=-1)
# #     time_differences = jnp.abs(selected_photon_times[:, None] - true_times[None, :])
    
# #     # Compute weights using a more efficient method
# #     normalized_distances = distances / jnp.max(distances, axis=1, keepdims=True)
# #     weights = jax.nn.softmax(-normalized_distances / 0.1, axis=1)  # Using softmax for normalization
    
# #     # Compute weighted average of time differences
# #     weighted_time_differences = jnp.sum(weights * time_differences, axis=1)
    
# #     # Compute spatial component using minimum distances
# #     min_distances = jnp.min(distances, axis=1)
    
# #     # Combine the components
# #     return 3 * jnp.mean(weighted_time_differences) + jnp.mean(min_distances)



# @partial(jax.jit, static_argnums=(14,))
# def smooth_combined_loss_function(true_indices, true_hits, true_times, reflection_prob, cone_opening, track_origin, track_direction, \
#     photon_norm, att_L, trk_L, scatt_L, detector_points, detector_radius, detector_height, Nphot, key):        
    
#     final_closest_points, final_closest_detector_indices, final_photon_times, same_detector_count, closest_points, closest_detector_indices, photon_times, ray_weights = differentiable_photon_pmt_distance(
#         reflection_prob, cone_opening, track_origin, track_direction, detector_points, detector_radius, detector_height, Nphot, key
#     )

#     reflection_mask = ray_weights > 0.5
#     selected_points = jnp.where(reflection_mask[:, None], final_closest_points, closest_points)
#     selected_photon_times = jnp.where(reflection_mask, final_photon_times, photon_times)
#     # selected_detector_indices = jnp.where(reflection_mask, final_closest_detector_indices, closest_detector_indices)
#     # hit_flag = jnp.linalg.norm(selected_points - detector_points[selected_detector_indices], axis=1) < 0.04 ## need to pass another argument here to be sensor radius!

#     # valid_indices = jnp.where(hit_flag, selected_detector_indices, -1)

#     # max_index = len(detector_points)
#     # reco_hits = jax.ops.segment_sum(jnp.ones_like(valid_indices), valid_indices, num_segments=max_index+1)[:max_index]

#     true_hit_positions = detector_points[true_indices]
    

#     diff = selected_points[:, None, :] - true_hit_positions[None, :, :]
#     distances = jnp.linalg.norm(diff, axis=-1)
#     time_differences = jnp.abs(selected_photon_times[:, None] - true_times[None, :])
    
#     # Compute weights using a more efficient method
#     normalized_distances = distances / jnp.max(distances, axis=1, keepdims=True)
#     weights = jax.nn.softmax(-normalized_distances / 0.1, axis=1)  # Using softmax for normalization
    
#     # Compute weighted average of time differences
#     weighted_time_differences = jnp.sum(weights *time_differences, axis=1)
    
#     # Compute spatial component using minimum distances
#     min_distances = jnp.min(distances, axis=1)

#     loss = 3 * jnp.mean(weighted_time_differences)+jnp.mean(min_distances)





#     #selected_detector_indices = jnp.where(reflection_mask, final_closest_detector_indices, closest_detector_indices)
#     hit_flag_direct = jnp.linalg.norm(closest_points - detector_points[closest_detector_indices], axis=1) < 0.04
#     hit_flag_reflec = jnp.linalg.norm(final_closest_points - detector_points[final_closest_detector_indices], axis=1) < 0.04

#     indices_direct = jnp.where(hit_flag_direct, closest_detector_indices, -1)
#     indices_reflec = jnp.where(hit_flag_reflec, final_closest_detector_indices, -1)

#     max_index = len(detector_points)
#     reco_hits_direct = jax.ops.segment_sum((1-ray_weights), indices_direct, num_segments=max_index+1)[:max_index]
#     reco_hits_reflec = jax.ops.segment_sum(ray_weights, indices_reflec, num_segments=max_index+1)[:max_index]

#     IDXA = jnp.nonzero(reco_hits_direct)[0]
#     IDXB = jnp.nonzero(reco_hits_reflec)[0]

#     pseudo_true_hits_direct = jax.ops.segment_sum(true_hits, IDXA, num_segments=max_index+1)[:max_index]
#     pseudo_true_hits_reflec = jax.ops.segment_sum(true_hits, IDXB, num_segments=max_index+1)[:max_index]


#     # # Create a mask for direct hits (opposite of reflection_mask)
#     # direct_hit_mask = ~reflection_mask

#     # # Use the direct_hit_mask to select indices
#     # direct_hit_indices = jnp.where(direct_hit_mask, closest_detector_indices, -1)

#     # # Create a boolean mask for the detectors that were hit directly
#     # detector_direct_hit_mask = jnp.zeros_like(true_hits, dtype=float).at[direct_hit_indices].set(True)
#     # detector_reflec_hit_mask = jnp.zeros_like(true_hits, dtype=float).at[direct_hit_indices].set(True)

#     Q_loss = jnp.mean((reco_hits_direct*photon_norm - pseudo_true_hits_direct) ** 2)+jnp.mean((reco_hits_reflec*photon_norm - pseudo_true_hits_reflec) ** 2)
#     #Q_loss = jnp.mean((reco_hits_reflec*photon_norm - true_hits) ** 2)


#     # selected_detector_indices = jnp.where(reflection_mask, final_closest_detector_indices, closest_detector_indices)
#     # hit_flag = jnp.linalg.norm(selected_points - detector_points[selected_detector_indices], axis=1) < 0.04 ## need to pass another argument here to be sensor radius!

#     # valid_indices = jnp.where(hit_flag, selected_detector_indices, -1)

#     # max_index = len(detector_points)
#     # reco_hits = jax.ops.segment_sum(jnp.ones_like(valid_indices), valid_indices, num_segments=max_index+1)[:max_index]



#     #Q_loss = (jnp.sum(reco_hits)*photon_norm-jnp.sum(true_hits))**2*jnp.mean((reco_hits*photon_norm - true_hits) ** 2)







#     return loss + Q_loss
#     # #### ---- not reflected -----
#     # A_diff = closest_points[:, None, :] - true_hit_positions[None, :, :]
#     # A_distances = jnp.linalg.norm(A_diff, axis=-1)
#     # A_time_differences = jnp.abs(photon_times[:, None] - true_times[None, :])
    
#     # # Compute weights using a more efficient method
#     # A_normalized_distances = A_distances / jnp.max(A_distances, axis=1, keepdims=True)
#     # A_weights = jax.nn.softmax(-A_normalized_distances / 0.1, axis=1)  # Using softmax for normalization
    
#     # # Compute weighted average of time differences
#     # A_weighted_time_differences = jnp.sum(A_weights *A_time_differences, axis=1)
    
#     # # Compute spatial component using minimum distances
#     # A_min_distances = jnp.min(A_distances, axis=1)

#     # A_loss = 3 * jnp.mean(A_weighted_time_differences)+jnp.mean(A_min_distances* (1-ray_weights))


#     # #### ---- not reflected -----
#     # B_diff = final_closest_points[:, None, :] - true_hit_positions[None, :, :]
#     # B_distances = jnp.linalg.norm(B_diff, axis=-1)
#     # B_time_differences = jnp.abs(final_photon_times[:, None] - true_times[None, :])
    
#     # # Compute weights using a more efficient method
#     # B_normalized_distances = B_distances / jnp.max(B_distances, axis=1, keepdims=True)
#     # B_weights = jax.nn.softmax(-B_normalized_distances / 0.1, axis=1)  # Using softmax for normalization
    
#     # # Compute weighted average of time differences
#     # B_weighted_time_differences = jnp.sum(B_weights *B_time_differences, axis=1)
    
#     # # Compute spatial component using minimum distances
#     # B_min_distances = jnp.min(B_distances, axis=1)

#     # B_loss = 3 * jnp.mean(B_weighted_time_differences)+jnp.mean(B_min_distances* (ray_weights))


#     # Q_loss = (jnp.sum(reco_hits)*photon_norm-jnp.sum(true_hits))**2*jnp.mean((reco_hits*photon_norm - true_hits) ** 2)

#     # return A_loss+B_loss+Q_loss

    




# # @partial(jax.jit, static_argnums=(14,))
# # def smooth_combined_loss_function(true_indices, true_hits, true_times, reflection_prob, cone_opening, track_origin, track_direction, \
# #     photon_norm, att_L, trk_L, scatt_L, detector_points, detector_radius, detector_height, Nphot, key):        
    
# #     final_closest_points, final_closest_detector_indices, final_photon_times, same_detector_count, closest_points, closest_detector_indices, photon_times, ray_weights = differentiable_photon_pmt_distance(
# #         reflection_prob, cone_opening, track_origin, track_direction, detector_points, detector_radius, detector_height, Nphot, key
# #     )

# #     reflection_mask = ray_weights > 0.5
# #     selected_points = jnp.where(reflection_mask[:, None], final_closest_points, closest_points)
# #     selected_photon_times = jnp.where(reflection_mask, final_photon_times, photon_times)
# #     selected_detector_indices = jnp.where(reflection_mask, final_closest_detector_indices, closest_detector_indices)
# #     hit_flag = jnp.linalg.norm(selected_points - detector_points[selected_detector_indices], axis=1) < 0.04

# #     valid_indices = jnp.where(hit_flag, selected_detector_indices, -1)

# #     max_index = len(detector_points)
# #     reco_hits = jax.ops.segment_sum(jnp.ones_like(valid_indices), valid_indices, num_segments=max_index+1)[:max_index]

# #     true_hit_positions = detector_points[true_indices]
    





# #     #### ---- not reflected -----
# #     A_diff = closest_points[:, None, :] - true_hit_positions[None, :, :]
# #     A_distances = jnp.linalg.norm(A_diff, axis=-1)
# #     A_time_differences = jnp.abs(photon_times[:, None] - true_times[None, :])
    
# #     # Compute weights using a more efficient method
# #     A_normalized_distances = A_distances / jnp.max(A_distances, axis=1, keepdims=True)
# #     A_weights = jax.nn.softmax(-A_normalized_distances / 0.1, axis=1)  # Using softmax for normalization
    
# #     # Compute weighted average of time differences
# #     A_weighted_time_differences = jnp.sum(A_weights *A_time_differences, axis=1)
    
# #     # Compute spatial component using minimum distances
# #     A_min_distances = jnp.min(A_distances, axis=1)

# #     A_loss = 3 * jnp.mean(A_weighted_time_differences)+jnp.mean(A_min_distances* (1-ray_weights))


# #     #### ---- not reflected -----
# #     B_diff = final_closest_points[:, None, :] - true_hit_positions[None, :, :]
# #     B_distances = jnp.linalg.norm(B_diff, axis=-1)
# #     B_time_differences = jnp.abs(final_photon_times[:, None] - true_times[None, :])
    
# #     # Compute weights using a more efficient method
# #     B_normalized_distances = B_distances / jnp.max(B_distances, axis=1, keepdims=True)
# #     B_weights = jax.nn.softmax(-B_normalized_distances / 0.1, axis=1)  # Using softmax for normalization
    
# #     # Compute weighted average of time differences
# #     B_weighted_time_differences = jnp.sum(B_weights *B_time_differences, axis=1)
    
# #     # Compute spatial component using minimum distances
# #     B_min_distances = jnp.min(B_distances, axis=1)

# #     B_loss = 3 * jnp.mean(B_weighted_time_differences)+jnp.mean(B_min_distances* (ray_weights))


# #     Q_loss = (jnp.sum(reco_hits)*photon_norm-jnp.sum(true_hits))**2*jnp.mean((reco_hits*photon_norm - true_hits) ** 2)

# #     return A_loss+B_loss+Q_loss

    












#     #return jnp.mean((reco_hits*photon_norm - true_hits) ** 2) #(jnp.sum(reco_hits)*photon_norm/Nphot-jnp.sum(true_hits))**2

#     #return (jnp.sum(reco_hits)*photon_norm-jnp.sum(true_hits))**2*jnp.mean((reco_hits*photon_norm - true_hits) ** 2)


#     #return jnp.mean((reco_hits*photon_norm/Nphot - true_hits) ** 2)



#     # reflection_mask = ray_weights > 0.5
#     # selected_points = jnp.where(reflection_mask[:, None], final_closest_points, closest_points)
#     # selected_detector_indices = jnp.where(reflection_mask, final_closest_detector_indices, closest_detector_indices)
#     # selected_photon_times = jnp.where(reflection_mask, final_photon_times, photon_times)

#     # hit_flag = jnp.linalg.norm(selected_points - detector_points[selected_detector_indices], axis=1) < 0.04




#     # # NOTE: ray_weights similar to 1 means reflection happens

#     # true_hit_positions = detector_points[true_indices]

#     # #### ---- not reflected -----
#     # A_diff = closest_points[:, None, :] - true_hit_positions[None, :, :]
#     # A_distances = jnp.linalg.norm(A_diff, axis=-1)
#     # A_time_differences = jnp.abs(photon_times[:, None] - true_times[None, :])
    
#     # # Compute weights using a more efficient method
#     # A_normalized_distances = A_distances / jnp.max(A_distances, axis=1, keepdims=True)
#     # A_weights = jax.nn.softmax(-A_normalized_distances / 0.1, axis=1)  # Using softmax for normalization
    
#     # # Compute weighted average of time differences
#     # A_weighted_time_differences = jnp.sum(A_weights *A_time_differences, axis=1)
    
#     # # Compute spatial component using minimum distances
#     # A_min_distances = jnp.min(A_distances, axis=1)

#     # #A_loss = 3 * jnp.mean(A_weighted_time_differences* (1-ray_weights))#+ jnp.mean(A_min_distances* (1-ray_weights[:, None]))
#     # #A_loss = jnp.mean(A_min_distances* (1-ray_weights))

#     # A_loss = 3 * jnp.mean(A_weighted_time_differences)+jnp.mean(A_min_distances* (1-ray_weights))


#     # #### ---- not reflected -----
#     # B_diff = final_closest_points[:, None, :] - true_hit_positions[None, :, :]
#     # B_distances = jnp.linalg.norm(B_diff, axis=-1)
#     # B_time_differences = jnp.abs(final_photon_times[:, None] - true_times[None, :])
    
#     # # Compute weights using a more efficient method
#     # B_normalized_distances = B_distances / jnp.max(B_distances, axis=1, keepdims=True)
#     # B_weights = jax.nn.softmax(-B_normalized_distances / 0.1, axis=1)  # Using softmax for normalization
    
#     # # Compute weighted average of time differences
#     # B_weighted_time_differences = jnp.sum(B_weights *B_time_differences, axis=1)
    
#     # # Compute spatial component using minimum distances
#     # B_min_distances = jnp.min(B_distances, axis=1)

#     # #B_loss = 3 * jnp.mean(B_weighted_time_differences*ray_weights) #+ jnp.mean((ray_weights) * B_min_distances)
#     # #B_loss = jnp.mean(B_min_distances* (ray_weights))

#     # B_loss = 3 * jnp.mean(B_weighted_time_differences)+jnp.mean(B_min_distances* (ray_weights))



#     #return A_loss+B_loss


