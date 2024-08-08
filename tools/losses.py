

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
    
    # final_closest_points, final_closest_detector_indices, final_photon_times, same_detector_count, closest_points, closest_detector_indices, photon_times, ray_weights, scatter_mask = differentiable_photon_pmt_distance(
    #     reflection_prob, cone_opening, track_origin, track_direction, detector_points, detector_radius, detector_height, Nphot, key
    # )

    true_hit_positions = detector_points[true_indices]

    direct_closest_points, direct_closest_detector_indices, direct_photon_times, \
    reflected_closest_points, reflected_closest_detector_indices, reflected_photon_times, \
    scattered_closest_points, scattered_closest_detector_indices, scattered_photon_times, \
    ray_weights, scattering_distances, scatter_mask = differentiable_photon_pmt_distance(
        reflection_prob, cone_opening, track_origin, track_direction, detector_points, detector_radius, detector_height, Nphot, key
    )

    selected_points, selected_detector_indices, selected_photon_times, hit_flag_direct, hit_flag_reflec, hit_flag_scatter, global_hit_flag =  calculate_selected_points(
        direct_closest_points, direct_closest_detector_indices, direct_photon_times,
        reflected_closest_points, reflected_closest_detector_indices, reflected_photon_times,
        scattered_closest_points, scattered_closest_detector_indices, scattered_photon_times,
        ray_weights, scatter_mask, detector_points
    )

    #### --loss based on the space and time distance of the photons to the recorded hits--
    ### Probably it is a good idea to look at this only for direct photons...
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

    Pattern_Loss = 3 * jnp.mean(weighted_time_differences)+jnp.mean(min_distances)
    #### -------------------------------------------------------------------------------
    
    fraction_of_scattered_photons = jnp.sum(scatter_mask>0)/Nphot

    max_index = len(detector_points)
    reco_hits_direct = jax.ops.segment_sum((1-reflection_prob)*(1-fraction_of_scattered_photons), jnp.where(hit_flag_direct, direct_closest_detector_indices, -1),    num_segments=max_index+1)[:max_index]
    reco_hits_reflec = jax.ops.segment_sum(reflection_prob*(1-fraction_of_scattered_photons),     jnp.where(hit_flag_reflec, reflected_closest_detector_indices, -1), num_segments=max_index+1)[:max_index]
    
    # Create a mask for non-zero elements
    mask_direct = reco_hits_direct > 0
    mask_reflec = reco_hits_reflec > 0
    
    # Use the masks to filter true_hits
    filtered_true_hits_direct = jnp.where(mask_direct, true_hits, 0.0)
    filtered_true_hits_reflec = jnp.where(mask_reflec, true_hits, 0.0)
    
    Q_sym = (jnp.sum(reco_hits_direct)*lax.stop_gradient(photon_norm) -jnp.sum(filtered_true_hits_direct))**2 + (jnp.sum(reco_hits_reflec)*lax.stop_gradient(photon_norm) -jnp.sum(filtered_true_hits_reflec))**2

    reco_hits = jax.ops.segment_sum(1, jnp.where(global_hit_flag, selected_detector_indices, -1), num_segments=max_index+1)[:max_index]
    Q_tot = (jnp.sum(reco_hits)*photon_norm-jnp.sum(true_hits))**2*jnp.mean((reco_hits*photon_norm - true_hits) ** 2)

    return Pattern_Loss+0.1*Q_sym/(Nphot**2)+Q_tot/(Nphot**2)






