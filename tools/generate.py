
import jax
from functools import partial
from jax import random
import sys, os
import h5py
import numpy as np
import jax.numpy as jnp
import time
from jax import vmap


def gumbel_softmax_sample(prob, temperature, key):
    logits = jnp.array([jnp.log(prob), jnp.log(1-prob)])
    u = jax.random.uniform(key, shape=logits.shape)
    gumbel_noise = -jnp.log(-jnp.log(u + 1e-8) + 1e-8)
    y = jax.nn.softmax((logits + gumbel_noise) / temperature)
    return y[0]  # Return probability of going right

@jax.jit
def normalize(vector):
    return vector / jnp.linalg.norm(vector)

@partial(jax.jit, static_argnums=(2,3))
def generate_vectors_on_cone_surface_jax(R, theta, num_vectors=1, key=random.PRNGKey(0)):
    """ Generate vectors on the surface of a cone around R. """
    R = normalize(R)
    # Generate random azimuthal angles from 0 to 2pi
    phi_values = random.uniform(key, (num_vectors,), minval=0, maxval=2 * jnp.pi)
    # Generate vectors in the local coordinate system
    x_local = jnp.sin(theta) * jnp.cos(phi_values)
    y_local = jnp.sin(theta) * jnp.sin(phi_values)
    z_local = jnp.cos(theta) * jnp.ones_like(phi_values)
    local_vectors = jnp.stack([x_local, y_local, z_local], axis=-1)
    # Compute the rotation matrix to align [0, 0, 1] with R
    v = jnp.cross(jnp.array([0., 0., 1.]), R)
    s = jnp.linalg.norm(v)
    c = R[2]  # dot product of [0, 0, 1] and R
    v_cross = jnp.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    rotation_matrix = jnp.where(
        s > 1e-6,
        jnp.eye(3) + v_cross + v_cross.dot(v_cross) * (1 - c) / (s ** 2),
        jnp.where(c > 0, jnp.eye(3), jnp.diag(jnp.array([1., 1., -1.])))
    )
    # Apply the rotation to all vectors
    rotated_vectors = jnp.einsum('ij,kj->ki', rotation_matrix, local_vectors)
    return rotated_vectors


@partial(jax.jit, static_argnums=(3,))
def differentiable_get_rays(track_origin, track_direction, cone_opening, Nphot, key):
    # Generate ray vectors
    key, subkey = random.split(key)
    ray_vectors = generate_vectors_on_cone_surface_jax(track_direction, jnp.radians(cone_opening), Nphot, subkey)
    
    key, subkey = random.split(key)
    random_lengths = random.uniform(subkey, (Nphot, 1), minval=0, maxval=1)
    ray_origins = jnp.ones((Nphot, 3)) * track_origin + random_lengths * track_direction
    
    return ray_vectors, ray_origins


import time
import jax

def generate_and_store_event(filename, reflection_prob, cone_opening, track_origin, track_direction, photon_norm, att_L, trk_L, scatt_L, detector, Nphot, key):
    start_time = time.time()
    
    N_photosensors = len(detector.all_points)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    file_creation_start = time.time()
    with h5py.File(filename, 'w') as f_outfile:
        f_outfile.create_dataset("evt_id", data=np.array([0], dtype=np.int32))
        f_outfile.create_dataset("positions", data=np.array([track_origin], dtype=np.float32).reshape(1, 1, 3))

        f_outfile.create_dataset("reflection_prob", data=np.array([reflection_prob]))
        f_outfile.create_dataset("cone_opening", data=np.array([cone_opening]))
        f_outfile.create_dataset("track_origin", data=track_origin)
        f_outfile.create_dataset("track_direction", data=track_direction)

        f_outfile.create_dataset("photon_norm", data=photon_norm)
        f_outfile.create_dataset("att_L", data=att_L)
        f_outfile.create_dataset("trk_L", data=trk_L)
        f_outfile.create_dataset("scatt_L", data=scatt_L)

    file_creation_end = time.time()
    
    detector_points = jnp.array(detector.all_points)
    detector_radius = jnp.array(detector.r)
    detector_height = jnp.array(detector.H)
    detector_sensor_radius = jnp.array(detector.S_radius)
    
    photon_generation_start = time.time()
    final_closest_points, final_closest_detector_indices, final_photon_times, same_detector_count, closest_points, closest_detector_indices, photon_times, ray_weights = differentiable_photon_pmt_distance(
        reflection_prob, cone_opening, track_origin, track_direction, detector_points, detector_radius, detector_height, Nphot, key)

    print(f"Number of photons hitting the same detector after reflection: {same_detector_count}")
    
    jax.block_until_ready(final_closest_points)
    jax.block_until_ready(final_closest_detector_indices)
    jax.block_until_ready(final_photon_times)
    jax.block_until_ready(closest_points)
    jax.block_until_ready(closest_detector_indices)
    jax.block_until_ready(photon_times)
    jax.block_until_ready(ray_weights)
    photon_generation_end = time.time()
    
    hit_calculation_start = time.time()
    # Use ray_weights to choose between reflected and non-reflected data
    reflection_mask = ray_weights > 0.5
    selected_points = jnp.where(reflection_mask[:, None], final_closest_points, closest_points)
    selected_detector_indices = jnp.where(reflection_mask, final_closest_detector_indices, closest_detector_indices)
    selected_photon_times = jnp.where(reflection_mask, final_photon_times, photon_times)

    hit_flag = jnp.linalg.norm(selected_points - detector_points[selected_detector_indices], axis=1) < 0.04 # this needs to be read from geometry file!!
    valid_indices = selected_detector_indices[hit_flag]
    valid_photon_times = selected_photon_times[hit_flag]
    jax.block_until_ready(hit_flag)
    jax.block_until_ready(valid_indices)
    jax.block_until_ready(valid_photon_times)
    hit_calculation_end = time.time()
    
    unique_calculation_start = time.time()
    idx, idx_inverse = jnp.unique(valid_indices, return_inverse=True)
    cts = jnp.bincount(idx_inverse)
    Nhits = len(idx)
    jax.block_until_ready(idx)
    jax.block_until_ready(cts)
    unique_calculation_end = time.time()
    
    data_storage_start = time.time()
    with h5py.File(filename, 'a') as f_outfile:
        f_outfile.create_dataset("event_hits_index", data=np.array([Nhits], dtype=np.int64))

        h5_evt_hit_IDs = f_outfile.create_dataset("hit_pmt", shape=(Nhits,), dtype=np.int32)
        h5_evt_hit_Qs = f_outfile.create_dataset("hit_charge", shape=(Nhits,), dtype=np.float32)
        h5_evt_hit_Ts = f_outfile.create_dataset("hit_time", shape=(Nhits,), dtype=np.float32)

        h5_evt_hit_IDs[:] = idx
        h5_evt_hit_Qs[:] = cts
        h5_evt_hit_Ts[:] = [np.mean(valid_photon_times[valid_indices == i]) for i in idx]
    data_storage_end = time.time()
    
    end_time = time.time()
    
    print(f"Total execution time: {end_time - start_time:.4f} seconds")
    print(f"File creation time: {file_creation_end - file_creation_start:.4f} seconds")
    print(f"Photon generation time: {photon_generation_end - photon_generation_start:.4f} seconds")
    print(f"Hit calculation time: {hit_calculation_end - hit_calculation_start:.4f} seconds")
    print(f"Unique calculation time: {unique_calculation_end - unique_calculation_start:.4f} seconds")
    print(f"Data storage time: {data_storage_end - data_storage_start:.4f} seconds")
    
    return filename

@jax.jit
def propagate_single_photon(ray_vector, ray_origin, detector_points):
    ray_to_detector = detector_points - ray_origin
    dot_product = jnp.sum(ray_vector * ray_to_detector, axis=-1)
    ray_mag_squared = jnp.sum(ray_vector ** 2)
    t = dot_product / ray_mag_squared
    
    # Ensure t is strictly positive (closest point is in front of the ray origin)
    t = jnp.maximum(t, 1e-3)
    
    closest_points = ray_origin + t[:, None] * ray_vector
    distances = jnp.linalg.norm(closest_points - detector_points, axis=-1)
    
    closest_detector_index = jnp.argmin(distances)
    closest_point = closest_points[closest_detector_index]
    photon_time = jnp.linalg.norm(closest_point - ray_origin)
    
    return closest_point, closest_detector_index, photon_time

@jax.jit
def propagate(ray_vectors, ray_origins, detector_points):
    # Use vmap to apply the single-photon function to all photons
    return jax.vmap(propagate_single_photon, in_axes=(0, 0, None))(ray_vectors, ray_origins, detector_points)


def create_surface_masks(closest_detector_indices):
    barrel_mask = (closest_detector_indices < 6720)
    top_cap_mask = (closest_detector_indices >= 6720) & (closest_detector_indices < 6720 + 1613)
    bottom_cap_mask = (closest_detector_indices >= 6720 + 1613)
    return barrel_mask, top_cap_mask, bottom_cap_mask

def calculate_reflections(ray_vectors, ray_origins, surface_masks, detector_radius, detector_height, epsilon):
    barrel_mask, top_cap_mask, bottom_cap_mask = surface_masks
    
    # Barrel reflection calculation
    new_ray_vectors_barrel, t_barrel = calculate_barrel_reflection(ray_vectors, ray_origins, detector_radius, epsilon)
    
    # Top cap reflection calculation
    new_ray_vectors_top, t_top = calculate_cap_reflection(ray_vectors, ray_origins, detector_height/2, epsilon)
    
    # Bottom cap reflection calculation
    new_ray_vectors_bottom, t_bottom = calculate_cap_reflection(ray_vectors, ray_origins, -detector_height/2, epsilon)
    
    # Combine new ray vectors and intersection points
    new_ray_vectors = jnp.where(barrel_mask[:, None], new_ray_vectors_barrel,
                                jnp.where(top_cap_mask[:, None], new_ray_vectors_top,
                                          jnp.where(bottom_cap_mask[:, None], new_ray_vectors_bottom,
                                                    ray_vectors)))
    
    new_ray_origins = jnp.where(barrel_mask[:, None], ray_origins + t_barrel[:, None] * ray_vectors,
                                jnp.where(top_cap_mask[:, None], ray_origins + t_top[:, None] * ray_vectors,
                                          jnp.where(bottom_cap_mask[:, None], ray_origins + t_bottom[:, None] * ray_vectors,
                                                    ray_origins)))
    
    time_to_reflection = jnp.where(barrel_mask, t_barrel,
                                   jnp.where(top_cap_mask, t_top,
                                             jnp.where(bottom_cap_mask, t_bottom, 0.0)))
    
    return new_ray_vectors, new_ray_origins, time_to_reflection

def calculate_barrel_reflection(ray_vectors, ray_origins, detector_radius, epsilon):
    a = jnp.sum(ray_vectors[:, :2]**2, axis=1)
    b = 2 * jnp.sum(ray_origins[:, :2] * ray_vectors[:, :2], axis=1)
    c = jnp.sum(ray_origins[:, :2]**2, axis=1) - detector_radius**2
    discriminant = jnp.maximum(b**2 - 4*a*c, epsilon)
    t_barrel = jnp.where(discriminant >= 0, (-b + jnp.sqrt(discriminant)) / (2*a + epsilon), jnp.inf)
    barrel_intersection_points = ray_origins + t_barrel[:, None] * ray_vectors
    normal_vectors = barrel_intersection_points[:, :2] / (detector_radius + epsilon)
    normal_vectors = jnp.concatenate([normal_vectors, jnp.zeros((len(ray_vectors), 1))], axis=1)
    dot_product = jnp.clip(jnp.sum(ray_vectors * normal_vectors, axis=1, keepdims=True), -1 + epsilon, 1 - epsilon)
    new_ray_vectors_barrel = ray_vectors - 2 * dot_product * normal_vectors
    return new_ray_vectors_barrel, t_barrel

def calculate_cap_reflection(ray_vectors, ray_origins, cap_z, epsilon):
    t_cap = (cap_z - ray_origins[:, 2]) / (ray_vectors[:, 2] + epsilon)
    new_ray_vectors_cap = ray_vectors.at[:, 2].multiply(-1)
    return new_ray_vectors_cap, t_cap

def step_along_ray(ray_origins, ray_vectors, delta):
    return ray_origins + delta * ray_vectors

def check_within_detector(points, detector_radius, detector_height):
    r_squared = points[:, 0]**2 + points[:, 1]**2
    z_abs = jnp.abs(points[:, 2])
    return (r_squared <= detector_radius**2) & (z_abs <= detector_height / 2)

def update_rays(new_origins, new_vectors, old_origins, old_vectors, within_detector):
    return (jnp.where(within_detector[:, None], new_origins, old_origins),
            jnp.where(within_detector[:, None], new_vectors, old_vectors))

def combine_results(closest_points, closest_detector_indices, photon_times,
                    new_closest_points, new_closest_detector_indices, new_photon_times,
                    surface_masks, within_detector, time_to_reflection):
    barrel_mask, top_cap_mask, bottom_cap_mask = surface_masks
    reflection_mask = (barrel_mask | top_cap_mask | bottom_cap_mask) & within_detector
    same_detector_mask = closest_detector_indices == new_closest_detector_indices
    valid_reflection_mask = reflection_mask & ~same_detector_mask
    
    final_closest_points = jnp.where(valid_reflection_mask[:, None], new_closest_points, closest_points)
    final_closest_detector_indices = jnp.where(valid_reflection_mask, new_closest_detector_indices, closest_detector_indices)
    final_photon_times = jnp.where(valid_reflection_mask, time_to_reflection + new_photon_times, photon_times)
    
    return final_closest_points, final_closest_detector_indices, final_photon_times

def count_same_detectors(final_results, surface_masks):
    _, final_closest_detector_indices, _ = final_results
    barrel_mask, top_cap_mask, bottom_cap_mask = surface_masks
    reflection_mask = barrel_mask | top_cap_mask | bottom_cap_mask
    same_detector_mask = final_closest_detector_indices == final_closest_detector_indices  # This line needs to be adjusted
    return jnp.sum(reflection_mask & same_detector_mask)

def calculate_ray_weights(reflection_prob, keys):
    return vmap(lambda k: gumbel_softmax_sample(reflection_prob, 0.1, k))(keys)

@jax.jit
def inverse_theta_cdf(y):
    return jnp.arccos(jnp.sqrt(y))


@partial(jax.jit, static_argnums=(7,))
def differentiable_photon_pmt_distance(
    reflection_prob, cone_opening, track_origin, track_direction, 
    detector_points, detector_radius, detector_height, Nphot, key):
    
    attenuation_length = 10
    epsilon = 1e-4  # Small value to prevent division by zero
    
    # Generate initial rays
    ray_vectors, ray_origins = differentiable_get_rays(track_origin, track_direction, cone_opening, Nphot, key)
    
    # First propagation
    closest_points, closest_detector_indices, photon_times = propagate(ray_vectors, ray_origins, detector_points)
    
    # Calculate ray lengths
    ray_lengths = jnp.linalg.norm(closest_points - ray_origins, axis=-1)
    
    # Sample scattering distances
    key, subkey1, subkey2 = random.split(key,3)
    scattering_distances = random.exponential(subkey1, shape=(Nphot,)) * attenuation_length
    
    # Determine which rays scatter
    scatter_mask = scattering_distances < ray_lengths
    
    # Calculate new origins for scattered rays
    scattered_origins = ray_origins + scatter_mask[:, None] * ray_vectors * scattering_distances[:, None]
    
    # Generate new directions for scattered rays
    # phi = random.uniform(subkey1, shape=(Nphot,), minval=0, maxval=2 * jnp.pi)
    y = random.uniform(subkey2, shape=(Nphot,))
    theta = inverse_theta_cdf(y)

    scattered_vectors = jax.vmap(generate_vectors_on_cone_surface_jax)(ray_vectors, theta).reshape(len(ray_vectors),3)

    keys = random.split(key, Nphot)
    # Use vmap to apply the function over the rays and keys
    scattered_vectors = jax.vmap(generate_vectors_on_cone_surface_jax, in_axes=(0, 0, None, 0))(ray_vectors, theta, 1, keys).reshape(Nphot, 3)

    # Combine scattered and non-scattered rays
    new_ray_origins = jnp.where(scatter_mask[:, jnp.newaxis], scattered_origins, ray_origins)
    new_ray_vectors = jnp.where(scatter_mask[:, jnp.newaxis], scattered_vectors, ray_vectors)
    
    # Second propagation with potentially scattered rays
    new_closest_points, new_closest_detector_indices, new_photon_times = propagate(new_ray_vectors, new_ray_origins, detector_points)
    
    # Handle reflections
    keys = jax.random.split(key, len(new_closest_detector_indices))
    surface_masks = create_surface_masks(new_closest_detector_indices)
    
    barrel_mask, top_cap_mask, bottom_cap_mask = surface_masks
    reflected_ray_vectors, reflected_ray_origins, time_to_reflection = calculate_reflections(
        new_ray_vectors, new_ray_origins, surface_masks, detector_radius, detector_height, epsilon)
    reflected_ray_origins_stepped = step_along_ray(reflected_ray_origins, reflected_ray_vectors, delta=0.2)
    within_detector = check_within_detector(reflected_ray_origins_stepped, detector_radius, detector_height)
    final_ray_origins, final_ray_vectors = update_rays(
        reflected_ray_origins_stepped, reflected_ray_vectors, new_ray_origins, new_ray_vectors, within_detector)
    final_closest_points, final_closest_detector_indices, final_photon_times = propagate(final_ray_vectors, final_ray_origins, detector_points)
    final_results = combine_results(
        new_closest_points, new_closest_detector_indices, new_photon_times,
        final_closest_points, final_closest_detector_indices, final_photon_times,
        surface_masks, within_detector, time_to_reflection
    )
    same_detector_count = count_same_detectors(final_results, surface_masks)
    ray_weights = calculate_ray_weights(reflection_prob, keys)
    #return (*final_results, same_detector_count, new_closest_points, new_closest_detector_indices, new_photon_times, ray_weights, scatter_mask)
    return (*final_results, same_detector_count, new_closest_points, new_closest_detector_indices, new_photon_times, ray_weights)









# import jax
# import jax.numpy as jnp
# from functools import partial

# @partial(jax.jit, static_argnums=(7,))
# def differentiable_photon_pmt_distance(
#     reflection_prob, cone_opening, track_origin, track_direction, 
#     detector_points, detector_radius, detector_height, Nphot, key):
#     attenuation_length = 0.1
#     epsilon = 1e-4  # Small value to prevent division by zero
    
#     # Generate initial rays
#     ray_vectors, ray_origins = differentiable_get_rays(track_origin, track_direction, cone_opening, Nphot, key)
    
#     # First propagation
#     closest_points, closest_detector_indices, photon_times = propagate(ray_vectors, ray_origins, detector_points)
    
#     # Calculate ray lengths
#     ray_lengths = jnp.linalg.norm(closest_points - ray_origins, axis=-1)
    
#     # Sample scattering distances
#     key, subkey = jax.random.split(key)
#     scattering_distances = jax.random.exponential(subkey, shape=(Nphot,)) * attenuation_length
    
#     # Determine which rays scatter
#     scatter_mask = scattering_distances < ray_lengths
    
#     # Calculate new origins for scattered rays
#     scattered_origins = ray_origins + scatter_mask[:, None] * ray_vectors * scattering_distances[:, None]
    
#     # Generate new directions for scattered rays
#     key, subkey1, subkey2 = jax.random.split(key, 3)
#     phi = jax.random.uniform(subkey1, shape=(Nphot,), minval=0, maxval=2 * jnp.pi)
#     y = jax.random.uniform(subkey2, shape=(Nphot,))
#     theta = inverse_theta_cdf(y)
    
#     # Use vmap to vectorize generate_vectors_on_cone_surface_jax
#     scattered_vectors = jax.vmap(generate_vectors_on_cone_surface_jax)(ray_vectors, theta)
    
#     # Combine scattered and non-scattered rays
#     new_ray_origins = jnp.where(scatter_mask[:, None], scattered_origins, ray_origins)
#     new_ray_vectors = jnp.where(scatter_mask[:, None], scattered_vectors, ray_vectors)
    
#     # Second propagation with potentially scattered rays
#     new_closest_points, new_closest_detector_indices, new_photon_times = propagate(new_ray_vectors, new_ray_origins, detector_points)
    
#     # Handle reflections
#     keys = jax.random.split(key, len(new_closest_detector_indices))
#     surface_masks = create_surface_masks(new_closest_detector_indices)
#     barrel_mask, top_cap_mask, bottom_cap_mask = surface_masks
    
#     # Use vmap for calculate_reflections
#     reflected_ray_vectors, reflected_ray_origins, time_to_reflection = jax.vmap(calculate_reflections)(
#         new_ray_vectors, new_ray_origins, surface_masks, detector_radius, detector_height, epsilon)
    
#     reflected_ray_origins_stepped = step_along_ray(reflected_ray_origins, reflected_ray_vectors, delta=0.2)
#     within_detector = check_within_detector(reflected_ray_origins_stepped, detector_radius, detector_height)
    
#     final_ray_origins, final_ray_vectors = update_rays(
#         reflected_ray_origins_stepped, reflected_ray_vectors, new_ray_origins, new_ray_vectors, within_detector)
    
#     final_closest_points, final_closest_detector_indices, final_photon_times = propagate(final_ray_vectors, final_ray_origins, detector_points)
    
#     final_results = combine_results(
#         new_closest_points, new_closest_detector_indices, new_photon_times,
#         final_closest_points, final_closest_detector_indices, final_photon_times,
#         surface_masks, within_detector, time_to_reflection
#     )
    
#     same_detector_count = count_same_detectors(final_results, surface_masks)
#     ray_weights = calculate_ray_weights(reflection_prob, keys)
    
#     return (*final_results, same_detector_count, new_closest_points, new_closest_detector_indices, new_photon_times, ray_weights, scatter_mask)



