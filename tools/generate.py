
import jax
from functools import partial
from jax import random
import sys, os
import h5py
import numpy as np
import jax.numpy as jnp
import time


@jax.jit
def normalize(vector):
    return vector / jnp.linalg.norm(vector)

@partial(jax.jit, static_argnums=(2,3))
def generate_vectors_on_cone_surface_jax(R, theta, num_vectors=10, key=random.PRNGKey(0)):
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

def check_hits_vectorized_per_track_jax(ray_origin, ray_direction, sensor_radius, points):
    # Ensure inputs are JAX arrays
    ray_origin_jax = jnp.array(ray_origin, dtype=jnp.float32)
    ray_direction_jax = jnp.array(ray_direction, dtype=jnp.float32)
    points_jax = jnp.array(points, dtype=jnp.float32)

    # Calculate vectors from ray origin to all points
    vectors_to_points = points_jax - ray_origin_jax[:, None, :]

    # Project all vectors onto the ray direction
    dot_products_numerator = jnp.einsum('ijk,ik->ij', vectors_to_points, ray_direction_jax)
    dot_products_denominator = jnp.sum(ray_direction_jax * ray_direction_jax, axis=-1)

    # Calculate t_values
    t_values = dot_products_numerator / dot_products_denominator[:, None]

    # Calculate the points on the ray closest to the given points
    closest_points_on_ray = ray_origin_jax[:, None, :] + t_values[:, :, None] * ray_direction_jax[:, None, :]

    # Calculate the Euclidean distances between all points and their closest points on the ray
    distances = jnp.linalg.norm(points_jax - closest_points_on_ray, axis=2)

    # Apply the mask
    mask = t_values < 0
    distances = jnp.where(mask, 999.0, distances)

    # Find the indices of the minimum distances
    indices = jnp.argmin(distances, axis=1)

    # True if the photon is on the photosensor False otherwise
    hit_flag = distances[jnp.arange(indices.size), indices] < sensor_radius

    # Get the good indices based on sensor_radius
    sensor_indices = indices[hit_flag]

    return sensor_indices, hit_flag, closest_points_on_ray


@partial(jax.jit, static_argnums=(3,))
def differentiable_get_rays(track_origin, track_direction, cone_opening, Nphot, key):
    # Generate ray vectors
    key, subkey = random.split(key)
    ray_vectors = generate_vectors_on_cone_surface_jax(track_direction, jnp.radians(cone_opening), Nphot, key=subkey)
    
    # Generate ray origins
    key, subkey = random.split(key)
    random_lengths = random.uniform(subkey, (Nphot, 1), minval=0, maxval=1)
    ray_origins = jnp.ones((Nphot, 3)) * track_origin + random_lengths * track_direction
    
    return ray_vectors, ray_origins

@jax.jit
def generate_rotation_matrix(vector):
    v = normalize(vector)
    z = jnp.array([0., 0., 1.])
    axis = jnp.cross(z, v)
    cos_theta = jnp.dot(z, v)
    sin_theta = jnp.linalg.norm(axis)
    
    # Instead of using an if statement, use jnp.where
    identity = jnp.eye(3)
    flipped = jnp.diag(jnp.array([1., 1., -1.]))
    
    axis = jnp.where(sin_theta > 1e-6, axis / sin_theta, axis)
    K = jnp.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    
    rotation = identity + sin_theta * K + (1 - cos_theta) * jnp.dot(K, K)
    
    return jnp.where(sin_theta > 1e-6, 
                     rotation, 
                     jnp.where(cos_theta > 0, identity, flipped))


# def generate_and_store_event(filename, cone_opening, track_origin, track_direction, detector, Nphot, key):
#     N_photosensors = len(detector.all_points)

#     os.makedirs(os.path.dirname(filename), exist_ok=True)

#     # Create the output h5 file
#     with h5py.File(filename, 'w') as f_outfile:
#         # Add evt_id, positions, and event_hits_index datasets
#         f_outfile.create_dataset("evt_id", data=np.array([0], dtype=np.int32))
#         f_outfile.create_dataset("positions", data=np.array([track_origin], dtype=np.float32).reshape(1, 1, 3))
        
#         f_outfile.create_dataset("true_cone_opening", data=np.array([cone_opening]))
#         f_outfile.create_dataset("true_track_origin", data=track_origin)
#         f_outfile.create_dataset("true_track_direction", data=track_direction)

#         # Convert inputs to JAX arrays
#         cone_opening_jax = jnp.array(cone_opening)
#         track_origin_jax = jnp.array(track_origin)
#         track_direction_jax = jnp.array(track_direction)
#         detector_points_jax = jnp.array(detector.all_points)
#         detector_radius_jax = jnp.array(detector.S_radius)

#         # Use differentiable_photon_pmt_distance to generate photon data
#         closest_points, closest_detector_indices, photon_times = differentiable_photon_pmt_distance(
#             cone_opening_jax, track_origin_jax, track_direction_jax, detector_points_jax, detector_radius_jax, Nphot, key)

#         # Calculate hit information
#         hit_flag = jnp.linalg.norm(closest_points - detector_points_jax[closest_detector_indices], axis=1) < detector_radius_jax
#         valid_indices = closest_detector_indices[hit_flag]
#         valid_photon_times = photon_times[hit_flag]

#         # Use JAX's unique function
#         idx, idx_inverse = jnp.unique(valid_indices, return_inverse=True)
#         cts = jnp.bincount(idx_inverse)
#         Nhits = len(idx)

#         # Add event_hits_index dataset
#         f_outfile.create_dataset("event_hits_index", data=np.array([Nhits], dtype=np.int64))

#         h5_evt_hit_IDs = f_outfile.create_dataset("hit_pmt", shape=(Nhits,), dtype=np.int32)
#         h5_evt_hit_Qs = f_outfile.create_dataset("hit_charge", shape=(Nhits,), dtype=np.float32)
#         h5_evt_hit_Ts = f_outfile.create_dataset("hit_time", shape=(Nhits,), dtype=np.float32)

#         h5_evt_hit_IDs[:] = idx
#         h5_evt_hit_Qs[:] = cts
#         h5_evt_hit_Ts[:] = [np.mean(valid_photon_times[valid_indices == i]) for i in idx]

#     return filename

import time
import jax

def generate_and_store_event(filename, cone_opening, track_origin, track_direction, detector, Nphot, key):
    start_time = time.time()
    
    N_photosensors = len(detector.all_points)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    file_creation_start = time.time()
    with h5py.File(filename, 'w') as f_outfile:
        f_outfile.create_dataset("evt_id", data=np.array([0], dtype=np.int32))
        f_outfile.create_dataset("positions", data=np.array([track_origin], dtype=np.float32).reshape(1, 1, 3))

        f_outfile.create_dataset("true_cone_opening", data=np.array([cone_opening]))
        f_outfile.create_dataset("true_track_origin", data=track_origin)
        f_outfile.create_dataset("true_track_direction", data=track_direction)
    file_creation_end = time.time()
    
    jax_conversion_start = time.time()
    cone_opening_jax = jnp.array(cone_opening)
    track_origin_jax = jnp.array(track_origin)
    track_direction_jax = jnp.array(track_direction)
    detector_points_jax = jnp.array(detector.all_points)
    detector_radius_jax = jnp.array(detector.S_radius)
    jax_conversion_end = time.time()
    
    photon_generation_start = time.time()
    closest_points, closest_detector_indices, photon_times = differentiable_photon_pmt_distance(
        cone_opening_jax, track_origin_jax, track_direction_jax, detector_points_jax, detector_radius_jax, Nphot, key)
    jax.block_until_ready(closest_points)
    jax.block_until_ready(closest_detector_indices)
    jax.block_until_ready(photon_times)
    photon_generation_end = time.time()
    
    hit_calculation_start = time.time()
    hit_flag = jnp.linalg.norm(closest_points - detector_points_jax[closest_detector_indices], axis=1) < detector_radius_jax
    valid_indices = closest_detector_indices[hit_flag]
    valid_photon_times = photon_times[hit_flag]
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
    print(f"JAX conversion time: {jax_conversion_end - jax_conversion_start:.4f} seconds")
    print(f"Photon generation time: {photon_generation_end - photon_generation_start:.4f} seconds")
    print(f"Hit calculation time: {hit_calculation_end - hit_calculation_start:.4f} seconds")
    print(f"Unique calculation time: {unique_calculation_end - unique_calculation_start:.4f} seconds")
    print(f"Data storage time: {data_storage_end - data_storage_start:.4f} seconds")
    
    return filename


# @partial(jax.jit, static_argnums=(5,))
# def differentiable_photon_pmt_distance(cone_opening, track_origin, track_direction, detector_points, detector_radius, Nphot, key):
#     ray_vectors, ray_origins = differentiable_get_rays(track_origin, track_direction, cone_opening, Nphot, key)

#     t = jnp.linspace(0, 10, 100)[:, None]
#     points_along_rays = ray_origins[:, None, :] + t * ray_vectors[:, None, :]

#     distances = jnp.linalg.norm(points_along_rays[:, :, None, :] - detector_points[None, None, :, :], axis=-1)
#     min_distances = jnp.min(distances, axis=1)
#     closest_detector_indices = jnp.argmin(min_distances, axis=1)

#     closest_points = points_along_rays[jnp.arange(Nphot), jnp.argmin(distances, axis=1)[jnp.arange(Nphot), closest_detector_indices]]
    
#     # Calculate time for each photon (assuming speed of light = 1)
#     photon_times = jnp.linalg.norm(closest_points - ray_origins, axis=-1)

#     return closest_points, closest_detector_indices, photon_times




# @partial(jax.jit, static_argnums=(5,))
# def differentiable_photon_pmt_distance(cone_opening, track_origin, track_direction, detector_points, detector_radius, Nphot, key):
#     ray_vectors, ray_origins = differentiable_get_rays(track_origin, track_direction, cone_opening, Nphot, key)
    
#     # Calculate the vector from ray origin to each detector
#     ray_to_detector = detector_points[None, :, :] - ray_origins[:, None, :]
    
#     # Calculate the dot product of ray vector and ray-to-detector vector
#     dot_product = jnp.sum(ray_vectors[:, None, :] * ray_to_detector, axis=-1)
    
#     # Calculate the squared magnitude of ray vectors
#     ray_mag_squared = jnp.sum(ray_vectors ** 2, axis=-1)[:, None]
    
#     # Calculate the parameter t for the closest point on each ray
#     t = dot_product / ray_mag_squared
    
#     # Calculate the closest points
#     closest_points = ray_origins[:, None, :] + t[:, :, None] * ray_vectors[:, None, :]
    
#     # Calculate the distances to the closest points
#     distances = jnp.linalg.norm(closest_points - detector_points[None, :, :], axis=-1)
    
#     # Find the closest detector for each photon
#     closest_detector_indices = jnp.argmin(distances, axis=1)
    
#     # Get the actual closest points
#     closest_points = closest_points[jnp.arange(Nphot), closest_detector_indices]
    
#     # Calculate photon times
#     photon_times = jnp.linalg.norm(closest_points - ray_origins, axis=-1)
    
#     return closest_points, closest_detector_indices, photon_times


@partial(jax.jit, static_argnums=(5,))
def differentiable_photon_pmt_distance(cone_opening, track_origin, track_direction, detector_points, detector_radius, Nphot, key):
    ray_vectors, ray_origins = differentiable_get_rays(track_origin, track_direction, cone_opening, Nphot, key)
    
    # Calculate the vector from ray origin to each detector
    ray_to_detector = detector_points[None, :, :] - ray_origins[:, None, :]
    
    # Calculate the dot product of ray vector and ray-to-detector vector
    dot_product = jnp.sum(ray_vectors[:, None, :] * ray_to_detector, axis=-1)
    
    # Calculate the squared magnitude of ray vectors
    ray_mag_squared = jnp.sum(ray_vectors ** 2, axis=-1)[:, None]
    
    # Calculate the parameter t for the closest point on each ray
    t = dot_product / ray_mag_squared
    
    # Ensure t is non-negative (closest point is in the direction of the ray)
    t = jnp.maximum(t, 0)
    
    # Calculate the closest points
    closest_points = ray_origins[:, None, :] + t[:, :, None] * ray_vectors[:, None, :]
    
    # Calculate the distances to the closest points
    distances = jnp.linalg.norm(closest_points - detector_points[None, :, :], axis=-1)
    
    # Find the closest detector for each photon
    closest_detector_indices = jnp.argmin(distances, axis=1)
    
    # Get the actual closest points
    closest_points = closest_points[jnp.arange(Nphot), closest_detector_indices]
    
    # Calculate photon times
    photon_times = jnp.linalg.norm(closest_points - ray_origins, axis=-1)
    
    return closest_points, closest_detector_indices, photon_times































# def generate_and_store_event(filename, cone_opening, track_origin, track_direction, detector, Nphot, key):
#     start_total = time.time()

#     N_photosensors = len(detector.all_points)
#     os.makedirs(os.path.dirname(filename), exist_ok=True)

#     start_hdf5 = time.time()
#     with h5py.File(filename, 'w') as f_outfile:
#         f_outfile.create_dataset("evt_id", data=np.array([0], dtype=np.int32))
#         f_outfile.create_dataset("positions", data=np.array([track_origin], dtype=np.float32).reshape(1, 1, 3))
        
#         f_outfile.create_dataset("true_cone_opening", data=np.array([cone_opening]))
#         f_outfile.create_dataset("true_track_origin", data=track_origin)
#         f_outfile.create_dataset("true_track_direction", data=track_direction)
#     end_hdf5_initial = time.time()

#     start_jax_conversion = time.time()
#     cone_opening_jax = jnp.array(cone_opening)
#     track_origin_jax = jnp.array(track_origin)
#     track_direction_jax = jnp.array(track_direction)
#     detector_points_jax = jnp.array(detector.all_points)
#     detector_radius_jax = jnp.array(detector.S_radius)
#     end_jax_conversion = time.time()

#     start_compilation = time.time()
#     jitted_photon_pmt_distance = jax.jit(differentiable_photon_pmt_distance, static_argnums=(5,))
#     end_compilation = time.time()

#     start_photon_simulation = time.time()
#     closest_points, closest_detector_indices, photon_times = jitted_photon_pmt_distance(
#         cone_opening_jax, track_origin_jax, track_direction_jax, detector_points_jax, detector_radius_jax, Nphot, key)
#     end_photon_simulation = time.time()

#     start_hit_calculation = time.time()
#     hit_flag = jnp.linalg.norm(closest_points - detector_points_jax[closest_detector_indices], axis=1) < detector_radius_jax
#     valid_indices = closest_detector_indices[hit_flag]
#     valid_photon_times = photon_times[hit_flag]

#     idx, idx_inverse = jnp.unique(valid_indices, return_inverse=True)
#     cts = jnp.bincount(idx_inverse)
#     Nhits = len(idx)
#     end_hit_calculation = time.time()

#     start_hdf5_final = time.time()
#     with h5py.File(filename, 'a') as f_outfile:
#         f_outfile.create_dataset("event_hits_index", data=np.array([Nhits], dtype=np.int64))
#         h5_evt_hit_IDs = f_outfile.create_dataset("hit_pmt", shape=(Nhits,), dtype=np.int32)
#         h5_evt_hit_Qs = f_outfile.create_dataset("hit_charge", shape=(Nhits,), dtype=np.float32)
#         h5_evt_hit_Ts = f_outfile.create_dataset("hit_time", shape=(Nhits,), dtype=np.float32)

#         h5_evt_hit_IDs[:] = np.array(idx)
#         h5_evt_hit_Qs[:] = np.array(cts)
#         h5_evt_hit_Ts[:] = np.array([jnp.mean(valid_photon_times[valid_indices == i]) for i in idx])
#     end_hdf5_final = time.time()

#     end_total = time.time()

#     # Calculate and print durations
#     duration_total = end_total - start_total
#     duration_hdf5_initial = end_hdf5_initial - start_hdf5
#     duration_jax_conversion = end_jax_conversion - start_jax_conversion
#     duration_photon_simulation = end_photon_simulation - start_photon_simulation
#     duration_hit_calculation = end_hit_calculation - start_hit_calculation
#     duration_hdf5_final = end_hdf5_final - start_hdf5_final

#     print(f"Total time: {duration_total:.6f} seconds")
#     print(f"Initial HDF5 operations: {duration_hdf5_initial:.6f} seconds")
#     print(f"JAX array conversion: {duration_jax_conversion:.6f} seconds")
#     print(f"Photon simulation: {duration_photon_simulation:.6f} seconds")
#     print(f"Hit calculation: {duration_hit_calculation:.6f} seconds")
#     print(f"Final HDF5 operations: {duration_hdf5_final:.6f} seconds")

#     print(f"Compilation time: {end_compilation - start_compilation:.6f} seconds")
#     print(f"Photon simulation execution time: {end_photon_simulation - start_photon_simulation:.6f} seconds")

#     return filename

# @partial(jax.jit, static_argnums=(5,))
# def differentiable_photon_pmt_distance(cone_opening, track_origin, track_direction, detector_points, detector_radius, Nphot, key):
#     # Start total time measurement
#     start_total = time.time()

#     # Start timer for differentiable_get_rays
#     start_get_rays = time.time()
#     ray_vectors, ray_origins = differentiable_get_rays(track_origin, track_direction, cone_opening, Nphot, key)
#     end_get_rays = time.time()

#     # Start timer for the rest of the computation
#     start_rest = time.time()

#     t = jnp.linspace(0, 10, 100)[:, None]
#     points_along_rays = ray_origins[:, None, :] + t * ray_vectors[:, None, :]

#     distances = jnp.linalg.norm(points_along_rays[:, :, None, :] - detector_points[None, None, :, :], axis=-1)
#     min_distances = jnp.min(distances, axis=1)
#     closest_detector_indices = jnp.argmin(min_distances, axis=1)

#     closest_points = points_along_rays[jnp.arange(Nphot), jnp.argmin(distances, axis=1)[jnp.arange(Nphot), closest_detector_indices]]
    
#     # Calculate time for each photon (assuming speed of light = 1)
#     photon_times = jnp.linalg.norm(closest_points - ray_origins, axis=-1)

#     end_rest = time.time()
#     end_total = time.time()

#     # Calculate durations
#     duration_get_rays = end_get_rays - start_get_rays
#     duration_rest = end_rest - start_rest
#     duration_total = end_total - start_total

#     # Print timing information
#     print(f"Time for differentiable_get_rays: {duration_get_rays:.6f} seconds")
#     print(f"Time for rest of computation: {duration_rest:.6f} seconds")
#     print(f"Total time: {duration_total:.6f} seconds")
#     print("-------")

#     return closest_points, closest_detector_indices, photon_times
