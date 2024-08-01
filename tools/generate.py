
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


@partial(jax.jit, static_argnums=(3,))
def differentiable_get_rays(track_origin, track_direction, cone_opening, Nphot, key):
    # Generate ray vectors
    key, subkey = random.split(key)
    ray_vectors = generate_vectors_on_cone_surface_jax(track_direction, jnp.radians(cone_opening), Nphot, subkey)
    
    # Generate ray origins
    key, subkey = random.split(key)
    random_lengths = random.uniform(subkey, (Nphot, 1), minval=0, maxval=1)
    ray_origins = jnp.ones((Nphot, 3)) * track_origin + random_lengths * track_direction
    
    return ray_vectors, ray_origins


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


@partial(jax.jit, static_argnums=(5,))
def differentiable_photon_pmt_distance(cone_opening, track_origin, track_direction, detector_points, detector_radius, Nphot, key):
    ray_vectors, ray_origins = differentiable_get_rays(track_origin, track_direction, cone_opening, Nphot, key)
    
    # closest_points = ray_origins
    # closest_detector_indices = jnp.ones(Nphot, dtype=jnp.int32)
    # photon_times = jnp.linalg.norm(closest_points - ray_origins, axis=-1)

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