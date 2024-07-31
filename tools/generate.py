
import jax
from functools import partial
from jax import random
import sys, os
import h5py
import numpy as np
import jax.numpy as jnp


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


def generate_and_store_event(filename, cone_opening, track_origin, track_direction, detector, Nphot, key):
    N_photosensors = len(detector.all_points)

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create the output h5 file
    with h5py.File(filename, 'w') as f_outfile:
        # Add evt_id, positions, and event_hits_index datasets
        f_outfile.create_dataset("evt_id", data=np.array([0], dtype=np.int32))
        f_outfile.create_dataset("positions", data=np.array([track_origin], dtype=np.float32).reshape(1, 1, 3))
        
        f_outfile.create_dataset("true_cone_opening", data=np.array([cone_opening]))
        f_outfile.create_dataset("true_track_origin", data=track_origin)
        f_outfile.create_dataset("true_track_direction", data=track_direction)


        ray_vectors, ray_origins = differentiable_get_rays(track_origin, track_direction, cone_opening, Nphot, key)# get_rays(track_origin, track_direction, cone_opening, Nphot)

        # Convert JAX arrays to NumPy arrays
        ray_origins_np = np.array(ray_origins)
        ray_vectors_np = np.array(ray_vectors)
        detector_points_np = np.array(detector.all_points)

        sensor_indices, hit_flag, closest_points = check_hits_vectorized_per_track_jax(
            ray_origins_np, ray_vectors_np, detector.S_radius, detector_points_np)

        # Convert JAX arrays to NumPy arrays
        sensor_indices = np.array(sensor_indices)
        hit_flag = np.array(hit_flag)
        closest_points = np.array(closest_points)

        # Filter out invalid indices
        valid_mask = sensor_indices < len(detector_points_np)
        valid_sensor_indices = sensor_indices[valid_mask]

        # Get the correct closest points
        valid_closest_points = closest_points[np.arange(len(hit_flag))[hit_flag], valid_sensor_indices]

        # Calculate photon times for valid hits
        valid_ray_origins = ray_origins_np[np.arange(len(hit_flag))[hit_flag]]
        photon_times = np.linalg.norm(valid_closest_points - valid_ray_origins, axis=1)

        idx, cts = np.unique(valid_sensor_indices, return_counts=True)
        Nhits = len(idx)

        # Add event_hits_index dataset
        f_outfile.create_dataset("event_hits_index", data=np.array([Nhits], dtype=np.int64))

        h5_evt_hit_IDs = f_outfile.create_dataset("hit_pmt", shape=(Nhits,), dtype=np.int32)
        h5_evt_hit_Qs = f_outfile.create_dataset("hit_charge", shape=(Nhits,), dtype=np.float32)
        h5_evt_hit_Ts = f_outfile.create_dataset("hit_time", shape=(Nhits,), dtype=np.float32)

        h5_evt_hit_IDs[:] = idx
        h5_evt_hit_Qs[:] = cts
        h5_evt_hit_Ts[:] = [np.mean(photon_times[valid_sensor_indices == i]) for i in idx]

    return filename

@partial(jax.jit, static_argnums=(5,6))
def differentiable_toy_mc_simulator(cone_opening, track_origin, track_direction, detector_points, detector_radius, Nphot, key):
    ray_vectors, ray_origins = differentiable_get_rays(track_origin, track_direction, cone_opening, Nphot, key)

    t = jnp.linspace(0, 10, 100)[:, None]
    points_along_rays = ray_origins[:, None, :] + t * ray_vectors[:, None, :]

    distances = jnp.linalg.norm(points_along_rays[:, :, None, :] - detector_points[None, None, :, :], axis=-1)
    min_distances = jnp.min(distances, axis=1)
    closest_detector_indices = jnp.argmin(min_distances, axis=1)

    closest_points = points_along_rays[jnp.arange(Nphot), jnp.argmin(distances, axis=1)[jnp.arange(Nphot), closest_detector_indices]]
    
    # Calculate time for each photon (assuming speed of light = 1)
    photon_times = jnp.linalg.norm(closest_points - ray_origins, axis=-1)

    return closest_points, closest_detector_indices, photon_times
