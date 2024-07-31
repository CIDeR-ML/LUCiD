import time
import pandas as pd
import h5py
import json
import numpy as np
import argparse

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.geometry import *

import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax import random
from jax.lax import scan, while_loop

from functools import partial

import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="unhashable type: .*. Attempting to hash a tracer will lead to an error in a future JAX release.")

Nphot = 200

def relative_angle(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude_vector1 = np.linalg.norm(vector1)
    magnitude_vector2 = np.linalg.norm(vector2)
    cosine_angle = dot_product / (magnitude_vector1 * magnitude_vector2)

    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


class Logger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.origins         = []
        self.directions      = []
        self.losses          = []
        self.dir_err         = []
        self.ori_err         = []
        self.ch_angles       = []
        self.ch_angles_err   = []

    def add_data(self, origin, direction, true_dir, true_ori, ch_angle, true_ch_angle, loss):
        self.origins.append(origin)
        self.directions.append(direction)
        self.losses.append(float(loss))
        self.dir_err.append(relative_angle(true_dir, direction))
        self.ori_err.append(np.linalg.norm(true_ori-origin))
        self.ch_angles.append(ch_angle)
        self.ch_angles_err.append(true_ch_angle-ch_angle)

    def plot_angle_err(self):
        plt.plot(range(len(self.dir_err[:])), self.dir_err[:], label='Direction angle error', color='darkorange')
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Angle Error (degrees)')
        plt.legend(frameon=False, loc='best')
        plt.ylim(bottom=0.)

    def plot_distance_err(self):
        plt.plot(range(len(self.ori_err[:])), self.ori_err[:], label='Origin distance error', color='cornflowerblue')
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Distance Error (meters)')
        plt.legend(frameon=False, loc='best')
        plt.ylim(bottom=0.)

    def plot_ch_angle(self):
        self.expected_cone_opening = 40
        plt.plot(range(len(self.ch_angles[:])), self.ch_angles[:], color='hotpink')
        plt.axhline(self.expected_cone_opening, color='darkgray', linestyle='--', label='expected')
        plt.xlim(1, len(self.losses))
        plt.ylim(bottom=min(self.expected_cone_opening, min(self.ch_angles[:])) / 1.43, top=max(self.expected_cone_opening, max(self.ch_angles[:])) * 1.3)
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Cone Opening')
        plt.legend(frameon=False, loc='best')

    def plot_loss(self):
        plt.plot(range(len(self.losses[:])), self.losses[:], color='k')
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Loss')
        plt.xlim(0, len(self.losses))
        plt.yscale('log')

    def plot_all(self):
        plt.figure(figsize=(12, 9))

        plt.subplot(2, 2, 1)
        self.plot_angle_err()

        plt.subplot(2, 2, 2)
        self.plot_distance_err()

        plt.subplot(2, 2, 3)
        self.plot_ch_angle()

        plt.subplot(2, 2, 4)
        self.plot_loss()

        plt.tight_layout()
        plt.savefig('optimization_results.pdf')
        plt.show()


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [jnp.zeros_like(param) for param in params]
            self.v = [jnp.zeros_like(param) for param in params]

        self.t += 1
        new_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            new_param = param - self.learning_rate * m_hat / (jnp.sqrt(v_hat) + self.epsilon)
            new_params.append(new_param)
        return new_params

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

def load_data(filename):
    with h5py.File(filename, 'r') as f:
        hit_pmt = np.array(f['hit_pmt'])
        hit_charge = np.array(f['hit_charge'])
        hit_time = np.array(f['hit_time'])
        true_cone_opening = np.array(f['true_cone_opening'])[0]
        true_track_origin = np.array(f['true_track_origin'])
        true_track_direction = np.array(f['true_track_direction'])

    return hit_pmt, hit_charge, hit_time, true_cone_opening, true_track_origin, true_track_direction

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

        print("Shapes:")
        print(f"ray_origins_np: {ray_origins_np.shape}")
        print(f"sensor_indices: {sensor_indices.shape}")
        print(f"hit_flag: {hit_flag.shape}")
        print(f"closest_points: {closest_points.shape}")

        print("Max sensor index:", np.max(sensor_indices))
        print("Number of hits:", np.sum(hit_flag))

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

        print(f"Number of valid hits: {Nhits}")

        # Add event_hits_index dataset
        f_outfile.create_dataset("event_hits_index", data=np.array([Nhits], dtype=np.int64))

        h5_evt_hit_IDs = f_outfile.create_dataset("hit_pmt", shape=(Nhits,), dtype=np.int32)
        h5_evt_hit_Qs = f_outfile.create_dataset("hit_charge", shape=(Nhits,), dtype=np.float32)
        h5_evt_hit_Ts = f_outfile.create_dataset("hit_time", shape=(Nhits,), dtype=np.float32)

        h5_evt_hit_IDs[:] = idx
        h5_evt_hit_Qs[:] = cts
        h5_evt_hit_Ts[:] = [np.mean(photon_times[valid_sensor_indices == i]) for i in idx]

        print(h5_evt_hit_Ts[:])

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
    
    return 3*avg_time_diff+avg_min_dist#+avg_time_diff

def run_tests(detector, true_indices, true_times, detector_points, detector_radius, Nphot, true_params, use_time_loss):
    def test_parameter(param_name, true_params, param_range, param_index=None):
        results = []
        loss_and_grad = jax.value_and_grad(combined_loss_function, argnums=(2, 3, 4))
        
        for i, param_value in enumerate(param_range):
            print(i)
            if param_name == 'cone_opening':
                params = [param_value, true_params[1], true_params[2]]
            elif param_name.startswith('track_origin'):
                params = list(true_params)
                if isinstance(params[1], jnp.ndarray):
                    params[1] = params[1].at[param_index].set(param_value)
                else:  # NumPy array
                    params[1] = params[1].copy()
                    params[1][param_index] = param_value
            elif param_name.startswith('track_direction'):
                params = list(true_params)
                if isinstance(params[2], jnp.ndarray):
                    params[2] = params[2].at[param_index].set(param_value)
                else:  # NumPy array
                    params[2] = params[2].copy()
                    params[2][param_index] = param_value
                params[2] = normalize(params[2])
            
            key = random.PRNGKey(0)
            loss, (grad_cone, grad_origin, grad_direction) = loss_and_grad(
                true_indices, true_times, *params, detector_points, detector_radius, Nphot, key, use_time_loss
            )

            if param_name == 'cone_opening':
                grad = grad_cone
            elif param_name.startswith('track_origin'):
                grad = grad_origin[param_index]
            elif param_name.startswith('track_direction'):
                grad = grad_direction[param_index]
            
            # Generate and store event
            event_filename = f'test_events/{param_name}_step_{i}.h5'
            generate_and_store_event(event_filename, *params, detector, Nphot, key)
            
            results.append({
                'param_value': param_value,
                'loss': loss,
                'grad': grad,
                'event_filename': event_filename
            })
        
        return results

    Nsteps = 11

    # Test 1: Cone opening angle
    cone_results = test_parameter('cone_opening', true_params, jnp.linspace(20, 60, 11))

    # Test 2: X component of track origin
    origin_x_results = test_parameter('track_origin_x', true_params, jnp.linspace(0, 2, Nsteps), 0)

    # Test 3: Y component of track direction
    direction_y_results = test_parameter('track_direction_y', true_params, jnp.linspace(-0.4, 0.4, Nsteps), 1)

    return cone_results, origin_x_results, direction_y_results

def main():
    # Set default values
    default_json_filename = 'cyl_geom_config.json'
    output_filename = 'autodiff_datasets/data_events.h5'
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--is_data', type=bool, default=False, help='This creates the data event.')
    parser.add_argument('--json_filename', type=str, default=default_json_filename, help='The JSON filename')
    parser.add_argument('--test', action='store_true', help='Run tests instead of optimization')
    parser.add_argument('--use_time_loss', action='store_true', help='Use time-based loss function')

    args = parser.parse_args()
    
    json_filename = args.json_filename
    use_time_loss = args.use_time_loss

    if args.is_data:
        print('Using data mode')
        # Use specific parameters for data generation
        true_cone_opening = 40.
        true_track_origin = np.array([1., 0., 0.])
        true_track_direction = np.array([1., 0., 0.])
        #generate_data(json_filename, output_filename, true_cone_opening, true_track_origin, true_track_direction)

        detector = generate_detector(json_filename)
        key = random.PRNGKey(0)
        generate_and_store_event(output_filename, true_cone_opening, true_track_origin, true_track_direction, detector, Nphot, key)

    elif args.test:
        print('Running tests')
        detector = generate_detector(json_filename)
        true_indices, _, true_times, true_cone_opening, true_track_origin, true_track_direction = load_data(output_filename)
        
        detector_points = jnp.array(detector.all_points)
        detector_radius = detector.S_radius
        
        true_params = [true_cone_opening, true_track_origin, true_track_direction]

        cone_results, origin_x_results, direction_y_results = run_tests(
            detector, true_indices, true_times, detector_points, detector_radius, Nphot, true_params, args.use_time_loss
        )

        # Plot and analyze results
        import matplotlib.pyplot as plt
        
        def plot_results(figname, results, title, xlabel, true_value):
            param_values = [r['param_value'] for r in results]
            losses = [r['loss'] for r in results]
            grads = [r['grad'] for r in results]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
            
            # Plot loss
            ax1.plot(param_values, losses)
            ax1.set_ylabel('Loss')
            ax1.set_title(title)
            ax1.axvline(x=true_value, color='r', linestyle='--', label='True Value')
            ax1.legend()
            
            # Plot gradient
            ax2.plot(param_values, grads)
            ax2.set_ylabel('Gradient')
            ax2.set_xlabel(xlabel)
            ax2.axvline(x=true_value, color='r', linestyle='--')
            
            plt.tight_layout()
            plt.savefig(figname)
            plt.close(fig)  # Close the figure to free up memory

        # In the main function, modify the calls to plot_results:
        plot_results('test1.pdf', cone_results, 'Cone Opening Angle Test', 'Cone Opening Angle (degrees)', true_cone_opening)
        plot_results('test2.pdf', origin_x_results, 'Track Origin X Test', 'Track Origin X', true_track_origin[0])
        plot_results('test3.pdf', direction_y_results, 'Track Direction Y Test', 'Track Direction Y', true_track_direction[1])

    else:
        print('Inference mode')
        detector = generate_detector(json_filename)
        true_indices, _, true_times, true_cone_opening, true_track_origin, true_track_direction = load_data(output_filename)
        
        log = Logger()

        # Start with random parameters for inference
        cone_opening = np.random.uniform(48., 52)
        track_origin = np.random.uniform(0., 0., size=3)
        track_direction = normalize(np.random.uniform(-1., 1., size=3))


        # cone_opening = np.random.uniform(45., 45)
        # track_origin = np.array([1.,0.,0.]) 
        # track_direction = normalize(np.array([0.,1.,1.]))

        # cone_opening = np.random.uniform(40., 40)
        # track_origin = np.array([1.,0.,0.]) 
        # track_direction = normalize(np.array([1.,0.,0.]))

        key = random.PRNGKey(0)

        filename = 'test_events/optimization_start.h5'
        generate_and_store_event(filename, cone_opening, track_origin, track_direction, detector, Nphot, key)


        detector_points = jnp.array(detector.all_points)
        detector_radius = detector.S_radius

        loss_and_grad = jax.value_and_grad(combined_loss_function, argnums=(2, 3, 4))

        # Optimization parameters
        num_iterations = 100
        patience = 50  # number of iterations to wait before early stopping
        min_delta = 1e-6  # minimum change in loss to qualify as an improvement
        
        # Initialize Adam optimizer
        adam = Adam(learning_rate=0.01)
        
        best_loss = float('inf')
        best_params = None
        patience_counter = 0
        
        # Optimization loop
        for i in range(num_iterations):
            loss, (grad_cone, grad_origin, grad_direction) = loss_and_grad(
                true_indices, true_times, cone_opening, track_origin, track_direction, 
                detector_points, detector_radius, Nphot, key, args.use_time_loss
            )

            Scale = 1
            # if loss < 2:
            #     Scale = 1

            cone_opening -= Scale*10*grad_cone
            track_origin -= Scale*0.05*grad_origin
            track_direction -= Scale*0.02*grad_direction


            #track_origin -= Scale*2*grad_origin

            # if loss<1.5:
            #     cone_opening -= 10*grad_cone
            #     track_origin -= 0.1*grad_origin
            #     track_direction -= 0.03*grad_direction
            # else:
            #     cone_opening -= 10*grad_cone
            #     track_direction -= 0.03*grad_direction

            #print(grad_origin,grad_direction,grad_cone)



            # track_origin -= 0.1*grad_origin

            log.add_data(track_origin, track_direction, true_track_direction, true_track_origin, cone_opening, true_cone_opening, loss)

            #print(grad_direction)


            # # Update parameters using Adam
            # cone_opening, track_origin, track_direction = adam.update(
            #     [cone_opening, track_origin, track_direction],
            #     [grad_cone, grad_origin, grad_direction]
            # )
            
            # Normalize track_direction
            track_direction = normalize(track_direction)
            
            # # Check for improvement
            # if loss < best_loss - min_delta:
            #     best_loss = loss
            #     best_params = (cone_opening, track_origin, track_direction)
            #     patience_counter = 0
            # else:
            #     patience_counter += 1
            
            # # Early stopping
            # if patience_counter >= patience:
            #     print(f"Early stopping at iteration {i}")
            #     break
            
            # Print progress every 10 iterations
            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss}")
                print(f"Cone opening: {cone_opening}")
                print(f"Track origin: {track_origin}")
                print(f"Track direction: {track_direction}")
                print()

        # # Use the best parameters found
        # cone_opening, track_origin, track_direction = best_params

        print("\nOptimization complete.")
        print("Final parameters:")
        print(f"Cone opening: {cone_opening}")
        print(f"Track origin: {track_origin}")
        print(f"Track direction: {track_direction}")

        print("\nTrue parameters:")
        print(f"Cone opening: {true_cone_opening}")
        print(f"Track origin: {true_track_origin}")
        print(f"Track direction: {true_track_direction}")

        print(f"\nFinal Loss: {best_loss}")

        filename = 'test_events/optimization_result.h5'
        generate_and_store_event(filename, cone_opening, track_origin, track_direction, detector, Nphot, key)

        log.plot_all()

if __name__ == "__main__":
    stime = time.perf_counter()
    main()
    print('Total exec. time: ', f"{time.perf_counter()-stime:.2f} s.")
