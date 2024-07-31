import jax
import jax.numpy as jnp
from jax import random
from functools import partial
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys, os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # or 'gpu' or 'tpu'
os.environ['JAX_ENABLE_X64'] = 'True'  # optional
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.generate import *

@partial(jax.jit, static_argnums=(5,))
def differentiable_photon_pmt_distance(cone_opening, track_origin, track_direction, detector_points, detector_radius, Nphot, key):
    def timed_operation(name, operation):
        start = time.perf_counter()
        with jax.profiler.trace(name):
            result = operation()
        end = time.perf_counter()
        print(f"{name} took {(end - start) * 1000:.3f} ms")
        return result

    ray_vectors, ray_origins = timed_operation(
        "differentiable_get_rays",
        lambda: differentiable_get_rays(track_origin, track_direction, cone_opening, Nphot, key)
    )
    
    ray_to_detector = timed_operation(
        "Calculate ray_to_detector",
        lambda: detector_points[None, :, :] - ray_origins[:, None, :]
    )
    
    dot_product = timed_operation(
        "Calculate dot_product",
        lambda: jnp.sum(ray_vectors[:, None, :] * ray_to_detector, axis=-1)
    )
    
    ray_mag_squared = timed_operation(
        "Calculate ray_mag_squared",
        lambda: jnp.sum(ray_vectors ** 2, axis=-1)[:, None]
    )
    
    t = timed_operation(
        "Calculate t",
        lambda: dot_product / ray_mag_squared
    )
    
    t = timed_operation(
        "Ensure t is non-negative",
        lambda: jnp.maximum(t, 0)
    )
    
    closest_points = timed_operation(
        "Calculate closest_points",
        lambda: ray_origins[:, None, :] + t[:, :, None] * ray_vectors[:, None, :]
    )
    
    distances = timed_operation(
        "Calculate distances",
        lambda: jnp.linalg.norm(closest_points - detector_points[None, :, :], axis=-1)
    )
    
    closest_detector_indices = timed_operation(
        "Find closest_detector_indices",
        lambda: jnp.argmin(distances, axis=1)
    )
    
    closest_points = timed_operation(
        "Get actual closest_points",
        lambda: closest_points[jnp.arange(Nphot), closest_detector_indices]
    )
    
    photon_times = timed_operation(
        "Calculate photon_times",
        lambda: jnp.linalg.norm(closest_points - ray_origins, axis=-1)
    )
    
    return closest_points, closest_detector_indices, photon_times

# Example usage:
key = random.PRNGKey(0)
cone_opening = 0.5
track_origin = jnp.array([0., 0., 0.])
track_direction = jnp.array([1., 0., 0.])
detector_points = jnp.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
detector_radius = 0.1
Nphot = 1000

# Compile the function
print("Compiling function...")
compiled_fn = jax.jit(differentiable_photon_pmt_distance, static_argnums=(5,))

# Warm-up run
print("Warm-up run...")
_ = compiled_fn(cone_opening, track_origin, track_direction, detector_points, detector_radius, Nphot, key)

# Actual timed run
print("Timed run...")
start = time.perf_counter()
with jax.profiler.trace("full_function"):
    result = compiled_fn(cone_opening, track_origin, track_direction, detector_points, detector_radius, Nphot, key)
end = time.perf_counter()

print(f"Total execution time: {(end - start) * 1000:.3f} ms")

# To save the trace
jax.profiler.save_device_memory_profile("memory_profile.prof")