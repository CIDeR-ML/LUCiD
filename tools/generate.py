import jax
from functools import partial
from jax import random
import sys, os
import h5py
import numpy as np
import jax.numpy as jnp
import time
from tools.siren import *
from tools.table import *

def normalize(v, epsilon=1e-8):
    """Normalize a vector with numerical stability.

    Parameters
    ----------
    v : jnp.ndarray
        Input vector to normalize
    epsilon : float, optional
        Small constant for numerical stability, by default 1e-8

    Returns
    -------
    jnp.ndarray
        Normalized vector
    """
    return v / (jnp.linalg.norm(v) + epsilon)


def generate_orthonormal_basis(v):
    """Generate an orthonormal basis with v as one of the vectors.

    Parameters
    ----------
    v : jnp.ndarray
        Input vector that will be the third basis vector

    Returns
    -------
    jnp.ndarray
        3x3 matrix where columns are orthonormal basis vectors
    """
    v = normalize(v)

    # Find a vector not parallel to v by trying [1,0,0] or [0,1,0]
    not_v = jnp.array([1.0, 0.0, 0.0])
    cond = jnp.abs(jnp.dot(v, not_v)) > 0.9
    not_v = jnp.where(cond, jnp.array([0.0, 1.0, 0.0]), not_v)

    # Use cross product to find two vectors orthogonal to v
    u = normalize(jnp.cross(v, not_v))
    w = jnp.cross(v, u)

    return jnp.stack([u, w, v], axis=-1)


@partial(jax.jit, static_argnums=(2,))
def generate_random_cone_vectors(R, theta, num_vectors, key):
    """Generate random vectors uniformly distributed on a cone surface.

    Parameters
    ----------
    R : jnp.ndarray
        Direction vector of the cone axis
    theta : float
        Opening angle of the cone in radians
    num_vectors : int
        Number of random vectors to generate
    key : jax.random.PRNGKey
        Random number generator key

    Returns
    -------
    jnp.ndarray
        Array of shape (num_vectors, 3) containing random unit vectors on cone surface
    """
    R = normalize(R)
    theta = jnp.clip(theta, 1e-6, jnp.pi - 1e-6)

    key1, key2 = random.split(key)
    phi = random.uniform(key1, (num_vectors,), minval=0, maxval=2 * jnp.pi)

    # Convert from polar to cartesian coordinates on cone surface
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    x = jnp.cos(phi) * sin_theta
    y = jnp.sin(phi) * sin_theta
    z = cos_theta * jnp.ones_like(x)

    basis = generate_orthonormal_basis(R)
    vectors = jnp.column_stack((x, y, z))
    rotated_vectors = jnp.einsum('ij,kj->ki', basis, vectors)

    return rotated_vectors


@partial(jax.jit, static_argnums=(3,))
def differentiable_get_rays(track_origin, track_direction, cone_opening, Nphot, key):
    """Generate ray origins and directions for a cone-shaped beam of rays.

    Parameters
    ----------
    track_origin : jnp.ndarray
        Starting point of the track
    track_direction : jnp.ndarray
        Direction vector of the track
    cone_opening : float
        Opening angle of the cone in degrees
    Nphot : int
        Number of rays to generate
    key : jax.random.PRNGKey
        Random number generator key

    Returns
    -------
    tuple
        (ray_vectors, ray_origins) where each is an array of shape (Nphot, 3)
    """
    key, subkey = random.split(key)
    ray_vectors = generate_random_cone_vectors(track_direction, jnp.radians(cone_opening), Nphot, key=subkey)

    key, subkey = random.split(key)

    # track length
    length = 1.0

    # # Generate uniformly distributed lengths
    random_lengths = random.uniform(subkey, (Nphot, 1), minval=-0.5, maxval=0.5) * length

    # Generate normally distributed lengths, clipped to [-1, 1]
    # random_lengths = random.normal(subkey, (Nphot, 1)) * length
    random_lengths = jnp.clip(random_lengths, -1.0, 1.0)

    ray_origins = jnp.ones((Nphot, 3)) * track_origin + random_lengths * track_direction

    return ray_vectors, ray_origins


@jax.jit
def jax_linear_interp(x_data, y_data, x):
    """
    Simple linear interpolation implementation using JAX
    """
    # Find the index of the closest point below x for each point
    idx = jnp.searchsorted(x_data, x, side='right') - 1
    # Clip index to valid range
    idx = jnp.clip(idx, 0, len(x_data) - 2)

    # Get x and y values for interpolation
    x0 = x_data[idx]
    x1 = x_data[idx + 1]
    y0 = y_data[idx]
    y1 = y_data[idx + 1]

    # Linear interpolation
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x - x0)


@partial(jax.jit, static_argnums=(3))
def new_differentiable_get_rays(track_origin, track_direction, energy, Nphot, table_data, model_params, key):
    key, subkey = random.split(key)
    cos_bins, trk_bins, cos_trk_mesh, (x_data, y_data), grid_shape = table_data

    # Create the evaluation grid using JAX operations - using efficient meshgrid
    energy_interp = jax_linear_interp(x_data, y_data, energy)
    cos_mesh, trk_mesh = jnp.meshgrid(cos_bins, trk_bins, indexing='ij')  # Note the indexing='ij'
    evaluation_grid = jnp.stack([
        jnp.full_like(cos_mesh, energy_interp).ravel(),
        cos_mesh.ravel(),  # Maintain correct order: cos first
        trk_mesh.ravel(),  # trk second
    ], axis=1)

    # Initialize SIREN model
    model = SIREN(
        hidden_features=256,
        hidden_layers=3,
        out_features=1,
        outermost_linear=True
    )

    # Apply SIREN model
    photon_weights, _ = model.apply(model_params, evaluation_grid)

    # After getting selected_cos and selected_trk:
    key, sampling_key = random.split(key)
    key, noise_key_cos = random.split(key)
    key, noise_key_trk = random.split(key)

    # this calculate the number of good seeds based on linear
    # interpolation given the trained SIREN model under evaluation
    # with the specific grid settings in create_siren_grid
    # if you change create_siren_grid, change also the numbers below.
    # you can use code in siren/cut_off_study notebook.
    # num_seeds = jnp.int32(energy*94.14714286-17750.)          # using 500 x 500 binning and 0.008  cut-off value
    # num_seeds = jnp.int32(energy*84.77857143-15115.71428571)  # using 500 x 500 binning and 0.01  cut-off value
    # num_seeds = jnp.int32(energy*75.0525-12907.35714286)      # using 500 x 500 binning and 0.015 cut-off value
    num_seeds = jnp.int32(energy * 69.77142857 - 11980.42857143)  # using 500 x 500 binning and 0.02  cut-off value

    seed_indices = random.randint(sampling_key, (Nphot,), 0, num_seeds)
    indices_by_weight = jnp.argsort(-photon_weights.squeeze())[seed_indices]

    cos_trk_mesh = jnp.array(cos_trk_mesh)
    selected_cos_trk = cos_trk_mesh[indices_by_weight]

    # Split into separate cos and trk arrays
    sampled_cos = selected_cos_trk[:, 0]
    sampled_trk = selected_cos_trk[:, 1]

    # Add Gaussian noise
    sigma_cos = 0.001
    sigma_trk = 0.001

    noise_cos = random.normal(noise_key_cos, (Nphot,)) * sigma_cos
    noise_trk = random.normal(noise_key_trk, (Nphot,)) * sigma_trk

    smeared_cos = sampled_cos + noise_cos
    smeared_trk = sampled_trk + noise_trk

    # kill those events sampled outside of the normalized SIREN binning
    # the sigma and binning are chosen such that the fraction of those events is negligibly small.
    smeared_trk = jnp.where(smeared_trk < -1, 0, smeared_trk)
    smeared_trk = jnp.where(smeared_trk > 1, 0, smeared_trk)

    smeared_cos = jnp.where(smeared_cos < -1, 0, smeared_cos)
    smeared_cos = jnp.where(smeared_cos > 1, 0, smeared_cos)

    # Create new evaluation grid with smeared values
    energy_interp = jax_linear_interp(x_data, y_data, energy)
    new_evaluation_grid = jnp.stack([
        jnp.full_like(smeared_cos, energy_interp),
        smeared_cos,
        smeared_trk,
    ], axis=1)

    # Run the model with new grid
    new_photon_weights, _ = model.apply(model_params, new_evaluation_grid)

    photon_thetas = jnp.arccos(smeared_cos)

    # Generate ray vectors and origins
    subkey, subkey2 = random.split(subkey)
    ray_vectors = generate_random_cone_vectors(track_direction, photon_thetas, Nphot, subkey)

    # Convert ranges to meters and compute ray origins
    ranges = (smeared_trk * 300 + 300) / 100
    ray_origins = jnp.ones((Nphot, 3)) * track_origin[None, :] + ranges[:, None] * normalize(track_direction[None, :])

    return ray_vectors, ray_origins, jnp.squeeze(new_photon_weights)




