import jax
from functools import partial
from jax import random
import sys, os
import h5py
import numpy as np
import jax.numpy as jnp
import time
from tqdm import tqdm 
from tools.siren.core import *
from tools.utils import save_single_event_with_extended_info, get_random_root_entry_index, superimpose_multiple_events, merge_event_files
from jax import jit

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

@jax.jit
def denormalize_log_predictions(predictions, log_max, log_min):
    log_predictions = predictions * (log_max - log_min) + log_min 
    return 10 ** log_predictions - 1e-10

@jax.jit
def normalize_inputs_jit(inputs, energy_min, energy_max, angle_min, angle_max, distance_min, distance_max):
    """
    Normalize inputs to the range [-1, 1] in all dimensions.
    
    Args:
        inputs: Array of shape (..., 3) containing [energy, angle, distance] values.
        energy_min, energy_max: The minimum and maximum energy values.
        angle_min, angle_max: The minimum and maximum angle values.
        distance_min, distance_max: The minimum and maximum distance values.
    
    Returns:
        Array of shape (..., 3) with normalized values in the range [-1, 1].
    """
    # Extract the individual components
    energy = inputs[:, 0]
    angle = inputs[:, 1]
    distance = inputs[:, 2]
    
    # Normalize each component to [-1, 1]
    normalized_energy = 2.0 * (energy - energy_min) / (energy_max - energy_min) - 1.0
    normalized_angle = 2.0 * (angle - angle_min) / (angle_max - angle_min) - 1.0
    normalized_distance = 2.0 * (distance - distance_min) / (distance_max - distance_min) - 1.0
    
    # Stack the normalized components back together
    normalized_inputs = jnp.stack([normalized_energy, normalized_angle, normalized_distance], axis=1)
    
    return normalized_inputs

@partial(jax.jit, static_argnums=(3,))
def photonsim_differentiable_get_rays(track_origin, track_direction, energy, Nphot,
                                         table_data, model_params, key, num_seeds_a=0.035882, num_seeds_b=1.417106, num_seeds_c=2.75):
    """
    Generate photon rays using SIREN model for photon generation.

    Parameters
    ----------
    track_origin : jnp.ndarray
        3D position of the track origin
    track_direction : jnp.ndarray
        3D direction vector of the track
    energy : float
        Energy of the particle in MeV
    Nphot : int
        Number of photons to generate
    table_data : tuple
        Grid data for SIREN model evaluation
    model_params : dict
        SIREN model parameters
    key : jax.random.PRNGKey
        Random key for sampling
    num_seeds_a : float, optional
        Parameter 'a' for power law num_seeds calculation, default 0.035882
    num_seeds_b : float, optional
        Parameter 'b' for power law num_seeds calculation, default 1.417106
    num_seeds_c : float, optional
        Parameter 'c' for power law num_seeds calculation, default 2.75

    Returns
    -------
    ray_vectors : jnp.ndarray
        Array of photon direction vectors
    ray_origins : jnp.ndarray
        Array of photon origin positions
    photon_weights : jnp.ndarray
        Array of photon weights
    """
    key, subkey = random.split(key)

    n_bins, energy_min, energy_max, angle_min, angle_max, distance_min, distance_max, angle_bins, distance_bins, angle_dist_grid, angle_mesh, distance_mesh, log_min, log_max = table_data

    # ============================================================================
    # FIRST EVALUATION: Full grid to get photon weights for sampling
    # ============================================================================
    evaluation_grid = jnp.stack([
        jnp.full_like(angle_mesh, energy).ravel(),  # Energy (MeV)
        angle_mesh.ravel(),                         # Angle (radians)
        distance_mesh.ravel(),                      # Distance (mm)
    ], axis=1)

    normalized_grid = normalize_inputs_jit(evaluation_grid, energy_min, energy_max, angle_min, angle_max, distance_min, distance_max)

    # Initialize SIREN model
    model = SIREN(
        hidden_features=256,
        hidden_layers=3,
        out_features=1,
    )

    # FIRST MODEL CALL
    photon_weights, _ = model.apply(model_params, normalized_grid)

    # Sample points based on weights
    key, sampling_key = random.split(key)
    key, noise_key_angle = random.split(key)
    key, noise_key_dist = random.split(key)

    # Calculate number of seeds using power law: num_seeds = int32(a * energy^b + c)
    num_seeds = jnp.int32(num_seeds_a * jnp.power(energy, num_seeds_b) + num_seeds_c)

    seed_indices = random.randint(sampling_key, (Nphot,), 0, num_seeds)
    indices_by_weight = jnp.argsort(-photon_weights.squeeze())[seed_indices]

    angle_dist_mesh = jnp.array(angle_dist_grid)
    selected_angle_dist = angle_dist_mesh[indices_by_weight]

    # Split into separate angle and distance arrays
    sampled_angle = selected_angle_dist[:, 0]
    sampled_dist  = selected_angle_dist[:, 1]

    # ============================================================================
    # STRATIFIED SAMPLING: Better coverage than pure MC
    # ============================================================================
    bin_width_angle = (angle_max - angle_min) / n_bins
    bin_width_dist = (distance_max - distance_min) / n_bins
    
    # Create stratified samples: divide [0, 1] into Nphot strata
    # Then shuffle to avoid systematic bias
    key, subkey_angle = random.split(key)
    key, subkey_dist = random.split(key)
    key, subkey_jitter_angle = random.split(key)
    key, subkey_jitter_dist = random.split(key)
    
    # Permute stratum indices for random assignment
    strata_indices_angle = random.permutation(subkey_angle, Nphot)
    strata_indices_dist = random.permutation(subkey_dist, Nphot)
    
    # Sample within each stratum: (stratum_index + uniform[0,1]) / Nphot
    jitter_angle = random.uniform(subkey_jitter_angle, (Nphot,))
    jitter_dist = random.uniform(subkey_jitter_dist, (Nphot,))
    
    strata_angle = (strata_indices_angle + jitter_angle) / Nphot
    strata_dist = (strata_indices_dist + jitter_dist) / Nphot
    
    # Map from [0, 1] to [-bin_width/2, bin_width/2]
    stratified_angle = (strata_angle - 0.5) * bin_width_angle
    stratified_dist = (strata_dist - 0.5) * bin_width_dist

    smeared_angle = sampled_angle + stratified_angle
    smeared_dist = sampled_dist + stratified_dist

    # ============================================================================
    # SECOND EVALUATION: Smeared points to get final photon weights
    # ============================================================================
    new_evaluation_grid = jnp.stack([
        jnp.full_like(smeared_angle, energy),
        smeared_angle,
        smeared_dist,
    ], axis=1)

    new_normalized_grid = normalize_inputs_jit(new_evaluation_grid, energy_min, energy_max, angle_min, angle_max, distance_min, distance_max)
    
    # SECOND MODEL CALL
    new_photon_weights, _ = model.apply(model_params, new_normalized_grid)

    photon_thetas = smeared_angle

    # Generate ray vectors and origins
    subkey, subkey2 = random.split(subkey)
    ray_vectors = generate_random_cone_vectors(track_direction, photon_thetas, Nphot, subkey)

    # Convert ranges to meters and compute ray origins
    ranges = smeared_dist / 1000
    ray_origins = jnp.ones((Nphot, 3)) * track_origin[None, :] + ranges[:, None] * normalize(track_direction[None, :])

    # Apply boundary conditions
    new_photon_weights = jnp.squeeze(new_photon_weights)
    new_photon_weights = jnp.where(smeared_angle < angle_min, 0, new_photon_weights)
    new_photon_weights = jnp.where(smeared_angle > angle_max, 0, new_photon_weights)
    new_photon_weights = jnp.where(smeared_dist < distance_min, 0, new_photon_weights)
    new_photon_weights = jnp.where(smeared_dist > distance_max, 0, new_photon_weights)

    return ray_vectors, ray_origins, denormalize_log_predictions(new_photon_weights, log_max, log_min)


@partial(jax.jit, static_argnums=(2,))
def get_isotropic_rays(source_position, source_intensity, Nphot, key):
    """
    Generate photons isotropically from a point source using spherical coordinates.
    """
    # Split the random key
    key, key_phi, key_theta = random.split(key, 3)
    
    # Generate spherically isotropic directions
    phi = random.uniform(key_phi, (Nphot,)) * 2 * jnp.pi
    cos_theta = random.uniform(key_theta, (Nphot,)) * 2 - 1
    sin_theta = jnp.sqrt(1 - cos_theta**2)
    
    # Convert to Cartesian coordinates
    x = sin_theta * jnp.cos(phi)
    y = sin_theta * jnp.sin(phi)
    z = cos_theta
    
    # Stack into direction vectors
    ray_vectors_unnormalized = jnp.stack([x, y, z], axis=1)
    
    # Normalize using vmap (even though they should already be unit vectors)
    ray_vectors = jax.vmap(normalize)(ray_vectors_unnormalized)
    
    # All ray origins are at the source position
    ray_origins = jnp.tile(source_position, (Nphot, 1))
    
    # Uniform weights
    photon_weights = jnp.ones(Nphot) * (source_intensity / Nphot)
    
    return ray_vectors, ray_origins, photon_weights


def generate_random_direction(key):
    """
    Generate a random direction uniformly distributed on a unit sphere.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX
    
    Returns
    -------
    jnp.ndarray
        Normalized 3D vector representing a random direction
    """
    key, subkey = jax.random.split(key)
    # Generate random points on a sphere using the Marsaglia method
    while True:
        # Generate two random numbers between -1 and 1
        u1, u2 = jax.random.uniform(subkey, shape=(2,), minval=-1.0, maxval=1.0)
        s = u1**2 + u2**2
        # Reject if s is outside the unit circle
        if s < 1.0:
            break
        key, subkey = jax.random.split(key)
    
    # Convert to Cartesian coordinates
    x = 2 * u1 * jnp.sqrt(1 - s)
    y = 2 * u2 * jnp.sqrt(1 - s)
    z = 1 - 2 * s
    
    # Return normalized vector
    return normalize(jnp.array([x, y, z]))

def generate_random_vertex(key):
    """
    Generate a random vertex within the volume [-1,1]^3.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for JAX
    
    Returns
    -------
    jnp.ndarray
        3D point within the volume [-1,1]^3
    """
    return jax.random.uniform(key, shape=(3,), minval=-0.1, maxval=0.1)

@jit
def predict_t0(distance, energy, baseline_slope, baseline_intercept, 
                   A_slope, A_intercept, B_slope, B_intercept, offset):
    """
    JAX JIT-compatible version of predict_t0.
    Parameters are passed as individual arrays/scalars instead of nested dict.
    """
    # Baseline from 1000 MeV linear fit
    baseline = baseline_slope * distance + baseline_intercept
    
    # Calculate delta timing
    log10_A = A_slope * energy + A_intercept
    B = B_slope * energy + B_intercept
    delta = 10**log10_A * jnp.power(distance, B) + offset
    
    return baseline + delta

# Helper function to unpack your existing params dict
def predict_t0_wrapper(distance, energy, params):
    """Wrapper to use your existing params dict structure"""
    return predict_t0(
        distance, energy,
        params['baseline_1000MeV']['slope'],
        params['baseline_1000MeV']['intercept'],
        params['delta_parameterization']['A_slope'],
        params['delta_parameterization']['A_intercept'],
        params['delta_parameterization']['B_slope'],
        params['delta_parameterization']['B_intercept'],
        params['delta_parameterization']['offset']
    )

def generate_multi_folder_events(event_simulator, root_file_path, folder_names, events_per_folder, 
                               n_rings_list=None, pion_root_file_path=None,
                               max_sensors_per_cell=4, batch_size=100):
    """
    Generate events across multiple folders, each with sequentially numbered events.
    
    Parameters
    ----------
    root_file_path : str
        Path to the ROOT file for muons
    folder_names : list of str
        List of folder names to create and populate with events
    events_per_folder : int or list of int
        Number of events to generate per folder. Can be a single int for all folders
        or a list of ints matching the length of folder_names
    n_rings_list : list of int, optional
        Number of rings for each folder, by default None (1 ring for all folders)
    pion_root_file_path : str, optional
        Path to ROOT file for pions, required if n_rings > 1 in any folder, by default None
    max_sensors_per_cell : int, optional
        Maximum sensors per cell, by default 4
    batch_size : int, optional
        Number of events to accumulate before saving in parallel, by default 100
        
    Returns
    -------
    dict
        Dictionary mapping folder names to lists of saved file paths
    """
    import os
    
    # Validate and normalize inputs
    if isinstance(events_per_folder, int):
        events_per_folder = [events_per_folder] * len(folder_names)
    elif len(events_per_folder) != len(folder_names):
        raise ValueError("If events_per_folder is a list, it must match the length of folder_names")
    
    if n_rings_list is None:
        n_rings_list = [1] * len(folder_names)
    elif len(n_rings_list) != len(folder_names):
        raise ValueError("If n_rings_list is provided, it must match the length of folder_names")
    
    # Check if pion file is needed but not provided
    if any(n_rings > 1 for n_rings in n_rings_list) and pion_root_file_path is None:
        raise ValueError("pion_root_file_path is required when n_rings > 1 in any folder")
        
    # Create base directory if it doesn't exist
    base_dir = os.path.dirname(folder_names[0])
    if base_dir and not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    
    # Generate events for each folder
    results = {}
    for folder_idx, folder_name in enumerate(folder_names):
        n_events = events_per_folder[folder_idx]
        n_rings = n_rings_list[folder_idx]
        
        print(f"\n{'-'*80}")
        print(f"Processing folder {folder_idx+1}/{len(folder_names)}: {folder_name}")
        print(f"Generating {n_events} events with {n_rings} ring(s)")
        print(f"{'-'*80}\n")
        
        saved_files = generate_events_from_root(
            event_simulator=event_simulator,
            root_file_path=root_file_path,
            output_dir=folder_name,
            n_events=n_events,
            n_rings=n_rings,
            pion_root_file_path=pion_root_file_path,
            max_sensors_per_cell=max_sensors_per_cell,
            batch_size=batch_size
        )
        
        results[folder_name] = saved_files
        
    # Print summary
    print("\nGeneration Summary:")
    print("=" * 50)
    total_events = sum(len(files) for files in results.values())
    print(f"Total events generated: {total_events}")
    for folder_name, files in results.items():
        print(f"  - {folder_name}: {len(files)} events")
    
    return results


def read_photon_data_from_photonsim(root_file_path, entry_index):
    """
    Read photon data from a PhotonSim ROOT file for a specific entry.
    
    Parameters
    ----------
    root_file_path : str
        Path to the PhotonSim ROOT file
    entry_index : int
        Entry index to read from the file
        
    Returns
    -------
    dict
        Dictionary containing photon_origins, photon_directions, and energy
    """
    import uproot
    import numpy as np
    import jax.numpy as jnp
    
    # Open the ROOT file
    root_file = uproot.open(root_file_path)
    
    # Access the tree
    tree = root_file['OpticalPhotons']
    
    # Read the data for the specified entry
    tree_data = tree.arrays(['PrimaryEnergy', 'PhotonPosX', 'PhotonPosY', 'PhotonPosZ', 
                           'PhotonDirX', 'PhotonDirY', 'PhotonDirZ', 'PhotonTime'], 
                          entry_start=entry_index, entry_stop=entry_index+1, library='np')
    
    # Extract primary energy (already in MeV)
    energy = float(tree_data['PrimaryEnergy'][0])
    
    # Extract photon positions (convert mm to cm)
    photon_posx = tree_data['PhotonPosX'][0] / 10.0  # mm to cm
    photon_posy = tree_data['PhotonPosY'][0] / 10.0
    photon_posz = tree_data['PhotonPosZ'][0] / 10.0
    
    # Extract photon directions
    photon_dirx = tree_data['PhotonDirX'][0]
    photon_diry = tree_data['PhotonDirY'][0]
    photon_dirz = tree_data['PhotonDirZ'][0]
    
    # Stack the components to form position and direction arrays
    photon_positions = np.column_stack((photon_posx, photon_posy, photon_posz))
    photon_directions = np.column_stack((photon_dirx, photon_diry, photon_dirz))
    
    return {
        'photon_origins': jnp.array(photon_positions),     # Combined position vectors in cm
        'photon_directions': jnp.array(photon_directions), # Combined direction vectors
        'photon_times': jnp.array(tree_data['PhotonTime'][0]),
        'energy': energy  # Energy in MeV
    }

def generate_events_from_photonsim(event_simulator, particles_dict, sensor_params, output_dir=None,
                                  n_events=None, batch_size=100, master_seed=None,
                                  merge_output=True, merged_filename='merged_events.h5'):
    """
    Generate and save events from PhotonSim ROOT files for multiple particle types.
    Events are saved with sequential numbering: event_0.h5, event_1.h5, etc.
    Each event contains multiple particles sharing the same vertex but with independent track parameters.

    Parameters
    ----------
    event_simulator : function
        The event simulation function to use
    particles_dict : dict
        Dictionary mapping particle types to ROOT file paths
        Example: {'mu-': 'path/to/muon.root', 'pi-': 'path/to/pion.root'}
    sensor_params: tuple
        scattering length, reflection rate, absorption length and gumbel_softmax
    output_dir : str, optional
        Directory to save output files, by default 'events'
    n_events : int, optional
        Number of events to generate, by default None
    batch_size : int, optional
        Number of events to accumulate before saving in parallel, by default 100
    master_seed : int, optional
        Random seed for JAX PRNG key generation. If None, generates a random seed based on current time, by default None
    merge_output : bool, optional
        Whether to merge individual event files into a single HDF5 file, by default True
    merged_filename : str, optional
        Name of the merged output file (only used if merge_output=True), by default 'merged_events.h5'

    Returns
    -------
    list or str
        List of saved file paths, or path to merged file if merge_output=True
    """
    import uproot
    import concurrent.futures
    import time
    import numpy as np

    # Generate random seed based on time if not provided
    if master_seed is None:
        master_seed = int(time.time() * 1000000) % (2**32)
        print(f"Generated random master seed from time: {master_seed}")
    else:
        print(f"Using provided master seed: {master_seed}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open each ROOT file and get number of entries
    num_entries = {}
    particle_types = list(particles_dict.keys())

    print(f"Loading ROOT files for {len(particle_types)} particle types:")
    for particle_type, root_file_path in particles_dict.items():
        root_file = uproot.open(root_file_path)
        tree = root_file['OpticalPhotons']
        num_entries[particle_type] = tree.num_entries
        print(f"  - {particle_type}: {num_entries[particle_type]} entries in {root_file_path}")
        root_file.close()

    # Determine number of events to generate
    if n_events is None:
        # Use the minimum number of entries across all particle types
        n_events = min(num_entries.values())
        print(f"No n_events specified, using minimum entries: {n_events}")

    print(f"\nGenerating {n_events} events with {len(particle_types)} particles each...")
    print(f"Using batch size of {batch_size} events for multithreaded I/O")
    print(f"Saving events to directory: {output_dir}")

    saved_files = []
    
    # Create batches
    num_batches = (n_events + batch_size - 1) // batch_size
    
    # Process each batch
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_events)
        batch_size_actual = end_idx - start_idx
        
        print(f"Processing batch {batch_idx+1}/{num_batches} (events {start_idx} to {end_idx-1})")
        
        # Lists to accumulate batch data
        batch_data = []
        batch_track_params = []
        batch_sensor_params = []
        batch_filenames = []
        batch_indices = []

        # Process each entry in the current batch
        for event_idx in tqdm(range(start_idx, end_idx), desc=f"Generating batch {batch_idx+1}", unit="event"):
            # Initialize master random key for this event using the master seed
            master_key = jax.random.PRNGKey(master_seed + event_idx)

            # Generate a random vertex (shared by all particles in this event)
            vertex_key, master_key = jax.random.split(master_key)
            vertex = generate_random_vertex(vertex_key)

            # Lists to collect data for all particles in this event
            event_charges_list = []
            event_times_list = []
            event_particle_types = []
            event_energies = []
            event_directions = []
            event_original_indices = []

            # Process each particle type
            for particle_type in particle_types:
                root_file_path = particles_dict[particle_type]

                # Sample a random entry index for this particle
                sample_key, master_key = jax.random.split(master_key)
                entry_index = int(jax.random.randint(sample_key, (), 0, num_entries[particle_type]))

                # Read photon data from PhotonSim at the random index
                photon_data = read_photon_data_from_photonsim(root_file_path, entry_index)

                # Set up parameters
                particle_energy = photon_data['energy']

                # Generate random direction for this particle
                dir_key, master_key = jax.random.split(master_key)
                direction = generate_random_direction(dir_key)

                # Create parameters tuple
                track_params = (
                    particle_energy,
                    vertex,
                    direction
                )

                # Get a key for the simulation
                sim_key, master_key = jax.random.split(master_key)

                # Process photon data
                photon_origins = photon_data['photon_origins']
                photon_directions = photon_data['photon_directions']
                photon_times = photon_data['photon_times']
                N = len(photon_origins)

                # the number 1_000_000 is hard coded also in _simulation_core
                padding_size = max(0, 1_000_000-N)

                # Pad the origins array (2D array with shape [N,3])
                photon_data['photon_origins'] = jnp.pad(photon_origins, ((0, padding_size), (0, 0)),
                                                    mode='constant', constant_values=0)

                # Pad the directions array with a default unit vector [0,0,1]
                default_direction = jnp.array([0.0, 0.0, 1.0])
                padding_directions = jnp.tile(default_direction, (padding_size, 1))
                if padding_size > 0:
                    photon_data['photon_directions'] = jnp.concatenate([photon_directions, padding_directions], axis=0)
                else:
                    photon_data['photon_directions'] = photon_directions

                # Pad the times array (1D array with shape [N])
                photon_data['photon_times'] = jnp.pad(photon_times, (0, padding_size),
                                                      mode='constant', constant_values=0)

                photon_data['N'] = N

                # Run simulation for this particle
                charges, times = event_simulator(track_params, sensor_params, sim_key, photon_data)

                # Collect data for this particle
                event_charges_list.append(charges)
                event_times_list.append(times)
                event_particle_types.append(particle_type)
                event_energies.append(particle_energy)
                event_directions.append(direction.tolist())
                event_original_indices.append(entry_index)

            # Create filename with sequential numbering (event_0.h5, event_1.h5, etc.)
            event_number = event_idx
            filename = os.path.join(output_dir, f'event_{event_number}.h5')

            # Extended info with updated field names
            extended_info = {
                'n_particles': len(particle_types),
                'particle_types': event_particle_types,
                'energies': event_energies,
                'directions': event_directions,
                'vertex': vertex.tolist(),
                'original_indices': event_original_indices,
                'source': 'PhotonSim'
            }

            # Store the event data for batch processing
            # Note: track_params is just a placeholder since we have multiple particles
            dummy_track_params = (event_energies[0], vertex, jnp.array(event_directions[0]))
            batch_data.append((event_charges_list, event_times_list, extended_info))
            batch_track_params.append(dummy_track_params)
            batch_sensor_params.append(sensor_params)
            batch_filenames.append(filename)
            batch_indices.append(event_number)
        
        # Now save all the events in the batch using multithreading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create a list of future objects
            futures = [
                executor.submit(
                    save_single_event_with_extended_info,
                    data[0], data[1],  # lists of charges and times (one per particle)
                    t_params,  # dummy track params (not used in extended save function)
                    extended_info=data[2],  # extended info
                    event_number=idx,
                    filename=filename
                )
                for data, t_params, filename, idx in zip(
                    batch_data, batch_track_params, batch_filenames, batch_indices
                )
            ]
            
            # Collect results as they complete
            for future in tqdm(
                concurrent.futures.as_completed(futures), 
                desc=f"Saving batch {batch_idx+1}", 
                total=len(futures),
                unit="file"
            ):
                try:
                    saved_file = future.result()
                    saved_files.append(saved_file)
                except Exception as e:
                    print(f"Error saving file: {e}")
    
    print(f"\nSuccessfully processed {len(saved_files)} events.")
    print(f"Each event contains {len(particle_types)} particles: {', '.join(particle_types)}")
    print(f"All events saved to {output_dir} with sequential naming (event_0.h5, event_1.h5, ...)")

    # Merge files if requested
    if merge_output:
        print("\nMerging individual event files...")
        merged_path = merge_event_files(output_dir, merged_filename=merged_filename, remove_individuals=True)
        return merged_path

    return saved_files
