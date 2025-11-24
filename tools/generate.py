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
from tools.utils import save_single_event_with_extended_info, save_single_event_with_label_info, get_random_root_entry_index, superimpose_multiple_events, merge_event_files
from jax import jit

def jax_rotate_vector_local(vector, axis, angle):
    """
    Rotate a vector around an axis by an angle using Rodrigues' rotation formula.
    Local copy to avoid circular import.
    """
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    dot = jnp.dot(axis, vector)
    cross = jnp.cross(axis, vector)
    return vector * cos_angle + cross * sin_angle + axis * dot * (1.0 - cos_angle)

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

def read_label_data_from_photonsim(root_file_path, entry_index):
    """
    Read label-based photon data from a PhotonSim ROOT file for a specific entry.
    This function reads the new label system that classifies photons by genealogy.

    Parameters
    ----------
    root_file_path : str
        Path to the PhotonSim ROOT file
    entry_index : int
        Entry index to read from the file

    Returns
    -------
    dict
        Dictionary containing:
        - 'n_labels': int, number of unique labels
        - 'labels': list of dicts, each containing:
            - 'genealogy': list of track IDs in the genealogy
            - 'photon_indices': list of photon indices belonging to this label
            - 'track_info': dict with track information (position, direction, energy, time, pdg, category, etc.)
        - 'photon_origins': array (N_photons, 3) in cm
        - 'photon_directions': array (N_photons, 3)
        - 'photon_times': array (N_photons,) in ns
        - 'primary_energy': float, energy in MeV
    """
    import uproot
    import numpy as np
    import jax.numpy as jnp

    # Open the ROOT file
    root_file = uproot.open(root_file_path)
    tree = root_file['OpticalPhotons']

    # Read all necessary data for the specified entry
    branches_to_read = [
        'PrimaryEnergy',
        'PhotonPosX', 'PhotonPosY', 'PhotonPosZ',
        'PhotonDirX', 'PhotonDirY', 'PhotonDirZ',
        'PhotonTime',
        'NLabels',
        'Label_GenealogySize', 'Label_GenealogyData',
        'Label_PhotonIDsSize', 'Label_PhotonIDsData',
        'TrackInfo_TrackID', 'TrackInfo_Category', 'TrackInfo_SubID',
        'TrackInfo_PosX', 'TrackInfo_PosY', 'TrackInfo_PosZ',
        'TrackInfo_DirX', 'TrackInfo_DirY', 'TrackInfo_DirZ',
        'TrackInfo_Energy', 'TrackInfo_Time',
        'TrackInfo_ParentTrackID', 'TrackInfo_PDG'
    ]

    tree_data = tree.arrays(branches_to_read, entry_start=entry_index, entry_stop=entry_index+1, library='np')

    # Extract primary energy
    primary_energy = float(tree_data['PrimaryEnergy'][0])

    # Extract photon data (convert mm to cm)
    photon_posx = tree_data['PhotonPosX'][0] / 10.0
    photon_posy = tree_data['PhotonPosY'][0] / 10.0
    photon_posz = tree_data['PhotonPosZ'][0] / 10.0
    photon_positions = np.column_stack((photon_posx, photon_posy, photon_posz))

    photon_dirx = tree_data['PhotonDirX'][0]
    photon_diry = tree_data['PhotonDirY'][0]
    photon_dirz = tree_data['PhotonDirZ'][0]
    photon_directions = np.column_stack((photon_dirx, photon_diry, photon_dirz))

    photon_times = tree_data['PhotonTime'][0]

    # Extract label system
    n_labels = int(tree_data['NLabels'][0])

    # Parse genealogy data
    genealogy_sizes = tree_data['Label_GenealogySize'][0]
    genealogy_data = tree_data['Label_GenealogyData'][0]

    # Parse photon IDs data
    photon_ids_sizes = tree_data['Label_PhotonIDsSize'][0]
    photon_ids_data = tree_data['Label_PhotonIDsData'][0]

    # Extract track info arrays
    track_ids = tree_data['TrackInfo_TrackID'][0]
    track_categories = tree_data['TrackInfo_Category'][0]
    track_subids = tree_data['TrackInfo_SubID'][0]
    track_posx = tree_data['TrackInfo_PosX'][0] / 10.0  # mm to cm
    track_posy = tree_data['TrackInfo_PosY'][0] / 10.0
    track_posz = tree_data['TrackInfo_PosZ'][0] / 10.0
    track_dirx = tree_data['TrackInfo_DirX'][0]
    track_diry = tree_data['TrackInfo_DirY'][0]
    track_dirz = tree_data['TrackInfo_DirZ'][0]
    track_energies = tree_data['TrackInfo_Energy'][0]
    track_times = tree_data['TrackInfo_Time'][0]
    track_parent_ids = tree_data['TrackInfo_ParentTrackID'][0]
    track_pdgs = tree_data['TrackInfo_PDG'][0]

    # Build track info dictionary for quick lookup
    track_info_dict = {}
    for i in range(len(track_ids)):
        track_info_dict[int(track_ids[i])] = {
            'track_id': int(track_ids[i]),
            'category': int(track_categories[i]),
            'sub_id': int(track_subids[i]),
            'position': np.array([track_posx[i], track_posy[i], track_posz[i]]),
            'direction': np.array([track_dirx[i], track_diry[i], track_dirz[i]]),
            'energy': float(track_energies[i]),
            'time': float(track_times[i]),
            'parent_id': int(track_parent_ids[i]),
            'pdg': int(track_pdgs[i])
        }

    # Parse labels
    labels = []
    genealogy_offset = 0
    photon_ids_offset = 0

    for label_idx in range(n_labels):
        # Extract genealogy for this label
        gen_size = int(genealogy_sizes[label_idx])
        genealogy = [int(genealogy_data[genealogy_offset + i]) for i in range(gen_size)]
        genealogy_offset += gen_size

        # Extract photon indices for this label
        photon_ids_size = int(photon_ids_sizes[label_idx])
        photon_indices = [int(photon_ids_data[photon_ids_offset + i]) for i in range(photon_ids_size)]
        photon_ids_offset += photon_ids_size

        # Get track info for the LAST track in this label's genealogy
        # Genealogy is ordered parent->child, so last track is the actual particle that produced photons
        last_track_id = genealogy[-1] if genealogy else None
        track_info = track_info_dict.get(last_track_id, None) if last_track_id is not None else None

        labels.append({
            'genealogy': genealogy,
            'photon_indices': photon_indices,
            'track_info': track_info
        })

    return {
        'n_labels': n_labels,
        'labels': labels,
        'photon_origins': jnp.array(photon_positions),
        'photon_directions': jnp.array(photon_directions),
        'photon_times': jnp.array(photon_times),
        'primary_energy': primary_energy,
        'track_info_dict': track_info_dict  # Include full track info for reference
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

def generate_events_from_photonsim_labels(event_simulator, root_file_path, sensor_params, output_dir=None,
                                         n_events=None, batch_size=100, master_seed=None,
                                         apply_smearing=False, apply_rotation=False, apply_translation=False,
                                         detector_config_path=None, merge_output=True, merged_filename='merged_events.h5'):
    """
    Generate and save events from PhotonSim ROOT file using label-based processing.
    Each label (genealogy) is processed independently through LUCiD simulation.

    This is the NEW workflow where:
    - PhotonSim runs multiple primaries per job
    - Photons are classified by labels (genealogies)
    - Q and T arrays have shape (N_labels, N_sensors) instead of (N_particles, N_sensors)
    - Q_true and T_true are computed by aggregating across labels
    - Q_reco and T_reco can optionally be computed by applying smearing

    Parameters
    ----------
    event_simulator : function
        The event simulation function to use (should have apply_smearing parameter)
    root_file_path : str
        Path to PhotonSim ROOT file containing multiple primaries and labels
    sensor_params : tuple
        scattering length, reflection rate, absorption length and gumbel_softmax
    output_dir : str, optional
        Directory to save output files, by default 'events'
    n_events : int, optional
        Number of events to generate, by default None (uses all entries)
    batch_size : int, optional
        Number of events to accumulate before saving in parallel, by default 100
    master_seed : int, optional
        Random seed for JAX PRNG key generation. If None, generates random seed, by default None
    apply_smearing : bool, optional
        If True, apply smearing to Q_true and T_true to get Q_reco and T_reco, by default False
    apply_rotation : bool, optional
        If True, apply random rotation per primary to all photons and tracks, by default False
    apply_translation : bool, optional
        If True, apply random translation per event to all photons and tracks (after rotation), by default False
    detector_config_path : str, optional
        Path to detector configuration JSON file (required if apply_translation=True), by default None
    merge_output : bool, optional
        Whether to merge individual event files into a single HDF5 file, by default True
    merged_filename : str, optional
        Name of the merged output file, by default 'merged_events.h5'

    Returns
    -------
    list or str
        List of saved file paths, or path to merged file if merge_output=True
    """
    import uproot
    import concurrent.futures
    import time
    import numpy as np
    import json
    from tools.simulation import smear_charges_SK_like, smear_times

    # Generate random seed if not provided
    if master_seed is None:
        master_seed = int(time.time() * 1000000) % (2**32)
        print(f"Generated random master seed from time: {master_seed}")
    else:
        print(f"Using provided master seed: {master_seed}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open ROOT file and get number of entries
    print(f"Loading ROOT file: {root_file_path}")
    root_file = uproot.open(root_file_path)
    tree = root_file['OpticalPhotons']
    num_entries = tree.num_entries
    print(f"  Found {num_entries} entries")
    root_file.close()

    # Determine number of events to generate
    if n_events is None:
        n_events = num_entries
        print(f"No n_events specified, using all {n_events} entries")

    print(f"\nGenerating {n_events} events using label-based processing...")
    print(f"Using batch size of {batch_size} events for multithreaded I/O")
    print(f"Apply smearing: {apply_smearing}")
    print(f"Apply rotation: {apply_rotation}")
    print(f"Apply translation: {apply_translation}")
    print(f"Saving events to directory: {output_dir}")

    # Load detector bounds if translation is requested
    detector_bounds = None
    if apply_translation:
        if detector_config_path is None:
            raise ValueError("detector_config_path must be provided when apply_translation=True")

        with open(detector_config_path, 'r') as f:
            config = json.load(f)

        detector_type = config.get('detector_type', 'cylinder')
        geom_def = config['geometry_definitions']

        if detector_type == 'cylinder':
            detector_bounds = {
                'type': 'cylinder',
                'radius': geom_def['radius'],  # meters
                'height': geom_def['height']   # meters
            }
        elif detector_type == 'sphere':
            detector_bounds = {
                'type': 'sphere',
                'radius': geom_def['radius']
            }
        elif detector_type == 'box':
            detector_bounds = {
                'type': 'box',
                'length': geom_def['length'],
                'width': geom_def['width'],
                'height': geom_def['height']
            }

        print(f"Detector bounds loaded: {detector_bounds}")

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
        batch_filenames = []
        batch_indices = []

        # Process each entry in the current batch
        for event_idx in tqdm(range(start_idx, end_idx), desc=f"Generating batch {batch_idx+1}", unit="event"):
            # Initialize master random key for this event
            master_key = jax.random.PRNGKey(master_seed + event_idx)

            # Read label data from PhotonSim
            label_data = read_label_data_from_photonsim(root_file_path, event_idx)

            n_labels = label_data['n_labels']
            labels = label_data['labels']
            all_photon_origins = label_data['photon_origins']
            all_photon_directions = label_data['photon_directions']
            all_photon_times = label_data['photon_times']

            # Group labels by primary and generate rotations if requested
            rotation_per_label = {}  # Maps label_idx -> (rotation_axis, rotation_angle)

            if apply_rotation:
                # Group labels by their root primary (genealogy[0])
                primary_groups = {}  # Maps primary_track_id -> list of label indices
                for label_idx, label in enumerate(labels):
                    genealogy = label['genealogy']
                    if len(genealogy) > 0:
                        primary_track_id = genealogy[0]  # Root of the genealogy tree
                        if primary_track_id not in primary_groups:
                            primary_groups[primary_track_id] = []
                        primary_groups[primary_track_id].append(label_idx)

                # Generate one rotation per primary group
                for primary_track_id, label_indices in primary_groups.items():
                    # Get the primary's true direction from the first label in the group
                    first_label_idx = label_indices[0]
                    first_label = labels[first_label_idx]

                    # Find the primary track info in genealogy
                    primary_track_info = label_data['track_info_dict'].get(primary_track_id)
                    if primary_track_info is not None:
                        source_direction = jnp.array(primary_track_info['direction'])
                    else:
                        # Fallback: use [0,0,1] if we can't find primary info
                        source_direction = jnp.array([0.0, 0.0, 1.0])

                    # Generate random target direction using spherical coordinates
                    rotation_key, master_key = jax.random.split(master_key)
                    random_values = jax.random.uniform(rotation_key, shape=(2,))
                    cos_theta = 2.0 * random_values[0] - 1.0
                    phi = 2.0 * jnp.pi * random_values[1]
                    sin_theta = jnp.sqrt(1.0 - cos_theta**2)

                    target_direction = jnp.array([
                        sin_theta * jnp.cos(phi),
                        sin_theta * jnp.sin(phi),
                        cos_theta
                    ])

                    # Calculate rotation from source to target
                    source_norm = source_direction / (jnp.linalg.norm(source_direction) + 1e-8)
                    target_norm = target_direction / (jnp.linalg.norm(target_direction) + 1e-8)

                    rotation_axis = jnp.cross(source_norm, target_norm)
                    axis_norm = jnp.linalg.norm(rotation_axis)

                    # Handle parallel or anti-parallel cases
                    rotation_axis = jnp.where(
                        axis_norm < 1e-6,
                        jnp.array([1.0, 0.0, 0.0]),  # Arbitrary axis for zero rotation
                        rotation_axis / (axis_norm + 1e-8)
                    )

                    rotation_angle = jnp.arccos(jnp.clip(jnp.dot(source_norm, target_norm), -1.0, 1.0))

                    # Store rotation for all labels in this primary group
                    for label_idx in label_indices:
                        rotation_per_label[label_idx] = (rotation_axis, rotation_angle)

            # Generate random translation vector for this event (applied AFTER rotation)
            translation_vector = jnp.array([0.0, 0.0, 0.0])  # Default: no translation
            if apply_translation:
                translation_key, master_key = jax.random.split(master_key)

                if detector_bounds['type'] == 'cylinder':
                    # Sample within a fiducial fraction (e.g., 90%) of detector bounds
                    frac = 0.9
                    r_max = detector_bounds['radius'] * frac
                    h_max = detector_bounds['height'] * frac / 2.0  # Half-height

                    # Sample random radius and angle for cylindrical coordinates
                    random_vals = jax.random.uniform(translation_key, shape=(3,))
                    r_sample = r_max * jnp.sqrt(random_vals[0])  # sqrt for uniform in circle
                    theta_sample = 2.0 * jnp.pi * random_vals[1]
                    z_sample = (2.0 * random_vals[2] - 1.0) * h_max  # Uniform in [-h_max, h_max]

                    translation_vector = jnp.array([
                        r_sample * jnp.cos(theta_sample),
                        r_sample * jnp.sin(theta_sample),
                        z_sample
                    ])

                elif detector_bounds['type'] == 'sphere':
                    # Sample uniformly within sphere (fiducial fraction)
                    frac = 0.9
                    r_max = detector_bounds['radius'] * frac

                    random_vals = jax.random.uniform(translation_key, shape=(3,))
                    # Sample uniformly in ball using rejection sampling
                    r_sample = r_max * (random_vals[0] ** (1.0/3.0))  # Cube root for uniform in ball
                    cos_theta = 2.0 * random_vals[1] - 1.0
                    phi = 2.0 * jnp.pi * random_vals[2]
                    sin_theta = jnp.sqrt(1.0 - cos_theta**2)

                    translation_vector = r_sample * jnp.array([
                        sin_theta * jnp.cos(phi),
                        sin_theta * jnp.sin(phi),
                        cos_theta
                    ])

                elif detector_bounds['type'] == 'box':
                    # Sample uniformly within box (fiducial fraction)
                    frac = 0.9
                    l_max = detector_bounds['length'] * frac / 2.0
                    w_max = detector_bounds['width'] * frac / 2.0
                    h_max = detector_bounds['height'] * frac / 2.0

                    random_vals = jax.random.uniform(translation_key, shape=(3,))
                    translation_vector = jnp.array([
                        (2.0 * random_vals[0] - 1.0) * l_max,
                        (2.0 * random_vals[1] - 1.0) * w_max,
                        (2.0 * random_vals[2] - 1.0) * h_max
                    ])

            # Lists to collect Q and T for each label
            Q_per_label_list = []
            T_per_label_list = []

            # Get number of sensors from simulation (will be determined from first run)
            n_sensors = None

            # Process each label
            for label_idx, label in enumerate(labels):
                photon_indices = label['photon_indices']

                if len(photon_indices) == 0:
                    # No photons for this label - skip or use zeros
                    if n_sensors is None:
                        # We don't know sensor count yet, will handle after first label
                        continue
                    Q_per_label_list.append(jnp.zeros(n_sensors))
                    T_per_label_list.append(jnp.zeros(n_sensors))
                    continue

                # Convert photon indices to numpy array for JAX indexing
                photon_indices_array = np.array(photon_indices, dtype=np.int32)

                # Extract photons for this label
                label_photon_origins = all_photon_origins[photon_indices_array]
                label_photon_directions = all_photon_directions[photon_indices_array]
                label_photon_times = all_photon_times[photon_indices_array]

                N = len(label_photon_origins)

                # Padding (1_000_000 is hardcoded in _simulation_core)
                padding_size = max(0, 1_000_000 - N)

                # Pad arrays
                padded_origins = jnp.pad(label_photon_origins, ((0, padding_size), (0, 0)),
                                        mode='constant', constant_values=0)

                default_direction = jnp.array([0.0, 0.0, 1.0])
                padding_directions = jnp.tile(default_direction, (padding_size, 1))
                if padding_size > 0:
                    padded_directions = jnp.concatenate([label_photon_directions, padding_directions], axis=0)
                else:
                    padded_directions = label_photon_directions

                padded_times = jnp.pad(label_photon_times, (0, padding_size),
                                       mode='constant', constant_values=0)

                # Track parameters (use track info from label)
                track_info = label['track_info']
                if track_info is not None:
                    track_energy = track_info['energy']
                    # Convert position from cm (PhotonSim) to meters (LUCiD)
                    track_position = jnp.array(track_info['position']) / 100.0
                    track_direction = jnp.array(track_info['direction'])
                else:
                    # Fallback if no track info
                    track_energy = label_data['primary_energy']
                    track_position = jnp.array([0.0, 0.0, 0.0])
                    track_direction = jnp.array([0.0, 0.0, 1.0])

                # Apply rotation to track info if this label has a rotation
                if label_idx in rotation_per_label:
                    rotation_axis, rotation_angle = rotation_per_label[label_idx]

                    # Rotate track position (in meters) and direction
                    track_position = jax_rotate_vector_local(track_position, rotation_axis, rotation_angle)
                    track_direction = jax_rotate_vector_local(track_direction, rotation_axis, rotation_angle)

                # Apply translation to track position (applied AFTER rotation, in meters)
                if apply_translation:
                    track_position = track_position + translation_vector

                # Update the label's track_info with final values (STORE IN METERS)
                if track_info is not None:
                    track_info['position'] = np.array(track_position)  # Store in meters
                    track_info['direction'] = np.array(track_direction)

                # Prepare photonsim data dict
                # Always include rotation parameters (with defaults if no rotation)
                if label_idx in rotation_per_label:
                    rotation_axis, rotation_angle = rotation_per_label[label_idx]
                    do_apply_rotation = True
                else:
                    rotation_axis = jnp.array([1.0, 0.0, 0.0])  # Dummy axis
                    rotation_angle = 0.0  # No rotation
                    do_apply_rotation = False

                photonsim_data = {
                    'photon_origins': padded_origins,
                    'photon_directions': padded_directions,
                    'photon_times': padded_times,
                    'N': N,
                    'apply_rotation': do_apply_rotation,
                    'rotation_axis': rotation_axis,
                    'rotation_angle': rotation_angle,
                    'apply_translation': apply_translation,
                    'translation_vector': translation_vector
                }

                track_params = (track_energy, track_position, track_direction)

                # Get simulation key
                sim_key, master_key = jax.random.split(master_key)

                # Run simulation for this label (WITHOUT smearing)
                # The event_simulator should respect apply_smearing=False internally
                Q_label, T_label = event_simulator(track_params, sensor_params, sim_key, photonsim_data)

                # Store sensor count from first result
                if n_sensors is None:
                    n_sensors = len(Q_label)

                Q_per_label_list.append(Q_label)
                T_per_label_list.append(T_label)

            # Handle case where all labels have no photons
            if n_sensors is None:
                print(f"Warning: Event {event_idx} has no photons in any label, skipping...")
                continue

            # Ensure we have arrays for all labels (fill with zeros if needed)
            while len(Q_per_label_list) < n_labels:
                Q_per_label_list.append(jnp.zeros(n_sensors))
                T_per_label_list.append(jnp.zeros(n_sensors))

            # Stack into arrays (N_labels, N_sensors)
            Q_per_label = jnp.stack(Q_per_label_list, axis=0)
            T_per_label = jnp.stack(T_per_label_list, axis=0)

            # Calculate Q_true and T_true by aggregating across labels
            Q_true = jnp.sum(Q_per_label, axis=0)  # Sum charges
            T_true = jnp.min(jnp.where(T_per_label > 0, T_per_label, jnp.inf), axis=0)  # Min times
            T_true = jnp.where(jnp.isfinite(T_true), T_true, 0.0)  # Replace inf with 0

            # Apply smearing if requested
            if apply_smearing:
                reco_key, master_key = jax.random.split(master_key)
                smear_q_key, smear_t_key = jax.random.split(reco_key)
                Q_reco = smear_charges_SK_like(Q_true, key=smear_q_key)
                T_reco = smear_times(T_true, key=smear_t_key)
            else:
                Q_reco = Q_true
                T_reco = T_true

            # Create filename
            event_number = event_idx
            filename = os.path.join(output_dir, f'event_{event_number}.h5')

            # Extended info with label structure
            extended_info = {
                'n_labels': n_labels,
                'labels': labels,
                'track_info_dict': label_data['track_info_dict'],
                'primary_energy': label_data['primary_energy'],
                'Q_per_label': Q_per_label,
                'T_per_label': T_per_label,
                'Q_true': Q_true,
                'T_true': T_true,
                'Q_reco': Q_reco,
                'T_reco': T_reco,
                'apply_smearing': apply_smearing,
                'source': 'PhotonSim_Labels'
            }

            # Store for batch processing
            batch_data.append(extended_info)
            batch_filenames.append(filename)
            batch_indices.append(event_number)

        # Save all events in the batch using multithreading
        # Note: We'll need to update save function to handle label-based structure
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    save_single_event_with_label_info,  # New save function
                    data,
                    event_number=idx,
                    filename=filename
                )
                for data, filename, idx in zip(batch_data, batch_filenames, batch_indices)
            ]

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

    print(f"\nSuccessfully processed {len(saved_files)} events with label-based structure.")
    print(f"All events saved to {output_dir}")

    # Merge files if requested
    if merge_output:
        print("\nMerging individual event files...")
        merged_path = merge_event_files(output_dir, merged_filename=merged_filename, remove_individuals=True)
        return merged_path

    return saved_files
