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
from tools.utils import save_single_event_with_extended_info, save_single_event_with_particle_info, get_random_root_entry_index, superimpose_multiple_events, merge_event_files
from tools.production.voxelize import VoxelGridConfig, voxelize_from_photon_indices, pack_voxel_data_for_hdf5
from jax import jit


def get_max_photons_per_particle(root_file_path, n_events=None):
    """
    Efficiently scan a ROOT file to find the maximum number of photons in any single particle.

    This function reads only the Particle_PhotonIDsSize branch for all events at once,
    which is much faster than iterating through events one by one.

    Parameters
    ----------
    root_file_path : str
        Path to the PhotonSim ROOT file
    n_events : int, optional
        Number of events to scan. If None, scans all events.

    Returns
    -------
    int
        Maximum number of photons found in any single particle across all scanned events
    """
    import uproot

    root_file = uproot.open(root_file_path)
    tree = root_file['OpticalPhotons']
    num_entries = tree.num_entries

    # Limit to n_events if specified
    entry_stop = min(n_events, num_entries) if n_events is not None else num_entries

    # Read Particle_PhotonIDsSize for all events at once (jagged array)
    photon_ids_sizes = tree['Particle_PhotonIDsSize'].array(
        entry_start=0, entry_stop=entry_stop, library='np'
    )

    # Find global maximum across all events and all particles
    max_photons = 0
    for event_sizes in photon_ids_sizes:
        if len(event_sizes) > 0:
            event_max = int(max(event_sizes))
            if event_max > max_photons:
                max_photons = event_max

    root_file.close()
    return max_photons


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

def read_particle_data_from_photonsim(root_file_path, entry_index, include_track_segments=False):
    """
    Read particle-based photon data from a PhotonSim ROOT file for a specific entry.
    This function reads the particle system that classifies photons by genealogy.

    Parameters
    ----------
    root_file_path : str
        Path to the PhotonSim ROOT file
    entry_index : int
        Entry index to read from the file
    include_track_segments : bool, optional
        If True, also read meaningful tracks and segment data, by default False

    Returns
    -------
    dict
        Dictionary containing:
        - 'n_particles': int, number of unique particles
        - 'particles': list of dicts, each containing:
            - 'genealogy': list of track IDs in the genealogy
            - 'photon_indices': list of photon indices belonging to this particle
            - 'track_info': dict with track information (position, direction, energy, time, pdg, category, etc.)
            - 'extended_genealogy': list of meaningful track IDs (if include_track_segments=True)
        - 'photon_origins': array (N_photons, 3) in cm
        - 'photon_directions': array (N_photons, 3)
        - 'photon_times': array (N_photons,) in ns
        - 'primary_energy': float, energy in MeV
        - 'meaningful_tracks': dict (if include_track_segments=True)
        - 'segments': dict (if include_track_segments=True)
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
        'NParticles',
        'Particle_GenealogySize', 'Particle_GenealogyData',
        'Particle_PhotonIDsSize', 'Particle_PhotonIDsData',
        'TrackInfo_TrackID', 'TrackInfo_Category', 'TrackInfo_SubID',
        'TrackInfo_PosX', 'TrackInfo_PosY', 'TrackInfo_PosZ',
        'TrackInfo_DirX', 'TrackInfo_DirY', 'TrackInfo_DirZ',
        'TrackInfo_Energy', 'TrackInfo_Time',
        'TrackInfo_ParentTrackID', 'TrackInfo_PDG'
    ]

    # Add branches for track segments if requested
    if include_track_segments:
        branches_to_read.extend([
            'Particle_ExtGenealogySize', 'Particle_ExtGenealogyData',
            'NMeaningfulTracks',
            'MTrack_TrackID', 'MTrack_ParentID', 'MTrack_PDG',
            'MTrack_InitialEnergy', 'MTrack_ParticleName', 'MTrack_NCherenkov',
            'MTrack_SegmentOffset', 'MTrack_NSegments',
            'NSegments',
            'Segment_StartX', 'Segment_StartY', 'Segment_StartZ',
            'Segment_EndX', 'Segment_EndY', 'Segment_EndZ',
            'Segment_DirX', 'Segment_DirY', 'Segment_DirZ',
            'Segment_Edep', 'Segment_Time'
        ])

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

    # Extract particle system
    n_particles = int(tree_data['NParticles'][0])

    # Parse genealogy data
    genealogy_sizes = tree_data['Particle_GenealogySize'][0]
    genealogy_data = tree_data['Particle_GenealogyData'][0]

    # Parse photon IDs data
    photon_ids_sizes = tree_data['Particle_PhotonIDsSize'][0]
    photon_ids_data = tree_data['Particle_PhotonIDsData'][0]

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

    # Parse extended genealogy data if requested
    ext_genealogy_sizes = None
    ext_genealogy_data = None
    if include_track_segments:
        ext_genealogy_sizes = tree_data['Particle_ExtGenealogySize'][0]
        ext_genealogy_data = tree_data['Particle_ExtGenealogyData'][0]

    # Parse particles
    particles = []
    genealogy_offset = 0
    photon_ids_offset = 0
    ext_genealogy_offset = 0

    for particle_idx in range(n_particles):
        # Extract genealogy for this particle
        gen_size = int(genealogy_sizes[particle_idx])
        genealogy = [int(genealogy_data[genealogy_offset + i]) for i in range(gen_size)]
        genealogy_offset += gen_size

        # Extract photon indices for this particle
        photon_ids_size = int(photon_ids_sizes[particle_idx])
        photon_indices = [int(photon_ids_data[photon_ids_offset + i]) for i in range(photon_ids_size)]
        photon_ids_offset += photon_ids_size

        # Extract extended genealogy if available
        extended_genealogy = None
        if include_track_segments and ext_genealogy_sizes is not None:
            ext_gen_size = int(ext_genealogy_sizes[particle_idx])
            extended_genealogy = [int(ext_genealogy_data[ext_genealogy_offset + i]) for i in range(ext_gen_size)]
            ext_genealogy_offset += ext_gen_size

        # Get track info for the LAST track in this particle's genealogy
        # Genealogy is ordered parent->child, so last track is the actual particle that produced photons
        last_track_id = genealogy[-1] if genealogy else None
        track_info = track_info_dict.get(last_track_id, None) if last_track_id is not None else None

        particle_dict = {
            'genealogy': genealogy,
            'photon_indices': photon_indices,
            'track_info': track_info
        }
        if extended_genealogy is not None:
            particle_dict['extended_genealogy'] = extended_genealogy

        particles.append(particle_dict)

    # Build result dictionary
    result = {
        'n_particles': n_particles,
        'particles': particles,
        'photon_origins': photon_positions,  # Keep as NumPy (avoid JAX conversion overhead)
        'photon_directions': photon_directions,  # Keep as NumPy
        'photon_times': photon_times,  # Keep as NumPy
        'primary_energy': primary_energy,
        'track_info_dict': track_info_dict  # Include full track info for reference
    }

    # Parse meaningful tracks and segments if requested
    if include_track_segments:
        n_meaningful_tracks = int(tree_data['NMeaningfulTracks'][0])
        n_segments = int(tree_data['NSegments'][0])

        # Build meaningful tracks dictionary (keyed by track ID for easy lookup)
        meaningful_tracks = {}
        mtrack_ids = tree_data['MTrack_TrackID'][0]
        mtrack_parent_ids = tree_data['MTrack_ParentID'][0]
        mtrack_pdgs = tree_data['MTrack_PDG'][0]
        mtrack_energies = tree_data['MTrack_InitialEnergy'][0]
        mtrack_names = tree_data['MTrack_ParticleName'][0]
        mtrack_ncherenkov = tree_data['MTrack_NCherenkov'][0]
        mtrack_seg_offsets = tree_data['MTrack_SegmentOffset'][0]
        mtrack_nsegs = tree_data['MTrack_NSegments'][0]

        for i in range(n_meaningful_tracks):
            track_id = int(mtrack_ids[i])
            meaningful_tracks[track_id] = {
                'track_id': track_id,
                'parent_id': int(mtrack_parent_ids[i]),
                'pdg': int(mtrack_pdgs[i]),
                'initial_energy': float(mtrack_energies[i]),
                'particle_name': str(mtrack_names[i]),
                'n_cherenkov': int(mtrack_ncherenkov[i]),
                'segment_offset': int(mtrack_seg_offsets[i]),
                'n_segments': int(mtrack_nsegs[i])
            }

        # Extract segment arrays (positions in mm, convert to cm)
        segments = {
            'start_x': tree_data['Segment_StartX'][0] / 10.0,  # mm to cm
            'start_y': tree_data['Segment_StartY'][0] / 10.0,
            'start_z': tree_data['Segment_StartZ'][0] / 10.0,
            'end_x': tree_data['Segment_EndX'][0] / 10.0,
            'end_y': tree_data['Segment_EndY'][0] / 10.0,
            'end_z': tree_data['Segment_EndZ'][0] / 10.0,
            'dir_x': tree_data['Segment_DirX'][0],
            'dir_y': tree_data['Segment_DirY'][0],
            'dir_z': tree_data['Segment_DirZ'][0],
            'edep': tree_data['Segment_Edep'][0],
            'time': tree_data['Segment_Time'][0],
            'n_segments': n_segments
        }

        result['meaningful_tracks'] = meaningful_tracks
        result['segments'] = segments

    return result

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

def generate_events_from_photonsim_particles(event_simulator, root_file_path, sensor_params, output_dir=None,
                                             n_events=None, batch_size=100, master_seed=None,
                                             apply_smearing=False, apply_rotation=False, apply_translation=False,
                                             detector_config_path=None, merge_output=True, merged_filename='merged_events.h5',
                                             include_track_segments=False):
    """
    VMAP-OPTIMIZED VERSION: Generate and save events using batched particle processing.

    This version dynamically determines the optimal PAD_SIZE based on the maximum photons
    per particle in the ROOT file, then uses jax.vmap to process all particles in parallel,
    eliminating the Python loop and achieving significant speedup.

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
        If True, apply smearing to PE_true and T_true to get PE_reco and T_reco, by default False
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

    # Dynamically determine PAD_SIZE based on actual data in the ROOT file
    print(f"Scanning ROOT file to determine optimal PAD_SIZE...")
    max_photons_in_file = get_max_photons_per_particle(root_file_path, n_events)
    PAD_SIZE = max_photons_in_file + 1  # +1 for safety margin
    print(f"  Max photons per particle in file: {max_photons_in_file:,}")
    print(f"  Using PAD_SIZE: {PAD_SIZE:,}")

    print(f"\nGenerating {n_events} events using VMAP-OPTIMIZED particle-based processing...")
    print(f"Using batch size of {batch_size} events for multithreaded I/O")
    print(f"Apply smearing: {apply_smearing}")
    print(f"Apply translation: {apply_translation}")
    # Note: Rotation is not applied in this workflow because PhotonSim already generates
    # primaries with random isotropic directions (/gun/randomDirection true). The photon
    # and track data are already in randomized coordinate frames, so rotation in LUCiD
    # would be redundant. Only translation is needed to place the vertex in the detector.
    if apply_rotation:
        print(f"WARNING: apply_rotation=True was passed but rotation is disabled in this workflow.")
        print(f"         PhotonSim already generates tracks with random directions, so rotation is unnecessary.")
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
                'radius': geom_def['radius'],
                'height': geom_def['height']
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
    event_times = []  # Track event processing times

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
        for event_idx in range(start_idx, end_idx):
            event_start_time = time.time()
            print(f"\n  Event {event_idx+1}/{n_events} (index {event_idx}):", flush=True)

            # Initialize master random key for this event
            master_key = jax.random.PRNGKey(master_seed + event_idx)

            # Read particle data from PhotonSim
            print(f"    Reading particle data from ROOT file...", flush=True)
            particle_data = read_particle_data_from_photonsim(root_file_path, event_idx, include_track_segments=include_track_segments)

            n_particles = particle_data['n_particles']
            particles = particle_data['particles']
            all_photon_origins = particle_data['photon_origins']
            all_photon_directions = particle_data['photon_directions']
            all_photon_times = particle_data['photon_times']
            total_photons = len(all_photon_origins)
            print(f"    Found {n_particles} particles, {total_photons:,} total photons", flush=True)

            # ========================================================================
            # VECTORIZED NUMPY PREPROCESSING + EFFICIENT JAX TRANSFER
            # All preprocessing in NumPy, single efficient transfer using device_put
            # ========================================================================
            print(f"    Preprocessing photon data...", flush=True)

            # PAD_SIZE is now computed dynamically at the function level
            default_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            # Data already NumPy - ensure float32 and convert to meters (avoid unnecessary copies)
            all_photon_origins_np = all_photon_origins.astype(np.float32, copy=False) / 100.0
            all_photon_directions_np = all_photon_directions.astype(np.float32, copy=False)
            all_photon_times_np = all_photon_times.astype(np.float32, copy=False)

            # Generate random translation vector
            translation_vector = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            if apply_translation:
                translation_seed = (master_seed + event_idx * 1000) % (2**32)
                rng = np.random.default_rng(seed=translation_seed)

                if detector_bounds['type'] == 'cylinder':
                    frac, r_max = 0.9, detector_bounds['radius'] * 0.9
                    h_max = detector_bounds['height'] * 0.9 / 2.0
                    random_vals = rng.uniform(0, 1, size=3).astype(np.float32)
                    r_sample = r_max * np.sqrt(random_vals[0])
                    theta_sample = 2.0 * np.pi * random_vals[1]
                    z_sample = (2.0 * random_vals[2] - 1.0) * h_max
                    translation_vector = np.array([
                        r_sample * np.cos(theta_sample),
                        r_sample * np.sin(theta_sample),
                        z_sample
                    ], dtype=np.float32)
                elif detector_bounds['type'] == 'sphere':
                    r_max = detector_bounds['radius'] * 0.9
                    random_vals = rng.uniform(0, 1, size=3).astype(np.float32)
                    r_sample = r_max * (random_vals[0] ** (1.0/3.0))
                    cos_theta, phi = 2.0 * random_vals[1] - 1.0, 2.0 * np.pi * random_vals[2]
                    sin_theta = np.sqrt(1.0 - cos_theta**2)
                    translation_vector = r_sample * np.array([
                        sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta
                    ], dtype=np.float32)
                elif detector_bounds['type'] == 'box':
                    random_vals = rng.uniform(0, 1, size=3).astype(np.float32)
                    translation_vector = np.array([
                        (2.0 * random_vals[0] - 1.0) * detector_bounds['length'] * 0.45,
                        (2.0 * random_vals[1] - 1.0) * detector_bounds['width'] * 0.45,
                        (2.0 * random_vals[2] - 1.0) * detector_bounds['height'] * 0.45
                    ], dtype=np.float32)

                all_photon_origins_np += translation_vector[None, :]

                # Apply translation to segment positions if track segments are included
                if include_track_segments and 'segments' in particle_data:
                    segments = particle_data['segments']
                    # Convert translation from meters to cm (segments are in cm)
                    translation_cm = translation_vector * 100.0
                    segments['start_x'] = segments['start_x'] + translation_cm[0]
                    segments['start_y'] = segments['start_y'] + translation_cm[1]
                    segments['start_z'] = segments['start_z'] + translation_cm[2]
                    segments['end_x'] = segments['end_x'] + translation_cm[0]
                    segments['end_y'] = segments['end_y'] + translation_cm[1]
                    segments['end_z'] = segments['end_z'] + translation_cm[2]

            # Pre-allocate batched arrays
            batched_origins_np = np.zeros((n_particles, PAD_SIZE, 3), dtype=np.float32)
            batched_directions_np = np.tile(default_direction, (n_particles, PAD_SIZE, 1))
            batched_times_np = np.zeros((n_particles, PAD_SIZE), dtype=np.float32)

            # Build track parameters as NumPy arrays (more efficient than lists)
            N_per_particle_np = np.zeros(n_particles, dtype=np.int32)
            track_energies_np = np.zeros(n_particles, dtype=np.float32)
            track_positions_np = np.zeros((n_particles, 3), dtype=np.float32)
            track_directions_np = np.zeros((n_particles, 3), dtype=np.float32)

            # Process each particle
            for particle_idx, particle in enumerate(particles):
                photon_indices = particle['photon_indices']
                N = len(photon_indices)
                N_per_particle_np[particle_idx] = N

                # Extract track parameters
                track_info = particle['track_info']
                if track_info is not None:
                    track_energies_np[particle_idx] = track_info['energy']
                    track_positions_np[particle_idx] = track_info['position'] / 100.0  # cm to m
                    track_directions_np[particle_idx] = track_info['direction']
                else:
                    track_energies_np[particle_idx] = particle_data['primary_energy']
                    track_directions_np[particle_idx] = [0.0, 0.0, 1.0]

                if apply_translation:
                    track_positions_np[particle_idx] += translation_vector
                    # Update particles data structure with transformed position (convert back to cm)
                    if track_info is not None:
                        track_info['position'] = track_positions_np[particle_idx] * 100.0

                # Scatter photons
                if N > 0:
                    batched_origins_np[particle_idx, :N] = all_photon_origins_np[photon_indices]
                    batched_directions_np[particle_idx, :N] = all_photon_directions_np[photon_indices]
                    batched_times_np[particle_idx, :N] = all_photon_times_np[photon_indices]

            # Efficient transfer to JAX device (avoids unnecessary copies)
            batched_origins = jax.device_put(batched_origins_np)
            batched_directions = jax.device_put(batched_directions_np)
            batched_times = jax.device_put(batched_times_np)
            N_per_particle_array = jax.device_put(N_per_particle_np)
            track_energies_array = jax.device_put(track_energies_np)
            track_positions_array = jax.device_put(track_positions_np)
            track_directions_array = jax.device_put(track_directions_np)

            # ========================================================================
            # VMAP OPTIMIZATION: Process all particles in parallel using vmap
            # ========================================================================
            print(f"    Running VMAP simulation for {n_particles} particles...", flush=True)
            sim_start_time = time.time()

            # Create a wrapper function that processes a single particle
            def simulate_single_particle(track_energy, track_pos, track_dir, photon_origins,
                                         photon_dirs, photon_times, N, sim_key):
                """Process a single particle - will be vmapped over all particles."""
                track_params = (track_energy, track_pos, track_dir)

                photonsim_data = {
                    'photon_origins': photon_origins,
                    'photon_directions': photon_dirs,
                    'photon_times': photon_times,
                    'N': N,
                    'apply_rotation': False,
                    'rotation_axis': jnp.array([1.0, 0.0, 0.0]),
                    'rotation_angle': 0.0,
                    'apply_translation': apply_translation,
                    'translation_vector': translation_vector
                }

                return event_simulator(track_params, sensor_params, sim_key, photonsim_data)

            # Create vectorized version using vmap
            # in_axes: (0, 0, 0, 0, 0, 0, 0, 0) means vectorize over first axis of all arguments
            simulate_all_particles = jax.vmap(
                simulate_single_particle,
                in_axes=(0, 0, 0, 0, 0, 0, 0, 0)
            )

            # Generate random keys for all particles
            particle_keys = jax.random.split(master_key, n_particles)

            # Process all particles in one vectorized call!
            PE_per_particle, T_per_particle = simulate_all_particles(
                track_energies_array,
                track_positions_array,
                track_directions_array,
                batched_origins,
                batched_directions,
                batched_times,
                N_per_particle_array,
                particle_keys
            )
            sim_elapsed = time.time() - sim_start_time
            print(f"    Simulation completed in {sim_elapsed:.2f}s", flush=True)

            # Calculate PE_true and T_true by aggregating across particles
            PE_true = jnp.sum(PE_per_particle, axis=0)
            T_true = jnp.min(jnp.where(T_per_particle > 0, T_per_particle, jnp.inf), axis=0)
            T_true = jnp.where(jnp.isfinite(T_true), T_true, 0.0)

            # Apply smearing if requested
            if apply_smearing:
                reco_key, master_key = jax.random.split(master_key)
                smear_pe_key, smear_t_key = jax.random.split(reco_key)
                PE_reco = smear_charges_SK_like(PE_true, key=smear_pe_key)
                T_reco = smear_times(T_true, key=smear_t_key)
            else:
                PE_reco = PE_true
                T_reco = T_true

            # Convert JAX arrays to numpy BEFORE storing in extended_info
            # This is critical for thread-safe saving with ThreadPoolExecutor
            PE_per_particle = np.asarray(PE_per_particle, dtype=np.float32)
            T_per_particle = np.asarray(T_per_particle, dtype=np.float32)
            PE_true = np.asarray(PE_true, dtype=np.float32)
            T_true = np.asarray(T_true, dtype=np.float32)
            PE_reco = np.asarray(PE_reco, dtype=np.float32)
            T_reco = np.asarray(T_reco, dtype=np.float32)

            # Calculate light containment
            light_containment_by_particle = np.zeros(n_particles, dtype=np.float64)
            overall_light_containment = 0.0

            if apply_translation and detector_bounds is not None:
                # Calculate which photons are inside detector bounds (after translation)
                if detector_bounds['type'] == 'cylinder':
                    r = np.sqrt(all_photon_origins_np[:, 0]**2 + all_photon_origins_np[:, 1]**2)
                    z = all_photon_origins_np[:, 2]
                    all_photons_inside_mask = (r <= detector_bounds['radius']) & (np.abs(z) <= detector_bounds['height'] / 2.0)
                elif detector_bounds['type'] == 'sphere':
                    r = np.sqrt(np.sum(all_photon_origins_np**2, axis=1))
                    all_photons_inside_mask = r <= detector_bounds['radius']
                elif detector_bounds['type'] == 'box':
                    all_photons_inside_mask = ((np.abs(all_photon_origins_np[:, 0]) <= detector_bounds['length'] / 2.0) &
                                               (np.abs(all_photon_origins_np[:, 1]) <= detector_bounds['width'] / 2.0) &
                                               (np.abs(all_photon_origins_np[:, 2]) <= detector_bounds['height'] / 2.0))

                # Calculate per-particle containment
                total_photons_all_particles = 0
                total_photons_inside_all_particles = 0

                for particle_idx, particle in enumerate(particles):
                    photon_indices = particle['photon_indices']
                    N = len(photon_indices)
                    if N > 0:
                        mask_for_particle = all_photons_inside_mask[photon_indices]
                        n_inside = int(np.sum(mask_for_particle))
                        light_containment_by_particle[particle_idx] = float(n_inside) / N
                        total_photons_all_particles += N
                        total_photons_inside_all_particles += n_inside

                # Calculate overall containment
                if total_photons_all_particles > 0:
                    overall_light_containment = float(total_photons_inside_all_particles) / total_photons_all_particles

            # ========================================================================
            # VOXELIZATION
            # Convert photon positions to sparse voxel representation
            # ========================================================================
            print(f"    Voxelizing photon positions...", flush=True)
            voxel_start_time = time.time()

            # Get photon indices for each particle
            particle_photon_indices_list = [particle['photon_indices'] for particle in particles]

            # Voxelize using positions in meters
            voxel_config = VoxelGridConfig()
            voxel_data = voxelize_from_photon_indices(
                all_photon_origins_np,  # Already in meters
                particle_photon_indices_list,
                voxel_config
            )

            # Pack for HDF5 storage
            packed_voxel_data = pack_voxel_data_for_hdf5(voxel_data)

            voxel_elapsed = time.time() - voxel_start_time
            total_voxels = np.sum(voxel_data['n_nonzero_voxels'])
            print(f"    Voxelization: {total_voxels:,} voxels in {voxel_elapsed:.3f}s", flush=True)

            # Create filename
            event_number = event_idx
            filename = os.path.join(output_dir, f'event_{event_number}.h5')

            # Generate event time offset t0 (simulates unknown event start time)
            t0 = np.random.uniform(-15.0, 15.0)

            # Extended info with particle structure
            extended_info = {
                'n_particles': n_particles,
                'particles': particles,
                'track_info_dict': particle_data['track_info_dict'],
                't0': t0,
                'PE_per_particle': PE_per_particle,
                'T_per_particle': T_per_particle,
                'PE_reco': PE_reco,
                'T_reco': T_reco,
                'source': 'PhotonSim_Particles_VMAP',
                'overall_light_containment': overall_light_containment,
                'light_containment_by_particle': light_containment_by_particle,
                # Voxel data (sparse representation)
                'voxel_n_nonzero': packed_voxel_data['voxel_n_nonzero'],
                'voxel_offsets': packed_voxel_data['voxel_offsets'],
                'voxel_flat_indices': packed_voxel_data['voxel_flat_indices'],
                'voxel_counts': packed_voxel_data['voxel_counts'],
                # Track segment data (if included)
                'include_track_segments': include_track_segments
            }

            # Add meaningful tracks and segments if requested
            if include_track_segments and 'meaningful_tracks' in particle_data:
                extended_info['meaningful_tracks'] = particle_data['meaningful_tracks']
                extended_info['segments'] = particle_data['segments']

            # Store for batch processing
            batch_data.append(extended_info)
            batch_filenames.append(filename)
            batch_indices.append(event_number)

            event_total_time = time.time() - event_start_time
            event_times.append(event_total_time)
            print(f"    Event total time: {event_total_time:.2f}s", flush=True)

        # Save all events in the batch using multithreading
        print(f"Saving batch {batch_idx+1}...")
        t_save_start = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    save_single_event_with_particle_info,
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
                    import traceback
                    print(f"Error saving file: {e}")
                    traceback.print_exc()

        t_save = time.time() - t_save_start
        print(f"Batch {batch_idx+1} save time: {t_save:.3f}s\n")

    print(f"\nSuccessfully processed {len(saved_files)} events with VMAP-OPTIMIZED particle-based structure.")
    print(f"All events saved to {output_dir}")

    # Print average event time
    if event_times:
        avg_time = sum(event_times) / len(event_times)
        print(f"Average event processing time: {avg_time:.3f}s")

    # Merge files if requested
    if merge_output:
        print("\nMerging individual event files...")
        merged_path = merge_event_files(output_dir, merged_filename=merged_filename, remove_individuals=True)
        return merged_path

    return saved_files
