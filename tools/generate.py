import jax
from functools import partial
from jax import random
import sys, os
import h5py
import numpy as np
import jax.numpy as jnp
import time
from tqdm import tqdm 
from tools.siren import *
from tools.table import *
from tools.utils import read_photon_data_from_root
from tools.utils import save_single_event_with_extended_info, get_random_root_entry_index, superimpose_multiple_events


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


def generate_events_from_root(event_simulator, root_file_path, output_dir='events', n_events=None, 
                            n_rings=1, pion_root_file_path=None,
                            max_detectors_per_cell=4, batch_size=100):
    """
    Generate and save events from a ROOT file, with support for N rings of particles.
    Ring 1 (N=1) is always a muon, and additional rings (N>1) are pions.
    Events are saved with sequential numbering: event_0.h5, event_1.h5, etc.
    
    Parameters
    ----------
    root_file_path : str
        Path to the ROOT file for muons
    output_dir : str, optional
        Directory to save output files, by default 'events'
    n_events : int, optional
        Number of events to process (None for all), by default None
    n_rings : int, optional
        Number of rings (particles) to superimpose, by default 1
        First ring is always a muon, additional rings are pions
    pion_root_file_path : str, optional
        Path to ROOT file for pions, required if n_rings > 1, by default None
    max_detectors_per_cell : int, optional
        Maximum detectors per cell, by default 4
    batch_size : int, optional
        Number of events to accumulate before saving in parallel, by default 100
        
    Returns
    -------
    list
        List of saved file paths
    """
    import uproot
    import concurrent.futures
    
    # Validate arguments
    if n_rings < 1:
        raise ValueError("n_rings must be at least 1")
    
    # If n_rings > 1, we need a pion ROOT file
    if n_rings > 1 and pion_root_file_path is None:
        raise ValueError("When n_rings > 1, pion_root_file_path must be provided")
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open ROOT file to get number of entries
    root_file = uproot.open(root_file_path)
    tree = root_file['v_photon']
    total_entries = tree.num_entries
    
    if n_events is None:
        n_events = total_entries
    else:
        n_events = min(n_events, total_entries)
    
    # Prepare descriptor for printing
    ring_description = f"{n_rings} ring{'s' if n_rings > 1 else ''}"
    particle_description = "muon" if n_rings == 1 else f"muon + {n_rings-1} pion{'s' if n_rings > 1 else ''}"
    
    print(f"Processing {n_events} events with {ring_description} ({particle_description})...")
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
        batch_params = []
        batch_filenames = []
        batch_indices = []
        
        # Process each entry in the current batch
        for i in tqdm(range(start_idx, end_idx), desc=f"Generating batch {batch_idx+1}", unit="event"):
            # Initialize master random key for this event
            master_key = jax.random.PRNGKey(i * 1000)
            
            # Generate a random vertex for all events in this iteration
            vertex_key, master_key = jax.random.split(master_key)
            shared_vertex = generate_random_vertex(vertex_key)
            
            # Lists to store charges and times for all rings
            all_charges = []
            all_times = []
            all_energies = []
            all_directions = []
            all_indices = []  # To store the indices of each particle for filename
            
            # Process the first ring - always a muon
            muon_data = read_photon_data_from_root(root_file_path, i, 'muon')
            
            # Set up parameters
            muon_energy = muon_data['energy']
            initial_intensity = 1.0
            scatter_length = 10.0  # default scattering length
            reflection_rate = 0.1  # default reflection rate
            absorption_length = 10.0  # default absorption length
            temperature = 0. 
            
            # Generate random direction for muon
            dir_key, master_key = jax.random.split(master_key)
            muon_direction = generate_random_direction(dir_key)
            
            # Create parameters tuple for muon
            muon_params = (
                muon_energy,
                shared_vertex,   # Use random vertex
                muon_direction,  # Use random direction
                initial_intensity,
                scatter_length,
                reflection_rate,
                absorption_length,
                temperature  
            )
            
            # Get a key for the muon simulation
            sim_key, master_key = jax.random.split(master_key)

            # Process muon data
            photon_origins = muon_data['photon_origins']  # Now contains direct origins (cm)
            photon_directions = muon_data['photon_directions']  # Now contains direct directions
            N = len(photon_origins)

            # the number 1_000_000 is hard coded also in _simulation_core
            padding_size = max(0, 1_000_000-N)

            # Pad the origins array (2D array with shape [N,3])
            muon_data['photon_origins'] = jnp.pad(photon_origins, ((0, padding_size), (0, 0)), 
                                                mode='constant', constant_values=0)

            # Pad the directions array with a default unit vector [0,0,1]
            default_direction = jnp.array([0.0, 0.0, 1.0])
            padding_directions = jnp.tile(default_direction, (padding_size, 1))
            if padding_size > 0:
                muon_data['photon_directions'] = jnp.concatenate([photon_directions, padding_directions], axis=0)
            else:
                muon_data['photon_directions'] = photon_directions
                
            muon_data['N'] = N
                        
            # Run simulation for muon
            muon_charges, muon_times = event_simulator(muon_params, sim_key, muon_data)
            
            # Store muon data
            all_charges.append(muon_charges)
            all_times.append(muon_times)
            all_energies.append(muon_energy)
            all_directions.append(muon_direction)
            all_indices.append(i)  # Store the original index
            
            # Process additional rings (pions) if n_rings > 1
            for ring_idx in range(1, n_rings):
                # Get a random entry index from the pion file
                random_idx = get_random_root_entry_index(pion_root_file_path)
                
                # Read photon data for pion
                pion_data = read_photon_data_from_root(pion_root_file_path, random_idx, 'pion')

                photon_origins = pion_data['photon_origins']
                photon_directions = pion_data['photon_directions']
                N = len(photon_origins)

                # the number 1_000_000 is hard coded also in _simulation_core
                padding_size = max(0, 1_000_000-N)

                # Pad the origins array (2D array with shape [N,3])
                pion_data['photon_origins'] = jnp.pad(photon_origins, ((0, padding_size), (0, 0)), 
                                                     mode='constant', constant_values=0)

                # Pad the directions array with a default unit vector [0,0,1]
                default_direction = jnp.array([0.0, 0.0, 1.0])
                padding_directions = jnp.tile(default_direction, (padding_size, 1))
                if padding_size > 0:
                    pion_data['photon_directions'] = jnp.concatenate([photon_directions, padding_directions], axis=0)
                else:
                    pion_data['photon_directions'] = photon_directions
                    
                pion_data['N'] = N
                
                # Generate a new random direction for the pion
                pion_dir_key, master_key = jax.random.split(master_key)
                pion_direction = generate_random_direction(pion_dir_key)
                
                # Create parameters tuple for pion - use same vertex but different direction
                pion_params = (
                    pion_data['energy'],
                    shared_vertex,    # Same vertex as muon
                    pion_direction,   # Different random direction
                    initial_intensity,
                    scatter_length,
                    reflection_rate,
                    absorption_length,
                    temperature
                )
                
                # Get a new key for the pion simulation
                pion_sim_key, master_key = jax.random.split(master_key)
                
                # Run simulation for pion
                pion_charges, pion_times = event_simulator(pion_params, pion_sim_key, pion_data)
                
                # Store pion data
                all_charges.append(pion_charges)
                all_times.append(pion_times)
                all_energies.append(pion_data['energy'])
                all_directions.append(pion_direction)
                all_indices.append(random_idx)  # Store the random index
            
            # Combine all rings using the superimpose_multiple_events function
            # (We still compute this for possible backward compatibility)
            if n_rings > 1:
                combined_charges, combined_times = superimpose_multiple_events(all_charges, all_times)
            else:
                # If only one ring, use the muon data directly
                combined_charges, combined_times = all_charges[0], all_times[0]
            
            # Create filename with sequential numbering (event_0.h5, event_1.h5, etc.)
            event_number = i - start_idx + batch_idx * batch_size
            filename = os.path.join(output_dir, f'event_{event_number}.h5')
            
            # Store original indices in extended_info for reference
            particle_indices = []
            particle_indices.append(all_indices[0])  # Store the muon index
            for ring_idx in range(1, n_rings):
                particle_indices.append(all_indices[ring_idx])  # Store pion indices
            
            # Store the first energy and direction for the save parameters
            save_params = (all_energies[0], shared_vertex, all_directions[0])
            
            # Extended save parameters for multi-ring events
            extended_info = {
                'n_rings': n_rings,
                'particle_types': ['muon'] + ['pion'] * (n_rings - 1),
                'energies': all_energies,
                'directions': [dir.tolist() for dir in all_directions],
                'indices': all_indices,
                'vertex': shared_vertex.tolist(),  # Add vertex to extended_info
                'original_indices': particle_indices  # Store original indices for reference
            }
                
            # Store the event data for batch processing
            batch_data.append((all_charges, all_times, extended_info))
            batch_params.append(save_params)
            batch_filenames.append(filename)
            batch_indices.append(event_number)  # Use sequential event number
        
        # Now save all the events in the batch using multithreading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create a list of future objects
            futures = [
                executor.submit(
                    save_single_event_with_extended_info, 
                    data[0], data[1],  # lists of individual charges and times
                    params, 
                    extended_info=data[2],  # extended info
                    event_number=idx, 
                    filename=filename
                )
                for data, params, filename, idx in zip(
                    batch_data, batch_params, batch_filenames, batch_indices
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
    
    print(f"Successfully processed {len(saved_files)} events.")
    print(f"All events saved to {output_dir} with sequential naming (event_0.h5, event_1.h5, ...)")
    return saved_files

def generate_multi_folder_events(event_simulator, root_file_path, folder_names, events_per_folder, 
                               n_rings_list=None, pion_root_file_path=None,
                               max_detectors_per_cell=4, batch_size=100):
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
    max_detectors_per_cell : int, optional
        Maximum detectors per cell, by default 4
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
            max_detectors_per_cell=max_detectors_per_cell,
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
