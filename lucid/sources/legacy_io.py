"""Legacy single-event HDF5 I/O and old ROOT I/O (v_photon tree).

Contains ``save_single_event``, ``load_single_event``,
``full_to_sparse``, ``sparse_to_full``, ``get_random_root_entry_index``,
``read_photon_data_from_root``, and ``generate_events_from_root``.
"""
from __future__ import annotations

import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


def full_to_sparse(charges, times):
    """Convert full arrays to sparse representation by removing zero elements.

    Parameters
    ----------
    charges : jnp.ndarray
        Array of charge values for all sensors
    times : jnp.ndarray
        Array of time values for all sensors

    Returns
    -------
    non_zero_indices : jnp.ndarray
        Indices where charges are non-zero
    non_zero_charges : jnp.ndarray
        Charge values at non-zero locations
    non_zero_times : jnp.ndarray
        Time values at non-zero locations
    """
    non_zero_indices = jnp.nonzero(charges)[0]
    non_zero_charges = charges[non_zero_indices]
    non_zero_times = times[non_zero_indices]
    return non_zero_indices, non_zero_charges, non_zero_times


def sparse_to_full(sparse_indices, sparse_values, full_size):
    """Convert sparse representation back to full array with zeros.

    Parameters
    ----------
    sparse_indices : jnp.ndarray
        Indices of non-zero elements
    sparse_values : jnp.ndarray
        Values at the non-zero indices
    full_size : int
        Size of the output array

    Returns
    -------
    jnp.ndarray
        Full array with sparse values inserted at specified indices
    """
    full_data = jnp.zeros(full_size)
    return full_data.at[sparse_indices].set(sparse_values)


def save_single_event(event_data, particle_params, sensor_params, event_number=0, filename=None, calibration_mode=False):
    """Save single event simulation data to an HDF5 file in sparse format.

    Parameters
    ----------
    event_data : tuple
        (charges, average_times) arrays for the event
    particle_params : ParticleParams or IsotropicSource
        if calibration_mode is True: IsotropicSource with position, intensity
        if calibration_mode is False: ParticleParams with energy, position, theta, phi, t0
    sensor_params : DetectorParams
        DetectorParams NamedTuple with all detector calibration fields
    event_number : int, optional
        Event identifier number, defaults to 0
    filename : str, optional
        Custom path to output HDF5 file. If None, auto-generates name
        in 'events' folder as 'event_X.h5' or 'event_X_TIMESTAMP.h5'

    Returns
    -------
    str
        Path to the saved file

    Notes
    -----
    Saves data in a hierarchical structure with two groups:
    - params: contains simulation parameters
    - event: contains sparse event data (indices, charges, times)
    """
    charges, average_times = event_data
    indices, sparse_charges, sparse_times = full_to_sparse(charges, average_times)

    # Generate filename if not provided
    if filename is None:
        from datetime import datetime

        # Create events directory if it doesn't exist
        os.makedirs('events', exist_ok=True)

        base_filename = os.path.join('events', f'event_{event_number}.h5')

        # If file exists, add timestamp
        if os.path.exists(base_filename):
            timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            filename = os.path.join('events', f'event_{event_number}_{timestamp}.h5')
        else:
            filename = base_filename

    with h5py.File(filename, 'w') as f:
        # Save simulation parameters
        if calibration_mode:
            params_group = f.create_group('calibration_params')
            params_group.create_dataset('source_position', data=np.array(particle_params.position))
            params_group.create_dataset('source_intensity', data=np.array(particle_params.intensity))

        else:
            params_group = f.create_group('particle_params')
            params_group.create_dataset('track_energy', data=np.array(particle_params.energy))
            params_group.create_dataset('track_origin', data=np.array(particle_params.position))
            params_group.create_dataset('track_direction', data=np.array(particle_params.direction))

        detector_group = f.create_group('sensor_params')
        detector_group.create_dataset('scatter_length', data=np.array(sensor_params.scatter_length))
        detector_group.create_dataset('wall_reflection_rate', data=np.array(sensor_params.wall_reflection_rate))
        detector_group.create_dataset('sensor_reflection_rate', data=np.array(sensor_params.sensor_reflection_rate))
        detector_group.create_dataset('absorption_length', data=np.array(sensor_params.absorption_length))
        detector_group.create_dataset('qe', data=np.array(sensor_params.qe))
        detector_group.create_dataset('qe_corrections', data=np.array(sensor_params.qe_corrections))

        # Save event data and number
        event_group = f.create_group('event')
        event_group.create_dataset('event_number', data=np.array(event_number))
        event_group.create_dataset('indices', data=np.array(indices))
        event_group.create_dataset('charges', data=np.array(sparse_charges))
        event_group.create_dataset('times', data=np.array(sparse_times))

    return filename


def load_single_event(filename, num_sensors, sparse=True, calibration_mode=False):
    """Load single event simulation data from an HDF5 file.

    Parameters
    ----------
    filename : str
        Path to HDF5 file
    num_sensors : int
        Total number of sensors (needed for dense format)
    sparse : bool, default=True
        If True, returns data in sparse format
        If False, converts to dense arrays
    calibration_mode : bool, default=False
        If True, loads calibration parameters instead of particle parameters

    Returns
    -------
    particle_params : ParticleParams or IsotropicSource
        if calibration_mode is True: IsotropicSource
        if calibration_mode is False: ParticleParams
    sensor_params : DetectorParams
        DetectorParams NamedTuple
    If sparse=True:
        indices, charges, times
    If sparse=False:
        charges, times (dense)
    """
    from lucid.detector_params import ParticleParams, DetectorParams, isotropic_source

    with h5py.File(filename, 'r') as f:
        if calibration_mode:
            params_group = f['calibration_params']
            source_position = jnp.array(params_group['source_position'][()])
            source_intensity = jnp.array(params_group['source_intensity'][()])
            particle_params = isotropic_source(position=source_position, intensity=source_intensity)
        else:
            params_group = f['particle_params']
            track_energy = jnp.array(params_group['track_energy'][()])
            track_origin = jnp.array(params_group['track_origin'][()])
            track_direction = jnp.array(params_group['track_direction'][()])
            particle_params = ParticleParams.from_cartesian(
                energy=track_energy, position=track_origin,
                direction=track_direction, t0=jnp.array(0.0))

        # Load detector parameters
        detector_group = f['sensor_params']
        sensor_params = DetectorParams(
            scatter_length=jnp.array(detector_group['scatter_length'][()]),
            wall_reflection_rate=jnp.array(detector_group['wall_reflection_rate'][()]),
            sensor_reflection_rate=jnp.array(detector_group['sensor_reflection_rate'][()]),
            absorption_length=jnp.array(detector_group['absorption_length'][()]),
            qe=jnp.array(detector_group['qe'][()]),
            qe_corrections=jnp.array(detector_group['qe_corrections'][()]),
        )

        # Load event data
        event_group = f['event']
        event_number = int(event_group['event_number'][()])
        indices = jnp.array(event_group['indices'][()])
        charges = jnp.array(event_group['charges'][()])
        times = jnp.array(event_group['times'][()])

    if sparse:
        return particle_params, sensor_params, indices, charges, times
    else:
        # Convert sparse arrays to full dense arrays
        dense_charges = sparse_to_full(indices, charges, num_sensors)
        dense_times = sparse_to_full(indices, times, num_sensors)

        return particle_params, sensor_params, dense_charges, dense_times


def get_random_root_entry_index(root_file_path):
    """
    Get a random valid entry index from a ROOT file.

    Parameters
    ----------
    root_file_path : str
        Path to the ROOT file

    Returns
    -------
    int
        Random valid entry index
    """
    import uproot

    root_file = uproot.open(root_file_path)
    tree = root_file['v_photon']
    total_entries = tree.num_entries

    return np.random.randint(0, total_entries - 1)

def read_photon_data_from_root(root_file_path, entry_index, particle_type='muon'):
    """
    Read photon data from a ROOT file for a specific entry, using the component vectors.

    Parameters
    ----------
    root_file_path : str
        Path to the ROOT file
    entry_index : int
        Entry index to read from the file
    particle_type : str, optional
        Type of particle ('muon' or 'pion'), by default 'muon'

    Returns
    -------
    dict
        Dictionary containing photon_origins, photon_directions, and energy
    """
    import uproot

    # Open the ROOT file
    root_file = uproot.open(root_file_path)

    # Access the tree
    tree = root_file['v_photon']

    # Read position components
    photon_posx = tree['photon_posx'].array(entry_start=entry_index, entry_stop=entry_index+1)[0]
    photon_posy = tree['photon_posy'].array(entry_start=entry_index, entry_stop=entry_index+1)[0]
    photon_posz = tree['photon_posz'].array(entry_start=entry_index, entry_stop=entry_index+1)[0]

    # Read direction components
    photon_dirx = tree['photon_dirx'].array(entry_start=entry_index, entry_stop=entry_index+1)[0]
    photon_diry = tree['photon_diry'].array(entry_start=entry_index, entry_stop=entry_index+1)[0]
    photon_dirz = tree['photon_dirz'].array(entry_start=entry_index, entry_stop=entry_index+1)[0]

    # Read momentum
    initmom = float(tree['initmom'].array(entry_start=entry_index, entry_stop=entry_index+1)[0])

    # Stack the components to form position and direction arrays
    photon_positions = np.column_stack((photon_posx, photon_posy, photon_posz))
    photon_directions = np.column_stack((photon_dirx, photon_diry, photon_dirz))

    # Convert initmom (momentum) to kinetic energy based on particle type
    if particle_type.lower() == 'muon':
        mass = 105.7  # MeV/c^2 (muon rest mass)
    elif particle_type.lower() == 'pion':
        mass = 139.6  # MeV/c^2 (charged pion rest mass)
    else:
        raise ValueError(f"Unsupported particle type: {particle_type}")

    # E_kinetic = sqrt(p^2 + m^2) - m
    energy = np.sqrt(initmom**2 + mass**2) - mass

    return {
        'photon_origins': jnp.array(photon_positions),     # Combined position vectors
        'photon_directions': jnp.array(photon_directions), # Combined direction vectors
        'energy': float(energy)
    }

def generate_events_from_root(event_simulator, root_file_path, output_dir='events', n_events=None,
                            n_rings=1, pion_root_file_path=None,
                            sensor_params=None, max_sensors_per_cell=4, batch_size=100):
    """
    Generate and save events from a ROOT file, with support for N rings of particles.
    Ring 1 (N=1) is always a muon, and additional rings (N>1) are pions.
    Events are saved with sequential numbering: event_0.h5, event_1.h5, etc.

    Parameters
    ----------
    event_simulator : function
        The event simulation function to use
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
    sensor_params : tuple, optional
        Sensor parameters tuple passed to event_simulator, by default None
    max_sensors_per_cell : int, optional
        Maximum sensors per cell, by default 4
    batch_size : int, optional
        Number of events to accumulate before saving in parallel, by default 100

    Returns
    -------
    list
        List of saved file paths
    """
    import uproot
    import concurrent.futures
    from lucid.sources.calibration_sources import generate_random_direction, generate_random_vertex
    from lucid.utils import superimpose_multiple_events

    # Validate arguments
    if n_rings < 1:
        raise ValueError("n_rings must be at least 1")

    from lucid.detector_params import ParticleParams
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
            all_indices = []

            # Process the first ring - always a muon
            muon_data = read_photon_data_from_root(root_file_path, i, 'muon')

            # Set up parameters
            muon_energy = muon_data['energy']

            # Generate random direction for muon
            dir_key, master_key = jax.random.split(master_key)
            muon_direction = generate_random_direction(dir_key)

            # Create parameters for muon
            track_params = ParticleParams.from_cartesian(
                energy=muon_energy,
                position=shared_vertex,
                direction=muon_direction,
                t0=0.0,
            )

            # Get a key for the muon simulation
            sim_key, master_key = jax.random.split(master_key)

            # Process muon data
            photon_origins = muon_data['photon_origins']
            photon_directions = muon_data['photon_directions']
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
            muon_charges, muon_times = event_simulator(track_params, sensor_params, sim_key, muon_data)

            # Store muon data
            all_charges.append(muon_charges)
            all_times.append(muon_times)
            all_energies.append(muon_energy)
            all_directions.append(muon_direction)
            all_indices.append(i)

            # Process additional rings (pions) if n_rings > 1
            for ring_idx in range(1, n_rings):
                # Get a random entry index from the pion file
                random_idx = get_random_root_entry_index(pion_root_file_path)

                # Read photon data for pion
                pion_data = read_photon_data_from_root(pion_root_file_path, random_idx, 'pion')

                photon_origins = pion_data['photon_origins']
                photon_directions = pion_data['photon_directions']
                N = len(photon_origins)

                padding_size = max(0, 1_000_000-N)

                pion_data['photon_origins'] = jnp.pad(photon_origins, ((0, padding_size), (0, 0)),
                                                     mode='constant', constant_values=0)

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

                # Create parameters for pion
                pion_track_params = ParticleParams.from_cartesian(
                    energy=pion_data['energy'],
                    position=shared_vertex,
                    direction=pion_direction,
                    t0=0.0,
                )

                # Get a new key for the pion simulation
                pion_sim_key, master_key = jax.random.split(master_key)

                # Run simulation for pion
                pion_charges, pion_times = event_simulator(pion_track_params, sensor_params, pion_sim_key, pion_data)

                # Store pion data
                all_charges.append(pion_charges)
                all_times.append(pion_times)
                all_energies.append(pion_data['energy'])
                all_directions.append(pion_direction)
                all_indices.append(random_idx)

            # Combine all rings
            if n_rings > 1:
                combined_charges, combined_times = superimpose_multiple_events(all_charges, all_times)
            else:
                combined_charges, combined_times = all_charges[0], all_times[0]

            # Create filename with sequential numbering
            event_number = i - start_idx + batch_idx * batch_size
            filename = os.path.join(output_dir, f'event_{event_number}.h5')

            # Store original indices in extended_info
            particle_indices = [all_indices[ring_idx] for ring_idx in range(n_rings)]

            save_params = (all_energies[0], shared_vertex, all_directions[0])

            extended_info = {
                'n_rings': n_rings,
                'particle_types': ['muon'] + ['pion'] * (n_rings - 1),
                'energies': all_energies,
                'directions': [dir.tolist() for dir in all_directions],
                'indices': all_indices,
                'vertex': shared_vertex.tolist(),
                'original_indices': particle_indices
            }

            batch_data.append((all_charges, all_times, extended_info))
            batch_params.append(save_params)
            batch_filenames.append(filename)
            batch_indices.append(event_number)

        # Save all events in the batch using multithreading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    save_single_event_with_extended_info,
                    data[0], data[1],
                    params,
                    extended_info=data[2],
                    event_number=idx,
                    filename=filename
                )
                for data, params, filename, idx in zip(
                    batch_data, batch_params, batch_filenames, batch_indices
                )
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

    print(f"Successfully processed {len(saved_files)} events.")
    print(f"All events saved to {output_dir} with sequential naming (event_0.h5, event_1.h5, ...)")
    return saved_files
