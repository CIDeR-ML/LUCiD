"""ROOT I/O and event generation functions.

Moved from lucid/generate.py during Phase 2.2 refactor.
Additional I/O and event analysis functions moved from lucid/utils.py
during Phase 2.5 refactor.
"""

import jax
import jax.numpy as jnp
import numpy as np
import h5py
import os
import time
from glob import glob as _glob
from tqdm import tqdm


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


def generate_multi_folder_events(event_simulator, root_file_path, folder_names, events_per_folder,
                               n_rings_list=None, pion_root_file_path=None,
                               sensor_params=None, max_sensors_per_cell=4, batch_size=100):
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
    sensor_params : tuple, optional
        Sensor parameters tuple passed to event_simulator, by default None
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
            sensor_params=sensor_params,
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
    from lucid.detector_params import ParticleParams
    from lucid.sources.calibration_sources import generate_random_direction, generate_random_vertex
    pass  # save_single_event_with_extended_info and merge_event_files are now local
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

                # Create parameters
                track_params = ParticleParams.from_cartesian(
                    energy=particle_energy,
                    position=vertex,
                    direction=direction,
                    t0=0.0,
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
                                             include_track_segments=False, include_voxels=False):
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
    from lucid.detector_params import ParticleParams
    import numpy as np
    import json
    from lucid.utils import smear_charges_SK_like, smear_times
    pass  # save_single_event_with_particle_info and merge_event_files are now local
    from lucid.production.voxelize import VoxelGridConfig, voxelize_from_photon_indices, pack_voxel_data_for_hdf5

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
                track_params = ParticleParams.from_cartesian(
                    energy=track_energy,
                    position=track_pos,
                    direction=track_dir,
                    t0=0.0,
                )

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
            # VOXELIZATION (optional)
            # Convert photon positions to sparse voxel representation
            # ========================================================================
            packed_voxel_data = None
            if include_voxels:
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
                # Track segment data (if included)
                'include_track_segments': include_track_segments
            }

            # Add voxel data if included
            if packed_voxel_data is not None:
                extended_info['voxel_n_nonzero'] = packed_voxel_data['voxel_n_nonzero']
                extended_info['voxel_offsets'] = packed_voxel_data['voxel_offsets']
                extended_info['voxel_flat_indices'] = packed_voxel_data['voxel_flat_indices']
                extended_info['voxel_counts'] = packed_voxel_data['voxel_counts']

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


# ---------------------------------------------------------------------------
# Functions moved from lucid/utils.py during Phase 2.5 refactor
# ---------------------------------------------------------------------------

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

def get_pdg_code(particle_type):
    """
    Convert particle type string to PDG code.

    Parameters
    ----------
    particle_type : str
        Particle type string (e.g., 'mu-', 'mu+', 'e-', 'e+', 'pi-', 'pi+', 'pi0')

    Returns
    -------
    int
        PDG code for the particle
    """
    pdg_map = {
        'mu-': 13,
        'mu+': -13,
        'muon': 13,  # backward compatibility
        'e-': 11,
        'e+': -11,
        'electron': 11,
        'positron': -11,
        'pi-': -211,
        'pi+': 211,
        'pi0': 111,
        'pion': 211,  # backward compatibility, assume pi+
        'gamma': 22,
        'photon': 22,
        'proton': 2212,
        'p': 2212,
        'neutron': 2112,
        'n': 2112
    }

    if particle_type in pdg_map:
        return pdg_map[particle_type]
    else:
        raise ValueError(f"Unknown particle type: {particle_type}")

def get_particle_mass(particle_type):
    """
    Get particle rest mass in MeV/c^2.

    Parameters
    ----------
    particle_type : str
        Particle type string (e.g., 'mu-', 'mu+', 'e-', 'e+', 'pi-', 'pi+', 'pi0')

    Returns
    -------
    float
        Rest mass in MeV/c^2
    """
    # Normalize particle type by removing charge for mass lookup
    particle_base = particle_type.replace('-', '').replace('+', '')

    mass_map = {
        'mu': 105.7,      # muon
        'muon': 105.7,
        'e': 0.511,       # electron
        'electron': 0.511,
        'positron': 0.511,
        'pi': 139.6,      # charged pion (pi+ and pi-)
        'pion': 139.6,
        'pi0': 135.0,     # neutral pion
        'gamma': 0.0,
        'photon': 0.0,
        'proton': 938.3,
        'p': 938.3,
        'neutron': 939.6,
        'n': 939.6
    }

    if particle_base in mass_map:
        return mass_map[particle_base]
    elif particle_type == 'pi0':  # special case for pi0
        return mass_map['pi0']
    else:
        raise ValueError(f"Unknown particle type: {particle_type}")

def save_single_event_with_extended_info(charges, times, params, extended_info=None, event_number=0, filename=None):
    """
    Save a single event to an HDF5 file with the following structure:
    - PDG (shape N, ): The PDG code of each track (particle)
    - Q (shape N, L): The observed charge for each track in each PMT
    - Q_tot (shape N, ): The total observed charge for each track
    - T (shape N, L): The observed time for each track in each PMT
    - P (shape N, 3): The 3D particle momentum
    - V (shape N, 3): The 3D origin of each particle

    Where N is the number of tracks and L is the number of sensors.
    """
    # If no filename is provided, generate one
    if filename is None:
        filename = f'event_{event_number}.h5'

    # Get number of tracks and detectors
    n_tracks = extended_info['n_particles']
    n_detectors = charges[0].shape[0]  # Assuming all charge arrays have the same shape

    # Create PDG array - use standard PDG codes
    pdg_array = jnp.array([get_pdg_code(pt) for pt in extended_info['particle_types']])

    # Create Q array (charge for each track in each PMT)
    q_array = jnp.zeros((n_tracks, n_detectors))
    for i in range(n_tracks):
        q_array = q_array.at[i].set(charges[i])

    # Calculate Q_tot (total observed charge for each track)
    q_tot = jnp.sum(q_array, axis=1)

    # Create T array (time for each track in each PMT) - same shape as Q
    t_array = jnp.zeros((n_tracks, n_detectors))
    for i in range(n_tracks):
        t_array = t_array.at[i].set(times[i])

    # Create momentum array
    p_array = jnp.zeros((n_tracks, 3))
    for i in range(n_tracks):
        energy = extended_info['energies'][i]
        direction = jnp.array(extended_info['directions'][i])

        # For relativistic particles, we need to convert energy to momentum
        mass = get_particle_mass(extended_info['particle_types'][i])

        # Calculate momentum magnitude: |p| = sqrt(E^2 - m^2)
        # E_kinetic = E_total - m, so E_total = E_kinetic + m
        total_energy = energy + mass
        momentum_mag = jnp.sqrt(total_energy**2 - mass**2)

        # Calculate momentum vector
        p_array = p_array.at[i].set(momentum_mag * direction)

    # Create vertex array (same vertex for all tracks)
    vertex = jnp.array(extended_info['vertex'])
    v_array = jnp.tile(vertex, (n_tracks, 1))

    with h5py.File(filename, 'w') as f:
        # Save data in the requested format
        f.create_dataset('PDG', data=pdg_array)  # shape (N,)
        f.create_dataset('Q', data=q_array)      # shape (N, L)
        f.create_dataset('Q_tot', data=q_tot)    # shape (N,)
        f.create_dataset('T', data=t_array)      # shape (N, L)
        f.create_dataset('P', data=p_array)      # shape (N, 3)
        f.create_dataset('V', data=v_array)      # shape (N, 3)

        # Also save event number for reference
        f.create_dataset('event_number', data=event_number)

    return filename

def save_single_event_with_particle_info(extended_info, event_number=0, filename=None):
    """
    Save a single event with categorized particle structure to an HDF5 file.

    This function saves events generated from PhotonSim where photons are classified
    by particle genealogy into categories (Primary, DecayElectron, Gamma, SecondaryPion).

    The HDF5 structure includes:

    Event Metadata:
    - event_number: Event index
    - n_particles: Number of categorized particles
    - t0: Event time offset (ns), sampled U(-15, 15)

    Reconstructed Sensor Data:
    - PE (N_sensors,): Observed photoelectrons
    - T (N_sensors,): Observed first-hit time (ns)

    Sensor Data by Categorized Particle:
    - PE_per_particle (n_particles, N_sensors): True PE per particle per sensor
    - T_per_particle (n_particles, N_sensors): True first-hit time per particle

    Categorized Particles Metadata:
    - Particle_Category (n_particles,): Category ID (0-3)
    - Particle_CategorizedGenealogy: Ancestry chain of categorized particles
    - Particle_TrackGenealogy: Full G4 track ID ancestry (optional)

    Track Information (optional):
    - TrackID, ParentID, PDG, InitialEnergy, NCherenkov, SegmentOffset, NSegments

    Segments (optional):
    - Start/End positions, Direction, Edep, Time

    Voxel data (sparse representation):
    - voxel_n_nonzero, voxel_offsets, voxel_flat_indices, voxel_counts

    Parameters
    ----------
    extended_info : dict
        Dictionary containing event information
    event_number : int, optional
        Event number, by default 0
    filename : str, optional
        Output filename, by default None (generates event_{number}.h5)

    Returns
    -------
    str
        Path to saved file
    """

    # If no filename provided, generate one
    if filename is None:
        filename = f'event_{event_number}.h5'

    n_particles = extended_info['n_particles']
    particles = extended_info['particles']
    PE_per_particle = extended_info['PE_per_particle']
    T_per_particle = extended_info['T_per_particle']
    PE = extended_info['PE_reco']  # Observed (smeared) values
    T = extended_info['T_reco']
    t0 = extended_info.get('t0', 0.0)  # Event time offset

    # Extract category and genealogy for each particle
    particle_categories = []
    particle_genealogies = []

    for particle in particles:
        track_info = particle['track_info']
        genealogy = particle['genealogy']

        if track_info is not None:
            particle_categories.append(track_info['category'])
        else:
            particle_categories.append(-1)

        particle_genealogies.append(genealogy)

    # Convert to numpy arrays
    particle_categories = np.array(particle_categories, dtype=np.int32)

    # Ensure all arrays are numpy arrays (converts JAX arrays if needed)
    PE_per_particle = np.asarray(PE_per_particle, dtype=np.float32)
    T_per_particle = np.asarray(T_per_particle, dtype=np.float32)
    PE = np.asarray(PE, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32)

    with h5py.File(filename, 'w') as f:
        # Event metadata
        f.create_dataset('event_number', data=np.int32(event_number))
        f.create_dataset('n_particles', data=np.int32(n_particles))
        f.create_dataset('t0', data=np.float32(t0))

        # Reconstructed sensor data
        f.create_dataset('PE', data=PE)  # (N_sensors,) - observed photoelectrons
        f.create_dataset('T', data=T)    # (N_sensors,) - observed first-hit time

        # Sensor data by categorized particle
        f.create_dataset('PE_per_particle', data=PE_per_particle)  # (n_particles, N_sensors)
        f.create_dataset('T_per_particle', data=T_per_particle)    # (n_particles, N_sensors)

        # Categorized particles metadata
        f.create_dataset('Particle_Category', data=particle_categories)

        # Save genealogies as variable-length array
        vlen_int_dtype = h5py.vlen_dtype(np.dtype('int32'))

        # Pre-convert all genealogies to int32 arrays
        genealogy_arrays = []
        for g in particle_genealogies:
            arr = np.asarray(g, dtype=np.int32)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            genealogy_arrays.append(arr.flatten())

        # Create object array by assignment to preserve int32 dtype of inner arrays
        genealogy_data = np.empty(len(genealogy_arrays), dtype=object)
        for i, arr in enumerate(genealogy_arrays):
            genealogy_data[i] = arr

        f.create_dataset('Particle_CategorizedGenealogy', data=genealogy_data, dtype=vlen_int_dtype)

        # Containment metrics
        f.create_dataset('overall_light_containment', data=np.float64(extended_info['overall_light_containment']))
        f.create_dataset('light_containment_by_particle', data=np.array(extended_info['light_containment_by_particle'], dtype=np.float64))

        # Voxel data (sparse representation)
        if 'voxel_n_nonzero' in extended_info:
            f.create_dataset('voxel_n_nonzero', data=np.asarray(extended_info['voxel_n_nonzero'], dtype=np.int32))
            f.create_dataset('voxel_offsets', data=np.asarray(extended_info['voxel_offsets'], dtype=np.int32))
            f.create_dataset('voxel_flat_indices', data=np.asarray(extended_info['voxel_flat_indices'], dtype=np.int64))
            f.create_dataset('voxel_counts', data=np.asarray(extended_info['voxel_counts'], dtype=np.int32))

        # File attributes
        f.attrs['source'] = extended_info['source']
        f.attrs['n_particles'] = np.int32(n_particles)
        f.attrs['n_sensors'] = np.int32(PE.shape[0])

        # Save track information and segments if included
        if extended_info.get('include_track_segments', False) and 'meaningful_tracks' in extended_info:
            meaningful_tracks = extended_info['meaningful_tracks']
            segments = extended_info['segments']

            # Create TrackInformation group
            tracks_group = f.create_group('TrackInformation')

            # Save track-level arrays
            n_tracks = len(meaningful_tracks)
            tracks_group.attrs['n_tracks'] = np.int32(n_tracks)

            if n_tracks > 0:
                # Build arrays from meaningful_tracks dict
                track_ids = np.array([t['track_id'] for t in meaningful_tracks.values()], dtype=np.int32)
                parent_ids = np.array([t['parent_id'] for t in meaningful_tracks.values()], dtype=np.int32)
                pdgs = np.array([t['pdg'] for t in meaningful_tracks.values()], dtype=np.int32)
                energies = np.array([t['initial_energy'] for t in meaningful_tracks.values()], dtype=np.float32)
                n_cherenkov = np.array([t['n_cherenkov'] for t in meaningful_tracks.values()], dtype=np.int32)
                seg_offsets = np.array([t['segment_offset'] for t in meaningful_tracks.values()], dtype=np.int32)
                n_segs = np.array([t['n_segments'] for t in meaningful_tracks.values()], dtype=np.int32)

                tracks_group.create_dataset('TrackID', data=track_ids)
                tracks_group.create_dataset('ParentID', data=parent_ids)
                tracks_group.create_dataset('PDG', data=pdgs)
                tracks_group.create_dataset('InitialEnergy', data=energies)
                tracks_group.create_dataset('NCherenkov', data=n_cherenkov)
                tracks_group.create_dataset('SegmentOffset', data=seg_offsets)
                tracks_group.create_dataset('NSegments', data=n_segs)

            # Create Segments group
            segments_group = f.create_group('Segments')
            n_segments = segments['n_segments']
            segments_group.attrs['n_segments'] = np.int32(n_segments)

            if n_segments > 0:
                # Save segment arrays (positions in cm)
                segments_group.create_dataset('StartX', data=np.asarray(segments['start_x'], dtype=np.float32))
                segments_group.create_dataset('StartY', data=np.asarray(segments['start_y'], dtype=np.float32))
                segments_group.create_dataset('StartZ', data=np.asarray(segments['start_z'], dtype=np.float32))
                segments_group.create_dataset('EndX', data=np.asarray(segments['end_x'], dtype=np.float32))
                segments_group.create_dataset('EndY', data=np.asarray(segments['end_y'], dtype=np.float32))
                segments_group.create_dataset('EndZ', data=np.asarray(segments['end_z'], dtype=np.float32))
                segments_group.create_dataset('DirX', data=np.asarray(segments['dir_x'], dtype=np.float32))
                segments_group.create_dataset('DirY', data=np.asarray(segments['dir_y'], dtype=np.float32))
                segments_group.create_dataset('DirZ', data=np.asarray(segments['dir_z'], dtype=np.float32))
                segments_group.create_dataset('Edep', data=np.asarray(segments['edep'], dtype=np.float32))
                segments_group.create_dataset('Time', data=np.asarray(segments['time'], dtype=np.float32))

            # Save track genealogy for each particle if available
            ext_genealogies = []
            for particle in particles:
                if 'extended_genealogy' in particle and particle['extended_genealogy'] is not None:
                    ext_genealogies.append(np.asarray(particle['extended_genealogy'], dtype=np.int32))
                else:
                    ext_genealogies.append(np.array([], dtype=np.int32))

            if ext_genealogies:
                ext_gen_data = np.empty(len(ext_genealogies), dtype=object)
                for i, arr in enumerate(ext_genealogies):
                    ext_gen_data[i] = arr
                f.create_dataset('Particle_TrackGenealogy', data=ext_gen_data, dtype=vlen_int_dtype)

    return filename

def merge_event_files(output_dir, merged_filename='merged_events.h5', remove_individuals=True):
    """
    Merge individual event HDF5 files into a single merged file.

    Events are stored in groups: /event_0/, /event_1/, etc.
    Each group contains: PDG, Q, Q_tot, T, P, V, event_number

    Parameters
    ----------
    output_dir : str
        Directory containing individual event files (event_0.h5, event_1.h5, etc.)
    merged_filename : str, optional
        Name of the merged output file, by default 'merged_events.h5'
    remove_individuals : bool, optional
        Whether to remove individual event files after merging, by default True

    Returns
    -------
    str
        Path to the merged file
    """
    import glob

    # Find all event files in the directory
    event_files = sorted(glob.glob(os.path.join(output_dir, 'event_*.h5')))

    if not event_files:
        print(f"No event files found in {output_dir}")
        return None

    print(f"Merging {len(event_files)} event files...")

    merged_path = os.path.join(output_dir, merged_filename)

    # Create merged file
    with h5py.File(merged_path, 'w') as merged_file:
        # Store number of events as an attribute
        merged_file.attrs['n_events'] = len(event_files)

        # Process each event file
        for event_file in tqdm(event_files, desc="Merging events", unit="file"):
            # Extract event number from filename
            event_name = os.path.basename(event_file)
            event_num = int(event_name.replace('event_', '').replace('.h5', ''))

            # Read the individual event file
            with h5py.File(event_file, 'r') as f:
                # Create a group for this event
                event_group = merged_file.create_group(f'event_{event_num}')

                # Recursively copy all datasets and groups from individual file
                def copy_item(src, dst, name):
                    """Recursively copy HDF5 items (datasets and groups)."""
                    item = src[name]
                    if isinstance(item, h5py.Dataset):
                        dst.create_dataset(name, data=item[()])
                    elif isinstance(item, h5py.Group):
                        grp = dst.create_group(name)
                        for subkey in item.keys():
                            copy_item(item, grp, subkey)

                for key in f.keys():
                    copy_item(f, event_group, key)

    print(f"Successfully merged events into: {merged_path}")

    # Remove individual files if requested
    if remove_individuals:
        print("Removing individual event files...")
        for event_file in tqdm(event_files, desc="Removing files", unit="file"):
            os.remove(event_file)
        print(f"Removed {len(event_files)} individual event files")

    return merged_path

def read_multi_folder_events(folder_names, max_files_per_folder=None, summary_only=True):
    """
    Read events from multiple folders.

    Parameters
    ----------
    folder_names : list of str
        List of folder names containing event files
    max_files_per_folder : int, optional
        Maximum number of files to read per folder, by default None (all files)
    summary_only : bool, optional
        Whether to print only summary statistics and not individual files, by default True

    Returns
    -------
    dict
        Dictionary mapping folder names to lists of data dictionaries
    """
    results = {}

    total_events = 0
    total_tracks = 0
    total_muons = 0
    total_pions = 0

    print(f"\nReading events from {len(folder_names)} folders:")
    for folder_idx, folder_name in enumerate(folder_names):
        print(f"\n{'-'*50}")
        print(f"Folder {folder_idx+1}/{len(folder_names)}: {folder_name}")
        print(f"{'-'*50}")

        data_list = analyze_event_directory(
            directory=folder_name,
            pattern="event_*.h5",
            max_files=max_files_per_folder,
            summary_only=summary_only
        )

        results[folder_name] = data_list

        # Accumulate statistics
        folder_tracks = sum(data['PDG'].shape[0] for data in data_list)
        folder_muons = sum(np.sum(data['PDG'] == 13) for data in data_list)
        folder_pions = sum(np.sum(data['PDG'] == 211) for data in data_list)

        total_events += len(data_list)
        total_tracks += folder_tracks
        total_muons += folder_muons
        total_pions += folder_pions

    # Print overall summary
    print("\n" + "="*60)
    print(f"Overall Summary for {len(folder_names)} Folders")
    print("="*60)
    print(f"Total events: {total_events}")
    print(f"Total tracks: {total_tracks}")
    print(f"Total muons: {total_muons} ({total_muons/total_tracks*100:.1f}%)")
    print(f"Total pions: {total_pions} ({total_pions/total_tracks*100:.1f}%)")

    # Print folder comparison
    print("\nFolder Comparison:")
    print("-" * 80)
    print(f"{'Folder':<20}{'Events':<10}{'Tracks':<10}{'Muons':<10}{'Pions':<10}")
    print("-" * 80)

    for folder_name, data_list in results.items():
        folder_tracks = sum(data['PDG'].shape[0] for data in data_list)
        folder_muons = sum(np.sum(data['PDG'] == 13) for data in data_list)
        folder_pions = sum(np.sum(data['PDG'] == 211) for data in data_list)

        print(f"{folder_name:<20}{len(data_list):<10}{folder_tracks:<10}{folder_muons:<10}{folder_pions:<10}")

    return results

def read_event_file(filename, verbose=True):
    """
    Read an event file in the new format and print its contents.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file
    verbose : bool, optional
        Whether to print detailed information, by default True

    Returns
    -------
    dict
        Dictionary containing the event data
    """
    with h5py.File(filename, 'r') as f:
        # Read all datasets
        pdg = np.array(f['PDG'])
        q = np.array(f['Q'])
        q_tot = np.array(f['Q_tot'])
        t = np.array(f['T'])
        p = np.array(f['P'])
        v = np.array(f['V'])

        # Check if event_number is present
        event_number = np.array(f['event_number']) if 'event_number' in f else None

        data = {
            'PDG': pdg,
            'Q': q,
            'Q_tot': q_tot,
            'T': t,
            'P': p,
            'V': v,
            'event_number': event_number,
            'filename': filename
        }

    # Print information if verbose
    if verbose:
        print(f"\n{'='*50}")
        print(f"File: {os.path.basename(filename)}")
        if event_number is not None:
            print(f"Event Number: {event_number}")
        print(f"{'='*50}")

        n_tracks = pdg.shape[0]
        n_detectors = q.shape[1]

        print(f"Number of tracks: {n_tracks}")
        print(f"Number of detectors: {n_detectors}")
        print(f"\nParticle Information:")
        print("-" * 80)
        print(f"{'Track #':<8}{'PDG':<8}{'Q_tot':<12}{'P_mag (MeV/c)':<16}{'Direction':<25}{'Vertex':<25}")
        print("-" * 80)

        for i in range(n_tracks):
            # Convert PDG code to particle name
            particle = "Muon" if pdg[i] == 13 else "Pion" if pdg[i] == 211 else f"Unknown ({pdg[i]})"

            # Calculate momentum magnitude
            p_mag = np.sqrt(np.sum(p[i]**2))

            # Normalize direction
            direction = p[i] / (p_mag if p_mag > 0 else 1)

            print(f"{i:<8}{particle:<8}{q_tot[i]:<12.2f}{p_mag:<16.2f}{str(direction):<25}{str(v[i]):<25}")

        print("\nDetector Statistics:")
        print(f"Total charge detected: {np.sum(q_tot):.2f}")
        print(f"Mean charge per track: {np.mean(q_tot):.2f}")
        print(f"Mean charge per PMT: {np.mean(np.sum(q, axis=0)):.2f}")
        print(f"Number of PMTs with signal: {np.sum(np.sum(q, axis=0) > 0)}")

        # Print Q values for each track
        print("\nCharge Matrix (Q) - First 10 PMTs:")
        print("-" * 80)
        header = "Track #  "
        for j in range(min(10, n_detectors)):
            header += f"PMT-{j:<5} "
        print(header)
        print("-" * 80)

        for i in range(n_tracks):
            row = f"{i:<8}  "
            for j in range(min(10, n_detectors)):
                row += f"{q[i,j]:<7.2f} "
            row += f"... (showing 10/{n_detectors} PMTs)"
            print(row)

        # Print timing information
        print("\nTiming Information:")
        # T is now shape (N, L) like Q
        valid_times = t[t > 0]
        if valid_times.size > 0:
            print(f"Mean detection time: {np.mean(valid_times):.2f} ns")
            print(f"Min detection time: {np.min(valid_times):.2f} ns")
            print(f"Max detection time: {np.max(valid_times):.2f} ns")
        else:
            print("No valid timing data available")

        # Print T values for each track (similar to Q matrix)
        print("\nTime Matrix (T) - First 10 PMTs:")
        print("-" * 80)
        header = "Track #  "
        for j in range(min(10, n_detectors)):
            header += f"PMT-{j:<5} "
        print(header)
        print("-" * 80)

        for i in range(n_tracks):
            row = f"{i:<8}  "
            for j in range(min(10, n_detectors)):
                if t[i,j] > 0:
                    row += f"{t[i,j]:<7.2f} "
                else:
                    row += f"{'--':<7} "
            row += f"... (showing 10/{n_detectors} PMTs)"
            print(row)

    return data

def extract_particle_properties(momentum, pdg_code):
    """
    Extract theta, phi angles and energy from particle momentum.

    Parameters
    ----------
    momentum : array_like
        3D momentum vector [px, py, pz] in MeV/c
    pdg_code : int
        PDG particle code (13 for muon, 211 for pion, etc.)

    Returns
    -------
    tuple
        (theta, phi, kinetic_energy) where:
        - theta: polar angle from z-axis in radians
        - phi: azimuthal angle in xy-plane in radians
        - kinetic_energy: kinetic energy in MeV
    """
    px, py, pz = momentum

    # Calculate momentum magnitude
    p_mag = np.sqrt(px**2 + py**2 + pz**2)

    # Calculate angles
    theta = np.arccos(pz / p_mag) if p_mag > 0 else 0.0  # polar angle from z-axis
    phi = np.arctan2(py, px)  # azimuthal angle in xy-plane

    # Get particle mass based on PDG code
    if pdg_code == 13 or pdg_code == -13:  # muon/antimuon
        mass = 105.7  # MeV/c^2
    elif pdg_code == 211 or pdg_code == -211:  # charged pion
        mass = 139.6  # MeV/c^2
    elif pdg_code == 11 or pdg_code == -11:  # electron/positron
        mass = 0.511  # MeV/c^2
    else:
        # Default to muon mass for unknown particles
        mass = 105.7
        print(f"Warning: Unknown PDG code {pdg_code}, using muon mass")

    # Calculate total energy: E^2 = p^2 + m^2
    total_energy = np.sqrt(p_mag**2 + mass**2)

    # Kinetic energy = Total energy - rest mass
    kinetic_energy = total_energy - mass

    return theta, phi, kinetic_energy

def analyze_loaded_particle(loaded_mom, loaded_vtx, pdg_code):
    """
    Analyze particle properties from loaded HDF5 data.

    Parameters
    ----------
    loaded_mom : array_like
        3D momentum vector [px, py, pz] in MeV/c
    loaded_vtx : array_like
        3D vertex position [x, y, z] in meters
    pdg_code : int
        PDG particle code

    Returns
    -------
    dict
        Dictionary containing particle properties
    """
    theta, phi, kinetic_energy = extract_particle_properties(loaded_mom, pdg_code)

    # Convert angles to degrees for easier interpretation
    theta_deg = np.degrees(theta)
    phi_deg = np.degrees(phi)

    # Calculate momentum magnitude
    p_mag = np.sqrt(np.sum(loaded_mom**2))

    # Particle type name
    particle_names = {13: 'muon', -13: 'antimuon', 211: 'pion+', -211: 'pion-',
                     11: 'electron', -11: 'positron'}
    particle_name = particle_names.get(pdg_code, f'unknown (PDG={pdg_code})')

    return {
        'particle_type': particle_name,
        'pdg_code': pdg_code,
        'momentum_magnitude': p_mag,
        'momentum_vector': loaded_mom,
        'theta_rad': theta,
        'phi_rad': phi,
        'theta_deg': theta_deg,
        'phi_deg': phi_deg,
        'kinetic_energy': kinetic_energy,
        'vertex': loaded_vtx,
        'direction': loaded_mom / p_mag if p_mag > 0 else np.array([0, 0, 1])
    }

def analyze_event_directory(directory, pattern="*.h5", max_files=None, summary_only=False):
    """
    Analyze multiple event files in a directory.

    Parameters
    ----------
    directory : str
        Directory containing HDF5 event files
    pattern : str, optional
        File pattern to match, by default "*.h5"
    max_files : int, optional
        Maximum number of files to analyze, by default None (all files)
    summary_only : bool, optional
        Whether to print only summary statistics and not individual files, by default False

    Returns
    -------
    list of dict
        List of data dictionaries for each event
    """
    # Find all files matching the pattern
    file_paths = _glob(os.path.join(directory, pattern))

    if max_files is not None:
        file_paths = file_paths[:max_files]

    print(f"Found {len(file_paths)} files to analyze")

    # Read all files
    all_data = []
    for file_path in file_paths:
        data = read_event_file(file_path, verbose=not summary_only)
        all_data.append(data)

    # Calculate summary statistics
    total_tracks = sum(data['PDG'].shape[0] for data in all_data)
    muon_count = sum(np.sum(data['PDG'] == 13) for data in all_data)
    pion_count = sum(np.sum(data['PDG'] == 211) for data in all_data)

    # Print summary
    print("\n" + "="*60)
    print(f"Summary Statistics for {len(file_paths)} Events")
    print("="*60)
    print(f"Total number of tracks: {total_tracks}")
    print(f"Total muons: {muon_count} ({muon_count/total_tracks*100:.1f}%)")
    print(f"Total pions: {pion_count} ({pion_count/total_tracks*100:.1f}%)")

    # Calculate charge statistics
    all_q_tot = np.concatenate([data['Q_tot'] for data in all_data])
    print(f"\nCharge Statistics:")
    print(f"Mean charge per track: {np.mean(all_q_tot):.2f}")
    print(f"Min charge: {np.min(all_q_tot):.2f}")
    print(f"Max charge: {np.max(all_q_tot):.2f}")

    # Calculate momentum statistics
    all_p_mag = np.concatenate([
        np.sqrt(np.sum(data['P']**2, axis=1)) for data in all_data
    ])
    print(f"\nMomentum Statistics:")
    print(f"Mean momentum magnitude: {np.mean(all_p_mag):.2f} MeV/c")
    print(f"Min momentum: {np.min(all_p_mag):.2f} MeV/c")
    print(f"Max momentum: {np.max(all_p_mag):.2f} MeV/c")

    # PMT statistics across all events
    if all_data:
        n_detectors = all_data[0]['Q'].shape[1]
        all_pmt_charges = np.zeros(n_detectors)

        for data in all_data:
            all_pmt_charges += np.sum(data['Q'], axis=0)

        active_pmts = np.where(all_pmt_charges > 0)[0]
        print(f"\nPMT Statistics Across All Events:")
        print(f"Number of active PMTs: {len(active_pmts)} / {n_detectors}")
        print(f"Mean charge per active PMT: {np.mean(all_pmt_charges[active_pmts]):.2f}")


    return all_data


# Particle physics constants (rest masses in MeV/c^2)
PARTICLE_MASSES = {
    13: 105.7,   # muon
    -13: 105.7,  # anti-muon
    211: 139.6,  # charged pion
    -211: 139.6, # negative pion
    111: 134.98, # neutral pion
    11: 0.511,   # electron
    -11: 0.511,  # positron
    22: 0.0,     # photon
    2212: 938.3, # proton
    2112: 939.6, # neutron
}

def momentum_to_angles_and_energy(momentum_vector, pdg_code):
    """
    Extract theta, phi angles and kinetic energy from particle momentum vector.

    Parameters
    ----------
    momentum_vector : jnp.ndarray
        3D momentum vector [px, py, pz] in MeV/c
    pdg_code : int
        PDG particle code (13 for muon, 211 for pion, etc.)

    Returns
    -------
    tuple
        (theta, phi, kinetic_energy) where:
        - theta: polar angle from z-axis in radians [0, pi]
        - phi: azimuthal angle in xy-plane in radians [0, 2*pi]
        - kinetic_energy: kinetic energy in MeV

    Notes
    -----
    - theta = 0 corresponds to positive z-direction
    - phi = 0 corresponds to positive x-direction
    - Uses relativistic energy-momentum relation: E^2 = p^2 + m^2
    - Kinetic energy = Total energy - Rest mass
    """
    # Get particle mass
    if pdg_code not in PARTICLE_MASSES:
        raise ValueError(f"Unknown PDG code: {pdg_code}. Supported codes: {list(PARTICLE_MASSES.keys())}")

    mass = PARTICLE_MASSES[pdg_code]

    # Extract momentum components
    px, py, pz = momentum_vector[0], momentum_vector[1], momentum_vector[2]

    # Calculate momentum magnitude
    p_magnitude = jnp.sqrt(px**2 + py**2 + pz**2)

    # Calculate polar angle theta (angle from z-axis)
    # theta = arccos(pz / |p|)
    theta = jnp.arccos(jnp.clip(pz / p_magnitude, -1.0, 1.0))

    # Calculate azimuthal angle phi (angle in xy-plane from x-axis)
    # phi = arctan2(py, px), adjusted to [0, 2*pi] range
    phi = jnp.arctan2(py, px)
    phi = jnp.where(phi < 0, phi + 2*jnp.pi, phi)  # Ensure phi is in [0, 2*pi]

    # Calculate total energy using relativistic energy-momentum relation
    # E^2 = p^2 + m^2
    total_energy = jnp.sqrt(p_magnitude**2 + mass**2)

    # Calculate kinetic energy
    kinetic_energy = total_energy - mass

    return theta, phi, kinetic_energy


def analyze_event_kinematics(event_data):
    """
    Wrapper function to analyze kinematics for all tracks in an event.

    Parameters
    ----------
    event_data : dict
        Event data dictionary containing 'P' (momentum) and 'PDG' arrays
        Expected format from read_event_file():
        - 'P': shape (N, 3) momentum vectors in MeV/c
        - 'PDG': shape (N,) PDG particle codes

    Returns
    -------
    dict
        Dictionary containing kinematic analysis results:
        - 'theta': polar angles in radians, shape (N,)
        - 'phi': azimuthal angles in radians, shape (N,)
        - 'kinetic_energy': kinetic energies in MeV, shape (N,)
        - 'momentum_magnitude': momentum magnitudes in MeV/c, shape (N,)
        - 'particle_types': list of particle type strings
        - 'n_tracks': number of tracks

    Example
    -------
    >>> # Load event data
    >>> event_data = read_event_file('event_0.h5')
    >>> # Analyze kinematics
    >>> kinematics = analyze_event_kinematics(event_data)
    >>> print(f"Track 0: theta={kinematics['theta'][0]:.3f} rad, "
    ...       f"phi={kinematics['phi'][0]:.3f} rad, "
    ...       f"KE={kinematics['kinetic_energy'][0]:.1f} MeV")
    """
    if 'P' not in event_data or 'PDG' not in event_data:
        raise ValueError("Event data must contain 'P' (momentum) and 'PDG' arrays")

    momentum_array = jnp.array(event_data['P'])  # Shape: (N, 3)
    pdg_array = jnp.array(event_data['PDG'])     # Shape: (N,)

    n_tracks = momentum_array.shape[0]

    # Initialize output arrays
    theta_array = jnp.zeros(n_tracks)
    phi_array = jnp.zeros(n_tracks)
    kinetic_energy_array = jnp.zeros(n_tracks)
    momentum_magnitude_array = jnp.zeros(n_tracks)

    # Process each track
    for i in range(n_tracks):
        theta, phi, kinetic_energy = momentum_to_angles_and_energy(
            momentum_array[i], int(pdg_array[i])
        )

        theta_array = theta_array.at[i].set(theta)
        phi_array = phi_array.at[i].set(phi)
        kinetic_energy_array = kinetic_energy_array.at[i].set(kinetic_energy)
        momentum_magnitude_array = momentum_magnitude_array.at[i].set(
            jnp.sqrt(jnp.sum(momentum_array[i]**2))
        )

    # Convert PDG codes to particle type strings
    particle_types = []
    for pdg in pdg_array:
        if pdg == 13:
            particle_types.append("muon")
        elif pdg == -13:
            particle_types.append("anti-muon")
        elif pdg == 211:
            particle_types.append("pi+")
        elif pdg == -211:
            particle_types.append("pi-")
        elif pdg == 111:
            particle_types.append("pi0")
        elif pdg == 11:
            particle_types.append("electron")
        elif pdg == -11:
            particle_types.append("positron")
        elif pdg == 22:
            particle_types.append("photon")
        elif pdg == 2212:
            particle_types.append("proton")
        elif pdg == 2112:
            particle_types.append("neutron")
        else:
            particle_types.append(f"unknown_{pdg}")

    return {
        'theta': theta_array,
        'phi': phi_array,
        'kinetic_energy': kinetic_energy_array,
        'momentum_magnitude': momentum_magnitude_array,
        'particle_types': particle_types,
        'n_tracks': n_tracks
    }


def print_event_kinematics(event_data, show_details=True):
    """
    Print kinematic analysis results for an event in a formatted way.

    Parameters
    ----------
    event_data : dict
        Event data dictionary containing 'P' and 'PDG' arrays
    show_details : bool, optional
        Whether to show detailed information for each track, by default True
    """
    kinematics = analyze_event_kinematics(event_data)

    print("\n" + "="*70)
    print("KINEMATIC ANALYSIS")
    print("="*70)
    print(f"Number of tracks: {kinematics['n_tracks']}")

    if show_details:
        print("\nTrack Details:")
        print("-" * 95)
        print(f"{'Track':<6}{'Particle':<12}{'P_mag':<12}{'KE':<12}{'Theta':<12}{'Phi':<12}{'Direction':<25}")
        print(f"{'#':<6}{'Type':<12}{'(MeV/c)':<12}{'(MeV)':<12}{'(rad)':<12}{'(rad)':<12}{'(unit vector)':<25}")
        print("-" * 95)

        for i in range(kinematics['n_tracks']):
            # Calculate unit direction vector
            theta = kinematics['theta'][i]
            phi = kinematics['phi'][i]
            direction = jnp.array([
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta)
            ])

            print(f"{i:<6}{kinematics['particle_types'][i]:<12}"
                  f"{kinematics['momentum_magnitude'][i]:<12.1f}"
                  f"{kinematics['kinetic_energy'][i]:<12.1f}"
                  f"{theta:<12.3f}"
                  f"{phi:<12.3f}"
                  f"[{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]")

    # Summary statistics
    print(f"\nSummary Statistics:")
    print(f"Mean kinetic energy: {jnp.mean(kinematics['kinetic_energy']):.1f} MeV")
    print(f"Mean momentum magnitude: {jnp.mean(kinematics['momentum_magnitude']):.1f} MeV/c")
    print(f"Theta range: {jnp.min(kinematics['theta']):.3f} - {jnp.max(kinematics['theta']):.3f} rad")
    print(f"Phi range: {jnp.min(kinematics['phi']):.3f} - {jnp.max(kinematics['phi']):.3f} rad")

    # Particle type distribution
    from collections import Counter
    particle_counts = Counter(kinematics['particle_types'])
    print(f"\nParticle Distribution:")
    for particle_type, count in particle_counts.items():
        print(f"  {particle_type}: {count}")

    print("="*70)
