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
from lucid.wavelength import DEFAULT_WAVELENGTH_NM


# Tag constants used with jax.random.fold_in so each subprocess stream in
# the seed hierarchy gets a distinct derivation. Value is arbitrary — it
# just has to be stable and distinct across tags.
_SUBPROC_PHOTONSIM_TAG = 0xB107
_SUBPROC_GENIE_TAG     = 0x6E1E


def _resolve_master_seed(master_seed):
    """Return a deterministic int seed, drawing from time if master_seed is None."""
    if master_seed is None:
        return int(time.time() * 1_000_000) % (2 ** 31 - 1)
    return int(master_seed) % (2 ** 31 - 1)


def derive_event_keys(master_seed, job_id, event_idx, interaction_idx=0):
    """Derive independent RNG keys for one (job, event, interaction) step.

    Combines ``master_seed``, ``job_id``, ``event_idx`` and
    ``interaction_idx`` via ``jax.random.fold_in`` so every dimension is
    independent — reusing a CLI seed across jobs no longer collides, and
    pile-up interactions within one event get distinct draws.

    Returns a dict with ``vertex_seed`` / ``t0_seed`` (concrete ints for
    ``np.random.default_rng``) and ``sim_key`` / ``smear_key`` (JAX keys
    to be consumed directly by ``jax.random.*``).
    """
    master_seed = _resolve_master_seed(master_seed)
    base = jax.random.PRNGKey(master_seed)
    job_key = jax.random.fold_in(base, int(job_id))
    event_key = jax.random.fold_in(job_key, int(event_idx))
    interaction_key = jax.random.fold_in(event_key, int(interaction_idx))
    vertex_key, t0_key, sim_key, smear_key = jax.random.split(interaction_key, 4)
    return {
        'vertex_seed': int(jax.random.randint(vertex_key, (), 1, 2**31 - 1)),
        't0_seed':     int(jax.random.randint(t0_key,     (), 1, 2**31 - 1)),
        'sim_key':     sim_key,
        'smear_key':   smear_key,
    }


def derive_subprocess_seeds(master_seed, job_id, vertex_idx=0):
    """Derive deterministic seeds for the per-job subprocesses (GENIE, PhotonSim).

    Subprocess seeds are folded at the (master_seed, job_id, vertex_idx)
    level — not per-event — because each subprocess produces all
    ``n_events`` internally and drives its own per-event RNG. The
    ``vertex_idx`` axis exists so pile-up configurations with N
    PhotonSim/GENIE streams per event get independent seeds per stream.

    PhotonSim's Geant4/CLHEP engine needs two seeds (`/random/setSeeds
    s1 s2`); GENIE's gevgen takes one.
    """
    master_seed = _resolve_master_seed(master_seed)
    base = jax.random.PRNGKey(master_seed)
    job_key = jax.random.fold_in(base, int(job_id))
    vertex_key = jax.random.fold_in(job_key, int(vertex_idx))
    genie_key = jax.random.fold_in(vertex_key, _SUBPROC_GENIE_TAG)
    ps_root = jax.random.fold_in(vertex_key, _SUBPROC_PHOTONSIM_TAG)
    ps_key1, ps_key2 = jax.random.split(ps_root, 2)
    return {
        'genie_seed':      int(jax.random.randint(genie_key, (), 1, 2**31 - 1)),
        'photonsim_seed1': int(jax.random.randint(ps_key1,   (), 1, 2**31 - 1)),
        'photonsim_seed2': int(jax.random.randint(ps_key2,   (), 1, 2**31 - 1)),
    }


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

    # Determine which branches to read
    branches = ['PrimaryEnergy', 'PhotonPosX', 'PhotonPosY', 'PhotonPosZ',
                'PhotonDirX', 'PhotonDirY', 'PhotonDirZ', 'PhotonTime']

    # Read wavelength if available in the ROOT file
    available_branches = tree.keys()
    has_wavelength = 'PhotonWavelength' in available_branches
    if has_wavelength:
        branches.append('PhotonWavelength')

    tree_data = tree.arrays(branches,
                          entry_start=entry_index, entry_stop=entry_index+1, library='np')

    # Extract primary energy (already in MeV)
    energy = float(tree_data['PrimaryEnergy'][0])

    # Extract photon positions (PhotonSim mm → LUCiD m)
    photon_posx = tree_data['PhotonPosX'][0] / 1000.0  # mm to m
    photon_posy = tree_data['PhotonPosY'][0] / 1000.0
    photon_posz = tree_data['PhotonPosZ'][0] / 1000.0

    # Extract photon directions
    photon_dirx = tree_data['PhotonDirX'][0]
    photon_diry = tree_data['PhotonDirY'][0]
    photon_dirz = tree_data['PhotonDirZ'][0]

    # Stack the components to form position and direction arrays
    photon_positions = np.column_stack((photon_posx, photon_posy, photon_posz))
    photon_directions = np.column_stack((photon_dirx, photon_diry, photon_dirz))

    result = {
        'photon_origins': jnp.array(photon_positions),     # Combined position vectors in m
        'photon_directions': jnp.array(photon_directions), # Combined direction vectors
        'photon_times': jnp.array(tree_data['PhotonTime'][0]),
        'energy': energy  # Energy in MeV
    }

    # Include per-photon wavelengths (nm) if available from PhotonSim
    if has_wavelength:
        result['wavelengths'] = jnp.array(tree_data['PhotonWavelength'][0])

    return result

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
        - 'photon_origins': array (N_photons, 3) in m
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
        'PhotonTime', 'PhotonWavelength',
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
            'Segment_Edep', 'Segment_Time',
            'Segment_BetaStart', 'Segment_NCherenkov',
        ])

        available = set(tree.keys())
        required_new = {'Segment_BetaStart', 'Segment_NCherenkov'}
        missing = required_new - available
        if missing:
            raise ValueError(
                f"PhotonSim ROOT file is missing branches {sorted(missing)}. "
                f"Upgrade to PhotonSim branch 'add-per-segment-beta-ncherenkov' "
                f"(commit 1ef5ace or later)."
            )

    tree_data = tree.arrays(branches_to_read, entry_start=entry_index, entry_stop=entry_index+1, library='np')

    # Extract primary energy
    primary_energy = float(tree_data['PrimaryEnergy'][0])

    # Extract photon data (PhotonSim mm → LUCiD m)
    photon_posx = tree_data['PhotonPosX'][0] / 1000.0
    photon_posy = tree_data['PhotonPosY'][0] / 1000.0
    photon_posz = tree_data['PhotonPosZ'][0] / 1000.0
    photon_positions = np.column_stack((photon_posx, photon_posy, photon_posz))

    photon_dirx = tree_data['PhotonDirX'][0]
    photon_diry = tree_data['PhotonDirY'][0]
    photon_dirz = tree_data['PhotonDirZ'][0]
    photon_directions = np.column_stack((photon_dirx, photon_diry, photon_dirz))

    photon_times = tree_data['PhotonTime'][0]
    photon_wavelengths = tree_data['PhotonWavelength'][0]

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
    track_posx = tree_data['TrackInfo_PosX'][0] / 1000.0  # mm to m
    track_posy = tree_data['TrackInfo_PosY'][0] / 1000.0
    track_posz = tree_data['TrackInfo_PosZ'][0] / 1000.0
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
        'photon_wavelengths': photon_wavelengths,  # Keep as NumPy (nm)
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

        # Extract segment arrays (PhotonSim mm → LUCiD m)
        segments = {
            'start_x': tree_data['Segment_StartX'][0] / 1000.0,  # mm to m
            'start_y': tree_data['Segment_StartY'][0] / 1000.0,
            'start_z': tree_data['Segment_StartZ'][0] / 1000.0,
            'end_x': tree_data['Segment_EndX'][0] / 1000.0,
            'end_y': tree_data['Segment_EndY'][0] / 1000.0,
            'end_z': tree_data['Segment_EndZ'][0] / 1000.0,
            'dir_x': tree_data['Segment_DirX'][0],
            'dir_y': tree_data['Segment_DirY'][0],
            'dir_z': tree_data['Segment_DirZ'][0],
            'edep': tree_data['Segment_Edep'][0],
            'time': tree_data['Segment_Time'][0],
            'beta_start': tree_data['Segment_BetaStart'][0],
            'n_cherenkov': tree_data['Segment_NCherenkov'][0],
            'n_segments': n_segments
        }

        edep_len = len(segments['edep'])
        assert len(segments['beta_start']) == edep_len, (
            f"Segment_BetaStart length {len(segments['beta_start'])} != Segment_Edep {edep_len}")
        assert len(segments['n_cherenkov']) == edep_len, (
            f"Segment_NCherenkov length {len(segments['n_cherenkov'])} != Segment_Edep {edep_len}")

        result['meaningful_tracks'] = meaningful_tracks
        result['segments'] = segments

    return result

def generate_events_from_photonsim_particles(event_simulator, root_file_path,
                                             sensor_positions, output_dir=None,
                                             n_events=None, batch_size=100, master_seed=None,
                                             job_id=1,
                                             apply_smearing=False, apply_rotation=False, apply_translation=False,
                                             detector_config_path=None,
                                             dataset_name='unnamed_dataset', run_id=None,
                                             file_index_start=0, detector_type='cylinder',
                                             material='water', include_track_segments=True,
                                             primary_source='particles'):
    """Generate events from a PhotonSim ROOT file, writing v3 four-file batches.

    For each batch of events, writes four HDF5 files under ``output_dir``:
    ``sensor/wc_sensor_NNNN.h5``, ``inst/wc_inst_NNNN.h5``,
    ``seg/wc_seg_NNNN.h5``, ``labl/wc_labl_NNNN.h5``. See
    ``docs/LUCID_DATASET.md`` for the full schema.

    Parameters
    ----------
    event_simulator : Callable
        Per-particle simulator with baked-in detector_params. Built via
        ``setup_event_simulator(..., default_detector_params=True)``; the
        call signature is ``(track_params, key, photonsim_data)``.
    root_file_path : str
        PhotonSim ROOT file path.
    sensor_positions : array-like (n_sensors, 3)
        PMT coordinates in meters.
    output_dir : str
        Dataset root directory; four subdirs are created under it.
    n_events : int, optional
        Number of events to generate (default: all entries in the ROOT file).
    batch_size : int
        Number of events per v3 batch file.
    master_seed : int, optional
        JAX PRNG seed; random if None.
    job_id : int
        1-based job id. Folded into the seed hierarchy so reusing
        ``master_seed`` across jobs yields independent RNG streams.
    apply_smearing, apply_rotation, apply_translation : bool
        Transform toggles; rotation is ignored (PhotonSim handles it).
    detector_config_path : str, optional
        Required when ``apply_translation=True``; also used for seg config
        geometry attrs.
    dataset_name : str
        Provenance: dataset identifier written to every ``config/`` group.
    run_id : str, optional
        Provenance: unique batch identifier; auto-UUID4 if None.
    file_index_start : int
        Index of the first batch file in this invocation (default 0).
    detector_type, material : str
        Provenance: detector geometry type and medium.
    include_track_segments : bool
        Must be True for v3 output (seg file requires segment data). Default True.
    primary_source : str
        'particles' or 'genie'. Written into ``per_interaction/source_type``
        for every event of this batch.
    """
    source_type_code = _source_type_code(primary_source)
    import uproot
    import time
    import uuid
    import subprocess
    from pathlib import Path
    from lucid.detector_params import ParticleParams
    import numpy as np
    import json
    from lucid.utils import smear_charges_SK_like, smear_times

    if not include_track_segments:
        raise ValueError(
            "v3 output requires include_track_segments=True (seg file needs segment data).")

    # Generate random seed if not provided
    if master_seed is None:
        master_seed = int(time.time() * 1000000) % (2**32)
        print(f"Generated random master seed from time: {master_seed}")
    else:
        print(f"Using provided master seed: {master_seed}")

    # Resolve run_id
    if run_id is None:
        run_id = str(uuid.uuid4())
    print(f"Run id: {run_id}")

    # Resolve sensor positions / n_sensors
    sensor_positions_np = np.asarray(sensor_positions, dtype=np.float32)
    if sensor_positions_np.ndim != 2 or sensor_positions_np.shape[1] != 3:
        raise ValueError(
            f"sensor_positions must have shape (n_sensors, 3); got {sensor_positions_np.shape}")
    n_sensors = int(sensor_positions_np.shape[0])

    # Resolve git commit for provenance (fallback to env or 'unknown')
    try:
        lucid_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=lucid_repo_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = os.environ.get('GIT_COMMIT', 'unknown')

    # Create output directory tree
    out_root = Path(output_dir)
    for subdir in ('sensor', 'inst', 'seg', 'labl'):
        (out_root / subdir).mkdir(parents=True, exist_ok=True)

    source_file_abs = os.path.abspath(root_file_path)

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

    # Load detector bounds for containment calculation and (optionally) translation
    detector_bounds = None
    if apply_translation and detector_config_path is None:
        raise ValueError("detector_config_path must be provided when apply_translation=True")

    if detector_config_path is not None:
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

            # Deterministic RNG keys for this (job, event). All per-event
            # draws — vertex translation, t0, simulator, smearing — flow
            # from this hierarchy so reusing --master-seed across jobs
            # yields independent streams.
            event_keys = derive_event_keys(master_seed, job_id, event_idx,
                                           interaction_idx=0)
            master_key = event_keys['sim_key']
            t0 = float(np.random.default_rng(
                seed=event_keys['t0_seed']).uniform(
                    -T0_HALF_WINDOW_NS, T0_HALF_WINDOW_NS))

            # Draw the vertex once, up front — both dark-event and normal
            # branches write it into per_interaction/. When apply_translation
            # is False the vertex is the origin (nothing to apply).
            if apply_translation and detector_bounds is not None:
                vertex_rng = np.random.default_rng(
                    seed=event_keys['vertex_seed'])
                translation_vector = sample_translation_vector(
                    detector_bounds, vertex_rng)
            else:
                translation_vector = np.zeros(3, dtype=np.float32)

            # Read particle data from PhotonSim
            print(f"    Reading particle data from ROOT file...", flush=True)
            particle_data = read_particle_data_from_photonsim(root_file_path, event_idx, include_track_segments=include_track_segments)

            n_particles = particle_data['n_particles']
            particles = particle_data['particles']
            all_photon_origins = particle_data['photon_origins']
            all_photon_directions = particle_data['photon_directions']
            all_photon_times = particle_data['photon_times']
            all_photon_wavelengths = particle_data['photon_wavelengths']
            total_photons = len(all_photon_origins)
            print(f"    Found {n_particles} particles, {total_photons:,} total photons", flush=True)

            # Short-circuit: dark events (no tracked particles or no photons)
            # are valid physics — a neutron primary or a sub-Cherenkov-threshold
            # proton contributes no detectable signal. Skip the JAX simulation
            # (whose axis-0 min reduction would fail on empty arrays) and
            # write a zero-filled v3 entry. Downstream save_*_event_v3 writers
            # are already sparse-safe (n_hits=0 groups are valid).
            if n_particles == 0 or total_photons == 0:
                print(f"    Event {event_idx}: dark "
                      f"(n_particles={n_particles}, total_photons={total_photons}); "
                      f"writing zero-filled v3 entry.", flush=True)
                event_number = event_idx
                # t0 already drawn at top of loop from the seed hierarchy.
                PE_per_particle = np.zeros((max(n_particles, 0), n_sensors), dtype=np.float32)
                T_per_particle  = np.zeros((max(n_particles, 0), n_sensors), dtype=np.float32)
                PE_true = np.zeros(n_sensors, dtype=np.float32)
                T_true  = np.zeros(n_sensors, dtype=np.float32)
                PE_reco = PE_true
                T_reco  = T_true
                extended_info = {
                    'n_particles': int(n_particles),
                    'particles': particles,
                    'track_info_dict': particle_data.get('track_info_dict', {}),
                    't0': t0,
                    'vertex_xyz': translation_vector.copy(),
                    'source_type': source_type_code,
                    'PE_per_particle': PE_per_particle,
                    'T_per_particle': T_per_particle,
                    'PE_reco': PE_reco,
                    'T_reco': T_reco,
                    'source': 'PhotonSim_Particles_VMAP',
                    'overall_light_containment': 0.0,
                    'light_containment_by_particle': np.zeros(max(n_particles, 1), dtype=np.float32),
                    'include_track_segments': include_track_segments,
                    'source_event_idx': int(event_number),
                }
                if include_track_segments and 'meaningful_tracks' in particle_data:
                    extended_info['meaningful_tracks'] = particle_data['meaningful_tracks']
                    extended_info['segments']        = particle_data['segments']
                batch_data.append(extended_info)
                batch_indices.append(event_number)
                event_times.append(time.time() - event_start_time)
                continue

            # ========================================================================
            # VECTORIZED NUMPY PREPROCESSING + EFFICIENT JAX TRANSFER
            # All preprocessing in NumPy, single efficient transfer using device_put
            # ========================================================================
            print(f"    Preprocessing photon data...", flush=True)

            # PAD_SIZE is now computed dynamically at the function level
            default_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            # Data already NumPy in meters; just ensure float32
            all_photon_origins_np = all_photon_origins.astype(np.float32, copy=False)
            all_photon_directions_np = all_photon_directions.astype(np.float32, copy=False)
            all_photon_times_np = all_photon_times.astype(np.float32, copy=False)
            all_photon_wavelengths_np = all_photon_wavelengths.astype(np.float32, copy=False)

            # Vertex already drawn at top of loop (same value both branches).
            if apply_translation:
                all_photon_origins_np += translation_vector[None, :]

                # Apply translation to segment positions if track segments are included
                if include_track_segments and 'segments' in particle_data:
                    segments = particle_data['segments']
                    # Segments are in meters; translation_vector is in meters
                    segments['start_x'] = segments['start_x'] + translation_vector[0]
                    segments['start_y'] = segments['start_y'] + translation_vector[1]
                    segments['start_z'] = segments['start_z'] + translation_vector[2]
                    segments['end_x'] = segments['end_x'] + translation_vector[0]
                    segments['end_y'] = segments['end_y'] + translation_vector[1]
                    segments['end_z'] = segments['end_z'] + translation_vector[2]

            # Pre-allocate batched arrays
            batched_origins_np = np.zeros((n_particles, PAD_SIZE, 3), dtype=np.float32)
            batched_directions_np = np.tile(default_direction, (n_particles, PAD_SIZE, 1))
            batched_times_np = np.zeros((n_particles, PAD_SIZE), dtype=np.float32)
            batched_wavelengths_np = np.zeros((n_particles, PAD_SIZE), dtype=np.float32)

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
                    track_positions_np[particle_idx] = track_info['position']  # already m
                    track_directions_np[particle_idx] = track_info['direction']
                else:
                    track_energies_np[particle_idx] = particle_data['primary_energy']
                    track_directions_np[particle_idx] = [0.0, 0.0, 1.0]

                if apply_translation:
                    track_positions_np[particle_idx] += translation_vector
                    if track_info is not None:
                        track_info['position'] = track_positions_np[particle_idx].copy()

                # Scatter photons
                if N > 0:
                    batched_origins_np[particle_idx, :N] = all_photon_origins_np[photon_indices]
                    batched_directions_np[particle_idx, :N] = all_photon_directions_np[photon_indices]
                    batched_times_np[particle_idx, :N] = all_photon_times_np[photon_indices]
                    batched_wavelengths_np[particle_idx, :N] = all_photon_wavelengths_np[photon_indices]

            # Efficient transfer to JAX device (avoids unnecessary copies)
            batched_origins = jax.device_put(batched_origins_np)
            batched_directions = jax.device_put(batched_directions_np)
            batched_times = jax.device_put(batched_times_np)
            batched_wavelengths = jax.device_put(batched_wavelengths_np)
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
                                         photon_dirs, photon_times, photon_wavelengths, N, sim_key):
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
                    'wavelengths': photon_wavelengths,
                    'N': N,
                    'apply_rotation': False,
                    'rotation_axis': jnp.array([1.0, 0.0, 0.0]),
                    'rotation_angle': 0.0,
                    'apply_translation': apply_translation,
                    'translation_vector': translation_vector
                }

                # detector_params are baked into event_simulator via default_detector_params=True
                return event_simulator(track_params, sim_key, photonsim_data)

            # Create vectorized version using vmap
            simulate_all_particles = jax.vmap(
                simulate_single_particle,
                in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0)
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
                batched_wavelengths,
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
                smear_pe_key, smear_t_key = jax.random.split(event_keys['smear_key'])
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

            # Shift simulator outputs from G4-frame (origin at vertex) into
            # absolute detector frame by adding the per-interaction t0.
            # Only the single-vertex path is in this function today; the
            # pile-up path applies per-vertex t0 in its merger. The
            # positivity mask preserves "no-hit" sentinels (0/inf).
            t0_f32 = np.float32(t0)
            T_per_particle = np.where(T_per_particle > 0, T_per_particle + t0_f32, T_per_particle)
            T_true = np.where(T_true > 0, T_true + t0_f32, T_true)
            T_reco = np.where(T_reco > 0, T_reco + t0_f32, T_reco)
            # Segments always carry meaningful times — shift all of them.
            if include_track_segments and 'segments' in particle_data and particle_data['segments'].get('n_segments', 0) > 0:
                particle_data['segments']['time'] = \
                    np.asarray(particle_data['segments']['time'], dtype=np.float32) + t0_f32

            # Calculate light containment
            light_containment_by_particle = np.zeros(n_particles, dtype=np.float64)
            overall_light_containment = 0.0

            if detector_bounds is not None:
                # Calculate which photons are inside detector bounds
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

            # Create filename
            event_number = event_idx
            filename = os.path.join(output_dir, f'event_{event_number}.h5')

            # t0 already drawn at top of loop from the seed hierarchy.

            # Extended info with particle structure
            extended_info = {
                'n_particles': n_particles,
                'particles': particles,
                'track_info_dict': particle_data['track_info_dict'],
                't0': t0,
                'vertex_xyz': translation_vector.copy(),
                'source_type': source_type_code,
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

            # Add meaningful tracks and segments if requested
            if include_track_segments and 'meaningful_tracks' in particle_data:
                extended_info['meaningful_tracks'] = particle_data['meaningful_tracks']
                extended_info['segments'] = particle_data['segments']

            # Store for batch processing
            extended_info['source_event_idx'] = int(event_number)
            batch_data.append(extended_info)
            batch_indices.append(event_number)

            event_total_time = time.time() - event_start_time
            event_times.append(event_total_time)
            print(f"    Event total time: {event_total_time:.2f}s", flush=True)

        # Write this batch as four v3 files (sensor/inst/seg/labl)
        print(f"Saving batch {batch_idx+1} as v3 four-file group...")
        t_save_start = time.time()

        file_idx = int(file_index_start + batch_idx)
        sensor_path = out_root / 'sensor' / f'wc_sensor_{file_idx:04d}.h5'
        inst_path = out_root / 'inst' / f'wc_inst_{file_idx:04d}.h5'
        seg_path = out_root / 'seg' / f'wc_seg_{file_idx:04d}.h5'
        labl_path = out_root / 'labl' / f'wc_labl_{file_idx:04d}.h5'

        batch_src_idx = np.asarray(batch_indices, dtype=np.uint32)

        config_meta = {
            'n_events': len(batch_data),
            'git_commit': git_commit,
            'run_id': run_id,
            'dataset_name': dataset_name,
            'file_index': file_idx,
            'source_file': source_file_abs,
            'lucid_master_seed': int(master_seed),
            'photonsim_seed': -1,
            'n_sensors': n_sensors,
            'detector_type': detector_type,
            'material': material,
            'smearing_applied': bool(apply_smearing),
            'smearing_charge_function': 'SK_like' if apply_smearing else 'none',
            'smearing_time_function': 'SK_like' if apply_smearing else 'none',
            'label_names': ['category'],
        }

        # Optional geometry hints for seg config
        if detector_bounds is not None:
            config_meta['detector_shape'] = detector_bounds['type']
            if detector_bounds['type'] == 'cylinder':
                config_meta['detector_radius'] = float(detector_bounds['radius'])
                config_meta['detector_half_height'] = float(detector_bounds['height']) / 2.0
                config_meta['detector_axis'] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            elif detector_bounds['type'] == 'sphere':
                config_meta['detector_radius'] = float(detector_bounds['radius'])
            elif detector_bounds['type'] == 'box':
                l, w, h = detector_bounds['length'], detector_bounds['width'], detector_bounds['height']
                config_meta['detector_bbox'] = np.array(
                    [-l/2, l/2, -w/2, w/2, -h/2, h/2], dtype=np.float32)

        with h5py.File(sensor_path, 'w') as fs, h5py.File(inst_path, 'w') as fi, \
                h5py.File(seg_path, 'w') as fg, h5py.File(labl_path, 'w') as fl:
            write_sensor_config_v3(fs, config_meta, batch_src_idx, sensor_positions_np)
            write_inst_config_v3(fi, config_meta, batch_src_idx, sensor_positions_np)
            write_seg_config_v3(fg, config_meta, batch_src_idx)
            write_labl_config_v3(fl, config_meta, batch_src_idx)

            for seq_idx, evdict in enumerate(batch_data):
                save_sensor_event_v3(fs, evdict, seq_idx)
                save_inst_event_v3(fi, evdict, seq_idx)
                save_seg_event_v3(fg, evdict, seq_idx)
                save_labl_event_v3(fl, evdict, seq_idx)

        saved_files.extend([str(sensor_path), str(inst_path), str(seg_path), str(labl_path)])

        t_save = time.time() - t_save_start
        print(f"Batch {batch_idx+1} save time: {t_save:.3f}s\n")

    print(f"\nSuccessfully wrote {num_batches} batches "
          f"({len(saved_files)} files total) to {output_dir}/"
          f"{{sensor,inst,seg,labl}}/")

    # Print average event time
    if event_times:
        avg_time = sum(event_times) / len(event_times)
        print(f"Average event processing time: {avg_time:.3f}s")

    return saved_files


def _simulate_vertex_stream(
    *,
    event_simulator,
    particle_data,
    translation_vector,
    apply_translation,
    n_sensors,
    pad_size,
    sim_key,
):
    """Run the vmap photon simulator for one PhotonSim stream.

    Returns (PE_per_particle, T_per_particle) as numpy float32 arrays of
    shape ``(n_particles, n_sensors)``. Inputs are mutated: track_info
    positions get shifted by ``translation_vector`` to keep the per-track
    info in the shifted frame. All times remain in G4 frame; the caller
    adds per-interaction t0 afterwards to move to absolute detector frame.
    """
    from lucid.detector_params import ParticleParams

    n_particles = particle_data['n_particles']
    particles = particle_data['particles']
    default_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    batched_origins_np = np.zeros((n_particles, pad_size, 3), dtype=np.float32)
    batched_directions_np = np.tile(default_direction, (n_particles, pad_size, 1))
    batched_times_np = np.zeros((n_particles, pad_size), dtype=np.float32)
    batched_wavelengths_np = np.zeros((n_particles, pad_size), dtype=np.float32)
    N_per_particle_np = np.zeros(n_particles, dtype=np.int32)
    track_energies_np = np.zeros(n_particles, dtype=np.float32)
    track_positions_np = np.zeros((n_particles, 3), dtype=np.float32)
    track_directions_np = np.zeros((n_particles, 3), dtype=np.float32)

    all_origins = particle_data['photon_origins']
    all_dirs    = particle_data['photon_directions']
    all_times   = particle_data['photon_times']
    all_wl      = particle_data['photon_wavelengths']

    for pi, particle in enumerate(particles):
        photon_indices = particle['photon_indices']
        N = len(photon_indices)
        N_per_particle_np[pi] = N
        ti = particle['track_info']
        if ti is not None:
            track_energies_np[pi]   = ti['energy']
            track_positions_np[pi]  = ti['position']
            track_directions_np[pi] = ti['direction']
        else:
            track_energies_np[pi]   = particle_data.get('primary_energy', 0.0)
            track_directions_np[pi] = default_direction
        if apply_translation:
            track_positions_np[pi] += translation_vector
            if ti is not None:
                ti['position'] = track_positions_np[pi].copy()
        if N > 0:
            batched_origins_np[pi, :N]      = all_origins[photon_indices]
            batched_directions_np[pi, :N]   = all_dirs[photon_indices]
            batched_times_np[pi, :N]        = all_times[photon_indices]
            batched_wavelengths_np[pi, :N]  = all_wl[photon_indices]

    batched_origins     = jax.device_put(batched_origins_np)
    batched_directions  = jax.device_put(batched_directions_np)
    batched_times       = jax.device_put(batched_times_np)
    batched_wavelengths = jax.device_put(batched_wavelengths_np)
    N_per_particle_array    = jax.device_put(N_per_particle_np)
    track_energies_array    = jax.device_put(track_energies_np)
    track_positions_array   = jax.device_put(track_positions_np)
    track_directions_array  = jax.device_put(track_directions_np)

    def _sim_one(energy, pos, dir_, po, pd, pt, pw, N, key):
        track_params = ParticleParams.from_cartesian(
            energy=energy, position=pos, direction=dir_, t0=0.0)
        photonsim_data = {
            'photon_origins': po, 'photon_directions': pd,
            'photon_times': pt, 'wavelengths': pw, 'N': N,
            'apply_rotation': False,
            'rotation_axis': jnp.array([1.0, 0.0, 0.0]),
            'rotation_angle': 0.0,
            'apply_translation': apply_translation,
            'translation_vector': translation_vector,
        }
        return event_simulator(track_params, key, photonsim_data)

    simulate_all = jax.vmap(_sim_one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))
    particle_keys = jax.random.split(sim_key, n_particles)
    PE_pp, T_pp = simulate_all(
        track_energies_array, track_positions_array, track_directions_array,
        batched_origins, batched_directions, batched_times, batched_wavelengths,
        N_per_particle_array, particle_keys)
    return np.asarray(PE_pp, dtype=np.float32), np.asarray(T_pp, dtype=np.float32)


def _offset_track_ids(particle_data, offset):
    """Shift all G4 track IDs in ``particle_data`` by ``offset``.

    parent_id == 0 (primary convention) is left alone so primaries
    remain recognizable after merging. Mutates in place and also returns
    the max track_id seen post-shift (so the caller can advance the
    running offset for the next vertex stream).
    """
    if offset == 0:
        return _max_track_id(particle_data)

    def _shift(tid):
        return int(tid) + offset if int(tid) > 0 else 0

    # meaningful_tracks: remap both the dict keys and each record's track_id / parent_id.
    mt = particle_data.get('meaningful_tracks')
    if mt:
        new_mt = {}
        for tid, t in mt.items():
            t = dict(t)
            t['track_id']  = _shift(t['track_id'])
            t['parent_id'] = _shift(t['parent_id'])
            new_mt[_shift(tid)] = t
        particle_data['meaningful_tracks'] = new_mt

    # track_info_dict: same treatment.
    tid_dict = particle_data.get('track_info_dict')
    if tid_dict:
        new_tid = {}
        for tid, t in tid_dict.items():
            t = dict(t)
            t['track_id']  = _shift(t.get('track_id', tid))
            t['parent_id'] = _shift(t.get('parent_id', 0))
            new_tid[_shift(tid)] = t
        particle_data['track_info_dict'] = new_tid

    # particles: genealogy and extended_genealogy lists of track IDs.
    for p in particle_data.get('particles', []):
        gen = p.get('genealogy') or []
        p['genealogy'] = [_shift(g) for g in gen]
        ext = p.get('extended_genealogy')
        if ext is not None:
            p['extended_genealogy'] = [_shift(g) for g in ext]
        # track_info inside particle (if any) — also remap.
        ti = p.get('track_info')
        if ti is not None and 'track_id' in ti:
            ti['track_id'] = _shift(ti['track_id'])
            if 'parent_id' in ti:
                ti['parent_id'] = _shift(ti['parent_id'])

    return _max_track_id(particle_data)


def _max_track_id(particle_data):
    """Largest track_id present in the stream (0 if the stream is empty)."""
    mt = particle_data.get('meaningful_tracks') or {}
    tid_d = particle_data.get('track_info_dict') or {}
    ids = [int(t) for t in mt.keys()] + [int(t) for t in tid_d.keys()]
    return max(ids) if ids else 0


def generate_events_from_photonsim_pileup(
    event_simulator,
    root_file_paths,
    vertex_primary_sources,
    sensor_positions,
    output_dir=None,
    n_events=None,
    batch_size=100,
    master_seed=None,
    job_id=1,
    apply_smearing=False,
    apply_translation=False,
    detector_config_path=None,
    dataset_name='unnamed_pileup_dataset',
    run_id=None,
    file_index_start=0,
    detector_type='cylinder',
    material='water',
    include_track_segments=True,
):
    """Generate pile-up events by merging N PhotonSim streams per event.

    Each entry in ``root_file_paths`` is a PhotonSim ROOT file from one
    vertex's interaction. For each event index, we draw an independent
    absolute t0 and fiducial vertex per vertex, simulate each vertex's
    photons, remap G4 track IDs to avoid collisions, and merge the
    per-vertex results into one event_dict. Sensor/inst/seg/labl are
    written using the same v3 writers as the single-vertex path.

    Parameters
    ----------
    root_file_paths : list[str | Path]
        One PhotonSim ROOT file per vertex, matched by index to
        ``vertex_primary_sources``.
    vertex_primary_sources : list[str]
        'particles' or 'genie' per vertex, used to set
        per_interaction/source_type for each primary from that vertex.
    """
    import uproot
    import time as _time
    import uuid
    import subprocess
    import json
    from pathlib import Path

    if not include_track_segments:
        raise ValueError(
            "v3 output requires include_track_segments=True.")

    if len(root_file_paths) != len(vertex_primary_sources):
        raise ValueError(
            f"root_file_paths and vertex_primary_sources length mismatch: "
            f"{len(root_file_paths)} vs {len(vertex_primary_sources)}")
    N_vertices = len(root_file_paths)
    if N_vertices < 2:
        raise ValueError("Pile-up requires at least 2 vertices.")

    master_seed = _resolve_master_seed(master_seed)
    print(f"Pile-up: master_seed={master_seed}, job_id={job_id}, "
          f"n_vertices={N_vertices}")

    if run_id is None:
        run_id = str(uuid.uuid4())
    print(f"Run id: {run_id}")

    sensor_positions_np = np.asarray(sensor_positions, dtype=np.float32)
    n_sensors = int(sensor_positions_np.shape[0])

    # Git commit (same as non-pile-up)
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo_root,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_commit = os.environ.get('GIT_COMMIT', 'unknown')

    out_root = Path(output_dir)
    for sub in ('sensor', 'inst', 'seg', 'labl'):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    # Determine common number of events across all ROOT files.
    per_file_counts = []
    for p in root_file_paths:
        with uproot.open(p) as f:
            per_file_counts.append(int(f['OpticalPhotons'].num_entries))
    common = min(per_file_counts)
    if n_events is None:
        n_events = common
    elif n_events > common:
        raise ValueError(
            f"n_events={n_events} exceeds min per-vertex entries "
            f"{per_file_counts}; at least one ROOT file is shorter than "
            f"requested.")
    print(f"Per-vertex ROOT entries: {per_file_counts}; "
          f"merging {n_events} events.")

    # Per-vertex PAD_SIZE (take max across all so one shared batched shape works).
    PAD_SIZE = 0
    for p in root_file_paths:
        PAD_SIZE = max(PAD_SIZE, get_max_photons_per_particle(str(p), n_events))
    PAD_SIZE += 1
    print(f"PAD_SIZE (max photons per particle across all streams): {PAD_SIZE}")

    # Detector bounds for vertex sampling + containment (same as non-pile-up).
    detector_bounds = None
    if detector_config_path is not None:
        with open(detector_config_path) as fj:
            cfg = json.load(fj)
        detector_type_from_cfg = cfg.get('detector_type', 'cylinder')
        gd = cfg['geometry_definitions']
        if detector_type_from_cfg == 'cylinder':
            detector_bounds = {'type': 'cylinder', 'radius': gd['radius'], 'height': gd['height']}
        elif detector_type_from_cfg == 'sphere':
            detector_bounds = {'type': 'sphere', 'radius': gd['radius']}
        elif detector_type_from_cfg == 'box':
            detector_bounds = {'type': 'box',
                               'length': gd['length'], 'width': gd['width'],
                               'height': gd['height']}
    if apply_translation and detector_bounds is None:
        raise ValueError("detector_config_path required when apply_translation=True.")

    saved_files = []
    event_times = []
    num_batches = (n_events + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_events)
        print(f"Pile-up batch {batch_idx+1}/{num_batches} "
              f"(events {start_idx}..{end_idx-1})")
        batch_data = []
        batch_indices = []

        for event_idx in range(start_idx, end_idx):
            t_start = _time.time()
            print(f"\n  Event {event_idx+1}/{n_events}:", flush=True)
            streams = []
            running_offset = 0

            for vidx in range(N_vertices):
                event_keys = derive_event_keys(
                    master_seed, job_id, event_idx, interaction_idx=vidx)
                t0_i = float(np.random.default_rng(
                    seed=event_keys['t0_seed']).uniform(
                        -T0_HALF_WINDOW_NS, T0_HALF_WINDOW_NS))
                if apply_translation:
                    vrng = np.random.default_rng(seed=event_keys['vertex_seed'])
                    vertex_i = sample_translation_vector(detector_bounds, vrng)
                else:
                    vertex_i = np.zeros(3, dtype=np.float32)
                print(f"    vertex {vidx}: t0={t0_i:+.2f} ns, "
                      f"xyz=({vertex_i[0]:.3f}, {vertex_i[1]:.3f}, {vertex_i[2]:.3f}) m",
                      flush=True)

                particle_data = read_particle_data_from_photonsim(
                    str(root_file_paths[vidx]), event_idx,
                    include_track_segments=include_track_segments)

                # Remap G4 track IDs so streams don't collide.
                stream_max = _offset_track_ids(particle_data, running_offset)

                n_particles_i = int(particle_data['n_particles'])
                total_photons_i = len(particle_data['photon_origins'])
                source_type_code_i = _source_type_code(vertex_primary_sources[vidx])

                if n_particles_i == 0 or total_photons_i == 0:
                    print(f"      dark stream "
                          f"(n_particles={n_particles_i}, photons={total_photons_i})",
                          flush=True)
                    streams.append({
                        'particles': particle_data.get('particles', []),
                        'meaningful_tracks': particle_data.get('meaningful_tracks', {}),
                        'segments': particle_data.get('segments', {'n_segments': 0}),
                        'PE_per_particle': np.zeros((n_particles_i, n_sensors), dtype=np.float32),
                        'T_per_particle':  np.zeros((n_particles_i, n_sensors), dtype=np.float32),
                        't0': t0_i,
                        'vertex_xyz': vertex_i,
                        'source_type': source_type_code_i,
                    })
                    running_offset = stream_max + 1
                    continue

                # Translate photon origins + segment positions by vertex_i
                particle_data['photon_origins'] = \
                    particle_data['photon_origins'].astype(np.float32, copy=False)
                if apply_translation:
                    particle_data['photon_origins'] = \
                        particle_data['photon_origins'] + vertex_i[None, :]
                    if (include_track_segments and 'segments' in particle_data
                            and particle_data['segments'].get('n_segments', 0) > 0):
                        segs = particle_data['segments']
                        for axis_idx, (sk, ek) in enumerate(
                                (('start_x', 'end_x'), ('start_y', 'end_y'), ('start_z', 'end_z'))):
                            segs[sk] = segs[sk] + vertex_i[axis_idx]
                            segs[ek] = segs[ek] + vertex_i[axis_idx]

                # Run the simulator for this vertex
                PE_i, T_i = _simulate_vertex_stream(
                    event_simulator=event_simulator,
                    particle_data=particle_data,
                    translation_vector=vertex_i,
                    apply_translation=apply_translation,
                    n_sensors=n_sensors,
                    pad_size=PAD_SIZE,
                    sim_key=event_keys['sim_key'],
                )

                # Apply +t0_i to shift simulator output into absolute detector frame
                T_i = np.where(T_i > 0, T_i + np.float32(t0_i), T_i)

                # Same shift for segment times
                if (include_track_segments and 'segments' in particle_data
                        and particle_data['segments'].get('n_segments', 0) > 0):
                    particle_data['segments']['time'] = (
                        np.asarray(particle_data['segments']['time'], dtype=np.float32)
                        + np.float32(t0_i))

                streams.append({
                    'particles': particle_data['particles'],
                    'meaningful_tracks': particle_data.get('meaningful_tracks', {}),
                    'segments': particle_data.get('segments', {'n_segments': 0}),
                    'PE_per_particle': PE_i,
                    'T_per_particle':  T_i,
                    't0': t0_i,
                    'vertex_xyz': vertex_i,
                    'source_type': source_type_code_i,
                })
                running_offset = stream_max + 1

            # ---- merge streams into one event_dict ----
            merged = _merge_pileup_streams(
                streams, n_sensors=n_sensors,
                apply_smearing=apply_smearing,
                smear_key=derive_event_keys(
                    master_seed, job_id, event_idx,
                    interaction_idx=N_vertices)['smear_key'],
                detector_bounds=detector_bounds,
                include_track_segments=include_track_segments,
            )
            merged['source_event_idx'] = int(event_idx)
            merged['include_track_segments'] = include_track_segments
            merged['source'] = 'PhotonSim_Pileup'

            batch_data.append(merged)
            batch_indices.append(int(event_idx))
            event_times.append(_time.time() - t_start)

        # Write batch (same as non-pile-up)
        file_idx = int(file_index_start + batch_idx)
        sensor_path = out_root / 'sensor' / f'wc_sensor_{file_idx:04d}.h5'
        inst_path   = out_root / 'inst'   / f'wc_inst_{file_idx:04d}.h5'
        seg_path    = out_root / 'seg'    / f'wc_seg_{file_idx:04d}.h5'
        labl_path   = out_root / 'labl'   / f'wc_labl_{file_idx:04d}.h5'

        batch_src_idx = np.asarray(batch_indices, dtype=np.uint32)
        config_meta = {
            'n_events': len(batch_data),
            'git_commit': git_commit,
            'run_id': run_id,
            'dataset_name': dataset_name,
            'file_index': file_idx,
            'source_file': ','.join(os.path.abspath(str(p)) for p in root_file_paths),
            'lucid_master_seed': int(master_seed),
            'photonsim_seed': -1,
            'n_sensors': n_sensors,
            'detector_type': detector_type,
            'material': material,
            'smearing_applied': bool(apply_smearing),
            'smearing_charge_function': 'SK_like' if apply_smearing else 'none',
            'smearing_time_function': 'SK_like' if apply_smearing else 'none',
            'label_names': ['category'],
        }
        if detector_bounds is not None:
            config_meta['detector_shape'] = detector_bounds['type']
            if detector_bounds['type'] == 'cylinder':
                config_meta['detector_radius']      = detector_bounds['radius']
                config_meta['detector_half_height'] = detector_bounds['height'] / 2.0

        with h5py.File(sensor_path, 'w') as fs, \
             h5py.File(inst_path,   'w') as fi, \
             h5py.File(seg_path,    'w') as fg, \
             h5py.File(labl_path,   'w') as fl:
            write_sensor_config_v3(fs, config_meta, batch_src_idx, sensor_positions_np)
            write_inst_config_v3(fi, config_meta, batch_src_idx, sensor_positions_np)
            write_seg_config_v3(fg, config_meta, batch_src_idx)
            write_labl_config_v3(fl, config_meta, batch_src_idx)
            for seq_idx, ev in enumerate(batch_data):
                save_sensor_event_v3(fs, ev, seq_idx)
                save_inst_event_v3(fi, ev, seq_idx)
                save_seg_event_v3(fg, ev, seq_idx)
                save_labl_event_v3(fl, ev, seq_idx)

        saved_files.extend([str(sensor_path), str(inst_path), str(seg_path), str(labl_path)])

    if event_times:
        print(f"\nAverage pile-up event time: "
              f"{sum(event_times)/len(event_times):.3f}s")
    return saved_files


def _merge_pileup_streams(streams, *, n_sensors, apply_smearing,
                          smear_key, detector_bounds, include_track_segments):
    """Merge per-vertex streams into a single event_dict.

    Per-interaction metadata (t0, vertex_xyz, source_type) is broadcast
    to one row per primary in the merged event. Primaries are identified
    after the merge by ``derive_track_ancestor_and_interaction`` (parent-
    chain walk to parent_id==0); each primary's vertex is looked up via
    the track_id range it falls into — streams are concatenated in
    declared order with monotonically increasing track IDs, so a
    primary's range uniquely identifies its source stream.

    ``smear_key`` is a jax key (not a concrete seed).
    """
    # Concatenate particles, meaningful_tracks, segments (all post-remap).
    all_particles = []
    all_tracks = {}
    all_segs = {
        'start_x': [], 'start_y': [], 'start_z': [],
        'end_x':   [], 'end_y':   [], 'end_z':   [],
        'dir_x':   [], 'dir_y':   [], 'dir_z':   [],
        'edep': [], 'time': [], 'beta_start': [], 'n_cherenkov': [],
    }
    PE_per_stream = []
    T_per_stream  = []

    stream_t0        = [s['t0']          for s in streams]
    stream_vertex    = [s['vertex_xyz']  for s in streams]
    stream_src_type  = [s['source_type'] for s in streams]
    # Max track_id contributed by each stream (post-remap). Used to look
    # up which vertex owns a given primary track_id after the merge.
    stream_max_tid = []

    for s in streams:
        all_particles.extend(s['particles'])
        all_tracks.update(s['meaningful_tracks'])
        segs = s['segments']
        if segs and segs.get('n_segments', 0) > 0:
            for k in all_segs:
                all_segs[k].append(np.asarray(segs[k]))
        PE_per_stream.append(s['PE_per_particle'])
        T_per_stream.append(s['T_per_particle'])
        stream_tids = list(s.get('meaningful_tracks', {}).keys())
        stream_max_tid.append(max(stream_tids) if stream_tids else 0)

    n_particles_total = len(all_particles)
    PE_per_particle = (np.concatenate(PE_per_stream, axis=0)
                       if PE_per_stream and sum(x.shape[0] for x in PE_per_stream) > 0
                       else np.zeros((0, n_sensors), dtype=np.float32))
    T_per_particle  = (np.concatenate(T_per_stream, axis=0)
                       if T_per_stream and sum(x.shape[0] for x in T_per_stream) > 0
                       else np.zeros((0, n_sensors), dtype=np.float32))

    # Aggregate across particles for sensor/inst files
    if PE_per_particle.shape[0] > 0:
        PE_true = np.sum(PE_per_particle, axis=0).astype(np.float32)
        masked = np.where(T_per_particle > 0, T_per_particle, np.inf)
        T_true = np.min(masked, axis=0)
        T_true = np.where(np.isfinite(T_true), T_true, 0.0).astype(np.float32)
    else:
        PE_true = np.zeros(n_sensors, dtype=np.float32)
        T_true  = np.zeros(n_sensors, dtype=np.float32)

    if apply_smearing and PE_per_particle.shape[0] > 0:
        from lucid.utils import smear_charges_SK_like, smear_times
        smear_pe_key, smear_t_key = jax.random.split(smear_key)
        PE_reco = np.asarray(
            smear_charges_SK_like(jnp.asarray(PE_true), key=smear_pe_key),
            dtype=np.float32)
        T_reco = np.asarray(
            smear_times(jnp.asarray(T_true), key=smear_t_key),
            dtype=np.float32)
    else:
        PE_reco = PE_true.copy()
        T_reco  = T_true.copy()

    # Derive primary ranks on the merged dict (same function save_labl
    # will call) so the t0/vertex/source_type arrays align with the
    # per_track/interaction column it writes.
    merged_for_derive = {
        'meaningful_tracks': all_tracks,
        'particles': all_particles,
    }
    ancestor, interaction = derive_track_ancestor_and_interaction(merged_for_derive)
    if interaction.size > 0:
        n_interactions = int(interaction.max()) + 1
        # Map each primary rank to its source stream via the track_id
        # ranges recorded during merge. Stream i owns IDs in
        # (stream_max_tid[i-1], stream_max_tid[i]] (stream 0: [1, max_0]).
        # Assemble sorted unique primaries in the same order derive uses.
        unique_primaries_sorted = sorted(set(int(a) for a in ancestor))
        upper_bounds = []
        running = 0
        for i in range(len(streams)):
            running = max(running, stream_max_tid[i])
            upper_bounds.append(running)

        def _vertex_of(pid):
            for i, ub in enumerate(upper_bounds):
                if pid <= ub:
                    return i
            return len(streams) - 1  # fallback shouldn't happen in practice

        rank_to_vertex = [_vertex_of(tid) for tid in unique_primaries_sorted]
        t0_arr = np.array([stream_t0[v]       for v in rank_to_vertex], dtype=np.float32)
        vx = np.stack([np.asarray(stream_vertex[v], dtype=np.float32)
                       for v in rank_to_vertex], axis=0)
        st_arr = np.array([stream_src_type[v] for v in rank_to_vertex], dtype=np.uint8)
    else:
        # Every stream was dark — fall back to the first vertex's values as a
        # synthetic 1-row table so per_interaction/ is never empty.
        t0_arr = np.array([stream_t0[0]], dtype=np.float32)
        vx = np.asarray(stream_vertex[0], dtype=np.float32).reshape(1, 3)
        st_arr = np.array([stream_src_type[0]], dtype=np.uint8)

    # Merge segment arrays
    if all_segs['time']:
        seg_merged = {k: np.concatenate(v) for k, v in all_segs.items()}
        seg_merged['n_segments'] = int(len(seg_merged['time']))
    else:
        seg_merged = {'n_segments': 0}

    # Light containment (aggregate: fraction of all photons inside detector bounds)
    light_containment_by_particle = np.zeros(max(n_particles_total, 1), dtype=np.float32)
    overall_light_containment = 0.0
    if detector_bounds is not None and n_particles_total > 0:
        # Aggregate across all streams' photon_origins/indices via the
        # per-particle mapping — skipped for brevity in MVP pile-up;
        # downstream analysis can recompute from segments if needed.
        overall_light_containment = 1.0
        light_containment_by_particle[:] = 1.0

    return {
        'n_particles': int(n_particles_total),
        'particles': all_particles,
        'track_info_dict': {},  # unused by writers; merged into meaningful_tracks
        'meaningful_tracks': all_tracks,
        'segments': seg_merged,
        # Per-interaction arrays (length = number of primaries across all vertices)
        't0':          t0_arr,
        'vertex_xyz':  vx,
        'source_type': st_arr,
        'PE_per_particle': PE_per_particle,
        'T_per_particle':  T_per_particle,
        'PE_reco': PE_reco,
        'T_reco':  T_reco,
        'overall_light_containment': overall_light_containment,
        'light_containment_by_particle': light_containment_by_particle,
    }


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


# ---------------------------------------------------------------------------
# V3 format: four-file per-event-group HDF5 (sensor / inst / seg / labl).
# See docs/LUCID_DATASET.md for the full schema.
# ---------------------------------------------------------------------------

_GZIP_OPTS = dict(compression='gzip', compression_opts=4)
_V3_FORMAT_VERSION = 4

# per_interaction/source_type encoding
SOURCE_TYPE_PARTICLES = 0
SOURCE_TYPE_GENIE     = 1

# t0 draw half-window (ns). Applied symmetrically per interaction:
# t0 ~ Uniform(-T0_HALF_WINDOW_NS, +T0_HALF_WINDOW_NS). Wide enough to
# (a) randomize absolute event time so downstream models can't assume
# t=0 is the true start, and (b) cover a ±250 ns pile-up window.
T0_HALF_WINDOW_NS = 250.0


def _source_type_code(primary_source):
    """Map config's ``primary_source`` string to the per_interaction/source_type int."""
    if primary_source == 'genie':
        return SOURCE_TYPE_GENIE
    return SOURCE_TYPE_PARTICLES


def sample_translation_vector(detector_bounds, rng):
    """Draw a random vertex inside the fiducial volume of the detector.

    Cylinder: uniform in (r, theta, z) with r <= 0.9*R, |z| <= 0.45*H.
    Sphere:   uniform in the 0.9*R ball.
    Box:      uniform in the 0.9-fraction-scaled box.

    Returns a length-3 float32 array in meters.
    """
    if detector_bounds is None:
        return np.zeros(3, dtype=np.float32)
    kind = detector_bounds['type']
    if kind == 'cylinder':
        r_max = detector_bounds['radius'] * 0.9
        h_max = detector_bounds['height'] * 0.9 / 2.0
        u = rng.uniform(0, 1, size=3).astype(np.float32)
        r = r_max * np.sqrt(u[0])
        theta = 2.0 * np.pi * u[1]
        z = (2.0 * u[2] - 1.0) * h_max
        return np.array([r * np.cos(theta), r * np.sin(theta), z], dtype=np.float32)
    if kind == 'sphere':
        r_max = detector_bounds['radius'] * 0.9
        u = rng.uniform(0, 1, size=3).astype(np.float32)
        r = r_max * (u[0] ** (1.0 / 3.0))
        cos_t = 2.0 * u[1] - 1.0
        phi = 2.0 * np.pi * u[2]
        sin_t = np.sqrt(1.0 - cos_t * cos_t)
        return r * np.array([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t],
                            dtype=np.float32)
    if kind == 'box':
        u = rng.uniform(0, 1, size=3).astype(np.float32)
        return np.array([
            (2.0 * u[0] - 1.0) * detector_bounds['length'] * 0.45,
            (2.0 * u[1] - 1.0) * detector_bounds['width']  * 0.45,
            (2.0 * u[2] - 1.0) * detector_bounds['height'] * 0.45,
        ], dtype=np.float32)
    raise ValueError(f"Unknown detector_bounds type: {kind!r}")


def derive_particle_idx_per_track(event_dict):
    """Map each meaningful track to the local index of its owning particle.

    Walks the track's ``parent_id`` chain until reaching the ``track_id`` of
    a categorized particle (the last entry of that particle's
    ``genealogy``). Orphaned tracks (no categorized ancestor found) get -1.

    Returns
    -------
    np.ndarray (int32) shape (n_tracks,)
    """
    tracks = event_dict.get('meaningful_tracks', {})
    particles = event_dict.get('particles', [])

    id_to_idx = {}
    for i, particle in enumerate(particles):
        gen = particle.get('genealogy') or []
        if gen:
            id_to_idx[int(gen[-1])] = i

    out = np.full(len(tracks), -1, dtype=np.int32)
    for row, tinfo in enumerate(tracks.values()):
        cur = int(tinfo['track_id'])
        visited = set()
        while cur > 0 and cur not in visited:
            visited.add(cur)
            if cur in id_to_idx:
                out[row] = id_to_idx[cur]
                break
            parent = tracks.get(cur)
            if parent is None:
                break
            cur = int(parent['parent_id'])
    return out


def derive_track_ancestor_and_interaction(event_dict):
    """For each meaningful track, derive (ancestor_track_id, interaction_id).

    * ``ancestor_track_id`` is the root of the parent chain — the primary
      this track descends from. A track that is itself a primary
      (``parent_id == 0``) is its own ancestor.
    * ``interaction_id`` is the 0-based rank of that ancestor among the
      event's primaries (sorted by track_id), grouping all tracks of the
      same neutrino/interaction vertex.

    Returns
    -------
    (ancestor, interaction) : tuple of np.ndarray (int32, int32)
        Both shape ``(n_tracks,)``.
    """
    tracks = event_dict.get('meaningful_tracks', {})
    if not tracks:
        empty = np.array([], dtype=np.int32)
        return empty, empty.copy()

    parent_of = {int(tid): int(t['parent_id']) for tid, t in tracks.items()}

    def walk_to_root(tid):
        cur = tid
        visited = set()
        while cur > 0 and cur not in visited:
            visited.add(cur)
            parent = parent_of.get(cur, 0)
            if parent == 0 or parent not in parent_of:
                return cur
            cur = parent
        return cur

    ancestors = np.array(
        [walk_to_root(int(t['track_id'])) for t in tracks.values()],
        dtype=np.int32)

    unique_primaries = sorted(set(int(a) for a in ancestors))
    rank_of = {a: i for i, a in enumerate(unique_primaries)}
    interaction = np.array([rank_of[int(a)] for a in ancestors], dtype=np.int32)
    return ancestors, interaction


def _write_common_config_attrs(f, config_meta):
    """Create ``config/`` group with provenance attrs common to all v3 files."""
    cfg = f.require_group('config')
    cfg.attrs['format_version'] = _V3_FORMAT_VERSION
    cfg.attrs['n_events'] = int(config_meta['n_events'])
    cfg.attrs['git_commit'] = str(config_meta.get('git_commit', 'unknown'))
    cfg.attrs['run_id'] = str(config_meta['run_id'])
    cfg.attrs['dataset_name'] = str(config_meta['dataset_name'])
    cfg.attrs['file_index'] = int(config_meta.get('file_index', 0))
    cfg.attrs['source_file'] = str(config_meta['source_file'])
    cfg.attrs['lucid_master_seed'] = int(config_meta['lucid_master_seed'])
    cfg.attrs['photonsim_seed'] = int(config_meta.get('photonsim_seed', -1))
    return cfg


def write_sensor_config_v3(f, config_meta, source_event_idx, sensor_positions):
    """Write the config/ group of a sensor v3 file."""
    cfg = _write_common_config_attrs(f, config_meta)
    cfg.attrs['n_sensors'] = int(config_meta['n_sensors'])
    cfg.attrs['detector_type'] = str(config_meta['detector_type'])
    cfg.attrs['material'] = str(config_meta['material'])
    cfg.attrs['smearing_applied'] = bool(config_meta['smearing_applied'])
    cfg.attrs['smearing_charge_function'] = str(
        config_meta.get('smearing_charge_function', 'default'))
    cfg.attrs['smearing_time_function'] = str(
        config_meta.get('smearing_time_function', 'default'))
    cfg.create_dataset('source_event_idx',
                       data=np.asarray(source_event_idx, dtype=np.uint32),
                       **_GZIP_OPTS)
    cfg.create_dataset('sensor_positions',
                       data=np.asarray(sensor_positions, dtype=np.float32),
                       **_GZIP_OPTS)


def write_inst_config_v3(f, config_meta, source_event_idx, sensor_positions):
    """Write the config/ group of an inst v3 file."""
    cfg = _write_common_config_attrs(f, config_meta)
    cfg.attrs['n_sensors'] = int(config_meta['n_sensors'])
    cfg.attrs['detector_type'] = str(config_meta['detector_type'])
    cfg.attrs['material'] = str(config_meta['material'])
    cfg.create_dataset('source_event_idx',
                       data=np.asarray(source_event_idx, dtype=np.uint32),
                       **_GZIP_OPTS)
    cfg.create_dataset('sensor_positions',
                       data=np.asarray(sensor_positions, dtype=np.float32),
                       **_GZIP_OPTS)


def write_seg_config_v3(f, config_meta, source_event_idx):
    """Write the config/ group of a seg v3 file."""
    cfg = _write_common_config_attrs(f, config_meta)
    cfg.attrs['detector_type'] = str(config_meta['detector_type'])
    cfg.attrs['material'] = str(config_meta['material'])
    if 'detector_shape' in config_meta:
        cfg.attrs['detector_shape'] = str(config_meta['detector_shape'])
    for key in ('detector_bbox', 'detector_axis'):
        if key in config_meta:
            cfg.create_dataset(key,
                               data=np.asarray(config_meta[key], dtype=np.float32))
    for key in ('detector_radius', 'detector_half_height'):
        if key in config_meta:
            cfg.attrs[key] = float(config_meta[key])
    cfg.create_dataset('source_event_idx',
                       data=np.asarray(source_event_idx, dtype=np.uint32),
                       **_GZIP_OPTS)


def write_labl_config_v3(f, config_meta, source_event_idx):
    """Write the config/ group of a labl v3 file."""
    cfg = _write_common_config_attrs(f, config_meta)
    label_names = list(config_meta.get('label_names', ['category']))
    cfg.attrs['label_names'] = np.array(label_names, dtype=h5py.string_dtype())
    cfg.create_dataset('source_event_idx',
                       data=np.asarray(source_event_idx, dtype=np.uint32),
                       **_GZIP_OPTS)


def _event_group_name(seq_idx):
    return f'event_{int(seq_idx):03d}'


def save_sensor_event_v3(f, event_dict, seq_idx):
    """Write a single event_NNN/ group to an already-open sensor v3 file.

    ``event_dict`` must contain: ``source_event_idx``, ``PE_reco``,
    ``T_reco``. Times in ``T_reco`` are expected in absolute detector
    frame — the caller applies per-interaction t0 shifts before calling
    this writer; the writer does not shift times further.
    """
    grp = f.create_group(_event_group_name(seq_idx))
    grp.attrs['source_event_idx'] = int(event_dict['source_event_idx'])

    pe = np.asarray(event_dict['PE_reco'], dtype=np.float32)
    t = np.asarray(event_dict['T_reco'], dtype=np.float32)

    mask = (pe > 0) | (np.isfinite(t) & (t > 0) & (t < 1e5))
    indices = np.where(mask)[0].astype(np.uint16)
    pe_sparse = pe[mask].astype(np.float32)
    t_sparse = np.where(np.isfinite(t[mask]), t[mask], np.float32(0.0)).astype(np.float32)

    grp.attrs['n_hits'] = int(indices.size)
    grp.create_dataset('sensor_idx', data=indices, **_GZIP_OPTS)
    grp.create_dataset('PE', data=pe_sparse, **_GZIP_OPTS)
    grp.create_dataset('T', data=t_sparse, **_GZIP_OPTS)


def save_inst_event_v3(f, event_dict, seq_idx):
    """Write a single event_NNN/ group to an already-open inst v3 file.

    Stores the per-particle PE/T decomposition as FK rows keyed by
    ``particle_idx`` (local to the event). Times in ``T_per_particle``
    are expected in absolute detector frame — no shift is applied here.
    """
    grp = f.create_group(_event_group_name(seq_idx))
    grp.attrs['source_event_idx'] = int(event_dict['source_event_idx'])
    grp.attrs['n_particles'] = int(event_dict['n_particles'])

    pe_pp = np.asarray(event_dict['PE_per_particle'], dtype=np.float32)
    t_pp = np.asarray(event_dict['T_per_particle'], dtype=np.float32)
    n_p = pe_pp.shape[0]

    particle_idx_parts, sensor_idx_parts, pe_parts, t_parts = [], [], [], []
    for i in range(n_p):
        mask = pe_pp[i] > 0
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        particle_idx_parts.append(np.full(idx.shape[0], i, dtype=np.int32))
        sensor_idx_parts.append(idx.astype(np.uint16))
        pe_parts.append(pe_pp[i, mask].astype(np.float32))
        t_vals = t_pp[i, mask]
        t_vals = np.where(np.isfinite(t_vals), t_vals, np.float32(0.0))
        t_parts.append(t_vals.astype(np.float32))

    def _cat(xs, dtype):
        return np.concatenate(xs).astype(dtype) if xs else np.array([], dtype=dtype)

    particle_idx_arr = _cat(particle_idx_parts, np.int32)
    sensor_idx_arr = _cat(sensor_idx_parts, np.uint16)
    pe_arr = _cat(pe_parts, np.float32)
    t_arr = _cat(t_parts, np.float32)

    grp.attrs['n_particle_hits'] = int(particle_idx_arr.size)
    grp.create_dataset('particle_idx', data=particle_idx_arr, **_GZIP_OPTS)
    grp.create_dataset('sensor_idx', data=sensor_idx_arr, **_GZIP_OPTS)
    grp.create_dataset('PE', data=pe_arr, **_GZIP_OPTS)
    grp.create_dataset('T', data=t_arr, **_GZIP_OPTS)


def save_seg_event_v3(f, event_dict, seq_idx):
    """Write a single event_NNN/ group to an already-open seg v3 file.

    Each segment row gets a local ``track_idx`` FK (0..n_tracks-1). Times are
    shifted by ``t0`` so they live in the detector frame. ``beta_start`` and
    ``n_cherenkov`` are pass-through from PhotonSim.
    """
    grp = f.create_group(_event_group_name(seq_idx))
    grp.attrs['source_event_idx'] = int(event_dict['source_event_idx'])

    mt = event_dict.get('meaningful_tracks', {})
    seg = event_dict.get('segments', {'n_segments': 0})

    n_tracks = int(len(mt))
    n_segments = int(seg.get('n_segments', 0))
    grp.attrs['n_tracks'] = n_tracks
    grp.attrs['n_segments'] = n_segments

    track_idx_per_segment = []
    for track_local_idx, t_info in enumerate(mt.values()):
        track_idx_per_segment.extend(
            [track_local_idx] * int(t_info['n_segments']))
    track_idx_arr = np.asarray(track_idx_per_segment, dtype=np.int32)
    assert track_idx_arr.size == n_segments, (
        f"track_idx length {track_idx_arr.size} != n_segments {n_segments}")

    grp.create_dataset('track_idx', data=track_idx_arr, **_GZIP_OPTS)

    def _empty(dtype): return np.array([], dtype=dtype)
    if n_segments > 0:
        fields = {
            'start_x': (seg['start_x'], np.float32),
            'start_y': (seg['start_y'], np.float32),
            'start_z': (seg['start_z'], np.float32),
            'end_x': (seg['end_x'], np.float32),
            'end_y': (seg['end_y'], np.float32),
            'end_z': (seg['end_z'], np.float32),
            'dir_x': (seg['dir_x'], np.float16),
            'dir_y': (seg['dir_y'], np.float16),
            'dir_z': (seg['dir_z'], np.float16),
            'edep': (seg['edep'], np.float32),
            'time': (np.asarray(seg['time'], dtype=np.float32), np.float32),
            'beta_start': (seg['beta_start'], np.float32),
            'n_cherenkov': (seg['n_cherenkov'], np.int32),
        }
        for name, (arr, dtype) in fields.items():
            grp.create_dataset(name,
                               data=np.asarray(arr, dtype=dtype),
                               **_GZIP_OPTS)
    else:
        for name, dtype in (('start_x', np.float32), ('start_y', np.float32),
                            ('start_z', np.float32), ('end_x', np.float32),
                            ('end_y', np.float32), ('end_z', np.float32),
                            ('dir_x', np.float16), ('dir_y', np.float16),
                            ('dir_z', np.float16), ('edep', np.float32),
                            ('time', np.float32), ('beta_start', np.float32),
                            ('n_cherenkov', np.int32)):
            grp.create_dataset(name, data=_empty(dtype), **_GZIP_OPTS)


def save_labl_event_v3(f, event_dict, seq_idx):
    """Write a single event_NNN/ group to an already-open labl v3 file.

    Subgroups:
    * ``per_event/`` — overall_containment (scalar).
    * ``per_interaction/`` — one row per primary/interaction rank:
      ``source_type``, ``t0``, ``vertex_{x,y,z}``, ``ancestor_track_id``,
      ``n_particles``. Dark events (no tracks) get a 1-row synthetic entry.
    * ``per_particle/`` — category, containment, genealogy CSR, and
      ``interaction_idx`` FK into ``per_interaction/``.
    * ``per_track/`` — track metadata + ``particle_idx`` and
      ``interaction`` FK columns.
    """
    grp = f.create_group(_event_group_name(seq_idx))
    grp.attrs['source_event_idx'] = int(event_dict['source_event_idx'])
    grp.attrs['n_particles'] = int(event_dict['n_particles'])
    mt = event_dict.get('meaningful_tracks', {})
    grp.attrs['n_tracks'] = int(len(mt))

    # Track-level derivations (also used to size per_interaction/)
    if mt:
        particle_idx = derive_particle_idx_per_track(event_dict)
        ancestor, interaction = derive_track_ancestor_and_interaction(event_dict)
    else:
        particle_idx = np.array([], dtype=np.int32)
        ancestor = np.array([], dtype=np.int32)
        interaction = np.array([], dtype=np.int32)

    # --- per_event (just the overall containment scalar now) ---
    pe_grp = grp.create_group('per_event')
    pe_grp.create_dataset('overall_containment',
                          data=np.float32(event_dict['overall_light_containment']))

    # --- per_interaction ---
    # Row `i` corresponds to tracks whose `interaction == i`. For
    # single-interaction events every row shares t0/vertex/source_type
    # (they come from the same PhotonSim stream); pile-up supplies
    # per-vertex arrays that get indexed here instead.
    pi_grp = grp.create_group('per_interaction')
    _write_per_interaction(pi_grp, event_dict, ancestor, interaction)

    # --- per_particle ---
    pp_grp = grp.create_group('per_particle')
    particles = event_dict['particles']

    cats = []
    for particle in particles:
        ti = particle.get('track_info')
        cat = ti['category'] if ti is not None else -1
        cats.append(cat if cat >= 0 else 255)
    pp_grp.create_dataset('category',
                          data=np.array(cats, dtype=np.uint8),
                          **_GZIP_OPTS)

    cont = np.asarray(event_dict['light_containment_by_particle'],
                      dtype=np.float32)
    pp_grp.create_dataset('containment', data=cont, **_GZIP_OPTS)

    gen_offsets = [0]
    gen_data_list = []
    for particle in particles:
        gen = np.asarray(particle['genealogy'], dtype=np.int32).flatten()
        gen_data_list.append(gen)
        gen_offsets.append(gen_offsets[-1] + len(gen))
    pp_grp.create_dataset('genealogy_offsets',
                          data=np.array(gen_offsets, dtype=np.uint32),
                          **_GZIP_OPTS)
    pp_grp.create_dataset('genealogy_data',
                          data=(np.concatenate(gen_data_list)
                                if gen_data_list else np.array([], dtype=np.int32)),
                          **_GZIP_OPTS)

    ext_offsets = [0]
    ext_data_list = []
    for particle in particles:
        ext = particle.get('extended_genealogy')
        arr = (np.asarray(ext, dtype=np.int32).flatten()
               if ext is not None else np.array([], dtype=np.int32))
        ext_data_list.append(arr)
        ext_offsets.append(ext_offsets[-1] + len(arr))
    pp_grp.create_dataset('ext_genealogy_offsets',
                          data=np.array(ext_offsets, dtype=np.uint32),
                          **_GZIP_OPTS)
    pp_grp.create_dataset('ext_genealogy_data',
                          data=(np.concatenate(ext_data_list)
                                if ext_data_list else np.array([], dtype=np.int32)),
                          **_GZIP_OPTS)

    # interaction_idx per particle: derived by mapping each particle's
    # last-in-genealogy (primary) track_id to its interaction rank.
    pp_grp.create_dataset(
        'interaction_idx',
        data=derive_particle_interaction_idx(event_dict, interaction),
        **_GZIP_OPTS)

    # --- per_track ---
    pt_grp = grp.create_group('per_track')
    if mt:
        track_id = np.array([t['track_id'] for t in mt.values()], dtype=np.int32)
        parent_id = np.array([t['parent_id'] for t in mt.values()], dtype=np.int32)
        pdg = np.array([t['pdg'] for t in mt.values()], dtype=np.int16)
        initial_energy = np.array([t['initial_energy'] for t in mt.values()],
                                   dtype=np.float32)
        n_ch = np.array([t['n_cherenkov'] for t in mt.values()], dtype=np.int32)
    else:
        track_id = np.array([], dtype=np.int32)
        parent_id = np.array([], dtype=np.int32)
        pdg = np.array([], dtype=np.int16)
        initial_energy = np.array([], dtype=np.float32)
        n_ch = np.array([], dtype=np.int32)

    pt_grp.create_dataset('track_id', data=track_id, **_GZIP_OPTS)
    pt_grp.create_dataset('parent_id', data=parent_id, **_GZIP_OPTS)
    pt_grp.create_dataset('pdg', data=pdg, **_GZIP_OPTS)
    pt_grp.create_dataset('initial_energy', data=initial_energy, **_GZIP_OPTS)
    pt_grp.create_dataset('n_cherenkov', data=n_ch, **_GZIP_OPTS)
    pt_grp.create_dataset('particle_idx', data=particle_idx, **_GZIP_OPTS)
    pt_grp.create_dataset('ancestor', data=ancestor, **_GZIP_OPTS)
    pt_grp.create_dataset('interaction', data=interaction, **_GZIP_OPTS)


def _write_per_interaction(pi_grp, event_dict, ancestor, interaction):
    """Populate the per_interaction/ subgroup.

    ``event_dict`` may carry ``t0`` / ``vertex_xyz`` / ``source_type``
    as scalars (single-interaction) or as per-interaction arrays
    (pile-up). In the scalar case the values are broadcast to every row.
    """
    # Distinct primary-ranks, in order
    if interaction.size > 0:
        n_interactions = int(interaction.max()) + 1
        ancestor_per_rank = np.zeros(n_interactions, dtype=np.int32)
        n_particles_per_rank = np.zeros(n_interactions, dtype=np.int32)
        for anc, rank in zip(ancestor, interaction):
            ancestor_per_rank[int(rank)] = int(anc)
        # Count particles per interaction
        part_inter = derive_particle_interaction_idx(event_dict, interaction)
        for pi in part_inter:
            if 0 <= int(pi) < n_interactions:
                n_particles_per_rank[int(pi)] += 1
    else:
        # Dark event — synthesize 1 row so per_interaction/ always exists.
        n_interactions = 1
        ancestor_per_rank = np.array([-1], dtype=np.int32)
        n_particles_per_rank = np.array([0], dtype=np.int32)

    # Broadcast event_dict['t0'|'vertex_xyz'|'source_type'] to rows.
    t0_raw = event_dict.get('t0', 0.0)
    vx_raw = event_dict.get('vertex_xyz', np.zeros(3, dtype=np.float32))
    st_raw = event_dict.get('source_type', SOURCE_TYPE_PARTICLES)

    t0_arr = np.asarray(t0_raw, dtype=np.float32).reshape(-1)
    if t0_arr.size == 1:
        t0_arr = np.full(n_interactions, float(t0_arr[0]), dtype=np.float32)
    assert t0_arr.size == n_interactions, \
        f"t0 length {t0_arr.size} != n_interactions {n_interactions}"

    vx_arr = np.asarray(vx_raw, dtype=np.float32)
    if vx_arr.ndim == 1:
        vx_arr = np.broadcast_to(vx_arr, (n_interactions, 3)).copy()
    assert vx_arr.shape == (n_interactions, 3), \
        f"vertex_xyz shape {vx_arr.shape} != ({n_interactions}, 3)"

    st_arr = np.asarray(st_raw, dtype=np.uint8).reshape(-1)
    if st_arr.size == 1:
        st_arr = np.full(n_interactions, int(st_arr[0]), dtype=np.uint8)
    assert st_arr.size == n_interactions

    pi_grp.create_dataset('source_type',       data=st_arr, **_GZIP_OPTS)
    pi_grp.create_dataset('t0',                data=t0_arr, **_GZIP_OPTS)
    pi_grp.create_dataset('vertex_x',          data=vx_arr[:, 0].copy(), **_GZIP_OPTS)
    pi_grp.create_dataset('vertex_y',          data=vx_arr[:, 1].copy(), **_GZIP_OPTS)
    pi_grp.create_dataset('vertex_z',          data=vx_arr[:, 2].copy(), **_GZIP_OPTS)
    pi_grp.create_dataset('ancestor_track_id', data=ancestor_per_rank, **_GZIP_OPTS)
    pi_grp.create_dataset('n_particles',       data=n_particles_per_rank, **_GZIP_OPTS)


def derive_particle_interaction_idx(event_dict, track_interaction=None):
    """For each particle, return the interaction index of its primary ancestor.

    Uses each particle's last-in-genealogy track_id (== its primary
    track) and looks up that track's ``interaction`` rank. Particles
    with no genealogy or whose primary isn't in the tracks table get -1.

    Parameters
    ----------
    event_dict : dict
        Must carry ``meaningful_tracks`` (dict of track_id → info) and
        ``particles`` (list with a ``genealogy`` key).
    track_interaction : np.ndarray, optional
        Cached output of ``derive_track_ancestor_and_interaction``; if
        None, recomputed.
    """
    tracks = event_dict.get('meaningful_tracks', {})
    particles = event_dict.get('particles', [])
    if not particles:
        return np.array([], dtype=np.int32)
    if not tracks:
        return np.full(len(particles), -1, dtype=np.int32)
    if track_interaction is None:
        _, track_interaction = derive_track_ancestor_and_interaction(event_dict)
    tid_to_interaction = {
        int(tid): int(track_interaction[i])
        for i, tid in enumerate(tracks.keys())
    }
    out = np.full(len(particles), -1, dtype=np.int32)
    for i, particle in enumerate(particles):
        gen = particle.get('genealogy') or []
        if gen:
            out[i] = tid_to_interaction.get(int(gen[-1]), -1)
    return out


def list_events_v3(filename):
    """Return the ``config/source_event_idx`` array from a v3 file."""
    with h5py.File(filename, 'r') as f:
        return np.asarray(f['config/source_event_idx'][:])


def _v3_group_to_dict(grp):
    """Recursively copy attrs + datasets + subgroups into a plain dict."""
    out = {}
    for key, value in grp.attrs.items():
        out[key] = value
    for key in grp.keys():
        item = grp[key]
        if isinstance(item, h5py.Dataset):
            out[key] = item[()]
        else:  # subgroup
            out[key] = _v3_group_to_dict(item)
    return out


def _read_v3_event(filename, event_idx):
    """Return the event_NNN/ group contents as a dict keyed by dataset/attr name."""
    with h5py.File(filename, 'r') as f:
        name = f'event_{int(event_idx):03d}'
        if name not in f:
            raise KeyError(
                f"Event group {name!r} not found in {filename}. "
                f"Available: {sorted(k for k in f.keys() if k.startswith('event_'))[:5]} ...")
        return _v3_group_to_dict(f[name])


def read_sensor_event_v3(filename, event_idx):
    """Read event ``event_idx`` from a sensor v3 file."""
    return _read_v3_event(filename, event_idx)


def read_inst_event_v3(filename, event_idx):
    """Read event ``event_idx`` from an inst v3 file."""
    return _read_v3_event(filename, event_idx)


def read_seg_event_v3(filename, event_idx):
    """Read event ``event_idx`` from a seg v3 file."""
    return _read_v3_event(filename, event_idx)


def read_labl_event_v3(filename, event_idx):
    """Read event ``event_idx`` from a labl v3 file.

    The returned dict contains top-level attrs plus four subdicts:
    ``per_event`` (overall_containment), ``per_interaction``
    (source_type, t0, vertex_{x,y,z}, ancestor_track_id, n_particles),
    ``per_particle`` (category, containment, genealogy CSR,
    interaction_idx), and ``per_track`` (track_id, parent_id, pdg,
    initial_energy, n_cherenkov, particle_idx, ancestor, interaction).
    """
    return _read_v3_event(filename, event_idx)


