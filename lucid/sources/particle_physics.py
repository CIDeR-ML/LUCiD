"""Particle physics utilities: PDG codes, masses, and kinematic analysis.

Standalone module with no internal LUCiD dependencies. Provides
``PARTICLE_MASSES``, ``get_pdg_code``, ``get_particle_mass``,
``extract_particle_properties``, ``analyze_loaded_particle``,
``analyze_event_directory``, ``momentum_to_angles_and_energy``,
``analyze_event_kinematics``, ``print_event_kinematics``, and
``derive_particle_interaction_idx``.
"""
from __future__ import annotations

import os
from glob import glob as _glob

import jax.numpy as jnp
import numpy as np


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


def derive_particle_interaction_idx(event_dict, track_interaction=None):
    """For each particle, return the interaction index of its primary ancestor.

    Uses each particle's last-in-genealogy track_id (== its primary
    track) and looks up that track's ``interaction`` rank. Particles
    with no genealogy or whose primary isn't in the tracks table get -1.

    Parameters
    ----------
    event_dict : dict
        Must carry ``meaningful_tracks`` (dict of track_id -> info) and
        ``particles`` (list with a ``genealogy`` key).
    track_interaction : np.ndarray, optional
        Cached output of ``derive_track_ancestor_and_interaction``; if
        None, recomputed.
    """
    from lucid.sources.event_builder import derive_track_ancestor_and_interaction

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
