import h5py
import numpy as np
import jax.numpy as jnp
import jax
from glob import glob
import os
import json
import sys
from tqdm import tqdm 

def base_dir_path():
    return os.path.dirname(os.path.abspath(__file__))+'/../'

def setup_matplotlib_for_notebook(force_notebook_mode=None):
    """
    Configure matplotlib for notebook display when running scripts from Jupyter.
    
    This function detects if the script is running in a Jupyter notebook environment
    and modifies matplotlib's behavior to display plots inline in the notebook
    instead of trying to show them in separate windows.
    
    Args:
        force_notebook_mode: If True, force notebook mode regardless of detection.
                            If False, force regular mode. If None, auto-detect.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    
    if force_notebook_mode:
        print("📊 Jupyter notebook environment - configuring matplotlib for inline display")
        
        # Use non-interactive backend
        matplotlib.use('Agg')
        
        # Store original show function
        original_show = plt.show
        
        def notebook_show(*args, **kwargs):
            """Custom show function that displays plots inline in notebooks."""
            if plt.get_fignums():  # If there are active figures
                import tempfile
                from IPython.display import Image, display
                
                # Create temporary file for the plot
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                
                try:
                    # Save current figure with high quality
                    plt.savefig(temp_path, dpi=150, bbox_inches='tight', facecolor='white')
                    
                    # Display in notebook
                    display(Image(temp_path))
                    
                finally:
                    # Close the figure to prevent memory leaks
                    plt.close()
                    
                    # Clean up temporary file
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass  # Ignore cleanup errors
        
        # Replace plt.show with our custom function
        plt.show = notebook_show
        
    else:
        # Not in notebook, use default behavior
        pass

def normalize_particle_type_for_path(particle_type):
    """
    Normalize particle type string to match data directory structure.

    Parameters
    ----------
    particle_type : str
        Particle type string (e.g., 'mu-', 'mu+', 'e-', 'e+', 'pi-', 'pi+')

    Returns
    -------
    str
        Normalized particle type for directory name (e.g., 'muon', 'pion', 'electron')
    """
    path_map = {
        'mu-': 'muon',
        'mu+': 'muon',
        'muon': 'muon',
        'e-': 'electron',
        'e+': 'electron',
        'electron': 'electron',
        'positron': 'electron',
        'pi-': 'pion',
        'pi+': 'pion',
        'pi0': 'pion',
        'pion': 'pion'
    }

    if particle_type in path_map:
        return path_map[particle_type]
    else:
        # Default to the input if not in map (for backward compatibility)
        return particle_type

def get_refractive_index(material='water'):
    """
    Get the refractive index for a given material.

    Parameters
    ----------
    material : str
        Material type (e.g., 'water', 'ice')

    Returns
    -------
    float
        Refractive index of the material

    Raises
    ------
    ValueError
        If material is not supported
    """
    # Refractive indices for different materials
    refractive_indices = {
        'water': 1.33,
        # Future materials can be added here:
        # 'ice': 1.31,
        # 'liquid_scintillator': 1.50,
    }

    if material not in refractive_indices:
        supported_materials = ', '.join(refractive_indices.keys())
        raise ValueError(
            f"Material '{material}' is not currently supported.\n"
            f"Supported materials: {supported_materials}\n"
            f"Development for additional materials is ongoing."
        )

    # Print warning that only water is fully supported
    if material != 'water':
        print(f"⚠️  WARNING: Material '{material}' is experimental.")
        print(f"   Only 'water' is fully validated and supported.")
        print(f"   Development for other materials is ongoing.")

    return refractive_indices[material]

def get_speed_of_light_in_material(material='water'):
    """
    Calculate the speed of light in a given material.

    Parameters
    ----------
    material : str
        Material type (e.g., 'water', 'ice')

    Returns
    -------
    float
        Speed of light in the material (m/ns)

    Notes
    -----
    The speed of light in a material is calculated as c/n where:
    - c = 0.299792 m/ns (speed of light in vacuum)
    - n = refractive index of the material

    For water (n=1.33): c/n ≈ 0.2253 m/ns
    """
    # Speed of light in vacuum (m/ns)
    SPEED_OF_LIGHT_VACUUM = 0.299792  # m/ns

    # Get refractive index for the material
    n = get_refractive_index(material)

    # Calculate speed of light in material
    return SPEED_OF_LIGHT_VACUUM / n

def unpack_t0_params(particle_type='muon', material='water'):
    """
    Load and unpack t0 timing parameters for a given particle type and material.

    Parameters
    ----------
    particle_type : str, optional
        Particle type (e.g., 'mu-', 'mu+', 'e-', 'e+', 'pi-', 'pi+', 'muon', 'pion', 'electron'), by default 'muon'
    material : str, optional
        Material type, by default 'water'

    Returns
    -------
    tuple
        (baseline_slope, baseline_intercept, A_slope, A_intercept, B_slope, B_intercept, offset)
    """
    # Normalize particle type for file path
    normalized_type = normalize_particle_type_for_path(particle_type)
    with open(base_dir_path()+f'/data/{material}/{normalized_type}/t0.json', 'r') as f:
        t0_params = json.load(f)

    # Extract individual parameters from nested dict structure
    return (
        t0_params['baseline']['slope'],
        t0_params['baseline']['intercept'],
        t0_params['delta_parameterization']['A_slope'],
        t0_params['delta_parameterization']['A_intercept'],
        t0_params['delta_parameterization']['B_slope'],
        t0_params['delta_parameterization']['B_intercept'],
        t0_params['delta_parameterization']['offset']
    )

def unpack_photonsim_params(particle_type='muon', material='water'):
    """
    Load and unpack photon simulation parameters for a given particle type and material.

    Parameters
    ----------
    particle_type : str, optional
        Particle type (e.g., 'mu-', 'mu+', 'e-', 'e+', 'pi-', 'pi+', 'muon', 'pion', 'electron'), by default 'muon'
    material : str, optional
        Material type, by default 'water'

    Returns
    -------
    dict
        Dictionary containing:
        - 'tot_n_photons_normalization': tuple of (a, b, c) for power law: a * energy^b + c
        - 'num_seeds': tuple of (a, b, c) for power law: a * energy^b + c
        - 'siren_model_path': str, absolute path to SIREN model
    """
    # Normalize particle type for file path
    normalized_type = normalize_particle_type_for_path(particle_type)
    config_path = base_dir_path()+f'/data/{material}/{normalized_type}/photonsim_params.json'

    with open(config_path, 'r') as f:
        photonsim_params = json.load(f)

    # Construct absolute path to SIREN model
    data_dir = base_dir_path()+f'/data/{material}/{normalized_type}/'
    siren_model_path = data_dir + photonsim_params['siren_model']['path']

    # Extract individual parameters (power law: a * x^b + c)
    return {
        'tot_n_photons_normalization': (
            photonsim_params['tot_n_photons_normalization']['a'],
            photonsim_params['tot_n_photons_normalization']['b'],
            photonsim_params['tot_n_photons_normalization']['c']
        ),
        'num_seeds': (
            photonsim_params['num_seeds']['a'],
            photonsim_params['num_seeds']['b'],
            photonsim_params['num_seeds']['c']
        ),
        'siren_model_path': siren_model_path
    }

def spherical_to_cartesian(theta, phi):
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Parameters:
    theta (float): Inclination angle in radians (0 = z-axis, pi/2 = xy-plane)
    phi (float): Azimuthal angle in radians (0 = x-axis, pi/2 = y-axis)
    
    Returns:
    jnp.array: Unit vector [x, y, z] in Cartesian coordinates
    """
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    
    x = sin_theta * cos_phi
    y = sin_theta * sin_phi
    z = cos_theta
    
    return jnp.array([x, y, z])

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
    particle_params : tuple
        if calibration_mode is True:
            (source_position, source_intensity)
        if calibration_mode is False:
            (track_energy, track_origin, track_direction)
    sensor_params : tuple
        (scatter_length, reflection_rate, absorption_length, sim_temperature)
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
        import os
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
            params_group.create_dataset('source_position', data=np.array(particle_params[0]))
            params_group.create_dataset('source_intensity', data=np.array(particle_params[1]))

        else:
            params_group = f.create_group('particle_params')
            params_group.create_dataset('track_energy', data=np.array(particle_params[0]))
            params_group.create_dataset('track_origin', data=np.array(particle_params[1]))
            params_group.create_dataset('track_direction', data=np.array(particle_params[2]))

        params_group = f.create_group('sensor_params')
        params_group.create_dataset('scatter_length', data=np.array(sensor_params[0]))
        params_group.create_dataset('reflection_rate', data=np.array(sensor_params[1]))
        params_group.create_dataset('absorption_length', data=np.array(sensor_params[2]))
        params_group.create_dataset('sim_temperature', data=np.array(sensor_params[3]))

        # # Save event data and number
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
    particle_params : tuple
        if calibration_mode is True:
            (source_position, source_intensity)
        if calibration_mode is False:
            (track_energy, track_origin, track_direction)
    sensor_params : tuple
        (scatter_length, reflection_rate, absorption_length, sim_temperature)
    event_number : int
        Event identifier number
    If sparse=True:
        indices : jnp.ndarray
            Detector indices with non-zero values
        charges : jnp.ndarray
            Charge values at these indices
        times : jnp.ndarray
            Time values at these indices
    If sparse=False:
        charges : jnp.ndarray
            Full array of charges for all sensors
        times : jnp.ndarray
            Full array of times for all sensors
    """
    with h5py.File(filename, 'r') as f:
        if calibration_mode:
            # Load calibration parameters
            params_group = f['calibration_params']
            source_position = jnp.array(params_group['source_position'][()])
            source_intensity = jnp.array(params_group['source_intensity'][()])

            particle_params = (source_position, source_intensity)
        else:
            # Load particle parameters
            params_group = f['particle_params']
            track_energy = jnp.array(params_group['track_energy'][()])
            track_origin = jnp.array(params_group['track_origin'][()])
            track_direction = jnp.array(params_group['track_direction'][()])

            particle_params = (track_energy, track_origin, track_direction)

        # Load detector parameters
        detector_group = f['sensor_params']
        scatter_length = jnp.array(detector_group['scatter_length'][()])
        reflection_rate = jnp.array(detector_group['reflection_rate'][()])
        absorption_length = jnp.array(detector_group['absorption_length'][()])
        sim_temperature = jnp.array(detector_group['sim_temperature'][()])

        sensor_params = (scatter_length, reflection_rate, absorption_length, sim_temperature)

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


import jax
import jax.numpy as jnp

def generate_random_params(key, h=2, r=1):
    """
    Generate random parameters for particle simulation using angles for direction.
    
    Parameters:
    key: JAX PRNG key
    
    Returns:
    tuple: (energy, position, direction_angles, intensity)
        - energy: scalar energy value in MeV
        - position: 3D position vector [x, y, z]
        - direction_angles: tuple of (theta, phi) in radians
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    # Generate energy between 100 and 1000 MeV
    energy = 300. + 600. * jax.random.uniform(k1)
    
    # Generate random position inside detector volume (approximated as cylinder)
    position = generate_random_point_inside_cylinder(k2, h, r)
    
    # Generate random direction angles
    # theta: inclination angle (0 to pi)
    # phi: azimuthal angle (0 to 2*pi)
    theta = jnp.pi * jax.random.uniform(k3)
    phi = 2.0 * jnp.pi * jax.random.uniform(k4)
    direction_angles = jnp.array([theta, phi])
    
    return energy, position, direction_angles

@jax.jit
def generate_random_point_inside_cylinder(key, h=2, r=1, offset = 0.1):
    """
    Generate random point inside a cylinder with specified height and radius.

    Parameters
    ----------
    key : jax.random.PRNGKey
        JAX random number generator key
    h : float, optional
        Height of the cylinder, default=2
        Position will be generated in range [-h/2, h/2] for z-coordinate
    r : float, optional
        Radius of the cylinder, default=1
        Position will be generated within circle of radius r in xy-plane
    offset : float, optional
        Offset to avoid generating points on the cylinder surface, default=0.1

    Returns
    -------
    array(3,)
        Random position coordinates inside cylinder of height h and radius r
    """
    # Split the key for independent random operations
    key1, key2, key3 = jax.random.split(key, 3)

    effective_radius = r - offset
    effective_height = h - offset

    # Generate cylindrical coordinates
    # Random radius from 0 to r (using square root for uniform distribution in circle)
    radius = effective_radius * jnp.sqrt(jax.random.uniform(key1, shape=()))
    # Random angle from 0 to 2π
    theta = jax.random.uniform(key2, shape=(), minval=0, maxval=2*jnp.pi)
    # Random height from -h/2 to h/2
    z = jax.random.uniform(key3, shape=(), minval=-effective_height/2, maxval=effective_height/2)

    # Convert cylindrical to Cartesian coordinates
    return jnp.array([
        radius * jnp.cos(theta),  # x
        radius * jnp.sin(theta),  # y
        z                         # z
    ])


def print_particle_params(trk_params):
    """
    Print particle parameters in a readable format.
    
    Parameters:
    trk_params: tuple of (energy, position, direction_angles)
    """
    energy, position, direction_angles = trk_params
    theta, phi = direction_angles
    
    # Convert angles to Cartesian for display
    direction = spherical_to_cartesian(theta, phi)
    
    print("Particle Parameters:")
    print(f"  Energy: {energy:.2f} MeV")
    print(f"  Position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}] m")
    print(f"  Direction angles: theta={theta:.2f} rad, phi={phi:.2f} rad")
    print(f"  Direction vector: [{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]")

def print_propagation_params(sensor_params):
    """
    Pretty print the detector parameters.

    Parameters
    ----------
    sensor_params : dict
        Dictionary containing detector parameters

    Returns
    -------
    None
        Prints formatted parameter information to stdout

    Example
    -------
     sensor_params = (
        jnp.array(10.),         # scatter_length
        jnp.array(0.1),         # reflection_rate
        jnp.array(10.),         # absorption_length
        jnp.array(0.1)          # sim_temperature
    )
        print_propagation_params(sensor_params)
    Propagation Parameters:
    ───────────────────────
    Scatter Length: 10.00 m
    Reflection Rate: 0.10
    Absorption Length: 10.00 m
    Simulation Temperature for Gumbel-Softmax: 0.10
    ───────────────────────
    """
    # Unpack the parameter tuple
    scatter_length, reflection_rate, absorption_length, sim_temperature = sensor_params

    # Create formatted output with consistent decimal places
    print("Propagation Parameters:")
    print("─" * 20)
    print(f"Scatter Length: {scatter_length:.2f} m")
    print(f"Reflection Rate: {reflection_rate:.2f}")
    print(f"Absorption Length: {absorption_length:.2f} m")
    print(f"Simulation Temperature for Gumbel-Softmax: {sim_temperature:.4f}")
    print("─" * 20)

def superimpose_multiple_events(charges_list, times_list):
    """
    Superimpose multiple events by summing charges and calculating weighted average of times.
    
    Parameters
    ----------
    charges_list : list of jnp.ndarray
        List of charge arrays from each event
    times_list : list of jnp.ndarray
        List of time arrays from each event
        
    Returns
    -------
    tuple
        (combined_charges, combined_times)
    """
    if not charges_list or not times_list:
        raise ValueError("Empty charges or times list")
    
    if len(charges_list) != len(times_list):
        raise ValueError("charges_list and times_list must have the same length")
    
    # Initialize with the first event
    combined_charges = charges_list[0]
    combined_times = times_list[0]
    
    # Iteratively combine with subsequent events
    for i in range(1, len(charges_list)):
        # Sum the charges
        combined_charges = combined_charges + charges_list[i]
        
        # Calculate weighted average of times
        # Start with the product of the previous combined values
        time_product = combined_times * (combined_charges - charges_list[i])
        
        # Add the product for the current event
        time_product = time_product + times_list[i] * charges_list[i]
        
        # Divide by combined charges to get weighted average
        # When charge is 0, use 0 for time to avoid division by zero
        nonzero_mask = combined_charges > 0
        
        # Initialize combined times with zeros
        new_combined_times = jnp.zeros_like(combined_times)
        
        # Only calculate weighted average where there are non-zero charges
        weighted_times = jnp.where(
            nonzero_mask,
            time_product / combined_charges,
            0.0
        )
        
        # Apply the weighted times only where we have non-zero charges
        combined_times = jnp.where(nonzero_mask, weighted_times, new_combined_times)
    
    return combined_charges, combined_times

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
    import h5py
    
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

def save_single_event_with_label_info(extended_info, event_number=0, filename=None):
    """
    Save a single event with label-based structure to an HDF5 file.

    This function saves events generated from PhotonSim's label system where photons
    are classified by genealogy. The HDF5 structure includes:

    Label-level arrays:
    - Q_per_label (N_labels, N_sensors): Charge contribution per label
    - T_per_label (N_labels, N_sensors): Time per label
    - Label_Genealogy: Variable-length array of genealogies
    - Label_Category: Category code for each label
    - Track kinematics per label: Position, Direction, Energy, Time, PDG

    Aggregated values:
    - Q_true (N_sensors): Sum of charges across all labels
    - T_true (N_sensors): Min of times across all labels
    - Q_reco (N_sensors): Optionally smeared Q_true
    - T_reco (N_sensors): Optionally smeared T_true

    Parameters
    ----------
    extended_info : dict
        Dictionary containing label-based event information
    event_number : int, optional
        Event number, by default 0
    filename : str, optional
        Output filename, by default None (generates event_{number}.h5)

    Returns
    -------
    str
        Path to saved file
    """
    import h5py
    import numpy as np

    # If no filename provided, generate one
    if filename is None:
        filename = f'event_{event_number}.h5'

    n_labels = extended_info['n_labels']
    labels = extended_info['labels']
    Q_per_label = extended_info['Q_per_label']
    T_per_label = extended_info['T_per_label']
    Q_true = extended_info['Q_true']
    T_true = extended_info['T_true']
    Q_reco = extended_info['Q_reco']
    T_reco = extended_info['T_reco']

    # Extract track info for each label
    label_categories = []
    label_pdgs = []
    label_positions = []
    label_directions = []
    label_energies = []
    label_times = []
    label_parent_ids = []
    label_genealogies = []

    for label in labels:
        track_info = label['track_info']
        genealogy = label['genealogy']

        if track_info is not None:
            label_categories.append(track_info['category'])
            label_pdgs.append(track_info['pdg'])
            label_positions.append(track_info['position'])
            label_directions.append(track_info['direction'])
            label_energies.append(track_info['energy'])
            label_times.append(track_info['time'])
            label_parent_ids.append(track_info['parent_id'])
        else:
            # Fallback if no track info
            label_categories.append(-1)
            label_pdgs.append(0)
            label_positions.append([0.0, 0.0, 0.0])
            label_directions.append([0.0, 0.0, 1.0])
            label_energies.append(0.0)
            label_times.append(0.0)
            label_parent_ids.append(-1)

        label_genealogies.append(genealogy)

    # Convert to numpy arrays
    label_categories = np.array(label_categories, dtype=np.int32)
    label_pdgs = np.array(label_pdgs, dtype=np.int32)
    label_positions = np.array(label_positions, dtype=np.float64)
    label_directions = np.array(label_directions, dtype=np.float64)
    label_energies = np.array(label_energies, dtype=np.float64)
    label_times = np.array(label_times, dtype=np.float64)
    label_parent_ids = np.array(label_parent_ids, dtype=np.int32)

    # Category names mapping
    category_names = {
        0: 'Primary',
        1: 'DecayElectron',
        2: 'SecondaryPion',
        3: 'GammaShower',
        -1: 'Unknown'
    }
    label_category_names = [category_names.get(cat, 'Unknown') for cat in label_categories]

    with h5py.File(filename, 'w') as f:
        # Save Q and T arrays
        f.create_dataset('Q_per_label', data=np.array(Q_per_label))  # (N_labels, N_sensors)
        f.create_dataset('T_per_label', data=np.array(T_per_label))  # (N_labels, N_sensors)
        f.create_dataset('Q_true', data=np.array(Q_true))            # (N_sensors,)
        f.create_dataset('T_true', data=np.array(T_true))            # (N_sensors,)
        f.create_dataset('Q_reco', data=np.array(Q_reco))            # (N_sensors,)
        f.create_dataset('T_reco', data=np.array(T_reco))            # (N_sensors,)

        # Save label information
        f.create_dataset('n_labels', data=n_labels)
        f.create_dataset('Label_Category', data=label_categories)

        # Save category names as variable-length strings
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('Label_CategoryName', data=label_category_names, dtype=dt)

        # Save track kinematics
        f.create_dataset('Track_PDG', data=label_pdgs)
        f.create_dataset('Track_Position', data=label_positions)    # (N_labels, 3)
        f.create_dataset('Track_Direction', data=label_directions)  # (N_labels, 3)
        f.create_dataset('Track_Energy', data=label_energies)
        f.create_dataset('Track_Time', data=label_times)
        f.create_dataset('Track_ParentID', data=label_parent_ids)

        # Save genealogies as variable-length array
        # Create a special dtype for variable-length integer arrays
        vlen_int_dtype = h5py.vlen_dtype(np.dtype('int32'))
        genealogy_data = np.array([np.array(g, dtype=np.int32) for g in label_genealogies], dtype=object)
        f.create_dataset('Label_Genealogy', data=genealogy_data, dtype=vlen_int_dtype)

        # Save metadata (ensure scalars are numpy types)
        f.create_dataset('event_number', data=np.int32(event_number))
        f.create_dataset('primary_energy', data=np.float64(extended_info['primary_energy']))
        f.create_dataset('apply_smearing', data=np.bool_(extended_info['apply_smearing']))

        # Save source information (also ensure proper types for attributes)
        f.attrs['source'] = extended_info['source']
        f.attrs['n_labels'] = np.int32(n_labels)
        f.attrs['n_sensors'] = np.int32(Q_true.shape[0])

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
    import h5py
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

                # Copy all datasets from individual file to the group
                for key in f.keys():
                    event_group.create_dataset(key, data=f[key][()])

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
    
    # Calculate total energy: E² = p² + m²
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
    file_paths = glob(os.path.join(directory, pattern))
    
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
    - Uses relativistic energy-momentum relation: E² = p² + m²
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
    # E² = p² + m²
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


def load_range_params(particle, medium):
    """
    Load range parametrization for a given particle and medium.

    Args:
        particle: Particle type (e.g., 'muon', 'electron')
        medium: Medium type (e.g., 'water')

    Returns:
        dict: Range parameters containing 'a', 'b' coefficients and metadata

    The range formula is: Range = a * E + b
    where E is energy in MeV and Range is in mm.
    """
    base_dir = base_dir_path()
    params_file = os.path.join(base_dir, 'data', medium, particle, 'range_params.json')

    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Range parameters file not found: {params_file}")

    with open(params_file, 'r') as f:
        params = json.load(f)

    return params


def calculate_particle_range(energy_mev, range_params):
    """
    Calculate particle range in the medium given energy.

    Args:
        energy_mev: Particle energy in MeV
        range_params: Range parameters dict from load_range_params()

    Returns:
        float: Range in meters

    The input parametrization is in mm, this function returns range in meters.
    """
    a = range_params['parameters']['a']
    b = range_params['parameters']['b']

    # Calculate range in mm
    range_mm = a * energy_mev + b

    # Convert to meters
    range_m = range_mm / 1000.0

    return range_m


def check_track_endpoint_in_detector(position, direction, energy_mev, range_params, detector_bounds, fraction=0.9):
    """
    Check if the track endpoint (position + range * direction) is within detector bounds.

    Args:
        position: Track starting position [x, y, z] in meters
        direction: Track direction (unit vector)
        energy_mev: Particle energy in MeV
        range_params: Range parameters from load_range_params()
        detector_bounds: Dict with 'r' (radius) and 'H' (height) in meters
        fraction: Fraction of detector bounds to check (default 0.9)

    Returns:
        bool: True if endpoint is within bounds, False otherwise
    """
    # Calculate range in meters
    track_range = calculate_particle_range(energy_mev, range_params)

    # Calculate endpoint
    endpoint = position + track_range * direction

    # Extract detector bounds
    detector_r = detector_bounds['r'] * fraction
    detector_h = detector_bounds['H'] * fraction

    # Check cylindrical bounds
    # Radial check (x, y)
    radial_distance = np.sqrt(endpoint[0]**2 + endpoint[1]**2)
    if radial_distance > detector_r:
        return False

    # Height check (z)
    if abs(endpoint[2]) > detector_h / 2.0:
        return False

    return True


def generate_random_event_params(key, detector_bounds, fraction=0.7):
    """
    Generate random event parameters based on detector geometry.
    
    This is the same function from optimize.py, shared for consistency.
    """
    if detector_bounds['type'] == 'cylinder':
        r_vert = jax.random.uniform(key, shape=(), minval=0, maxval=detector_bounds['r'] * fraction)
        key, _ = jax.random.split(key)
        theta = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
        key, _ = jax.random.split(key)
        z_vert = jax.random.uniform(key, shape=(), minval=-detector_bounds['H']/2 * fraction, 
                                   maxval=detector_bounds['H']/2 * fraction)
        position = jnp.array([r_vert * jnp.cos(theta), r_vert * jnp.sin(theta), z_vert])
        
    elif detector_bounds['type'] == 'sphere':
        u = jax.random.uniform(key, shape=())
        key, _ = jax.random.split(key)
        cos_theta = jax.random.uniform(key, shape=(), minval=-1, maxval=1)
        key, _ = jax.random.split(key)
        phi = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
        
        r = detector_bounds['r'] * fraction * jnp.cbrt(u)
        sin_theta = jnp.sqrt(1 - cos_theta**2)
        position = jnp.array([r * sin_theta * jnp.cos(phi), 
                             r * sin_theta * jnp.sin(phi), 
                             r * cos_theta])
        
    elif detector_bounds['type'] == 'box':
        position = jax.random.uniform(key, shape=(3,), 
                                    minval=jnp.array([-detector_bounds['x']/2, 
                                                     -detector_bounds['y']/2, 
                                                     -detector_bounds['z']/2]) * fraction,
                                    maxval=jnp.array([detector_bounds['x']/2, 
                                                     detector_bounds['y']/2, 
                                                     detector_bounds['z']/2]) * fraction)
    
    # Random direction
    key, _ = jax.random.split(key)
    phi = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
    key, _ = jax.random.split(key)
    cos_theta = jax.random.uniform(key, shape=(), minval=-1, maxval=1)
    sin_theta = jnp.sqrt(1 - cos_theta**2)
    direction = jnp.array([sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), cos_theta])
    
    # Random energy
    key, _ = jax.random.split(key)
    energy = jax.random.uniform(key, shape=(), minval=500.0, maxval=1500.0)
    
    return position, direction, energy