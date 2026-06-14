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
        ``(a_coeffs, l_coeffs, b_coeffs)`` — the three length-4 cubic-in-log10E coefficient
        lists of the ``stretched_exp_delay`` t0 model (see :func:`lucid.sources.siren_rays.predict_t0`).
    """
    # Normalize particle type for file path
    normalized_type = normalize_particle_type_for_path(particle_type)
    with open(base_dir_path()+f'/data/{material}/{normalized_type}/t0.json', 'r') as f:
        t0_params = json.load(f)
    return (
        t0_params['A']['log10_poly_logE'],
        t0_params['lambda']['log10_poly_logE'],
        t0_params['beta']['poly_logE'],
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


def generate_random_params(key, h=2, r=1):
    """
    Generate random parameters for particle simulation using angles for direction.

    Parameters:
    key: JAX PRNG key

    Returns:
    ParticleParams: with energy, position, theta, phi, t0
    """
    from lucid.detector_params import ParticleParams

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

    return ParticleParams(energy=energy, position=position,
                          theta=theta, phi=phi, t0=jnp.array(0.0))

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
    trk_params: ParticleParams with energy, position, theta, phi, t0
    """
    # Convert angles to Cartesian for display
    direction = spherical_to_cartesian(trk_params.theta, trk_params.phi)

    print("Particle Parameters:")
    print(f"  Energy: {trk_params.energy:.2f} MeV")
    print(f"  Position: [{trk_params.position[0]:.2f}, {trk_params.position[1]:.2f}, {trk_params.position[2]:.2f}] m")
    print(f"  Direction angles: theta={trk_params.theta:.2f} rad, phi={trk_params.phi:.2f} rad")
    print(f"  Direction vector: [{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]")
    print(f"  t0: {trk_params.t0:.4f}")

def print_propagation_params(sensor_params):
    """
    Pretty print the detector parameters.

    Parameters
    ----------
    sensor_params : DetectorParams
        DetectorParams NamedTuple with named fields

    Returns
    -------
    None
        Prints formatted parameter information to stdout
    """
    print("Propagation Parameters:")
    print("─" * 20)
    print(f"Scatter Length: {sensor_params.scattering.scatter_length:.2f} m")
    print(f"Wall Reflection Rate: {sensor_params.reflection.wall_reflection_rate:.2f}")
    print(f"Sensor Reflection Rate: {sensor_params.reflection.sensor_reflection_rate:.2f}")
    print(f"Absorption Length: {sensor_params.absorption.absorption_length:.2f} m")
    print(f"QE: {sensor_params.response.qe:.4f}")
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

# ---------------------------------------------------------------------------
# The following I/O functions were moved to lucid.sources.event_io in Phase 2.5:
#   save_single_event, load_single_event, get_random_root_entry_index,
#   read_photon_data_from_root, get_pdg_code, get_particle_mass,
#   save_single_event_with_extended_info, save_single_event_with_particle_info,
#   merge_event_files, read_multi_folder_events, read_event_file,
#   extract_particle_properties, analyze_loaded_particle, analyze_event_directory,
#   PARTICLE_MASSES, momentum_to_angles_and_energy, analyze_event_kinematics,
#   print_event_kinematics, full_to_sparse (copy), sparse_to_full (copy)
# ---------------------------------------------------------------------------

# Backward-compat re-exports so existing imports keep working
from lucid.sources.event_io import (               # noqa: F401
    save_single_event,
    load_single_event,
    get_random_root_entry_index,
    read_photon_data_from_root,
    get_pdg_code,
    get_particle_mass,
    save_single_event_with_extended_info,
    save_single_event_with_particle_info,
    merge_event_files,
    read_multi_folder_events,
    read_event_file,
    extract_particle_properties,
    analyze_loaded_particle,
    analyze_event_directory,
    PARTICLE_MASSES,
    momentum_to_angles_and_energy,
    analyze_event_kinematics,
    print_event_kinematics,
)

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
    from lucid.detector_params import ParticleParams
    key, _ = jax.random.split(key)
    phi = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
    key, _ = jax.random.split(key)
    cos_theta = jax.random.uniform(key, shape=(), minval=-1, maxval=1)
    sin_theta = jnp.sqrt(1 - cos_theta**2)
    direction = jnp.array([sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), cos_theta])

    # Random energy
    key, _ = jax.random.split(key)
    energy = jax.random.uniform(key, shape=(), minval=500.0, maxval=1500.0)

    return ParticleParams.from_cartesian(energy=energy, position=position,
                                         direction=direction, t0=jnp.array(0.0))


def smear_times(times, time_resolution=0.4, key=None):
    """
    Gaussianly smear input times.
    The default time resoluition is that of SK.
    Reference: https://arxiv.org/pdf/1307.0162

    Args:
        times: array of input times (e.g., per detector).
        time_resolution: standard deviation (in ns) of Gaussian smearing.
        key: JAX random key for reproducibility.
    Returns:
        smeared_times: Gaussian-smeared times.
    """
    noise = jax.random.normal(key, shape=times.shape) * time_resolution
    smeared_times = times + noise
    smeared_times = jnp.where((jnp.isfinite(smeared_times)), smeared_times, 1e6)

    return smeared_times


def smear_charges_SK_like(counts, key=None):
    """
    Gaussianly smear input charge counts according to Super-Kamiokande-like resolution.

    Reference: https://arxiv.org/pdf/1307.0162 (Table 2)

    Args:
        counts: array of input charge counts (e.g., number of hits per PMT).
        key: JAX random key for reproducibility.

    Returns:
        smeared_counts: Gaussian-smeared charge counts.
    """
    if key is None:
        raise ValueError("key must be provided for reproducibility.")

    #Define sigma according to the count range
    sigma = jnp.where(
        counts < 20,
        counts * 0.012,
        jnp.where(counts < 130, counts * 0.0075, counts * 0.005)
    )

    # Apply Gaussian smearing
    noise = jax.random.normal(key, shape=counts.shape) * sigma
    smeared_counts = counts + noise

    # Handle non-finite results (e.g., NaN or inf)
    smeared_counts = jnp.where(jnp.isfinite(smeared_counts), smeared_counts, 0.0)

    # Avoid negative or unphysical charge values
    smeared_counts = jnp.clip(smeared_counts, 0.0, None)

    return smeared_counts


def time_digitizer(times, time_resolution=0.4):
    """
    Digitize input times to bin centers.

    Args:
        times: Input array of times to digitize
        time_resolution: Time resolution for binning (default=0.4 ns, Super-Kamiokande PMT resolution)

    Returns:
        Array with same shape as input where each time is replaced
        by its corresponding bin center
    """
    time_window = 500  # nanoseconds
    nbins = int(time_window / time_resolution)
    bins = jnp.linspace(0, time_window, int(nbins + 1))
    bin_centers = bins[:-1] + (bins[1] - bins[0]) / 2

    # Find which bin each time falls into
    bin_indices = jnp.digitize(times, bins) - 1
    digitized_times = jnp.where((jnp.isfinite(times)), bin_centers[bin_indices], 1e6)

    return digitized_times


# ---------------------------------------------------------------------------
# Shared math helpers (moved from generate.py during Phase 2.2 refactor)
# ---------------------------------------------------------------------------

def jax_rotate_vector(vector, axis, angle):
    """Rotate a vector around an axis by a given angle using Rodrigues' formula.

    The axis is normalized internally for safety.
    """
    norm = jnp.linalg.norm(axis)
    axis = jnp.where(norm > 1e-8, axis / norm, axis)
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    cross_product = jnp.cross(axis, vector)
    dot_product = jnp.dot(axis, vector) * (1 - cos_angle)
    return cos_angle * vector + sin_angle * cross_product + dot_product * axis


# Backward-compat alias
jax_rotate_vector_local = jax_rotate_vector

def normalize(v, epsilon=1e-8):
    """Normalize a vector (or batch of vectors) with numerical stability.

    Parameters
    ----------
    v : jnp.ndarray
        Input vector or batch of vectors to normalize.
    epsilon : float, optional
        Small constant for numerical stability, by default 1e-8.

    Returns
    -------
    jnp.ndarray
        Normalized vector(s), same shape as input.
    """
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + epsilon)


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