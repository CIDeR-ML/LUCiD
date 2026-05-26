from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jax
import os
import json
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lucid.detector_params import DetectorParams, ParticleParams

def base_dir_path() -> str:
    """Return the absolute path to the LUCiD project root directory."""
    return os.path.dirname(os.path.abspath(__file__))+'/../'

def setup_matplotlib_for_notebook(force_notebook_mode: bool | None = None) -> None:
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

def normalize_particle_type_for_path(particle_type: str) -> str:
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

def get_refractive_index(material: str = 'water') -> float:
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

    if material != 'water':
        print(f"⚠️  WARNING: Material '{material}' is experimental.")
        print(f"   Only 'water' is fully validated and supported.")
        print(f"   Development for other materials is ongoing.")

    return refractive_indices[material]

def get_speed_of_light_in_material(material: str = 'water') -> float:
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
    SPEED_OF_LIGHT_VACUUM = 0.299792  # m/ns

    n = get_refractive_index(material)

    return SPEED_OF_LIGHT_VACUUM / n

def unpack_t0_params(particle_type: str = 'muon', material: str = 'water') -> tuple[float, ...]:
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

def unpack_siren_params(particle_type: str = 'muon', material: str = 'water') -> dict:
    """
    Load SIREN inference parameters for a given particle type and material from
    `data/<material>/<particle>/siren_params.json`.

    `N_photons(E)` and `s_max(E)` come from the trained model's metadata (the
    `nphot` / `smax` blocks); this loader resolves the model paths and the
    ray-sampling knobs for both the Cherenkov and dE/dx surrogates.

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
        - 'siren_model_path': str, absolute path to the Cherenkov SIREN model.
        - 'ray_sampling': dict, ``{'grid_bins': int, 'threshold': float}`` —
          first-pass grid resolution and seed threshold for the Cherenkov
          importance-sampling ray generator (empty dict if not configured).
        - 'dedx_model_path': str or None, absolute path to the dE/dx SIREN
          model (None if no ``dedx_model`` block is present — only required
          when the medium enables scintillation).
        - 'dedx_sampling': dict, same shape as ``ray_sampling``, driving the
          scintillation surrogate's first-pass grid.
    """
    normalized_type = normalize_particle_type_for_path(particle_type)
    data_dir = base_dir_path() + f'/data/{material}/{normalized_type}/'
    config_path = data_dir + 'siren_params.json'

    with open(config_path, 'r') as f:
        siren_params = json.load(f)

    dedx_block = siren_params.get('dedx_model')
    dedx_path = data_dir + dedx_block['path'] if dedx_block else None

    return {
        'siren_model_path': data_dir + siren_params['siren_model']['path'],
        'ray_sampling': siren_params.get('ray_sampling', {}),
        'dedx_model_path': dedx_path,
        'dedx_sampling': siren_params.get('dedx_sampling', {}),
    }

def spherical_to_cartesian(theta: jax.Array, phi: jax.Array) -> jax.Array:
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

def full_to_sparse(charges: jax.Array, times: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
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


def sparse_to_full(sparse_indices: jax.Array, sparse_values: jax.Array, full_size: int) -> jax.Array:
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


def generate_random_params(key: jax.Array, h: float = 2, r: float = 1) -> ParticleParams:
    """
    Generate random parameters for particle simulation using angles for direction.

    Parameters:
    key: JAX PRNG key

    Returns:
    ParticleParams: with energy, position, theta, phi, t0
    """
    from lucid.detector_params import ParticleParams

    k1, k2, k3, k4 = jax.random.split(key, 4)

    energy = 300. + 600. * jax.random.uniform(k1)

    # Generate random position inside detector volume (approximated as cylinder)
    position = generate_random_point_inside_cylinder(k2, h, r)

    theta = jnp.pi * jax.random.uniform(k3)
    phi = 2.0 * jnp.pi * jax.random.uniform(k4)

    return ParticleParams(energy=energy, position=position,
                          theta=theta, phi=phi, t0=jnp.array(0.0))

@jax.jit
def generate_random_point_inside_cylinder(key: jax.Array, h: float = 2, r: float = 1, offset: float = 0.1) -> jax.Array:
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
    key1, key2, key3 = jax.random.split(key, 3)

    effective_radius = r - offset
    effective_height = h - offset

    # Using square root for uniform distribution in circle
    radius = effective_radius * jnp.sqrt(jax.random.uniform(key1, shape=()))
    theta = jax.random.uniform(key2, shape=(), minval=0, maxval=2*jnp.pi)
    z = jax.random.uniform(key3, shape=(), minval=-effective_height/2, maxval=effective_height/2)

    return jnp.array([
        radius * jnp.cos(theta),
        radius * jnp.sin(theta),
        z
    ])


def print_particle_params(trk_params: ParticleParams) -> None:
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

def print_propagation_params(sensor_params: DetectorParams) -> None:
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
    print(f"Scatter Length: {sensor_params.scatter_length:.2f} m")
    print(f"Wall Reflection Rate: {sensor_params.wall_reflection_rate:.2f}")
    print(f"Sensor Reflection Rate: {sensor_params.sensor_reflection_rate:.2f}")
    print(f"Absorption Length: {sensor_params.absorption_length:.2f} m")
    print(f"QE: {sensor_params.qe:.4f}")
    print("─" * 20)

def superimpose_multiple_events(charges_list: list[jax.Array], times_list: list[jax.Array]) -> tuple[jax.Array, jax.Array]:
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
    
    combined_charges = charges_list[0]
    combined_times = times_list[0]
    
    for i in range(1, len(charges_list)):
        combined_charges = combined_charges + charges_list[i]

        time_product = combined_times * (combined_charges - charges_list[i])
        time_product = time_product + times_list[i] * charges_list[i]

        # When charge is 0, use 0 for time to avoid division by zero
        nonzero_mask = combined_charges > 0

        new_combined_times = jnp.zeros_like(combined_times)

        weighted_times = jnp.where(
            nonzero_mask,
            time_product / combined_charges,
            0.0
        )

        combined_times = jnp.where(nonzero_mask, weighted_times, new_combined_times)
    
    return combined_charges, combined_times

# ---------------------------------------------------------------------------
# The following I/O functions live in lucid.sources.event_io (Phase 2.5+):
#   save_single_event, load_single_event, get_random_root_entry_index,
#   read_photon_data_from_root, get_pdg_code, get_particle_mass,
#   extract_particle_properties, analyze_loaded_particle, analyze_event_directory,
#   PARTICLE_MASSES, momentum_to_angles_and_energy, analyze_event_kinematics,
#   print_event_kinematics, full_to_sparse (copy), sparse_to_full (copy)
# ---------------------------------------------------------------------------

# Backward-compat re-exports so existing ``from lucid.utils import X``
# keeps working.  Deferred via __getattr__ to break the circular import
# chain (utils → event_io → sources/__init__ → siren_rays → utils).
_EVENT_IO_REEXPORTS = {
    'save_single_event', 'load_single_event', 'get_random_root_entry_index',
    'read_photon_data_from_root', 'get_pdg_code', 'get_particle_mass',
    'extract_particle_properties', 'analyze_loaded_particle',
    'analyze_event_directory', 'PARTICLE_MASSES',
    'momentum_to_angles_and_energy', 'analyze_event_kinematics',
    'print_event_kinematics',
}

def __getattr__(name: str) -> Any:
    if name in _EVENT_IO_REEXPORTS:
        from lucid.sources import event_io
        return getattr(event_io, name)
    raise AttributeError(f"module 'lucid.utils' has no attribute {name!r}")

def load_range_params(particle: str, medium: str) -> dict:
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


def calculate_particle_range(energy_mev: float, range_params: dict) -> float:
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

    range_mm = a * energy_mev + b
    range_m = range_mm / 1000.0

    return range_m


def check_track_endpoint_in_detector(
    position: np.ndarray, direction: np.ndarray, energy_mev: float,
    range_params: dict, detector_bounds: dict, fraction: float = 0.9,
) -> bool:
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
    track_range = calculate_particle_range(energy_mev, range_params)
    endpoint = position + track_range * direction

    detector_r = detector_bounds['r'] * fraction
    detector_h = detector_bounds['H'] * fraction

    radial_distance = np.sqrt(endpoint[0]**2 + endpoint[1]**2)
    if radial_distance > detector_r:
        return False

    if abs(endpoint[2]) > detector_h / 2.0:
        return False

    return True


def generate_random_event_params(key: jax.Array, detector_bounds: dict, fraction: float = 0.7) -> ParticleParams:
    """
    Generate random event parameters based on detector geometry.
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
    
    from lucid.detector_params import ParticleParams
    key, _ = jax.random.split(key)
    phi = jax.random.uniform(key, shape=(), minval=0, maxval=2*jnp.pi)
    key, _ = jax.random.split(key)
    cos_theta = jax.random.uniform(key, shape=(), minval=-1, maxval=1)
    sin_theta = jnp.sqrt(1 - cos_theta**2)
    direction = jnp.array([sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), cos_theta])

    key, _ = jax.random.split(key)
    energy = jax.random.uniform(key, shape=(), minval=500.0, maxval=1500.0)

    return ParticleParams.from_cartesian(energy=energy, position=position,
                                         direction=direction, t0=jnp.array(0.0))


def smear_times(times: jax.Array, time_resolution: float = 2.5, *, key: jax.Array) -> jax.Array:
    """
    Gaussianly smear input times.

    Default σ = 2.5 ns matches the single-PE TTS of the SK 20-inch R3600
    PMT (Fukuda et al. 2003, NIM A 501, 418). Use 1.15 ns for HK 20-inch
    HQE R12860 (Nishimura et al. 2022, NIM A 1027, 166248).

    Note: when applied at the *per-photon* level (e.g. inside
    ``make_hits_data`` before segment_min), this represents the physical
    transit-time jitter and the per-channel first-arrival narrowing emerges
    automatically from the order statistic. When applied to already-aggregated
    per-channel times (legacy paths, e.g. event_io.py), σ should instead be the
    effective high-PE channel resolution (~0.4 ns for SK), since the
    order-statistic narrowing has already happened — pass ``time_resolution``
    explicitly in those callers.

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


def smear_charges_SK_like(counts: jax.Array, key: jax.Array | None = None) -> jax.Array:
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

    noise = jax.random.normal(key, shape=counts.shape) * sigma
    smeared_counts = counts + noise

    smeared_counts = jnp.where(jnp.isfinite(smeared_counts), smeared_counts, 0.0)

    # Avoid negative or unphysical charge values
    smeared_counts = jnp.clip(smeared_counts, 0.0, None)

    return smeared_counts


def time_digitizer(times: jax.Array, time_resolution: float = 0.4) -> jax.Array:
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


def jax_rotate_vector(vector: jax.Array, axis: jax.Array, angle: float) -> jax.Array:
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

def normalize(v: jax.Array, epsilon: float = 1e-8) -> jax.Array:
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


def generate_orthonormal_basis(v: jax.Array) -> jax.Array:
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