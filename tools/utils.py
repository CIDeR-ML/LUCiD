import h5py
import numpy as np
import jax.numpy as jnp
import jax

def full_to_sparse(charges, times):
    """Convert full arrays to sparse representation by removing zero elements.

    Parameters
    ----------
    charges : jnp.ndarray
        Array of charge values for all detectors
    times : jnp.ndarray
        Array of time values for all detectors

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


def save_single_event(event_data, params, event_number=0, filename=None):
    """Save single event simulation data to an HDF5 file in sparse format.

    Parameters
    ----------
    event_data : tuple
        (charges, average_times) arrays for the event
    params : tuple
        (track_energy, track_origin, track_direction)
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
        params_group = f.create_group('params')
        params_group.create_dataset('track_energy', data=np.array(params[0]))
        params_group.create_dataset('track_origin', data=np.array(params[1]))
        params_group.create_dataset('track_direction', data=np.array(params[2]))

        # Save event data and number
        event_group = f.create_group('event')
        event_group.create_dataset('event_number', data=np.array(event_number))
        event_group.create_dataset('indices', data=np.array(indices))
        event_group.create_dataset('charges', data=np.array(sparse_charges))
        event_group.create_dataset('times', data=np.array(sparse_times))

    return filename


def load_single_event(filename, num_detectors, sparse=True):
    """Load single event simulation data from an HDF5 file.

    Parameters
    ----------
    filename : str
        Path to HDF5 file
    num_detectors : int
        Total number of detectors (needed for dense format)
    sparse : bool, default=True
        If True, returns data in sparse format
        If False, converts to dense arrays

    Returns
    -------
    params : tuple
        (track_energy, track_origin, track_direction)
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
            Full array of charges for all detectors
        times : jnp.ndarray
            Full array of times for all detectors
    """
    with h5py.File(filename, 'r') as f:
        # Load parameters
        params_group = f['params']
        track_energy = jnp.array(params_group['track_energy'][()])
        track_origin = jnp.array(params_group['track_origin'][()])
        track_direction = jnp.array(params_group['track_direction'][()])

        params = (track_energy, track_origin, track_direction)

        # Load event data
        event_group = f['event']
        event_number = int(event_group['event_number'][()])
        indices = jnp.array(event_group['indices'][()])
        charges = jnp.array(event_group['charges'][()])
        times = jnp.array(event_group['times'][()])

    if sparse:
        return params, indices, charges, times
    else:
        # Convert sparse arrays to full dense arrays
        dense_charges = sparse_to_full(indices, charges, num_detectors)
        dense_times = sparse_to_full(indices, times, num_detectors)

        return params, dense_charges, dense_times


import jax
import jax.numpy as jnp

def generate_random_params(key, energy_range=(180, 1000), h=2, r=1, offset = 0.1):
    """
    Generate random parameters for event simulation with position inside a cylinder.

    Parameters
    ----------
    key : jax.random.PRNGKey
        JAX random number generator key
    energy_range : tuple, optional
        Range for energy values in MeV, default=(50, 1000)
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
    tuple
        Random parameters for event simulation
        - random_energy: float
            Random energy value in MeV
        - initial_position: array(3,)
            Starting position coordinates (x, y, z)
        - initial_direction: array(3,)
            Initial direction vector
    """
    # Split the key for independent random operations
    key1, key2, key3, key4, key5 = jax.random.split(key, 5)

    # Generate random energy within the specified range
    random_energy = jax.random.uniform(key1, minval=energy_range[0], maxval=energy_range[1])

    # Generate random position inside the cylinder
    random_position = generate_random_point_inside_cylinder(key2, h, r, offset)

    # Generate initial direction: random unit vector from normal distribution
    random_direction = jax.random.normal(key5, shape=(3,))
    random_direction /= jnp.linalg.norm(random_direction)  # Normalize to unit vector

    return random_energy, random_position, random_direction

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


def print_params(params):
    """
    Pretty print the event simulation parameters.

    Parameters
    ----------
    params : tuple
        Tuple containing (initial_energy, initial_position, initial_direction)
        - initial_energy: float
            Energy in MeV
        - initial_position: array(3,)
            Starting position coordinates (x, y, z)
        - initial_direction: array(3,)
            Initial direction vector

    Returns
    -------
    None
        Prints formatted parameter information to stdout

    Example
    -------
     params = (30.0, jnp.array([1., 0., 0.]), jnp.array([0., 1., 0.]))
     print_params(params)
    Event Parameters:
    ────────────────
    Energy: 30.00 MeV
    Initial Position: (1.00, 0.00, 0.00)
    Initial Direction: (0.00, 1.00, 0.00)
    ────────────────
    """
    # Unpack the parameter tuple
    initial_energy, initial_position, initial_direction = params

    # Create formatted output with consistent decimal places
    print("Event Parameters:")
    print("─" * 20)
    print(f"Energy: {initial_energy:.2f} MeV")
    print(f"Initial Position: ({initial_position[0]:.2f}, {initial_position[1]:.2f}, {initial_position[2]:.2f})")
    print(f"Initial Direction: ({initial_direction[0]:.2f}, {initial_direction[1]:.2f}, {initial_direction[2]:.2f})")
    print("─" * 20)