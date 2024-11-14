import h5py
import numpy as np
import jax.numpy as jnp

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
        (cone_opening, track_origin, track_direction, initial_intensity)
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
        params_group.create_dataset('cone_opening', data=np.array(params[0]))
        params_group.create_dataset('track_origin', data=np.array(params[1]))
        params_group.create_dataset('track_direction', data=np.array(params[2]))
        params_group.create_dataset('initial_intensity', data=np.array(params[3]))

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
        (cone_opening, track_origin, track_direction, initial_intensity)
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
        cone_opening = float(params_group['cone_opening'][()])
        track_origin = jnp.array(params_group['track_origin'][()])
        track_direction = jnp.array(params_group['track_direction'][()])
        initial_intensity = float(params_group['initial_intensity'][()])

        params = (cone_opening, track_origin, track_direction, initial_intensity)

        # Load event data
        event_group = f['event']
        event_number = int(event_group['event_number'][()])
        indices = jnp.array(event_group['indices'][()])
        charges = jnp.array(event_group['charges'][()])
        times = jnp.array(event_group['times'][()])

    if sparse:
        return params, event_number, indices, charges, times
    else:
        # Convert sparse arrays to full dense arrays
        dense_charges = sparse_to_full(indices, charges, num_detectors)
        dense_times = sparse_to_full(indices, times, num_detectors)

        return params, event_number, dense_charges, dense_times