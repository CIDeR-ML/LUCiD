import h5py
import numpy as np
import jax.numpy as jnp


def read_multi_event_file(filename, event_index=None, verbose=True, show_matrices=False):
    """
    Read event(s) from an HDF5 file that may contain multiple events.
    
    This function handles the new file format where multiple events are stacked
    with an index. It can read a specific event by index or all events.
    
    Parameters
    ----------
    filename : str
        Path to the HDF5 file
    event_index : int or None, optional
        If specified, read only the event at this index.
        If None, read all events in the file.
        Index can be used in two ways:
        - If data has an 'event_index' dimension, use it to slice
        - If data is organized in groups, use it to select a group
    verbose : bool, optional
        Whether to print detailed information, by default True
    show_matrices : bool, optional
        Whether to show charge and time matrices in verbose output, by default False
        
    Returns
    -------
    dict or list of dict
        If event_index is specified, returns a single event dictionary.
        If event_index is None, returns a list of event dictionaries.
        Each dictionary contains:
        - 'PDG': particle PDG codes
        - 'Q': charge matrix
        - 'Q_tot': total charge per track
        - 'T': time matrix
        - 'P': momentum vectors
        - 'V': vertex positions
        - 'event_number': event identifier
        - 'filename': source filename
        
    Examples
    --------
    # Read all events
    >>> events = read_multi_event_file('events.h5')
    >>> print(f"Read {len(events)} events")
    
    # Read specific event with detailed matrix display
    >>> event_0 = read_multi_event_file('events.h5', event_index=0, show_matrices=True)
    >>> print(f"Event 0 has {len(event_0['PDG'])} tracks")
    """
    with h5py.File(filename, 'r') as f:
        # Determine file structure
        keys = list(f.keys())
        
        # Check if file has top-level datasets (new format with event dimension)
        if 'PDG' in keys:
            # New format: datasets have an event dimension
            pdg_full = np.array(f['PDG'])
            
            # Check if there's an event dimension (3D for Q and T, 2D for PDG)
            if len(pdg_full.shape) == 2:
                # Format: datasets have shape (num_events, ...) 
                n_events = pdg_full.shape[0]
                
                if event_index is not None:
                    # Read single event
                    if event_index >= n_events:
                        raise IndexError(f"Event index {event_index} out of range [0, {n_events})")
                    
                    data = {
                        'PDG': np.array(f['PDG'][event_index]),
                        'Q': np.array(f['Q'][event_index]),
                        'Q_tot': np.array(f['Q_tot'][event_index]) if 'Q_tot' in f else np.sum(f['Q'][event_index], axis=1),
                        'T': np.array(f['T'][event_index]),
                        'P': np.array(f['P'][event_index]),
                        'V': np.array(f['V'][event_index]),
                        'event_number': event_index,
                        'filename': filename
                    }
                    
                    if verbose:
                        print_event_info(data, f"Event {event_index}", show_matrices=show_matrices)
                    
                    return data
                else:
                    # Read all events
                    events = []
                    for i in range(n_events):
                        data = {
                            'PDG': np.array(f['PDG'][i]),
                            'Q': np.array(f['Q'][i]),
                            'Q_tot': np.array(f['Q_tot'][i]) if 'Q_tot' in f else np.sum(f['Q'][i], axis=1),
                            'T': np.array(f['T'][i]),
                            'P': np.array(f['P'][i]),
                            'V': np.array(f['V'][i]),
                            'event_number': i,
                            'filename': filename
                        }
                        events.append(data)
                        
                        if verbose:
                            print_event_info(data, f"Event {i}/{n_events-1}", show_matrices=show_matrices)
                    
                    if verbose:
                        print(f"\nTotal: Read {len(events)} events from {filename}")
                    
                    return events
            else:
                # Format: single event file (backward compatibility)
                data = {
                    'PDG': pdg_full,
                    'Q': np.array(f['Q']),
                    'Q_tot': np.array(f['Q_tot']) if 'Q_tot' in f else np.sum(f['Q'], axis=1),
                    'T': np.array(f['T']),
                    'P': np.array(f['P']),
                    'V': np.array(f['V']),
                    'event_number': np.array(f['event_number']) if 'event_number' in f else 0,
                    'filename': filename
                }
                
                if verbose:
                    print_event_info(data, "Single Event", show_matrices=show_matrices)
                
                return data
        
        # Check if file has event groups (alternative multi-event format)
        elif 'event_0' in keys or any(k.startswith('event_') for k in keys):
            # Group-based format: each event is in a separate group
            event_groups = sorted([k for k in keys if k.startswith('event_')])
            n_events = len(event_groups)
            
            if event_index is not None:
                # Read single event
                if event_index >= n_events:
                    raise IndexError(f"Event index {event_index} out of range [0, {n_events})")
                
                group_name = f'event_{event_index}'
                if group_name not in f:
                    raise KeyError(f"Event group {group_name} not found in file")
                
                g = f[group_name]
                data = {
                    'PDG': np.array(g['PDG']),
                    'Q': np.array(g['Q']),
                    'Q_tot': np.array(g['Q_tot']) if 'Q_tot' in g else np.sum(g['Q'], axis=1),
                    'T': np.array(g['T']),
                    'P': np.array(g['P']),
                    'V': np.array(g['V']),
                    'event_number': np.array(g['event_number']) if 'event_number' in g else event_index,
                    'filename': filename
                }
                
                if verbose:
                    print_event_info(data, f"Event {event_index}", show_matrices=show_matrices)
                
                return data
            else:
                # Read all events
                events = []
                for i, group_name in enumerate(event_groups):
                    g = f[group_name]
                    data = {
                        'PDG': np.array(g['PDG']),
                        'Q': np.array(g['Q']),
                        'Q_tot': np.array(g['Q_tot']) if 'Q_tot' in g else np.sum(g['Q'], axis=1),
                        'T': np.array(g['T']),
                        'P': np.array(g['P']),
                        'V': np.array(g['V']),
                        'event_number': np.array(g['event_number']) if 'event_number' in g else i,
                        'filename': filename
                    }
                    events.append(data)
                    
                    if verbose:
                        print_event_info(data, f"Event {i}/{n_events-1}", show_matrices=show_matrices)
                
                if verbose:
                    print(f"\nTotal: Read {len(events)} events from {filename}")
                
                return events
        else:
            raise ValueError(f"Unrecognized file format. Top-level keys: {keys}")


def print_event_info(data, title="Event", show_matrices=False, n_pmts_to_show=10):
    """
    Print summary information about an event.
    
    Parameters
    ----------
    data : dict
        Event data dictionary
    title : str
        Title for the printout
    show_matrices : bool, optional
        Whether to show charge and time matrices, by default False
    n_pmts_to_show : int, optional
        Number of PMTs to show in matrix display, by default 10
    """
    pdg = data['PDG']
    q = data['Q']
    q_tot = data['Q_tot']
    t = data['T']
    p = data['P']
    v = data['V']
    
    n_tracks = pdg.shape[0]
    n_detectors = q.shape[1]
    
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Event Number: {data['event_number']}")
    print(f"Number of tracks: {n_tracks}")
    print(f"Number of sensors: {n_detectors}")
    
    # Detector statistics
    print(f"\nDetector Statistics:")
    print(f"Total charge detected: {np.sum(q_tot):.2f}")
    print(f"Mean charge per track: {np.mean(q_tot):.2f}")
    print(f"Mean charge per PMT: {np.mean(np.sum(q, axis=0)):.2f}")
    print(f"Number of PMTs with signal: {np.sum(np.sum(q, axis=0) > 0)}")
    
    # Particle information
    print(f"\nParticle Information:")
    print("-" * 80)
    print(f"{'Track #':<8}{'PDG':<8}{'Q_tot':<12}{'P_mag (MeV/c)':<16}{'Direction':<25}{'Vertex':<25}")
    print("-" * 80)
    
    for i in range(n_tracks):
        # Convert PDG code to particle name
        particle = get_particle_name(pdg[i])
        
        # Calculate momentum magnitude
        p_mag = np.sqrt(np.sum(p[i]**2))
        
        # Normalize direction
        direction = p[i] / (p_mag if p_mag > 0 else 1)
        
        print(f"{i:<8}{particle:<8}{q_tot[i]:<12.2f}{p_mag:<16.2f}{str(direction):<25}{str(v[i]):<25}")
    
    if show_matrices:
        # Print Q values for each track
        print("\nCharge Matrix (Q) - First {} PMTs:".format(min(n_pmts_to_show, n_detectors)))
        print("-" * 80)
        header = "Track #  "
        for j in range(min(n_pmts_to_show, n_detectors)):
            header += f"PMT-{j:<5} "
        print(header)
        print("-" * 80)
        
        for i in range(n_tracks):
            row = f"{i:<8}  "
            for j in range(min(n_pmts_to_show, n_detectors)):
                row += f"{q[i,j]:<7.2f} "
            row += f"... (showing {min(n_pmts_to_show, n_detectors)}/{n_detectors} PMTs)"
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
        print("\nTime Matrix (T) - First {} PMTs:".format(min(n_pmts_to_show, n_detectors)))
        print("-" * 80)
        header = "Track #  "
        for j in range(min(n_pmts_to_show, n_detectors)):
            header += f"PMT-{j:<5} "
        print(header)
        print("-" * 80)
        
        for i in range(n_tracks):
            row = f"{i:<8}  "
            for j in range(min(n_pmts_to_show, n_detectors)):
                if t[i,j] > 0:
                    row += f"{t[i,j]:<7.2f} "
                else:
                    row += f"{'--':<7} "
            row += f"... (showing {min(n_pmts_to_show, n_detectors)}/{n_detectors} PMTs)"
            print(row)


def get_particle_name(pdg_code):
    """
    Convert PDG code to particle name.
    
    Parameters
    ----------
    pdg_code : int
        PDG particle code
        
    Returns
    -------
    str
        Particle name
    """
    particle_map = {
        13: 'mu-',
        -13: 'mu+',
        211: 'pi+',
        -211: 'pi-',
        111: 'pi0',
        11: 'e-',
        -11: 'e+',
        22: 'gamma',
        2212: 'p',
        2112: 'n'
    }

    return particle_map.get(pdg_code, f'unknown')


def get_track_hits(event, track_index):
    """
    Extract nonzero PMT indices, Q, and T for a specific track,
    keeping only the PMTs with Q > 0.

    Returns filtered arrays so that:
        len(nonzero_indices) == len(Q_filtered) == len(T_filtered)
    """
    Q = event['Q']
    T = event['T']

    if track_index < 0 or track_index >= Q.shape[0]:
        raise IndexError(
            f"Track index {track_index} out of range [0, {Q.shape[0]})"
        )

    Q_row = Q[track_index]
    T_row = T[track_index]

    # PMTs where charge is detected
    nonzero_indices = np.where(Q_row > 0)[0]

    # Filter Q and T to only these PMTs
    Q_filtered = Q_row[nonzero_indices]
    T_filtered = T_row[nonzero_indices]

    return nonzero_indices, Q_filtered, T_filtered
