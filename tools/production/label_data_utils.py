"""
Utilities for reading and analyzing label-based LUCiD output files.

This module provides tools for comprehensive analysis of the new label-based
workflow where photons are classified by genealogy rather than by primary particle.
"""

import h5py
import numpy as np


def get_particle_name(pdg_code):
    """Convert PDG code to particle name"""
    pdg_names = {
        11: "e-",
        -11: "e+",
        13: "mu-",
        -13: "mu+",
        22: "gamma",
        111: "pi0",
        211: "pi+",
        -211: "pi-",
        321: "K+",
        -321: "K-",
        2212: "proton",
        -2212: "antiproton",
        2112: "neutron",
        -2112: "antineutron",
        0: "geantino",
    }
    return pdg_names.get(int(pdg_code), f"PDG_{int(pdg_code)}")


def get_category_name(category_code):
    """Convert category code to name"""
    category_names = {
        0: "Primary",
        1: "DecayElectron",
        2: "SecondaryPion",
        3: "GammaShower",
        -1: "Unknown"
    }
    return category_names.get(int(category_code), f"Category_{int(category_code)}")


def read_label_event_file(filename, event_index=None, verbose=True, show_matrices=False,
                          show_genealogies=True, n_pmts_to_show=10):
    """
    Read label-based event(s) from an HDF5 file.

    This function reads the new label-based format where Q and T have shape
    (N_labels, N_sensors) instead of (N_particles, N_sensors).

    Parameters
    ----------
    filename : str
        Path to the HDF5 file
    event_index : int or None, optional
        If specified, read only the event at this index.
        If None, read all events in the file.
    verbose : bool, optional
        Whether to print detailed information, by default True
    show_matrices : bool, optional
        Whether to show charge and time matrices, by default False
    show_genealogies : bool, optional
        Whether to show genealogy information, by default True
    n_pmts_to_show : int, optional
        Number of PMTs to show in matrix display, by default 10

    Returns
    -------
    dict or list of dict
        If event_index is specified, returns a single event dictionary.
        If event_index is None, returns a list of event dictionaries.
    """
    with h5py.File(filename, 'r') as f:
        # Get list of events (stored as groups: event_0, event_1, etc.)
        event_groups = sorted([k for k in f.keys() if k.startswith('event_')])
        n_events = len(event_groups)

        if n_events == 0:
            # Check for attributes
            if 'n_events' in f.attrs:
                n_events = f.attrs['n_events']
                print(f"File reports {n_events} events but no event groups found")
            raise ValueError("No events found in file")

        if event_index is not None:
            # Read single event
            if event_index >= n_events:
                raise IndexError(f"Event index {event_index} out of range [0, {n_events})")

            group_name = f'event_{event_index}'
            event_data = _read_single_label_event(f[group_name], event_index, filename)

            if verbose:
                print_label_event_info(event_data, f"Event {event_index}",
                                      show_matrices=show_matrices,
                                      show_genealogies=show_genealogies,
                                      n_pmts_to_show=n_pmts_to_show)

            return event_data
        else:
            # Read all events
            events = []
            for i, group_name in enumerate(event_groups):
                event_data = _read_single_label_event(f[group_name], i, filename)
                events.append(event_data)

                if verbose:
                    print_label_event_info(event_data, f"Event {i}/{n_events-1}",
                                          show_matrices=show_matrices,
                                          show_genealogies=show_genealogies,
                                          n_pmts_to_show=n_pmts_to_show)

            if verbose:
                print(f"\n{'='*70}")
                print(f"SUMMARY: Read {len(events)} events from {filename}")
                print(f"{'='*70}\n")

            return events


def _read_single_label_event(group, event_index, filename):
    """Read a single event from an HDF5 group (new field naming convention)"""
    n_particles = int(group['n_particles'][()])

    data = {
        # Event metadata
        'n_particles': n_particles,
        'event_number': int(group['event_number'][()]) if 'event_number' in group else event_index,
        't0': float(group['t0'][()]) if 't0' in group else 0.0,
        'filename': filename,

        # Reconstructed sensor data
        'PE': np.array(group['PE']),  # (N_sensors,) - observed photoelectrons
        'T': np.array(group['T']),    # (N_sensors,) - observed first-hit time

        # Sensor data by categorized particle
        'PE_per_particle': np.array(group['PE_per_particle']),  # (n_particles, N_sensors)
        'T_per_particle': np.array(group['T_per_particle']),    # (n_particles, N_sensors)

        # Categorized particles metadata
        'Particle_Category': np.array(group['Particle_Category']),
        'Particle_CategorizedGenealogy': np.array(group['Particle_CategorizedGenealogy'], dtype=object),

        # Light containment metrics
        'overall_light_containment': float(group['overall_light_containment'][()]) if 'overall_light_containment' in group else 1.0,
        'light_containment_by_particle': np.array(group['light_containment_by_particle']) if 'light_containment_by_particle' in group else np.ones(n_particles),
    }

    # Track genealogy (optional - only present when track segments are included)
    if 'Particle_TrackGenealogy' in group:
        data['Particle_TrackGenealogy'] = np.array(group['Particle_TrackGenealogy'], dtype=object)

    # Add attributes if present
    if 'source' in group.attrs:
        data['source'] = group.attrs['source']
    if 'n_sensors' in group.attrs:
        data['n_sensors'] = group.attrs['n_sensors']

    return data


def print_label_event_info(data, title="Event", show_matrices=False,
                           show_genealogies=True, n_pmts_to_show=10):
    """
    Print comprehensive information about a categorized particle event.

    Parameters
    ----------
    data : dict
        Event data dictionary
    title : str
        Title for the printout
    show_matrices : bool, optional
        Whether to show charge and time matrices, by default False
    show_genealogies : bool, optional
        Whether to show genealogy details, by default True
    n_pmts_to_show : int, optional
        Number of PMTs to show in matrix display, by default 10
    """
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

    # Basic event info
    print(f"Event Number: {data['event_number']}")
    print(f"Source: {data.get('source', 'Unknown')}")
    print(f"t0: {data.get('t0', 0.0):.2f} ns")

    # Dimensions
    n_particles = data['n_particles']
    n_sensors = data['PE'].shape[0]
    print(f"\nDimensions:")
    print(f"  Number of particles: {n_particles}")
    print(f"  Number of sensors: {n_sensors}")
    print(f"  PE_per_particle shape: {data['PE_per_particle'].shape}")
    print(f"  T_per_particle shape: {data['T_per_particle'].shape}")
    print(f"  PE shape: {data['PE'].shape}")

    # Overall charge statistics
    total_PE = np.sum(data['PE'])
    sensors_hit = np.sum(data['PE'] > 0)

    print(f"\nCharge Statistics:")
    print(f"  Total PE: {total_PE:.2f}")
    print(f"  Sensors with signal: {sensors_hit} / {n_sensors} ({100*sensors_hit/n_sensors:.1f}%)")
    print(f"  Mean PE per hit sensor: {total_PE/max(sensors_hit, 1):.2f}")

    # Verify aggregation
    PE_sum = np.sum(data['PE_per_particle'], axis=0)
    # PE is smeared, so compare to sum of PE_per_particle (true values)
    print(f"  Sum of PE_per_particle: {np.sum(PE_sum):.2f}")

    # Particle summary
    print(f"\nParticle Summary:")
    print(f"  Total particles: {n_particles}")

    # Count particles by category
    categories = data['Particle_Category']
    category_counts = {}
    for cat_code in categories:
        cat_name = get_category_name(cat_code)
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

    for cat_name, count in sorted(category_counts.items()):
        print(f"    {cat_name}: {count}")

    # Detailed particle information
    print(f"\n{'='*70}")
    print(f"CATEGORIZED PARTICLES")
    print(f"{'='*70}")

    header = f"{'Idx':<6} {'Category':<16} {'PE':<12} {'Containment':<12}"
    print(header)
    print("-" * 70)

    for i in range(n_particles):
        category_name = get_category_name(categories[i])
        PE_particle = np.sum(data['PE_per_particle'][i])
        containment = data['light_containment_by_particle'][i]

        print(f"{i:<6} {category_name:<16} {PE_particle:<12.2f} {containment*100:<11.1f}%")

    # Genealogy information
    if show_genealogies:
        print(f"\n{'='*70}")
        print(f"GENEALOGY INFORMATION")
        print(f"{'='*70}")

        for i in range(n_particles):
            category_name = get_category_name(categories[i])
            genealogy = data['Particle_CategorizedGenealogy'][i]
            if isinstance(genealogy, np.ndarray):
                genealogy_list = genealogy.tolist()
            else:
                genealogy_list = genealogy
            print(f"Particle {i} ({category_name}): Genealogy = {genealogy_list}")

            # Charge statistics
            PE_particle = data['PE_per_particle'][i]
            T_particle = data['T_per_particle'][i]
            particle_charge = np.sum(PE_particle)
            particle_sensors_hit = np.sum(PE_particle > 0)

            print(f"  Charge contribution: {particle_charge:.2f} ({100*particle_charge/max(total_PE,1):.1f}% of total)")
            print(f"  Sensors hit: {particle_sensors_hit}")

            if particle_sensors_hit > 0:
                mean_T = np.sum(T_particle * PE_particle) / particle_charge if particle_charge > 0 else 0
                valid_times = T_particle[T_particle > 0]
                if len(valid_times) > 0:
                    min_T = np.min(valid_times)
                    print(f"  Mean hit time (charge-weighted): {mean_T:.2f} ns")
                    print(f"  Min hit time: {min_T:.2f} ns")

    # Matrix display
    if show_matrices:
        print(f"\n{'='*70}")
        print(f"PE MATRIX (First {n_pmts_to_show} sensors)")
        print(f"{'='*70}")

        print(f"\n{'Particle':<10}", end="")
        for j in range(min(n_pmts_to_show, n_sensors)):
            print(f"PMT{j:<7}", end="")
        print()
        print("-" * 70)

        PE_per_particle = data['PE_per_particle']
        for i in range(n_particles):
            print(f"{i:<10}", end="")
            for j in range(min(n_pmts_to_show, n_sensors)):
                print(f"{PE_per_particle[i,j]:<10.2f}", end="")
            print()

        # Show PE (observed)
        print(f"\n{'PE':<10}", end="")
        for j in range(min(n_pmts_to_show, n_sensors)):
            print(f"{data['PE'][j]:<10.2f}", end="")
        print()


def print_genealogy_tree(data, title="Event"):
    """
    Print particle genealogy tree with light containment information.

    Parameters
    ----------
    data : dict
        Event data dictionary
    title : str
        Title for the printout
    """
    n_particles = data['n_particles']

    # Define colors (for terminal, we'll just use labels)
    colors_palette = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow',
                      'brown', 'pink', 'olive', 'navy', 'teal', 'maroon']

    category_names_map = {0: "Primary", 1: "DecayElectron", 2: "SecondaryPion", 3: "GammaShower", -1: "Unknown"}

    # Build tree from genealogy data
    particle_tree = {}
    PE_per_particle = data['PE_per_particle']
    T_per_particle = data['T_per_particle']

    # Process each particle
    for particle_idx in range(n_particles):
        genealogy = data['Particle_CategorizedGenealogy'][particle_idx]
        if isinstance(genealogy, np.ndarray):
            genealogy_list = genealogy.tolist()
        else:
            genealogy_list = genealogy

        category_code = data['Particle_Category'][particle_idx]
        category_name = category_names_map.get(int(category_code), f'Category_{int(category_code)}')

        # Calculate charge and average time
        PE_particle = PE_per_particle[particle_idx]
        T_particle = T_per_particle[particle_idx]
        total_charge = np.sum(PE_particle)

        # Calculate weighted average time
        finite_mask = np.isfinite(T_particle) & (PE_particle > 0)
        if np.any(finite_mask):
            finite_charges = PE_particle[finite_mask]
            finite_times = T_particle[finite_mask]
            avg_time = np.sum(finite_charges * finite_times) / np.sum(finite_charges)
        else:
            avg_time = 0.0

        containment = data['light_containment_by_particle'][particle_idx]

        # Store particle info
        particle_tree[particle_idx] = {
            'category': category_name,
            'charge': total_charge,
            'time': avg_time,
            'containment': containment,
            'genealogy': genealogy_list
        }

    # Print header
    print(f"\n{'='*80}")
    print(f"{title} - PARTICLE GENEALOGY")
    print(f"{'='*80}")

    # Print particles
    for particle_idx in range(n_particles):
        info = particle_tree[particle_idx]
        print(f"\nParticle {particle_idx} ({info['category']})")
        print(f"  Genealogy: {info['genealogy']}")
        print(f"  Charge: {info['charge']:.1f} PE, Avg Time: {info['time']:.1f} ns")
        print(f"  Containment: {info['containment']*100:.1f}%")

    # Print overall light containment
    overall_containment = data['overall_light_containment']
    print(f"\n{'='*80}")
    print("LIGHT CONTAINMENT")
    print(f"{'='*80}")
    print(f"Overall: {overall_containment*100:.1f}% of photons inside detector")

    # Print per-particle containment summary
    light_containment_by_particle = data['light_containment_by_particle']
    print("\nPer-particle:")
    for particle_idx in range(n_particles):
        category_name = get_category_name(data['Particle_Category'][particle_idx])
        containment = light_containment_by_particle[particle_idx]
        print(f"  Particle {particle_idx} ({category_name}): {containment*100:.1f}%")
    print()


def print_file_summary(filename, max_events_to_show=None):
    """
    Print a summary of all events in a file.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file
    max_events_to_show : int or None, optional
        Maximum number of events to show in detail.
        If None, shows all events.
    """
    events = read_label_event_file(filename, event_index=None, verbose=False)

    print(f"\n{'='*70}")
    print(f"FILE SUMMARY: {filename}")
    print(f"{'='*70}")
    print(f"Total events: {len(events)}")

    # Aggregate statistics
    total_particles = sum(e['n_particles'] for e in events)
    total_charge = sum(np.sum(e['PE']) for e in events)

    # Count categories across all events
    all_categories = {}
    for event in events:
        for cat_code in event['Particle_Category']:
            cat_name = get_category_name(cat_code)
            all_categories[cat_name] = all_categories.get(cat_name, 0) + 1

    print(f"\nAggregate Statistics:")
    print(f"  Total particles across all events: {total_particles}")
    print(f"  Total PE across all events: {total_charge:.2f}")
    print(f"  Average particles per event: {total_particles/len(events):.1f}")
    print(f"  Average PE per event: {total_charge/len(events):.2f}")

    print(f"\nParticle Category Distribution:")
    for cat, count in sorted(all_categories.items()):
        print(f"  {cat}: {count} ({100*count/total_particles:.1f}%)")

    # Show individual events
    n_to_show = len(events) if max_events_to_show is None else min(max_events_to_show, len(events))

    print(f"\n{'='*70}")
    print(f"INDIVIDUAL EVENTS (showing {n_to_show}/{len(events)})")
    print(f"{'='*70}")

    for i in range(n_to_show):
        print_label_event_info(events[i], f"Event {i}", show_matrices=False,
                              show_genealogies=True, n_pmts_to_show=5)


if __name__ == "__main__":
    import sys

    # Example usage
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        event_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None

        if event_idx is not None:
            # Show single event
            read_label_event_file(filename, event_index=event_idx,
                                 verbose=True, show_matrices=True,
                                 show_genealogies=True, n_pmts_to_show=10)
        else:
            # Show all events
            print_file_summary(filename, max_events_to_show=3)
    else:
        print("Usage:")
        print("  python label_data_utils.py <filename> [event_index]")
        print("\nExamples:")
        print("  python label_data_utils.py events.h5           # Show all events")
        print("  python label_data_utils.py events.h5 0         # Show event 0 in detail")
