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
    """Read a single label-based event from an HDF5 group"""
    data = {
        # Label-level arrays
        'Q_per_label': np.array(group['Q_per_label']),
        'T_per_label': np.array(group['T_per_label']),
        'Label_Category': np.array(group['Label_Category']),
        'Label_CategoryName': np.array(group['Label_CategoryName']),
        'Label_Genealogy': np.array(group['Label_Genealogy'], dtype=object),

        # Track information
        'Track_PDG': np.array(group['Track_PDG']),
        'Track_Position': np.array(group['Track_Position']),
        'Track_Direction': np.array(group['Track_Direction']),
        'Track_Energy': np.array(group['Track_Energy']),
        'Track_Time': np.array(group['Track_Time']),
        'Track_ParentID': np.array(group['Track_ParentID']),

        # Aggregated values
        'Q_true': np.array(group['Q_true']),
        'T_true': np.array(group['T_true']),
        'Q_reco': np.array(group['Q_reco']),
        'T_reco': np.array(group['T_reco']),

        # Metadata (with defaults for optional fields)
        'n_labels': int(group['n_labels'][()]),
        'event_number': int(group['event_number'][()]) if 'event_number' in group else event_index,
        'primary_energy': float(group['primary_energy'][()]) if 'primary_energy' in group else 0.0,
        'apply_smearing': bool(group['apply_smearing'][()]) if 'apply_smearing' in group else False,
        'filename': filename,

        # Light containment metrics (with defaults for backward compatibility)
        'overall_light_containment': float(group['overall_light_containment'][()]) if 'overall_light_containment' in group else 1.0,
        'light_containment_by_label': np.array(group['light_containment_by_label']) if 'light_containment_by_label' in group else np.ones(int(group['n_labels'][()])),
    }

    # Add attributes if present
    if 'source' in group.attrs:
        data['source'] = group.attrs['source']
    if 'n_sensors' in group.attrs:
        data['n_sensors'] = group.attrs['n_sensors']

    return data


def print_label_event_info(data, title="Event", show_matrices=False,
                           show_genealogies=True, n_pmts_to_show=10):
    """
    Print comprehensive information about a label-based event.

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
    print(f"Primary Energy: {data['primary_energy']:.1f} MeV")
    print(f"Smearing Applied: {data['apply_smearing']}")

    # Dimensions
    n_labels = data['n_labels']
    n_sensors = data['Q_true'].shape[0]
    print(f"\nDimensions:")
    print(f"  Number of labels: {n_labels}")
    print(f"  Number of sensors: {n_sensors}")
    print(f"  Q_per_label shape: {data['Q_per_label'].shape}")
    print(f"  T_per_label shape: {data['T_per_label'].shape}")
    print(f"  Q_true shape: {data['Q_true'].shape}")

    # Overall charge statistics
    total_charge_true = np.sum(data['Q_true'])
    total_charge_reco = np.sum(data['Q_reco'])
    sensors_hit = np.sum(data['Q_true'] > 0)

    print(f"\nCharge Statistics:")
    print(f"  Total Q_true: {total_charge_true:.2f}")
    print(f"  Total Q_reco: {total_charge_reco:.2f}")
    print(f"  Sensors with signal: {sensors_hit} / {n_sensors} ({100*sensors_hit/n_sensors:.1f}%)")
    print(f"  Mean Q_true per hit sensor: {total_charge_true/max(sensors_hit, 1):.2f}")

    # Verify aggregation
    Q_sum = np.sum(data['Q_per_label'], axis=0)
    aggregation_ok = np.allclose(data['Q_true'], Q_sum)
    print(f"  Q_true == sum(Q_per_label): {aggregation_ok}")
    if not aggregation_ok:
        print(f"    WARNING: Aggregation mismatch!")
        print(f"    Max difference: {np.max(np.abs(data['Q_true'] - Q_sum)):.6f}")

    # Label summary
    print(f"\nLabel Summary:")
    print(f"  Total labels: {n_labels}")

    # Count labels by category
    categories = data['Label_Category']
    category_names = [name.decode('utf-8') if isinstance(name, bytes) else name
                     for name in data['Label_CategoryName']]

    category_counts = {}
    for cat_name in category_names:
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

    for cat_name, count in sorted(category_counts.items()):
        print(f"    {cat_name}: {count}")

    # Detailed track information
    print(f"\n{'='*70}")
    print(f"TRACK INFORMATION (ALL LABELS)")
    print(f"{'='*70}")

    header = f"{'Label':<6} {'Category':<16} {'PDG':<10} {'Particle':<10} {'Energy':<12} {'Time':<10} {'Parent':<8}"
    print(header)
    print("-" * 70)

    for i in range(n_labels):
        category_name = category_names[i]
        pdg = data['Track_PDG'][i]
        particle_name = get_particle_name(pdg)
        energy = data['Track_Energy'][i]
        time = data['Track_Time'][i]
        parent = data['Track_ParentID'][i]

        print(f"{i:<6} {category_name:<16} {pdg:<10} {particle_name:<10} {energy:<12.2f} {time:<10.3f} {parent:<8}")

    # Detailed label information
    print(f"\n{'='*70}")
    print(f"DETAILED LABEL INFORMATION")
    print(f"{'='*70}")

    for i in range(n_labels):
        print(f"\nLabel {i}: {category_names[i]}")
        print(f"  PDG: {data['Track_PDG'][i]} ({get_particle_name(data['Track_PDG'][i])})")
        print(f"  Energy: {data['Track_Energy'][i]:.3f} MeV")
        print(f"  Time: {data['Track_Time'][i]:.3f} ns")
        print(f"  Position: ({data['Track_Position'][i][0]:.3f}, {data['Track_Position'][i][1]:.3f}, {data['Track_Position'][i][2]:.3f}) cm")
        print(f"  Direction: ({data['Track_Direction'][i][0]:.4f}, {data['Track_Direction'][i][1]:.4f}, {data['Track_Direction'][i][2]:.4f})")
        print(f"  Parent Track ID: {data['Track_ParentID'][i]}")

        # Genealogy
        if show_genealogies:
            genealogy = data['Label_Genealogy'][i]
            if isinstance(genealogy, np.ndarray):
                genealogy_list = genealogy.tolist()
            else:
                genealogy_list = genealogy
            print(f"  Genealogy: {genealogy_list}")

        # Charge statistics for this label
        Q_label = data['Q_per_label'][i]
        T_label = data['T_per_label'][i]
        label_charge = np.sum(Q_label)
        label_sensors_hit = np.sum(Q_label > 0)

        print(f"  Charge contribution: {label_charge:.2f} ({100*label_charge/max(total_charge_true,1):.1f}% of total)")
        print(f"  Sensors hit: {label_sensors_hit}")

        if label_sensors_hit > 0:
            mean_T = np.sum(T_label * Q_label) / label_charge if label_charge > 0 else 0
            min_T = np.min(T_label[T_label > 0])
            print(f"  Mean hit time (charge-weighted): {mean_T:.2f} ns")
            print(f"  Min hit time: {min_T:.2f} ns")

    # Matrix display
    if show_matrices:
        print(f"\n{'='*70}")
        print(f"CHARGE MATRIX (First {n_pmts_to_show} sensors)")
        print(f"{'='*70}")

        print(f"\n{'Label':<8}", end="")
        for j in range(min(n_pmts_to_show, n_sensors)):
            print(f"PMT{j:<7}", end="")
        print()
        print("-" * 70)

        Q_per_label = data['Q_per_label']
        for i in range(n_labels):
            print(f"{i:<8}", end="")
            for j in range(min(n_pmts_to_show, n_sensors)):
                print(f"{Q_per_label[i,j]:<10.2f}", end="")
            print()

        # Also show Q_true for comparison
        print(f"\n{'Q_true':<8}", end="")
        for j in range(min(n_pmts_to_show, n_sensors)):
            print(f"{data['Q_true'][j]:<10.2f}", end="")
        print()

        # Show Q_reco
        print(f"{'Q_reco':<8}", end="")
        for j in range(min(n_pmts_to_show, n_sensors)):
            print(f"{data['Q_reco'][j]:<10.2f}", end="")
        print()


def print_genealogy_tree(data, title="Event"):
    """
    Print track genealogy tree with light containment information.

    Parameters
    ----------
    data : dict
        Event data dictionary
    title : str
        Title for the printout
    """
    n_labels = data['n_labels']

    # Define colors (for terminal, we'll just use labels)
    colors_palette = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow',
                      'brown', 'pink', 'olive', 'navy', 'teal', 'maroon']

    category_names_map = {0: "Primary", 1: "DecayElectron", 2: "SecondaryPion", 3: "GammaShower", -1: "Unknown"}

    # Build track tree from HDF5 genealogy data
    track_tree = {}
    Q_per_label = data['Q_per_label']
    T_per_label = data['T_per_label']

    # Process each label to build the track tree
    for label_idx in range(n_labels):
        genealogy = data['Label_Genealogy'][label_idx]
        if isinstance(genealogy, np.ndarray):
            genealogy_list = genealogy.tolist()
        else:
            genealogy_list = genealogy

        pdg = data['Track_PDG'][label_idx]
        particle_name = get_particle_name(int(pdg))
        category_code = data['Label_Category'][label_idx]
        category_name = category_names_map.get(int(category_code), f'Category_{int(category_code)}')
        energy = data['Track_Energy'][label_idx]
        parent_id = data['Track_ParentID'][label_idx]

        # Calculate charge and average time for this label
        Q_label = Q_per_label[label_idx]
        T_label = T_per_label[label_idx]
        total_charge = np.sum(Q_label)

        # Calculate weighted average time (only for sensors with finite times)
        finite_mask = np.isfinite(T_label) & (Q_label > 0)
        if np.any(finite_mask):
            finite_charges = Q_label[finite_mask]
            finite_times = T_label[finite_mask]
            avg_time = np.sum(finite_charges * finite_times) / np.sum(finite_charges)
        else:
            avg_time = 0.0

        # Add all tracks in the genealogy to the tree
        for depth, track_id in enumerate(genealogy_list):
            if track_id not in track_tree:
                track_tree[track_id] = {
                    'particle': particle_name if depth == len(genealogy_list) - 1 else f'Track_{track_id}',
                    'category': category_name if depth == len(genealogy_list) - 1 else 'Unknown',
                    'energy': energy if depth == len(genealogy_list) - 1 else 0.0,
                    'parent_id': parent_id if depth == len(genealogy_list) - 1 else (genealogy_list[depth - 1] if depth > 0 else 0),
                    'children': set(),
                    'charge': 0.0,
                    'time': 0.0,
                    'label_id': None,
                    'pdg': pdg if depth == len(genealogy_list) - 1 else 0,
                    'containment': 0.0
                }

            # If this is the photon-producing track (last in genealogy), store its data
            if depth == len(genealogy_list) - 1:
                containment = data['light_containment_by_label'][label_idx]
                track_tree[track_id].update({
                    'particle': particle_name,
                    'category': category_name,
                    'energy': energy,
                    'parent_id': parent_id,
                    'charge': total_charge,
                    'time': avg_time,
                    'label_id': label_idx,
                    'pdg': pdg,
                    'containment': containment
                })

            # Link parent-child relationship
            if depth > 0:
                parent_track_id = genealogy_list[depth - 1]
                if parent_track_id in track_tree:
                    track_tree[parent_track_id]['children'].add(track_id)

    def format_track_tree_terminal(track_id, track_tree, depth=0):
        """Recursively format track tree as text"""
        if track_id not in track_tree:
            return []

        track = track_tree[track_id]
        indent = "  " + "  " * depth
        arrow = "└─ " if depth > 0 else ""

        # Format label ID with containment info
        if track['label_id'] is not None:
            label_str = f" [Label {track['label_id']}]"
            containment_str = f" - Containment: {track['containment']*100:.1f}%"
        else:
            label_str = ""
            containment_str = ""

        lines = [
            f"{indent}{arrow}{track['particle']} ({track['category']}) - TrackID: {track_id}{label_str}",
            f"{indent}    Energy: {track['energy']:.1f} MeV, Charge: {track['charge']:.1f} PE, Avg Time: {track['time']:.1f} ns{containment_str}"
        ]

        # Add children recursively
        for child_id in sorted(track['children']):
            lines.extend(format_track_tree_terminal(child_id, track_tree, depth + 1))

        return lines

    # Find root tracks (parent_id == 0 or not in tree)
    root_tracks = [tid for tid, data_item in track_tree.items() if data_item['parent_id'] == 0]

    # Print header
    print(f"\n{'='*80}")
    print(f"{title} - TRACK GENEALOGY")
    print(f"{'='*80}")

    # Print tree
    for root_id in sorted(root_tracks):
        print()
        for line in format_track_tree_terminal(root_id, track_tree, depth=0):
            print(line)

    # Print overall light containment
    overall_containment = data['overall_light_containment']
    print(f"\n{'='*80}")
    print("LIGHT CONTAINMENT")
    print(f"{'='*80}")
    print(f"Overall: {overall_containment*100:.1f}% of photons inside detector")

    # Print per-label containment
    light_containment_by_label = data['light_containment_by_label']
    print("\nPer-label:")
    for label_idx in range(n_labels):
        particle_name = get_particle_name(int(data['Track_PDG'][label_idx]))
        containment = light_containment_by_label[label_idx]
        print(f"  Label {label_idx} ({particle_name}): {containment*100:.1f}%")
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
    total_labels = sum(e['n_labels'] for e in events)
    total_charge = sum(np.sum(e['Q_true']) for e in events)

    # Count categories across all events
    all_categories = {}
    for event in events:
        category_names = [name.decode('utf-8') if isinstance(name, bytes) else name
                         for name in event['Label_CategoryName']]
        for cat in category_names:
            all_categories[cat] = all_categories.get(cat, 0) + 1

    print(f"\nAggregate Statistics:")
    print(f"  Total labels across all events: {total_labels}")
    print(f"  Total charge across all events: {total_charge:.2f}")
    print(f"  Average labels per event: {total_labels/len(events):.1f}")
    print(f"  Average charge per event: {total_charge/len(events):.2f}")

    print(f"\nLabel Category Distribution:")
    for cat, count in sorted(all_categories.items()):
        print(f"  {cat}: {count} ({100*count/total_labels:.1f}%)")

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
