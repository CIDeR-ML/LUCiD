#!/usr/bin/env python3
"""
Validation script for LUCiD label-based workflow.

This script visualizes:
1. Photons from PhotonSim (input) colored by label/category
2. True track information (positions, directions) with cylinders/arrows
3. LUCiD predicted sensor hits (charge and time values on detector sensors)

This allows comprehensive validation that the label-based workflow produces
sensible results from PhotonSim input through LUCiD processing to detector hits.
"""
import sys
sys.path.append('/Users/cjesus/Software/LUCiD')

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tools.generate import read_label_data_from_photonsim
from tools.production.label_data_utils import read_label_event_file
from tools.geometry import generate_detector
from tools.geometry.utils import calculate_surface_normals, create_disc_mesh
import argparse

def create_cylinder(start, end, radius, n_segments=16):
    """
    Create a 3D cylinder mesh between two points.

    Args:
        start: Starting point [x, y, z]
        end: Ending point [x, y, z]
        radius: Cylinder radius in world coordinates (cm)
        n_segments: Number of segments around the cylinder

    Returns:
        x, y, z: Arrays of vertex coordinates
        i, j, k: Arrays defining triangular faces
    """
    start = np.array(start)
    end = np.array(end)

    # Direction vector and length
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-10:
        # Degenerate case: start and end are the same
        return np.array([]), np.array([]), np.array([]), [], [], []
    direction = direction / length

    # Find perpendicular vectors to create cylinder cross-section
    # Choose a vector not parallel to direction
    if abs(direction[0]) < 0.9:
        perp1 = np.cross(direction, [1, 0, 0])
    else:
        perp1 = np.cross(direction, [0, 1, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)

    # Create vertices
    vertices = []
    angles = np.linspace(0, 2*np.pi, n_segments, endpoint=False)

    # Bottom circle
    for angle in angles:
        point = start + radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
        vertices.append(point)

    # Top circle
    for angle in angles:
        point = end + radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
        vertices.append(point)

    vertices = np.array(vertices)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    # Create triangular faces for the cylinder surface
    i, j, k = [], [], []
    for seg in range(n_segments):
        next_seg = (seg + 1) % n_segments

        # Two triangles per rectangular face
        # Triangle 1
        i.append(seg)
        j.append(next_seg)
        k.append(seg + n_segments)

        # Triangle 2
        i.append(next_seg)
        j.append(next_seg + n_segments)
        k.append(seg + n_segments)

    return x, y, z, i, j, k


# Parse command line arguments
parser = argparse.ArgumentParser(description='Validate LUCiD label-based workflow')
parser.add_argument('root_file', type=str, help='Input ROOT file from PhotonSim')
parser.add_argument('hdf5_file', type=str, help='Input HDF5 file from LUCiD (label-based output)')
parser.add_argument('detector_config', type=str, help='Detector configuration JSON file')
parser.add_argument('--events', type=int, default=5, help='Number of events to validate (default: 5)')
parser.add_argument('--photons', type=int, default=500, help='Number of photons to sample per label (default: 500)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for photon sampling (default: 42)')
parser.add_argument('--plot-time', action='store_true', help='Color sensor hits by time instead of charge')
parser.add_argument('--log-scale', action='store_true', help='Use log scale for sensor hit colors')
args = parser.parse_args()

root_file = args.root_file
hdf5_file = args.hdf5_file
detector_config = args.detector_config
events_to_validate = list(range(args.events))
n_photons_to_sample = args.photons
master_seed = args.seed

print("="*70)
print(f"LUCID LABEL-BASED WORKFLOW VALIDATION")
print("="*70)
print(f"PhotonSim ROOT file: {root_file}")
print(f"LUCiD HDF5 file: {hdf5_file}")
print(f"Detector config: {detector_config}")
print(f"Events to validate: {args.events}")
print(f"Photons to sample per label: {n_photons_to_sample}")
print(f"Plot mode: {'Time' if args.plot_time else 'Charge'}")
print(f"Scale: {'Logarithmic' if args.log_scale else 'Linear'}")
print()

# Load detector geometry
print("Loading detector geometry...")
detector = generate_detector(detector_config)
print(f"Detector: {detector.__class__.__name__}")
print(f"  Number of sensors: {detector.n_sensors}")
print(f"  Sensor radius: {detector.S_radius} m")

# IMPORTANT: Convert detector coordinates from meters to centimeters
# PhotonSim uses cm, LUCiD detector geometry uses m
detector.all_points = detector.all_points * 100.0  # m -> cm
detector.S_radius = detector.S_radius * 100.0      # m -> cm
print(f"  Converted to cm for visualization (sensor radius: {detector.S_radius:.2f} cm)")
print()

# Define colors for labels
colors_palette = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow',
                  'brown', 'pink', 'olive', 'navy', 'teal', 'maroon']
category_names = {0: 'Primary', 1: 'DecayElectron', 2: 'SecondaryPion', 3: 'GammaShower'}

# PDG code to particle name mapping
pdg_to_name = {
    -211: 'pi-',
    211: 'pi+',
    111: 'pi0',
    -13: 'mu+',
    13: 'mu-',
    -11: 'e+',
    11: 'e-',
    22: 'gamma',
    2212: 'proton',
    2112: 'neutron',
    -2212: 'antiproton',
}

# Statistics tracking
total_events = 0
category_counts = {0: 0, 1: 0, 2: 0, 3: 0}
events_with_category = {0: 0, 1: 0, 2: 0, 3: 0}
primary_particles = set()


def process_event(event_idx):
    """Process a single event and return data for plotting"""
    global total_events, category_counts, events_with_category, primary_particles

    total_events += 1

    print(f"\n{'='*70}")
    print(f"EVENT {event_idx}")
    print(f"{'='*70}")

    # Read PhotonSim label data
    print("Reading PhotonSim data...")
    photonsim_data = read_label_data_from_photonsim(root_file, event_idx)

    n_labels = photonsim_data['n_labels']
    labels = photonsim_data['labels']
    all_photon_origins = photonsim_data['photon_origins']
    all_photon_directions = photonsim_data['photon_directions']

    print(f"  PhotonSim: {n_labels} labels, {len(all_photon_origins)} photons")

    # Read LUCiD HDF5 data
    print("Reading LUCiD HDF5 data...")
    lucid_data = read_label_event_file(hdf5_file, event_index=event_idx, verbose=False)

    Q_per_label = lucid_data['Q_per_label']  # Shape: (N_labels, N_sensors)
    T_per_label = lucid_data['T_per_label']
    Q_true = lucid_data['Q_true']  # Shape: (N_sensors,)
    T_true = lucid_data['T_true']

    print(f"  LUCiD: Q_per_label shape {Q_per_label.shape}, Q_true shape {Q_true.shape}")

    # Verify dimensions match
    if Q_per_label.shape[0] != n_labels:
        print(f"  WARNING: Label count mismatch - PhotonSim: {n_labels}, LUCiD: {Q_per_label.shape[0]}")
    if Q_per_label.shape[1] != detector.n_sensors:
        print(f"  WARNING: Sensor count mismatch - Detector: {detector.n_sensors}, LUCiD: {Q_per_label.shape[1]}")

    print()

    # Sample photons for each label and prepare visualization data
    sampled_photons = []
    track_positions = []
    track_directions = []
    label_colors = []
    label_names = []
    label_categories = []
    label_q_values = []  # Total charge per label

    local_category_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for label_idx, label in enumerate(labels):
        photon_indices = label['photon_indices']
        track_info = label['track_info']

        if track_info is None:
            continue

        cat_name = category_names.get(track_info['category'], f"Unknown_{track_info['category']}")
        particle_name = pdg_to_name.get(track_info['pdg'], f"PDG{track_info['pdg']}")
        color = colors_palette[label_idx % len(colors_palette)]

        # Get kinetic energy
        kinetic_energy = track_info['energy']  # MeV

        # Get total charge for this label
        q_label_total = np.sum(Q_per_label[label_idx]) if label_idx < Q_per_label.shape[0] else 0.0

        # Update statistics
        category = track_info['category']
        if category in category_counts:
            category_counts[category] += 1
            local_category_counts[category] += 1

        # Track primary particle types
        if category == 0:  # Primary
            primary_particles.add(particle_name)

        print(f"Label {label_idx} ({cat_name}):")
        print(f"  Particle: {particle_name} (PDG: {track_info['pdg']})")
        print(f"  Kinetic Energy: {kinetic_energy:.2f} MeV")
        print(f"  Color: {color}")
        print(f"  Track position: {track_info['position']}")
        print(f"  Track direction: {track_info['direction']}")
        print(f"  Total photons: {len(photon_indices)}")
        print(f"  Total charge (LUCiD): {q_label_total:.1f} PE")

        if len(photon_indices) == 0:
            print(f"  WARNING: No photons for this label")
            print()
            continue

        # Sample photons
        photon_indices_array = np.array(photon_indices, dtype=np.int32)
        n_to_sample = min(n_photons_to_sample, len(photon_indices))
        np.random.seed(master_seed + label_idx + event_idx * 100)
        sampled_indices = np.random.choice(len(photon_indices), size=n_to_sample, replace=False)
        selected_photon_indices = photon_indices_array[sampled_indices]

        photon_origins = all_photon_origins[selected_photon_indices]

        sampled_photons.append(photon_origins)
        track_positions.append(track_info['position'])
        track_directions.append(track_info['direction'])
        label_colors.append(color)
        label_q_values.append(q_label_total)

        # Create label name with energy and charge
        label_name = f"Label {label_idx} ({cat_name} - {particle_name}, {kinetic_energy:.1f} MeV, {q_label_total:.0f} PE)"
        label_names.append(label_name)
        label_categories.append(track_info['category'])

        print(f"  Sampled {n_to_sample} photons")
        print()

    # Update events with category counts
    for cat, count in local_category_counts.items():
        if count > 0:
            events_with_category[cat] += 1

    # Prepare sensor hit data
    hit_indices = np.where(Q_true > 0)[0]
    hit_charges = Q_true[hit_indices]
    hit_times = T_true[hit_indices]

    print(f"Sensor hits: {len(hit_indices)} / {detector.n_sensors} sensors with charge")
    print(f"  Total charge: {np.sum(hit_charges):.1f} PE")
    print(f"  Mean time: {np.mean(hit_times):.1f} ns")
    print()

    return {
        'event_idx': event_idx,
        'n_labels': len(sampled_photons),
        'sampled_photons': sampled_photons,
        'track_positions': track_positions,
        'track_directions': track_directions,
        'label_colors': label_colors,
        'label_names': label_names,
        'label_categories': label_categories,
        'label_q_values': label_q_values,
        'hit_indices': hit_indices,
        'hit_charges': hit_charges,
        'hit_times': hit_times,
        'Q_per_label': Q_per_label,
        'T_per_label': T_per_label,
    }


# Process all events
events_data = []
for event_idx in events_to_validate:
    try:
        event_data = process_event(event_idx)
        if event_data is not None:
            events_data.append(event_data)
    except Exception as e:
        print(f"ERROR processing event {event_idx}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Print summary statistics
print()
print("="*70)
print("VALIDATION STATISTICS")
print("="*70)
print(f"Total events processed: {total_events}")
print()
print("Primary particle types detected:", ', '.join(sorted(primary_particles)))
print()
print("Category breakdown:")
for cat_id in sorted(category_counts.keys()):
    cat_name = category_names.get(cat_id, f"Unknown_{cat_id}")
    count = category_counts[cat_id]
    events_with = events_with_category[cat_id]
    if total_events > 0:
        avg_per_event = count / total_events
        pct_events = 100 * events_with / total_events
        print(f"  {cat_name:20s}: {count:4d} total | {events_with:3d} events ({pct_events:.1f}%) | {avg_per_event:.2f} per event")
print()

# Create interactive plots
print()
print("="*70)
print("CREATING INTERACTIVE VISUALIZATIONS")
print("="*70)
print(f"Creating plots for {len(events_data)} events")
print()

for event_data in events_data:
    event_idx = event_data['event_idx']
    print(f"Creating visualization for event {event_idx}...")

    # Create figure with subplots: 3D plot on top, histograms below
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.78, 0.22],
        column_widths=[0.5, 0.5],
        specs=[[{"type": "scatter3d", "colspan": 2}, None],
               [{"type": "bar"}, {"type": "bar"}]],
        subplot_titles=("3D Event Visualization", "True Charge per Label", "True Time per Label"),
        vertical_spacing=0.10,
        horizontal_spacing=0.1
    )

    # Track number of traces for each component and their actual indices
    n_photon_traces = 0
    n_arrow_traces = 0
    photon_trace_indices = []
    arrow_trace_indices = []
    current_trace_index = 0

    # Plot each label (photons + track arrows) in the 3D subplot
    for i in range(event_data['n_labels']):
        photons = event_data['sampled_photons'][i]
        track_pos = event_data['track_positions'][i]
        track_dir = event_data['track_directions'][i]
        color = event_data['label_colors'][i]
        label_name = event_data['label_names'][i]
        category = event_data['label_categories'][i]

        # Plot photons in 3D subplot (row=1, col=1)
        fig.add_trace(
            go.Scatter3d(
                x=photons[:, 0], y=photons[:, 1], z=photons[:, 2],
                mode='markers',
                marker=dict(size=2, color=color, opacity=0.3),
                name=label_name,
                legendgroup=f'label{i}',
                showlegend=True,
                visible=True
            ),
            row=1, col=1
        )
        photon_trace_indices.append(current_trace_index)
        current_trace_index += 1
        n_photon_traces += 1

        # Plot track arrow
        scale = 20  # Arrow length in cm
        cylinder_radius = 1.5  # Cylinder radius in cm

        # Draw the cylinder - HIDDEN BY DEFAULT
        cylinder_end = track_pos + track_dir * scale
        cyl_x, cyl_y, cyl_z, cyl_i, cyl_j, cyl_k = create_cylinder(
            track_pos, cylinder_end, cylinder_radius
        )

        if len(cyl_x) > 0:
            fig.add_trace(
                go.Mesh3d(
                    x=cyl_x, y=cyl_y, z=cyl_z,
                    i=cyl_i, j=cyl_j, k=cyl_k,
                    color=color, opacity=1.0,
                    name=f'{label_name} track',
                    legendgroup=f'label{i}',
                    showlegend=False,
                    visible=False,  # Hidden by default
                    lighting=dict(ambient=0.8, diffuse=0.8, specular=0.2),
                    flatshading=False
                ),
                row=1, col=1
            )
            arrow_trace_indices.append(current_trace_index)
            current_trace_index += 1
            n_arrow_traces += 1

            # Add cone arrowhead - HIDDEN BY DEFAULT
            tip_x = track_pos[0] + track_dir[0]*scale
            tip_y = track_pos[1] + track_dir[1]*scale
            tip_z = track_pos[2] + track_dir[2]*scale

            fig.add_trace(
                go.Cone(
                    x=[tip_x], y=[tip_y], z=[tip_z],
                    u=[track_dir[0]], v=[track_dir[1]], w=[track_dir[2]],
                    colorscale=[[0, color], [1, color]],
                    sizemode="absolute",
                    sizeref=20,
                    showscale=False,
                    name=f'{label_name} direction',
                    legendgroup=f'label{i}',
                    showlegend=False,
                    visible=False  # Hidden by default
                ),
                row=1, col=1
            )
            arrow_trace_indices.append(current_trace_index)
            current_trace_index += 1
            n_arrow_traces += 1

    # Add sensor hits colored by LABEL (not charge/time)
    hit_indices = event_data['hit_indices']
    hit_charges = event_data['hit_charges']
    Q_per_label = event_data['Q_per_label']  # Shape: (N_labels, N_sensors)
    T_per_label = event_data['T_per_label']

    # Calculate disc radius
    disc_radius = detector.S_radius * 1.0

    # Determine label contributions for each sensor
    n_sensors_hit = len(hit_indices)
    sensor_label_assignments = []  # List of (sensor_idx, contributing_labels, charges_per_label)

    for sensor_idx in hit_indices:
        # Find which labels contribute to this sensor
        label_charges = Q_per_label[:, sensor_idx]
        contributing_labels = np.where(label_charges > 0)[0]
        sensor_label_assignments.append((sensor_idx, contributing_labels, label_charges[contributing_labels]))

    # Compute sensor colors based on label with max charge
    sensor_colors_maxcharge = []
    sensor_charges_list = []

    for sensor_idx, contrib_labels, charges in sensor_label_assignments:
        total_charge = hit_charges[hit_indices == sensor_idx][0]
        sensor_charges_list.append(total_charge)

        if len(contrib_labels) == 1:
            # Single label: use its color
            label_idx = contrib_labels[0]
            color = event_data['label_colors'][label_idx]
            sensor_colors_maxcharge.append(color)
        else:
            # Multiple labels: use color of label with max charge
            max_label_idx = contrib_labels[np.argmax(charges)]
            sensor_colors_maxcharge.append(event_data['label_colors'][max_label_idx])

    # Get hit sensor positions and normals
    if len(hit_indices) > 0:
        hit_positions = detector.all_points[hit_indices]
        hit_normals = calculate_surface_normals(detector, hit_indices)

        # Sort by depth for rendering
        depth_order = np.argsort(hit_positions[:, 2])
        hit_positions_sorted = hit_positions[depth_order]
        hit_normals_sorted = hit_normals[depth_order]
        sensor_colors_maxcharge_sorted = [sensor_colors_maxcharge[i] for i in depth_order]
        sensor_charges_sorted = [sensor_charges_list[i] for i in depth_order]

        # Create sensor discs for each threshold level
        # Threshold levels: 1, 5, 10, 20, 50 PE (removed 0 since 0 PE = no hit)
        threshold_levels = [1, 5, 10, 20, 50]
        n_sensor_trace_sets = len(threshold_levels)
        sensor_a_trace_indices = []

        for threshold_idx, threshold in enumerate(threshold_levels):
            # Filter sensors by threshold
            mask = np.array(sensor_charges_sorted) >= threshold

            if not np.any(mask):
                continue  # Skip if no sensors above threshold

            positions_filtered = hit_positions_sorted[mask]
            normals_filtered = hit_normals_sorted[mask]
            colors_maxcharge_filtered = [c for i, c in enumerate(sensor_colors_maxcharge_sorted) if mask[i]]

            # Create disc meshes colored by label with max charge
            all_vertices_a = []
            all_faces_a = []
            all_colors_a = []
            vertex_offset = 0

            for pos, normal, color in zip(positions_filtered, normals_filtered, colors_maxcharge_filtered):
                vertices, faces = create_disc_mesh(pos, normal, disc_radius, n_segments=12)
                faces_adjusted = faces + vertex_offset
                all_vertices_a.append(vertices)
                all_faces_a.append(faces_adjusted)
                # Assign same color to all vertices of this disc
                all_colors_a.extend([color] * len(vertices))
                vertex_offset += len(vertices)

            if all_vertices_a:
                combined_vertices_a = np.vstack(all_vertices_a)
                combined_faces_a = np.vstack(all_faces_a)

                # Convert color names to RGB for proper vertex coloring
                # We'll use vertexcolor instead of intensity
                fig.add_trace(
                    go.Mesh3d(
                        x=combined_vertices_a[:, 0],
                        y=combined_vertices_a[:, 1],
                        z=combined_vertices_a[:, 2],
                        i=combined_faces_a[:, 0],
                        j=combined_faces_a[:, 1],
                        k=combined_faces_a[:, 2],
                        vertexcolor=all_colors_a,
                        opacity=0.9,
                        name='Sensor Hits',
                        showlegend=(threshold_idx == 0),  # Only show first in legend
                        lighting=dict(ambient=0.8, diffuse=0.8, specular=0.1),
                        visible=(threshold_idx == 0),  # Show first threshold by default
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
                sensor_a_trace_indices.append(current_trace_index)
                current_trace_index += 1

    # Add stacked histograms for Q_true and T_true
    # Sort labels by total charge (most charge at bottom of stack)
    label_total_charges = [event_data['label_q_values'][i] for i in range(event_data['n_labels'])]
    sorted_label_indices = np.argsort(label_total_charges)[::-1]  # Descending order (largest first, will be at bottom)

    # Filter to only include labels that exist in LUCiD data
    n_lucid_labels = Q_per_label.shape[0]
    sorted_label_indices = [idx for idx in sorted_label_indices if idx < n_lucid_labels]

    # Track actual number of histogram traces added
    n_q_histogram_traces = 0

    # Q_true histogram (stacked by label) - distribution of charge values
    for idx in sorted_label_indices:
        q_values = Q_per_label[idx, :]
        # Only include sensors with charge > 0
        q_nonzero = q_values[q_values > 0]

        if len(q_nonzero) > 0:
            fig.add_trace(
                go.Histogram(
                    x=q_nonzero,
                    name=event_data['label_names'][idx],
                    marker=dict(color=event_data['label_colors'][idx]),
                    legendgroup=f'label{idx}',
                    showlegend=False,  # Already shown in 3D plot
                    opacity=0.8,
                    nbinsx=50,
                    hovertemplate='Charge: %{x:.1f} PE<br>Count: %{y}<extra></extra>'
                ),
                row=2, col=1
            )
            n_q_histogram_traces += 1

    # Track actual number of T histogram traces added
    n_t_histogram_traces = 0

    # T_true histogram (stacked by label) - distribution of time values
    for idx in sorted_label_indices:
        t_values = T_per_label[idx, :]
        q_values = Q_per_label[idx, :]
        # Only include times where charge > 1 PE
        valid_mask = q_values > 1.0
        t_valid = t_values[valid_mask]

        if len(t_valid) > 0:
            fig.add_trace(
                go.Histogram(
                    x=t_valid,
                    name=event_data['label_names'][idx],
                    marker=dict(color=event_data['label_colors'][idx]),
                    legendgroup=f'label{idx}',
                    showlegend=False,
                    opacity=0.8,
                    nbinsx=50,
                    hovertemplate='Time: %{x:.1f} ns<br>Count: %{y}<extra></extra>'
                ),
                row=2, col=2
            )
            n_t_histogram_traces += 1

    # Update subplot axes (linear scale for counts)
    fig.update_xaxes(title_text="Charge (PE)", row=2, col=1, color='white', gridcolor='gray')
    fig.update_yaxes(title_text="Count", row=2, col=1, color='white', gridcolor='gray')
    fig.update_xaxes(title_text="Time (ns)", row=2, col=2, color='white', gridcolor='gray', exponentformat='none')
    fig.update_yaxes(title_text="Count", row=2, col=2, color='white', gridcolor='gray')

    # Calculate trace counts
    # Structure: photons, arrows, sensors (per threshold), Q_histograms, T_histograms
    n_sensor_traces = n_sensor_trace_sets if len(hit_indices) > 0 else 0
    n_histogram_traces_per_plot = event_data['n_labels']
    n_total_traces = len(fig.data)

    # Create visibility arrays
    # Helper function to build visibility array
    def make_visibility(show_photons, show_arrows, show_sensors, show_histograms):
        vis = []
        # Photons
        vis.extend([show_photons] * n_photon_traces)
        # Arrows
        vis.extend([show_arrows] * n_arrow_traces)
        # Sensors (per threshold)
        vis.extend([show_sensors] * n_sensor_traces)
        # Q histograms
        vis.extend([show_histograms] * n_histogram_traces_per_plot)
        # T histograms
        vis.extend([show_histograms] * n_histogram_traces_per_plot)
        return vis

    # Use the tracked trace indices for button logic
    photon_indices = photon_trace_indices
    arrow_indices = arrow_trace_indices
    sensor_indices = sensor_a_trace_indices

    # Button visibility states
    # Build show_all manually to only show first threshold
    show_all = []
    show_all.extend([True] * n_photon_traces)  # All photons visible
    show_all.extend([True] * n_arrow_traces)   # All arrows visible
    # Only first threshold visible
    for i in range(len(sensor_indices)):
        show_all.append(i == 0)  # Only first threshold visible
    show_all.extend([True] * n_q_histogram_traces)  # Q histograms visible
    show_all.extend([True] * n_t_histogram_traces)  # T histograms visible

    # Create slider steps for charge threshold
    slider_steps = []
    for threshold_idx, threshold in enumerate(threshold_levels):
        # Create visibility for this threshold
        step_vis = []
        # Photons and arrows: keep current state (use None to not change)
        step_vis.extend([None] * (n_photon_traces + n_arrow_traces))
        # Sensors: show only this threshold level
        for i in range(n_sensor_traces):
            step_vis.append(i == threshold_idx)
        # Histograms: keep visible
        step_vis.extend([None] * (n_q_histogram_traces + n_t_histogram_traces))

        slider_steps.append(dict(
            method="update",
            args=[{"visible": step_vis}],
            label=f"{threshold} PE"
        ))

    # Update layout
    primary_desc = ', '.join(sorted(primary_particles)) if primary_particles else "Unknown"
    fig.update_layout(
        title=dict(
            text=f'Event {event_idx}: LUCiD Label Validation ({primary_desc})',
            font=dict(color='white', size=16)
        ),
        scene=dict(
            xaxis_title='X (cm)',
            yaxis_title='Y (cm)',
            zaxis_title='Z (cm)',
            aspectmode='data',
            bgcolor='black',
            xaxis=dict(gridcolor='gray', color='white'),
            yaxis=dict(gridcolor='gray', color='white'),
            zaxis=dict(gridcolor='gray', color='white')
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        legend=dict(
            itemsizing='constant',
            itemwidth=50,
            tracegroupgap=5,
            font=dict(size=14),
            bgcolor='rgba(0,0,0,0.5)',
            uirevision='legend',  # Prevent legend from resizing when traces are toggled
            x=0.01,  # Position in the reserved left space
            xanchor='left',
            y=0.85,  # Moved down to avoid title and slider
            yanchor='top'
        ),
        barmode='stack',  # Stack the histogram bars
        updatemenus=[
            # Show All button
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": show_all}],
                        label="Show All",
                        method="update"
                    )
                ],
                pad={"r": 5, "t": 5},
                showactive=False,
                x=0.87,
                xanchor="left",
                y=0.22,
                yanchor="top",
                bgcolor='rgba(50,50,50,0.8)',
                bordercolor='white',
                font=dict(color='white', size=14)
            ),
            # Toggle Photons
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": True}, photon_indices],
                        label="Photons ON",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": False}, photon_indices],
                        label="Photons OFF",
                        method="restyle"
                    )
                ],
                pad={"r": 5, "t": 5},
                showactive=False,
                x=0.87,
                xanchor="left",
                y=0.18,
                yanchor="top",
                bgcolor='rgba(50,50,50,0.8)',
                bordercolor='white',
                font=dict(color='white', size=14)
            ),
            # Toggle Arrows
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": True}, arrow_indices],
                        label="Arrows ON",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": False}, arrow_indices],
                        label="Arrows OFF",
                        method="restyle"
                    )
                ],
                pad={"r": 5, "t": 5},
                showactive=False,
                x=0.87,
                xanchor="left",
                y=0.14,
                yanchor="top",
                bgcolor='rgba(50,50,50,0.8)',
                bordercolor='white',
                font=dict(color='white', size=14)
            ),
            # Toggle Sensors
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": True}, sensor_indices],
                        label="Sensors ON",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": False}, sensor_indices],
                        label="Sensors OFF",
                        method="restyle"
                    )
                ],
                pad={"r": 5, "t": 5},
                showactive=False,
                x=0.87,
                xanchor="left",
                y=0.10,
                yanchor="top",
                bgcolor='rgba(50,50,50,0.8)',
                bordercolor='white',
                font=dict(color='white', size=14)
            ),
        ],
        sliders=[
            dict(
                active=0,  # Start with threshold=0
                yanchor="top",
                y=1.08,
                xanchor="left",
                x=0.0,
                currentvalue=dict(
                    prefix="Charge Threshold: ",
                    visible=True,
                    xanchor="left",
                    font=dict(color='white')
                ),
                pad={"b": 10, "t": 50},
                len=0.6,
                steps=slider_steps
            )
        ],
        height=1000,
        width=1900,  # Increased width to accommodate legend and buttons
        uirevision='constant'  # Prevent layout changes when traces are toggled
    )

    # Fix scene domain to prevent resizing when traces are toggled
    # Reserve space on the left for legend (0-0.29), 3D plot (0.30-0.95), buttons on right (0.96+)
    fig.update_scenes(
        domain=dict(x=[0.2, 0.95], y=[0.25, 0.99])
    )

    # Fix subplot domains for histograms (keep them in original position)
    fig.update_xaxes(domain=[0.03, 0.41], row=2, col=1)
    fig.update_xaxes(domain=[0.47, 0.85], row=2, col=2)

    # Set initial visibility state
    for i, vis in enumerate(show_all):
        fig.data[i].visible = vis

    # Save as HTML
    root_basename = os.path.splitext(os.path.basename(root_file))[0]
    hdf5_basename = os.path.splitext(os.path.basename(hdf5_file))[0]
    filename = f'lucid_validation_{root_basename}_{hdf5_basename}_event{event_idx}.html'
    fig.write_html(filename)
    print(f"  ✓ Saved event {event_idx} to {filename}")

print()
print("="*70)
print("VALIDATION COMPLETE")
print("="*70)
print(f"\nGenerated {len(events_data)} HTML files")
print("Open the HTML files in a browser to interactively explore the 3D plots")
print("Toggle between different views using the buttons at the top of the plot")
