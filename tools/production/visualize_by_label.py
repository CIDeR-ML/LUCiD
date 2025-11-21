#!/usr/bin/env python3
"""
Simplified sensor visualization by label.

This script visualizes sensor hits colored by charge value, with a slider to
select which label's sensors to display. Provides a focused view for exploring
how each label (particle type) contributes charge to different sensors.

Usage:
    python visualize_by_label.py <root_file> <hdf5_file> <detector_config> --event 0
"""
import sys
sys.path.append('/Users/cjesus/Software/LUCiD')

import os
import numpy as np
import plotly.graph_objects as go
from tools.generate import read_label_data_from_photonsim
from tools.production.label_data_utils import read_label_event_file
from tools.geometry import generate_detector
from tools.geometry.utils import calculate_surface_normals, create_disc_mesh
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize sensor hits by label')
parser.add_argument('root_file', type=str, help='Input ROOT file from PhotonSim')
parser.add_argument('hdf5_file', type=str, help='Input HDF5 file from LUCiD (label-based output)')
parser.add_argument('detector_config', type=str, help='Detector configuration JSON file')
parser.add_argument('--event', type=int, default=0, help='Event index to visualize (default: 0)')
parser.add_argument('--min-charge', type=float, default=1.0, help='Minimum charge threshold in PE (default: 1.0)')
args = parser.parse_args()

root_file = args.root_file
hdf5_file = args.hdf5_file
detector_config = args.detector_config
event_idx = args.event
min_charge = args.min_charge

print("="*70)
print(f"SENSOR VISUALIZATION BY LABEL")
print("="*70)
print(f"PhotonSim ROOT file: {root_file}")
print(f"LUCiD HDF5 file: {hdf5_file}")
print(f"Detector config: {detector_config}")
print(f"Event: {event_idx}")
print(f"Min charge threshold: {min_charge} PE")
print()

# Load detector geometry
print("Loading detector geometry...")
detector = generate_detector(detector_config)
print(f"Detector: {detector.__class__.__name__}")
print(f"  Number of sensors: {detector.n_sensors}")
print(f"  Sensor radius: {detector.S_radius:.3f} m")
print()

# PDG code to particle name mapping
pdg_to_name = {
    -211: 'pi-',
    211: 'pi+',
    -13: 'mu+',
    13: 'mu-',
    -11: 'e+',
    11: 'e-',
    22: 'gamma',
    2212: 'p',
    2112: 'n',
}

# Category names
category_names = {0: 'Primary', 1: 'DecayElectron', 2: 'SecondaryPion', 3: 'GammaShower'}

# Define colors for labels
colors_palette = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow',
                  'brown', 'pink', 'olive', 'navy', 'teal', 'maroon']


def print_genealogy_chain(genealogy, track_info_dict, pdg_to_name, category_names):
    """Print the genealogy chain with indentation showing hierarchy"""
    print("Genealogy chain (parent → child):")
    for depth, track_id in enumerate(genealogy):
        track = track_info_dict.get(track_id)
        if track:
            particle = pdg_to_name.get(track['pdg'], f"PDG{track['pdg']}")
            category = category_names.get(track['category'], 'Unknown')
            indent = "   " + " └─" * depth if depth > 0 else "  "
            print(f"{indent}[{depth+1}] {particle} ({category}) - "
                  f"TrackID: {track_id}, ParentID: {track['parent_id']}, "
                  f"Energy: {track['energy']:.2f} MeV")
        else:
            print(f"  Warning: Track {track_id} not found in track_info_dict")


print(f"Loading event {event_idx}...")
print()

# Load PhotonSim data
photonsim_data = read_label_data_from_photonsim(root_file, event_idx)
n_labels = photonsim_data['n_labels']
labels = photonsim_data['labels']
track_info_dict = photonsim_data['track_info_dict']
print(f"PhotonSim: {n_labels} labels")

# Load LUCiD data
lucid_data = read_label_event_file(hdf5_file, event_index=event_idx, verbose=False)
Q_per_label = lucid_data['Q_per_label']  # Shape: (N_labels, N_sensors)
T_per_label = lucid_data['T_per_label']
Q_true = lucid_data['Q_true']

print(f"LUCiD: Q_per_label shape {Q_per_label.shape}")

# Find global max charge for colorbar normalization
global_max_charge = np.max(Q_per_label)
print(f"Global max charge: {global_max_charge:.1f} PE")
print()

# Prepare label information
label_info = []
for label_idx, label in enumerate(labels):
    if label_idx >= Q_per_label.shape[0]:
        # LUCiD has fewer labels than PhotonSim
        break

    track_info = label['track_info']
    if track_info is None:
        continue

    pdg = track_info['pdg']
    particle_name = pdg_to_name.get(pdg, f'PDG{pdg}')
    category = category_names.get(track_info['category'], 'Unknown')
    kinetic_energy = track_info['energy']  # MeV
    total_charge = np.sum(Q_per_label[label_idx, :])

    label_name = f"Label {label_idx}: {particle_name}"
    label_info.append({
        'idx': label_idx,
        'name': label_name,
        'particle': particle_name,
        'category': category,
        'energy': kinetic_energy,
        'total_charge': total_charge,
        'color': colors_palette[label_idx % len(colors_palette)]
    })

    # Print detailed label information
    print("=" * 80)
    print(f"{label_name} ({category})")
    print("=" * 80)

    # Print genealogy chain
    if len(label['genealogy']) > 0:
        print_genealogy_chain(label['genealogy'], track_info_dict, pdg_to_name, category_names)
        print()

    # Print detailed track info for the photon-producing track
    photon_count = len(label['photon_indices'])
    # Convert position from cm (PhotonSim) to m for display
    pos_m = [track_info['position'][0]/100.0, track_info['position'][1]/100.0, track_info['position'][2]/100.0]
    print(f"Track producing photons: TrackID {track_info['track_id']} ({particle_name})")
    print(f"  Category: {category}")
    print(f"  Kinetic Energy: {kinetic_energy:.2f} MeV")
    print(f"  Position: [{pos_m[0]:.3f}, {pos_m[1]:.3f}, {pos_m[2]:.3f}] m")
    print(f"  Direction: [{track_info['direction'][0]:.3f}, {track_info['direction'][1]:.3f}, {track_info['direction'][2]:.3f}]")
    print(f"  Total charge deposited: {total_charge:.1f} PE")
    print(f"  Number of photons: {photon_count}")

print()

# Calculate charge and average time for each track
track_charge_time = {}  # {track_id: (charge, avg_time)}
for label_idx in range(min(Q_per_label.shape[0], len(labels))):
    label = labels[label_idx]
    genealogy = label['genealogy']
    last_track_id = genealogy[-1] if genealogy else None

    charges = Q_per_label[label_idx, :]
    times = T_per_label[label_idx, :]

    # Weighted average time
    total_charge = np.sum(charges)
    if total_charge > 0:
        # Average time weighted by charge
        avg_time = np.sum(charges * times) / total_charge
    else:
        avg_time = 0

    if last_track_id:
        track_charge_time[last_track_id] = (total_charge, avg_time)

# Build unified genealogy tree (each track appears only once)
track_tree = {}

for label_idx in range(min(Q_per_label.shape[0], len(labels))):
    label = labels[label_idx]
    genealogy = label['genealogy']

    # Get charge and time for the photon-producing track (last in genealogy)
    last_track_id = genealogy[-1] if genealogy else None
    charge, avg_time = track_charge_time.get(last_track_id, (0.0, 0.0))

    # Add each track in genealogy to tree
    for depth, track_id in enumerate(genealogy):
        track = track_info_dict.get(track_id)
        if not track:
            continue

        if track_id not in track_tree:
            track_tree[track_id] = {
                'particle': pdg_to_name.get(track['pdg'], f"PDG{track['pdg']}"),
                'category': category_names.get(track['category'], 'Unknown'),
                'energy': track['energy'],
                'parent_id': track['parent_id'],
                'children': set(),
                'charge': 0.0,
                'time': 0.0,
                'label_id': None
            }

        # If this is the photon-producing track, store its charge/time/label
        if track_id == last_track_id:
            track_tree[track_id]['charge'] = charge
            track_tree[track_id]['time'] = avg_time
            track_tree[track_id]['label_id'] = label_idx

        # Link parent-child relationship
        if depth > 0:
            parent_id = genealogy[depth - 1]
            if parent_id in track_tree:
                track_tree[parent_id]['children'].add(track_id)


def format_track_tree(track_id, track_tree, depth=0):
    """Recursively format track tree as text with color coding"""
    if track_id not in track_tree:
        return []

    track = track_tree[track_id]
    indent = "&nbsp;&nbsp;" + "&nbsp;&nbsp;" * depth
    arrow = "└─ " if depth > 0 else ""

    # Format label ID with color if present
    if track['label_id'] is not None:
        color = colors_palette[track['label_id'] % len(colors_palette)]
        particle_colored = f"<span style='color:{color};font-weight:bold'>{track['particle']}</span>"
        label_str = f" [Label {track['label_id']}]"
    else:
        particle_colored = track['particle']
        label_str = ""

    lines = [
        f"{indent}{arrow}{particle_colored} ({track['category']}) - "
        f"TrackID: {track_id}{label_str}",
        f"{indent}&nbsp;&nbsp;&nbsp;&nbsp;Energy: {track['energy']:.1f} MeV, "
        f"Charge: {track['charge']:.1f} PE, Avg Time: {track['time']:.1f} ns"
    ]

    # Add children recursively
    for child_id in sorted(track['children']):
        lines.extend(format_track_tree(child_id, track_tree, depth + 1))

    return lines


# Find root tracks (parent_id == 0 or not in tree)
root_tracks = [tid for tid, data in track_tree.items() if data['parent_id'] == 0]

# Build formatted text
genealogy_text_lines = [f"<b>EVENT {event_idx} - TRACK GENEALOGY</b>"]
for root_id in sorted(root_tracks):
    genealogy_text_lines.append("<br>")
    genealogy_text_lines.extend(format_track_tree(root_id, track_tree, depth=0))

event_genealogy_text = "<br>".join(genealogy_text_lines) + "<br>&nbsp;<br>&nbsp;"  # Add spacing with non-breaking spaces

print(f"Creating visualization...")

# Create figure
fig = go.Figure()

# Calculate sensor disc radius
disc_radius = detector.S_radius * 1.0

# Create "All" trace showing total charge across all labels
all_hit_mask = Q_true >= min_charge
all_hit_indices = np.where(all_hit_mask)[0]

if len(all_hit_indices) > 0:
    all_charges = Q_true[all_hit_indices]
    all_positions = detector.all_points[all_hit_indices]
    all_normals = calculate_surface_normals(detector, all_hit_indices)

    # Sort by depth
    depth_order = np.argsort(all_positions[:, 2])
    all_positions_sorted = all_positions[depth_order]
    all_normals_sorted = all_normals[depth_order]
    all_charges_sorted = all_charges[depth_order]
    all_hit_indices_sorted = all_hit_indices[depth_order]

    # Build combined mesh for "All"
    all_vertices = []
    all_faces = []
    all_intensities = []
    vertex_offset = 0

    for pos, normal, charge in zip(all_positions_sorted, all_normals_sorted, all_charges_sorted):
        vertices, faces = create_disc_mesh(pos, normal, disc_radius, n_segments=12)
        faces_adjusted = faces + vertex_offset
        all_vertices.append(vertices)
        all_faces.append(faces_adjusted)
        all_intensities.extend([charge] * len(vertices))
        vertex_offset += len(vertices)

    combined_vertices_all = np.vstack(all_vertices)
    combined_faces_all = np.vstack(all_faces)

    fig.add_trace(
        go.Mesh3d(
            x=combined_vertices_all[:, 0],
            y=combined_vertices_all[:, 1],
            z=combined_vertices_all[:, 2],
            i=combined_faces_all[:, 0],
            j=combined_faces_all[:, 1],
            k=combined_faces_all[:, 2],
            intensity=all_intensities,
            colorscale='Viridis',
            cmin=0,
            cmax=global_max_charge,
            opacity=0.9,
            name='All Labels',
            showlegend=False,
            lighting=dict(ambient=0.8, diffuse=0.8, specular=0.1),
            visible=False,  # Hidden by default, "By Label" shows first
            colorbar=dict(
                title="Charge (PE)",
                x=0.80,  # Much closer to plot
                xanchor='left',
                thickness=20,
                len=0.7
            ),
            hovertemplate='Charge: %{intensity:.1f} PE<extra></extra>'
        )
    )
    all_trace_idx = len(fig.data) - 1
    print(f"  All: {len(all_hit_indices)} sensors above {min_charge} PE")
else:
    all_trace_idx = None
    print(f"  Warning: No sensors above {min_charge} PE for 'All'")

# Create "By Label" trace with discrete colors per label
if len(all_hit_indices) > 0:
    by_label_vertices = []
    by_label_faces = []
    by_label_colors = []
    vertex_offset = 0

    for idx, sensor_idx in enumerate(all_hit_indices_sorted):
        # Find which label contributed most charge
        label_charges = Q_per_label[:, sensor_idx]
        if np.max(label_charges) > 0:
            max_label_idx = np.argmax(label_charges)
            color = colors_palette[max_label_idx % len(colors_palette)]
        else:
            color = 'gray'

        pos = all_positions_sorted[idx]
        normal = all_normals_sorted[idx]
        vertices, faces = create_disc_mesh(pos, normal, disc_radius, n_segments=12)
        faces_adjusted = faces + vertex_offset
        by_label_vertices.append(vertices)
        by_label_faces.append(faces_adjusted)
        by_label_colors.extend([color] * len(vertices))
        vertex_offset += len(vertices)

    combined_vertices_by_label = np.vstack(by_label_vertices)
    combined_faces_by_label = np.vstack(by_label_faces)

    fig.add_trace(
        go.Mesh3d(
            x=combined_vertices_by_label[:, 0],
            y=combined_vertices_by_label[:, 1],
            z=combined_vertices_by_label[:, 2],
            i=combined_faces_by_label[:, 0],
            j=combined_faces_by_label[:, 1],
            k=combined_faces_by_label[:, 2],
            vertexcolor=by_label_colors,  # Discrete colors, no colorscale
            opacity=0.9,
            name='By Label',
            showlegend=False,  # No legend
            lighting=dict(ambient=0.8, diffuse=0.8, specular=0.1),
            visible=True,  # Show by default (first slider item)
            hoverinfo='skip'
        )
    )
    by_label_trace_idx = len(fig.data) - 1
    print(f"  By Label: {len(all_hit_indices)} sensors (color-coded)")
else:
    by_label_trace_idx = None

# Create sensor meshes for each individual label
sensor_trace_indices = []

for label_idx, info in enumerate(label_info):
    # Get sensors with charge >= min_charge for this label
    charge_values = Q_per_label[label_idx, :]
    hit_mask = charge_values >= min_charge
    hit_sensor_indices = np.where(hit_mask)[0]

    if len(hit_sensor_indices) == 0:
        print(f"  Warning: Label {label_idx} has no sensors above {min_charge} PE")
        # Add dummy trace to maintain indexing
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=0),
            visible=False,
            showlegend=False,
            hoverinfo='skip'
        ))
        sensor_trace_indices.append(len(fig.data) - 1)
        continue

    hit_charges = charge_values[hit_sensor_indices]
    hit_positions = detector.all_points[hit_sensor_indices]
    hit_normals = calculate_surface_normals(detector, hit_sensor_indices)

    # Sort by depth (z-coordinate) for proper rendering
    depth_order = np.argsort(hit_positions[:, 2])
    hit_positions_sorted = hit_positions[depth_order]
    hit_normals_sorted = hit_normals[depth_order]
    hit_charges_sorted = hit_charges[depth_order]

    # Create combined mesh for all sensors of this label
    all_vertices = []
    all_faces = []
    all_intensities = []
    vertex_offset = 0

    for pos, normal, charge in zip(hit_positions_sorted, hit_normals_sorted, hit_charges_sorted):
        vertices, faces = create_disc_mesh(pos, normal, disc_radius, n_segments=12)
        faces_adjusted = faces + vertex_offset
        all_vertices.append(vertices)
        all_faces.append(faces_adjusted)
        # Assign charge value to all vertices of this disc
        all_intensities.extend([charge] * len(vertices))
        vertex_offset += len(vertices)

    combined_vertices = np.vstack(all_vertices)
    combined_faces = np.vstack(all_faces)

    # Add mesh trace
    fig.add_trace(
        go.Mesh3d(
            x=combined_vertices[:, 0],
            y=combined_vertices[:, 1],
            z=combined_vertices[:, 2],
            i=combined_faces[:, 0],
            j=combined_faces[:, 1],
            k=combined_faces[:, 2],
            intensity=all_intensities,
            colorscale='Viridis',
            cmin=0,
            cmax=global_max_charge,
            opacity=0.9,
            name=info['name'],
            showlegend=False,
            lighting=dict(ambient=0.8, diffuse=0.8, specular=0.1),
            visible=False,  # Hidden by default, "By Label" shows first
            colorbar=dict(
                title="Charge (PE)",
                x=0.80,  # Much closer to plot
                xanchor='left',
                thickness=20,
                len=0.7
            ),
            hovertemplate='Charge: %{intensity:.1f} PE<extra></extra>'
        )
    )
    sensor_trace_indices.append(len(fig.data) - 1)

    print(f"  Label {label_idx}: {len(hit_sensor_indices)} sensors above {min_charge} PE")

print()

# Create slider steps
slider_steps = []

# Step 0: "By Label" - show discrete color-coded sensors (default view)
if by_label_trace_idx is not None:
    step_vis = [False] * len(fig.data)
    step_vis[by_label_trace_idx] = True
    slider_steps.append(dict(
        method="update",
        args=[{"visible": step_vis}],
        label="By Label"
    ))

# Step 1: "All" - show total charge across all labels
if all_trace_idx is not None:
    step_vis = [False] * len(fig.data)
    step_vis[all_trace_idx] = True
    slider_steps.append(dict(
        method="update",
        args=[{"visible": step_vis}],
        label="All"
    ))

# Steps 2+: Individual labels
for label_idx, info in enumerate(label_info):
    # Create visibility array: show only this label's sensors
    step_vis = [False] * len(fig.data)
    step_vis[sensor_trace_indices[label_idx]] = True

    slider_steps.append(dict(
        method="update",
        args=[{"visible": step_vis}],  # Only update trace visibility
        label=f"{label_idx}: {info['particle']}"
    ))

# Update layout
fig.update_layout(
    title=dict(
        text=f'Event {event_idx}: Sensor Hits by Label',
        font=dict(color='white', size=16)
    ),
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        aspectmode='data',
        bgcolor='black',
        xaxis=dict(gridcolor='gray', color='white'),
        yaxis=dict(gridcolor='gray', color='white'),
        zaxis=dict(gridcolor='gray', color='white'),
        domain=dict(x=[0.0, 1.0], y=[0.10, 0.95])  # Larger 3D plot, more vertical space
    ),
    paper_bgcolor='black',
    plot_bgcolor='black',
    font=dict(color='white'),
    margin=dict(b=700, t=120, l=50, r=50),  # Generous bottom margin for text box
    annotations=[{
        "text": event_genealogy_text,  # Static text showing entire event
        "showarrow": False,
        "xref": "paper",
        "yref": "paper",
        "x": 0.5,
        "y": -0.05,  # Below the plot area with more spacing
        "xanchor": "center",
        "yanchor": "top",
        "align": "left",
        "bgcolor": "rgba(0, 0, 0, 0.7)",
        "bordercolor": "white",
        "borderwidth": 1,
        "borderpad": 10,
        "font": dict(color="white", size=14, family="monospace"),
        "width": 800,  # Wider to accommodate more info
        "height": None  # Auto height
    }],
    sliders=[
        dict(
            active=0,
            yanchor="top",
            y=1.08,
            xanchor="left",
            x=0.0,
            currentvalue=dict(
                prefix="Label: ",
                visible=True,
                xanchor="left",
                font=dict(color='white', size=14)
            ),
            pad={"b": 10, "t": 50},
            len=0.8,
            steps=slider_steps
        )
    ],
    height=1600,  # Generous height with larger 3D plot
    width=1600
)

# Save as HTML
root_basename = os.path.splitext(os.path.basename(root_file))[0]
hdf5_basename = os.path.splitext(os.path.basename(hdf5_file))[0]
filename = f'label_sensors_{root_basename}_{hdf5_basename}_event{event_idx}.html'
fig.write_html(filename)

print(f"Saved to: {filename}")
print()
print("="*70)
print("DONE")
print("="*70)
