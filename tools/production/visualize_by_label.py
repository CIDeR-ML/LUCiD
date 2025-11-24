#!/usr/bin/env python3
"""
Simplified sensor visualization by label - HDF5 only version.

Shows sensor hits colored by charge value, with arrows showing track directions.
No photon visualization - only tracks (arrows) and sensor hits.

Usage:
    python visualize_by_label.py <hdf5_file> <detector_config> --event 0
"""
import sys
sys.path.append('/Users/cjesus/Software/LUCiD')

import os
import numpy as np
import plotly.graph_objects as go
from tools.production.label_data_utils import read_label_event_file
from tools.geometry import generate_detector
from tools.geometry.utils import calculate_surface_normals, create_disc_mesh
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize sensor hits by label')
parser.add_argument('hdf5_file', type=str, help='Input HDF5 file from LUCiD (label-based output)')
parser.add_argument('detector_config', type=str, help='Detector configuration JSON file')
parser.add_argument('--event', type=int, default=0, help='Event index to visualize (default: 0)')
parser.add_argument('--min-charge', type=float, default=1.0, help='Minimum charge threshold in PE (default: 1.0)')
args = parser.parse_args()

hdf5_file = args.hdf5_file
detector_config = args.detector_config
event_idx = args.event
min_charge = args.min_charge

print("="*70)
print(f"SENSOR VISUALIZATION BY LABEL")
print("="*70)
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
    11: 'e-', -11: 'e+', 13: 'mu-', -13: 'mu+', 22: 'gamma',
    111: 'pi0', 211: 'pi+', -211: 'pi-',
    321: 'K+', -321: 'K-',
    2212: 'proton', -2212: 'antiproton',
    2112: 'neutron', -2112: 'antineutron'
}

# Define colors for labels (matching old visualization)
colors_palette = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow',
                  'brown', 'pink', 'olive', 'navy', 'teal', 'maroon']

def create_cylinder(start, end, radius, n_segments=12):
    """Create a cylinder mesh from start to end point with given radius."""
    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-6:
        return [], [], [], [], [], []

    direction = direction / length

    # Find perpendicular vectors
    if abs(direction[2]) < 0.9:
        perp1 = np.cross(direction, np.array([0, 0, 1]))
    else:
        perp1 = np.cross(direction, np.array([1, 0, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)

    # Generate cylinder vertices
    angles = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
    vertices = []

    # Bottom circle
    for angle in angles:
        offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
        vertices.append(start + offset)

    # Top circle
    for angle in angles:
        offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
        vertices.append(end + offset)

    vertices = np.array(vertices)

    # Generate faces (triangles)
    faces_i, faces_j, faces_k = [], [], []
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        # Side faces (2 triangles per segment)
        faces_i.append(i)
        faces_j.append(next_i)
        faces_k.append(i + n_segments)
        faces_i.append(next_i)
        faces_j.append(next_i + n_segments)
        faces_k.append(i + n_segments)

    if len(vertices) == 0:
        return [], [], [], [], [], []

    return vertices[:, 0], vertices[:, 1], vertices[:, 2], faces_i, faces_j, faces_k


print(f"Loading event {event_idx}...")
print()

# Load LUCiD data (contains all track and sensor information)
lucid_data = read_label_event_file(hdf5_file, event_index=event_idx, verbose=False)
n_labels = lucid_data['n_labels']
Q_per_label = lucid_data['Q_per_label']  # Shape: (N_labels, N_sensors)
T_per_label = lucid_data['T_per_label']
Q_true = lucid_data['Q_true']

print(f"Event: {n_labels} labels")
print(f"Q_per_label shape: {Q_per_label.shape}")

# Find global max charge for colorbar normalization
global_max_charge = np.max(Q_per_label)
print(f"Global max charge: {global_max_charge:.1f} PE")
print()

# Prepare label information from HDF5 data
label_info = []
for label_idx in range(n_labels):
    pdg = lucid_data['Track_PDG'][label_idx]
    particle_name = pdg_to_name.get(int(pdg), f'PDG{int(pdg)}')

    # Get category name
    category_name = lucid_data['Label_CategoryName'][label_idx]
    if isinstance(category_name, bytes):
        category_name = category_name.decode('utf-8')

    kinetic_energy = lucid_data['Track_Energy'][label_idx]
    total_charge = np.sum(Q_per_label[label_idx, :])

    label_name = f"Label {label_idx}: {particle_name}"
    label_info.append({
        'idx': label_idx,
        'name': label_name,
        'particle': particle_name,
        'category': category_name,
        'energy': kinetic_energy,
        'total_charge': total_charge,
        'color': colors_palette[label_idx % len(colors_palette)]
    })

    # Print detailed label information
    print("=" * 80)
    print(f"{label_name} ({category_name})")
    print("=" * 80)

    # Print genealogy
    genealogy = lucid_data['Label_Genealogy'][label_idx]
    if len(genealogy) > 0:
        print(f"  Genealogy: {genealogy}")

    # Print detailed track info
    pos_m = lucid_data['Track_Position'][label_idx]
    dir_vec = lucid_data['Track_Direction'][label_idx]
    print(f"  PDG: {pdg} ({particle_name})")
    print(f"  Category: {category_name}")
    print(f"  Kinetic Energy: {kinetic_energy:.2f} MeV")
    print(f"  Position: [{pos_m[0]:.3f}, {pos_m[1]:.3f}, {pos_m[2]:.3f}] m")
    print(f"  Direction: [{dir_vec[0]:.3f}, {dir_vec[1]:.3f}, {dir_vec[2]:.3f}]")
    print(f"  Total charge deposited: {total_charge:.1f} PE")

print()

# Build track tree from HDF5 genealogy data
track_tree = {}
category_names_map = {0: "Primary", 1: "DecayElectron", 2: "SecondaryPion", 3: "GammaShower", -1: "Unknown"}

# Process each label to build the track tree
for label_idx in range(n_labels):
    genealogy = lucid_data['Label_Genealogy'][label_idx]
    if isinstance(genealogy, np.ndarray):
        genealogy_list = genealogy.tolist()
    else:
        genealogy_list = genealogy

    pdg = lucid_data['Track_PDG'][label_idx]
    particle_name = pdg_to_name.get(int(pdg), f'PDG{int(pdg)}')
    category_code = lucid_data['Label_Category'][label_idx]
    category_name = category_names_map.get(int(category_code), f'Category_{int(category_code)}')
    energy = lucid_data['Track_Energy'][label_idx]
    parent_id = lucid_data['Track_ParentID'][label_idx]

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
            # For tracks not at the end of genealogy, we don't have full info yet
            # We'll update them as we encounter them
            track_tree[track_id] = {
                'particle': particle_name if depth == len(genealogy_list) - 1 else f'Track_{track_id}',
                'category': category_name if depth == len(genealogy_list) - 1 else 'Unknown',
                'energy': energy if depth == len(genealogy_list) - 1 else 0.0,
                'parent_id': parent_id if depth == len(genealogy_list) - 1 else (genealogy_list[depth - 1] if depth > 0 else 0),
                'children': set(),
                'charge': 0.0,
                'time': 0.0,
                'label_id': None,
                'pdg': pdg if depth == len(genealogy_list) - 1 else 0
            }

        # If this is the photon-producing track (last in genealogy), store its data
        if depth == len(genealogy_list) - 1:
            track_tree[track_id].update({
                'particle': particle_name,
                'category': category_name,
                'energy': energy,
                'parent_id': parent_id,
                'charge': total_charge,
                'time': avg_time,
                'label_id': label_idx,
                'pdg': pdg
            })

        # Link parent-child relationship
        if depth > 0:
            parent_track_id = genealogy_list[depth - 1]
            if parent_track_id in track_tree:
                track_tree[parent_track_id]['children'].add(track_id)


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

event_genealogy_text = "<br>".join(genealogy_text_lines) + "<br>&nbsp;<br>&nbsp;"

print("Creating visualization...")

# Create figure
fig = go.Figure()

# Calculate sensor disc radius
disc_radius = detector.S_radius * 1.0

# Track indices for arrow traces
arrow_trace_indices = []

# Create arrow traces for each label
print("Creating arrow traces...")
for i, label in enumerate(label_info):
    color = label['color']
    label_name = label['name']

    # Get track position and direction (from LUCiD HDF5, stored in meters)
    track_pos = lucid_data['Track_Position'][label['idx']]
    track_dir = lucid_data['Track_Direction'][label['idx']]

    # Add arrow (cylinder + cone) to show track direction
    arrow_scale = 0.20  # Arrow length in meters (20 cm)
    cylinder_radius = 0.015  # Cylinder radius in meters (1.5 cm)

    # Draw the cylinder shaft
    cylinder_end = track_pos + track_dir * arrow_scale
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
                visible=True,
                lighting=dict(ambient=0.8, diffuse=0.8, specular=0.2),
                flatshading=False
            )
        )
        arrow_trace_indices.append(len(fig.data) - 1)

        # Add cone arrowhead
        tip_x = track_pos[0] + track_dir[0] * arrow_scale
        tip_y = track_pos[1] + track_dir[1] * arrow_scale
        tip_z = track_pos[2] + track_dir[2] * arrow_scale

        fig.add_trace(
            go.Cone(
                x=[tip_x], y=[tip_y], z=[tip_z],
                u=[track_dir[0]], v=[track_dir[1]], w=[track_dir[2]],
                colorscale=[[0, color], [1, color]],
                sizemode="absolute",
                sizeref=0.2,  # Cone size in meters (20 cm)
                showscale=False,
                name=f'{label_name} direction',
                legendgroup=f'label{i}',
                showlegend=False,
                visible=True
            )
        )
        arrow_trace_indices.append(len(fig.data) - 1)

        print(f"  Added arrow for {label_name}")

print(f"  Total arrow traces: {len(arrow_trace_indices)}")
print()

# Create "All" trace showing total charge across all labels
all_hit_mask = Q_true >= min_charge
all_hit_indices = np.where(all_hit_mask)[0]

if len(all_hit_indices) > 0:
    all_charges = Q_true[all_hit_indices]
    all_positions = detector.all_points[all_hit_indices]
    all_normals = calculate_surface_normals(detector, all_hit_indices)

    # Build combined mesh for "All"
    all_vertices = []
    all_faces = []
    all_intensities = []
    vertex_offset = 0

    for pos, normal, charge in zip(all_positions, all_normals, all_charges):
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
            colorbar=dict(title="Charge (PE)", x=1.15),
            name='All',
            showscale=True,
            visible=True,
            lighting=dict(ambient=0.8, diffuse=0.5, specular=0.1),
            flatshading=True
        )
    )
    print(f"  All: {len(all_hit_indices)} sensors above {min_charge} PE")

# Create "By Label" trace with discrete colors per label
# Resolve overlaps by assigning each sensor to the label with max charge contribution
if len(all_hit_indices) > 0:
    by_label_vertices = []
    by_label_faces = []
    by_label_colors = []
    vertex_offset = 0

    # Sort sensors by position for consistent visualization
    all_positions_sorted = all_positions
    all_normals_sorted = all_normals
    all_hit_indices_sorted = all_hit_indices

    for idx, sensor_idx in enumerate(all_hit_indices_sorted):
        # Find which label contributed most charge to this sensor
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
else:
    by_label_vertices = []

# Add "By Label" trace
if len(by_label_vertices) > 0:
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
            vertexcolor=by_label_colors,
            name='By Label',
            showscale=False,
            visible=False,
            lighting=dict(ambient=0.8, diffuse=0.5, specular=0.1),
            flatshading=True
        )
    )
    print(f"  By Label: {len(all_hit_indices)} sensors (color-coded)")

# Create sensor meshes for each individual label
for label_idx in range(n_labels):
    charges = Q_per_label[label_idx, :]
    hit_mask = charges >= min_charge
    hit_indices = np.where(hit_mask)[0]

    if len(hit_indices) == 0:
        # Add dummy trace to maintain indexing
        fig.add_trace(go.Mesh3d(
            x=[0], y=[0], z=[0],
            i=[0], j=[0], k=[0],
            opacity=0,
            showscale=False,
            visible=False,
            name=f'Label {label_idx}'
        ))
        print(f"  Warning: Label {label_idx} has no sensors above {min_charge} PE")
        continue

    label_charges = charges[hit_indices]
    positions = detector.all_points[hit_indices]
    normals = calculate_surface_normals(detector, hit_indices)

    label_vertices = []
    label_faces = []
    label_intensities = []
    vertex_offset = 0

    for pos, normal, charge in zip(positions, normals, label_charges):
        vertices, faces = create_disc_mesh(pos, normal, disc_radius, n_segments=12)
        faces_adjusted = faces + vertex_offset
        label_vertices.append(vertices)
        label_faces.append(faces_adjusted)
        label_intensities.extend([charge] * len(vertices))
        vertex_offset += len(vertices)

    combined_vertices = np.vstack(label_vertices)
    combined_faces = np.vstack(label_faces)

    fig.add_trace(
        go.Mesh3d(
            x=combined_vertices[:, 0],
            y=combined_vertices[:, 1],
            z=combined_vertices[:, 2],
            i=combined_faces[:, 0],
            j=combined_faces[:, 1],
            k=combined_faces[:, 2],
            intensity=label_intensities,
            colorscale='Viridis',
            cmin=0,
            cmax=global_max_charge,
            colorbar=dict(title="Charge (PE)", x=1.15),
            name=f'Label {label_idx}',
            showscale=True,
            visible=False,
            lighting=dict(ambient=0.8, diffuse=0.5, specular=0.1),
            flatshading=True
        )
    )
    print(f"  Label {label_idx}: {len(hit_indices)} sensors above {min_charge} PE")

print()

# Add detector geometry
print("Adding detector geometry...")
sensor_positions = detector.all_points
fig.add_trace(
    go.Scatter3d(
        x=sensor_positions[:, 0],
        y=sensor_positions[:, 1],
        z=sensor_positions[:, 2],
        mode='markers',
        marker=dict(size=1, color='lightgray', opacity=0.1),
        name='Detector',
        showlegend=False,
        visible=True
    )
)

# Create slider steps
slider_steps = []

# Step 0: "Arrows" - show arrows only (default view since no photons)
step_vis = [False] * len(fig.data)
for idx in arrow_trace_indices:
    step_vis[idx] = True
slider_steps.append(dict(
    method="update",
    args=[{"visible": step_vis}],
    label="Arrows"
))

# Step 1: "By Label" - show discrete color-coded sensors
step_vis = [False] * len(fig.data)
step_vis[len(arrow_trace_indices) + 1] = True  # By Label trace
slider_steps.append(dict(
    method="update",
    args=[{"visible": step_vis}],
    label="By Label"
))

# Step 2: "All" - show total charge across all labels
step_vis = [False] * len(fig.data)
step_vis[len(arrow_trace_indices)] = True  # All trace
slider_steps.append(dict(
    method="update",
    args=[{"visible": step_vis}],
    label="All"
))

# Steps 3+: Individual labels
for label_idx, info in enumerate(label_info):
    step_vis = [False] * len(fig.data)
    step_vis[len(arrow_trace_indices) + 2 + label_idx] = True  # Individual label trace
    slider_steps.append(dict(
        method="update",
        args=[{"visible": step_vis}],
        label=f"{label_idx}: {info['particle']}"
    ))

# Update layout with black theme matching old style
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
    updatemenus=[
        # Toggle Arrows
        dict(
            type="buttons",
            direction="left",
            buttons=[
                dict(
                    args=[{"visible": True}, arrow_trace_indices],
                    label="Arrows ON",
                    method="restyle"
                ),
                dict(
                    args=[{"visible": False}, arrow_trace_indices],
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
    ],
    sliders=[
        dict(
            active=0,
            yanchor="top",
            y=1.08,
            xanchor="left",
            x=0.0,
            currentvalue=dict(
                prefix="View: ",
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
    width=1600,
    showlegend=False
)

# Save to HTML
root_basename = os.path.splitext(os.path.basename(hdf5_file))[0]
filename = f'label_sensors_{root_basename}_event{event_idx}.html'
fig.write_html(filename)

print(f"Saved to: {filename}")

print()
print("="*70)
print("DONE")
print("="*70)
