#!/usr/bin/env python3
"""
Simplified sensor visualization by label - HDF5 only version.

Shows sensor hits colored by charge value, with arrows showing track directions.
No photon visualization - only tracks (arrows) and sensor hits.

Usage:
    python visualize_labeled_events.py <hdf5_file> <detector_config> --event 0
"""
import sys
sys.path.append('/Users/cjesus/Software/LUCiD')

import os
import numpy as np
import plotly.graph_objects as go
from tools.production.label_data_utils import read_label_event_file
from tools.production.voxelize import flat_index_to_position, VoxelGridConfig
from tools.geometry import generate_detector
from tools.geometry.utils import calculate_surface_normals, create_disc_mesh
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize sensor hits by label')
parser.add_argument('hdf5_file', type=str, help='Input HDF5 file from LUCiD (label-based output)')
parser.add_argument('detector_config', type=str, help='Detector configuration JSON file')
parser.add_argument('--event', type=int, default=0, help='Event index to visualize (default: 0)')
parser.add_argument('--min-charge', type=float, default=1.0, help='Minimum charge threshold in PE (default: 1.0)')
parser.add_argument('--output-dir', type=str, default=None, help='Output directory for HTML file (default: current directory)')
args = parser.parse_args()

hdf5_file = args.hdf5_file
detector_config = args.detector_config
event_idx = args.event
min_charge = args.min_charge
output_dir = args.output_dir

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

# Category names mapping
category_names_map = {0: "Primary", 1: "DecayElectron", 2: "SecondaryPion", 3: "GammaShower", -1: "Unknown"}

def get_category_name(code):
    return category_names_map.get(int(code), f'Category_{int(code)}')

# Load LUCiD data (contains sensor and particle information)
lucid_data = read_label_event_file(hdf5_file, event_index=event_idx, verbose=False)
n_particles = lucid_data['n_particles']
PE_per_particle = lucid_data['PE_per_particle']  # Shape: (n_particles, N_sensors)
T_per_particle = lucid_data['T_per_particle']
PE = lucid_data['PE']

# Load track info for PDG lookup and arrow visualization
import h5py
track_id_to_info = {}  # Map track ID to (pdg, position, direction, energy)
with h5py.File(hdf5_file, 'r') as f:
    if f'event_{event_idx}' in f:
        event_group = f[f'event_{event_idx}']
    else:
        event_group = f

    if 'TrackInformation' in event_group and 'Segments' in event_group:
        tracks_grp = event_group['TrackInformation']
        segs_grp = event_group['Segments']

        track_ids = np.array(tracks_grp['TrackID'])
        track_pdgs = np.array(tracks_grp['PDG'])
        track_energies = np.array(tracks_grp['InitialEnergy'])
        track_seg_offsets = np.array(tracks_grp['SegmentOffset'])
        track_n_segs = np.array(tracks_grp['NSegments'])

        # Get segment data (in cm, convert to m)
        seg_start_x = np.array(segs_grp['StartX']) / 100.0
        seg_start_y = np.array(segs_grp['StartY']) / 100.0
        seg_start_z = np.array(segs_grp['StartZ']) / 100.0
        seg_dir_x = np.array(segs_grp['DirX'])
        seg_dir_y = np.array(segs_grp['DirY'])
        seg_dir_z = np.array(segs_grp['DirZ'])

        for i, tid in enumerate(track_ids):
            offset = track_seg_offsets[i]
            if track_n_segs[i] > 0:
                pos = np.array([seg_start_x[offset], seg_start_y[offset], seg_start_z[offset]])
                dir_vec = np.array([seg_dir_x[offset], seg_dir_y[offset], seg_dir_z[offset]])
                # Normalize direction
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > 0:
                    dir_vec = dir_vec / dir_norm
            else:
                pos = np.array([0, 0, 0])
                dir_vec = np.array([0, 0, 1])

            track_id_to_info[int(tid)] = {
                'pdg': int(track_pdgs[i]),
                'position': pos,
                'direction': dir_vec,
                'energy': float(track_energies[i])
            }

print(f"Event: {n_particles} categorized particles")
print(f"t0: {lucid_data.get('t0', 0.0):.2f} ns")
print(f"PE_per_particle shape: {PE_per_particle.shape}")

# Find global max charge for colorbar normalization
global_max_charge = np.max(PE_per_particle)
print(f"Global max charge: {global_max_charge:.1f} PE")
print()

# Prepare particle information from HDF5 data
particle_info = []
for particle_idx in range(n_particles):
    category_code = lucid_data['Particle_Category'][particle_idx]
    category_name = get_category_name(category_code)
    total_charge = np.sum(PE_per_particle[particle_idx, :])
    containment = lucid_data['light_containment_by_particle'][particle_idx]

    # Get particle type from PDG using extended genealogy (last track ID)
    ext_genealogy = lucid_data.get('Particle_TrackGenealogy', [None] * n_particles)[particle_idx]
    particle_type = "unknown"
    particle_energy = 0.0
    particle_position = None
    particle_direction = None

    if ext_genealogy is not None and len(ext_genealogy) > 0:
        # Get the last track ID in the genealogy (the track that produced photons)
        last_track_id = int(ext_genealogy[-1])
        if last_track_id in track_id_to_info:
            pdg = track_id_to_info[last_track_id]['pdg']
            particle_type = pdg_to_name.get(pdg, f'PDG{pdg}')
            particle_energy = track_id_to_info[last_track_id]['energy']
            particle_position = track_id_to_info[last_track_id]['position']
            particle_direction = track_id_to_info[last_track_id]['direction']

    particle_name = f"Particle {particle_idx}"
    particle_info.append({
        'idx': particle_idx,
        'name': particle_name,
        'category': category_name,
        'particle_type': particle_type,
        'energy': particle_energy,
        'position': particle_position,
        'direction': particle_direction,
        'total_charge': total_charge,
        'containment': containment,
        'color': colors_palette[particle_idx % len(colors_palette)]
    })

    # Print detailed particle information
    print("=" * 80)
    print(f"{particle_name} ({category_name}) - {particle_type}")
    print("=" * 80)

    # Print genealogy
    genealogy = lucid_data['Particle_CategorizedGenealogy'][particle_idx]
    if len(genealogy) > 0:
        print(f"  Genealogy: {genealogy}")

    print(f"  Category: {category_name}")
    print(f"  Particle type: {particle_type} ({particle_energy:.1f} MeV)")
    print(f"  Total charge deposited: {total_charge:.1f} PE")
    print(f"  Light containment: {containment*100:.1f}%")

print()

# Keep label_info for compatibility with rest of code
label_info = particle_info

# Build particle tree from HDF5 genealogy data
particle_tree = {}

# Process each categorized particle
for particle_idx in range(n_particles):
    genealogy = lucid_data['Particle_CategorizedGenealogy'][particle_idx]
    if isinstance(genealogy, np.ndarray):
        genealogy_list = genealogy.tolist()
    else:
        genealogy_list = genealogy

    category_code = lucid_data['Particle_Category'][particle_idx]
    category_name = get_category_name(category_code)

    # Calculate charge and average time for this particle
    PE_particle = PE_per_particle[particle_idx]
    T_particle = T_per_particle[particle_idx]
    total_charge = np.sum(PE_particle)

    # Calculate weighted average time (only for sensors with finite times)
    finite_mask = np.isfinite(T_particle) & (PE_particle > 0)
    if np.any(finite_mask):
        finite_charges = PE_particle[finite_mask]
        finite_times = T_particle[finite_mask]
        avg_time = np.sum(finite_charges * finite_times) / np.sum(finite_charges)
    else:
        avg_time = 0.0

    containment = lucid_data['light_containment_by_particle'][particle_idx]

    # Get particle type and position/direction from particle_info
    pinfo = particle_info[particle_idx]

    # Store particle info
    particle_tree[particle_idx] = {
        'category': category_name,
        'particle_type': pinfo['particle_type'],
        'energy': pinfo['energy'],
        'position': pinfo['position'],
        'direction': pinfo['direction'],
        'genealogy': genealogy_list,
        'charge': total_charge,
        'time': avg_time,
        'containment': containment
    }

# Alias for compatibility with rest of code
track_tree = particle_tree
n_labels = n_particles
Q_per_label = PE_per_particle
T_per_label = T_per_particle
Q_true = PE


def build_particle_hierarchy(particle_tree, n_particles):
    """Build parent-child relationships between particles based on genealogy."""
    # Find which particles are "children" of others based on genealogy overlap
    # A particle B is a child of particle A if B's genealogy starts with A's genealogy
    children = {i: [] for i in range(n_particles)}
    roots = []

    for i in range(n_particles):
        gen_i = particle_tree[i]['genealogy']
        is_root = True

        for j in range(n_particles):
            if i == j:
                continue
            gen_j = particle_tree[j]['genealogy']

            # Check if particle i's genealogy starts with particle j's genealogy
            # (meaning i is a descendant of j)
            if len(gen_j) < len(gen_i) and gen_i[:len(gen_j)] == gen_j:
                # j is an ancestor of i - but we want the closest ancestor
                # Check if there's a closer ancestor
                is_closest = True
                for k in range(n_particles):
                    if k == i or k == j:
                        continue
                    gen_k = particle_tree[k]['genealogy']
                    # k is between j and i if j's gen is prefix of k's gen and k's gen is prefix of i's gen
                    if (len(gen_j) < len(gen_k) < len(gen_i) and
                        gen_i[:len(gen_k)] == gen_k and gen_k[:len(gen_j)] == gen_j):
                        is_closest = False
                        break

                if is_closest:
                    children[j].append(i)
                    is_root = False
                    break

        if is_root:
            roots.append(i)

    return roots, children


def format_particle_tree(particle_idx, particle_tree, children, depth=0):
    """Recursively format particle tree as HTML text with arrows and indentation."""
    if particle_idx not in particle_tree:
        return []

    particle = particle_tree[particle_idx]
    color = colors_palette[particle_idx % len(colors_palette)]

    # Format particle type with energy
    particle_type = particle.get('particle_type', 'unknown')
    energy = particle.get('energy', 0.0)
    type_str = f"{particle_type} ({energy:.1f} MeV)" if energy > 0 else particle_type

    category_colored = f"<span style='color:{color};font-weight:bold'>{particle['category']}: {type_str}</span>"

    indent = "&nbsp;&nbsp;" * (depth + 1)
    arrow = "└─ " if depth > 0 else ""

    containment_str = f", Containment: {particle['containment']*100:.1f}%"

    lines = [
        f"{indent}{arrow}{category_colored} [Particle {particle_idx}]",
        f"{indent}&nbsp;&nbsp;&nbsp;&nbsp;Charge: {particle['charge']:.1f} PE, Avg Time: {particle['time']:.1f} ns{containment_str}"
    ]

    # Add children recursively
    for child_idx in sorted(children.get(particle_idx, [])):
        lines.extend(format_particle_tree(child_idx, particle_tree, children, depth + 1))

    return lines


# Build particle hierarchy from genealogy
roots, children = build_particle_hierarchy(particle_tree, n_particles)

# Build formatted text for particle genealogy display
genealogy_text_lines = [f"<b>EVENT {event_idx} - CATEGORIZED PARTICLES</b>"]
for root_idx in sorted(roots):
    genealogy_text_lines.append("<br>")
    genealogy_text_lines.extend(format_particle_tree(root_idx, particle_tree, children, depth=0))

# Add overall light containment at the end
genealogy_text_lines.append("<br>")
overall_containment = lucid_data['overall_light_containment']
genealogy_text_lines.append(f"<br><b>Overall Light Containment:</b> {overall_containment*100:.1f}% of photons inside detector")

event_genealogy_text = "<br>".join(genealogy_text_lines) + "<br>&nbsp;<br>&nbsp;"

print("Creating visualization...")

# Create figure
fig = go.Figure()

# Calculate sensor disc radius
disc_radius = detector.S_radius * 1.0

# Create arrow traces for each categorized particle
print("Creating arrow traces...")
arrow_trace_indices = []
for i, pinfo in enumerate(particle_info):
    color = pinfo['color']
    particle_type = pinfo['particle_type']
    category = pinfo['category']

    # Get track position and direction
    track_pos = pinfo['position']
    track_dir = pinfo['direction']

    if track_pos is None or track_dir is None:
        print(f"  Skipping arrow for Particle {i} (no track info)")
        continue

    # Add arrow (cylinder + cone) to show track direction
    arrow_scale = 3.0  # Arrow length in meters (3 m)
    cylinder_radius = 0.075  # Cylinder radius in meters (7.5 cm)

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
                name=f'{particle_type} ({category})',
                legendgroup=f'particle{i}',
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
                sizeref=1.5,  # Cone size in meters (1.5 m)
                showscale=False,
                name=f'{particle_type} direction',
                legendgroup=f'particle{i}',
                showlegend=False,
                visible=True
            )
        )
        arrow_trace_indices.append(len(fig.data) - 1)

        print(f"  Added arrow for Particle {i}: {particle_type} ({category})")

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
            colorbar=dict(title="Charge (PE)", x=0.92, len=0.8),
            name='All',
            showscale=True,
            visible=False,
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
            colorbar=dict(title="Charge (PE)", x=0.92, len=0.8),
            name=f'Label {label_idx}',
            showscale=True,
            visible=False,
            lighting=dict(ambient=0.8, diffuse=0.5, specular=0.1),
            flatshading=True
        )
    )
    print(f"  Label {label_idx}: {len(hit_indices)} sensors above {min_charge} PE")

print()

# ============================================================================
# VOXEL VISUALIZATION
# ============================================================================
print("Creating voxel visualization...")

# Check if voxel data exists in the HDF5 file
has_voxel_data = False
voxel_trace_index = None

# Try to load voxel data directly from HDF5
import h5py
with h5py.File(hdf5_file, 'r') as f:
    # Navigate to the correct event group
    if f'event_{event_idx}' in f:
        event_group = f[f'event_{event_idx}']
    else:
        event_group = f  # Single event file

    if 'voxel_flat_indices' in event_group:
        has_voxel_data = True
        voxel_n_nonzero = np.array(event_group['voxel_n_nonzero'])
        voxel_offsets = np.array(event_group['voxel_offsets'])
        voxel_flat_indices = np.array(event_group['voxel_flat_indices'])
        voxel_counts = np.array(event_group['voxel_counts'])

# ============================================================================
# TRACK SEGMENT VISUALIZATION
# ============================================================================
print("Loading track segment data...")

# Check if track segment data exists
has_segment_data = False
segment_trace_indices = []

with h5py.File(hdf5_file, 'r') as f:
    if f'event_{event_idx}' in f:
        event_group = f[f'event_{event_idx}']
    else:
        event_group = f

    if 'TrackInformation' in event_group and 'Segments' in event_group:
        has_segment_data = True
        tracks_group = event_group['TrackInformation']
        segs_group = event_group['Segments']

        # Load track data
        track_ids = np.array(tracks_group['TrackID'])
        track_parent_ids = np.array(tracks_group['ParentID'])
        track_pdgs = np.array(tracks_group['PDG'])
        track_seg_offsets = np.array(tracks_group['SegmentOffset'])
        track_n_segs = np.array(tracks_group['NSegments'])
        track_n_cherenkov = np.array(tracks_group['NCherenkov'])
        # Get particle names from PDG codes
        track_names = [pdg_to_name.get(int(pdg), f'PDG{int(pdg)}') for pdg in track_pdgs]

        # Load segment data (positions in cm, convert to meters for display)
        seg_start_x = np.array(segs_group['StartX']) / 100.0
        seg_start_y = np.array(segs_group['StartY']) / 100.0
        seg_start_z = np.array(segs_group['StartZ']) / 100.0
        seg_end_x = np.array(segs_group['EndX']) / 100.0
        seg_end_y = np.array(segs_group['EndY']) / 100.0
        seg_end_z = np.array(segs_group['EndZ']) / 100.0

        print(f"  Found {len(track_ids)} meaningful tracks with {len(seg_start_x)} segments")

if has_segment_data and len(track_ids) > 0:
    MIN_SEGMENTS_TO_DISPLAY = 5  # Only show tracks with significant trajectory

    # Filter tracks with enough segments
    tracks_to_display = [i for i in range(len(track_ids))
                         if track_n_segs[i] > MIN_SEGMENTS_TO_DISPLAY]
    print(f"  Displaying {len(tracks_to_display)} tracks with >{MIN_SEGMENTS_TO_DISPLAY} segments")

    # Color mapping for particles
    particle_colors = {
        'mu-': 'red', 'mu+': 'magenta',
        'e-': 'blue', 'e+': 'cyan',
        'pi+': 'green', 'pi-': 'lime',
        'pi0': 'yellow',
        'gamma': 'orange',
        'proton': 'white', 'neutron': 'gray'
    }

    for track_idx in tracks_to_display:
        offset = track_seg_offsets[track_idx]
        n_segs = track_n_segs[track_idx]

        particle_name = track_names[track_idx]
        color = particle_colors.get(particle_name, 'white')

        # Create lines for each segment (with None separators for disconnected lines)
        x_coords = []
        y_coords = []
        z_coords = []

        for seg_idx in range(n_segs):
            i = offset + seg_idx
            x_coords.extend([seg_start_x[i], seg_end_x[i], None])
            y_coords.extend([seg_start_y[i], seg_end_y[i], None])
            z_coords.extend([seg_start_z[i], seg_end_z[i], None])

        fig.add_trace(go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='lines',
            line=dict(color=color, width=3),
            name=f'{particle_name} (Track {track_ids[track_idx]})',
            legendgroup=f'track_{track_idx}',
            showlegend=False,
            visible=False,
            hoverinfo='text',
            hovertext=f'{particle_name}<br>TrackID: {track_ids[track_idx]}<br>Cherenkov: {track_n_cherenkov[track_idx]}'
        ))
        segment_trace_indices.append(len(fig.data) - 1)

    print(f"  Created {len(segment_trace_indices)} track segment traces")
else:
    print("  No track segment data found in HDF5 file")

print()

if has_voxel_data and len(voxel_flat_indices) > 0:
    print(f"  Found voxel data: {len(voxel_flat_indices)} total voxels")

    # Convert flat indices to positions
    voxel_config = VoxelGridConfig()
    voxel_positions = flat_index_to_position(voxel_flat_indices, voxel_config)

    # Build arrays for scatter plot with colors per label
    all_voxel_x = []
    all_voxel_y = []
    all_voxel_z = []
    all_voxel_colors = []
    all_voxel_sizes = []
    all_voxel_text = []

    for particle_idx in range(n_particles):
        start = voxel_offsets[particle_idx]
        end = start + voxel_n_nonzero[particle_idx]

        particle_positions = voxel_positions[start:end]
        particle_counts = voxel_counts[start:end]
        particle_color = colors_palette[particle_idx % len(colors_palette)]

        # Get particle info for hover text
        category_name = particle_info[particle_idx]['category']

        for i, (pos, count) in enumerate(zip(particle_positions, particle_counts)):
            all_voxel_x.append(pos[0])
            all_voxel_y.append(pos[1])
            all_voxel_z.append(pos[2])
            all_voxel_colors.append(particle_color)
            # Scale marker size by log of photon count
            size = np.log10(count + 1) * 3 + 2
            all_voxel_sizes.append(size)
            all_voxel_text.append(
                f"Particle {particle_idx}: {category_name}<br>"
                f"Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m<br>"
                f"Photons: {count}"
            )

        print(f"    Particle {particle_idx} ({category_name}): {voxel_n_nonzero[particle_idx]} voxels, {np.sum(particle_counts)} photons")

    # Add voxel trace
    fig.add_trace(
        go.Scatter3d(
            x=all_voxel_x,
            y=all_voxel_y,
            z=all_voxel_z,
            mode='markers',
            marker=dict(
                size=all_voxel_sizes,
                color=all_voxel_colors,
                opacity=0.8,
                line=dict(width=0)
            ),
            text=all_voxel_text,
            hoverinfo='text',
            name='Voxels',
            showlegend=False,
            visible=False
        )
    )
    voxel_trace_index = len(fig.data) - 1
    print(f"  Total: {len(all_voxel_x)} voxels visualized")
else:
    print("  No voxel data found in HDF5 file")

print()

# Add detector geometry
print("Adding detector geometry...")

# Add detector cylinder surface (barely visible)
theta = np.linspace(0, 2*np.pi, 40)
z = np.linspace(-detector.H/2, detector.H/2, 40)
theta_grid, z_grid = np.meshgrid(theta, z)
x_cyl = detector.r * np.cos(theta_grid)
y_cyl = detector.r * np.sin(theta_grid)

fig.add_trace(
    go.Surface(
        x=x_cyl,
        y=y_cyl,
        z=z_grid,
        colorscale=[[0, 'gray'], [1, 'gray']],
        opacity=0.1,
        showscale=False,
        name='Detector Volume',
        hoverinfo='skip',
        showlegend=False,
        visible=True
    )
)
detector_surface_index = len(fig.data) - 1

# Add sensor positions as small dots
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
        visible=False
    )
)
detector_trace_index = len(fig.data) - 1

# Create slider steps
slider_steps = []

# Track indices for main visualization traces
# Trace order: Arrows..., "All", "By Particle", Individual particles..., voxel (if present), segments..., detector surface, detector points
n_arrow_traces = len(arrow_trace_indices)
all_trace_index = n_arrow_traces  # After arrow traces
by_particle_trace_index = n_arrow_traces + 1  # After "All"
individual_particle_start = n_arrow_traces + 2  # After "By Particle"

# Step: "Arrows" - show direction arrows for each categorized particle
if len(arrow_trace_indices) > 0:
    step_vis = [False] * len(fig.data)
    for idx in arrow_trace_indices:
        step_vis[idx] = True
    step_vis[detector_surface_index] = True  # Show detector surface
    slider_steps.append(dict(
        method="update",
        args=[{"visible": step_vis}],
        label="Arrows"
    ))

# Step: "Track Segments" - show track segment trajectories and detector surface
if len(segment_trace_indices) > 0:
    step_vis = [False] * len(fig.data)
    for idx in segment_trace_indices:
        step_vis[idx] = True
    step_vis[detector_surface_index] = True  # Show detector surface
    slider_steps.append(dict(
        method="update",
        args=[{"visible": step_vis}],
        label="Track Segments"
    ))

# Step: "Voxels" (if available) - show voxels and detector surface
if voxel_trace_index is not None:
    step_vis = [False] * len(fig.data)
    step_vis[voxel_trace_index] = True  # Voxel trace
    step_vis[detector_surface_index] = True  # Show detector surface
    slider_steps.append(dict(
        method="update",
        args=[{"visible": step_vis}],
        label="Voxels"
    ))

# Step: "By Particle" - show discrete color-coded sensors and detector surface
step_vis = [False] * len(fig.data)
step_vis[by_particle_trace_index] = True  # By Particle trace
step_vis[detector_surface_index] = True  # Show detector surface
slider_steps.append(dict(
    method="update",
    args=[{"visible": step_vis}],
    label="By Particle"
))

# Step: "All" - show total charge across all particles and detector surface
step_vis = [False] * len(fig.data)
step_vis[all_trace_index] = True  # All trace
step_vis[detector_surface_index] = True  # Show detector surface
slider_steps.append(dict(
    method="update",
    args=[{"visible": step_vis}],
    label="All"
))

# Steps: Individual particles and detector surface
for particle_idx, info in enumerate(particle_info):
    step_vis = [False] * len(fig.data)
    step_vis[individual_particle_start + particle_idx] = True  # Individual particle trace
    step_vis[detector_surface_index] = True  # Show detector surface
    slider_steps.append(dict(
        method="update",
        args=[{"visible": step_vis}],
        label=f"{particle_idx}: {info['category']}"
    ))

# Update layout with black theme matching old style
fig.update_layout(
    title=dict(
        text=f'Event {event_idx}: Sensor Hits by Particle',
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
        domain=dict(x=[0.0, 0.85], y=[0.10, 0.95])  # Fixed domain leaving room for colorbar
    ),
    paper_bgcolor='black',
    plot_bgcolor='black',
    font=dict(color='white'),
    margin=dict(b=700, t=120, l=50, r=50),  # Bottom margin for text box
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
                prefix="View: ",
                visible=True,
                xanchor="left",
                font=dict(color='white', size=14)
            ),
            pad={"b": 10, "t": 50},
            len=0.95,
            ticklen=10,
            minorticklen=0,
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

# Use output directory if specified
if output_dir is not None:
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, filename)

fig.write_html(filename)

print(f"Saved to: {filename}")

print()
print("="*70)
print("DONE")
print("="*70)
