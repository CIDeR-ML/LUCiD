#!/usr/bin/env python3
"""
Visualize sensor hits for a single event from a v3 four-file dataset.

Shows sensor hits colored by charge value, with arrows showing track
directions and optional track segment polylines. Reads the four v3 files
(sensor/inst/seg/labl) for one batch and renders one event as HTML.

Usage:
    python visualize_particle_events.py <dataset_root> <detector_config> \\
        --event 0 [--file-index 0]

``<dataset_root>`` must contain ``sensor/``, ``inst/``, ``seg/``, ``labl/``
subdirectories with ``wc_*_{file_index:04d}.h5`` files.
"""
import os
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from lucid.sources.event_io import (
    read_sensor_event_v3,
    read_inst_event_v3,
    read_seg_event_v3,
    read_labl_event_v3,
)
from lucid.geometry import generate_detector
from lucid.geometry.utils import calculate_surface_normals, create_disc_mesh
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize sensor hits from a v3 four-file dataset')
parser.add_argument('dataset_root', type=str,
                    help='Dataset root directory containing sensor/, inst/, seg/, labl/ subdirs.')
parser.add_argument('detector_config', type=str, help='Detector configuration JSON file')
parser.add_argument('--event', type=int, default=0,
                    help='Event sequence index within the batch file (default: 0).')
parser.add_argument('--file-index', type=int, default=0,
                    help='Batch file index NNNN in wc_*_NNNN.h5 (default: 0).')
parser.add_argument('--min-charge', type=float, default=1.0,
                    help='Minimum charge threshold in PE (default: 1.0)')
parser.add_argument('--output-dir', type=str, default=None,
                    help='Output directory for HTML file (default: current directory)')
args = parser.parse_args()

dataset_root = Path(args.dataset_root)
detector_config = args.detector_config
event_idx = args.event
file_index = args.file_index
min_charge = args.min_charge
output_dir = args.output_dir

sensor_file = dataset_root / 'sensor' / f'wc_sensor_{file_index:04d}.h5'
inst_file = dataset_root / 'inst' / f'wc_inst_{file_index:04d}.h5'
seg_file = dataset_root / 'seg' / f'wc_seg_{file_index:04d}.h5'
labl_file = dataset_root / 'labl' / f'wc_labl_{file_index:04d}.h5'

for p in (sensor_file, inst_file, seg_file, labl_file):
    if not p.exists():
        raise FileNotFoundError(f"v3 batch file missing: {p}")

print("="*70)
print(f"SENSOR VISUALIZATION BY PARTICLE")
print("="*70)
print(f"Dataset root: {dataset_root}")
print(f"Detector config: {detector_config}")
print(f"Event: {event_idx} (file index {file_index})")
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


print(f"Loading event {event_idx} from v3 batch file_index={file_index}...")
print()

# Category names mapping (255 = Unknown, sentinel for uncategorized)
category_names_map = {0: "Primary", 1: "DecayElectron", 2: "SecondaryPion",
                      3: "Gamma", 255: "Unknown", -1: "Unknown"}

def get_category_name(code):
    return category_names_map.get(int(code), f'Category_{int(code)}')

# Read the four v3 files
sensor_data = read_sensor_event_v3(str(sensor_file), event_idx)
inst_data = read_inst_event_v3(str(inst_file), event_idx)
seg_data = read_seg_event_v3(str(seg_file), event_idx)
labl_data = read_labl_event_v3(str(labl_file), event_idx)

n_sensors = detector.n_sensors
n_particles = int(labl_data['n_particles'])
n_tracks = int(labl_data['n_tracks'])
t0 = float(labl_data['per_interaction']['t0'][0])
overall_light_containment = float(labl_data['per_event']['overall_containment'])

# Reconstruct dense PE_per_particle / T_per_particle from inst sparse rows
PE_per_particle = np.zeros((n_particles, n_sensors), dtype=np.float32)
T_per_particle = np.full((n_particles, n_sensors), np.inf, dtype=np.float32)
if int(inst_data.get('n_particle_hits', 0)) > 0:
    pi_arr = np.asarray(inst_data['particle_idx'], dtype=np.int32)
    si_arr = np.asarray(inst_data['sensor_idx'], dtype=np.int32)
    pe_arr = np.asarray(inst_data['PE'], dtype=np.float32)
    t_arr = np.asarray(inst_data['T'], dtype=np.float32)
    PE_per_particle[pi_arr, si_arr] = pe_arr
    T_per_particle[pi_arr, si_arr] = np.where(t_arr > 0, t_arr, np.inf)

# Reconstruct dense per-sensor PE from sensor sparse
PE = np.zeros(n_sensors, dtype=np.float32)
if int(sensor_data.get('n_hits', 0)) > 0:
    PE[np.asarray(sensor_data['sensor_idx'], dtype=np.int32)] = np.asarray(
        sensor_data['PE'], dtype=np.float32)

# Decompose labl/per_particle CSR genealogy arrays
pp = labl_data['per_particle']

def _decompose_csr(data, offsets, count):
    data = np.asarray(data)
    offsets = np.asarray(offsets)
    return [np.asarray(data[offsets[i]:offsets[i + 1]]) for i in range(count)]

particle_categorized_genealogy = _decompose_csr(
    pp['genealogy_data'], pp['genealogy_offsets'], n_particles)
particle_track_genealogy = _decompose_csr(
    pp['ext_genealogy_data'], pp['ext_genealogy_offsets'], n_particles)
particle_category = np.asarray(pp['category'], dtype=np.int32)
light_containment_by_particle = np.asarray(pp['containment'], dtype=np.float32)

# Build the dict that the rendering code below indexes into
lucid_data = {
    'n_particles': n_particles,
    'PE_per_particle': PE_per_particle,
    'T_per_particle': T_per_particle,
    'PE': PE,
    't0': t0,
    'Particle_Category': particle_category,
    'light_containment_by_particle': light_containment_by_particle,
    'Particle_TrackGenealogy': particle_track_genealogy,
    'Particle_CategorizedGenealogy': particle_categorized_genealogy,
    'overall_light_containment': overall_light_containment,
}

# Build track_id_to_info using labl/per_track + seg first-segment per track
track_id_to_info = {}
pt = labl_data['per_track']
if n_tracks > 0:
    track_ids_labl = np.asarray(pt['track_id'], dtype=np.int64)
    track_pdgs_labl = np.asarray(pt['pdg'], dtype=np.int32)
    track_energies_labl = np.asarray(pt['initial_energy'], dtype=np.float32)

    seg_track_idx = np.asarray(seg_data['track_idx'], dtype=np.int32)
    first_seg_for_track = np.full(n_tracks, -1, dtype=np.int32)
    for seg_row in range(seg_track_idx.size):
        ti = int(seg_track_idx[seg_row])
        if 0 <= ti < n_tracks and first_seg_for_track[ti] == -1:
            first_seg_for_track[ti] = seg_row

    seg_start_x_m = np.asarray(seg_data['start_x'], dtype=np.float32)
    seg_start_y_m = np.asarray(seg_data['start_y'], dtype=np.float32)
    seg_start_z_m = np.asarray(seg_data['start_z'], dtype=np.float32)
    seg_dx = np.asarray(seg_data['dir_x'], dtype=np.float32)
    seg_dy = np.asarray(seg_data['dir_y'], dtype=np.float32)
    seg_dz = np.asarray(seg_data['dir_z'], dtype=np.float32)

    for k in range(n_tracks):
        fs = int(first_seg_for_track[k])
        if fs >= 0:
            pos = np.array([seg_start_x_m[fs], seg_start_y_m[fs], seg_start_z_m[fs]])
            dir_vec = np.array([seg_dx[fs], seg_dy[fs], seg_dz[fs]], dtype=np.float32)
            n = float(np.linalg.norm(dir_vec))
            if n > 0:
                dir_vec = dir_vec / n
        else:
            pos = np.array([0.0, 0.0, 0.0])
            dir_vec = np.array([0.0, 0.0, 1.0])
        track_id_to_info[int(track_ids_labl[k])] = {
            'pdg': int(track_pdgs_labl[k]),
            'position': pos,
            'direction': dir_vec,
            'energy': float(track_energies_labl[k]),
        }

print(f"Event: {n_particles} categorized particles, {n_tracks} meaningful tracks")
print(f"t0: {t0:.2f} ns")
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

# Alias for rest of visualization code
track_tree = particle_tree
PE_per_part = PE_per_particle
T_per_part = T_per_particle
PE_total = PE


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

# Create "All" trace showing total charge across all particles
all_hit_mask = PE_total >= min_charge
all_hit_indices = np.where(all_hit_mask)[0]

if len(all_hit_indices) > 0:
    all_charges = PE_total[all_hit_indices]
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

# Create "By Particle" trace with discrete colors per particle
# Resolve overlaps by assigning each sensor to the particle with max charge contribution
if len(all_hit_indices) > 0:
    by_particle_vertices = []
    by_particle_faces = []
    by_particle_colors = []
    vertex_offset = 0

    # Sort sensors by position for consistent visualization
    all_positions_sorted = all_positions
    all_normals_sorted = all_normals
    all_hit_indices_sorted = all_hit_indices

    for idx, sensor_idx in enumerate(all_hit_indices_sorted):
        # Find which particle contributed most charge to this sensor
        particle_charges = PE_per_part[:, sensor_idx]
        if np.max(particle_charges) > 0:
            max_particle_idx = np.argmax(particle_charges)
            color = colors_palette[max_particle_idx % len(colors_palette)]
        else:
            color = 'gray'

        pos = all_positions_sorted[idx]
        normal = all_normals_sorted[idx]

        vertices, faces = create_disc_mesh(pos, normal, disc_radius, n_segments=12)
        faces_adjusted = faces + vertex_offset
        by_particle_vertices.append(vertices)
        by_particle_faces.append(faces_adjusted)
        by_particle_colors.extend([color] * len(vertices))
        vertex_offset += len(vertices)
else:
    by_particle_vertices = []

# Add "By Particle" trace
if len(by_particle_vertices) > 0:
    combined_vertices_by_particle = np.vstack(by_particle_vertices)
    combined_faces_by_particle = np.vstack(by_particle_faces)

    fig.add_trace(
        go.Mesh3d(
            x=combined_vertices_by_particle[:, 0],
            y=combined_vertices_by_particle[:, 1],
            z=combined_vertices_by_particle[:, 2],
            i=combined_faces_by_particle[:, 0],
            j=combined_faces_by_particle[:, 1],
            k=combined_faces_by_particle[:, 2],
            vertexcolor=by_particle_colors,
            name='By Particle',
            showscale=False,
            visible=False,
            lighting=dict(ambient=0.8, diffuse=0.5, specular=0.1),
            flatshading=True
        )
    )
    print(f"  By Particle: {len(all_hit_indices)} sensors (color-coded)")

# Create sensor meshes for each individual particle
for particle_idx in range(n_particles):
    charges = PE_per_part[particle_idx, :]
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
            name=f'Particle {particle_idx}'
        ))
        print(f"  Warning: Particle {particle_idx} has no sensors above {min_charge} PE")
        continue

    particle_charges = charges[hit_indices]
    positions = detector.all_points[hit_indices]
    normals = calculate_surface_normals(detector, hit_indices)

    particle_vertices = []
    particle_faces = []
    particle_intensities = []
    vertex_offset = 0

    for pos, normal, charge in zip(positions, normals, particle_charges):
        vertices, faces = create_disc_mesh(pos, normal, disc_radius, n_segments=12)
        faces_adjusted = faces + vertex_offset
        particle_vertices.append(vertices)
        particle_faces.append(faces_adjusted)
        particle_intensities.extend([charge] * len(vertices))
        vertex_offset += len(vertices)

    combined_vertices = np.vstack(particle_vertices)
    combined_faces = np.vstack(particle_faces)

    fig.add_trace(
        go.Mesh3d(
            x=combined_vertices[:, 0],
            y=combined_vertices[:, 1],
            z=combined_vertices[:, 2],
            i=combined_faces[:, 0],
            j=combined_faces[:, 1],
            k=combined_faces[:, 2],
            intensity=particle_intensities,
            colorscale='Viridis',
            cmin=0,
            cmax=global_max_charge,
            colorbar=dict(title="Charge (PE)", x=0.92, len=0.8),
            name=f'Particle {particle_idx}',
            showscale=True,
            visible=False,
            lighting=dict(ambient=0.8, diffuse=0.5, specular=0.1),
            flatshading=True
        )
    )
    print(f"  Particle {particle_idx}: {len(hit_indices)} sensors above {min_charge} PE")

print()

# ============================================================================
# TRACK SEGMENT VISUALIZATION (from v3 seg + labl/per_track)
# ============================================================================
print("Loading track segment data from v3 seg file...")

has_segment_data = n_tracks > 0
segment_trace_indices = []

if has_segment_data:
    track_ids = track_ids_labl
    track_parent_ids = np.asarray(pt['parent_id'], dtype=np.int32)
    track_pdgs = track_pdgs_labl
    track_n_cherenkov = np.asarray(pt['n_cherenkov'], dtype=np.int32)
    track_names = [pdg_to_name.get(int(pdg), f'PDG{int(pdg)}') for pdg in track_pdgs]

    # Derive per-track segment offsets + counts from the track_idx FK column.
    # Segments are written ordered by track (writer guarantee), so offsets are
    # the first occurrence of each track_idx; counts are bincount.
    track_n_segs = np.bincount(seg_track_idx, minlength=n_tracks).astype(np.int32)
    track_seg_offsets = np.zeros(n_tracks, dtype=np.int32)
    if n_tracks > 1:
        track_seg_offsets[1:] = np.cumsum(track_n_segs[:-1])

    # Segment geometry in meters (seg file stores meters)
    seg_start_x = np.asarray(seg_data['start_x'], dtype=np.float32)
    seg_start_y = np.asarray(seg_data['start_y'], dtype=np.float32)
    seg_start_z = np.asarray(seg_data['start_z'], dtype=np.float32)
    seg_end_x = np.asarray(seg_data['end_x'], dtype=np.float32)
    seg_end_y = np.asarray(seg_data['end_y'], dtype=np.float32)
    seg_end_z = np.asarray(seg_data['end_z'], dtype=np.float32)

    print(f"  Found {n_tracks} meaningful tracks with {seg_start_x.size} segments")

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
    print("  No track segment data available")

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
# Trace order: Arrows..., "All", "By Particle", Individual particles..., segments..., detector surface, detector points
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
root_basename = dataset_root.name if dataset_root.name else 'dataset'
filename = f'particle_sensors_{root_basename}_file{file_index:04d}_event{event_idx:03d}.html'

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
