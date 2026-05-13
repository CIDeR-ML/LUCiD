"""
Generate IceCube-86 reference geometry NPZ files.

Produces two variants:
  icecube86_simple.npz   — 78 main strings, 60 DOMs each, uniform 17m spacing
  icecube86_full.npz     — 78 main + 8 DeepCore strings (non-uniform z)

Run: python string/gen_icecube86.py
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

SENSOR_RADIUS = 0.165  # 13" DOM outer radius (m)

# ──────────────────────────────────────────────────────────────────────
# Hex grid generator
# ──────────────────────────────────────────────────────────────────────

def hex_ring_positions(spacing, n_rings):
    """Generate positions for concentric hex rings."""
    positions = [(0.0, 0.0)]
    for ring in range(1, n_rings + 1):
        for side in range(6):
            corner_angle = np.pi / 3 * side
            for step in range(ring):
                angle = corner_angle + np.pi / 3 * (side + 2)
                x = ring * spacing * np.cos(corner_angle) + step * spacing * np.cos(angle)
                y = ring * spacing * np.sin(corner_angle) + step * spacing * np.sin(angle)
                positions.append((x, y))
    return np.array(positions)


# ──────────────────────────────────────────────────────────────────────
# Main string z-table
# ──────────────────────────────────────────────────────────────────────

def main_string_z(n_doms=60, dz=17.0, z_top=-1450.0):
    """Standard IceCube string: 60 DOMs at 17m spacing from z_top downward."""
    return np.array([z_top - k * dz for k in range(n_doms)])


# ──────────────────────────────────────────────────────────────────────
# DeepCore string z-table
# ──────────────────────────────────────────────────────────────────────

def deepcore_string_z():
    """DeepCore string: 50 DOMs at 7m (physics) + 10 DOMs at 10m (veto cap).

    Physics region: z = -2450 to -2107, 7m spacing (50 DOMs)
    Dust gap: -2107 to -1860 (uninstrumented)
    Veto cap: z = -1860 to -1770, 10m spacing (10 DOMs)

    Returns z positions top-to-bottom (descending z order) for consistency.
    """
    physics = np.array([-2450.0 + k * 7.0 for k in range(50)])  # ascending
    veto = np.array([-1860.0 + k * 10.0 for k in range(10)])    # ascending
    all_z = np.concatenate([veto[::-1], physics[::-1]])          # descending (top to bottom)
    return all_z


# ──────────────────────────────────────────────────────────────────────
# Pack into string-telescope NPZ format
# ──────────────────────────────────────────────────────────────────────

def pack_string_npz(string_xy, string_z_tables, sensor_radius,
                    envelope_radius=None, envelope_z_min=None, envelope_z_max=None):
    """Pack string positions into the NPZ format expected by StringTelescope.

    Parameters
    ----------
    string_xy : (N_str, 2)       xy positions of each string
    string_z_tables : list of (n_dom_i,) arrays   z positions per string (descending)
    sensor_radius : float

    Returns
    -------
    dict of arrays suitable for np.savez
    """
    n_str = len(string_xy)
    max_dom = max(len(z) for z in string_z_tables)
    n_dom_per_str = np.array([len(z) for z in string_z_tables], dtype=np.int32)
    total_doms = int(n_dom_per_str.sum())

    dom_xyz = np.full((n_str, max_dom, 3), np.nan)
    for i in range(n_str):
        n = len(string_z_tables[i])
        z_sorted = np.sort(string_z_tables[i])  # ascending z for dom_s_offsets
        for k in range(n):
            dom_xyz[i, k] = [string_xy[i, 0], string_xy[i, 1], z_sorted[k]]

    # Envelope: auto-derive if not specified
    all_xy = string_xy
    if envelope_radius is None:
        envelope_radius = float(np.max(np.linalg.norm(all_xy, axis=1))) + 50.0
    if envelope_z_min is None:
        envelope_z_min = float(min(z.min() for z in string_z_tables)) - 50.0
    if envelope_z_max is None:
        envelope_z_max = float(max(z.max() for z in string_z_tables)) + 50.0

    return {
        'dom_xyz': dom_xyz,
        'n_dom_per_str': n_dom_per_str,
        'sensor_radius': np.float64(sensor_radius),
        'envelope_radius': np.float64(envelope_radius),
        'envelope_z_min': np.float64(envelope_z_min),
        'envelope_z_max': np.float64(envelope_z_max),
        'n_strings': np.int32(n_str),
        'max_dom_per_str': np.int32(max_dom),
        'total_doms': np.int32(total_doms),
    }


# ──────────────────────────────────────────────────────────────────────
# Generate both variants
# ──────────────────────────────────────────────────────────────────────

def generate_simple():
    """78 main strings, uniform 60×17m."""
    all_pos = hex_ring_positions(125.0, 5)  # 91 positions through 5 rings
    main_xy = all_pos[:78]
    z_tables = [main_string_z() for _ in range(78)]
    return pack_string_npz(main_xy, z_tables, SENSOR_RADIUS)


def generate_full():
    """78 main + 8 DeepCore strings."""
    main_pos = hex_ring_positions(125.0, 5)[:78]

    # DeepCore: 8 strings in the center, ~40-70m spacing
    dc_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    dc_radius = 50.0
    dc_xy = np.column_stack([dc_radius * np.cos(dc_angles),
                             dc_radius * np.sin(dc_angles)])

    all_xy = np.vstack([main_pos, dc_xy])

    z_tables = [main_string_z() for _ in range(78)]
    z_tables += [deepcore_string_z() for _ in range(8)]

    return pack_string_npz(all_xy, z_tables, SENSOR_RADIUS)


def write_geom_config(npz_filename, output_json, material="water"):
    """Write a detector config JSON pointing at the NPZ."""
    config = {
        "material": material,
        "detector_type": "string",
        "geometry_definitions": {
            "npz_file_path": npz_filename
        }
    }
    with open(output_json, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  wrote {output_json}")


if __name__ == "__main__":
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")

    # Simple (78 main strings)
    data_simple = generate_simple()
    npz_simple = os.path.join(config_dir, "icecube86_simple.npz")
    np.savez(npz_simple, **data_simple)
    print(f"  wrote {npz_simple}")
    print(f"    strings={data_simple['n_strings']}, total_doms={data_simple['total_doms']}, "
          f"max_dom_per_str={data_simple['max_dom_per_str']}")
    write_geom_config("icecube86_simple.npz",
                      os.path.join(config_dir, "IceCube86_simple_geom_config.json"))

    # Full (78 main + 8 DeepCore)
    data_full = generate_full()
    npz_full = os.path.join(config_dir, "icecube86_full.npz")
    np.savez(npz_full, **data_full)
    print(f"  wrote {npz_full}")
    print(f"    strings={data_full['n_strings']}, total_doms={data_full['total_doms']}, "
          f"max_dom_per_str={data_full['max_dom_per_str']}")
    write_geom_config("icecube86_full.npz",
                      os.path.join(config_dir, "IceCube86_full_geom_config.json"))

    # Physics config (shared)
    physics = {
        "medium_model": "materials/water.json",
        "qe_curve": "pmt/SK_QE.json",
        "wall_reflection_rate": 0.0,
        "sensor_reflection_rate": 0.0,
        "qe_corrections": 1.0
    }
    phys_path = os.path.join(config_dir, "IceCube86_physics_config.json")
    with open(phys_path, 'w') as f:
        json.dump(physics, f, indent=2)
    print(f"  wrote {phys_path}")
