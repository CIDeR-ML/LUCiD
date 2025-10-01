import json
import math
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from tools.optimization.losses import origin_time_loss


def load_optimization_config(config_path):
    """
    Load optimization configuration from JSON file.
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def print_optimization_parameters(config, detector_r, detector_h, num_detectors, verbosity=None):
    """
    Print all optimization parameters in a formatted way.
    
    Args:
        config: Configuration dictionary loaded from JSON
        detector_r: Detector radius (from detector object)
        detector_h: Detector height (from detector object)
        num_detectors: Number of detectors (from detector object)
        verbosity: Verbosity level (0, 1, 2). If None, uses config value.
    """
    if verbosity is None:
        verbosity = config.get('verbosity', {}).get('level', 2)
    
    if verbosity >= 2:
        print(f"Number of sensors: {num_detectors}")
        print(f"Detector dimensions: R={detector_r:.1f}m, H={detector_h:.1f}m")
        print(f"Fixed temperature: {config['basic_config']['temperature']}")
        print(f"Speed of light in medium: {config['basic_config']['c_medium']}")
        print(f"Number of events for analysis: {config['basic_config']['n_events']}")
        print(f"Position grid search parameters:")
        print(f"  Number of divisions: {config['position_grid_search']['pos_n_div']}")
        print(f"  Refinement levels: {config['position_grid_search']['pos_levels']}")
        print(f"  Search fraction: {config['position_grid_search']['pos_fraction']}")
        print(f"  Minimum grid spacing: {config['position_grid_search']['pos_min_L']}")
        print(f"Hierarchical cone direction search parameters:")
        print(f"  Levels: {config['cone_direction_search']['cone_levels']}")
        print(f"  Initial divisions: {config['cone_direction_search']['cone_initial_div']}")
        print(f"  Max cone angle: {config['cone_direction_search']['cone_max_angle_deg']}°")
        print(f"  Cone reduction factor: {config['cone_direction_search']['cone_reduction']}")
        print(f"Energy optimization parameters:")
        print(f"  Energy delta: ±{config['energy_optimization']['energy_delta']}")
        print(f"  Energy scan steps: {config['energy_optimization']['energy_scan_steps']}")


def get_detector_params_from_config(config):
    """
    Convert detector parameters from config to JAX array format.

    Args:
        config: Configuration dictionary

    Returns:
        tuple: JAX arrays for detector parameters
    """
    detector_config = config['detector_params']
    return (
        jnp.array(detector_config['scatter_length']),
        jnp.array(detector_config['reflection_rate']),
        jnp.array(detector_config['absorption_length']),
        jnp.array(detector_config['gumbel_softmax_temp'])
    )


def get_detector_bounds(detector):
    """
    Extract detector bounds based on detector type.

    Args:
        detector: Detector object (Cylinder, Sphere, or Box)

    Returns:
        dict: Dictionary with detector type and bounds
    """
    detector_type = detector.__class__.__name__

    if 'Cylinder' in detector_type:
        return {
            'type': 'cylinder',
            'r': detector.r,
            'H': detector.H
        }
    elif 'Sphere' in detector_type:
        return {
            'type': 'sphere',
            'r': detector.r
        }
    elif 'Box' in detector_type:
        return {
            'type': 'box',
            'x': detector.L,
            'y': detector.W,
            'z': detector.H
        }
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


def is_point_inside_detector(point, detector_bounds, fraction=1.0):
    """
    Check if a point is inside the detector bounds with given fraction.

    Args:
        point: (x, y, z) coordinates
        detector_bounds: Dictionary from get_detector_bounds()
        fraction: Fraction of detector dimensions to use (0 < fraction <= 1)

    Returns:
        bool: True if point is inside detector bounds
    """
    x, y, z = point

    if detector_bounds['type'] == 'cylinder':
        r_max = detector_bounds['r'] * fraction
        h_max = detector_bounds['H'] * fraction / 2
        r_point = np.sqrt(x**2 + y**2)
        return r_point <= r_max and abs(z) <= h_max

    elif detector_bounds['type'] == 'sphere':
        r_max = detector_bounds['r'] * fraction
        r_point = np.sqrt(x**2 + y**2 + z**2)
        return r_point <= r_max

    elif detector_bounds['type'] == 'box':
        hx = detector_bounds['x'] * fraction / 2
        hy = detector_bounds['y'] * fraction / 2
        hz = detector_bounds['z'] * fraction / 2
        return abs(x) <= hx and abs(y) <= hy and abs(z) <= hz

    return False


def generate_grid_points_local(center, local_size, L, detector_bounds, fraction=1.0):
    """
    Generate 3D cubic grid points within a local region, filtered by detector bounds.

    Args:
        center: Center of local search region (x, y, z)
        local_size: Half-size of local cubic search region
        L: Grid spacing
        detector_bounds: Detector geometry bounds from get_detector_bounds()
        fraction: Fraction of detector dimensions to use (0 < fraction <= 1)

    Returns:
        np.ndarray: Array of grid points (N, 3)
    """
    cx, cy, cz = center

    # Cubic grid spacing
    dx = dy = dz = L

    # Bounds of local search region
    x_min, x_max = cx - local_size, cx + local_size
    y_min, y_max = cy - local_size, cy + local_size
    z_min, z_max = cz - local_size, cz + local_size

    xs = np.arange(x_min, x_max + dx, dx)
    ys = np.arange(y_min, y_max + dy, dy)
    zs = np.arange(z_min, z_max + dz, dz)

    pts = []
    for x in xs:
        for y in ys:
            for z in zs:
                if is_point_inside_detector((x, y, z), detector_bounds, fraction):
                    pts.append((x, y, z))

    return np.array(pts)


def hierarchical_position_grid_search(hit_detector_positions, observed_times, observed_charge,
                                     true_position, true_t0, detector_bounds,
                                     n_div=5, levels=6, fraction=1.0, min_L=0.01, verbosity=2):
    """
    Unified hierarchical position search for any detector geometry using origin_time_loss.
    Uses a regular 3D cubic grid and filters points based on detector bounds.

    Args:
        hit_detector_positions: Positions of detectors with hits
        observed_times: Timing data
        observed_charge: Charge data
        true_position: True position for comparison
        true_t0: True t0 for loss evaluation
        detector_bounds: Detector geometry bounds from get_detector_bounds()
        n_div: Number of divisions per dimension in the grid
        levels: Number of refinement levels
        fraction: Fraction of the detector where we look for the vertex position
        min_L: Minimum grid spacing
        verbosity: Verbosity level (0, 1, 2)

    Returns:
        dict: Search results with best position and statistics
    """
    if verbosity >= 2:
        print(f"    Performing hierarchical position grid search...")
        print(f"    Parameters: n_div={n_div}, levels={levels}, fraction={fraction}")
        print(f"    Detector bounds: {detector_bounds}")

    center = (0.0, 0.0, 0.0)  # Start at detector center

    # Initial local search size based on detector type
    if detector_bounds['type'] == 'cylinder':
        local_size = max(detector_bounds['r'], detector_bounds['H']/2) * fraction
    elif detector_bounds['type'] == 'sphere':
        local_size = detector_bounds['r'] * fraction
    elif detector_bounds['type'] == 'box':
        local_size = max(detector_bounds['x'], detector_bounds['y'], detector_bounds['z']) * fraction / 2

    all_results = []
    best_overall_loss = float('inf')
    best_overall_position = None

    for level in range(levels):
        L = (2 * local_size) / n_div
        if L < min_L:
            if verbosity >= 2:
                print(f"      Level {level}: L={L:.4f} < min_L={min_L}, stopping")
            break

        pts = generate_grid_points_local(center, local_size, L, detector_bounds, fraction)
        if pts.size == 0:
            if verbosity >= 2:
                print(f"      Level {level}: No valid grid points, stopping")
            break

        if verbosity >= 2:
            print(f"      Level {level}: L={L:.4f}, local_size={local_size:.4f}, num_points={len(pts)}")

        # Evaluate origin_time_loss at each grid point
        level_results = []
        best_level_loss = float('inf')
        best_level_position = None

        for i, point in enumerate(pts):
            position = jnp.array(point)

            try:
                # Evaluate origin_time_loss
                loss = origin_time_loss(position, hit_detector_positions,
                                      observed_times, observed_charge, true_t0)

                level_results.append({
                    'position': np.array(position),
                    'loss': float(loss),
                    'distance_to_true': float(jnp.linalg.norm(position - true_position))
                })

                # Track best for this level
                if loss < best_level_loss:
                    best_level_loss = loss
                    best_level_position = position

                # Track best overall
                if loss < best_overall_loss:
                    best_overall_loss = loss
                    best_overall_position = position

            except Exception as e:
                if verbosity >= 2:
                    print(f"        Error evaluating point {i}: {e}")
                continue

        level_summary = {
            'level': level,
            'L': L,
            'center': center,
            'local_size': local_size,
            'num_points': len(pts),
            'grid_points': pts,
            'point_results': level_results,
            'best_position': np.array(best_level_position) if best_level_position is not None else None,
            'best_loss': best_level_loss
        }

        all_results.append(level_summary)

        if verbosity >= 2:
            print(f"      Level {level} best loss: {best_level_loss:.6f}")
            if best_level_position is not None:
                print(f"      Level {level} best position: {best_level_position}")

        # Prepare for next level refinement
        if best_level_position is not None:
            center = (float(best_level_position[0]),
                     float(best_level_position[1]),
                     float(best_level_position[2]))
            local_size = L / 2  # Shrink search region for next level
        else:
            break

    # Calculate final statistics
    final_position_error = float(jnp.linalg.norm(best_overall_position - true_position)) if best_overall_position is not None else float('inf')

    if verbosity >= 2:
        print(f"    Grid search complete. Best overall loss: {best_overall_loss:.6f}")
        print(f"    Best position: {best_overall_position}")
        print(f"    Position error: {final_position_error:.3f}m")

    return {
        'all_levels': all_results,
        'best_position': np.array(best_overall_position) if best_overall_position is not None else None,
        'best_loss': best_overall_loss,
        'position_error': final_position_error,
        'len_all_levels': len(all_results)
    }