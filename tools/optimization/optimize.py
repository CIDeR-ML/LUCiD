import json
import math
import numpy as np
import jax
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
        print(f"Position+t0 grid search parameters (t0-loop approach):")
        print(f"  Spatial divisions: {config['position_grid_search']['pos_n_div']}")
        print(f"  t0 divisions: {config['position_grid_search']['t0_n_div']}")
        print(f"  Refinement levels (per 3D search): {config['position_grid_search']['pos_levels']}")
        print(f"  Search fraction: {config['position_grid_search']['pos_fraction']}")
        print(f"  Minimum grid spacing: {config['position_grid_search']['pos_min_L']}")
        print(f"  t0 range: [{config['position_grid_search']['t0_min']}, {config['position_grid_search']['t0_max']}]")
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
        jnp.array(detector_config['qe'])
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


def evaluate_loss_batch(positions, hit_detector_positions, observed_times, observed_charge, t0_fixed):
    """
    Vectorized batch evaluation of origin_time_loss for multiple positions.

    Args:
        positions: Array of positions to evaluate (N, 3)
        hit_detector_positions: Detector positions with hits
        observed_times: Timing data
        observed_charge: Charge data
        t0_fixed: Fixed t0 value

    Returns:
        Array of losses (N,)
    """
    # Convert to JAX array if needed
    positions_jax = jnp.asarray(positions)

    # Create vectorized version of origin_time_loss over the first argument (origin)
    # All other arguments are fixed
    vectorized_loss = jax.vmap(
        lambda pos: origin_time_loss(pos, hit_detector_positions, observed_times, observed_charge, t0_fixed)
    )

    return vectorized_loss(positions_jax)


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

    # Vectorized grid generation using meshgrid
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    all_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # Vectorized detector bounds filtering
    if detector_bounds['type'] == 'cylinder':
        r = detector_bounds['r'] * fraction
        H = detector_bounds['H'] * fraction
        rho_sq = all_points[:, 0]**2 + all_points[:, 1]**2
        mask = (rho_sq <= r**2) & (np.abs(all_points[:, 2]) <= H / 2)
    elif detector_bounds['type'] == 'sphere':
        r = detector_bounds['r'] * fraction
        dist_sq = np.sum(all_points**2, axis=1)
        mask = dist_sq <= r**2
    elif detector_bounds['type'] == 'box':
        x_lim = detector_bounds['x'] * fraction / 2
        y_lim = detector_bounds['y'] * fraction / 2
        z_lim = detector_bounds['z'] * fraction / 2
        mask = ((np.abs(all_points[:, 0]) <= x_lim) &
                (np.abs(all_points[:, 1]) <= y_lim) &
                (np.abs(all_points[:, 2]) <= z_lim))
    else:
        mask = np.ones(len(all_points), dtype=bool)

    pts = all_points[mask]
    return pts


def generate_grid_points_4d_local(center_xyz, t0_center, local_size, t0_range, L, dt0,
                                   detector_bounds, fraction=1.0):
    """
    Generate 4D cubic grid points (x, y, z, t0) within a local region, filtered by detector bounds.

    Args:
        center_xyz: Center of spatial search region (x, y, z)
        t0_center: Center of t0 search
        local_size: Half-size of local cubic spatial search region
        t0_range: Half-range of t0 search
        L: Spatial grid spacing
        dt0: t0 grid spacing
        detector_bounds: Detector geometry bounds from get_detector_bounds()
        fraction: Fraction of detector dimensions to use (0 < fraction <= 1)

    Returns:
        np.ndarray: Array of 4D grid points (N, 4) where each row is [x, y, z, t0]
    """
    cx, cy, cz = center_xyz

    # Spatial bounds
    x_min, x_max = cx - local_size, cx + local_size
    y_min, y_max = cy - local_size, cy + local_size
    z_min, z_max = cz - local_size, cz + local_size

    # t0 bounds
    t0_min, t0_max = t0_center - t0_range, t0_center + t0_range

    xs = np.arange(x_min, x_max + L, L)
    ys = np.arange(y_min, y_max + L, L)
    zs = np.arange(z_min, z_max + L, L)
    t0s = np.arange(t0_min, t0_max + dt0, dt0)

    pts = []
    for x in xs:
        for y in ys:
            for z in zs:
                if is_point_inside_detector((x, y, z), detector_bounds, fraction):
                    for t0 in t0s:
                        pts.append((x, y, z, t0))

    return np.array(pts)


def hierarchical_position_grid_search_4d(hit_detector_positions, observed_times, observed_charge,
                                     true_position, true_t0, t0_guess, detector_bounds,
                                     n_div=5, t0_n_div=5, levels=6, fraction=1.0,
                                     t0_min=-15.0, t0_max=15.0, t0_min_range=0.1,
                                     min_L=0.01, verbosity=2):
    """
    Unified hierarchical position and t0 search for any detector geometry using origin_time_loss.
    Uses a regular 4D cubic grid (x, y, z, t0) and filters spatial points based on detector bounds.

    Args:
        hit_detector_positions: Positions of detectors with hits
        observed_times: Timing data
        observed_charge: Charge data
        true_position: True position for comparison
        true_t0: True t0 for comparison (for error tracking only)
        t0_guess: Initial t0 guess (center of initial t0 search range)
        detector_bounds: Detector geometry bounds from get_detector_bounds()
        n_div: Number of divisions per spatial dimension in the grid
        t0_n_div: Number of divisions for t0 dimension
        levels: Number of refinement levels
        fraction: Fraction of the detector where we look for the vertex position
        t0_min: Minimum t0 value for initial search
        t0_max: Maximum t0 value for initial search
        t0_min_range: Minimum t0 range (stops refinement when t0_range < t0_min_range)
        min_L: Minimum spatial grid spacing
        verbosity: Verbosity level (0, 1, 2)

    Returns:
        dict: Search results with best position, best t0, and statistics
    """
    if verbosity >= 2:
        print(f"    Performing hierarchical 4D position+t0 grid search...")
        print(f"    Parameters: n_div={n_div}, t0_n_div={t0_n_div}, levels={levels}, fraction={fraction}")
        print(f"    t0 range: [{t0_min}, {t0_max}]")
        print(f"    Detector bounds: {detector_bounds}")

    center_xyz = (0.0, 0.0, 0.0)  # Start at detector center
    t0_center = t0_guess  # Start t0 search centered on initial guess

    # Initial local search size based on detector type
    if detector_bounds['type'] == 'cylinder':
        local_size = max(detector_bounds['r'], detector_bounds['H']/2) * fraction
    elif detector_bounds['type'] == 'sphere':
        local_size = detector_bounds['r'] * fraction
    elif detector_bounds['type'] == 'box':
        local_size = max(detector_bounds['x'], detector_bounds['y'], detector_bounds['z']) * fraction / 2

    # Initial t0 search range
    t0_range = max(abs(t0_max - t0_center), abs(t0_min - t0_center))

    all_results = []
    best_overall_loss = float('inf')
    best_overall_position = None
    best_overall_t0 = None

    for level in range(levels):
        L = (2 * local_size) / n_div
        dt0 = (2 * t0_range) / t0_n_div

        # Check stopping criteria
        if L < min_L and t0_range < t0_min_range:
            if verbosity >= 2:
                print(f"      Level {level}: Both L={L:.4f} < min_L={min_L} and t0_range={t0_range:.4f} < t0_min_range={t0_min_range}, stopping")
            break

        pts_4d = generate_grid_points_4d_local(center_xyz, t0_center, local_size, t0_range,
                                                L, dt0, detector_bounds, fraction)
        if pts_4d.size == 0:
            if verbosity >= 2:
                print(f"      Level {level}: No valid grid points, stopping")
            break

        if verbosity >= 2:
            print(f"      Level {level}: L={L:.4f}, local_size={local_size:.4f}, dt0={dt0:.4f}, t0_range={t0_range:.4f}, num_points={len(pts_4d)}")

        # Evaluate origin_time_loss at each 4D grid point
        level_results = []
        best_level_loss = float('inf')
        best_level_position = None
        best_level_t0 = None

        for i, point_4d in enumerate(pts_4d):
            position = jnp.array(point_4d[:3])
            t0 = point_4d[3]

            try:
                # Evaluate origin_time_loss with this position and t0
                loss = origin_time_loss(position, hit_detector_positions,
                                      observed_times, observed_charge, t0)

                level_results.append({
                    'position': np.array(position),
                    't0': float(t0),
                    'loss': float(loss),
                    'distance_to_true': float(jnp.linalg.norm(position - true_position)),
                    't0_error': float(abs(t0 - true_t0))
                })

                # Track best for this level
                if loss < best_level_loss:
                    best_level_loss = loss
                    best_level_position = position
                    best_level_t0 = t0

                # Track best overall
                if loss < best_overall_loss:
                    best_overall_loss = loss
                    best_overall_position = position
                    best_overall_t0 = t0

            except Exception as e:
                if verbosity >= 2:
                    print(f"        Error evaluating point {i}: {e}")
                continue

        level_summary = {
            'level': level,
            'L': L,
            'dt0': dt0,
            'center_xyz': center_xyz,
            't0_center': t0_center,
            'local_size': local_size,
            't0_range': t0_range,
            'num_points': len(pts_4d),
            'grid_points_4d': pts_4d,
            'point_results': level_results,
            'best_position': np.array(best_level_position) if best_level_position is not None else None,
            'best_t0': float(best_level_t0) if best_level_t0 is not None else None,
            'best_loss': best_level_loss
        }

        all_results.append(level_summary)

        if verbosity >= 2:
            print(f"      Level {level} best loss: {best_level_loss:.6f}")
            if best_level_position is not None:
                print(f"      Level {level} best position: {best_level_position}, best t0: {best_level_t0:.4f}")

        # Prepare for next level refinement
        if best_level_position is not None and best_level_t0 is not None:
            center_xyz = (float(best_level_position[0]),
                         float(best_level_position[1]),
                         float(best_level_position[2]))
            t0_center = float(best_level_t0)
            local_size = L / 2  # Shrink spatial search region for next level
            t0_range = dt0 / 2  # Shrink t0 search range for next level
        else:
            break

    # Calculate final statistics
    final_position_error = float(jnp.linalg.norm(best_overall_position - true_position)) if best_overall_position is not None else float('inf')
    final_t0_error = float(abs(best_overall_t0 - true_t0)) if best_overall_t0 is not None else float('inf')

    if verbosity >= 2:
        print(f"    4D grid search complete. Best overall loss: {best_overall_loss:.6f}")
        print(f"    Best position: {best_overall_position}")
        print(f"    Best t0: {best_overall_t0:.4f}")
        print(f"    Position error: {final_position_error:.3f}m, t0 error: {final_t0_error:.4f}")

    return {
        'all_levels': all_results,
        'best_position': np.array(best_overall_position) if best_overall_position is not None else None,
        'best_t0': float(best_overall_t0) if best_overall_t0 is not None else None,
        'best_loss': best_overall_loss,
        'position_error': final_position_error,
        't0_error': final_t0_error,
        'len_all_levels': len(all_results)
    }


def hierarchical_position_grid_search_3d(hit_detector_positions, observed_times, observed_charge,
                                         true_position, t0_fixed, detector_bounds,
                                         n_div=5, levels=6, fraction=1.0, min_L=0.01, verbosity=2):
    """
    Hierarchical 3D position search for a FIXED t0 value using origin_time_loss.
    Uses a regular 3D cubic grid and filters points based on detector bounds.

    Args:
        hit_detector_positions: Positions of detectors with hits
        observed_times: Timing data
        observed_charge: Charge data
        true_position: True position for comparison
        t0_fixed: Fixed t0 value to use for this search
        detector_bounds: Detector geometry bounds from get_detector_bounds()
        n_div: Number of divisions per dimension in the grid
        levels: Number of refinement levels
        fraction: Fraction of the detector where we look for the vertex position
        min_L: Minimum grid spacing
        verbosity: Verbosity level (0, 1, 2)

    Returns:
        dict: Search results with best position and statistics
    """
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
            break

        pts = generate_grid_points_local(center, local_size, L, detector_bounds, fraction)
        if pts.size == 0:
            break

        # Vectorized batch evaluation of all grid points
        losses = evaluate_loss_batch(pts, hit_detector_positions,
                                     observed_times, observed_charge, t0_fixed)

        # Find best position in this level
        best_idx = jnp.argmin(losses)
        best_level_loss = float(losses[best_idx])
        best_level_position = jnp.array(pts[best_idx])

        # Track best overall
        if best_level_loss < best_overall_loss:
            best_overall_loss = best_level_loss
            best_overall_position = best_level_position

        level_summary = {
            'level': level,
            'L': L,
            'center': center,
            'local_size': local_size,
            'num_points': len(pts),
            'best_position': np.array(best_level_position) if best_level_position is not None else None,
            'best_loss': best_level_loss
        }

        all_results.append(level_summary)

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

    return {
        'all_levels': all_results,
        'best_position': np.array(best_overall_position) if best_overall_position is not None else None,
        'best_loss': best_overall_loss,
        'position_error': final_position_error,
        'len_all_levels': len(all_results)
    }


def hierarchical_position_grid_search(hit_detector_positions, observed_times, observed_charge,
                                     true_position, true_t0, t0_guess, detector_bounds,
                                     n_div=5, t0_n_div=5, levels=6, fraction=1.0,
                                     t0_min=-15.0, t0_max=15.0,
                                     min_L=0.01, verbosity=2):
    """
    Hierarchical position and t0 search using multiple 3D searches.
    For each t0 value in a grid, runs a complete 3D hierarchical position search.
    Returns the (position, t0) combination with the lowest loss.

    Args:
        hit_detector_positions: Positions of detectors with hits
        observed_times: Timing data
        observed_charge: Charge data
        true_position: True position for comparison
        true_t0: True t0 for comparison (for error tracking only)
        t0_guess: Initial t0 guess (used as center for t0 grid)
        detector_bounds: Detector geometry bounds from get_detector_bounds()
        n_div: Number of divisions per spatial dimension in the grid
        t0_n_div: Number of t0 values to test
        levels: Number of refinement levels for each 3D search
        fraction: Fraction of the detector where we look for the vertex position
        t0_min: Minimum t0 value
        t0_max: Maximum t0 value
        min_L: Minimum spatial grid spacing
        verbosity: Verbosity level (0, 1, 2)

    Returns:
        dict: Search results with best position, best t0, and statistics
    """
    if verbosity >= 2:
        print(f"    Performing hierarchical position+t0 grid search (t0-loop approach)...")
        print(f"    Parameters: n_div={n_div}, t0_n_div={t0_n_div}, levels={levels}, fraction={fraction}")
        print(f"    t0 range: [{t0_min}, {t0_max}]")
        print(f"    Detector bounds: {detector_bounds}")

    # Generate t0 grid
    t0_values = np.linspace(t0_min, t0_max, t0_n_div)

    if verbosity >= 3:
        print(f"    Testing {len(t0_values)} t0 values: {t0_values}")

    all_t0_results = []
    best_overall_loss = float('inf')
    best_overall_position = None
    best_overall_t0 = None

    # Loop over each t0 value
    for i, t0_val in enumerate(t0_values):
        if verbosity >= 3:
            print(f"      t0 [{i+1}/{len(t0_values)}] = {t0_val:.4f}")

        # Run complete 3D hierarchical search for this fixed t0
        search_3d_results = hierarchical_position_grid_search_3d(
            hit_detector_positions, observed_times, observed_charge,
            true_position, t0_val, detector_bounds,
            n_div=n_div, levels=levels, fraction=fraction, min_L=min_L, verbosity=0
        )

        t0_result = {
            't0_value': float(t0_val),
            't0_error': float(abs(t0_val - true_t0)),
            'best_position': search_3d_results['best_position'],
            'best_loss': search_3d_results['best_loss'],
            'position_error': search_3d_results['position_error'],
            'search_3d_results': search_3d_results
        }

        all_t0_results.append(t0_result)

        if verbosity >= 3:
            print(f"        → loss: {search_3d_results['best_loss']:.6f}, pos_err: {search_3d_results['position_error']:.3f}m")

        # Track best overall
        if search_3d_results['best_loss'] < best_overall_loss:
            best_overall_loss = search_3d_results['best_loss']
            best_overall_position = search_3d_results['best_position']
            best_overall_t0 = t0_val

    # Calculate final statistics
    final_position_error = float(jnp.linalg.norm(best_overall_position - true_position)) if best_overall_position is not None else float('inf')
    final_t0_error = float(abs(best_overall_t0 - true_t0)) if best_overall_t0 is not None else float('inf')

    if verbosity >= 2:
        print(f"    Grid search complete. Best overall loss: {best_overall_loss:.6f}")
        print(f"    Best position: {best_overall_position}")
        print(f"    Best t0: {best_overall_t0:.4f} (true: {true_t0:.4f})")
        print(f"    Position error: {final_position_error:.3f}m, t0 error: {final_t0_error:.4f}")

    return {
        'all_t0_results': all_t0_results,
        'best_position': np.array(best_overall_position) if best_overall_position is not None else None,
        'best_t0': float(best_overall_t0) if best_overall_t0 is not None else None,
        'best_loss': best_overall_loss,
        'position_error': final_position_error,
        't0_error': final_t0_error,
        'n_t0_tested': len(t0_values)
    }