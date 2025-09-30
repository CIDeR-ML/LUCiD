
from jax import jit
import jax.numpy as jnp
import math
import numpy as np
from tools.optimization.losses import *

@jit
def estimate_muon_energy_from_photon_count(N):
    """
    Estimate energy from photon count using empirical fit:
    E_guess = 1.782 * N^0.674 + -97.460
    
    Parameters:
    -----------
    N : int or jnp.ndarray
        Number of observed photons
        
    Returns:
    --------
    float
        Estimated energy
    """
    return 1.782 * jnp.power(N, 0.674) - 97.460



# HCP lattice grid generation functions
def cylinder_hcp_points_local(center_xy, z_center, R_local, H_local, L, R, H):
    """
    Generate HCP lattice points inside a local cylinder.
    - center_xy, z_center: center of local cylinder
    - R_local, H_local: half-sizes of local cylinder
    - L: nearest-neighbor spacing
    - R, H: global cylinder dimensions
    """
    cx, cy = center_xy
    cz = z_center

    # in-plane triangular lattice spacing
    dx = L
    dy = math.sqrt(3) / 2 * L
    dz = math.sqrt(3) / 2 * L  # vertical spacing

    # bounds of local search region
    x_min, x_max = cx - R_local, cx + R_local
    y_min, y_max = cy - R_local, cy + R_local
    z_min, z_max = cz - H_local, cz + H_local

    xs = np.arange(x_min, x_max + dx, dx)
    ys = np.arange(y_min, y_max + dy, dy)
    zs = np.arange(z_min, z_max + dz, dz)

    pts = []
    for i, z in enumerate(zs):
        z_global = z
        # if z_global < 0 or z_global > H:
        #     continue

        layer_shift_x = 0.0
        layer_shift_y = 0.0
        if i % 2 == 1:
            layer_shift_x = 0.5 * dx
            layer_shift_y = 0.5 * dy

        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                xg = x + (iy % 2) * 0.5 * dx + layer_shift_x
                yg = y + layer_shift_y

                # check global cylinder constraint
                if (xg**2 + yg**2) <= R**2:
                    pts.append((xg, yg, z_global))

    return np.array(pts)


# Hierarchical cone-based direction search functions
def cone_points(axis, max_angle_rad, num_rings):
    """
    Generate points on a cone around 'axis' (unit vector).
    max_angle_rad: maximum opening angle of the cone
    num_rings: number of rings along the opening angle
    Returns array of points (N x 3)
    """
    axis = axis / np.linalg.norm(axis)

    # Find two orthogonal vectors perpendicular to axis
    if abs(axis[2]) < 0.99:
        u = np.cross(axis, [0,0,1])
    else:
        u = np.cross(axis, [0,1,0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    points = []

    for i in range(num_rings):
        theta = (i+1) * max_angle_rad / num_rings  # opening angle from axis
        ring_radius = np.sin(theta)
        z = np.cos(theta)

        # number of points on ring to match spacing along theta
        num_points_ring = max(1, int(2*np.pi*ring_radius*num_rings))
        for j in range(num_points_ring):
            phi = 2*np.pi*j / num_points_ring
            point = z*axis + ring_radius*(np.cos(phi)*u + np.sin(phi)*v)
            points.append(point)

    return np.array(points)


def hierarchical_direction_search_cone(prediction_simulator, detector_params, position, initial_t0, hit_detector_positions, 
                                     observed_times, observed_charge, true_data, energy_guess,
                                     levels, initial_div, max_angle_deg, reduction):
    """
    Hierarchical cone-based direction search using combined loss evaluation.
    
    Args:
        position: [x, y, z] position (from HCP search)
        initial_t0: starting t0 value
        energy_guess: energy estimate from photon count
        levels: number of hierarchical levels
        initial_div: initial divisions for global sphere sampling
        max_angle_deg: maximum cone opening angle in degrees
        reduction: cone angle reduction factor between levels
    
    Returns:
        dict with optimal direction and hierarchical search results
    """
    
    print(f"    Performing {levels}-level hierarchical cone direction search...")
    print(f"    Parameters: initial_div={initial_div}, max_angle={max_angle_deg}°, reduction={reduction}")
    
    # Initial global grid over sphere
    num_theta, num_phi = initial_div, initial_div*2
    thetas = np.linspace(0, np.pi, num_theta)
    phis = np.linspace(0, 2*np.pi, num_phi, endpoint=False)
    directions = np.array([[np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)] 
                          for t in thetas for p in phis])
    
    best_direction = None
    max_angle_rad = np.radians(max_angle_deg)
    path = []
    
    # Random key for loss evaluations
    search_key = jax.random.PRNGKey(789)
    
    for lvl in range(levels):
        print(f"      Level {lvl}: Evaluating {len(directions)} directions")
        
        # Evaluate combined loss for each direction
        level_results = []
        best_loss = float('inf')
        best_level_direction = None
        
        for i, direction in enumerate(directions):
            # Convert direction to spherical coordinates
            theta = np.arccos(np.clip(direction[2], -1.0, 1.0))
            phi = np.arctan2(direction[1], direction[0])
            
            # # Create parameter vector for this direction
            # test_params = jnp.array([
            #     position[0], position[1], position[2],
            #     initial_t0, theta, phi, energy_guess
            # ])
            
            search_key, _ = jax.random.split(search_key)
            
            try:
                # # Evaluate combined loss at this direction
                # combined_loss, vertex_loss, wc_loss, energy_loss_val = combined_product_loss(
                #     test_params, hit_detector_positions, observed_times, observed_charge,
                #     true_data, detector_params, search_key
                # )

                track_params = (energy_guess, position, jnp.array([theta, phi]))
                simulated_data = prediction_simulator(track_params, detector_params, search_key)
                loss = counts_loss(observed_charge, simulated_data[0])

                # loss, _ = spatial_loss_component(
                #     position, theta, phi, energy_guess, true_data, detector_params, search_key
                # )

                direction_result = {
                    'direction': direction.copy(),
                    'theta': float(theta),
                    'phi': float(phi),
                    'loss': float(loss),
                    # 'vertex_loss': float(vertex_loss),
                    # 'wc_loss': float(wc_loss),
                    # 'energy_loss': float(energy_loss_val)
                }
                
                level_results.append(direction_result)
                
                # Track best result for this level
                if loss < best_loss:
                    best_loss = loss
                    best_level_direction = direction.copy()
                    
            except Exception as e:
                print(f"        Error evaluating direction {i}: {e}")
                continue
        
        if best_level_direction is None:
            print(f"      ERROR: No valid directions found at level {lvl}")
            break
            
        # Store level results
        level_summary = {
            'level': lvl,
            'num_directions': len(directions),
            'directions': directions.copy(),
            'direction_results': level_results,
            'best_direction': best_level_direction.copy(),
            'best_loss': best_loss,
            'max_angle_rad': max_angle_rad
        }
        
        path.append(level_summary)
        best_direction = best_level_direction
        
        print(f"      Level {lvl} best loss: {best_loss:.6f}")
        print(f"      Level {lvl} best direction: {best_level_direction}")
        
        # Prepare next level: generate cone around best direction
        if lvl < levels - 1:  # Don't generate cone for last level
            num_rings = 4  # Fixed number of rings for cone
            directions = cone_points(best_level_direction, max_angle_rad, num_rings)
            
            # Shrink cone angle for next level
            max_angle_rad *= reduction
    
    # Convert final best direction to spherical coordinates
    final_theta = np.arccos(np.clip(best_direction[2], -1.0, 1.0))
    final_phi = np.arctan2(best_direction[1], best_direction[0])
    
    print(f"    Hierarchical cone search complete. Best loss: {path[-1]['best_loss']:.6f}")
    print(f"    Best direction: {best_direction}")
    print(f"    Best angles: θ={final_theta:.3f}, φ={final_phi:.3f}")
    
    return {
        'best_direction': best_direction,
        'best_theta': float(final_theta),
        'best_phi': float(final_phi),
        'best_loss': path[-1]['best_loss'],
        'search_path': path,
        'total_levels': len(path)
    }


@jit
def spherical_to_cartesian(theta, phi):
    """Convert spherical angles to Cartesian direction vector"""
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    
    return jnp.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])

@jit
def cartesian_to_spherical(direction):
    """Convert Cartesian direction vector to spherical angles"""
    # Normalize direction
    direction = direction / (jnp.linalg.norm(direction) + 1e-8)
    
    theta = jnp.arccos(jnp.clip(direction[2], -1.0, 1.0))
    phi = jnp.arctan2(direction[1], direction[0])
    
    return theta, phi


def hcp_position_grid_search(hit_detector_positions, observed_times, observed_charge, 
                            true_position, true_t0, R, H, L0, levels, reduction):
    """
    Perform HCP lattice grid search for optimal origin position using origin_time_loss
    
    Args:
        hit_detector_positions: positions of detectors with hits
        observed_times: timing data
        observed_charge: charge data 
        true_position: true position for comparison
        true_t0: true t0 for loss evaluation
        R, H: detector cylinder dimensions
        L0: initial HCP lattice spacing
        levels: number of refinement levels
        reduction: reduction factor between levels
    
    Returns:
        dict with HCP search results and optimal position
    """
    
    print(f"    Performing HCP lattice position grid search...")
    print(f"    Parameters: L0={L0}, levels={levels}, reduction={reduction}")
    print(f"    Detector dimensions: R={R:.1f}m, H={H:.1f}m")
    
    # Generate all HCP levels
    all_hcp_results = []
    best_overall_loss = float('inf')
    best_overall_position = None
    
    # Start from detector center
    center_xy = (0.0, 0.0)
    z_center = 0.0
    R_local = R
    H_local = H / 2.0
    L = L0
    
    for level in range(levels):
        print(f"      HCP Level {level}: L={L:.3f}, R_local={R_local:.3f}, H_local={H_local:.3f}")
        
        # Generate HCP lattice points for this level
        hcp_points = cylinder_hcp_points_local(center_xy, z_center, R_local, H_local, L, R, H)
        
        if len(hcp_points) == 0:
            print(f"      No valid HCP points at level {level}")
            break
            
        print(f"      Generated {len(hcp_points)} HCP points")
        
        # Evaluate origin_time_loss at each HCP point
        level_results = []
        best_level_loss = float('inf')
        best_level_position = None
        
        for i, point in enumerate(hcp_points):
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
                print(f"        Error evaluating point {i}: {e}")
                continue
        
        level_summary = {
            'level': level,
            'L': L,
            'center_xy': center_xy,
            'z_center': z_center,
            'R_local': R_local,
            'H_local': H_local,
            'num_points': len(hcp_points),
            'hcp_points': hcp_points,
            'point_results': level_results,
            'best_position': np.array(best_level_position) if best_level_position is not None else None,
            'best_loss': best_level_loss
        }
        
        all_hcp_results.append(level_summary)
        
        print(f"      Level {level} best loss: {best_level_loss:.6f}")
        print(f"      Level {level} best position: {best_level_position}")
        
        # Prepare for next level refinement
        if best_level_position is not None:
            center_xy = (float(best_level_position[0]), float(best_level_position[1]))
            z_center = float(best_level_position[2])
            
            # Shrink search region for next level
            L_next = L * reduction
            R_local = math.sqrt(3) / 2 * L/2
            H_local = math.sqrt(3) / 2 * L/2
            L = L_next
        else:
            break
    
    # Calculate final statistics
    final_position_error = float(jnp.linalg.norm(best_overall_position - true_position)) if best_overall_position is not None else float('inf')
    
    print(f"    HCP search complete. Best overall loss: {best_overall_loss:.6f}")
    print(f"    Best position: {best_overall_position}")
    print(f"    Position error: {final_position_error:.3f}m")
    
    return {
        'hcp_levels': all_hcp_results,
        'best_position': np.array(best_overall_position) if best_overall_position is not None else None,
        'best_loss': best_overall_loss,
        'position_error': final_position_error,
        'total_levels': len(all_hcp_results)
    }


def energy_scan_optimization(prediction_simulator, detector_params, position, theta, phi, initial_t0, hit_detector_positions,
                           observed_times, observed_charge, true_data, energy_guess,
                           energy_delta, n_steps):
    """
    Perform energy scan around initial energy guess to find optimal energy.
    
    Args:
        position: [x, y, z] position from HCP search
        theta, phi: direction angles from cone search
        initial_t0: t0 estimate
        energy_guess: initial energy estimate from photon count
        energy_delta: scan range (±energy_delta)
        n_steps: number of scan steps
    
    Returns:
        dict with energy scan results and optimal energy
    """
    
    print(f"    Performing energy scan around E_guess={energy_guess:.1f}")
    print(f"    Scan range: [{energy_guess-energy_delta:.1f}, {energy_guess+energy_delta:.1f}] in {n_steps} steps")
    
    # Generate energy scan points
    energy_min = energy_guess - energy_delta
    energy_max = energy_guess + energy_delta
    energy_scan_points = np.linspace(energy_min, energy_max, n_steps)
    
    scan_results = []
    best_energy = energy_guess
    best_loss = float('inf')
    
    # Random key for loss evaluations
    scan_key = jax.random.PRNGKey(456)
    
    for i, energy in enumerate(energy_scan_points):
        # Create parameter vector for this energy
        test_params = jnp.array([
            position[0], position[1], position[2],
            initial_t0, theta, phi, energy
        ])
        
        scan_key, _ = jax.random.split(scan_key)
        
        try:

            current_track_params = (energy, position, jnp.array([theta, phi]))
            simulated_data = prediction_simulator(current_track_params, detector_params, scan_key)
            loss = energy_loss(simulated_data[0], observed_charge)
            # combined_loss, vertex_loss, wc_loss, energy_loss_val = energy_loss(
            #     test_params, hit_detector_positions, observed_times, observed_charge,
            #     true_data, detector_params, scan_key
            # )
            
            energy_result = {
                'energy': float(energy),
                'loss': float(loss),
                # 'vertex_loss': float(vertex_loss),
                # 'wc_loss': float(wc_loss),
                # 'energy_loss': float(energy_loss_val)
            }
            
            scan_results.append(energy_result)
            
            # Track best energy
            if loss < best_loss:
                best_loss = loss
                best_energy = energy
                
        except Exception as e:
            print(f"        Error evaluating energy {energy:.1f}: {e}")
            continue
    
    print(f"    Energy scan complete. Best energy: {best_energy:.1f} (loss: {best_loss:.6f})")
    
    return {
        'energy_guess': float(energy_guess),
        'energy_min': float(energy_min),
        'energy_max': float(energy_max),
        'n_steps': n_steps,
        'scan_results': scan_results,
        'best_energy': float(best_energy),
        'best_loss': best_loss,
        'energy_improvement': float(abs(best_energy - energy_guess))
    }