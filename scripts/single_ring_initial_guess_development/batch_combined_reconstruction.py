#!/usr/bin/env python3
"""
Batch reconstruction of all events using combined spatial-temporal track reconstruction.
Processes all events, saves individual images, and generates summary statistics.
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import sys
from datetime import datetime
from scipy.optimize import minimize
from sklearn.decomposition import PCA

# Add parent directories to path to access tools
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.utils import base_dir_path
from tools.geometry import generate_detector


def compute_cone_cylinder_intersection(cone_vertex, cone_direction, cone_angle, 
                                     cylinder_radius, cylinder_height, n_points=200):
    """
    Compute the intersection curve of a cone with a cylinder.
    Returns sorted intersection points for smooth curve visualization.
    """
    # Normalize cone direction
    cone_direction = np.array(cone_direction) / np.linalg.norm(cone_direction)
    cone_vertex = np.array(cone_vertex)
    
    intersection_points = []
    
    # Method 1: Parameterize by angle around cylinder side and solve for z
    theta_values = np.linspace(0, 2*np.pi, n_points)
    
    for theta in theta_values:
        # Point on cylinder surface
        x_cyl = cylinder_radius * np.cos(theta)
        y_cyl = cylinder_radius * np.sin(theta)
        
        # Solve for z where the point lies on the cone
        x0, y0, z0 = cone_vertex
        dx, dy, dz = cone_direction
        
        vx = x_cyl - x0
        vy = y_cyl - y0
        
        cos_angle = np.cos(cone_angle)
        cos_angle_sq = cos_angle**2
        
        A = vx*dx + vy*dy
        B = dz
        C = vx**2 + vy**2
        
        # Quadratic equation: a*vz² + b*vz + c = 0
        a = B**2 - cos_angle_sq
        b = 2*A*B
        c = A**2 - cos_angle_sq*C
        
        # Solve quadratic equation
        discriminant = b**2 - 4*a*c
        
        if discriminant >= 0 and abs(a) > 1e-10:
            sqrt_disc = np.sqrt(discriminant)
            vz1 = (-b + sqrt_disc) / (2*a)
            vz2 = (-b - sqrt_disc) / (2*a)
            
            z1 = vz1 + z0
            z2 = vz2 + z0
            
            # Check if solutions are within cylinder height bounds AND on forward cone
            half_height = cylinder_height / 2
            
            if -half_height <= z1 <= half_height:
                point1 = np.array([x_cyl, y_cyl, z1])
                vec_to_point1 = point1 - cone_vertex
                if np.dot(vec_to_point1, cone_direction) > 0:  # Forward cone only
                    intersection_points.append([x_cyl, y_cyl, z1])
                    
            if -half_height <= z2 <= half_height and abs(z2 - z1) > 1e-6:
                point2 = np.array([x_cyl, y_cyl, z2])
                vec_to_point2 = point2 - cone_vertex
                if np.dot(vec_to_point2, cone_direction) > 0:  # Forward cone only
                    intersection_points.append([x_cyl, y_cyl, z2])
        elif abs(a) <= 1e-10 and abs(b) > 1e-10:
            # Linear case
            vz = -c / b
            z = vz + z0
            half_height = cylinder_height / 2
            if -half_height <= z <= half_height:
                point = np.array([x_cyl, y_cyl, z])
                vec_to_point = point - cone_vertex
                if np.dot(vec_to_point, cone_direction) > 0:  # Forward cone only
                    intersection_points.append([x_cyl, y_cyl, z])
    
    if len(intersection_points) == 0:
        return np.array([])
    
    intersection_points = np.array(intersection_points)
    
    # Sort intersection points using nearest neighbor for smooth curve
    sorted_points = sort_intersection_points(intersection_points)
    
    return sorted_points


def sort_intersection_points(points):
    """Sort intersection points for smooth curve using nearest neighbor algorithm."""
    if len(points) == 0:
        return points
    
    points = np.array(points)
    n_points = len(points)
    
    if n_points == 1:
        return points
    
    # Start with the point with smallest angle
    angles = np.arctan2(points[:, 1], points[:, 0])
    start_idx = np.argmin(angles)
    
    sorted_points = [points[start_idx]]
    remaining_indices = set(range(n_points)) - {start_idx}
    
    current_point = points[start_idx]
    
    while remaining_indices:
        min_distance = float('inf')
        nearest_idx = None
        
        for idx in remaining_indices:
            candidate_point = points[idx]
            distance = np.linalg.norm(candidate_point - current_point)
            
            if distance < min_distance:
                min_distance = distance
                nearest_idx = idx
        
        if nearest_idx is not None:
            sorted_points.append(points[nearest_idx])
            remaining_indices.remove(nearest_idx)
            current_point = points[nearest_idx]
        else:
            nearest_idx = remaining_indices.pop()
            sorted_points.append(points[nearest_idx])
            current_point = points[nearest_idx]
    
    return np.array(sorted_points)


def calculate_emission_point_distance(vertex, direction, hit_pos, cherenkov_angle):
    """
    Calculate the distance along the track where photon must be emitted 
    to reach hit_pos at the Cherenkov angle.
    """
    # Vector from vertex to hit
    vec_to_hit = hit_pos - vertex
    
    # Project hit onto track direction
    projection = np.dot(vec_to_hit, direction)
    
    # Vector perpendicular to track (from track to hit)
    vec_perp = vec_to_hit - projection * direction
    perp_distance = np.linalg.norm(vec_perp)
    
    # For Cherenkov emission at angle θ_c:
    # tan(θ_c) = perp_distance / d_emission
    # Therefore: d_emission = perp_distance / tan(θ_c)
    
    tan_angle = np.tan(cherenkov_angle)
    if tan_angle > 1e-10:  # Avoid division by zero
        d_emission = perp_distance / tan_angle
    else:
        d_emission = projection  # Fallback for small angles
    
    return d_emission


def identify_outer_edge_hits(hit_positions, hit_charges, percentile=85):
    """
    Identify hits that are likely on the outer edge of the Cherenkov ring.
    """
    # Calculate distance from center of hit distribution
    hit_center = np.mean(hit_positions, axis=0)
    distances_from_center = np.linalg.norm(hit_positions - hit_center, axis=1)
    
    # Consider hits in the outer percentile as edge hits
    distance_threshold = np.percentile(distances_from_center, percentile)
    outer_edge_mask = distances_from_center >= distance_threshold
    
    return outer_edge_mask


def fit_combined_spatial_temporal(hit_positions, hit_times, hit_charges, detector_radius=5.0, detector_height=8.0):
    """
    Fit track parameters using BOTH spatial and temporal constraints.
    Uses the same approach as your edited combined_spatial_temporal_fit.py
    """
    
    # Cherenkov angle (fixed)
    n_water = 1.33
    cherenkov_angle = np.arccos(1.0 / n_water)
    
    # Physical constants (simulation units)
    c_water = 1.0  # Speed of light in water (simulation units)
    v_particle = 1.0  # Particle speed ≈ c
    
    # Identify outer edge hits for spatial weighting
    outer_edge_mask = identify_outer_edge_hits(hit_positions, hit_charges, percentile=85)

    def objective_function(params):
        """
        Objective function using BOTH spatial and temporal constraints.
        Uses your edited weighting scheme.
        """
        vertex = params[:3]
        t0 = params[3]  # Time offset
        direction_unnorm = params[4:7]
        
        # Normalize direction
        direction_norm = np.linalg.norm(direction_unnorm)
        if direction_norm < 1e-8:
            return 1e10  # Penalty for degenerate direction
        
        direction = direction_unnorm / direction_norm
        
        # Check if vertex is inside detector
        vertex_r = np.sqrt(vertex[0]**2 + vertex[1]**2)
        if vertex_r > detector_radius or abs(vertex[2]) > detector_height/2:
            return 1e6  # Large penalty for vertex outside detector
        
        # Calculate BOTH spatial and temporal residuals
        total_residual = 0.0
        cos_angle = np.cos(cherenkov_angle)
        
        for i, (hit_pos, hit_time, charge) in enumerate(zip(hit_positions, hit_times, hit_charges)):
            # Vector from vertex to hit
            vec_to_hit = hit_pos - vertex
            vec_length = np.linalg.norm(vec_to_hit)
            
            if vec_length < 1e-8:
                continue  # Skip if hit is at vertex
            
            # Check if hit is on forward side of cone
            forward_projection = np.dot(vec_to_hit, direction)
            if forward_projection <= 0:
                # Hit is behind vertex, add penalty
                total_residual += 1000.0 * np.sqrt(charge)
                continue
            
            # SPATIAL CONSTRAINT: Cherenkov cone geometry
            dot_product = np.dot(vec_to_hit, direction)
            expected_dot = cos_angle * vec_length
            spatial_residual = abs(dot_product - expected_dot)
            
            # TEMPORAL CONSTRAINT: Corrected Cherenkov timing
            d_emission = calculate_emission_point_distance(vertex, direction, hit_pos, cherenkov_angle)
            t_emission = d_emission / v_particle
            t_photon = vec_length / c_water
            expected_time = t_emission + t_photon + t0
            temporal_residual = abs(hit_time - expected_time)
            
            # COMBINED RESIDUAL with your edited weighting scheme
            if outer_edge_mask[i]:
                # Outer edge hits: strong spatial constraint, moderate temporal
                spatial_weight = 1.0
                temporal_weight = 0.5
            else:
                # Inner hits: moderate spatial constraint, strong temporal
                spatial_weight = 0.1
                temporal_weight = 0.1
            
            combined_residual = (spatial_weight * spatial_residual + 
                               temporal_weight * temporal_residual) * np.sqrt(charge)
            
            total_residual += combined_residual
        
        return total_residual
    
    # Initial guess based on hit distribution
    hit_center = np.mean(hit_positions, axis=0)
    
    # Estimate direction from PCA of hits
    pca = PCA(n_components=3)
    pca.fit(hit_positions)
    initial_direction = pca.components_[2]  # Normal to the plane (smallest variance)
    
    # Try both directions and see which gives a vertex inside detector
    direction_option1 = initial_direction
    direction_option2 = -initial_direction
    
    vertex_option1 = hit_center - 2.0 * direction_option1
    vertex_option2 = hit_center - 2.0 * direction_option2
    
    # Check which vertex option is better (inside detector bounds)
    def is_inside_detector_bounds(vertex):
        r = np.sqrt(vertex[0]**2 + vertex[1]**2)
        return r <= detector_radius * 0.9 and abs(vertex[2]) <= detector_height/2 * 0.9
    
    inside1 = is_inside_detector_bounds(vertex_option1)
    inside2 = is_inside_detector_bounds(vertex_option2)
    
    if inside1 and not inside2:
        initial_vertex = vertex_option1
        initial_direction = direction_option1
    elif inside2 and not inside1:
        initial_vertex = vertex_option2
        initial_direction = direction_option2
    else:
        # Choose the one closer to detector center
        dist1 = np.linalg.norm(vertex_option1)
        dist2 = np.linalg.norm(vertex_option2)
        if dist1 < dist2:
            initial_vertex = vertex_option1
            initial_direction = direction_option1
        else:
            initial_vertex = vertex_option2
            initial_direction = direction_option2
    
    # Final safety check: ensure vertex is inside detector
    vertex_r = np.sqrt(initial_vertex[0]**2 + initial_vertex[1]**2)
    if vertex_r > detector_radius * 0.9:
        scale_factor = (detector_radius * 0.8) / vertex_r
        initial_vertex = initial_vertex * scale_factor
    if abs(initial_vertex[2]) > detector_height/2 * 0.9:
        initial_vertex[2] = np.sign(initial_vertex[2]) * detector_height/2 * 0.8
    
    # Initial t0 guess
    initial_t0 = -6.5
    
    initial_params = np.concatenate([initial_vertex, [initial_t0], initial_direction])
    
    # Parameter bounds (using your edited bounds)
    bounds = [
        (-detector_radius*0.9, detector_radius*0.9),      # x0
        (-detector_radius*0.9, detector_radius*0.9),      # y0
        (-detector_height/2*0.9, detector_height/2*0.9),  # z0
        (-25.0, 0.0),  # t0 (time offset)
        (-1.00, 1.00),  # dx
        (-1.00, 1.00),  # dy
        (-1.00, 1.00),  # dz
    ]
    
    # Optimize
    try:
        result = minimize(objective_function, initial_params, bounds=bounds, 
                         method='SLSQP', options={'maxiter': 100})
        
        if result.success and result.fun < 1e6:
            fitted_vertex = result.x[:3]
            fitted_t0 = result.x[3]
            fitted_direction_unnorm = result.x[4:7]
            fitted_direction = fitted_direction_unnorm / np.linalg.norm(fitted_direction_unnorm)
            
            return fitted_vertex, fitted_direction, fitted_t0, cherenkov_angle, True
        else:
            return initial_vertex, initial_direction, initial_t0, cherenkov_angle, False
            
    except Exception:
        return initial_vertex, initial_direction, initial_t0, cherenkov_angle, False


def create_cylinder_surface(radius, height, center=(0, 0, 0), n_points=15):
    """Create a transparent cylinder surface for detector boundaries."""
    theta = np.linspace(0, 2*np.pi, n_points)
    z = np.linspace(-height/2, height/2, n_points)
    theta_mesh, z_mesh = np.meshgrid(theta, z)
    
    x_mesh = center[0] + radius * np.cos(theta_mesh)
    y_mesh = center[1] + radius * np.sin(theta_mesh)
    z_mesh = center[2] + z_mesh
    
    return x_mesh, y_mesh, z_mesh


def process_single_event(event, sensor_positions, detector, min_charge=50.0, save_plot=True, output_dir='generated_data/batch_combined_reconstruction'):
    """Process a single event and return combined reconstruction results."""
    event_id = event['event_id']
    true_position = jnp.array(event['vertex_position'])
    true_direction = jnp.array(event['direction'])
    energy = event['energy']
    hit_times = jnp.array(event['hit_times'])
    hit_charges = jnp.array(event['hit_charges'])
    
    # Filter significant hits
    significant_mask = hit_charges >= min_charge
    significant_positions = sensor_positions[significant_mask]
    significant_charges = hit_charges[significant_mask]
    significant_times = hit_times[significant_mask]
    
    if len(significant_positions) < 20:
        # Not enough hits for reliable reconstruction
        return {
            'event_id': event_id,
            'success': False,
            'error': 'Not enough significant hits',
            'n_hits': len(significant_positions),
            'true_position': np.array(true_position),
            'true_direction': np.array(true_direction),
            'energy': energy
        }
    
    try:
        # Convert to numpy
        hit_positions = np.array(significant_positions)
        hit_charges_array = np.array(significant_charges)
        hit_times_array = np.array(significant_times)
        
        # Perform combined spatial-temporal reconstruction
        fitted_vertex, fitted_direction, fitted_t0, fitted_cherenkov_angle, success = fit_combined_spatial_temporal(
            hit_positions, hit_times_array, hit_charges_array, detector.r, detector.H
        )
        
        if not success:
            return {
                'event_id': event_id,
                'success': False,
                'error': 'Combined fitting failed',
                'n_hits': len(significant_positions),
                'true_position': np.array(true_position),
                'true_direction': np.array(true_direction),
                'energy': energy
            }
        
        # Calculate errors
        position_error = np.linalg.norm(fitted_vertex - true_position)
        direction_error = np.arccos(np.clip(np.abs(np.dot(fitted_direction, true_direction)), 0, 1))
        direction_error_deg = np.degrees(direction_error)
        
        # Calculate residuals
        spatial_residuals = []
        temporal_residuals = []
        cos_angle = np.cos(fitted_cherenkov_angle)
        
        for hit_pos, hit_time in zip(hit_positions, hit_times_array):
            # Spatial residual
            vec_to_hit = hit_pos - fitted_vertex
            vec_length = np.linalg.norm(vec_to_hit)
            if vec_length > 1e-8:
                dot_product = np.dot(vec_to_hit, fitted_direction)
                expected_dot = cos_angle * vec_length
                spatial_residual = abs(dot_product - expected_dot)
                spatial_residuals.append(spatial_residual)
                
                # Temporal residual
                d_emission = calculate_emission_point_distance(fitted_vertex, fitted_direction, hit_pos, fitted_cherenkov_angle)
                t_emission = d_emission / 1.0
                t_photon = vec_length / 1.0
                expected_time = t_emission + t_photon + fitted_t0
                temporal_residual = abs(hit_time - expected_time)
                temporal_residuals.append(temporal_residual)
        
        spatial_residuals = np.array(spatial_residuals)
        temporal_residuals = np.array(temporal_residuals)
        
        # Track angle relative to detector axis
        track_angle_to_z = np.degrees(np.arccos(np.abs(true_direction[2])))
        
        # Create and save visualization if requested
        if save_plot:
            create_event_visualization(event_id, true_position, true_direction, fitted_vertex, fitted_direction,
                                     hit_positions, hit_charges_array, hit_times_array, detector, 
                                     position_error, direction_error_deg, fitted_t0,
                                     energy, track_angle_to_z, fitted_cherenkov_angle, 
                                     spatial_residuals, temporal_residuals, output_dir)
        
        # Return results
        return {
            'event_id': event_id,
            'success': True,
            'true_position': np.array(true_position),
            'true_direction': np.array(true_direction),
            'recon_position': fitted_vertex,
            'recon_direction': fitted_direction,
            'position_error': position_error,
            'direction_error_deg': direction_error_deg,
            'fitted_t0': fitted_t0,
            'energy': energy,
            'n_hits': len(significant_positions),
            'track_angle_to_z': track_angle_to_z,
            'cherenkov_angle': fitted_cherenkov_angle,
            'spatial_rms': np.sqrt(np.mean(spatial_residuals**2)) if len(spatial_residuals) > 0 else np.nan,
            'temporal_rms': np.sqrt(np.mean(temporal_residuals**2)) if len(temporal_residuals) > 0 else np.nan
        }
        
    except Exception as e:
        return {
            'event_id': event_id,
            'success': False,
            'error': str(e),
            'n_hits': len(significant_positions) if 'significant_positions' in locals() else 0,
            'true_position': np.array(true_position),
            'true_direction': np.array(true_direction),
            'energy': energy
        }


def create_event_visualization(event_id, true_position, true_direction, recon_origin, recon_direction,
                              hit_positions, hit_charges, hit_times, detector, position_error, direction_error_deg, 
                              fitted_t0, energy, track_angle_to_z, cherenkov_angle, 
                              spatial_residuals, temporal_residuals, output_dir):
    """Create and save visualization for a single event."""
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot detector hits (color by time, size by charge)
    scatter = ax.scatter(hit_positions[:, 0], hit_positions[:, 1], hit_positions[:, 2], 
                        c=hit_times, s=hit_charges*1.5, cmap='plasma', alpha=0.4)
    
    # Track parameters for plotting
    t_vals = np.linspace(0, 8, 100)
    
    # Plot true track (high z-order to appear above everything)
    true_track_points = true_position[:, np.newaxis] + t_vals[np.newaxis, :] * true_direction[:, np.newaxis]
    ax.plot(true_track_points[0], true_track_points[1], true_track_points[2], 
            'blue', linewidth=5, label='True Track', zorder=30)
    ax.scatter(true_position[0], true_position[1], true_position[2], 
               c='blue', s=400, marker='*', edgecolors='black', linewidth=3, 
               label='True Origin', zorder=35)
    
    # Plot reconstructed track (high z-order to appear above everything)
    recon_track_points = recon_origin[:, np.newaxis] + t_vals[np.newaxis, :] * recon_direction[:, np.newaxis]
    ax.plot(recon_track_points[0], recon_track_points[1], recon_track_points[2], 
            'red', linewidth=5, label='Combined Reconstructed Track', zorder=30)
    ax.scatter(recon_origin[0], recon_origin[1], recon_origin[2], 
               c='red', s=400, marker='*', edgecolors='black', linewidth=3, 
               label='Combined Reconstructed Origin', zorder=35)
    
    # Compute and plot theoretical intersection curves
    try:
        true_intersection = compute_cone_cylinder_intersection(
            true_position, true_direction, cherenkov_angle, detector.r, detector.H, n_points=200
        )
        fitted_intersection = compute_cone_cylinder_intersection(
            recon_origin, recon_direction, cherenkov_angle, detector.r, detector.H, n_points=200
        )
        
        # Plot theoretical intersection curves (thick lines, high z-order)
        if len(true_intersection) > 0:
            ax.plot(true_intersection[:, 0], true_intersection[:, 1], true_intersection[:, 2],
                   'blue', linewidth=6, alpha=0.9, label='True Intersection Curve', zorder=20)
        
        if len(fitted_intersection) > 0:
            ax.plot(fitted_intersection[:, 0], fitted_intersection[:, 1], fitted_intersection[:, 2],
                   'red', linewidth=6, alpha=0.9, label='Fitted Intersection Curve', zorder=20)
    except Exception:
        pass  # Skip intersection curves if computation fails
    
    # Add transparent cylinder surface for detector boundaries
    x_cyl, y_cyl, z_cyl = create_cylinder_surface(detector.r, detector.H)
    ax.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.05, color='gray')
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('Z (m)', fontsize=10)
    
    spatial_rms = np.sqrt(np.mean(spatial_residuals**2)) if len(spatial_residuals) > 0 else 0
    temporal_rms = np.sqrt(np.mean(temporal_residuals**2)) if len(temporal_residuals) > 0 else 0
    
    ax.set_title(f'Event {event_id}: Combined Spatial-Temporal Reconstruction\\n' +
                f'Pos Err: {position_error:.2f}m, Dir Err: {direction_error_deg:.1f}°, t0: {fitted_t0:.3f}\\n' +
                f'Spatial RMS: {spatial_rms:.3f}m, Temporal RMS: {temporal_rms:.3f}ns\\n' +
                f'Energy: {energy:.0f}MeV, Track Angle: {track_angle_to_z:.1f}°, Hits: {len(hit_positions)}',
                fontsize=10)
    
    # Add colorbar for time
    plt.colorbar(scatter, ax=ax, label='Hit Time (ns)', shrink=0.6)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=8)
    
    # Set axis limits
    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    ax.set_zlim([-5, 5])
    
    # Add grid
    ax.grid(True, alpha=0.2)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}/event_{event_id:02d}_combined_reconstruction.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()


def generate_summary_statistics(results, output_dir='generated_data/batch_combined_reconstruction'):
    """Generate comprehensive summary statistics and plots."""
    
    # Filter successful reconstructions
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    n_total = len(results)
    n_successful = len(successful_results)
    n_failed = len(failed_results)
    
    print(f"\\n{'='*60}")
    print(f"BATCH COMBINED RECONSTRUCTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total events processed: {n_total}")
    print(f"Successful reconstructions: {n_successful} ({100*n_successful/n_total:.1f}%)")
    print(f"Failed reconstructions: {n_failed} ({100*n_failed/n_total:.1f}%)")
    
    if n_failed > 0:
        print(f"\\nFailed events:")
        for r in failed_results:
            print(f"  Event {r['event_id']}: {r.get('error', 'Unknown error')} (hits: {r.get('n_hits', 'N/A')})")
    
    if n_successful == 0:
        print("No successful reconstructions to analyze.")
        return
    
    # Extract arrays for analysis
    position_errors = np.array([r['position_error'] for r in successful_results])
    direction_errors = np.array([r['direction_error_deg'] for r in successful_results])
    fitted_t0s = np.array([r['fitted_t0'] for r in successful_results])
    energies = np.array([r['energy'] for r in successful_results])
    n_hits = np.array([r['n_hits'] for r in successful_results])
    track_angles = np.array([r['track_angle_to_z'] for r in successful_results])
    spatial_rms = np.array([r['spatial_rms'] for r in successful_results if not np.isnan(r['spatial_rms'])])
    temporal_rms = np.array([r['temporal_rms'] for r in successful_results if not np.isnan(r['temporal_rms'])])
    
    # Position error statistics
    print(f"\\nPOSITION ERROR STATISTICS:")
    print(f"  Mean: {np.mean(position_errors):.3f} m")
    print(f"  Median: {np.median(position_errors):.3f} m")
    print(f"  Std: {np.std(position_errors):.3f} m")
    print(f"  Min: {np.min(position_errors):.3f} m")
    print(f"  Max: {np.max(position_errors):.3f} m")
    print(f"  90th percentile: {np.percentile(position_errors, 90):.3f} m")
    
    # Direction error statistics
    print(f"\\nDIRECTION ERROR STATISTICS:")
    print(f"  Mean: {np.mean(direction_errors):.1f}°")
    print(f"  Median: {np.median(direction_errors):.1f}°")
    print(f"  Std: {np.std(direction_errors):.1f}°")
    print(f"  Min: {np.min(direction_errors):.1f}°")
    print(f"  Max: {np.max(direction_errors):.1f}°")
    print(f"  90th percentile: {np.percentile(direction_errors, 90):.1f}°")
    
    # Timing statistics
    print(f"\\nTIMING STATISTICS:")
    print(f"  t0 offset - Mean: {np.mean(fitted_t0s):.3f} ns")
    print(f"  t0 offset - Std: {np.std(fitted_t0s):.3f} ns")
    if len(spatial_rms) > 0:
        print(f"  Spatial RMS - Mean: {np.mean(spatial_rms):.3f} m")
    if len(temporal_rms) > 0:
        print(f"  Temporal RMS - Mean: {np.mean(temporal_rms):.3f} ns")
    
    # Create summary plots
    create_summary_plots(successful_results, position_errors, direction_errors, fitted_t0s, energies, 
                        n_hits, track_angles, spatial_rms, temporal_rms, output_dir)
    
    # Save detailed results
    save_detailed_results(results, output_dir)


def create_summary_plots(results, position_errors, direction_errors, fitted_t0s, energies, n_hits, track_angles, spatial_rms, temporal_rms, output_dir):
    """Create comprehensive summary plots."""
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Plot 1: Position error histogram
    axes[0, 0].hist(position_errors, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(np.mean(position_errors), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(position_errors):.2f}m')
    axes[0, 0].axvline(np.median(position_errors), color='orange', linestyle='--', 
                      label=f'Median: {np.median(position_errors):.2f}m')
    axes[0, 0].set_xlabel('Position Error (m)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Position Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Direction error histogram
    axes[0, 1].hist(direction_errors, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(np.mean(direction_errors), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(direction_errors):.1f}°')
    axes[0, 1].axvline(np.median(direction_errors), color='orange', linestyle='--', 
                      label=f'Median: {np.median(direction_errors):.1f}°')
    axes[0, 1].set_xlabel('Direction Error (degrees)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Direction Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: t0 offset histogram
    axes[0, 2].hist(fitted_t0s, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[0, 2].axvline(np.mean(fitted_t0s), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(fitted_t0s):.1f}ns')
    axes[0, 2].set_xlabel('Fitted t0 Offset (ns)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('t0 Offset Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Error vs Energy
    axes[1, 0].scatter(energies, position_errors, alpha=0.6, color='blue', label='Position')
    axes[1, 0].scatter(energies, direction_errors/10, alpha=0.6, color='red', label='Direction/10')
    axes[1, 0].set_xlabel('Energy (MeV)')
    axes[1, 0].set_ylabel('Error')
    axes[1, 0].set_title('Reconstruction Error vs Energy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Error vs Track Angle
    axes[1, 1].scatter(track_angles, position_errors, alpha=0.6, color='blue')
    axes[1, 1].set_xlabel('Track Angle to Z-axis (degrees)')
    axes[1, 1].set_ylabel('Position Error (m)')
    axes[1, 1].set_title('Position Error vs Track Angle')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Direction error vs Track Angle
    axes[1, 2].scatter(track_angles, direction_errors, alpha=0.6, color='green')
    axes[1, 2].set_xlabel('Track Angle to Z-axis (degrees)')
    axes[1, 2].set_ylabel('Direction Error (degrees)')
    axes[1, 2].set_title('Direction Error vs Track Angle')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Plot 7: Spatial RMS histogram
    if len(spatial_rms) > 0:
        axes[2, 0].hist(spatial_rms, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[2, 0].axvline(np.mean(spatial_rms), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(spatial_rms):.3f}m')
        axes[2, 0].set_xlabel('Spatial RMS (m)')
        axes[2, 0].set_ylabel('Count')
        axes[2, 0].set_title('Spatial Residual RMS Distribution')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 8: Temporal RMS histogram
    if len(temporal_rms) > 0:
        axes[2, 1].hist(temporal_rms, bins=20, alpha=0.7, color='cyan', edgecolor='black')
        axes[2, 1].axvline(np.mean(temporal_rms), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(temporal_rms):.3f}ns')
        axes[2, 1].set_xlabel('Temporal RMS (ns)')
        axes[2, 1].set_ylabel('Count')
        axes[2, 1].set_title('Temporal Residual RMS Distribution')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
    
    # Plot 9: Performance summary
    axes[2, 2].axis('off')
    summary_text = f"""COMBINED RECONSTRUCTION PERFORMANCE SUMMARY

Position Error:
  Mean: {np.mean(position_errors):.2f} ± {np.std(position_errors):.2f} m
  Median: {np.median(position_errors):.2f} m
  90%ile: {np.percentile(position_errors, 90):.2f} m

Direction Error:
  Mean: {np.mean(direction_errors):.1f} ± {np.std(direction_errors):.1f}°
  Median: {np.median(direction_errors):.1f}°
  90%ile: {np.percentile(direction_errors, 90):.1f}°

Timing:
  Mean t0: {np.mean(fitted_t0s):.1f} ± {np.std(fitted_t0s):.1f} ns"""
    
    if len(spatial_rms) > 0:
        summary_text += f"\\n  Spatial RMS: {np.mean(spatial_rms):.3f} m"
    if len(temporal_rms) > 0:
        summary_text += f"\\n  Temporal RMS: {np.mean(temporal_rms):.3f} ns"
    
    summary_text += f"""

Performance:
  Mean Hits/Event: {np.mean(n_hits):.0f}
  Success Rate: {len(results)}/{len(results)} events

Method: Combined Spatial-Temporal"""
    
    axes[2, 2].text(0.05, 0.95, summary_text, transform=axes[2, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('Combined Spatial-Temporal Track Reconstruction: Batch Analysis Summary', fontsize=16)
    plt.tight_layout()
    
    # Save summary plot
    summary_filename = f'{output_dir}/combined_reconstruction_summary.png'
    plt.savefig(summary_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\\nSummary plots saved to: {summary_filename}")


def save_detailed_results(results, output_dir):
    """Save detailed results to JSON file."""
    
    # Prepare data for JSON serialization
    json_results = []
    for r in results:
        json_result = {}
        for key, value in r.items():
            if isinstance(value, np.ndarray):
                json_result[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                json_result[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                json_result[key] = int(value)
            else:
                json_result[key] = value
        json_results.append(json_result)
    
    # Add timestamp and metadata
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'total_events': len(results),
        'successful_events': len([r for r in results if r['success']]),
        'reconstruction_method': 'combined_spatial_temporal',
        'min_charge_threshold': 50.0,
        'results': json_results
    }
    
    # Save to file
    results_filename = f'{output_dir}/detailed_combined_results.json'
    with open(results_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Detailed results saved to: {results_filename}")


def main():
    """Main function to process all events with combined reconstruction."""
    # Load detector configuration
    detector = generate_detector(base_dir_path() + 'config/IWCD_geom_config.json')
    sensor_positions = jnp.array(detector.all_points)
    
    print(f"Detector geometry:")
    print(f"  Radius: {detector.r:.1f} m")
    print(f"  Height: {detector.H:.1f} m")
    print(f"  Total sensors: {len(sensor_positions)}")
    
    # Load generated data
    local_data_dir = os.path.join(os.path.dirname(__file__), 'generated_data')
    data_files = glob.glob(os.path.join(local_data_dir, 'lucid_simulated_events_*.json'))
    if not data_files:
        print("No generated data found. Please run generate_lucid_data.py first.")
        return
    
    # Use the most recent file
    data_file = sorted(data_files)[-1]
    print(f"Loading data from {data_file}")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    events = data['events']
    n_events = len(events)
    print(f"Processing {n_events} events with combined spatial-temporal reconstruction...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'generated_data', 'batch_combined_reconstruction')
    os.makedirs(output_dir, exist_ok=True)
    
    test_n_events = n_events #min(5, n_events)
    results = []
    for i, event in enumerate(events[:test_n_events]):
        print(f"\\nProcessing event {i+1}/{test_n_events} (ID: {event['event_id']})...")
        result = process_single_event(event, sensor_positions, detector, 
                                    min_charge=50.0, save_plot=True, output_dir=output_dir)
        results.append(result)
        
        if result['success']:
            print(f"  Success! Pos err: {result['position_error']:.2f}m, "\
                  f"Dir err: {result['direction_error_deg']:.1f}°, t0: {result['fitted_t0']:.3f}")
        else:
            print(f"  Failed: {result['error']}")
    
    # Generate summary statistics and plots
    print(f"\\nGenerating summary statistics...")
    generate_summary_statistics(results, output_dir)
    
    print(f"\\nBatch combined reconstruction complete!")
    print(f"Individual event plots saved in: {output_dir}/")
    print(f"Summary statistics and plots saved in: {output_dir}/")


if __name__ == "__main__":
    main()