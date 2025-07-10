#!/usr/bin/env python3
"""
Combined spatial-temporal track reconstruction.
Uses both Cherenkov cone spatial constraints AND corrected timing constraints.
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
from scipy.optimize import minimize
from sklearn.decomposition import PCA

# Add parent directories to path to access tools
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.utils import base_dir_path
from tools.geometry import generate_detector


def compute_cone_cylinder_intersection(cone_vertex, cone_direction, cone_angle, 
                                     cylinder_radius, cylinder_height, n_points=1000):
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
    
    print(f"Identified {np.sum(outer_edge_mask)} outer edge hits (>{percentile}th percentile)")
    
    return outer_edge_mask


def fit_combined_spatial_temporal(hit_positions, hit_times, hit_charges, detector_radius=5.0, detector_height=8.0):
    """
    Fit track parameters using BOTH spatial and temporal constraints.
    
    Parameters:
    -----------
    hit_positions : array_like
        (N, 3) array of hit positions
    hit_times : array_like
        (N,) array of hit times (with unknown t0 offset)
    hit_charges : array_like
        (N,) array of hit charges for weighting
    detector_radius : float
        Radius of cylindrical detector
    detector_height : float
        Height of cylindrical detector
    
    Returns:
    --------
    fitted_vertex : array
        Fitted track vertex position
    fitted_direction : array
        Fitted track direction (normalized)
    fitted_t0 : float
        Fitted time offset
    fitted_angle : float
        Cherenkov angle (fixed)
    success : bool
        Whether fitting was successful
    """
    
    # Cherenkov angle (fixed)
    n_water = 1.33
    cherenkov_angle = np.arccos(1.0 / n_water)
    
    # Physical constants (simulation units)
    c_water = 1.0  # Speed of light in water (simulation units)
    v_particle = 1.0  # Particle speed ≈ c
    
    print(f"Combined spatial-temporal fit to {len(hit_positions)} hits...")
    print(f"Using fixed Cherenkov angle: {np.degrees(cherenkov_angle):.1f}°")
    print(f"Time range (before t0 correction): {np.min(hit_times):.3f} to {np.max(hit_times):.3f}")
    
    # Identify outer edge hits for spatial weighting
    outer_edge_mask = identify_outer_edge_hits(hit_positions, hit_charges, percentile=85)

    def objective_function(params):
        """
        Objective function using BOTH spatial and temporal constraints.
        
        Parameters: [x0, y0, z0, t0, dx, dy, dz]
        where t0 is the global time offset for all hits
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
            
            # COMBINED RESIDUAL with weighting
            # Weight spatial more heavily for outer edge hits, temporal for all hits
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

            # combined_residual = (spatial_weight * spatial_residual) * np.sqrt(charge)
            
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
    
    # Initial t0 guess: use value from time-only analysis
    initial_t0 = -6.5  # From our previous analysis
    
    initial_params = np.concatenate([initial_vertex, [initial_t0], initial_direction])
    
    print(f"Initial guess - Vertex: {initial_vertex}, t0: {initial_t0:.3f}, Direction: {initial_direction}")
    
    # Parameter bounds
    bounds = [
        (-detector_radius*0.9, detector_radius*0.9),      # x0
        (-detector_radius*0.9, detector_radius*0.9),      # y0
        (-detector_height/2*0.9, detector_height/2*0.9),  # z0
        (-25.0, 0.0),  # t0 (time offset)
        (-1.0, 1.0),  # dx
        (-1.0, 1.0),  # dy
        (-1.0, 1.0),  # dz
    ]
    
    # Optimize with multiple attempts
    best_result = None
    best_loss = float('inf')
    
    # Try just the best method from previous tests
    methods = ['SLSQP']
    
    for method in methods:
        try:
            result = minimize(objective_function, initial_params, bounds=bounds, 
                             method=method, options={'maxiter': 500})
            
            if result.success and result.fun < best_loss:
                best_result = result
                best_loss = result.fun
                print(f"  {method} succeeded with loss: {result.fun:.2f}")
            else:
                print(f"  {method} failed or worse: {result.fun:.2f}")
                
        except Exception as e:
            print(f"  {method} error: {e}")
    
    if best_result is not None and best_result.fun < 1e6:
        fitted_vertex = best_result.x[:3]
        fitted_t0 = best_result.x[3]
        fitted_direction_unnorm = best_result.x[4:7]
        fitted_direction = fitted_direction_unnorm / np.linalg.norm(fitted_direction_unnorm)
        
        print(f"Combined optimization successful!")
        print(f"  Final loss: {best_result.fun:.2f}")
        print(f"  Fitted vertex: {fitted_vertex}")
        print(f"  Fitted t0: {fitted_t0:.3f}")
        print(f"  Fitted direction: {fitted_direction}")
        
        return fitted_vertex, fitted_direction, fitted_t0, cherenkov_angle, True
    else:
        print(f"Combined optimization failed!")
        return initial_vertex, initial_direction, initial_t0, cherenkov_angle, False


def create_cylinder_surface(radius, height, center=(0, 0, 0), n_points=20):
    """Create a transparent cylinder surface for detector boundaries."""
    theta = np.linspace(0, 2*np.pi, n_points)
    z = np.linspace(-height/2, height/2, n_points)
    theta_mesh, z_mesh = np.meshgrid(theta, z)
    
    x_mesh = center[0] + radius * np.cos(theta_mesh)
    y_mesh = center[1] + radius * np.sin(theta_mesh)
    z_mesh = center[2] + z_mesh
    
    return x_mesh, y_mesh, z_mesh


def main():
    """Main function for combined spatial-temporal reconstruction."""
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
    
    data_file = sorted(data_files)[-1]
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    events = data['events']
    
    # Test on available events
    # available_events = min(len(events), 3)  # Test first 3 events or all if fewer
    # test_events = list(range(available_events))
    

    test_events = [33]
    for event_idx in test_events:
        event = events[event_idx]
        event_id = event['event_id']
        true_position = jnp.array(event['vertex_position'])
        true_direction = jnp.array(event['direction'])
        energy = event['energy']
        hit_times = jnp.array(event['hit_times'])
        hit_charges = jnp.array(event['hit_charges'])
        
        print(f"\n{'='*60}")
        print(f"Analyzing event {event_id}:")
        print(f"True position: {true_position}")
        print(f"True direction: {true_direction}")
        print(f"Energy: {energy:.1f} MeV")
        
        # Track angle relative to detector axis
        track_angle_to_z = np.degrees(np.arccos(np.abs(true_direction[2])))
        print(f"Track angle to Z-axis: {track_angle_to_z:.1f}° (perpendicular = 90°)")
        
        # Filter significant hits
        min_charge = 50.0
        significant_mask = hit_charges >= min_charge
        significant_positions = sensor_positions[significant_mask]
        significant_charges = hit_charges[significant_mask]
        significant_times = hit_times[significant_mask]
        
        # Convert to numpy
        hit_positions = np.array(significant_positions)
        hit_charges_array = np.array(significant_charges)
        hit_times_array = np.array(significant_times)
        
        print(f"Using {len(hit_positions)} significant hits (charge >= {min_charge})")
        
        # Perform combined spatial-temporal reconstruction
        fitted_vertex, fitted_direction, fitted_t0, fitted_cherenkov_angle, success = fit_combined_spatial_temporal(
            hit_positions, hit_times_array, hit_charges_array, detector.r, detector.H
        )
        
        if success:
            # Calculate errors
            position_error = np.linalg.norm(fitted_vertex - true_position)
            direction_error = np.arccos(np.clip(np.abs(np.dot(fitted_direction, true_direction)), 0, 1))
            direction_error_deg = np.degrees(direction_error)
            
            print(f"\nCombined Spatial-Temporal Reconstruction Results:")
            print(f"  Position error: {position_error:.3f} m")
            print(f"  Direction error: {direction_error_deg:.1f}°")
            print(f"  Fitted t0 offset: {fitted_t0:.3f}")
            
            # Calculate both spatial and temporal residuals for diagnostics
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
            
            print(f"\nDiagnostics:")
            print(f"  Mean spatial residual: {np.mean(spatial_residuals):.3f} m")
            print(f"  RMS spatial residual: {np.sqrt(np.mean(spatial_residuals**2)):.3f} m")
            print(f"  Mean temporal residual: {np.mean(temporal_residuals):.3f} ns")
            print(f"  RMS temporal residual: {np.sqrt(np.mean(temporal_residuals**2)):.3f} ns")
            
            # Compute theoretical curves for visualization
            n_curve_points = 300  # High resolution curves
            
            print(f"Computing intersection curves with {n_curve_points} points...")
            print(f"Using Cherenkov angle: {np.degrees(fitted_cherenkov_angle):.1f}°")
            
            true_intersection = compute_cone_cylinder_intersection(
                true_position, true_direction, fitted_cherenkov_angle, detector.r, detector.H, n_points=n_curve_points
            )
            print(f"True intersection: {len(true_intersection)} points")
            
            fitted_intersection = compute_cone_cylinder_intersection(
                fitted_vertex, fitted_direction, fitted_cherenkov_angle, detector.r, detector.H, n_points=n_curve_points
            )
            print(f"Fitted intersection: {len(fitted_intersection)} points")
            
            # Create visualization
            fig = plt.figure(figsize=(16, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot detector hits (color by time, size by charge)
            scatter = ax.scatter(hit_positions[:, 0], hit_positions[:, 1], hit_positions[:, 2], 
                                c=hit_times_array, s=hit_charges_array*2, cmap='plasma', alpha=0.4,
                                label='Detector Hits (colored by time)')
            
            # Plot true track (high z-order to appear above everything)
            t_vals = np.linspace(0, 8, 100)
            true_track_points = true_position[:, np.newaxis] + t_vals[np.newaxis, :] * true_direction[:, np.newaxis]
            ax.plot(true_track_points[0], true_track_points[1], true_track_points[2], 
                    'blue', linewidth=5, label='True Track', zorder=30)
            ax.scatter(true_position[0], true_position[1], true_position[2], 
                       c='blue', s=400, marker='*', edgecolors='black', linewidth=3, 
                       label='True Origin', zorder=35)
            
            # Plot fitted track (high z-order to appear above everything)
            fitted_track_points = fitted_vertex[:, np.newaxis] + t_vals[np.newaxis, :] * fitted_direction[:, np.newaxis]
            ax.plot(fitted_track_points[0], fitted_track_points[1], fitted_track_points[2], 
                    'red', linewidth=5, label='Combined Fitted Track', zorder=30)
            ax.scatter(fitted_vertex[0], fitted_vertex[1], fitted_vertex[2], 
                       c='red', s=400, marker='*', edgecolors='black', linewidth=3, 
                       label='Combined Fitted Origin', zorder=35)
            
            # Plot theoretical intersection curves (thick lines, high z-order to appear above hits)
            if len(true_intersection) > 0:
                ax.plot(true_intersection[:, 0], true_intersection[:, 1], true_intersection[:, 2],
                       'blue', linewidth=6, alpha=0.9, label='True Intersection Curve', zorder=20)
            
            if len(fitted_intersection) > 0:
                ax.plot(fitted_intersection[:, 0], fitted_intersection[:, 1], fitted_intersection[:, 2],
                       'red', linewidth=6, alpha=0.9, label='Combined Fitted Intersection Curve', zorder=20)
            
            # Add cylinder boundaries
            x_cyl, y_cyl, z_cyl = create_cylinder_surface(detector.r, detector.H)
            ax.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.05, color='gray')
            
            # Set labels and title
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_zlabel('Z (m)', fontsize=12)
            ax.set_title(f'Event {event_id}: Combined Spatial-Temporal Reconstruction\\n' +
                        f'Pos Err: {position_error:.2f}m, Dir Err: {direction_error_deg:.1f}°, t0: {fitted_t0:.3f}\\n' +
                        f'Spatial RMS: {np.sqrt(np.mean(spatial_residuals**2)):.3f}m, Temporal RMS: {np.sqrt(np.mean(temporal_residuals**2)):.3f}ns',
                        fontsize=12)
            
            # Add colorbar and legend
            plt.colorbar(scatter, ax=ax, label='Hit Time', shrink=0.6)
            ax.legend(loc='upper right', fontsize=10)
            
            # Set axis limits
            ax.set_xlim([-6, 6])
            ax.set_ylim([-6, 6])
            ax.set_zlim([-5, 5])
            
            ax.grid(True, alpha=0.3)
            ax.view_init(elev=20, azim=-60)
            
            plt.tight_layout()
            plt.show()
            
        else:
            print("Combined spatial-temporal reconstruction failed!")


if __name__ == "__main__":
    main()