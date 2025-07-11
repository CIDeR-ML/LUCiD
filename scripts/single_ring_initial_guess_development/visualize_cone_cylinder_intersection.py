#!/usr/bin/env python3
"""
Visualize the intersection of a Cherenkov cone with a cylindrical detector.
This script develops the visualization to be used in adaptive_loss_search.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
import argparse
import sys


def create_cylinder_surface(radius, height, center=(0, 0, 0), n_points=50):
    """Create a transparent cylinder surface for detector boundaries."""
    theta = np.linspace(0, 2*np.pi, n_points)
    z = np.linspace(-height/2, height/2, n_points)
    theta_mesh, z_mesh = np.meshgrid(theta, z)
    
    x_mesh = center[0] + radius * np.cos(theta_mesh)
    y_mesh = center[1] + radius * np.sin(theta_mesh)
    z_mesh = center[2] + z_mesh
    
    return x_mesh, y_mesh, z_mesh


def create_cone_surface(vertex, direction, angle, length=10, n_radial=50, n_length=30):
    """
    Create a cone surface for Cherenkov emission visualization.
    
    Parameters:
    -----------
    vertex : array_like
        Cone vertex position (3D)
    direction : array_like
        Cone axis direction (normalized)
    angle : float
        Half-angle of the cone (Cherenkov angle in radians)
    length : float
        Length of the cone to visualize
    n_radial : int
        Number of points around the cone
    n_length : int
        Number of points along the cone axis
    """
    # Normalize direction
    direction = np.array(direction) / np.linalg.norm(direction)
    vertex = np.array(vertex)
    
    # Create parametric representation
    # t: distance along cone axis
    # phi: angle around cone axis
    t = np.linspace(0, length, n_length)
    phi = np.linspace(0, 2*np.pi, n_radial)
    t_mesh, phi_mesh = np.meshgrid(t, phi)
    
    # Radius at distance t is t * tan(angle)
    r_mesh = t_mesh * np.tan(angle)
    
    # Create orthonormal basis for the cone
    # Find two vectors perpendicular to direction
    if abs(direction[2]) < 0.9:
        perp1 = np.cross(direction, [0, 0, 1])
    else:
        perp1 = np.cross(direction, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    
    # Generate cone points
    x_mesh = vertex[0] + t_mesh * direction[0] + r_mesh * (np.cos(phi_mesh) * perp1[0] + np.sin(phi_mesh) * perp2[0])
    y_mesh = vertex[1] + t_mesh * direction[1] + r_mesh * (np.cos(phi_mesh) * perp1[1] + np.sin(phi_mesh) * perp2[1])
    z_mesh = vertex[2] + t_mesh * direction[2] + r_mesh * (np.cos(phi_mesh) * perp1[2] + np.sin(phi_mesh) * perp2[2])
    
    return x_mesh, y_mesh, z_mesh


def sort_3d_curve_points(points):
    """
    Sort 3D points to form a smooth curve using nearest neighbor approach.
    """
    if len(points) <= 2:
        return points
    
    sorted_points = [points[0]]
    remaining_points = list(points[1:])
    
    while remaining_points:
        current_point = sorted_points[-1]
        distances = [np.linalg.norm(p - current_point) for p in remaining_points]
        nearest_idx = np.argmin(distances)
        
        sorted_points.append(remaining_points.pop(nearest_idx))
    
    return np.array(sorted_points)


def compute_cone_cylinder_intersection(vertex, direction, cone_angle, 
                                     cylinder_radius, cylinder_height, n_points=500):
    """
    Compute the intersection curve of a cone with a cylinder using analytical approach.
    
    Parameters:
    -----------
    vertex : array_like
        Cone vertex position
    direction : array_like
        Cone axis direction (normalized)
    cone_angle : float
        Half-angle of the cone (Cherenkov angle)
    cylinder_radius : float
        Radius of the cylinder
    cylinder_height : float
        Height of the cylinder
    n_points : int
        Number of points to sample
        
    Returns:
    --------
    intersection_points : array
        (N, 3) array of intersection points
    """
    direction = np.array(direction) / np.linalg.norm(direction)
    vertex = np.array(vertex)
    
    # Create orthonormal basis for the cone
    if abs(direction[2]) < 0.9:
        perp1 = np.cross(direction, [0, 0, 1])
    else:
        perp1 = np.cross(direction, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    
    intersection_points = []
    
    # Sample angles around the cone axis
    phi_values = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    for phi in phi_values:
        # Direction vector on cone surface at angle phi
        cone_dir = (direction * np.cos(cone_angle) + 
                   perp1 * np.sin(cone_angle) * np.cos(phi) + 
                   perp2 * np.sin(cone_angle) * np.sin(phi))
        
        found_intersection = False
        
        # Find intersection with cylinder side using ray-cylinder intersection
        # Ray: P = vertex + t * cone_dir
        # Cylinder: x^2 + y^2 = R^2
        
        a = cone_dir[0]**2 + cone_dir[1]**2
        b = 2 * (vertex[0] * cone_dir[0] + vertex[1] * cone_dir[1])
        c = vertex[0]**2 + vertex[1]**2 - cylinder_radius**2
        
        if abs(a) > 1e-10:  # Not parallel to cylinder axis
            discriminant = b**2 - 4*a*c
            
            if discriminant >= 0:
                # Two possible intersections
                t1 = (-b + np.sqrt(discriminant)) / (2*a)
                t2 = (-b - np.sqrt(discriminant)) / (2*a)
                
                # Check both solutions, take the closest positive one
                valid_t = []
                for t in [t1, t2]:
                    if t > 0:  # Only forward direction
                        point = vertex + t * cone_dir
                        if -cylinder_height/2 <= point[2] <= cylinder_height/2:
                            valid_t.append(t)
                
                if valid_t:
                    t = min(valid_t)  # Take closest intersection
                    point = vertex + t * cone_dir
                    intersection_points.append(point)
                    found_intersection = True
        
        # If no cylinder side intersection, check top and bottom caps
        if not found_intersection:
            for z_cap in [cylinder_height/2, -cylinder_height/2]:
                if abs(cone_dir[2]) > 1e-10:  # Not parallel to cap
                    t = (z_cap - vertex[2]) / cone_dir[2]
                    if t > 0:  # Forward direction
                        point = vertex + t * cone_dir
                        r = np.sqrt(point[0]**2 + point[1]**2)
                        if r <= cylinder_radius:
                            intersection_points.append(point)
                            break
    
    if not intersection_points:
        return np.array([]).reshape(0, 3)
    
    # Convert to array and remove near-duplicates
    points = np.array(intersection_points)
    
    # Remove duplicates by checking distances
    unique_points = []
    for p in points:
        if not unique_points or all(np.linalg.norm(p - up) > 0.01 for up in unique_points):
            unique_points.append(p)
    
    return np.array(unique_points)


def visualize_cone_cylinder_intersection(vertex, direction, cone_angle, 
                                       cylinder_radius=5.0, cylinder_height=8.0,
                                       save_path='cone_cylinder_intersection.png',
                                       show_cone_surface=False):
    """
    Create visualization of cone-cylinder intersection.
    """
    # Ensure numpy arrays
    vertex = np.array(vertex)
    direction = np.array(direction) / np.linalg.norm(direction)
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot cylinder
    x_cyl, y_cyl, z_cyl = create_cylinder_surface(cylinder_radius, cylinder_height)
    ax.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.2, color='gray', label='Detector')
    
    # Plot cone surface if requested
    if show_cone_surface:
        cone_length = 8.0
        x_cone, y_cone, z_cone = create_cone_surface(vertex, direction, cone_angle, cone_length, n_radial=40, n_length=30)
        
        ax.plot_surface(x_cone, y_cone, z_cone, 
                       alpha=0.3, color='blue', label='Cherenkov Cone')
    
    # Plot intersection curve
    intersection_points = compute_cone_cylinder_intersection(vertex, direction, cone_angle, 
                                                           cylinder_radius, cylinder_height)
    
    if len(intersection_points) > 0:
        # Sort points for smooth curve - use 3D sorting algorithm
        sorted_points = sort_3d_curve_points(intersection_points)
        
        # Close the curve if endpoints are close
        if len(sorted_points) > 2:
            dist_to_start = np.linalg.norm(sorted_points[-1] - sorted_points[0])
            if dist_to_start < 1.0:  # If curve is closed
                sorted_points = np.vstack([sorted_points, sorted_points[0]])
        
        ax.plot(sorted_points[:, 0], sorted_points[:, 1], sorted_points[:, 2], 
                'red', linewidth=4, label='Intersection Curve', zorder=20)
        
        # Also plot as scatter for visibility
        ax.scatter(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2],
                  c='red', s=50, alpha=0.6, zorder=25)
        
        # Mark start and end points
        ax.scatter(sorted_points[0, 0], sorted_points[0, 1], sorted_points[0, 2],
                  c='green', s=100, marker='o', label='Start', zorder=26)
        ax.scatter(sorted_points[-1, 0], sorted_points[-1, 1], sorted_points[-1, 2],
                  c='blue', s=100, marker='s', label='End', zorder=26)
    
    # Plot cone vertex and direction
    ax.scatter(vertex[0], vertex[1], vertex[2], c='green', s=200, marker='*', 
               edgecolors='black', linewidth=2, label='Cone Vertex', zorder=30)
    
    # Plot cone axis
    axis_length = 8.0
    axis_end = vertex + axis_length * direction
    ax.plot([vertex[0], axis_end[0]], [vertex[1], axis_end[1]], [vertex[2], axis_end[2]],
            'green', linewidth=3, label='Cone Axis', zorder=15)
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'Cherenkov Cone-Cylinder Intersection\n' +
                f'Vertex: [{vertex[0]:.2f}, {vertex[1]:.2f}, {vertex[2]:.2f}], ' +
                f'Cherenkov Angle: {np.degrees(cone_angle):.1f}°',
                fontsize=14)
    
    # Set axis limits
    ax.set_xlim([-cylinder_radius*1.2, cylinder_radius*1.2])
    ax.set_ylim([-cylinder_radius*1.2, cylinder_radius*1.2])
    ax.set_zlim([-cylinder_height/2*1.2, cylinder_height/2*1.2])
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    
    # Show the plot interactively
    plt.show()


def main(test_case_index=None):
    """Test cone-cylinder intersection visualization with different scenarios."""
    
    # Detector parameters (typical values)
    cylinder_radius = 5.0
    cylinder_height = 8.0
    
    # Cherenkov angle in water (n ≈ 1.33)
    n_water = 1.33
    cherenkov_angle = np.arccos(1.0 / n_water)  # About 41.2 degrees
    
    print(f"Cherenkov angle in water: {np.degrees(cherenkov_angle):.1f}°")
    
    # Test cases
    test_cases = [
        {
            'name': 'centered_upward',
            'vertex': [0, 0, 0],
            'direction': [0, 0, 1],
            'description': 'Centered vertex, upward direction'
        },
        {
            'name': 'off_center_diagonal',
            'vertex': [2, 1, -2],
            'direction': np.array([1, 1, 1]) / np.sqrt(3),
            'description': 'Off-center vertex, diagonal direction'
        },
        {
            'name': 'near_wall_horizontal',
            'vertex': [3, 0, 0],
            'direction': [1, 0, 0],
            'description': 'Near wall, horizontal direction'
        },
        {
            'name': 'complex_angle',
            'vertex': [-1, 2, 1],
            'direction': np.array([-0.5, -0.3, 0.8]) / np.linalg.norm([-0.5, -0.3, 0.8]),
            'description': 'Complex positioning and angle'
        }
    ]
    
    # If specific test case requested, only run that one
    if test_case_index is not None:
        if 0 <= test_case_index < len(test_cases):
            test_cases = [test_cases[test_case_index]]
        else:
            print(f"Invalid test case index {test_case_index}. Valid range: 0-{len(test_cases)-1}")
            return
    
    for i, test_case in enumerate(test_cases):
        case_num = test_case_index if test_case_index is not None else i
        print(f"\nTest case {case_num+1}: {test_case['description']}")
        output_file = f"cone_cylinder_test_{test_case['name']}.png"
        
        visualize_cone_cylinder_intersection(
            test_case['vertex'],
            test_case['direction'],
            cherenkov_angle,
            cylinder_radius,
            cylinder_height,
            output_file,
            show_cone_surface=True  # Show cone for test cases
        )
        
        # Also compute and report number of intersection points
        intersection_points = compute_cone_cylinder_intersection(
            test_case['vertex'],
            test_case['direction'],
            cherenkov_angle,
            cylinder_radius,
            cylinder_height
        )
        print(f"  Found {len(intersection_points)} intersection points")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize cone-cylinder intersections')
    parser.add_argument('--case', '-c', type=int, 
                        help='Test case index (0-3) to run. If not specified, runs all cases.')
    
    args = parser.parse_args()
    main(test_case_index=args.case)