#!/usr/bin/env python3
"""
Utility functions for computing and visualizing cone-cylinder intersections.
To be used by adaptive_loss_search.py for Cherenkov cone visualization.
"""

import numpy as np


def compute_cone_cylinder_intersection_curve(vertex, direction, cone_angle, 
                                           cylinder_radius, cylinder_height, 
                                           n_samples=200):
    """
    Compute a smooth intersection curve of a cone with a cylinder.
    
    Parameters:
    -----------
    vertex : array_like
        Cone vertex position (3D)
    direction : array_like
        Cone axis direction (will be normalized)
    cone_angle : float
        Half-angle of the cone (Cherenkov angle in radians)
    cylinder_radius : float
        Radius of the cylinder
    cylinder_height : float
        Height of the cylinder
    n_samples : int
        Number of points to sample around the cone
        
    Returns:
    --------
    curve_points : array
        (N, 3) array of points forming the intersection curve
    """
    vertex = np.array(vertex)
    direction = np.array(direction) / np.linalg.norm(direction)
    
    # Create orthonormal basis for the cone
    if abs(direction[2]) < 0.9:
        perp1 = np.cross(direction, [0, 0, 1])
    else:
        perp1 = np.cross(direction, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    
    intersection_points = []
    
    # Sample angles around the cone axis
    phi_values = np.linspace(0, 2*np.pi, n_samples)
    
    for phi in phi_values:
        # Direction vector on cone surface at angle phi
        cone_dir = (direction * np.cos(cone_angle) + 
                   perp1 * np.sin(cone_angle) * np.cos(phi) + 
                   perp2 * np.sin(cone_angle) * np.sin(phi))
        
        # Find intersection with cylinder side
        # Ray equation: P = vertex + t * cone_dir
        # Cylinder equation: x^2 + y^2 = R^2
        
        # Substituting ray into cylinder equation gives quadratic in t
        a = cone_dir[0]**2 + cone_dir[1]**2
        b = 2 * (vertex[0] * cone_dir[0] + vertex[1] * cone_dir[1])
        c = vertex[0]**2 + vertex[1]**2 - cylinder_radius**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant >= 0 and abs(a) > 1e-10:
            # Two possible intersections
            t1 = (-b + np.sqrt(discriminant)) / (2*a)
            t2 = (-b - np.sqrt(discriminant)) / (2*a)
            
            # Check both solutions
            for t in [t1, t2]:
                if t > 0:  # Only forward direction
                    point = vertex + t * cone_dir
                    # Check if within cylinder height
                    if abs(point[2]) <= cylinder_height/2:
                        intersection_points.append(point)
                        break  # Take first valid intersection
        
        # Also check intersections with top and bottom caps
        for z_cap in [cylinder_height/2, -cylinder_height/2]:
            if abs(cone_dir[2]) > 1e-10:
                t = (z_cap - vertex[2]) / cone_dir[2]
                if t > 0:
                    point = vertex + t * cone_dir
                    r = np.sqrt(point[0]**2 + point[1]**2)
                    if r <= cylinder_radius:
                        # Check if this is the closest intersection
                        if not intersection_points or t < np.linalg.norm(intersection_points[-1] - vertex):
                            intersection_points.append(point)
    
    if not intersection_points:
        return np.array([]).reshape(0, 3)
    
    # Convert to array and remove duplicates
    points = np.array(intersection_points)
    
    # Sort points by angle around vertex for smooth curve
    if len(points) > 2:
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        points = points[sorted_indices]
    
    return points


def get_cherenkov_angle(refractive_index=1.33):
    """
    Calculate Cherenkov angle for given refractive index.
    
    Parameters:
    -----------
    refractive_index : float
        Refractive index of the medium (default 1.33 for water)
        
    Returns:
    --------
    angle : float
        Cherenkov angle in radians
    """
    return np.arccos(1.0 / refractive_index)


def create_cone_mesh(vertex, direction, angle, length=10, n_radial=30, n_length=20):
    """
    Create mesh points for cone surface visualization.
    
    Returns x_mesh, y_mesh, z_mesh for use with ax.plot_surface()
    """
    vertex = np.array(vertex)
    direction = np.array(direction) / np.linalg.norm(direction)
    
    # Create orthonormal basis
    if abs(direction[2]) < 0.9:
        perp1 = np.cross(direction, [0, 0, 1])
    else:
        perp1 = np.cross(direction, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    
    # Parametric representation
    t = np.linspace(0, length, n_length)
    phi = np.linspace(0, 2*np.pi, n_radial)
    t_mesh, phi_mesh = np.meshgrid(t, phi)
    
    # Radius at distance t
    r_mesh = t_mesh * np.tan(angle)
    
    # Generate cone points
    x_mesh = vertex[0] + t_mesh * direction[0] + r_mesh * (np.cos(phi_mesh) * perp1[0] + np.sin(phi_mesh) * perp2[0])
    y_mesh = vertex[1] + t_mesh * direction[1] + r_mesh * (np.cos(phi_mesh) * perp1[1] + np.sin(phi_mesh) * perp2[1])
    z_mesh = vertex[2] + t_mesh * direction[2] + r_mesh * (np.cos(phi_mesh) * perp1[2] + np.sin(phi_mesh) * perp2[2])
    
    return x_mesh, y_mesh, z_mesh


if __name__ == "__main__":
    # Simple test
    print("Testing cone-cylinder intersection computation...")
    
    vertex = [0, 0, 0]
    direction = [0, 0, 1]
    cherenkov_angle = get_cherenkov_angle()
    
    points = compute_cone_cylinder_intersection_curve(
        vertex, direction, cherenkov_angle, 
        cylinder_radius=5.0, cylinder_height=8.0
    )
    
    print(f"Cherenkov angle: {np.degrees(cherenkov_angle):.1f}°")
    print(f"Found {len(points)} intersection points")
    
    if len(points) > 0:
        print(f"Average radius: {np.mean(np.sqrt(points[:, 0]**2 + points[:, 1]**2)):.2f} m")
        print(f"Z range: [{np.min(points[:, 2]):.2f}, {np.max(points[:, 2]):.2f}] m")