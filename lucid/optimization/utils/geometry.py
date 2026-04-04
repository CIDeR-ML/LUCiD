"""
Utility functions for LUCiD optimization, including geometry calculations.
"""

import numpy as np


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


def create_cylinder_surface(radius, height, center=(0, 0, 0), n_points=20):
    """Create a transparent cylinder surface for detector boundaries."""
    theta = np.linspace(0, 2*np.pi, n_points)
    z = np.linspace(-height/2, height/2, n_points)
    theta_mesh, z_mesh = np.meshgrid(theta, z)
    
    x_mesh = center[0] + radius * np.cos(theta_mesh)
    y_mesh = center[1] + radius * np.sin(theta_mesh)
    z_mesh = center[2] + z_mesh
    
    return x_mesh, y_mesh, z_mesh


def create_sphere_surface(radius, center=(0, 0, 0), n_points=30):
    """Create a transparent sphere surface for detector boundaries."""
    phi = np.linspace(0, np.pi, n_points)
    theta = np.linspace(0, 2*np.pi, n_points)
    phi_mesh, theta_mesh = np.meshgrid(phi, theta)
    
    x_mesh = center[0] + radius * np.sin(phi_mesh) * np.cos(theta_mesh)
    y_mesh = center[1] + radius * np.sin(phi_mesh) * np.sin(theta_mesh)
    z_mesh = center[2] + radius * np.cos(phi_mesh)
    
    return x_mesh, y_mesh, z_mesh


def create_box_surface(x_size, y_size, z_size, center=(0, 0, 0)):
    """Create box edges for detector boundaries."""
    # Define vertices
    vertices = np.array([
        [-x_size/2, -y_size/2, -z_size/2],
        [x_size/2, -y_size/2, -z_size/2],
        [x_size/2, y_size/2, -z_size/2],
        [-x_size/2, y_size/2, -z_size/2],
        [-x_size/2, -y_size/2, z_size/2],
        [x_size/2, -y_size/2, z_size/2],
        [x_size/2, y_size/2, z_size/2],
        [-x_size/2, y_size/2, z_size/2]
    ])
    vertices += np.array(center)
    
    # Define edges connecting vertices
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    return vertices, edges


def compute_cone_cylinder_intersection(vertex, direction, cone_angle, 
                                     cylinder_radius, cylinder_height, n_points=500):
    """
    Compute the intersection curve of a cone with a cylinder using analytical approach.
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
        
        # Find intersection with cylinder side
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


def compute_cone_sphere_intersection(vertex, direction, cone_angle, sphere_radius, n_points=500):
    """
    Compute the intersection curve of a cone with a sphere.
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
        
        # Ray-sphere intersection
        # Ray: P = vertex + t * cone_dir
        # Sphere: ||P||^2 = R^2
        
        # Solve quadratic equation
        a = np.dot(cone_dir, cone_dir)  # Should be 1 for normalized direction
        b = 2 * np.dot(vertex, cone_dir)
        c = np.dot(vertex, vertex) - sphere_radius**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant >= 0:
            # Two possible intersections
            t1 = (-b + np.sqrt(discriminant)) / (2*a)
            t2 = (-b - np.sqrt(discriminant)) / (2*a)
            
            # Take the closest positive intersection
            valid_t = [t for t in [t1, t2] if t > 0]
            
            if valid_t:
                t = min(valid_t)
                point = vertex + t * cone_dir
                intersection_points.append(point)
    
    if not intersection_points:
        return np.array([]).reshape(0, 3)
    
    # Convert to array and sort for smooth curve
    points = np.array(intersection_points)
    
    # Sort points by angle around the cone axis for smooth plotting
    if len(points) > 2:
        # Project points onto a plane perpendicular to the cone axis
        projected = points - np.outer(np.dot(points, direction), direction)
        angles = np.arctan2(np.dot(projected, perp2), np.dot(projected, perp1))
        sorted_indices = np.argsort(angles)
        points = points[sorted_indices]
    
    return points


def compute_cone_box_intersection(vertex, direction, cone_angle, 
                                 box_x, box_y, box_z, n_points=500):
    """
    Compute the intersection curves of a cone with a box.
    Returns a list of curve segments.
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
    
    # Define box faces
    faces = [
        {'normal': [1, 0, 0], 'point': [box_x/2, 0, 0], 'bounds': {'y': [-box_y/2, box_y/2], 'z': [-box_z/2, box_z/2]}},
        {'normal': [-1, 0, 0], 'point': [-box_x/2, 0, 0], 'bounds': {'y': [-box_y/2, box_y/2], 'z': [-box_z/2, box_z/2]}},
        {'normal': [0, 1, 0], 'point': [0, box_y/2, 0], 'bounds': {'x': [-box_x/2, box_x/2], 'z': [-box_z/2, box_z/2]}},
        {'normal': [0, -1, 0], 'point': [0, -box_y/2, 0], 'bounds': {'x': [-box_x/2, box_x/2], 'z': [-box_z/2, box_z/2]}},
        {'normal': [0, 0, 1], 'point': [0, 0, box_z/2], 'bounds': {'x': [-box_x/2, box_x/2], 'y': [-box_y/2, box_y/2]}},
        {'normal': [0, 0, -1], 'point': [0, 0, -box_z/2], 'bounds': {'x': [-box_x/2, box_x/2], 'y': [-box_y/2, box_y/2]}}
    ]
    
    all_segments = []
    
    # For each face, find cone intersection
    for face in faces:
        face_points = []
        phi_values = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        
        for phi in phi_values:
            # Direction vector on cone surface
            cone_dir = (direction * np.cos(cone_angle) + 
                       perp1 * np.sin(cone_angle) * np.cos(phi) + 
                       perp2 * np.sin(cone_angle) * np.sin(phi))
            
            # Ray-plane intersection
            # Ray: P = vertex + t * cone_dir
            # Plane: (P - face_point) · face_normal = 0
            
            denom = np.dot(cone_dir, face['normal'])
            if abs(denom) > 1e-10:  # Not parallel to face
                t = np.dot(np.array(face['point']) - vertex, face['normal']) / denom
                
                if t > 0:  # Forward direction
                    point = vertex + t * cone_dir
                    
                    # Check if point is within face bounds
                    in_bounds = True
                    for coord, bounds in face['bounds'].items():
                        coord_idx = {'x': 0, 'y': 1, 'z': 2}[coord]
                        if not (bounds[0] <= point[coord_idx] <= bounds[1]):
                            in_bounds = False
                            break
                    
                    if in_bounds:
                        face_points.append(point)
        
        if face_points:
            # Sort points for smooth curve
            face_points = np.array(face_points)
            if len(face_points) > 2:
                # Sort by angle within the face
                center = np.mean(face_points, axis=0)
                # Find two orthogonal directions in the face
                face_normal = np.array(face['normal'])
                if abs(face_normal[2]) < 0.9:
                    u = np.cross(face_normal, [0, 0, 1])
                else:
                    u = np.cross(face_normal, [1, 0, 0])
                u = u / np.linalg.norm(u)
                v = np.cross(face_normal, u)
                
                # Project points onto face coordinates
                rel_points = face_points - center
                angles = np.arctan2(np.dot(rel_points, v), np.dot(rel_points, u))
                sorted_indices = np.argsort(angles)
                face_points = face_points[sorted_indices]
            
            all_segments.append(face_points)
    
    return all_segments


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