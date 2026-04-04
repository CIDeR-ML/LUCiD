"""
Utility functions for detector geometries including sensor patterns and visualization.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def generate_concentric_hexagons(n_sensors, radius_eff):
    """Generate hexagonal pattern using concentric rings.
    
    Returns exact number of sensors in a regular hexagonal pattern.
    Pattern: center (1) + rings of 6k sensors each, where k is ring number.
    Total sensors for n rings: 1 + 6(1 + 2 + ... + n) = 1 + 3n(n+1)
    """
    if n_sensors == 0:
        return np.array([]).reshape(0, 2)
    
    if n_sensors == 1:
        return np.array([[0, 0]])
    
    # Find how many complete rings we can fit
    # Solve: 1 + 3n(n+1) <= n_sensors for largest n
    n_rings = 0
    while 1 + 3 * (n_rings + 1) * (n_rings + 2) <= n_sensors:
        n_rings += 1
    
    # Calculate spacing to fit the outermost ring within radius_eff
    if n_rings == 0:
        spacing = radius_eff  # Single sensor at center
    else:
        spacing = radius_eff / n_rings
    
    points = []
    
    # Center point
    points.append([0, 0])
    
    # Generate concentric hexagonal rings
    for ring in range(1, n_rings + 1):
        ring_radius = ring * spacing
        n_sensors_in_ring = 6 * ring
        
        # Generate points around the ring
        for i in range(n_sensors_in_ring):
            angle = 2 * np.pi * i / n_sensors_in_ring
            x = ring_radius * np.cos(angle)
            y = ring_radius * np.sin(angle)
            points.append([x, y])
    
    # If we need more sensors and have space, add partial outer ring
    current_count = len(points)
    if current_count < n_sensors and n_rings * spacing < radius_eff:
        remaining = n_sensors - current_count
        next_ring = n_rings + 1
        next_ring_radius = next_ring * spacing
        
        if next_ring_radius <= radius_eff:
            # Add sensors from the next ring
            sensors_to_add = min(remaining, 6 * next_ring)
            for i in range(sensors_to_add):
                angle = 2 * np.pi * i / (6 * next_ring)
                x = next_ring_radius * np.cos(angle)
                y = next_ring_radius * np.sin(angle)
                points.append([x, y])
    
    return np.array(points[:n_sensors])  # Ensure exact count


def fibonacci_sphere_points_numpy(n_points, radius=1.0):
    """Generate approximately equidistant points on sphere surface using Fibonacci spiral.
    
    Parameters
    ----------
    n_points : int
        Number of points to generate
    radius : float, optional
        Radius of the sphere, default 1.0
        
    Returns
    -------
    np.ndarray
        Array of shape (n_points, 3) containing point coordinates
    """
    center = np.array([0.0, 0.0, 0.0])
    
    indices = np.arange(0, n_points, dtype=float)
    
    # Golden ratio
    golden_ratio = (1 + np.sqrt(5)) / 2
    
    # Fibonacci spiral algorithm
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * indices / n_points)
    
    # Convert to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    points = np.stack([x, y, z], axis=1) + center
    
    return points


def create_disc_mesh(center, normal, radius, n_segments=20):
    """
    Create a circular disc mesh with specified center, normal, and radius.
    
    Parameters:
    -----------
    center : array-like
        3D center position of the disc
    normal : array-like  
        3D normal vector (will be normalized)
    radius : float
        Radius of the disc
    n_segments : int
        Number of segments for the circle (higher = smoother)
        
    Returns:
    --------
    vertices : np.ndarray
        Array of shape (n_segments + 1, 3) containing vertex positions
    faces : np.ndarray
        Array of shape (n_segments, 3) containing triangle indices
    """
    center = np.array(center)
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)  # Normalize
    
    # Create circle in XY plane
    angles = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
    circle_2d = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles), 
        np.zeros(n_segments)
    ])
    
    # Add center point
    vertices_local = np.vstack([np.array([0, 0, 0]), circle_2d])
    
    # Calculate rotation from [0, 0, 1] to target normal
    z_axis = np.array([0, 0, 1])
    
    if np.allclose(normal, z_axis):
        # No rotation needed
        rotation_matrix = np.eye(3)
    elif np.allclose(normal, -z_axis):
        # 180 degree rotation around X axis
        rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        # General rotation
        axis = np.cross(z_axis, normal)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(z_axis, normal), -1, 1))
        rotation = Rotation.from_rotvec(axis * angle)
        rotation_matrix = rotation.as_matrix()
    
    # Apply rotation and translation
    vertices = (rotation_matrix @ vertices_local.T).T + center
    
    # Create triangular faces (fan triangulation from center)
    faces = []
    for i in range(n_segments):
        faces.append([0, i + 1, ((i + 1) % n_segments) + 1])
    
    return vertices, np.array(faces)


def calculate_surface_normals(detector, sensor_indices):
    """
    Calculate surface normal vectors for detector positions.
    
    Parameters:
    -----------
    detector : Detector (Cylinder, Sphere, or Box)
        Detector geometry object
    sensor_indices : array-like
        Indices of sensors to calculate normals for
        
    Returns:
    --------
    normals : np.ndarray
        Array of shape (len(sensor_indices), 3) containing normal vectors
    """
    positions = detector.all_points[sensor_indices]
    normals = np.zeros_like(positions)
    
    # Import here to avoid circular imports
    from .cylinder import Cylinder
    from .sphere import Sphere
    from .box import Box
    
    if isinstance(detector, Cylinder):
        for i, idx in enumerate(sensor_indices):
            pos = positions[i]
            case = detector.ID_to_case[idx]
            
            if case == 0:  # Barrel
                # Normal points radially outward from cylinder axis
                radial_vector = pos[:2] - detector.C[:2]  # Only X,Y components
                radial_vector = radial_vector / np.linalg.norm(radial_vector)
                normals[i] = np.array([radial_vector[0], radial_vector[1], 0])
            elif case == 1:  # Top cap
                normals[i] = np.array([0, 0, 1])
            elif case == 2:  # Bottom cap  
                normals[i] = np.array([0, 0, -1])
    elif isinstance(detector, Sphere):
        for i, pos in enumerate(positions):
            # Normal points radially outward from sphere center
            normal = pos - detector.C
            normals[i] = normal / np.linalg.norm(normal)
    elif isinstance(detector, Box):
        for i, idx in enumerate(sensor_indices):
            case = detector.ID_to_case[idx]
            
            # Box faces: 0=front(+y), 1=back(-y), 2=left(-x), 3=right(+x), 4=top(+z), 5=bottom(-z)
            if case == 0:  # Front face (+y)
                normals[i] = np.array([0, 1, 0])
            elif case == 1:  # Back face (-y)
                normals[i] = np.array([0, -1, 0])
            elif case == 2:  # Left face (-x)
                normals[i] = np.array([-1, 0, 0])
            elif case == 3:  # Right face (+x)
                normals[i] = np.array([1, 0, 0])
            elif case == 4:  # Top face (+z)
                normals[i] = np.array([0, 0, 1])
            elif case == 5:  # Bottom face (-z)
                normals[i] = np.array([0, 0, -1])
    
    return normals