"""
Box (rectangular prism) detector photon propagation functions.
"""

import jax
import jax.numpy as jnp
from functools import partial
from jax import lax

from tools.propagate_base import (
    process_intersection_normals, compute_sensor_intersections_base,
    find_closest_sensors
)
from tools.overlap import create_overlap_prob


@jax.jit
def intersect_box_face(ray_origin, ray_direction, face_normal, face_distance):
    """Calculate intersection of a ray with a box face (plane).

    Parameters
    ----------
    ray_origin : jnp.ndarray
        Starting point of the ray [x, y, z]
    ray_direction : jnp.ndarray
        Direction vector of the ray [dx, dy, dz]
    face_normal : jnp.ndarray
        Normal vector of the face [nx, ny, nz]
    face_distance : float
        Distance from origin to face along normal

    Returns
    -------
    tuple
        (bool, float) - (whether intersection exists, distance to intersection)
    """
    LARGE = 1e10
    
    # Ray-plane intersection: (p - p0) . n = 0, where p = ray_origin + t * ray_direction
    denom = jnp.dot(ray_direction, face_normal)
    
    def normal_intersection(_):
        t = (face_distance - jnp.dot(ray_origin, face_normal)) / denom
        intersects_ = t > 0
        tval_ = jnp.where(intersects_, t, LARGE)
        return (intersects_, tval_)
    
    def parallel_intersection(_):
        # Ray is parallel to plane
        on_plane = jnp.abs(jnp.dot(ray_origin, face_normal) - face_distance) < 1e-12
        intersects_ = on_plane
        tval_ = jnp.where(on_plane, 0.0, LARGE)
        return (intersects_, tval_)
    
    is_parallel = jnp.abs(denom) < 1e-12
    intersects, tval = lax.cond(is_parallel,
                                parallel_intersection,
                                normal_intersection,
                                operand=None)
    
    return intersects, tval


@jax.jit
def intersect_box(ray_origin, ray_direction, length, width, height):
    """Find the closest intersection point with any face of the box.

    Parameters
    ----------
    ray_origin : jnp.ndarray
        Starting point of the ray [x, y, z]
    ray_direction : jnp.ndarray
        Direction vector of the ray [dx, dy, dz]
    length : float
        Length of the box (x-dimension)
    width : float
        Width of the box (y-dimension)
    height : float
        Height of the box (z-dimension)

    Returns
    -------
    tuple
        (bool, float, int) - (whether intersection exists, distance, face index)
        face index: 0=front(+y), 1=back(-y), 2=left(-x), 3=right(+x), 4=top(+z), 5=bottom(-z)
    """
    # Define face normals and distances
    face_normals = jnp.array([
        [0., 1., 0.],   # Front face (+y)
        [0., -1., 0.],  # Back face (-y)
        [-1., 0., 0.],  # Left face (-x)
        [1., 0., 0.],   # Right face (+x)
        [0., 0., 1.],   # Top face (+z)
        [0., 0., -1.]   # Bottom face (-z)
    ])
    
    face_distances = jnp.array([
        width/2,   # Front face distance
        width/2,   # Back face distance
        length/2,  # Left face distance
        length/2,  # Right face distance
        height/2,  # Top face distance
        height/2   # Bottom face distance
    ])
    
    # Calculate intersections with all faces
    def intersect_single_face(i):
        intersects, t = intersect_box_face(ray_origin, ray_direction, 
                                          face_normals[i], face_distances[i])
        
        # Check if intersection point is within face bounds
        intersection_point = ray_origin + t * ray_direction
        
        if i == 0:  # Front face (+y)
            within_bounds = ((jnp.abs(intersection_point[0]) <= length/2) & 
                           (jnp.abs(intersection_point[2]) <= height/2))
        elif i == 1:  # Back face (-y)
            within_bounds = ((jnp.abs(intersection_point[0]) <= length/2) & 
                           (jnp.abs(intersection_point[2]) <= height/2))
        elif i == 2:  # Left face (-x)
            within_bounds = ((jnp.abs(intersection_point[1]) <= width/2) & 
                           (jnp.abs(intersection_point[2]) <= height/2))
        elif i == 3:  # Right face (+x)
            within_bounds = ((jnp.abs(intersection_point[1]) <= width/2) & 
                           (jnp.abs(intersection_point[2]) <= height/2))
        elif i == 4:  # Top face (+z)
            within_bounds = ((jnp.abs(intersection_point[0]) <= length/2) & 
                           (jnp.abs(intersection_point[1]) <= width/2))
        else:  # Bottom face (-z)
            within_bounds = ((jnp.abs(intersection_point[0]) <= length/2) & 
                           (jnp.abs(intersection_point[1]) <= width/2))
        
        valid_intersection = intersects & within_bounds
        final_t = jnp.where(valid_intersection, t, 1e10)
        
        return valid_intersection, final_t
    
    # Vectorized intersection calculation
    all_intersects = []
    all_ts = []
    
    for i in range(6):
        intersects_i, t_i = intersect_single_face(i)
        all_intersects.append(intersects_i)
        all_ts.append(t_i)
    
    intersects = jnp.stack(all_intersects)
    ts = jnp.stack(all_ts)
    
    min_t_index = jnp.argmin(ts)
    min_t = jnp.min(ts)
    any_intersects = jnp.any(intersects)
    
    return any_intersects, min_t, min_t_index


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def intersect_box_with_grid(ray_origin, ray_direction, length, width, height, 
                           n_x, n_y, n_z):
    """Find intersection with box and compute grid cell indices.

    Parameters
    ----------
    ray_origin : jnp.ndarray
        Starting point of the ray [x, y, z]
    ray_direction : jnp.ndarray
        Direction vector of the ray [dx, dy, dz]
    length : float
        Length of the box (x-dimension)
    width : float
        Width of the box (y-dimension)
    height : float
        Height of the box (z-dimension)
    n_x : int
        Number of grid divisions in x-direction
    n_y : int
        Number of grid divisions in y-direction
    n_z : int
        Number of grid divisions in z-direction

    Returns
    -------
    tuple
        (intersects, t, face_idx, grid_indices, intersection_point)
    """
    intersects, t, face = intersect_box(ray_origin, ray_direction, length, width, height)
    intersection_point = ray_origin + t * ray_direction
    
    # Calculate grid indices based on which face was hit
    x, y, z = intersection_point[0], intersection_point[1], intersection_point[2]
    
    # Face-specific grid index calculations
    def get_front_back_indices():  # Front/Back faces (y-planes)
        x_idx = jnp.floor((x + length/2) / length * n_x).astype(jnp.int32)
        z_idx = jnp.floor((z + height/2) / height * n_z).astype(jnp.int32)
        x_idx = jnp.clip(x_idx, 0, n_x - 1)
        z_idx = jnp.clip(z_idx, 0, n_z - 1)
        return jnp.array([x_idx, z_idx])
    
    def get_left_right_indices():  # Left/Right faces (x-planes)
        y_idx = jnp.floor((y + width/2) / width * n_y).astype(jnp.int32)
        z_idx = jnp.floor((z + height/2) / height * n_z).astype(jnp.int32)
        y_idx = jnp.clip(y_idx, 0, n_y - 1)
        z_idx = jnp.clip(z_idx, 0, n_z - 1)
        return jnp.array([y_idx, z_idx])
    
    def get_top_bottom_indices():  # Top/Bottom faces (z-planes)
        x_idx = jnp.floor((x + length/2) / length * n_x).astype(jnp.int32)
        y_idx = jnp.floor((y + width/2) / width * n_y).astype(jnp.int32)
        x_idx = jnp.clip(x_idx, 0, n_x - 1)
        y_idx = jnp.clip(y_idx, 0, n_y - 1)
        return jnp.array([x_idx, y_idx])
    
    # Select grid indices based on face
    grid_indices = jnp.where(
        (face == 0) | (face == 1),  # Front or Back
        get_front_back_indices(),
        jnp.where(
            (face == 2) | (face == 3),  # Left or Right
            get_left_right_indices(),
            get_top_bottom_indices()  # Top or Bottom
        )
    )
    
    return intersects, t, face, grid_indices, intersection_point


batch_intersect_box_with_grid = jax.vmap(intersect_box_with_grid,
                                         in_axes=(0, 0, None, None, None, None, None, None))


def calculate_box_normals(face_indices):
    """
    Calculate normals for box faces.
    
    Parameters
    ----------
    face_indices : ndarray
        Face indices for intersections
        
    Returns
    -------
    ndarray
        Normal vectors for box face intersections
    """
    # Define normals for each face
    face_normals = jnp.array([
        [0., 1., 0.],   # Front face (+y)
        [0., -1., 0.],  # Back face (-y)
        [-1., 0., 0.],  # Left face (-x)
        [1., 0., 0.],   # Right face (+x)
        [0., 0., 1.],   # Top face (+z)
        [0., 0., -1.]   # Bottom face (-z)
    ])
    
    return face_normals[face_indices]


def box_bounds_check(points, length, width, height):
    """
    Check if points are within box bounds.
    
    Parameters
    ----------
    points : ndarray
        Points to check [x, y, z]
    length : float
        Box length (x-dimension)
    width : float
        Box width (y-dimension)
    height : float
        Box height (z-dimension)
        
    Returns
    -------
    ndarray
        Boolean array indicating which points are inside box
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    inside_x = (x >= -length/2) & (x <= length/2)
    inside_y = (y >= -width/2) & (y <= width/2)
    inside_z = (z >= -height/2) & (z <= height/2)
    
    return True #inside_x & inside_y & inside_z


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def assign_sensors_to_box_grid(sensors, sensor_radius, length, width, height,
                                n_x, n_y, n_z):
    """Assign sensors to box grid cells, handling overlap across cell boundaries.

    Parameters
    ----------
    sensors : jnp.ndarray
        Array of sensor positions, shape (n_sensors, 3)
    sensor_radius : float
        Radius of each sensor
    length : float
        Box length (x-dimension)
    width : float
        Box width (y-dimension)
    height : float
        Box height (z-dimension)
    n_x : int
        Number of grid divisions in x-direction
    n_y : int
        Number of grid divisions in y-direction
    n_z : int
        Number of grid divisions in z-direction

    Returns
    -------
    jnp.ndarray
        Array of shape (n_sensors, 4, 3) containing up to 4 grid cell assignments
        per sensor. -1 indicates no assignment.
    """

    def assign_single_sensor(sensor):
        x, y, z = sensor
        
        # Determine which face the sensor is closest to
        dist_to_faces = jnp.array([
            jnp.abs(y - width/2),   # Front face
            jnp.abs(y + width/2),   # Back face
            jnp.abs(x + length/2),  # Left face
            jnp.abs(x - length/2),  # Right face
            jnp.abs(z - height/2),  # Top face
            jnp.abs(z + height/2)   # Bottom face
        ])
        
        closest_face = jnp.argmin(dist_to_faces)
        min_distance = jnp.min(dist_to_faces)
        
        # Check if sensor is close enough to any face
        on_surface = min_distance <= sensor_radius
        
        def assign_surface():
            def assign_front_back():
                # Front/Back faces use x,z coordinates
                x_idx = jnp.floor((x + length/2) / length * n_x).astype(jnp.int32)
                z_idx = jnp.floor((z + height/2) / height * n_z).astype(jnp.int32)
                x_idx = jnp.clip(x_idx, 0, n_x - 1)
                z_idx = jnp.clip(z_idx, 0, n_z - 1)
                
                # Calculate overlap with neighboring cells
                x_frac = ((x + length/2) / length * n_x) % 1
                z_frac = ((z + height/2) / height * n_z) % 1
                
                include_x_right = x_frac >= 1 - sensor_radius / (length / n_x)
                include_x_left = x_frac <= sensor_radius / (length / n_x)
                include_z_up = z_frac >= 1 - sensor_radius / (height / n_z)
                include_z_down = z_frac <= sensor_radius / (height / n_z)
                
                x_right = jnp.clip(x_idx + 1, 0, n_x - 1)
                x_left = jnp.clip(x_idx - 1, 0, n_x - 1)
                z_up = jnp.clip(z_idx + 1, 0, n_z - 1)
                z_down = jnp.clip(z_idx - 1, 0, n_z - 1)
                
                indices = jnp.array([
                    [x_idx, z_idx, closest_face],
                    [x_right, z_idx, closest_face],
                    [x_idx, z_up, closest_face],
                    [x_right, z_up, closest_face]
                ])
                
                selection = jnp.array([
                    1.0,
                    include_x_right,
                    include_z_up,
                    include_x_right * include_z_up
                ])
                
                sorted_indices = indices[jnp.argsort(-selection)]
                return jnp.where(jnp.arange(4)[:, None] < jnp.sum(selection), sorted_indices[:4], -1)
            
            def assign_left_right():
                # Left/Right faces use y,z coordinates
                y_idx = jnp.floor((y + width/2) / width * n_y).astype(jnp.int32)
                z_idx = jnp.floor((z + height/2) / height * n_z).astype(jnp.int32)
                y_idx = jnp.clip(y_idx, 0, n_y - 1)
                z_idx = jnp.clip(z_idx, 0, n_z - 1)
                
                # Calculate overlap with neighboring cells
                y_frac = ((y + width/2) / width * n_y) % 1
                z_frac = ((z + height/2) / height * n_z) % 1
                
                include_y_right = y_frac >= 1 - sensor_radius / (width / n_y)
                include_y_left = y_frac <= sensor_radius / (width / n_y)
                include_z_up = z_frac >= 1 - sensor_radius / (height / n_z)
                include_z_down = z_frac <= sensor_radius / (height / n_z)
                
                y_right = jnp.clip(y_idx + 1, 0, n_y - 1)
                y_left = jnp.clip(y_idx - 1, 0, n_y - 1)
                z_up = jnp.clip(z_idx + 1, 0, n_z - 1)
                z_down = jnp.clip(z_idx - 1, 0, n_z - 1)
                
                indices = jnp.array([
                    [y_idx, z_idx, closest_face],
                    [y_right, z_idx, closest_face],
                    [y_idx, z_up, closest_face],
                    [y_right, z_up, closest_face]
                ])
                
                selection = jnp.array([
                    1.0,
                    include_y_right,
                    include_z_up,
                    include_y_right * include_z_up
                ])
                
                sorted_indices = indices[jnp.argsort(-selection)]
                return jnp.where(jnp.arange(4)[:, None] < jnp.sum(selection), sorted_indices[:4], -1)
            
            def assign_top_bottom():
                # Top/Bottom faces use x,y coordinates
                x_idx = jnp.floor((x + length/2) / length * n_x).astype(jnp.int32)
                y_idx = jnp.floor((y + width/2) / width * n_y).astype(jnp.int32)
                x_idx = jnp.clip(x_idx, 0, n_x - 1)
                y_idx = jnp.clip(y_idx, 0, n_y - 1)
                
                # Calculate overlap with neighboring cells
                x_frac = ((x + length/2) / length * n_x) % 1
                y_frac = ((y + width/2) / width * n_y) % 1
                
                include_x_right = x_frac >= 1 - sensor_radius / (length / n_x)
                include_x_left = x_frac <= sensor_radius / (length / n_x)
                include_y_right = y_frac >= 1 - sensor_radius / (width / n_y)
                include_y_left = y_frac <= sensor_radius / (width / n_y)
                
                x_right = jnp.clip(x_idx + 1, 0, n_x - 1)
                x_left = jnp.clip(x_idx - 1, 0, n_x - 1)
                y_right = jnp.clip(y_idx + 1, 0, n_y - 1)
                y_left = jnp.clip(y_idx - 1, 0, n_y - 1)
                
                indices = jnp.array([
                    [x_idx, y_idx, closest_face],
                    [x_right, y_idx, closest_face],
                    [x_idx, y_right, closest_face],
                    [x_right, y_right, closest_face]
                ])
                
                selection = jnp.array([
                    1.0,
                    include_x_right,
                    include_y_right,
                    include_x_right * include_y_right
                ])
                
                sorted_indices = indices[jnp.argsort(-selection)]
                return jnp.where(jnp.arange(4)[:, None] < jnp.sum(selection), sorted_indices[:4], -1)
            
            return jnp.where(
                (closest_face == 0) | (closest_face == 1),  # Front or Back
                assign_front_back(),
                jnp.where(
                    (closest_face == 2) | (closest_face == 3),  # Left or Right
                    assign_left_right(),
                    assign_top_bottom()  # Top or Bottom
                )
            )
        
        def assign_off_surface():
            return jnp.full((4, 3), -1, dtype=jnp.int32)
        
        return lax.cond(on_surface, assign_surface, assign_off_surface)
    
    return jax.vmap(assign_single_sensor)(sensors)


@partial(jax.jit, static_argnums=(1, 2, 3))
def create_sensor_box_grid_map(assignments, n_x, n_y, n_z):
    """
    Creates a grid map counting the number of sensors in each cell of the box sensor grid.
    """
    # Each face has different grid sizes
    front_back_cells = n_x * n_z
    left_right_cells = n_y * n_z
    top_bottom_cells = n_x * n_y
    
    total_cells = 2 * (front_back_cells + left_right_cells + top_bottom_cells)
    grid = jnp.zeros(total_cells, dtype=jnp.int32)
    
    def update_grid(sensor_assignments):
        def update_cell(cell, g):
            i, j, k = cell
            is_valid = (i != -1) & (j != -1) & (k != -1)
            
            # Calculate linear index based on face
            # Front/Back faces
            face_offset_fb = k * front_back_cells
            idx_fb = face_offset_fb + i * n_z + j
            
            # Left/Right faces
            face_offset_lr = 2 * front_back_cells + (k - 2) * left_right_cells
            idx_lr = face_offset_lr + i * n_z + j
            
            # Top/Bottom faces
            face_offset_tb = 2 * (front_back_cells + left_right_cells) + (k - 4) * top_bottom_cells
            idx_tb = face_offset_tb + i * n_y + j
            
            # Select the appropriate index based on face
            idx = jnp.where(
                k <= 1,
                idx_fb,
                jnp.where(
                    k <= 3,
                    idx_lr,
                    idx_tb
                )
            )
            
            return g.at[idx].add(is_valid)
        
        return jax.lax.fori_loop(0, sensor_assignments.shape[0], 
                                lambda i, g: update_cell(sensor_assignments[i], g), grid)
    
    all_updates = jax.vmap(update_grid)(assignments)
    return all_updates.sum(axis=0)


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5))
def calculate_box_grid_centers(length, width, height, n_x, n_y, n_z):
    """Calculate center points of all box grid cells"""
    centers = []
    
    # Front and Back faces (x,z grid)
    x_positions = (jnp.arange(n_x) - n_x/2 + 0.5) * (length / n_x)
    z_positions = (jnp.arange(n_z) - n_z/2 + 0.5) * (height / n_z)
    
    for face_y in [width/2, -width/2]:  # Front, Back
        x_grid, z_grid = jnp.meshgrid(x_positions, z_positions, indexing='ij')
        face_centers = jnp.stack([
            x_grid.reshape(-1),
            jnp.full(n_x * n_z, face_y),
            z_grid.reshape(-1)
        ], axis=1)
        centers.append(face_centers)
    
    # Left and Right faces (y,z grid)
    y_positions = (jnp.arange(n_y) - n_y/2 + 0.5) * (width / n_y)
    
    for face_x in [-length/2, length/2]:  # Left, Right
        y_grid, z_grid = jnp.meshgrid(y_positions, z_positions, indexing='ij')
        face_centers = jnp.stack([
            jnp.full(n_y * n_z, face_x),
            y_grid.reshape(-1),
            z_grid.reshape(-1)
        ], axis=1)
        centers.append(face_centers)
    
    # Top and Bottom faces (x,y grid)
    y_positions = (jnp.arange(n_y) - n_y/2 + 0.5) * (width / n_y)
    
    for face_z in [height/2, -height/2]:  # Top, Bottom
        x_grid, y_grid = jnp.meshgrid(x_positions, y_positions, indexing='ij')
        face_centers = jnp.stack([
            x_grid.reshape(-1),
            y_grid.reshape(-1),
            jnp.full(n_x * n_y, face_z)
        ], axis=1)
        centers.append(face_centers)
    
    return jnp.concatenate(centers, axis=0)


def find_intersected_box_sensors_differentiable(ray_origins, ray_directions, sensor_positions, sensor_radius,
                                                  length, width, height, n_x, n_y, n_z, 
                                                  inverted_sensor_map, temperature, overlap_prob):
    """
    Finds sensors intersected by rays using a differentiable approximation with overlap-based weights.
    """
    single_ray = ray_origins.ndim == 1
    if single_ray:
        ray_origins = ray_origins[None, :]
        ray_directions = ray_directions[None, :]

    # Get box intersection points and grid indices
    intersects, t_box, face_indices, grid_indices, intersection_point = (
        batch_intersect_box_with_grid(ray_origins, ray_directions, length, width, height, n_x, n_y, n_z))

    def calculate_linear_index(face_indices, grid_indices):
        front_back_cells = n_x * n_z
        left_right_cells = n_y * n_z
        top_bottom_cells = n_x * n_y
        
        # Calculate linear index based on face
        front_back_idx = grid_indices[:, 0] * n_z + grid_indices[:, 1]
        left_right_idx = grid_indices[:, 0] * n_z + grid_indices[:, 1]
        top_bottom_idx = grid_indices[:, 0] * n_y + grid_indices[:, 1]
        
        face_offsets = jnp.array([
            0,  # Front
            front_back_cells,  # Back
            2 * front_back_cells,  # Left
            2 * front_back_cells + left_right_cells,  # Right
            2 * (front_back_cells + left_right_cells),  # Top
            2 * (front_back_cells + left_right_cells) + top_bottom_cells  # Bottom
        ])
        
        idx = jnp.where(
            face_indices <= 1,  # Front/Back
            face_offsets[face_indices] + front_back_idx,
            jnp.where(
                face_indices <= 3,  # Left/Right
                face_offsets[face_indices] + left_right_idx,
                face_offsets[face_indices] + top_bottom_idx  # Top/Bottom
            )
        )
        
        total_cells = 2 * (front_back_cells + left_right_cells + top_bottom_cells)
        return jnp.clip(idx, 0, total_cells - 1)

    idx = calculate_linear_index(face_indices, grid_indices)
    potential_sensors = jax.lax.stop_gradient(inverted_sensor_map[idx])

    # Create bounds check function
    bounds_check = lambda points: box_bounds_check(points, length, width, height)

    # Process all potential sensors
    sensor_results = jax.vmap(
        lambda det_idx: compute_sensor_intersections_base(
            det_idx, sensor_positions, sensor_radius,
            ray_origins, ray_directions, bounds_check, overlap_prob
        )
    )(potential_sensors.T)
    
    weights = sensor_results[0]
    sensor_times = sensor_results[1]
    sensor_indices = sensor_results[2]
    sensor_normals = sensor_results[3]
    inside_detector = sensor_results[4]
    sensor_hit_positions = sensor_results[5]

    # Calculate box face normals
    box_normals = calculate_box_normals(face_indices)

    intersection_results = process_intersection_normals(
        ray_origins, ray_directions, intersection_point,
        t_box, sensor_normals, sensor_hit_positions,
        inside_detector, box_normals
    )

    hit_positions = intersection_results['positions']
    final_normals = intersection_results['normals']

    result = {
        'times': sensor_times,
        'sensor_weights': weights,
        'sensor_indices': sensor_indices,
        'per_sensor_positions': sensor_hit_positions,
        'positions': hit_positions,
        'normals': final_normals,
        'sensor_normals': sensor_normals,
        'inside_detector': inside_detector
    }

    return result if not single_ray else jax.tree_map(lambda x: x[0], result)


def create_inverted_box_sensor_map(assignments_geometric, assignments_distance, 
                                    n_x, n_y, n_z, max_sensors_per_cell):
    """Create inverted sensor map for box geometry with proper grid indexing."""
    front_back_cells = n_x * n_z
    left_right_cells = n_y * n_z
    top_bottom_cells = n_x * n_y
    total_cells = 2 * (front_back_cells + left_right_cells + top_bottom_cells)
    
    inverted_map = jnp.full((total_cells, max_sensors_per_cell), -1, dtype=jnp.int32)
    
    def update_cell(inv_map, i):
        # Update cell i with sensors that geometrically intersect with it
        def add_geometric(carry, j):
            curr_map, curr_count = carry
            
            # Calculate which face and grid position this cell represents
            # Use JAX-compatible conditionals
            
            # Front/Back face calculations
            is_front_back = i < 2 * front_back_cells
            face_idx_fb = jnp.where(i < front_back_cells, 0, 1)
            local_idx_fb = i % front_back_cells
            cell_i_fb = local_idx_fb // n_z
            cell_j_fb = local_idx_fb % n_z
            
            # Left/Right face calculations
            is_left_right = (i >= 2 * front_back_cells) & (i < 2 * front_back_cells + 2 * left_right_cells)
            offset_lr = i - 2 * front_back_cells
            face_idx_lr = jnp.where(offset_lr < left_right_cells, 2, 3)
            local_idx_lr = offset_lr % left_right_cells
            cell_i_lr = local_idx_lr // n_z
            cell_j_lr = local_idx_lr % n_z
            
            # Top/Bottom face calculations
            offset_tb = i - 2 * (front_back_cells + left_right_cells)
            face_idx_tb = jnp.where(offset_tb < top_bottom_cells, 4, 5)
            local_idx_tb = offset_tb % top_bottom_cells
            cell_i_tb = local_idx_tb // n_y
            cell_j_tb = local_idx_tb % n_y
            
            # Select appropriate values based on which face type
            cell_i = jnp.where(is_front_back, cell_i_fb,
                              jnp.where(is_left_right, cell_i_lr, cell_i_tb))
            cell_j = jnp.where(is_front_back, cell_j_fb,
                              jnp.where(is_left_right, cell_j_lr, cell_j_tb))
            cell_k = jnp.where(is_front_back, face_idx_fb,
                              jnp.where(is_left_right, face_idx_lr, face_idx_tb))
            
            # Check if sensor j intersects with this cell
            sensor_assignments = assignments_geometric[j]
            matches = (sensor_assignments[:, 0] == cell_i) & \
                     (sensor_assignments[:, 1] == cell_j) & \
                     (sensor_assignments[:, 2] == cell_k)
            
            cell_matches = jnp.any(matches)
            should_add = cell_matches & (curr_count < max_sensors_per_cell)
            
            new_map = jnp.where(
                should_add,
                curr_map.at[i, curr_count].set(j),
                curr_map
            )
            
            return (new_map, curr_count + should_add), None
        
        # Add geometric intersections
        (new_map, geom_count), _ = jax.lax.scan(
            add_geometric,
            (inv_map, 0),
            jnp.arange(len(assignments_geometric))
        )
        
        # Get closest sensors for this cell
        closest = assignments_distance[i]
        
        # Add closest sensors if there's room
        def add_closest(carry, j):
            curr_map, curr_count = carry
            sensor_idx = closest[j]
            
            # Check for duplicates
            def check_duplicate(k, is_dup):
                return is_dup | (curr_map[i, k] == sensor_idx)
            
            is_duplicate = jax.lax.fori_loop(
                0, curr_count,
                check_duplicate,
                False
            )
            
            # Add if not duplicate and have space
            should_add = (~is_duplicate) & (curr_count < max_sensors_per_cell)
            
            new_map = jnp.where(
                should_add,
                curr_map.at[i, curr_count].set(sensor_idx),
                curr_map
            )
            
            return (new_map, curr_count + should_add), None
        
        # Fill remaining slots with closest sensors
        (final_map, _), _ = jax.lax.scan(
            add_closest,
            (new_map, geom_count),
            jnp.arange(len(closest))
        )
        
        return final_map, None
    
    final_map, _ = jax.lax.scan(
        update_cell,
        inverted_map,
        jnp.arange(total_cells)
    )
    
    return final_map


def create_box_photon_propagator(sensor_positions, sensor_radius, length=4.0, width=4.0, height=6.0,
                                 n_x=50, n_y=50, n_z=50, temperature=0.2, max_sensors_per_cell=4):
    """
    Creates a JIT-compiled function for efficient photon propagation simulation in box geometry.
    """
    
    if temperature is None:
        overlap_prob = create_overlap_prob(temperature, sensor_radius)
    else:
        overlap_prob = create_overlap_prob(temperature * sensor_radius, sensor_radius)

    # Convert sensor positions to JAX array
    sensor_positions_jax = jnp.array(sensor_positions)
    
    # Create sensor grid assignments
    assignments_geometric = assign_sensors_to_box_grid(
        sensor_positions_jax, sensor_radius, length, width, height, n_x, n_y, n_z
    )
    
    # Calculate grid centers for distance-based assignment
    grid_centers = calculate_box_grid_centers(length, width, height, n_x, n_y, n_z)
    assignments_distance = find_closest_sensors(grid_centers, sensor_positions_jax, max_sensors_per_cell)
    
    # Create inverted sensor map
    inverted_sensor_map = create_inverted_box_sensor_map(
        assignments_geometric, assignments_distance, n_x, n_y, n_z, max_sensors_per_cell
    )

    def propagate_photons(photon_origins, photon_directions):
        """
        Box propagation using proper grid-based sensor lookup.
        """
        return find_intersected_box_sensors_differentiable(
            photon_origins, photon_directions, sensor_positions_jax, sensor_radius,
            length, width, height, n_x, n_y, n_z,
            inverted_sensor_map, temperature, overlap_prob
        )

    return jax.jit(propagate_photons)