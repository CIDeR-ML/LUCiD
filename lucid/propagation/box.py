"""
Box (rectangular prism) detector photon propagation functions with optimizations.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import lax

from lucid.propagation.sensor_map import grid_face_cells



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
        Array of shape (n_sensors, 9, 3) containing up to 9 grid cell assignments
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
        
        # Cast to int32 like x_idx/z_idx (and the index components in cylinder.py and sphere.py):
        # under jax_enable_x64 `jnp.argmin` returns int64, which would promote `indices` to int64
        # while assign_off_surface stays int32, and `lax.cond` rejects the mismatched branches.
        closest_face = jnp.argmin(dist_to_faces).astype(jnp.int32)
        min_distance = jnp.min(dist_to_faces)
        
        # Check if sensor is close enough to any face
        on_surface = min_distance <= sensor_radius
        
        def assign_surface():
            def assign_face(u, extent_u, n_u, v, extent_v, n_v):
                # The sensor's own cell on this face, plus each of the 8 neighbours its disk
                # reaches (as cylinder.py does), the central cell first.
                pu = (u + extent_u / 2) / extent_u * n_u
                pv = (v + extent_v / 2) / extent_v * n_v
                iu = jnp.clip(jnp.floor(pu).astype(jnp.int32), 0, n_u - 1)
                iv = jnp.clip(jnp.floor(pv).astype(jnp.int32), 0, n_v - 1)
                fu, fv = pu % 1, pv % 1
                reach_u = sensor_radius / (extent_u / n_u)
                reach_v = sensor_radius / (extent_v / n_v)
                up_u, dn_u = fu >= 1 - reach_u, fu <= reach_u
                up_v, dn_v = fv >= 1 - reach_v, fv <= reach_v
                u_up, u_dn = jnp.clip(iu + 1, 0, n_u - 1), jnp.clip(iu - 1, 0, n_u - 1)
                v_up, v_dn = jnp.clip(iv + 1, 0, n_v - 1), jnp.clip(iv - 1, 0, n_v - 1)
                indices = jnp.array([
                    [iu, iv, closest_face],
                    [u_up, iv, closest_face], [iu, v_up, closest_face], [u_up, v_up, closest_face],
                    [iu, v_dn, closest_face], [u_up, v_dn, closest_face],
                    [u_dn, iv, closest_face], [u_dn, v_up, closest_face], [u_dn, v_dn, closest_face],
                ])
                selection = jnp.array([
                    1.0, up_u, up_v, up_u * up_v, dn_v, up_u * dn_v, dn_u, dn_u * up_v, dn_u * dn_v,
                ])
                sorted_indices = indices[jnp.argsort(-selection)]
                return jnp.where(jnp.arange(9)[:, None] < jnp.sum(selection), sorted_indices, -1)

            def assign_front_back():                  # Front/Back faces use x,z coordinates
                return assign_face(x, length, n_x, z, height, n_z)

            def assign_left_right():                  # Left/Right faces use y,z coordinates
                return assign_face(y, width, n_y, z, height, n_z)

            def assign_top_bottom():                  # Top/Bottom faces use x,y coordinates
                return assign_face(x, length, n_x, y, width, n_y)

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
            return jnp.full((9, 3), -1, dtype=jnp.int32)
        
        return lax.cond(on_surface, assign_surface, assign_off_surface)
    
    return jax.vmap(assign_single_sensor)(sensors)


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


def box_grid_cells(n_x, n_y, n_z):
    """Grid coordinates (i, j, face) of every box cell, by linear index: (total_cells, 3) int32.

    Faces 0-1 are (n_x, n_z), faces 2-3 (n_y, n_z) and faces 4-5 (n_x, n_y), in that order: the
    convention of :func:`assign_sensors_to_box_grid`.
    """
    return np.concatenate([grid_face_cells(n_x, n_z, 0), grid_face_cells(n_x, n_z, 1),
                           grid_face_cells(n_y, n_z, 2), grid_face_cells(n_y, n_z, 3),
                           grid_face_cells(n_x, n_y, 4), grid_face_cells(n_x, n_y, 5)])
