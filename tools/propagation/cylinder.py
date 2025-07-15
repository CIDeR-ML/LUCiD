"""
Cylinder-specific photon propagation functions.
"""

import jax
import jax.numpy as jnp
from functools import partial
from jax import lax

from .base import (
    process_intersection_normals, compute_sensor_intersections_base,
    find_closest_sensors
)
from ..overlap import create_overlap_prob


@jax.jit
def intersect_cylinder_wall(ray_origin, ray_direction, r, h):
    """Calculate intersection of a ray with a cylinder's wall.

    Parameters
    ----------
    ray_origin : jnp.ndarray
        Starting point of the ray [x, y, z]
    ray_direction : jnp.ndarray
        Direction vector of the ray [dx, dy, dz]
    r : float
        Radius of the cylinder
    h : float
        Height of the cylinder

    Returns
    -------
    tuple
        (bool, float) - (whether intersection exists, distance to intersection)
    """
    LARGE = 1e10

    # Quadratic equation coefficients for cylinder intersection
    a = ray_direction[0]**2 + ray_direction[1]**2
    b = 2.0 * (ray_origin[0]*ray_direction[0] + ray_origin[1]*ray_direction[1])
    c = ray_origin[0]**2 + ray_origin[1]**2 - r**2

    discriminant = b**2 - 4*a*c
    epsilon = 1e-6

    def side_branch(_):
        sqrt_disc = jnp.sqrt(jnp.maximum(0.0, discriminant))
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        t1, t2 = jnp.minimum(t1, t2), jnp.maximum(t1, t2)

        valid_t = (t1 > 0) | (t2 > 0)
        t_candidate = jnp.where(t1 > 0, t1, t2)

        ipt = ray_origin + t_candidate * ray_direction
        within_height = jnp.abs(ipt[2]) <= (h / 2)
        intersects_ = (discriminant >= -epsilon) & valid_t & within_height

        tval_ = jnp.where(intersects_, t_candidate, LARGE)
        return (intersects_, tval_)

    def parallel_side_branch(_):
        # Direction is purely along z => no side intersection
        return (False, jnp.array(LARGE, dtype=jnp.float32))

    use_parallel = jnp.abs(a) < 1e-12
    intersects, tval = lax.cond(use_parallel,
                                parallel_side_branch,
                                side_branch,
                                operand=None)

    return intersects, tval


@jax.jit
def intersect_cylinder_cap(ray_origin, ray_direction, r, z):
    """Calculate intersection of a ray with one of the cylinder's caps.

    Parameters
    ----------
    ray_origin : jnp.ndarray
        Starting point of the ray [x, y, z]
    ray_direction : jnp.ndarray
        Direction vector of the ray [dx, dy, dz]
    r : float
        Radius of the cylinder
    z : float
        Z-coordinate of the cap plane

    Returns
    -------
    tuple
        (bool, float) - (whether intersection exists, distance to intersection)
    """
    LARGE = 1e10

    def normal_cap_branch(_):
        t_plane = (z - ray_origin[2]) / ray_direction[2]
        ipt = ray_origin + t_plane * ray_direction
        within_circle = (ipt[0]**2 + ipt[1]**2) <= r**2

        intersects_ = (t_plane > 0) & within_circle
        tval_ = jnp.where(intersects_, t_plane, LARGE)
        return (intersects_, tval_)

    def parallel_cap_branch(_):
        # If dz == 0 => parallel. Check if we're exactly on the plane
        same_plane = jnp.abs(ray_origin[2] - z) < 1e-12
        intersects_ = same_plane
        tval_ = jnp.where(same_plane, 0.0, LARGE)
        return (intersects_, tval_)

    use_parallel = jnp.abs(ray_direction[2]) < 1e-12
    intersects, tval = lax.cond(use_parallel,
                                parallel_cap_branch,
                                normal_cap_branch,
                                operand=None)

    return intersects, tval


@jax.jit
def intersect_cylinder(ray_origin, ray_direction, r, h):
    """Find the closest intersection point with any part of the cylinder.

    Parameters
    ----------
    ray_origin : jnp.ndarray
        Starting point of the ray [x, y, z]
    ray_direction : jnp.ndarray
        Direction vector of the ray [dx, dy, dz]
    r : float
        Radius of the cylinder
    h : float
        Height of the cylinder

    Returns
    -------
    tuple
        (bool, float, int) - (whether intersection exists, distance, part index)
        part index: 0=wall, 1=top cap, 2=bottom cap
    """
    wall_intersects, wall_t = intersect_cylinder_wall(ray_origin, ray_direction, r, h)
    top_intersects, top_t = intersect_cylinder_cap(ray_origin, ray_direction, r, h / 2)
    bottom_intersects, bottom_t = intersect_cylinder_cap(ray_origin, ray_direction, r, -h / 2)

    # Combine them into shape (3,) arrays
    ts = jnp.stack([wall_t, top_t, bottom_t], axis=0)
    intersects = jnp.stack([wall_intersects, top_intersects, bottom_intersects], axis=0)

    min_t_index = jnp.argmin(ts)
    min_t = jnp.min(ts)
    any_intersects = jnp.any(intersects)

    return any_intersects, min_t, min_t_index


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def intersect_cylinder_with_grid(ray_origin, ray_direction, r, h, n_cap, n_angular, n_height):
    """Find intersection with cylinder and compute grid cell indices.

    Parameters
    ----------
    ray_origin : jnp.ndarray
        Starting point of the ray [x, y, z]
    ray_direction : jnp.ndarray
        Direction vector of the ray [dx, dy, dz]
    r : float
        Radius of the cylinder
    h : float
        Height of the cylinder
    n_cap : int
        Number of grid cells along each dimension of the cap
    n_angular : int
        Number of angular divisions for the wall
    n_height : int
        Number of vertical divisions for the wall

    Returns
    -------
    tuple
        (intersects, t, is_wall, is_top_cap, wall_indices, cap_indices, intersection_point)
    """
    intersects, t, part = intersect_cylinder(ray_origin, ray_direction, r, h)
    intersection_point = ray_origin + t * ray_direction

    # Calculate wall grid indices using polar coordinates
    angle = jnp.arctan2(intersection_point[1], intersection_point[0]) % (2 * jnp.pi)
    angular_idx = jnp.floor(angle / (2 * jnp.pi) * n_angular).astype(jnp.int32)
    angular_idx = angular_idx % n_angular  # Handle wraparound for angular dimension
    height_idx = jnp.floor((intersection_point[2] + h / 2) / h * n_height).astype(jnp.int32)
    height_idx = jnp.clip(height_idx, 0, n_height - 1)  # Clamp to valid range

    # Calculate cap grid indices using Cartesian coordinates
    cap_x = (intersection_point[0] + r) / (2 * r)
    cap_y = (intersection_point[1] + r) / (2 * r)
    cap_x_idx = jnp.floor(cap_x * n_cap).astype(jnp.int32)
    cap_y_idx = jnp.floor(cap_y * n_cap).astype(jnp.int32)
    cap_x_idx = jnp.clip(cap_x_idx, 0, n_cap - 1)  # Clamp to valid range
    cap_y_idx = jnp.clip(cap_y_idx, 0, n_cap - 1)  # Clamp to valid range

    wall_indices = jnp.array([angular_idx, height_idx])
    cap_indices = jnp.array([cap_x_idx, cap_y_idx])

    is_wall = part == 0
    is_top_cap = part == 1

    return intersects, t, is_wall, is_top_cap, wall_indices, cap_indices, intersection_point


batch_intersect_cylinder_with_grid = jax.vmap(intersect_cylinder_with_grid,
                                              in_axes=(0, 0, None, None, None, None, None))


def calculate_cylinder_normals(intersection_point, is_wall, is_top_cap):
    """
    Calculate normals for cylinder surfaces.
    
    Parameters
    ----------
    intersection_point : ndarray
        Points of intersection on the cylinder
    is_wall : ndarray
        Boolean array indicating wall hits
    is_top_cap : ndarray
        Boolean array indicating top cap hits
        
    Returns
    -------
    ndarray
        Normal vectors for cylinder intersections
    """
    # Wall normals
    wall_normals = -intersection_point[:, :2] / (
            jnp.linalg.norm(intersection_point[:, :2], axis=1, keepdims=True) + 1e-10)
    wall_normals = jnp.concatenate([wall_normals, jnp.zeros_like(intersection_point[:, :1])], axis=1)
    
    # Cap normals
    top_cap_normal = jnp.array([0., 0., 1.])
    bottom_cap_normal = jnp.array([0., 0., -1.])
    
    # Select appropriate normal
    return jnp.where(is_wall[:, None],
                    wall_normals,
                    jnp.where(is_top_cap[:, None],
                             top_cap_normal,
                             bottom_cap_normal))


def cylinder_bounds_check(points, r, h):
    """
    Check if points are within cylinder bounds.
    
    Parameters
    ----------
    points : ndarray
        Points to check [x, y, z]
    r : float
        Cylinder radius
    h : float
        Cylinder height
        
    Returns
    -------
    ndarray
        Boolean array indicating which points are inside cylinder
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # For x,y: check if point is within circle with radius r
    inside_xy_circle = (x**2 + y**2) <= r**2
    # For z: check if |z| ≤ h/2
    inside_z_bounds = (z >= -h/2) & (z <= h/2)
    
    return inside_xy_circle & inside_z_bounds


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def assign_sensors_to_grid(sensors, sensor_radius, r, h, n_cap, n_angular, n_height):
    """Assign sensors to grid cells, handling overlap across cell boundaries.

    Parameters
    ----------
    sensors : jnp.ndarray
        Array of sensor positions, shape (n_sensors, 3)
    sensor_radius : float
        Radius of each sensor
    r : float
        Cylinder radius
    h : float
        Cylinder height
    n_cap : int
        Grid resolution for caps
    n_angular : int
        Number of angular divisions
    n_height : int
        Number of height divisions

    Returns
    -------
    jnp.ndarray
        Array of shape (n_sensors, 4, 3) containing up to 4 grid cell assignments
        per sensor. -1 indicates no assignment.
    """

    def assign_single_sensor(sensor):
        x, y, z = sensor

        # Convert to cylindrical coordinates
        radius = jnp.sqrt(x ** 2 + y ** 2)
        angle = jnp.arctan2(y, x) % (2 * jnp.pi)

        # Determine sensor location (wall or cap)
        on_wall = jnp.abs(radius - r) <= sensor_radius
        on_top = z > h / 2 - sensor_radius
        on_bottom = z < -h / 2 + sensor_radius

        def assign_wall():
            wall_angle = angle
            wall_height = jnp.clip(z, -h / 2, h / 2)

            angular_idx = jnp.floor(wall_angle / (2 * jnp.pi) * n_angular).astype(jnp.int32)
            angular_idx = angular_idx % n_angular  # Handle wraparound for angular dimension
            height_idx = jnp.floor((wall_height + h / 2) / h * n_height).astype(jnp.int32)
            height_idx = jnp.clip(height_idx, 0, n_height - 1)  # Clamp to valid range

            # Calculate overlap with neighboring cells
            angular_frac = (wall_angle / (2 * jnp.pi) * n_angular) % 1
            height_frac = ((wall_height + h / 2) / h * n_height) % 1

            include_right = angular_frac >= 1 - sensor_radius / (2 * jnp.pi * r / n_angular)
            include_left = angular_frac <= sensor_radius / (2 * jnp.pi * r / n_angular)
            include_top = height_frac >= 1 - sensor_radius / (h / n_height)
            include_bottom = height_frac <= sensor_radius / (h / n_height)

            # Calculate valid height neighbors (don't wrap at boundaries)
            height_up = jnp.clip(height_idx + 1, 0, n_height - 1)
            height_down = jnp.clip(height_idx - 1, 0, n_height - 1)
            
            indices = jnp.array([
                [angular_idx, height_idx, 0],
                [(angular_idx + 1) % n_angular, height_idx, 0],
                [angular_idx, height_up, 0],
                [(angular_idx + 1) % n_angular, height_up, 0],
                [angular_idx, height_down, 0],
                [(angular_idx + 1) % n_angular, height_down, 0],
                [(angular_idx - 1) % n_angular, height_idx, 0],
                [(angular_idx - 1) % n_angular, height_up, 0],
                [(angular_idx - 1) % n_angular, height_down, 0]
            ])

            selection = jnp.array([
                1.0,  # Central cell always included
                include_right,
                include_top,
                include_right * include_top,
                include_bottom,
                include_right * include_bottom,
                include_left,
                include_left * include_top,
                include_left * include_bottom
            ])

            sorted_indices = indices[jnp.argsort(-selection)]

            return jnp.where(jnp.arange(4)[:, None] < jnp.sum(selection), sorted_indices[:4], -1)

        def assign_cap(is_top):
            cap_x = x
            cap_y = y

            x_idx = jnp.floor((cap_x + r) / (2 * r) * n_cap).astype(jnp.int32)
            x_idx = jnp.clip(x_idx, 0, n_cap - 1)  # Clamp to valid range
            y_idx = jnp.floor((cap_y + r) / (2 * r) * n_cap).astype(jnp.int32)
            y_idx = jnp.clip(y_idx, 0, n_cap - 1)  # Clamp to valid range

            # Calculate overlap with neighboring cells
            x_frac = ((cap_x + r) / (2 * r) * n_cap) % 1
            y_frac = ((cap_y + r) / (2 * r) * n_cap) % 1

            include_right = x_frac >= 1 - sensor_radius / (2 * r / n_cap)
            include_left = x_frac <= sensor_radius / (2 * r / n_cap)
            include_top = y_frac >= 1 - sensor_radius / (2 * r / n_cap)
            include_bottom = y_frac <= sensor_radius / (2 * r / n_cap)

            # Calculate valid cap neighbors (don't wrap at boundaries)
            x_right = jnp.clip(x_idx + 1, 0, n_cap - 1)
            x_left = jnp.clip(x_idx - 1, 0, n_cap - 1)
            y_up = jnp.clip(y_idx + 1, 0, n_cap - 1)
            y_down = jnp.clip(y_idx - 1, 0, n_cap - 1)
            
            indices = jnp.array([
                [x_idx, y_idx, 1 if is_top else 2],
                [x_right, y_idx, 1 if is_top else 2],
                [x_idx, y_up, 1 if is_top else 2],
                [x_right, y_up, 1 if is_top else 2],
                [x_idx, y_down, 1 if is_top else 2],
                [x_right, y_down, 1 if is_top else 2],
                [x_left, y_idx, 1 if is_top else 2],
                [x_left, y_up, 1 if is_top else 2],
                [x_left, y_down, 1 if is_top else 2]
            ])

            selection = jnp.array([
                1.0,
                include_right,
                include_top,
                include_right * include_top,
                include_bottom,
                include_right * include_bottom,
                include_left,
                include_left * include_top,
                include_left * include_bottom
            ])

            sorted_indices = indices[jnp.argsort(-selection)]

            return jnp.where(jnp.arange(4)[:, None] < jnp.sum(selection), sorted_indices[:4], -1)

        return jax.lax.cond(
            on_wall,
            assign_wall,
            lambda: jax.lax.cond(
                on_top,
                lambda: assign_cap(True),
                lambda: jax.lax.cond(
                    on_bottom,
                    lambda: assign_cap(False),
                    lambda: jnp.full((4, 3), -1, dtype=jnp.int32)
                )
            )
        )

    return jax.vmap(assign_single_sensor)(sensors)


@partial(jax.jit, static_argnums=(1, 2, 3))
def create_sensor_grid_map(assignments, n_cap, n_angular, n_height):
    """
    Creates a grid map counting the number of sensors in each cell of the sensor grid.
    """
    # Calculate grid size: wall cells + cells in both caps
    total_cells = n_angular * n_height + 2 * n_cap * n_cap
    grid = jnp.zeros(total_cells, dtype=jnp.int32)

    def update_grid(sensor_assignments):
        def update_cell(cell, g):
            i, j, k = cell
            is_valid = (i != -1) & (j != -1) & (k != -1)

            # Calculate linear index for either wall (k=0) or cap cells (k=1,2)
            idx = jnp.where(k == 0,
                            i * n_height + j,  # Wall cell indexing
                            n_angular * n_height + (k - 1) * n_cap * n_cap + i * n_cap + j)  # Cap cell indexing

            return g.at[idx].add(is_valid)

        return jax.lax.fori_loop(0, sensor_assignments.shape[0], lambda i, g: update_cell(sensor_assignments[i], g),
                                 grid)

    all_updates = jax.vmap(update_grid)(assignments)
    return all_updates.sum(axis=0)


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def calculate_grid_centers(r, h, n_cap, n_angular, n_height):
    """Calculate center points of all grid cells"""
    # Wall centers
    angular_step = 2 * jnp.pi / n_angular
    height_step = h / n_height

    angular_centers = (jnp.arange(n_angular) + 0.5) * angular_step
    height_centers = (jnp.arange(n_height) - n_height / 2 + 0.5) * height_step

    ang_grid, h_grid = jnp.meshgrid(angular_centers, height_centers, indexing='ij')

    wall_x = r * jnp.cos(ang_grid)
    wall_y = r * jnp.sin(ang_grid)
    wall_centers = jnp.stack([
        wall_x.reshape(-1),
        wall_y.reshape(-1),
        h_grid.reshape(-1)
    ], axis=1)

    # Cap centers
    cap_step = 2 * r / n_cap
    cap_positions = (jnp.arange(n_cap) - n_cap / 2 + 0.5) * cap_step
    x_grid, y_grid = jnp.meshgrid(cap_positions, cap_positions, indexing='ij')

    # Top and bottom cap centers
    top_z = jnp.full(n_cap * n_cap, h / 2)
    bottom_z = jnp.full(n_cap * n_cap, -h / 2)

    cap_centers = jnp.concatenate([
        jnp.stack([
            x_grid.reshape(-1),
            y_grid.reshape(-1),
            top_z
        ], axis=1),
        jnp.stack([
            x_grid.reshape(-1),
            y_grid.reshape(-1),
            bottom_z
        ], axis=1)
    ], axis=0)

    return jnp.concatenate([wall_centers, cap_centers], axis=0)


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6), device=jax.devices('cpu')[0])
def create_inverted_sensor_map(assignments_geometric, assignments_distance, n_cap, n_angular, n_height,
                                 max_sensors_per_cell, num_sensors):
    """Create inverted sensor map prioritizing geometric intersections then closest sensors"""
    total_cells = n_angular * n_height + 2 * n_cap * n_cap

    # Initialize map
    inverted_map = jnp.full((total_cells, max_sensors_per_cell), -1, dtype=jnp.int32)

    def update_cell(carry, i):
        inv_map = carry

        def add_geometric(carry, j):
            curr_map, curr_count = carry

            # Convert linear index i back to 3D coordinates using jnp.where instead of if/else
            is_wall_cell = i < n_angular * n_height
            is_top_cap_cell = (i >= n_angular * n_height) & (i < n_angular * n_height + n_cap * n_cap)

            # Wall cell calculations
            wall_i = i // n_height
            wall_j = i % n_height

            # Cap cell calculations
            cap_offset = i - n_angular * n_height
            cap_idx = cap_offset % (n_cap * n_cap)
            cap_i = cap_idx // n_cap
            cap_j = cap_idx % n_cap

            # Select correct indices based on cell type
            cell_i = jnp.where(is_wall_cell, wall_i, cap_i)
            cell_j = jnp.where(is_wall_cell, wall_j, cap_j)
            cell_k = jnp.where(is_wall_cell,
                               0,
                               jnp.where(is_top_cap_cell, 1, 2))

            # Check if sensor j intersects with this cell in 3D coordinates
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


def find_intersected_sensors_differentiable(ray_origins, ray_directions, sensor_positions, sensor_radius, r, h,
                                           n_cap, n_angular, n_height, inverted_sensor_map,
                                           temperature, overlap_prob):
    """
    Finds sensors intersected by rays using a differentiable approximation with overlap-based weights.
    """
    single_ray = ray_origins.ndim == 1
    if single_ray:
        ray_origins = ray_origins[None, :]
        ray_directions = ray_directions[None, :]

    # Get cylinder intersection points and grid indices
    intersects, t_cylinder, is_wall, is_top_cap, wall_indices, cap_indices, intersection_point = (
        batch_intersect_cylinder_with_grid(ray_origins, ray_directions, r, h, n_cap, n_angular, n_height))

    def calculate_linear_index(wall_indices, cap_indices, is_wall, is_top_cap):
        wall_linear = jnp.clip(wall_indices[:, 0] * n_height + wall_indices[:, 1],
                               0, n_angular * n_height - 1)
        cap_linear = jnp.clip(cap_indices[:, 0] * n_cap + cap_indices[:, 1],
                              0, n_cap * n_cap - 1)
        idx = jnp.where(is_wall,
                        wall_linear,
                        jnp.where(is_top_cap,
                                  n_angular * n_height + cap_linear,
                                  n_angular * n_height + n_cap * n_cap + cap_linear))
        total_cells = n_angular * n_height + 2 * n_cap * n_cap
        return jnp.clip(idx, 0, total_cells - 1)

    idx = calculate_linear_index(wall_indices, cap_indices, is_wall, is_top_cap)
    potential_sensors = jax.lax.stop_gradient(inverted_sensor_map[idx])

    # Create bounds check function
    bounds_check = lambda points: cylinder_bounds_check(points, r, h)

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

    # Calculate cylinder normals
    cylinder_normals = calculate_cylinder_normals(intersection_point, is_wall, is_top_cap)

    intersection_results = process_intersection_normals(
        ray_origins, ray_directions, intersection_point,
        t_cylinder, sensor_normals, sensor_hit_positions,
        inside_detector, cylinder_normals
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


def create_photon_propagator(sensor_positions, sensor_radius, r=4.0, h=6.0, n_cap=150, n_angular=250, n_height=150,
                           temperature=0.2, max_sensors_per_cell=4):
    """
    Creates a JIT-compiled function for efficient photon propagation simulation with overlap-based weights.
    """
    assignments_geometric = assign_sensors_to_grid(
        sensor_positions, sensor_radius, r, h, n_cap, n_angular, n_height)

    sensor_grid_map = create_sensor_grid_map(
        assignments_geometric, n_cap, n_angular, n_height)

    assignments_distance = find_closest_sensors(
        calculate_grid_centers(r, h, n_cap, n_angular, n_height),
        sensor_positions,
        max_sensors_per_cell
    )

    inverted_sensor_map = create_inverted_sensor_map(
        assignments_geometric,
        assignments_distance,
        n_cap, n_angular, n_height,
        max_sensors_per_cell, sensor_positions.shape[0]
    )

    if temperature is None:
        overlap_prob = create_overlap_prob(temperature, sensor_radius)
    else:
        # Create overlap probability function
        overlap_prob = create_overlap_prob(temperature * sensor_radius, sensor_radius)

    @jax.jit
    def propagate_photons(photon_origins, photon_directions):
        return find_intersected_sensors_differentiable(
            photon_origins, photon_directions, sensor_positions, sensor_radius,
            r, h, n_cap, n_angular, n_height, inverted_sensor_map,
            temperature, overlap_prob)

    return propagate_photons