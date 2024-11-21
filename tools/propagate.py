import jax
import jax.numpy as jnp
from functools import partial

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
    # Quadratic equation coefficients for cylinder intersection
    a = ray_direction[0] ** 2 + ray_direction[1] ** 2
    b = 2 * (ray_origin[0] * ray_direction[0] + ray_origin[1] * ray_direction[1])
    c = ray_origin[0] ** 2 + ray_origin[1] ** 2 - r ** 2

    discriminant = b ** 2 - 4 * a * c

    epsilon = 1e-6

    t1 = (-b - jnp.sqrt(jnp.maximum(0, discriminant))) / (2 * a)
    t2 = (-b + jnp.sqrt(jnp.maximum(0, discriminant))) / (2 * a)

    t1, t2 = jnp.minimum(t1, t2), jnp.maximum(t1, t2)

    valid_t = (t1 > 0) | (t2 > 0)

    t = jnp.where(t1 > 0, t1, t2)

    intersection_point = ray_origin + t * ray_direction
    within_height = jnp.abs(intersection_point[2]) <= h / 2

    intersects = (discriminant >= -epsilon) & valid_t & within_height
    return intersects, jnp.where(intersects, t, jnp.inf)


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
    t = jnp.where(ray_direction[2] != 0, (z - ray_origin[2]) / ray_direction[2], jnp.inf)

    intersection_point = ray_origin + t * ray_direction
    within_circle = (intersection_point[0] ** 2 + intersection_point[1] ** 2) <= r ** 2

    intersects = (t > 0) & within_circle
    return intersects, jnp.where(intersects, t, jnp.inf)


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

    ts = jnp.array([wall_t, top_t, bottom_t])
    intersects = jnp.array([wall_intersects, top_intersects, bottom_intersects])

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
    height_idx = jnp.floor((intersection_point[2] + h / 2) / h * n_height).astype(jnp.int32)

    # Calculate cap grid indices using Cartesian coordinates
    cap_x = (intersection_point[0] + r) / (2 * r)
    cap_y = (intersection_point[1] + r) / (2 * r)
    cap_x_idx = jnp.floor(cap_x * n_cap).astype(jnp.int32)
    cap_y_idx = jnp.floor(cap_y * n_cap).astype(jnp.int32)

    wall_indices = jnp.array([angular_idx, height_idx])
    cap_indices = jnp.array([cap_x_idx, cap_y_idx])

    is_wall = part == 0
    is_top_cap = part == 1

    return intersects, t, is_wall, is_top_cap, wall_indices, cap_indices, intersection_point


batch_intersect_cylinder_with_grid = jax.vmap(intersect_cylinder_with_grid,
                                              in_axes=(0, 0, None, None, None, None, None))


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def assign_detectors_to_grid(detectors, detector_radius, r, h, n_cap, n_angular, n_height):
    """Assign detectors to grid cells, handling overlap across cell boundaries.

    Parameters
    ----------
    detectors : jnp.ndarray
        Array of detector positions, shape (n_detectors, 3)
    detector_radius : float
        Radius of each detector
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
        Array of shape (n_detectors, 4, 3) containing up to 4 grid cell assignments
        per detector. -1 indicates no assignment.
    """

    def assign_single_detector(detector):
        x, y, z = detector

        # Convert to cylindrical coordinates
        radius = jnp.sqrt(x ** 2 + y ** 2)
        angle = jnp.arctan2(y, x) % (2 * jnp.pi)

        # Determine detector location (wall or cap)
        on_wall = jnp.abs(radius - r) <= detector_radius
        on_top = z > h / 2 - detector_radius
        on_bottom = z < -h / 2 + detector_radius

        def assign_wall():
            wall_angle = angle
            wall_height = jnp.clip(z, -h / 2, h / 2)

            angular_idx = jnp.floor(wall_angle / (2 * jnp.pi) * n_angular).astype(jnp.int32)
            height_idx = jnp.floor((wall_height + h / 2) / h * n_height).astype(jnp.int32)

            # Calculate overlap with neighboring cells
            angular_frac = (wall_angle / (2 * jnp.pi) * n_angular) % 1
            height_frac = ((wall_height + h / 2) / h * n_height) % 1

            include_right = angular_frac >= 1 - detector_radius / (2 * jnp.pi * r / n_angular)
            include_left = angular_frac <= detector_radius / (2 * jnp.pi * r / n_angular)
            include_top = height_frac >= 1 - detector_radius / (h / n_height)
            include_bottom = height_frac <= detector_radius / (h / n_height)

            indices = jnp.array([
                [angular_idx, height_idx, 0],
                [(angular_idx + 1) % n_angular, height_idx, 0],
                [angular_idx, (height_idx + 1) % n_height, 0],
                [(angular_idx + 1) % n_angular, (height_idx + 1) % n_height, 0],
                [angular_idx, (height_idx - 1) % n_height, 0],
                [(angular_idx + 1) % n_angular, (height_idx - 1) % n_height, 0],
                [(angular_idx - 1) % n_angular, height_idx, 0],
                [(angular_idx - 1) % n_angular, (height_idx + 1) % n_height, 0],
                [(angular_idx - 1) % n_angular, (height_idx - 1) % n_height, 0]
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
            y_idx = jnp.floor((cap_y + r) / (2 * r) * n_cap).astype(jnp.int32)

            # Calculate overlap with neighboring cells
            x_frac = ((cap_x + r) / (2 * r) * n_cap) % 1
            y_frac = ((cap_y + r) / (2 * r) * n_cap) % 1

            include_right = x_frac >= 1 - detector_radius / (2 * r / n_cap)
            include_left = x_frac <= detector_radius / (2 * r / n_cap)
            include_top = y_frac >= 1 - detector_radius / (2 * r / n_cap)
            include_bottom = y_frac <= detector_radius / (2 * r / n_cap)

            indices = jnp.array([
                [x_idx, y_idx, 1 if is_top else 2],
                [(x_idx + 1) % n_cap, y_idx, 1 if is_top else 2],
                [x_idx, (y_idx + 1) % n_cap, 1 if is_top else 2],
                [(x_idx + 1) % n_cap, (y_idx + 1) % n_cap, 1 if is_top else 2],
                [x_idx, (y_idx - 1) % n_cap, 1 if is_top else 2],
                [(x_idx + 1) % n_cap, (y_idx - 1) % n_cap, 1 if is_top else 2],
                [(x_idx - 1) % n_cap, y_idx, 1 if is_top else 2],
                [(x_idx - 1) % n_cap, (y_idx + 1) % n_cap, 1 if is_top else 2],
                [(x_idx - 1) % n_cap, (y_idx - 1) % n_cap, 1 if is_top else 2]
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

    return jax.vmap(assign_single_detector)(detectors)


@partial(jax.jit, static_argnums=(1, 2, 3))
def create_detector_grid_map(assignments, n_cap, n_angular, n_height):
    """
    Creates a grid map counting the number of detectors in each cell of the detector grid.

    Parameters
    ----------
    assignments : ndarray
        Array of detector assignments to grid cells, where each assignment is (i, j, k)
        coordinates representing angular, height, and cap positions
    n_cap : int
        Number of cells along each dimension in the cap regions
    n_angular : int
        Number of angular divisions in the cylindrical wall
    n_height : int
        Number of height divisions in the cylindrical wall

    Returns
    -------
    ndarray
        1D array containing detector counts for each cell in the grid
    """
    # Calculate grid size: wall cells + cells in both caps
    total_cells = n_angular * n_height + 2 * n_cap * n_cap
    grid = jnp.zeros(total_cells, dtype=jnp.int32)

    def update_grid(detector_assignments):
        def update_cell(cell, g):
            i, j, k = cell
            is_valid = (i != -1) & (j != -1) & (k != -1)

            # Calculate linear index for either wall (k=0) or cap cells (k=1,2)
            idx = jnp.where(k == 0,
                            i * n_height + j,  # Wall cell indexing
                            n_angular * n_height + (k - 1) * n_cap * n_cap + i * n_cap + j)  # Cap cell indexing

            return g.at[idx].add(is_valid)

        return jax.lax.fori_loop(0, detector_assignments.shape[0], lambda i, g: update_cell(detector_assignments[i], g),
                                 grid)

    all_updates = jax.vmap(update_grid)(assignments)
    return all_updates.sum(axis=0)


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def assign_and_map_detectors(detectors, detector_radius, r, h, n_cap, n_angular, n_height):
    """
    Assigns detectors to grid cells and creates a detector density map.

    Parameters
    ----------
    detectors : ndarray
        Array of detector positions
    detector_radius : float
        Radius of each detector
    r : float
        Cylinder radius
    h : float
        Cylinder height
    n_cap : int
        Number of cells along each dimension in cap regions
    n_angular : int
        Number of angular divisions
    n_height : int
        Number of height divisions

    Returns
    -------
    tuple
        (detector assignments, detector grid map)
    """
    assignments = assign_detectors_to_grid(detectors, detector_radius, r, h, n_cap, n_angular, n_height)
    detector_grid_map = create_detector_grid_map(assignments, n_cap, n_angular, n_height)
    return assignments, detector_grid_map

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


@partial(jax.jit, static_argnums=(2,))
def find_closest_detectors(grid_centers, detector_positions, max_detectors_per_cell):
    """Find closest detectors to each grid cell center"""
    squared_distances = jnp.sum(
        (grid_centers[:, None, :] - detector_positions[None, :, :]) ** 2,
        axis=2
    )
    return jax.lax.top_k(-squared_distances, max_detectors_per_cell)[1]


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6), device=jax.devices('cpu')[0])
def create_inverted_detector_map(assignments_geometric, assignments_distance, n_cap, n_angular, n_height,
                                 max_detectors_per_cell, num_detectors):
    """Create inverted detector map prioritizing geometric intersections then closest detectors"""
    total_cells = n_angular * n_height + 2 * n_cap * n_cap

    # Initialize map
    inverted_map = jnp.full((total_cells, max_detectors_per_cell), -1, dtype=jnp.int32)

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

            # Check if detector j intersects with this cell in 3D coordinates
            detector_assignments = assignments_geometric[j]
            matches = (detector_assignments[:, 0] == cell_i) & \
                      (detector_assignments[:, 1] == cell_j) & \
                      (detector_assignments[:, 2] == cell_k)

            cell_matches = jnp.any(matches)
            should_add = cell_matches & (curr_count < max_detectors_per_cell)

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

        # Get closest detectors for this cell
        closest = assignments_distance[i]

        # Add closest detectors if there's room
        def add_closest(carry, j):
            curr_map, curr_count = carry
            detector_idx = closest[j]

            # Check for duplicates
            def check_duplicate(k, is_dup):
                return is_dup | (curr_map[i, k] == detector_idx)

            is_duplicate = jax.lax.fori_loop(
                0, curr_count,
                check_duplicate,
                False
            )

            # Add if not duplicate and have space
            should_add = (~is_duplicate) & (curr_count < max_detectors_per_cell)

            new_map = jnp.where(
                should_add,
                curr_map.at[i, curr_count].set(detector_idx),
                curr_map
            )

            return (new_map, curr_count + should_add), None

        # Fill remaining slots with closest detectors
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


def find_intersected_detectors_differentiable(ray_origins, ray_directions, detector_positions, detector_radius, r, h,
                                              n_cap, n_angular, n_height, inverted_detector_map,
                                              temperature):
    """
    Finds detectors intersected by rays using a differentiable approximation with temperature-aware minimum weights.

    Parameters
    ----------
    ray_origins : ndarray
        Starting points of rays
    ray_directions : ndarray
        Direction vectors of rays
    detector_positions : ndarray
        Positions of all detectors
    detector_radius : float
        Radius of each detector
    r : float
        Cylinder radius
    h : float
        Cylinder height
    n_cap : int
        Number of cells along each dimension in cap regions
    n_angular : int
        Number of angular divisions
    n_height : int
        Number of height divisions
    inverted_detector_map : ndarray
        Mapping from grid cells to detector indices
    temperature : float
        Smoothing scale for differentiable approximation
        Higher values will make the approximation closer to a hard assignment

    Returns
    -------
    dict
        Contains intersection times, positions, weights, and indices
    """
    single_ray = ray_origins.ndim == 1
    if single_ray:
        ray_origins = ray_origins[None, :]
        ray_directions = ray_directions[None, :]

    # Get cylinder intersection points and grid indices
    intersects, t_cylinder, is_wall, is_top_cap, wall_indices, cap_indices, intersection_point = jax.lax.stop_gradient(
        batch_intersect_cylinder_with_grid(ray_origins, ray_directions, r, h, n_cap, n_angular, n_height)
    )

    # Convert to linear indices for wall and cap regions
    wall_idx = wall_indices[:, 0] * n_height + wall_indices[:, 1]
    cap_idx = cap_indices[:, 0] * n_cap + cap_indices[:, 1]

    def calculate_linear_index(wall_indices, cap_indices, is_wall, is_top_cap):
        # Wall indexing
        wall_linear = jnp.clip(wall_indices[:, 0] * n_height + wall_indices[:, 1],
                               0, n_angular * n_height - 1)

        # Cap indexing
        cap_linear = jnp.clip(cap_indices[:, 0] * n_cap + cap_indices[:, 1],
                              0, n_cap * n_cap - 1)

        # Combine with base offsets
        idx = jnp.where(is_wall,
                        wall_linear,
                        jnp.where(is_top_cap,
                                  n_angular * n_height + cap_linear,
                                  n_angular * n_height + n_cap * n_cap + cap_linear))

        # Final bounds check
        total_cells = n_angular * n_height + 2 * n_cap * n_cap
        return jnp.clip(idx, 0, total_cells - 1)

    # Use this function instead of direct indexing
    idx = calculate_linear_index(wall_indices, cap_indices, is_wall, is_top_cap)

    potential_detectors = jax.lax.stop_gradient(inverted_detector_map[idx])

    def compute_detector_intersections(detector_idx):
        valid = detector_idx != -1
        sphere_centers = jnp.where(valid[:, None], detector_positions[detector_idx], jnp.zeros(3))

        # Find closest approach of ray to detector center
        oc = ray_origins - sphere_centers
        ray_d = ray_directions / (jnp.linalg.norm(ray_directions, axis=1, keepdims=True) + 1e-10)
        t = -jnp.sum(oc * ray_d, axis=1, keepdims=True)
        closest = ray_origins + t * ray_d

        # Calculate smoothed intersection weights with temperature-aware minimum
        to_detector = closest - sphere_centers
        distance = jnp.linalg.norm(to_detector, axis=1)
        min_distance = jnp.maximum(distance - detector_radius, 0.0)

        # Temperature-aware weighting function
        base_weight = detector_radius / (detector_radius + (min_distance * temperature) ** 2)

        raw_weights = jnp.where(valid,
                                base_weight,
                                0.0)

        return raw_weights, t, detector_idx

    detector_results = jax.vmap(compute_detector_intersections)(potential_detectors.T)
    raw_weights = detector_results[0]
    detector_times = detector_results[1]
    detector_indices = detector_results[2]

    # Normalize weights only when their sum exceeds 1
    sum_weights = jnp.sum(raw_weights, axis=0) + 1e-10
    final_weights = jnp.where(
        sum_weights > 1.0,
        raw_weights / sum_weights[None, :],
        raw_weights
    )

    # Calculate intersection positions
    detector_times_expanded = detector_times
    ray_directions_expanded = ray_directions[None, :, :]
    ray_origins_expanded = ray_origins[None, :, :]
    hit_positions = (ray_origins_expanded +
                     detector_times_expanded * ray_directions_expanded)

    result = {
        'times': detector_times,
        'positions': hit_positions,
        'detector_weights': final_weights,
        'detector_indices': detector_indices
    }

    return result if not single_ray else jax.tree_map(lambda x: x[0], result)


def create_photon_propagator(detector_positions, detector_radius, r=4.0, h=6.0, n_cap=73, n_angular=168, n_height=82,
                             temperature=100, max_detectors_per_cell=4):
    """
    Creates a JIT-compiled function for efficient photon propagation simulation.

    Parameters
    ----------
    detector_positions : ndarray
        Array of detector positions
    detector_radius : float
        Radius of each detector
    r : float, optional
        Cylinder radius, default 4.0
    h : float, optional
        Cylinder height, default 6.0
    n_cap : int, optional
        Number of cells along each dimension in cap regions, default 39
    n_angular : int, optional
        Number of angular divisions, default 168
    n_height : int, optional
        Number of height divisions, default 28
    temperature : float, optional
        Temperature parameter for smoothing, default 100
    max_detectors_per_cell : int, optional
        Maximum number of detectors per grid cell, default 4

    Returns
    -------
    callable
        JIT-compiled function for photon propagation simulation
    """
    assignments_geometric = assign_detectors_to_grid(
        detector_positions, detector_radius, r, h, n_cap, n_angular, n_height)

    detector_grid_map = create_detector_grid_map(
        assignments_geometric, n_cap, n_angular, n_height)

    assignments_distance = find_closest_detectors(
        calculate_grid_centers(r, h, n_cap, n_angular, n_height),
        detector_positions,
        max_detectors_per_cell
    )

    inverted_detector_map = create_inverted_detector_map(
        assignments_geometric,
        assignments_distance,
        n_cap, n_angular, n_height,
        max_detectors_per_cell, detector_positions.shape[0]
    )

    @jax.jit
    def propagate_photons(photon_origins, photon_directions):
        return find_intersected_detectors_differentiable(
            photon_origins, photon_directions, detector_positions, detector_radius,
            r, h, n_cap, n_angular, n_height, inverted_detector_map,
            temperature)

    return propagate_photons