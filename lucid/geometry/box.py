"""
Box (rectangular prism) detector geometry implementation.
"""

from typing import NamedTuple

import numpy as np
from .base import Detector
from .registry import register_detector


class BoxGrid(NamedTuple):
    """The propagation grid, frozen: cells along x, y and z."""
    n_x: int
    n_y: int
    n_z: int


@register_detector('box')
class Box(Detector):
    """Box (rectangular prism) detector geometry"""
    
    def __init__(self, length, width, height, n_sensors, sensor_radius):
        """
        Initialize box detector.
        
        Parameters:
        -----------
        length : float
            Length of the box (x-dimension)
        width : float
            Width of the box (y-dimension)
        height : float
            Height of the box (z-dimension)
        n_sensors : int
            Number of photosensors
        sensor_radius : float
            Radius of individual sensors
        """
        super().__init__(n_sensors, sensor_radius)
        self.L = length
        self.W = width
        self.H = height
        self.place_photosensors()

    @classmethod
    def from_config(cls, geometry, config_dir):
        """Algorithmic layout from ``length / width / height / n_sensors / sensor_radius``."""
        return cls(geometry['length'], geometry['width'], geometry['height'],
                   geometry['n_sensors'], geometry['sensor_radius'])

    def make_grid(self, n_x=None, n_y=None, n_z=None, max_candidates_per_ray=4):
        """The propagation grid for this layout, as a :class:`BoxGrid`.

        Defaults follow the nearest-neighbour rule of :meth:`Detector._grid_cell_target`, as for
        the cylinder; explicit arguments win.
        """
        import math
        target = self._grid_cell_target(max_candidates_per_ray)
        return BoxGrid(n_x=n_x if n_x is not None else max(10, int(math.ceil(self.L / target))),
                       n_y=n_y if n_y is not None else max(10, int(math.ceil(self.W / target))),
                       n_z=n_z if n_z is not None else max(10, int(math.ceil(self.H / target))))

    def place_photosensors(self):
        """Position the photo sensor centers proportionally by surface area."""
        # Calculate surface areas for each face
        front_back_area = 2 * self.L * self.H
        left_right_area = 2 * self.W * self.H
        top_bottom_area = 2 * self.L * self.W
        total_area = front_back_area + left_right_area + top_bottom_area
        
        # Distribute sensors proportionally
        n_front_back = int(self.n_sensors * front_back_area / total_area)
        n_left_right = int(self.n_sensors * left_right_area / total_area)
        n_top_bottom = self.n_sensors - n_front_back - n_left_right
        
        n_per_front_back = n_front_back // 2
        n_per_left_right = n_left_right // 2
        n_per_top_bottom = n_top_bottom // 2
        
        # Place sensors on each face
        self.front_points = self._place_face_sensors(n_per_front_back, self.L, self.H, 'front')
        self.back_points = self._place_face_sensors(n_per_front_back, self.L, self.H, 'back')
        self.left_points = self._place_face_sensors(n_per_left_right, self.W, self.H, 'left')
        self.right_points = self._place_face_sensors(n_per_left_right, self.W, self.H, 'right')
        self.top_points = self._place_face_sensors(n_per_top_bottom, self.L, self.W, 'top')
        self.bottom_points = self._place_face_sensors(n_per_top_bottom, self.L, self.W, 'bottom')
        
        # Combine all points
        self.all_points = np.concatenate([
            self.front_points, self.back_points,
            self.left_points, self.right_points,
            self.top_points, self.bottom_points
        ], axis=0)
        
        # Create ID mappings
        self.ID_to_position = {i: self.all_points[i] for i in range(len(self.all_points))}
        
        # Create case mappings (0=front, 1=back, 2=left, 3=right, 4=top, 5=bottom)
        self.ID_to_case = {}
        n_front = len(self.front_points)
        n_back = len(self.back_points)
        n_left = len(self.left_points)
        n_right = len(self.right_points)
        n_top = len(self.top_points)
        
        for i in range(len(self.all_points)):
            if i < n_front:
                self.ID_to_case[i] = 0
            elif i < n_front + n_back:
                self.ID_to_case[i] = 1
            elif i < n_front + n_back + n_left:
                self.ID_to_case[i] = 2
            elif i < n_front + n_back + n_left + n_right:
                self.ID_to_case[i] = 3
            elif i < n_front + n_back + n_left + n_right + n_top:
                self.ID_to_case[i] = 4
            else:
                self.ID_to_case[i] = 5

    def _place_face_sensors(self, n_sensors, dim1, dim2, face):
        """Place sensors on a rectangular face with regular grid."""
        if n_sensors == 0:
            return np.array([]).reshape(0, 3)
            
        # Calculate effective dimensions (with margins)
        dim1_eff = dim1 - 3 * self.S_radius
        dim2_eff = dim2 - 3 * self.S_radius
        
        if dim1_eff <= 0 or dim2_eff <= 0:
            return np.array([]).reshape(0, 3)
        
        # Find optimal rows and columns for approximately square spacing
        aspect_ratio = dim1_eff / dim2_eff
        n_rows = int(np.sqrt(n_sensors / aspect_ratio))
        n_cols = n_sensors // n_rows if n_rows > 0 else n_sensors
        
        # Adjust to get closer to target
        while n_rows * n_cols < n_sensors and n_rows > 1:
            if (n_rows + 1) * n_cols <= n_sensors:
                n_rows += 1
            elif n_rows * (n_cols + 1) <= n_sensors:
                n_cols += 1
            else:
                break
        
        # Generate grid
        pos1 = np.linspace(-dim1_eff/2, dim1_eff/2, n_cols)
        pos2 = np.linspace(-dim2_eff/2, dim2_eff/2, n_rows)
        
        points = []
        for p2 in pos2:
            for p1 in pos1:
                if face == 'front':  # +y face
                    points.append([p1 + self.C[0], self.W/2 + self.C[1], p2 + self.C[2]])
                elif face == 'back':  # -y face
                    points.append([p1 + self.C[0], -self.W/2 + self.C[1], p2 + self.C[2]])
                elif face == 'left':  # -x face
                    points.append([-self.L/2 + self.C[0], p1 + self.C[1], p2 + self.C[2]])
                elif face == 'right':  # +x face
                    points.append([self.L/2 + self.C[0], p1 + self.C[1], p2 + self.C[2]])
                elif face == 'top':  # +z face
                    points.append([p1 + self.C[0], p2 + self.C[1], self.H/2 + self.C[2]])
                elif face == 'bottom':  # -z face
                    points.append([p1 + self.C[0], p2 + self.C[1], -self.H/2 + self.C[2]])
        
        return np.array(points[:n_sensors])  # Trim to exact count

    @property
    def volume(self):
        """The instrumented volume (:mod:`lucid.geometry.volume`)."""
        from lucid.geometry.volume import BoxVolume
        return BoxVolume(self.L / 2, self.W / 2, self.H / 2)

    def bounds_check(self, positions):
        """Test whether positions are inside the box."""
        return self.volume.contains(positions)

    # ── Propagation methods (Phase 9) ──────────────────────────────────

    def intersect_ray(self, origins, directions, grid):
        """Batch ray-box intersection with grid indexing."""
        from lucid.propagation.box import batch_intersect_box_with_grid
        results = batch_intersect_box_with_grid(
            origins, directions, self.L, self.W, self.H, grid.n_x, grid.n_y, grid.n_z)
        intersects, t, face_indices, grid_indices, intersection_point = results
        grid_info = (face_indices, grid_indices)
        surface_info = face_indices
        return intersection_point, t, grid_info, surface_info

    def compute_normal(self, intersection_point, surface_info):
        """Compute outward box face normals."""
        from lucid.propagation.box import calculate_box_normals
        return calculate_box_normals(surface_info)  # surface_info = face_indices

    def point_to_grid_cell(self, grid_info, grid):
        """Map box intersection to linear grid cell index."""
        import jax.numpy as jnp
        face_indices, grid_indices = grid_info
        n_x, n_y, n_z = grid

        fb_cells = n_x * n_z           # front/back face cells
        lr_cells = n_y * n_z           # left/right face cells
        tb_cells = n_x * n_y           # top/bottom face cells

        offsets = jnp.array([
            0,                           # face 0: front
            fb_cells,                    # face 1: back
            2 * fb_cells,                # face 2: left
            2 * fb_cells + lr_cells,     # face 3: right
            2 * (fb_cells + lr_cells),   # face 4: top
            2 * (fb_cells + lr_cells) + tb_cells,  # face 5: bottom
        ])

        # grid_indices[:, 0] and [:, 1] are the 2D cell coords on the face
        # For front/back/left/right: second_dim is n_z
        # For top/bottom: second_dim is n_y
        second_dim = jnp.where(face_indices <= 3, n_z, n_y)
        local_idx = grid_indices[:, 0] * second_dim + grid_indices[:, 1]

        total_cells = 2 * (fb_cells + lr_cells + tb_cells)
        idx = offsets[face_indices] + local_idx
        return jnp.clip(idx, 0, total_cells - 1)

    def assign_sensor_to_cells(self, sensors, sensor_radius, grid):
        """Map sensors to overlapping box grid cells."""
        from lucid.propagation.box import assign_sensors_to_box_grid
        return assign_sensors_to_box_grid(
            sensors, sensor_radius, self.L, self.W, self.H, grid.n_x, grid.n_y, grid.n_z)

    def grid_cell_centers(self, grid):
        """Compute centers of all box grid cells."""
        from lucid.propagation.box import calculate_box_grid_centers
        return calculate_box_grid_centers(self.L, self.W, self.H, grid.n_x, grid.n_y, grid.n_z)

    def total_grid_cells(self, grid):
        n_x, n_y, n_z = grid
        return 2 * (n_x * n_z + n_y * n_z + n_x * n_y)

    def point_to_grid_cell_from_coords(self, coords, grid):
        """Convert grid coordinates to linear index.

        For box: coords = [cell_i, cell_j, face_idx].
        """
        cell_i, cell_j, face_idx = int(coords[0]), int(coords[1]), int(coords[2])
        fb = grid.n_x * grid.n_z
        lr = grid.n_y * grid.n_z
        tb = grid.n_x * grid.n_y
        offsets = [0, fb, 2*fb, 2*fb+lr, 2*(fb+lr), 2*(fb+lr)+tb]
        second_dim = grid.n_z if face_idx <= 3 else grid.n_y
        return offsets[face_idx] + cell_i * second_dim + cell_j

    def build_inverted_sensor_map(self, assignments_geometric, assignments_distance,
                                   max_candidates_per_ray, grid):
        """Build cell→sensor lookup table for box."""
        from lucid.propagation.box import box_grid_cells
        from lucid.propagation.sensor_map import build_inverted_sensor_map
        return build_inverted_sensor_map(
            assignments_geometric, assignments_distance,
            box_grid_cells(grid.n_x, grid.n_y, grid.n_z), max_candidates_per_ray)