"""
Box (rectangular prism) detector geometry implementation.
"""

import numpy as np
import plotly.graph_objects as go
from .base import Detector
from .registry import register_detector


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
        self._n_x = None
        self._n_y = None
        self._n_z = None
        self.place_photosensors()

    def configure_grid(self, n_x=None, n_y=None, n_z=None,
                        max_candidates_per_ray=4):
        """Set grid parameters for propagation methods.

        If not provided, defaults are derived from detector dimensions to
        ensure no cell exceeds ``max_candidates_per_ray`` sensors.
        """
        import math
        n_placed = len(self.all_points) if self.all_points is not None else self.n_sensors
        total_area = 2 * (self.L * self.W + self.L * self.H + self.W * self.H)
        # Target cell size from: n_placed / max_candidates_per_ray × safety
        safety = 2.0
        target_cells = n_placed / max_candidates_per_ray * safety
        cell_size = math.sqrt(total_area / max(1, target_cells))

        self._n_x = n_x if n_x is not None else max(10, int(self.L / cell_size))
        self._n_y = n_y if n_y is not None else max(10, int(self.W / cell_size))
        self._n_z = n_z if n_z is not None else max(10, int(self.H / cell_size))


    def place_photosensors(self):
        """Position the photo sensor centers proportionally by surface area."""
        front_back_area = 2 * self.L * self.H
        left_right_area = 2 * self.W * self.H
        top_bottom_area = 2 * self.L * self.W
        total_area = front_back_area + left_right_area + top_bottom_area
        
        n_front_back = int(self.n_sensors * front_back_area / total_area)
        n_left_right = int(self.n_sensors * left_right_area / total_area)
        n_top_bottom = self.n_sensors - n_front_back - n_left_right
        
        n_per_front_back = n_front_back // 2
        n_per_left_right = n_left_right // 2
        n_per_top_bottom = n_top_bottom // 2
        
        self.front_points = self._place_face_sensors(n_per_front_back, self.L, self.H, 'front')
        self.back_points = self._place_face_sensors(n_per_front_back, self.L, self.H, 'back')
        self.left_points = self._place_face_sensors(n_per_left_right, self.W, self.H, 'left')
        self.right_points = self._place_face_sensors(n_per_left_right, self.W, self.H, 'right')
        self.top_points = self._place_face_sensors(n_per_top_bottom, self.L, self.W, 'top')
        self.bottom_points = self._place_face_sensors(n_per_top_bottom, self.L, self.W, 'bottom')
        
        self.all_points = np.concatenate([
            self.front_points, self.back_points,
            self.left_points, self.right_points,
            self.top_points, self.bottom_points
        ], axis=0)
        
        self.ID_to_position = {i: self.all_points[i] for i in range(len(self.all_points))}
        
        # Create case mappings (0=front, 1=back, 2=left, 3=right, 4=top, 5=bottom)
        self.ID_to_case = {}
        n_front = len(self.front_points)
        n_back = len(self.back_points)
        n_left = len(self.left_points)
        n_right = len(self.right_points)
        n_top = len(self.top_points)
        n_bottom = len(self.bottom_points)
        
        cumulative = 0
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
            
        dim1_eff = dim1 - 3 * self.S_radius
        dim2_eff = dim2 - 3 * self.S_radius
        
        if dim1_eff <= 0 or dim2_eff <= 0:
            return np.array([]).reshape(0, 3)
        
        # Find optimal rows and columns for approximately square spacing
        aspect_ratio = dim1_eff / dim2_eff
        n_rows = int(np.sqrt(n_sensors / aspect_ratio))
        n_cols = n_sensors // n_rows if n_rows > 0 else n_sensors
        
        while n_rows * n_cols < n_sensors and n_rows > 1:
            if (n_rows + 1) * n_cols <= n_sensors:
                n_rows += 1
            elif n_rows * (n_cols + 1) <= n_sensors:
                n_cols += 1
            else:
                break
        
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

    def visualize_geometry_wireframe(self, show_sensors=True):
        """Visualize the box as a wireframe with detectors"""
        fig = go.Figure()

        x_min, x_max = -self.L/2 + self.C[0], self.L/2 + self.C[0]
        y_min, y_max = -self.W/2 + self.C[1], self.W/2 + self.C[1]
        z_min, z_max = -self.H/2 + self.C[2], self.H/2 + self.C[2]
        
        edges = [
            [[x_min, x_max], [y_min, y_min], [z_min, z_min]],
            [[x_max, x_max], [y_min, y_max], [z_min, z_min]],
            [[x_max, x_min], [y_max, y_max], [z_min, z_min]],
            [[x_min, x_min], [y_max, y_min], [z_min, z_min]],
            [[x_min, x_max], [y_min, y_min], [z_max, z_max]],
            [[x_max, x_max], [y_min, y_max], [z_max, z_max]],
            [[x_max, x_min], [y_max, y_max], [z_max, z_max]],
            [[x_min, x_min], [y_max, y_min], [z_max, z_max]],
            [[x_min, x_min], [y_min, y_min], [z_min, z_max]],
            [[x_max, x_max], [y_min, y_min], [z_min, z_max]],
            [[x_max, x_max], [y_max, y_max], [z_min, z_max]],
            [[x_min, x_min], [y_max, y_max], [z_min, z_max]],
        ]
        
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=edge[0], y=edge[1], z=edge[2],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ))

        face_colors = ['lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightcoral', 'lightcoral']
        face_names = ['Front', 'Back', 'Left', 'Right', 'Top', 'Bottom']
        
        # Front face (+y)
        x_face = [x_min, x_max, x_max, x_min, x_min]
        z_face = [z_min, z_min, z_max, z_max, z_min]
        fig.add_trace(go.Scatter3d(
            x=x_face, y=[y_max]*5, z=z_face,
            mode='lines', fill='toself',
            line=dict(color='lightblue', width=1),
            opacity=0.1, name='Front Face'
        ))
        
        # Back face (-y)
        fig.add_trace(go.Scatter3d(
            x=x_face, y=[y_min]*5, z=z_face,
            mode='lines', fill='toself',
            line=dict(color='lightblue', width=1),
            opacity=0.1, name='Back Face'
        ))

        if show_sensors:
            colors = ['blue', 'darkblue', 'green', 'darkgreen', 'red', 'darkred']
            face_names_full = ['Front', 'Back', 'Left', 'Right', 'Top', 'Bottom']
            
            for case in range(6):
                indices = [i for i, c in self.ID_to_case.items() if c == case]
                if indices:
                    face_points = self.all_points[indices]
                    fig.add_trace(go.Scatter3d(
                        x=face_points[:, 0], 
                        y=face_points[:, 1], 
                        z=face_points[:, 2],
                        mode='markers',
                        marker=dict(size=4, color=colors[case], opacity=0.8),
                        name=f'{face_names_full[case]} Sensors ({len(indices)})'
                    ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            title=f'Box Detector Geometry (L={self.L}, W={self.W}, H={self.H})',
            height=800
        )

        fig.show()

    def bounds_check(self, positions):
        """Test whether positions are inside the box."""
        import jax.numpy as jnp
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        return ((x >= -self.L / 2) & (x <= self.L / 2) &
                (y >= -self.W / 2) & (y <= self.W / 2) &
                (z >= -self.H / 2) & (z <= self.H / 2))

    # ── Propagation methods (Phase 9) ──────────────────────────────────

    def intersect_ray(self, origins, directions):
        """Batch ray-box intersection with grid indexing."""
        from lucid.propagation.box import batch_intersect_box_with_grid
        n_x = self._n_x
        n_y = self._n_y
        n_z = self._n_z
        results = batch_intersect_box_with_grid(
            origins, directions, self.L, self.W, self.H, n_x, n_y, n_z)
        intersects, t, face_indices, grid_indices, intersection_point = results
        grid_info = (face_indices, grid_indices)
        surface_info = face_indices
        return intersection_point, t, grid_info, surface_info

    def compute_normal(self, intersection_point, surface_info):
        """Compute outward box face normals."""
        from lucid.propagation.box import calculate_box_normals
        return calculate_box_normals(surface_info)  # surface_info = face_indices

    def point_to_grid_cell(self, grid_info):
        """Map box intersection to linear grid cell index."""
        import jax.numpy as jnp
        face_indices, grid_indices = grid_info
        n_x = self._n_x
        n_y = self._n_y
        n_z = self._n_z

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

    def assign_sensor_to_cells(self, sensors, sensor_radius):
        """Map sensors to overlapping box grid cells."""
        from lucid.propagation.box import assign_sensors_to_box_grid
        n_x = self._n_x
        n_y = self._n_y
        n_z = self._n_z
        return assign_sensors_to_box_grid(
            sensors, sensor_radius, self.L, self.W, self.H, n_x, n_y, n_z)

    def grid_cell_centers(self):
        """Compute centers of all box grid cells."""
        from lucid.propagation.box import calculate_box_grid_centers
        n_x = self._n_x
        n_y = self._n_y
        n_z = self._n_z
        return calculate_box_grid_centers(self.L, self.W, self.H, n_x, n_y, n_z)

    def total_grid_cells(self):
        n_x = self._n_x
        n_y = self._n_y
        n_z = self._n_z
        return 2 * (n_x * n_z + n_y * n_z + n_x * n_y)

    def cell_index_to_coords(self, linear_idx):
        """Decode linear index to (cell_i, cell_j, face_idx) for box."""
        import jax.numpy as jnp
        n_x = self._n_x
        n_y = self._n_y
        n_z = self._n_z
        fb = n_x * n_z
        lr = n_y * n_z
        tb = n_x * n_y

        is_fb = linear_idx < 2 * fb
        is_lr = (linear_idx >= 2 * fb) & (linear_idx < 2 * fb + 2 * lr)

        # Front/Back
        fb_face = jnp.where(linear_idx < fb, 0, 1)
        fb_local = linear_idx % fb
        fb_i = fb_local // n_z
        fb_j = fb_local % n_z

        # Left/Right
        lr_offset = linear_idx - 2 * fb
        lr_face = jnp.where(lr_offset < lr, 2, 3)
        lr_local = lr_offset % lr
        lr_i = lr_local // n_z
        lr_j = lr_local % n_z

        # Top/Bottom
        tb_offset = linear_idx - 2 * fb - 2 * lr
        tb_face = jnp.where(tb_offset < tb, 4, 5)
        tb_local = tb_offset % tb
        tb_i = tb_local // n_y
        tb_j = tb_local % n_y

        cell_i = jnp.where(is_fb, fb_i, jnp.where(is_lr, lr_i, tb_i))
        cell_j = jnp.where(is_fb, fb_j, jnp.where(is_lr, lr_j, tb_j))
        face_idx = jnp.where(is_fb, fb_face, jnp.where(is_lr, lr_face, tb_face))
        return cell_i, cell_j, face_idx

    def point_to_grid_cell_from_coords(self, coords):
        """Convert grid coordinates to linear index.

        For box: coords = [cell_i, cell_j, face_idx].
        """
        cell_i, cell_j, face_idx = int(coords[0]), int(coords[1]), int(coords[2])
        fb = self._n_x * self._n_z
        lr = self._n_y * self._n_z
        tb = self._n_x * self._n_y
        offsets = [0, fb, 2*fb, 2*fb+lr, 2*(fb+lr), 2*(fb+lr)+tb]
        second_dim = self._n_z if face_idx <= 3 else self._n_y
        return offsets[face_idx] + cell_i * second_dim + cell_j

    def build_inverted_sensor_map(self, assignments_geometric, assignments_distance,
                                   max_candidates_per_ray, num_sensors):
        """Build cell→sensor lookup table for box."""
        from lucid.propagation.box import create_inverted_box_sensor_map
        n_x = self._n_x
        n_y = self._n_y
        n_z = self._n_z
        return create_inverted_box_sensor_map(
            assignments_geometric, assignments_distance,
            n_x, n_y, n_z, max_candidates_per_ray)