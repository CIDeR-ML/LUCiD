"""
Cylindrical detector geometry implementation.
"""

import numpy as np
import plotly.graph_objects as go
from .base import Detector
from .utils import generate_concentric_hexagons
from .registry import register_detector


@register_detector('cylinder')
class Cylinder(Detector):
    """Cylindrical detector geometry"""
    
    def __init__(self, radius, height, n_sensors, sensor_radius):
        """
        Initialize cylindrical detector.
        
        Parameters:
        -----------
        radius : float
            Radius of the cylinder
        height : float
            Height of the cylinder
        n_sensors : int
            Number of photosensors
        sensor_radius : float
            Radius of individual sensors
        """
        super().__init__(n_sensors, sensor_radius)
        self.r = radius
        self.H = height
        self.place_photosensors()

    def place_photosensors(self):
        """Position the photo sensor centers proportionally by surface area."""
        # Calculate surface areas
        barrel_area = 2 * np.pi * self.r * self.H
        caps_area = 2 * np.pi * self.r**2  # Both caps combined
        total_area = barrel_area + caps_area
        
        # Distribute sensors proportionally
        n_barrel = int(self.n_sensors * barrel_area / total_area)
        n_caps = self.n_sensors - n_barrel
        n_per_cap = n_caps // 2  # Split equally between top and bottom
        
        # Place barrel sensors
        self.barr_points = self._place_barrel_sensors(n_barrel)
        
        # Place cap sensors
        self.tcap_points = self._place_cap_sensors(n_per_cap, self.H/2)  # Top cap
        self.bcap_points = self._place_cap_sensors(n_per_cap, -self.H/2)  # Bottom cap
        
        # Combine all points
        self.all_points = np.concatenate([self.barr_points, self.tcap_points, self.bcap_points], axis=0)
        
        # Create ID mappings
        self.ID_to_position = {i: self.all_points[i] for i in range(len(self.all_points))}
        
        # Create case mappings (0=barrel, 1=top cap, 2=bottom cap)
        self.ID_to_case = {}
        n_barr = len(self.barr_points)
        n_tcap = len(self.tcap_points)
        n_bcap = len(self.bcap_points)
        
        for i in range(len(self.all_points)):
            if i < n_barr:
                self.ID_to_case[i] = 0
            elif i < n_barr + n_tcap:
                self.ID_to_case[i] = 1
            else:
                self.ID_to_case[i] = 2

    def _place_barrel_sensors(self, n_sensors):
        """Place sensors on barrel surface with rectangular grid."""
        if n_sensors == 0:
            return np.array([]).reshape(0, 3)
            
        # Calculate effective dimensions (with margins)
        height_eff = self.H - 3 * self.S_radius  # Top and bottom margins
        circumference_eff = 2 * np.pi * self.r
        
        # Find optimal rows and columns for approximately square spacing
        aspect_ratio = height_eff / circumference_eff
        n_rows = int(np.sqrt(n_sensors * aspect_ratio))
        n_cols = n_sensors // n_rows
        
        # Adjust to get closer to target
        while n_rows * n_cols < n_sensors and n_rows > 1:
            if (n_rows + 1) * n_cols <= n_sensors:
                n_rows += 1
            elif n_rows * (n_cols + 1) <= n_sensors:
                n_cols += 1
            else:
                break
        
        # Generate grid
        z_positions = np.linspace(-height_eff/2, height_eff/2, n_rows) + self.C[2]
        theta_positions = np.linspace(0, 2*np.pi, n_cols, endpoint=False)
        
        points = []
        for z in z_positions:
            for theta in theta_positions:
                x = self.r * np.cos(theta) + self.C[0]
                y = self.r * np.sin(theta) + self.C[1]
                points.append([x, y, z])
        
        return np.array(points[:n_sensors])  # Trim to exact count

    def _place_cap_sensors(self, n_sensors, z_position):
        """Place sensors on cap surface with concentric hexagonal rings."""
        if n_sensors == 0:
            return np.array([]).reshape(0, 3)
            
        # Calculate effective radius (with margin)
        radius_eff = self.r - 1.5 * self.S_radius
        
        if radius_eff <= 0:
            return np.array([]).reshape(0, 3)
        
        # Generate concentric hexagonal pattern
        hex_points = generate_concentric_hexagons(n_sensors, radius_eff)
        
        # Convert to 3D and translate
        points_3d = np.zeros((len(hex_points), 3))
        points_3d[:, 0] = hex_points[:, 0] + self.C[0]
        points_3d[:, 1] = hex_points[:, 1] + self.C[1]
        points_3d[:, 2] = z_position + self.C[2]
        
        return points_3d

    def visualize_geometry_wireframe(self, show_sensors=True):
        """Visualize the cylinder as a wireframe with detectors"""
        fig = go.Figure()

        # Create cylinder wireframe
        # Barrel surface
        theta = np.linspace(0, 2 * np.pi, 50)
        z_barrel = np.linspace(-self.H/2, self.H/2, 20)
        theta_mesh, z_mesh = np.meshgrid(theta, z_barrel)
        
        x_barrel = self.r * np.cos(theta_mesh) + self.C[0]
        y_barrel = self.r * np.sin(theta_mesh) + self.C[1]
        z_barrel_mesh = z_mesh + self.C[2]

        # Add barrel wireframe
        fig.add_trace(go.Surface(
            x=x_barrel, y=y_barrel, z=z_barrel_mesh,
            opacity=0.1,
            showscale=False,
            colorscale=[[0, 'lightblue'], [1, 'lightblue']],
            name='Barrel Surface'
        ))

        # Top cap
        r_cap = np.linspace(0, self.r, 20)
        theta_cap = np.linspace(0, 2 * np.pi, 50)
        r_mesh, theta_mesh = np.meshgrid(r_cap, theta_cap)
        
        x_top = r_mesh * np.cos(theta_mesh) + self.C[0]
        y_top = r_mesh * np.sin(theta_mesh) + self.C[1]
        z_top = np.full_like(x_top, self.H/2 + self.C[2])

        # Add top cap wireframe
        fig.add_trace(go.Surface(
            x=x_top, y=y_top, z=z_top,
            opacity=0.1,
            showscale=False,
            colorscale=[[0, 'lightgreen'], [1, 'lightgreen']],
            name='Top Cap'
        ))

        # Bottom cap
        z_bottom = np.full_like(x_top, -self.H/2 + self.C[2])

        # Add bottom cap wireframe
        fig.add_trace(go.Surface(
            x=x_top, y=y_top, z=z_bottom,
            opacity=0.1,
            showscale=False,
            colorscale=[[0, 'lightcoral'], [1, 'lightcoral']],
            name='Bottom Cap'
        ))

        if show_sensors:
            # Color code sensors by type
            barrel_indices = [i for i, case in self.ID_to_case.items() if case == 0]
            tcap_indices = [i for i, case in self.ID_to_case.items() if case == 1]
            bcap_indices = [i for i, case in self.ID_to_case.items() if case == 2]
            
            if barrel_indices:
                barrel_points = self.all_points[barrel_indices]
                fig.add_trace(go.Scatter3d(
                    x=barrel_points[:, 0], 
                    y=barrel_points[:, 1], 
                    z=barrel_points[:, 2],
                    mode='markers',
                    marker=dict(size=4, color='blue', opacity=0.8),
                    name=f'Barrel Sensors ({len(barrel_indices)})'
                ))
            
            if tcap_indices:
                tcap_points = self.all_points[tcap_indices]
                fig.add_trace(go.Scatter3d(
                    x=tcap_points[:, 0], 
                    y=tcap_points[:, 1], 
                    z=tcap_points[:, 2],
                    mode='markers',
                    marker=dict(size=4, color='green', opacity=0.8),
                    name=f'Top Cap Sensors ({len(tcap_indices)})'
                ))
            
            if bcap_indices:
                bcap_points = self.all_points[bcap_indices]
                fig.add_trace(go.Scatter3d(
                    x=bcap_points[:, 0], 
                    y=bcap_points[:, 1], 
                    z=bcap_points[:, 2],
                    mode='markers',
                    marker=dict(size=4, color='red', opacity=0.8),
                    name=f'Bottom Cap Sensors ({len(bcap_indices)})'
                ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            title=f'Cylindrical Detector Geometry (R={self.r}, H={self.H})',
            height=800
        )

        fig.show()

    def bounds_check(self, positions):
        """Test whether positions are inside the cylinder."""
        import jax.numpy as jnp
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        inside_xy = (x ** 2 + y ** 2) <= self.r ** 2
        inside_z = (z >= -self.H / 2) & (z <= self.H / 2)
        return inside_xy & inside_z

    # ── Propagation methods (Phase 9) ──────────────────────────────────

    def intersect_ray(self, origins, directions):
        """Batch ray-cylinder intersection with grid indexing."""
        from lucid.propagation.cylinder import batch_intersect_cylinder_with_grid
        # Default grid params matching create_photon_propagator defaults
        n_cap = getattr(self, '_n_cap', 150)
        n_angular = getattr(self, '_n_angular', 250)
        n_height = getattr(self, '_n_height', 150)
        results = batch_intersect_cylinder_with_grid(
            origins, directions, self.r, self.H, n_cap, n_angular, n_height)
        # Returns: (intersects, t, is_wall, is_top_cap, wall_indices, cap_indices, intersection_point)
        intersects, t, is_wall, is_top_cap, wall_indices, cap_indices, intersection_point = results
        grid_info = (is_wall, is_top_cap, wall_indices, cap_indices)
        surface_info = (is_wall, is_top_cap)
        return intersection_point, t, grid_info, surface_info

    def compute_normal(self, intersection_point, surface_info):
        """Compute outward cylinder surface normals."""
        from lucid.propagation.cylinder import calculate_cylinder_normals
        is_wall, is_top_cap = surface_info
        return calculate_cylinder_normals(intersection_point, is_wall, is_top_cap)

    def point_to_grid_cell(self, grid_info):
        """Map cylinder intersection to linear grid cell index."""
        import jax.numpy as jnp
        is_wall, is_top_cap, wall_indices, cap_indices = grid_info
        n_cap = getattr(self, '_n_cap', 150)
        n_angular = getattr(self, '_n_angular', 250)
        n_height = getattr(self, '_n_height', 150)

        wall_linear = wall_indices[:, 0] * n_height + wall_indices[:, 1]
        cap_linear = cap_indices[:, 0] * n_cap + cap_indices[:, 1]
        n_wall_cells = n_angular * n_height

        idx = jnp.where(is_wall, wall_linear,
                        jnp.where(is_top_cap,
                                  n_wall_cells + cap_linear,
                                  n_wall_cells + n_cap * n_cap + cap_linear))
        total_cells = n_wall_cells + 2 * n_cap * n_cap
        return jnp.clip(idx, 0, total_cells - 1)

    def assign_sensor_to_cells(self, sensors, sensor_radius):
        """Map sensors to overlapping cylinder grid cells."""
        from lucid.propagation.cylinder import assign_sensors_to_grid
        n_cap = getattr(self, '_n_cap', 150)
        n_angular = getattr(self, '_n_angular', 250)
        n_height = getattr(self, '_n_height', 150)
        return assign_sensors_to_grid(
            sensors, sensor_radius, self.r, self.H, n_cap, n_angular, n_height)

    def grid_cell_centers(self):
        """Compute centers of all cylinder grid cells."""
        from lucid.propagation.cylinder import calculate_grid_centers
        n_cap = getattr(self, '_n_cap', 150)
        n_angular = getattr(self, '_n_angular', 250)
        n_height = getattr(self, '_n_height', 150)
        return calculate_grid_centers(self.r, self.H, n_cap, n_angular, n_height)

    def total_grid_cells(self):
        n_cap = getattr(self, '_n_cap', 150)
        n_angular = getattr(self, '_n_angular', 250)
        n_height = getattr(self, '_n_height', 150)
        return n_angular * n_height + 2 * n_cap * n_cap

    def cell_index_to_coords(self, linear_idx):
        """Decode linear index to (cell_i, cell_j, cell_k) for cylinder."""
        import jax.numpy as jnp
        n_cap = getattr(self, '_n_cap', 150)
        n_angular = getattr(self, '_n_angular', 250)
        n_height = getattr(self, '_n_height', 150)
        n_wall = n_angular * n_height

        is_wall = linear_idx < n_wall
        is_top = (linear_idx >= n_wall) & (linear_idx < n_wall + n_cap * n_cap)

        wall_i = linear_idx // n_height
        wall_j = linear_idx % n_height
        cap_offset = linear_idx - n_wall
        cap_idx = cap_offset % (n_cap * n_cap)
        cap_i = cap_idx // n_cap
        cap_j = cap_idx % n_cap

        cell_i = jnp.where(is_wall, wall_i, cap_i)
        cell_j = jnp.where(is_wall, wall_j, cap_j)
        cell_k = jnp.where(is_wall, 0, jnp.where(is_top, 1, 2))
        return cell_i, cell_j, cell_k