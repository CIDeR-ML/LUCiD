"""
Spherical detector geometry implementation.
"""

import numpy as np
import plotly.graph_objects as go
from .base import Detector
from .utils import fibonacci_sphere_points_numpy
from .registry import register_detector


@register_detector('sphere')
class Sphere(Detector):
    """Spherical detector geometry"""
    
    def __init__(self, radius, n_sensors, sensor_radius):
        """
        Initialize spherical detector.
        
        Parameters:
        -----------
        radius : float
            Radius of the sphere
        n_sensors : int
            Number of photosensors
        sensor_radius : float
            Radius of individual sensors
        """
        super().__init__(n_sensors, sensor_radius)
        self.r = radius
        self.place_photosensors()

    def place_photosensors(self):
        """Position the photo sensor centers on the sphere surface using Fibonacci spiral."""
        self.all_points = fibonacci_sphere_points_numpy(self.n_sensors, self.r) + self.C
        
        # Create ID to position dictionary
        self.ID_to_position = {i: self.all_points[i] for i in range(len(self.all_points))}
        
        # For sphere, all sensors are on surface (case 0)
        self.ID_to_case = {i: 0 for i in range(len(self.all_points))}

    def visualize_geometry_wireframe(self, show_sensors=True):
        """Visualize the sphere as a wireframe with detectors"""
        fig = go.Figure()

        # Create sphere wireframe
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = self.r * np.outer(np.cos(u), np.sin(v)) + self.C[0]
        y_sphere = self.r * np.outer(np.sin(u), np.sin(v)) + self.C[1]
        z_sphere = self.r * np.outer(np.ones(np.size(u)), np.cos(v)) + self.C[2]

        # Add wireframe
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.1,
            showscale=False,
            colorscale=[[0, 'lightblue'], [1, 'lightblue']],
            name='Sphere Surface'
        ))

        if show_sensors:
            fig.add_trace(go.Scatter3d(
                x=self.all_points[:, 0], 
                y=self.all_points[:, 1], 
                z=self.all_points[:, 2],
                mode='markers',
                marker=dict(size=4, color='red', opacity=0.8),
                name=f'Sensors ({self.n_sensors})'
            ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            title=f'Spherical Detector Geometry (R={self.r})',
            height=800
        )

        fig.show()

    def bounds_check(self, positions):
        """Test whether positions are inside the sphere."""
        import jax.numpy as jnp
        return jnp.linalg.norm(positions, axis=1) <= self.r

    # ── Propagation methods (Phase 9) ──────────────────────────────────

    def intersect_ray(self, origins, directions):
        """Batch ray-sphere intersection with grid indexing."""
        from lucid.propagation.sphere import batch_intersect_sphere_with_grid
        n_div = getattr(self, '_n_divisions', 50)
        results = batch_intersect_sphere_with_grid(origins, directions, self.r, n_div)
        intersects, t, theta_idx, phi_idx, intersection_point = results
        grid_info = (theta_idx, phi_idx)
        surface_info = None  # sphere has uniform surface
        return intersection_point, t, grid_info, surface_info

    def compute_normal(self, intersection_point, surface_info):
        """Compute outward sphere surface normals."""
        from lucid.propagation.sphere import calculate_sphere_normals
        return calculate_sphere_normals(intersection_point)

    def point_to_grid_cell(self, grid_info):
        """Map sphere intersection to linear grid cell index."""
        import jax.numpy as jnp
        theta_idx, phi_idx = grid_info
        n_div = getattr(self, '_n_divisions', 50)
        n_theta = n_div
        n_phi = 2 * n_div
        idx = theta_idx * n_phi + phi_idx
        total = n_theta * n_phi
        return jnp.clip(idx, 0, total - 1)

    def assign_sensor_to_cells(self, sensors, sensor_radius):
        """Map sensors to overlapping sphere grid cells."""
        from lucid.propagation.sphere import assign_sensors_to_sphere_grid
        n_div = getattr(self, '_n_divisions', 50)
        return assign_sensors_to_sphere_grid(sensors, sensor_radius, self.r, n_div)

    def grid_cell_centers(self):
        """Compute centers of all sphere grid cells."""
        from lucid.propagation.sphere import calculate_sphere_grid_centers
        n_div = getattr(self, '_n_divisions', 50)
        return calculate_sphere_grid_centers(self.r, n_div)

    def total_grid_cells(self):
        n_div = getattr(self, '_n_divisions', 50)
        return n_div * (2 * n_div)

    def cell_index_to_coords(self, linear_idx):
        """Decode linear index to (theta_idx, phi_idx) for sphere."""
        n_div = getattr(self, '_n_divisions', 50)
        n_phi = 2 * n_div
        theta_idx = linear_idx // n_phi
        phi_idx = linear_idx % n_phi
        return theta_idx, phi_idx

    def build_inverted_sensor_map(self, assignments_geometric, assignments_distance,
                                   max_sensors_per_cell, num_sensors):
        """Build cell→sensor lookup table for sphere."""
        from lucid.propagation.sphere import create_inverted_sphere_sensor_map
        n_div = getattr(self, '_n_divisions', 50)
        return create_inverted_sphere_sensor_map(
            assignments_geometric, assignments_distance,
            n_div, max_sensors_per_cell, num_sensors)