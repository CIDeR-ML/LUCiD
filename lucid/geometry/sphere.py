"""
Spherical detector geometry implementation.
"""

from typing import NamedTuple

from .base import Detector
from .utils import fibonacci_sphere_points_numpy
from .registry import register_detector


class SphereGrid(NamedTuple):
    """The propagation grid, frozen: n_divisions polar cells by 2 * n_divisions azimuthal."""
    n_divisions: int


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

    @classmethod
    def from_config(cls, geometry, config_dir):
        """Algorithmic layout from ``radius / n_sensors / sensor_radius``."""
        return cls(geometry['radius'], geometry['n_sensors'], geometry['sensor_radius'])

    def make_grid(self, n_divisions=None, max_candidates_per_ray=4):
        """The propagation grid for this layout, as a :class:`SphereGrid`.

        If not provided, defaults are derived from sensor count to
        ensure no cell exceeds ``max_candidates_per_ray`` sensors.
        """
        import math
        if n_divisions is not None:
            return SphereGrid(n_divisions)
        n_placed = len(self.all_points) if self.all_points is not None else self.n_sensors
        # total cells = 2 × n_div². Target: n_placed / max_candidates_per_ray × safety.
        # Polar grid has non-uniform cell sizes — equatorial cells are much
        # larger than polar cells, causing sensor crowding at the equator.
        # Safety factor 6.0 ensures no cell exceeds max_candidates_per_ray.
        target_cells = n_placed / max_candidates_per_ray * 8.0
        return SphereGrid(max(10, int(math.sqrt(target_cells / 2))))

    def place_photosensors(self):
        """Position the photo sensor centers on the sphere surface using Fibonacci spiral."""
        self.all_points = fibonacci_sphere_points_numpy(self.n_sensors, self.r) + self.C
        
        # Create ID to position dictionary
        self.ID_to_position = {i: self.all_points[i] for i in range(len(self.all_points))}
        
        # For sphere, all sensors are on surface (case 0)
        self.ID_to_case = {i: 0 for i in range(len(self.all_points))}

    @property
    def volume(self):
        """The instrumented volume (:mod:`lucid.geometry.volume`)."""
        from lucid.geometry.volume import SphereVolume
        return SphereVolume(self.r)

    def bounds_check(self, positions):
        """Test whether positions are inside the sphere."""
        return self.volume.contains(positions)

    # ── Propagation methods (Phase 9) ──────────────────────────────────

    def intersect_ray(self, origins, directions, grid):
        """Batch ray-sphere intersection with grid indexing."""
        from lucid.propagation.sphere import batch_intersect_sphere_with_grid
        results = batch_intersect_sphere_with_grid(origins, directions, self.r, grid.n_divisions)
        intersects, t, theta_idx, phi_idx, intersection_point = results
        grid_info = (theta_idx, phi_idx)
        surface_info = None  # sphere has uniform surface
        return intersection_point, t, grid_info, surface_info

    def compute_normal(self, intersection_point, surface_info):
        """Compute outward sphere surface normals."""
        from lucid.propagation.sphere import calculate_sphere_normals
        return calculate_sphere_normals(intersection_point)

    def point_to_grid_cell(self, grid_info, grid):
        """Map sphere intersection to linear grid cell index."""
        import jax.numpy as jnp
        theta_idx, phi_idx = grid_info
        n_div = grid.n_divisions
        n_theta = n_div
        n_phi = 2 * n_div
        idx = theta_idx * n_phi + phi_idx
        total = n_theta * n_phi
        return jnp.clip(idx, 0, total - 1)

    def assign_sensor_to_cells(self, sensors, sensor_radius, grid):
        """Map sensors to overlapping sphere grid cells."""
        from lucid.propagation.sphere import assign_sensors_to_sphere_grid
        return assign_sensors_to_sphere_grid(sensors, sensor_radius, self.r, grid.n_divisions)

    def grid_cell_centers(self, grid):
        """Compute centers of all sphere grid cells."""
        from lucid.propagation.sphere import calculate_sphere_grid_centers
        return calculate_sphere_grid_centers(self.r, grid.n_divisions)

    def total_grid_cells(self, grid):
        n_div = grid.n_divisions
        return n_div * (2 * n_div)

    def point_to_grid_cell_from_coords(self, coords, grid):
        """Convert grid coordinates to linear index.

        For sphere: coords = [theta_idx, phi_idx].
        """
        theta_idx, phi_idx = int(coords[0]), int(coords[1])
        n_phi = 2 * grid.n_divisions
        return theta_idx * n_phi + phi_idx

    def build_inverted_sensor_map(self, assignments_geometric, assignments_distance,
                                   max_candidates_per_ray, grid):
        """Build cell→sensor lookup table for sphere."""
        from lucid.propagation.sphere import sphere_grid_cells
        from lucid.propagation.sensor_map import build_inverted_sensor_map
        return build_inverted_sensor_map(
            assignments_geometric, assignments_distance,
            sphere_grid_cells(grid.n_divisions), max_candidates_per_ray)