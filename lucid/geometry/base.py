"""Base detector class.

Public name: ``Detector``, the abstract base every geometry subclasses. Geometry holds no plotting
code; the 3-D event display is :func:`lucid.visualization.event_display_3d`.
"""

import numpy as np
from abc import ABC, abstractmethod


class Detector(ABC):
    """Base class for detector geometries"""
    
    def __init__(self, n_sensors, sensor_radius):
        """
        Initialize common detector attributes.
        
        Parameters:
        -----------
        n_sensors : int
            Number of photosensors
        sensor_radius : float
            Radius of individual sensors
        """
        self.C = np.array([0.0, 0.0, 0.0])  # Always centered at origin
        self.n_sensors = n_sensors
        self.S_radius = sensor_radius
        
        # These will be set by place_photosensors()
        self.all_points = None
        self.ID_to_position = None
        self.ID_to_case = None

    def _grid_cell_target(self, max_candidates_per_ray):
        """Default grid-cell edge (m): no longer than the closest sensor-to-sensor distance.

        At most one sensor centre then falls in a cell, and the overlap assignment brings a cell
        to about 4 geometric assignments, matching ``max_candidates_per_ray=4``; a larger
        ``max_candidates_per_ray`` scales the edge by ``sqrt(k/4)``. The 0.9 factor was
        calibrated on WCTE (the tightest layout in the repo; 0.92 keeps occupancy at 4). The
        floor at the sensor radius stops a coincident sensor pair from driving a huge grid.
        """
        import math
        scale = math.sqrt(max_candidates_per_ray / 4.0)
        if self.all_points is None or len(self.all_points) < 2:
            return 2 * self.S_radius * scale
        from scipy.spatial import cKDTree
        pts = np.asarray(self.all_points)
        nn_min = float(cKDTree(pts).query(pts, k=2)[0][:, 1].min())
        return max(0.9 * nn_min * scale, self.S_radius * scale)

    @classmethod
    @abstractmethod
    def from_config(cls, geometry, config_dir):
        """Build the detector from a config's ``geometry_definitions``.

        :func:`lucid.geometry.generate_detector` calls this on the class registered under the
        config's ``detector_type``, so registering a subclass that implements it is enough to
        make it constructible from a JSON config.

        Parameters
        ----------
        geometry : dict
            The config's ``geometry_definitions``.
        config_dir : str
            Directory of the config file; relative file paths in ``geometry`` resolve against it.
        """

    @abstractmethod
    def place_photosensors(self):
        """Position the photo sensor centers. Must be implemented by subclasses."""
        pass

    # ── Propagation abstract methods (Phase 9) ─────────────────────────
    # Subclasses implement these to enable the shared create_propagator().
    # bounds_check() was added in Phase 6. The grid is a frozen record from make_grid(),
    # passed in: the detector itself holds no grid state, so one detector can serve several
    # propagators.

    def make_grid(self, max_candidates_per_ray=4, **grid_params):
        """The propagation grid for this layout: a frozen, geometry-specific NamedTuple.

        Parameters
        ----------
        max_candidates_per_ray : int
            Sensor slots per cell the default sizing aims at.
        **grid_params
            Explicit cell counts (they win over the default sizing).
        """
        raise NotImplementedError

    def intersect_ray(self, origins, directions, grid):
        """Batch ray-geometry intersection.

        Parameters
        ----------
        origins : jnp.ndarray, shape (n_rays, 3)
        directions : jnp.ndarray, shape (n_rays, 3)
        grid : the record from make_grid

        Returns
        -------
        intersection_point : jnp.ndarray, shape (n_rays, 3)
        t_value : jnp.ndarray, shape (n_rays,)
        grid_info : geometry-specific opaque data passed to point_to_grid_cell
        surface_info : geometry-specific data passed to compute_normal
        """
        raise NotImplementedError

    def compute_normal(self, intersection_point, surface_info):
        """Compute outward surface normals at intersection points.

        Parameters
        ----------
        intersection_point : jnp.ndarray, shape (n_rays, 3)
        surface_info : from intersect_ray

        Returns
        -------
        normals : jnp.ndarray, shape (n_rays, 3), outward convention
        """
        raise NotImplementedError

    def point_to_grid_cell(self, grid_info, grid):
        """Map intersection data to linear grid cell indices.

        Parameters
        ----------
        grid_info : from intersect_ray
        grid : the record from make_grid

        Returns
        -------
        linear_idx : jnp.ndarray, shape (n_rays,), int
        """
        raise NotImplementedError

    def assign_sensor_to_cells(self, sensors, sensor_radius, grid):
        """Map each sensor to the grid cells it overlaps.

        Parameters
        ----------
        sensors : jnp.ndarray, shape (n_sensors, 3)
        sensor_radius : float
        grid : the record from make_grid

        Returns
        -------
        assignments : jnp.ndarray, geometry-specific shape
        """
        raise NotImplementedError

    def grid_cell_centers(self, grid):
        """Compute (n_cells, 3) array of grid cell center positions of ``grid``.

        Returns
        -------
        centers : jnp.ndarray, shape (n_total_cells, 3)
        """
        raise NotImplementedError

    def total_grid_cells(self, grid):
        """Return total number of cells of ``grid``.

        Returns
        -------
        int
        """
        raise NotImplementedError
