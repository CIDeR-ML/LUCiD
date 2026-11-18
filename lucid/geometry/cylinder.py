"""
Cylindrical detector geometry implementation.
"""

import os
from typing import NamedTuple

import numpy as np
from .base import Detector
from .utils import generate_concentric_hexagons
from .registry import register_detector


_FROM_FILE_REQUIRED_ARRAYS = (
    'positions_mm', 'directions', 'surfaces', 'pmt_id',
)
_FROM_FILE_REQUIRED_SCALARS = (
    'radius', 'height', 'sensor_radius',
)
_FROM_FILE_RESERVED = set(_FROM_FILE_REQUIRED_ARRAYS) | set(_FROM_FILE_REQUIRED_SCALARS)
_FROM_FILE_MM_TO_M = 0.001
_INACTIVE_PREFIX = 'inactive_'


class CylinderGrid(NamedTuple):
    """The propagation grid, frozen: cells per cap side, around the barrel, and along it."""
    n_cap: int
    n_angular: int
    n_height: int


@register_detector('cylinder')
class Cylinder(Detector):
    """Cylindrical detector geometry.

    Two construction paths share this class:

    * The default :meth:`__init__` algorithmically tiles the barrel
      and end caps with ``n_sensors`` photosensors.
    * :meth:`from_pmt_file` instead loads measured PMT positions from
      a `.npz` file (see ``docs/reference/pmt-npz-schema.md``) — used for SK, HK,
      WCTE, SK_official and any other geometry whose layout is given
      rather than computed.
    """

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

    def make_grid(self, n_cap=None, n_angular=None, n_height=None,
                  max_candidates_per_ray=4):
        """The propagation grid for this layout, as a :class:`CylinderGrid`.

        The default sizing rule keeps each grid-cell edge **no longer
        than the smallest sensor-to-sensor distance** in the detector
        (with a safety factor). That guarantees at most one sensor
        centre per cell; the 4-way overlap in
        :func:`assign_sensors_to_grid` then brings each cell to at
        most ~4 geometric assignments — matching the default
        ``max_candidates_per_ray=4``.

        The previous area/safety-factor heuristic worked for
        uniformly-placed sensors (SK, HK) but broke for clustered
        mPMT layouts: when 19 PMTs of a dome project onto a small
        patch of the cylinder wall, cells sized by global area end
        up several sensors too coarse and the inverse-map builder
        drops the overflowers. The nearest-neighbour rule adapts
        to whatever the sensor layout actually is.

        Explicit ``n_angular`` / ``n_height`` / ``n_cap`` arguments
        still win; the rule only supplies defaults for what is not
        passed.

        ``max_candidates_per_ray`` scales the target cell area
        proportionally — a caller passing ``max_candidates_per_ray=16``
        gets a coarser grid (edge ~2× larger) because each cell is
        allowed to hold ~4× more sensors.
        """
        import math
        target = self._grid_cell_target(max_candidates_per_ray)
        return CylinderGrid(
            n_cap=n_cap if n_cap is not None else max(10, int(math.ceil(2 * self.r / target))),
            n_angular=n_angular if n_angular is not None else max(
                10, int(math.ceil(2 * math.pi * self.r / target))),
            n_height=n_height if n_height is not None else max(10, int(math.ceil(self.H / target))))

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
        
        for i in range(len(self.all_points)):
            if i < n_barr:
                self.ID_to_case[i] = 0
            elif i < n_barr + n_tcap:
                self.ID_to_case[i] = 1
            else:
                self.ID_to_case[i] = 2

    @classmethod
    def from_config(cls, geometry, config_dir):
        """Measured PMTs when ``geometry`` names an ``npz_file_path``, else the algorithmic layout."""
        if 'npz_file_path' in geometry:
            return cls.from_pmt_file(os.path.join(config_dir, geometry['npz_file_path']))
        return cls(geometry['radius'], geometry['height'],
                   geometry['n_sensors'], geometry['sensor_radius'])

    # ── Construction from a PMT-positions npz file ────────────────────

    @classmethod
    def from_pmt_file(cls, npz_file_path, snap_to_wall=True):
        """Build a :class:`Cylinder` whose PMT positions are read from
        a unified-schema ``.npz`` file (see ``docs/reference/pmt-npz-schema.md``).

        Parameters
        ----------
        npz_file_path : str
            Path to the ``.npz`` file.
        snap_to_wall : bool, optional
            Project barrel PMTs to ``r_xy = radius`` and cap PMTs to
            ``z = ±height/2``. Default True. Geofiles can leave PMTs
            up to a few centimetres off the nominal surface (mPMT
            domes, alternating barrel rings); snapping puts them on
            the geometric boundary so the cylinder ray-tracer treats
            them correctly. The pre-snap positions are kept on
            ``self.raw_positions``.

        Returns
        -------
        Cylinder
            A cylinder whose ``all_points`` array holds the active
            sensors from the file (barrel first, then top cap, then
            bottom cap). Per-PMT metadata arrays found in the file
            are reordered to match ``all_points`` and attached as
            attributes named after their npz key. Inactive PMTs (any
            ``inactive_*`` arrays) are attached as-is.
        """
        data = np.load(npz_file_path, allow_pickle=True)

        # ── Validate schema ──
        for key in _FROM_FILE_REQUIRED_ARRAYS + _FROM_FILE_REQUIRED_SCALARS:
            if key not in data.files:
                raise KeyError(
                    f"PMT npz '{npz_file_path}' is missing required "
                    f"key '{key}'. See docs/reference/pmt-npz-schema.md."
                )

        positions_m = data['positions_mm'].astype(float) * _FROM_FILE_MM_TO_M
        directions = data['directions'].astype(float)
        surfaces = np.asarray(data['surfaces'])
        n = positions_m.shape[0]
        if directions.shape != (n, 3):
            raise ValueError(
                f"PMT npz '{npz_file_path}': directions shape "
                f"{directions.shape} does not match positions ({n}, 3)."
            )
        if surfaces.shape != (n,):
            raise ValueError(
                f"PMT npz '{npz_file_path}': surfaces shape "
                f"{surfaces.shape} does not match positions count {n}."
            )

        sensor_radius = float(data['sensor_radius'])
        radius = float(data['radius'])
        height = float(data['height'])

        # ── Build the instance, bypassing __init__ (which would do
        # the algorithmic placement) ──
        instance = cls.__new__(cls)
        Detector.__init__(instance, n, sensor_radius)
        instance.r = radius
        instance.H = height
        instance.npz_file_path = npz_file_path
        instance.snap_to_wall = bool(snap_to_wall)

        # ── Reorder to (barrel, top, bottom) and place ──
        order = np.concatenate([
            np.where(surfaces == 'barrel')[0],
            np.where(surfaces == 'top')[0],
            np.where(surfaces == 'bottom')[0],
        ])
        if order.size != n:
            unknown = np.setdiff1d(surfaces, ['barrel', 'top', 'bottom'])
            raise ValueError(
                f"PMT npz '{npz_file_path}': unknown surface labels "
                f"in active block: {unknown.tolist()}"
            )
        instance._reorder_indices = order

        positions_ord = positions_m[order]
        instance.raw_positions = positions_ord.copy()
        if instance.snap_to_wall:
            positions_ord = instance._snap_to_wall(positions_ord, surfaces[order])

        n_barrel = int(np.sum(surfaces == 'barrel'))
        n_top    = int(np.sum(surfaces == 'top'))
        instance.all_points = positions_ord
        instance.barr_points = positions_ord[:n_barrel]
        instance.tcap_points = positions_ord[n_barrel:n_barrel + n_top]
        instance.bcap_points = positions_ord[n_barrel + n_top:]

        # PMT viewing directions, reordered. Not used by the propagator
        # today (which uses the cylinder normal), but kept for any
        # future angular-response physics.
        instance.pmt_directions = directions[order]

        # Standard mappings
        instance.ID_to_position = {
            i: instance.all_points[i] for i in range(n)
        }
        case = np.empty(n, dtype=int)
        case[:n_barrel] = 0
        case[n_barrel:n_barrel + n_top] = 1
        case[n_barrel + n_top:] = 2
        instance.ID_to_case = {i: int(case[i]) for i in range(n)}

        # PMT-id → sequential index lookup (always built; convenient
        # for indexing real-data hits back into all_points order).
        pmt_id = np.asarray(data['pmt_id'])[order]
        instance.pmt_id = pmt_id
        instance.pmt_id_to_idx = {int(pid): i for i, pid in enumerate(pmt_id)}

        # ── Attach all other arrays from the npz as attributes ──
        # Per-active-PMT arrays (first dim == N) get reordered;
        # inactive_* arrays and scalars are stored as-is.
        for key in data.files:
            if key in _FROM_FILE_RESERVED or key == 'pmt_id':
                continue
            arr = data[key]
            if key.startswith(_INACTIVE_PREFIX):
                setattr(instance, key, arr)
                continue
            if arr.ndim >= 1 and arr.shape[0] == n:
                setattr(instance, key, arr[order])
            else:
                setattr(instance, key, arr)

        return instance

    def _snap_to_wall(self, positions, surfaces):
        """Project barrel PMTs radially onto ``r=self.r`` and cap PMTs
        axially onto ``z=±self.H/2``. ``positions`` and ``surfaces``
        must already share the same row order."""
        out = positions.copy()
        barrel = surfaces == 'barrel'
        xy = out[barrel, :2]
        r_xy = np.linalg.norm(xy, axis=1, keepdims=True)
        r_xy = np.where(r_xy > 0, r_xy, 1.0)  # guard against r=0
        out[barrel, :2] = xy * (self.r / r_xy)
        out[surfaces == 'top', 2] = self.H / 2
        out[surfaces == 'bottom', 2] = -self.H / 2
        return out

    # ── Algorithmic placement helpers ──────────────────────────────────

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

    @property
    def volume(self):
        """The instrumented volume (:mod:`lucid.geometry.volume`), built from ``r`` and ``H``."""
        from lucid.geometry.volume import CylinderVolume
        return CylinderVolume(self.r, -self.H / 2, self.H / 2)

    def bounds_check(self, positions):
        """Test whether positions are inside the cylinder."""
        return self.volume.contains(positions)

    # ── Propagation methods (Phase 9) ──────────────────────────────────

    def intersect_ray(self, origins, directions, grid):
        """Batch ray-cylinder intersection with grid indexing."""
        from lucid.propagation.cylinder import batch_intersect_cylinder_with_grid
        results = batch_intersect_cylinder_with_grid(
            origins, directions, self.r, self.H, grid.n_cap, grid.n_angular, grid.n_height)
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

    def point_to_grid_cell(self, grid_info, grid):
        """Map cylinder intersection to linear grid cell index."""
        import jax.numpy as jnp
        is_wall, is_top_cap, wall_indices, cap_indices = grid_info
        n_cap, n_angular, n_height = grid

        wall_linear = wall_indices[:, 0] * n_height + wall_indices[:, 1]
        cap_linear = cap_indices[:, 0] * n_cap + cap_indices[:, 1]
        n_wall_cells = n_angular * n_height

        idx = jnp.where(is_wall, wall_linear,
                        jnp.where(is_top_cap,
                                  n_wall_cells + cap_linear,
                                  n_wall_cells + n_cap * n_cap + cap_linear))
        total_cells = n_wall_cells + 2 * n_cap * n_cap
        return jnp.clip(idx, 0, total_cells - 1)

    def assign_sensor_to_cells(self, sensors, sensor_radius, grid):
        """Map sensors to overlapping cylinder grid cells."""
        from lucid.propagation.cylinder import assign_sensors_to_grid
        return assign_sensors_to_grid(
            sensors, sensor_radius, self.r, self.H, grid.n_cap, grid.n_angular, grid.n_height)

    def grid_cell_centers(self, grid):
        """Compute centers of all cylinder grid cells."""
        from lucid.propagation.cylinder import calculate_grid_centers
        return calculate_grid_centers(self.r, self.H, grid.n_cap, grid.n_angular, grid.n_height)

    def total_grid_cells(self, grid):
        return grid.n_angular * grid.n_height + 2 * grid.n_cap * grid.n_cap

    def point_to_grid_cell_from_coords(self, coords, grid):
        """Convert grid coordinates (from assignment array) to linear index.

        For cylinder: coords = [angular_idx, height_idx, part_type]
        where part_type: 0=wall, 1=top_cap, 2=bottom_cap.
        """
        cell_i, cell_j, cell_k = int(coords[0]), int(coords[1]), int(coords[2])
        n_wall = grid.n_angular * grid.n_height
        if cell_k == 0:  # wall
            return cell_i * grid.n_height + cell_j
        elif cell_k == 1:  # top cap
            return n_wall + cell_i * grid.n_cap + cell_j
        else:  # bottom cap
            return n_wall + grid.n_cap * grid.n_cap + cell_i * grid.n_cap + cell_j

    def build_inverted_sensor_map(self, assignments_geometric, assignments_distance,
                                   max_candidates_per_ray, grid):
        """Build cell→sensor lookup table for cylinder."""
        from lucid.propagation.cylinder import cylinder_grid_cells
        from lucid.propagation.sensor_map import build_inverted_sensor_map
        return build_inverted_sensor_map(
            assignments_geometric, assignments_distance,
            cylinder_grid_cells(grid.n_cap, grid.n_angular, grid.n_height),
            max_candidates_per_ray)