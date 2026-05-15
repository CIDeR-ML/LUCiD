"""
Cylindrical detector geometry implementation.
"""
from __future__ import annotations

from typing import Optional, Union

import jax
import numpy as np
import plotly.graph_objects as go
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


@register_detector('cylinder')
class Cylinder(Detector):
    """Cylindrical detector geometry.

    Two construction paths share this class:

    * The default :meth:`__init__` algorithmically tiles the barrel
      and end caps with ``n_sensors`` photosensors.
    * :meth:`from_pmt_file` instead loads measured PMT positions from
      a `.npz` file (see ``PMT_NPZ_SCHEMA.md``) — used for SK, HK,
      WCTE, SK_official and any other geometry whose layout is given
      rather than computed.
    """

    def __init__(self, radius: float, height: float, n_sensors: int, sensor_radius: float) -> None:
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
        # Grid params — set via configure_grid() before using propagation methods
        self._n_cap = None
        self._n_angular = None
        self._n_height = None
        self.place_photosensors()

    def configure_grid(self, n_cap: Optional[int] = None, n_angular: Optional[int] = None, n_height: Optional[int] = None,
                        max_candidates_per_ray: int = 4) -> None:
        """Set grid parameters for propagation methods.

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
        import numpy as np

        scale = math.sqrt(max_candidates_per_ray / 4.0)

        if self.all_points is not None and len(self.all_points) >= 2:
            from scipy.spatial import cKDTree
            pts = np.asarray(self.all_points)
            tree = cKDTree(pts)
            dists, _ = tree.query(pts, k=2)
            nn_min = float(dists[:, 1].min())
            # Safety factor — calibrated against WCTE (mPMT dome,
            # tightest layout in the repo): cell ≤ 0.92 × nn_min
            # keeps max occupancy at 4. 0.9 leaves headroom.
            target = 0.9 * nn_min * scale
            # Floor: never go below the sensor radius. Without this,
            # a single buggy sensor pair coincident-by-mm in the npz
            # would drive a multi-GB grid. The floor is well below
            # the natural target for every real detector in the repo
            # (WCTE has the tightest packing at 0.9*nn_min ≈ 62mm,
            # vs floor of 40mm). When the floor activates, expect
            # some cells to exceed max_candidates_per_ray — the
            # validator in shared.py will emit a warning.
            target = max(target, self.S_radius * scale)
        else:
            # No sensors placed yet, or only one — fall back to a
            # sensor-size-based target so callers get something sane.
            target = 2 * self.S_radius * scale

        self._n_angular = n_angular if n_angular is not None else max(
            10, int(math.ceil(2 * math.pi * self.r / target)))
        self._n_height = n_height if n_height is not None else max(
            10, int(math.ceil(self.H / target)))
        self._n_cap = n_cap if n_cap is not None else max(
            10, int(math.ceil(2 * self.r / target)))


    def place_photosensors(self) -> None:
        """Position the photo sensor centers proportionally by surface area."""
        barrel_area = 2 * np.pi * self.r * self.H
        caps_area = 2 * np.pi * self.r**2  # Both caps combined
        total_area = barrel_area + caps_area
        
        n_barrel = int(self.n_sensors * barrel_area / total_area)
        n_caps = self.n_sensors - n_barrel
        n_per_cap = n_caps // 2  # Split equally between top and bottom
        
        self.barr_points = self._place_barrel_sensors(n_barrel)
        self.tcap_points = self._place_cap_sensors(n_per_cap, self.H/2)
        self.bcap_points = self._place_cap_sensors(n_per_cap, -self.H/2)

        self.all_points = np.concatenate([self.barr_points, self.tcap_points, self.bcap_points], axis=0)

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

    # ── Construction from a PMT-positions npz file ────────────────────

    @classmethod
    def from_pmt_file(cls, npz_file_path: str, snap_to_wall: bool = True) -> Cylinder:
        """Build a :class:`Cylinder` whose PMT positions are read from
        a unified-schema ``.npz`` file (see ``PMT_NPZ_SCHEMA.md``).

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
                    f"key '{key}'. See lucid/geometry/PMT_NPZ_SCHEMA.md."
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
        instance._n_cap = None
        instance._n_angular = None
        instance._n_height = None
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

    def _snap_to_wall(self, positions: np.ndarray, surfaces: np.ndarray) -> np.ndarray:
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

    def _place_barrel_sensors(self, n_sensors: int) -> np.ndarray:
        """Place sensors on barrel surface with rectangular grid."""
        if n_sensors == 0:
            return np.array([]).reshape(0, 3)
            
        height_eff = self.H - 3 * self.S_radius
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
        
        z_positions = np.linspace(-height_eff/2, height_eff/2, n_rows) + self.C[2]
        theta_positions = np.linspace(0, 2*np.pi, n_cols, endpoint=False)
        
        points = []
        for z in z_positions:
            for theta in theta_positions:
                x = self.r * np.cos(theta) + self.C[0]
                y = self.r * np.sin(theta) + self.C[1]
                points.append([x, y, z])
        
        return np.array(points[:n_sensors])  # Trim to exact count

    def _place_cap_sensors(self, n_sensors: int, z_position: float) -> np.ndarray:
        """Place sensors on cap surface with concentric hexagonal rings."""
        if n_sensors == 0:
            return np.array([]).reshape(0, 3)
            
        radius_eff = self.r - 1.5 * self.S_radius
        
        if radius_eff <= 0:
            return np.array([]).reshape(0, 3)
        
        hex_points = generate_concentric_hexagons(n_sensors, radius_eff)

        points_3d = np.zeros((len(hex_points), 3))
        points_3d[:, 0] = hex_points[:, 0] + self.C[0]
        points_3d[:, 1] = hex_points[:, 1] + self.C[1]
        points_3d[:, 2] = z_position + self.C[2]
        
        return points_3d

    def visualize_geometry_wireframe(self, show_sensors: bool = True) -> None:
        """Visualize the cylinder as a wireframe with detectors"""
        fig = go.Figure()

        theta = np.linspace(0, 2 * np.pi, 50)
        z_barrel = np.linspace(-self.H/2, self.H/2, 20)
        theta_mesh, z_mesh = np.meshgrid(theta, z_barrel)
        
        x_barrel = self.r * np.cos(theta_mesh) + self.C[0]
        y_barrel = self.r * np.sin(theta_mesh) + self.C[1]
        z_barrel_mesh = z_mesh + self.C[2]

        fig.add_trace(go.Surface(
            x=x_barrel, y=y_barrel, z=z_barrel_mesh,
            opacity=0.1,
            showscale=False,
            colorscale=[[0, 'lightblue'], [1, 'lightblue']],
            name='Barrel Surface'
        ))

        r_cap = np.linspace(0, self.r, 20)
        theta_cap = np.linspace(0, 2 * np.pi, 50)
        r_mesh, theta_mesh = np.meshgrid(r_cap, theta_cap)
        
        x_top = r_mesh * np.cos(theta_mesh) + self.C[0]
        y_top = r_mesh * np.sin(theta_mesh) + self.C[1]
        z_top = np.full_like(x_top, self.H/2 + self.C[2])

        fig.add_trace(go.Surface(
            x=x_top, y=y_top, z=z_top,
            opacity=0.1,
            showscale=False,
            colorscale=[[0, 'lightgreen'], [1, 'lightgreen']],
            name='Top Cap'
        ))

        z_bottom = np.full_like(x_top, -self.H/2 + self.C[2])

        fig.add_trace(go.Surface(
            x=x_top, y=y_top, z=z_bottom,
            opacity=0.1,
            showscale=False,
            colorscale=[[0, 'lightcoral'], [1, 'lightcoral']],
            name='Bottom Cap'
        ))

        if show_sensors:
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

    def bounds_check(self, positions: jax.Array) -> jax.Array:
        """Test whether positions are inside the cylinder."""
        import jax.numpy as jnp
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        inside_xy = (x ** 2 + y ** 2) <= self.r ** 2
        inside_z = (z >= -self.H / 2) & (z <= self.H / 2)
        return inside_xy & inside_z

    # ── Propagation methods (Phase 9) ──────────────────────────────────

    def intersect_ray(self, origins: jax.Array, directions: jax.Array) -> tuple[jax.Array, jax.Array, tuple, tuple]:
        """Batch ray-cylinder intersection with grid indexing."""
        from lucid.propagation.cylinder import batch_intersect_cylinder_with_grid
        n_cap = self._n_cap
        n_angular = self._n_angular
        n_height = self._n_height
        results = batch_intersect_cylinder_with_grid(
            origins, directions, self.r, self.H, n_cap, n_angular, n_height)
        # Returns: (intersects, t, is_wall, is_top_cap, wall_indices, cap_indices, intersection_point)
        intersects, t, is_wall, is_top_cap, wall_indices, cap_indices, intersection_point = results
        grid_info = (is_wall, is_top_cap, wall_indices, cap_indices)
        surface_info = (is_wall, is_top_cap)
        return intersection_point, t, grid_info, surface_info

    def compute_normal(self, intersection_point: jax.Array, surface_info: tuple[jax.Array, jax.Array]) -> jax.Array:
        """Compute outward cylinder surface normals."""
        from lucid.propagation.cylinder import calculate_cylinder_normals
        is_wall, is_top_cap = surface_info
        return calculate_cylinder_normals(intersection_point, is_wall, is_top_cap)

    def point_to_grid_cell(self, grid_info: tuple[jax.Array, jax.Array, jax.Array, jax.Array]) -> jax.Array:
        """Map cylinder intersection to linear grid cell index."""
        import jax.numpy as jnp
        is_wall, is_top_cap, wall_indices, cap_indices = grid_info
        n_cap = self._n_cap
        n_angular = self._n_angular
        n_height = self._n_height

        wall_linear = wall_indices[:, 0] * n_height + wall_indices[:, 1]
        cap_linear = cap_indices[:, 0] * n_cap + cap_indices[:, 1]
        n_wall_cells = n_angular * n_height

        idx = jnp.where(is_wall, wall_linear,
                        jnp.where(is_top_cap,
                                  n_wall_cells + cap_linear,
                                  n_wall_cells + n_cap * n_cap + cap_linear))
        total_cells = n_wall_cells + 2 * n_cap * n_cap
        return jnp.clip(idx, 0, total_cells - 1)

    def assign_sensor_to_cells(self, sensors: jax.Array, sensor_radius: float) -> jax.Array:
        """Map sensors to overlapping cylinder grid cells."""
        from lucid.propagation.cylinder import assign_sensors_to_grid
        n_cap = self._n_cap
        n_angular = self._n_angular
        n_height = self._n_height
        return assign_sensors_to_grid(
            sensors, sensor_radius, self.r, self.H, n_cap, n_angular, n_height)

    def grid_cell_centers(self) -> jax.Array:
        """Compute centers of all cylinder grid cells."""
        from lucid.propagation.cylinder import calculate_grid_centers
        n_cap = self._n_cap
        n_angular = self._n_angular
        n_height = self._n_height
        return calculate_grid_centers(self.r, self.H, n_cap, n_angular, n_height)

    def total_grid_cells(self) -> int:
        n_cap = self._n_cap
        n_angular = self._n_angular
        n_height = self._n_height
        return n_angular * n_height + 2 * n_cap * n_cap

    def cell_index_to_coords(self, linear_idx: Union[int, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Decode linear index to (cell_i, cell_j, cell_k) for cylinder."""
        import jax.numpy as jnp
        n_cap = self._n_cap
        n_angular = self._n_angular
        n_height = self._n_height
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

    def point_to_grid_cell_from_coords(self, coords: Union[list, np.ndarray, jax.Array]) -> int:
        """Convert grid coordinates (from assignment array) to linear index.

        For cylinder: coords = [angular_idx, height_idx, part_type]
        where part_type: 0=wall, 1=top_cap, 2=bottom_cap.
        """
        cell_i, cell_j, cell_k = int(coords[0]), int(coords[1]), int(coords[2])
        n_wall = self._n_angular * self._n_height
        if cell_k == 0:  # wall
            return cell_i * self._n_height + cell_j
        elif cell_k == 1:  # top cap
            return n_wall + cell_i * self._n_cap + cell_j
        else:  # bottom cap
            return n_wall + self._n_cap * self._n_cap + cell_i * self._n_cap + cell_j

    def build_inverted_sensor_map(self, assignments_geometric: jax.Array, assignments_distance: jax.Array,
                                   max_candidates_per_ray: int, num_sensors: int) -> jax.Array:
        """Build cell→sensor lookup table for cylinder."""
        from lucid.propagation.cylinder import create_inverted_sensor_map
        n_cap = self._n_cap
        n_angular = self._n_angular
        n_height = self._n_height
        return create_inverted_sensor_map(
            assignments_geometric, assignments_distance,
            n_cap, n_angular, n_height, max_candidates_per_ray, num_sensors)