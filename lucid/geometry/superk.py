"""
Super-Kamiokande detector geometry implementation.

Reads real PMT positions from the SK5 ConnectionTable ROOT file
rather than algorithmically placing sensors. Converts ROOT file
positions (in cm) to meters to match the standard config format.
"""

import numpy as np
import plotly.graph_objects as go
from .base import Detector
from .cylinder import Cylinder
from .registry import register_detector


@register_detector('superk')
class SuperK(Cylinder):
    """
    Super-Kamiokande detector geometry.

    Reads PMT positions directly from a ConnectionTable ROOT file
    (e.g. ConnectionTable_SK5.root). Positions in the ROOT file are
    stored in cm and are converted to meters internally to match the
    standard geometry config format used by Cylinder, Sphere, etc.

    Parameters
    ----------
    connection_table_path : str
        Path to the ConnectionTable ROOT file.
    radius : float
        Detector radius in meters (from config, used for wireframe
        drawing and consistency with other detector types).
    height : float
        Detector height in meters (from config).
    n_sensors : int
        Number of sensors (from config; actual count is read from
        the ROOT file and may differ — a warning is printed if so).
    sensor_radius : float
        Radius of individual PMT sensors in meters.
    z_boundary : float, optional
        The |z| threshold (in meters) separating barrel from endcap
        PMTs. Default is 18.0 m (= 1800 cm).
    tree_name : str, optional
        Name of the TTree in the ROOT file. Default is 'ConnectionTable'.
    """

    CM_TO_M = 0.01  # Conversion factor from ROOT file (cm) to config (m)
    DEFAULT_Z_BOUNDARY = 18.0  # meters

    def __init__(self, connection_table_path, radius, height, n_sensors,
                 sensor_radius, z_boundary=None, tree_name='ConnectionTable'):

        self.connection_table_path = connection_table_path
        self.z_boundary = z_boundary if z_boundary is not None else self.DEFAULT_Z_BOUNDARY
        self.tree_name = tree_name

        # Load PMT data from ROOT file (positions converted to meters)
        self._load_connection_table()

        # Use config values for radius/height (for wireframe, compatibility)
        # but warn if they don't match the data
        actual_n = len(self._pmtx)
        if actual_n != n_sensors:
            print(f"SuperK: config says {n_sensors} sensors but ROOT file "
                  f"has {actual_n}. Using actual count from file.")

        # Initialize Detector base (skip Cylinder.__init__ which would place
        # sensors algorithmically — we place them from the ROOT file instead)
        Detector.__init__(self, actual_n, sensor_radius)
        self.r = radius
        self.H = height
        self._n_cap = None
        self._n_angular = None
        self._n_height = None

        # Place sensors using real positions from ConnectionTable
        self.place_photosensors()

    def _load_connection_table(self):
        """
        Load PMT data from the ConnectionTable ROOT file.

        All positions are converted from cm (ROOT file) to meters.
        """
        try:
            import uproot
        except ImportError:
            raise ImportError(
                "The 'uproot' package is required to read ROOT files. "
                "Install it with: pip install uproot awkward"
            )

        f = uproot.open(self.connection_table_path)
        tree = f[self.tree_name]

        # Core position and ID data — convert positions to meters
        self._cableid = tree['cableid'].array(library='np')
        self._pmtx = tree['pmtx'].array(library='np').astype(float) * self.CM_TO_M
        self._pmty = tree['pmty'].array(library='np').astype(float) * self.CM_TO_M
        self._pmtz = tree['pmtz'].array(library='np').astype(float) * self.CM_TO_M
        self._pmtflag = tree['pmtflag'].array(library='np')

        # Electronics and module mapping
        self._supserial = tree['supserial'].array(library='np')
        self._modserial = tree['modserial'].array(library='np')
        self._hutnum = tree['hutnum'].array(library='np')
        self._group = tree['group'].array(library='np')

        # HV info
        self._hvcrate = tree['hvcrate'].array(library='np')
        self._hvmodadd = tree['hvmodadd'].array(library='np')
        self._hvch = tree['hvch'].array(library='np')
        self._oldhv = tree['oldhv'].array(library='np')

        # Production year
        self._prodyear_sk4 = tree['prodyear_sk4'].array(library='np')
        self._prodyear_sk5 = tree['prodyear_sk5'].array(library='np')

        f.close()

    def place_photosensors(self):
        """
        Assign PMT positions from the ConnectionTable data to the
        standard detector arrays (all_points, barr_points, etc.).

        Surface classification uses the z_boundary threshold (in meters):
          - z > z_boundary   →  top endcap
          - z < -z_boundary  →  bottom endcap
          - otherwise        →  barrel
        """
        top_mask = self._pmtz > self.z_boundary
        bottom_mask = self._pmtz < -self.z_boundary
        barrel_mask = ~top_mask & ~bottom_mask

        # Build per-surface point arrays
        self.tcap_points = np.column_stack([
            self._pmtx[top_mask], self._pmty[top_mask], self._pmtz[top_mask]
        ])
        self.bcap_points = np.column_stack([
            self._pmtx[bottom_mask], self._pmty[bottom_mask], self._pmtz[bottom_mask]
        ])
        self.barr_points = np.column_stack([
            self._pmtx[barrel_mask], self._pmty[barrel_mask], self._pmtz[barrel_mask]
        ])

        # Combined array: barrel, then top cap, then bottom cap
        # (matches Cylinder convention)
        self.all_points = np.concatenate(
            [self.barr_points, self.tcap_points, self.bcap_points], axis=0
        )

        # ID mappings (sequential index → position)
        self.ID_to_position = {
            i: self.all_points[i] for i in range(len(self.all_points))
        }

        # Case mappings: 0=barrel, 1=top cap, 2=bottom cap
        n_barr = len(self.barr_points)
        n_tcap = len(self.tcap_points)
        self.ID_to_case = {}
        for i in range(len(self.all_points)):
            if i < n_barr:
                self.ID_to_case[i] = 0
            elif i < n_barr + n_tcap:
                self.ID_to_case[i] = 1
            else:
                self.ID_to_case[i] = 2

        # Cable ID mapping: sequential index → SK cable ID
        barrel_cables = self._cableid[~(self._pmtz > self.z_boundary) &
                                       ~(self._pmtz < -self.z_boundary)]
        top_cables = self._cableid[self._pmtz > self.z_boundary]
        bottom_cables = self._cableid[self._pmtz < -self.z_boundary]
        all_cables = np.concatenate([barrel_cables, top_cables, bottom_cables])
        self.ID_to_cableid = {i: int(c) for i, c in enumerate(all_cables)}
        self.cableid_to_ID = {v: k for k, v in self.ID_to_cableid.items()}

    def get_pmt_info(self, sequential_id):
        """
        Get full PMT information for a given sequential ID.

        Parameters
        ----------
        sequential_id : int
            The sequential index (0-based) in self.all_points.

        Returns
        -------
        dict
            Dictionary with PMT properties. Positions are in meters.
        """
        cable_id = self.ID_to_cableid[sequential_id]
        orig_idx = np.where(self._cableid == cable_id)[0][0]

        surface_names = {0: 'barrel', 1: 'top', 2: 'bottom'}
        return {
            'cable_id': int(cable_id),
            'position': self.all_points[sequential_id].copy(),
            'surface': surface_names[self.ID_to_case[sequential_id]],
            'pmtflag': int(self._pmtflag[orig_idx]),
            'supserial': int(self._supserial[orig_idx]),
            'modserial': int(self._modserial[orig_idx]),
            'group': int(self._group[orig_idx]),
            'hv': float(self._oldhv[orig_idx]),
            'prodyear_sk5': float(self._prodyear_sk5[orig_idx]),
        }

    def visualize_geometry_wireframe(self, show_sensors=True, color_by='surface'):
        """
        Visualize the Super-K detector geometry.

        Parameters
        ----------
        show_sensors : bool
            Whether to show PMT positions.
        color_by : str
            How to color PMTs: 'surface', 'pmtflag', or 'group'.
        """
        fig = go.Figure()

        # Draw cylinder wireframe
        theta = np.linspace(0, 2 * np.pi, 80)
        z_barrel = np.linspace(-self.H/2, self.H/2, 30)
        theta_mesh, z_mesh = np.meshgrid(theta, z_barrel)

        x_barrel = self.r * np.cos(theta_mesh)
        y_barrel = self.r * np.sin(theta_mesh)

        fig.add_trace(go.Surface(
            x=x_barrel, y=y_barrel, z=z_mesh,
            opacity=0.05, showscale=False,
            colorscale=[[0, 'lightblue'], [1, 'lightblue']],
            name='Barrel Surface'
        ))

        # Endcaps
        r_cap = np.linspace(0, self.r, 20)
        theta_cap = np.linspace(0, 2 * np.pi, 80)
        r_mesh, theta_mesh = np.meshgrid(r_cap, theta_cap)
        x_cap = r_mesh * np.cos(theta_mesh)
        y_cap = r_mesh * np.sin(theta_mesh)

        for z_val, color, name in [
            (self.H/2, 'lightgreen', 'Top Cap'),
            (-self.H/2, 'lightcoral', 'Bottom Cap')
        ]:
            fig.add_trace(go.Surface(
                x=x_cap, y=y_cap, z=np.full_like(x_cap, z_val),
                opacity=0.05, showscale=False,
                colorscale=[[0, color], [1, color]],
                name=name
            ))

        if show_sensors:
            if color_by == 'surface':
                self._plot_sensors_by_surface(fig)
            elif color_by == 'pmtflag':
                self._plot_sensors_by_flag(fig)
            elif color_by == 'group':
                self._plot_sensors_by_group(fig)

        fig.update_layout(
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            title=f'Super-Kamiokande Detector ({len(self.all_points)} PMTs)',
            height=800
        )
        fig.show()

    def _plot_sensors_by_surface(self, fig):
        """Add PMT scatter traces colored by surface type."""
        configs = [
            (0, 'blue', 'Barrel'),
            (1, 'green', 'Top Cap'),
            (2, 'red', 'Bottom Cap'),
        ]
        for case_id, color, label in configs:
            indices = [i for i, c in self.ID_to_case.items() if c == case_id]
            if indices:
                pts = self.all_points[indices]
                fig.add_trace(go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode='markers',
                    marker=dict(size=2, color=color, opacity=0.7),
                    name=f'{label} ({len(indices)})'
                ))

    def _plot_sensors_by_flag(self, fig):
        """Add PMT scatter traces colored by pmtflag."""
        top_mask = self._pmtz > self.z_boundary
        bottom_mask = self._pmtz < -self.z_boundary
        barrel_mask = ~top_mask & ~bottom_mask
        flags = np.concatenate([
            self._pmtflag[barrel_mask],
            self._pmtflag[top_mask],
            self._pmtflag[bottom_mask]
        ])
        unique_flags = np.unique(flags)
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta']

        for i, flag in enumerate(unique_flags):
            mask = flags == flag
            pts = self.all_points[mask]
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(size=2, color=colors[i % len(colors)], opacity=0.7),
                name=f'Flag {flag} ({np.sum(mask)})'
            ))

    def _plot_sensors_by_group(self, fig):
        """Add PMT scatter traces colored by group number."""
        top_mask = self._pmtz > self.z_boundary
        bottom_mask = self._pmtz < -self.z_boundary
        barrel_mask = ~top_mask & ~bottom_mask
        groups = np.concatenate([
            self._group[barrel_mask],
            self._group[top_mask],
            self._group[bottom_mask]
        ])

        fig.add_trace(go.Scatter3d(
            x=self.all_points[:, 0],
            y=self.all_points[:, 1],
            z=self.all_points[:, 2],
            mode='markers',
            marker=dict(
                size=2, color=groups, colorscale='Viridis',
                opacity=0.7, showscale=True,
                colorbar=dict(title='Group')
            ),
            name='PMTs by Group'
        ))