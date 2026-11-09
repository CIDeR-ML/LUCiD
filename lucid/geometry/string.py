"""
String-telescope detector geometry.

A detector consisting of vertical (or near-vertical) strings of DOMs
(Digital Optical Modules) in an open medium (ice or water). There is
no closed surface — photons exiting the envelope are lost.

The detector is bounded by a cylindrical envelope used for ray-entry/exit
clipping and the inside-detector check.
"""

import os
import numpy as np
import jax.numpy as jnp

from .base import Detector
from .registry import register_detector


@register_detector('string')
class StringTelescope(Detector):
    """String-array neutrino telescope geometry.

    Stores per-DOM positions as the canonical data source. Derives
    per-string fitted axes, arc-length offsets, and curvature at
    construction time.

    Parameters
    ----------
    dom_xyz : (N_str, max_dom, 3) — per-DOM positions, NaN-padded
    n_dom_per_str : (N_str,) int — actual DOM count per string
    sensor_radius : float — DOM glass sphere radius (m)
    envelope_radius : float — bounding cylinder radius
    envelope_z_min, envelope_z_max : float — bounding cylinder z range
    """

    def __init__(self, dom_xyz, n_dom_per_str, sensor_radius,
                 envelope_radius, envelope_z_min, envelope_z_max):
        n_str = len(dom_xyz)
        max_dom = dom_xyz.shape[1]
        total_doms = int(np.sum(n_dom_per_str))

        super().__init__(n_sensors=total_doms, sensor_radius=sensor_radius)

        self.n_str = n_str
        self.max_dom = max_dom
        self.n_dom_per_str_np = np.asarray(n_dom_per_str, dtype=np.int32)

        self.envelope_radius = float(envelope_radius)
        self.envelope_z_min = float(envelope_z_min)
        self.envelope_z_max = float(envelope_z_max)

        self._build_tables(np.asarray(dom_xyz, dtype=np.float64))
        self.place_photosensors()

    def _build_tables(self, dom_xyz_raw):
        """Derive all per-string and per-DOM tables from raw positions."""
        n_str = self.n_str
        max_dom = self.max_dom

        # ── Per-string fitted axis ──
        self.string_anchors = np.zeros((n_str, 3))
        self.string_tops = np.zeros((n_str, 3))
        self.string_axes = np.zeros((n_str, 3))
        self.string_lengths = np.zeros(n_str)

        for i in range(n_str):
            n = self.n_dom_per_str_np[i]
            self.string_anchors[i] = dom_xyz_raw[i, 0]
            self.string_tops[i] = dom_xyz_raw[i, n - 1]
            diff = self.string_tops[i] - self.string_anchors[i]
            length = np.linalg.norm(diff)
            self.string_lengths[i] = length
            self.string_axes[i] = diff / (length + 1e-30)

        # ── Per-DOM arc-length offsets along fitted axis ──
        self.dom_s_offsets = np.full((n_str, max_dom), np.inf, dtype=np.float64)
        self.dom_global_ids = np.full((n_str, max_dom), -1, dtype=np.int32)
        self.string_s_min = np.zeros(n_str)
        self.string_s_max = np.zeros(n_str)

        # Flatten all DOM positions for self.all_points
        all_positions = []
        global_id = 0

        for i in range(n_str):
            n = self.n_dom_per_str_np[i]
            for k in range(n):
                pos = dom_xyz_raw[i, k]
                offset_vec = pos - self.string_anchors[i]
                s = np.dot(offset_vec, self.string_axes[i])
                self.dom_s_offsets[i, k] = s
                self.dom_global_ids[i, k] = global_id
                all_positions.append(pos.copy())
                global_id += 1
            if n > 0:
                # Sort DOMs by arc-length so bracket-finding snap works
                sort_idx = np.argsort(self.dom_s_offsets[i, :n])
                self.dom_s_offsets[i, :n] = self.dom_s_offsets[i, :n][sort_idx]
                self.dom_global_ids[i, :n] = self.dom_global_ids[i, :n][sort_idx]
                self.string_s_min[i] = self.dom_s_offsets[i, 0]
                self.string_s_max[i] = self.dom_s_offsets[i, n - 1]

        self._all_positions = np.array(all_positions)

        # ── Per-string curvature (max perpendicular DOM-from-axis offset) ──
        self.string_curv = np.zeros(n_str)
        self.perp_delta_vec = np.zeros((n_str, 3))

        for i in range(n_str):
            n = self.n_dom_per_str_np[i]
            max_perp = 0.0
            max_perp_vec = np.zeros(3)
            for k in range(n):
                pos = dom_xyz_raw[i, k]
                proj = self.string_anchors[i] + self.dom_s_offsets[i, k] * self.string_axes[i]
                perp_vec = pos - proj
                perp = np.linalg.norm(perp_vec)
                if perp > max_perp:
                    max_perp = perp
                    max_perp_vec = perp_vec
            self.string_curv[i] = max_perp
            self.perp_delta_vec[i] = max_perp_vec

    def place_photosensors(self):
        """Set self.all_points from the flattened DOM position table."""
        self.all_points = self._all_positions
        self.ID_to_position = {i: self.all_points[i] for i in range(self.n_sensors)}

    @property
    def volume(self):
        """The cylindrical envelope (:mod:`lucid.geometry.volume`)."""
        from lucid.geometry.volume import CylinderVolume
        return CylinderVolume(self.envelope_radius, self.envelope_z_min, self.envelope_z_max)

    def bounds_check(self, positions):
        """Check if points are inside the cylindrical envelope."""
        return self.volume.contains(positions)

    def get_jax_tables(self):
        """Convert tables to JAX arrays for the propagator."""
        return {
            'string_anchors': jnp.array(self.string_anchors),
            'string_tops': jnp.array(self.string_tops),
            'string_axes': jnp.array(self.string_axes),
            'dom_s_offsets': jnp.array(self.dom_s_offsets),
            'dom_global_ids': jnp.array(self.dom_global_ids),
            'string_s_min': jnp.array(self.string_s_min),
            'string_s_max': jnp.array(self.string_s_max),
            'string_curv': jnp.array(self.string_curv),
            'n_str': self.n_str,
            'max_dom': self.max_dom,
            'envelope_radius': self.envelope_radius,
            'envelope_z_min': self.envelope_z_min,
            'envelope_z_max': self.envelope_z_max,
        }

    @classmethod
    def from_npz(cls, npz_path):
        """Load a StringTelescope from an NPZ file."""
        data = np.load(npz_path)
        return cls(
            dom_xyz=data['dom_xyz'],
            n_dom_per_str=data['n_dom_per_str'],
            sensor_radius=float(data['sensor_radius']),
            envelope_radius=float(data['envelope_radius']),
            envelope_z_min=float(data['envelope_z_min']),
            envelope_z_max=float(data['envelope_z_max']),
        )

    @classmethod
    def from_config(cls, geometry, config_dir):
        """Per-DOM positions from the NPZ that ``geometry['npz_file_path']`` names."""
        return cls.from_npz(os.path.join(config_dir, geometry['npz_file_path']))

    # ── The surface-propagator interface: unused, strings have their own propagator ──

    def intersect_ray(self, origins, directions, grid):
        raise NotImplementedError("StringTelescope uses create_string_propagator, not create_propagator")

    def compute_normal(self, point, surface_info):
        raise NotImplementedError("StringTelescope uses create_string_propagator, not create_propagator")

    def point_to_grid_cell(self, grid_info, grid):
        raise NotImplementedError("StringTelescope uses create_string_propagator, not create_propagator")

    def assign_sensor_to_cells(self, sensors, sensor_radius, grid):
        raise NotImplementedError("StringTelescope uses create_string_propagator, not create_propagator")

    def grid_cell_centers(self, grid):
        raise NotImplementedError("StringTelescope uses create_string_propagator, not create_propagator")

    def total_grid_cells(self, grid):
        raise NotImplementedError("StringTelescope uses create_string_propagator, not create_propagator")

