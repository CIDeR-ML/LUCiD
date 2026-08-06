"""Polygonal-prism detector geometry — the real SK inner-detector shape.

SK's barrel is not a cylinder. WCSim builds it as a ``G4Polyhedra`` with
``WCBarrelNumPMTHorizontal / WCPMTperCellHorizontal = 152/4 = 38`` sides, so the wall is
a regular 38-gon prism of apothem ``WCIDDiameter/2 = 16.840750 m`` and height
``WCIDHeight = 36.200 m``. Panel ``k`` spans azimuth [k*dphi, (k+1)*dphi], dphi = 2*pi/38.

:class:`PolygonalCylinder` subclasses :class:`~lucid.geometry.cylinder.Cylinder` so the
npz loading, the shared linear cell packing and the inverse-map builder are reused
verbatim. ``self.r`` is deliberately set to the **apothem**, i.e. the inscribed circle,
so every inherited use of ``self.r`` degrades to a conservative bound rather than a
wrong one.

Sensor placement follows the physics rather than the wall: a photocathode is a spherical
cap, and the sphere it belongs to is centred *behind* the blacksheet by
``pmt_offset = sphere_radius - expose_height``. The propagator's sphere-of-radius-R
sensor, intersected with the detector volume, is then exactly that cap — see
``docs/reference/pmt-npz-schema.md``. So ``S_radius`` is the radius of curvature and
``aperture_radius`` is the 20-inch rim that falls out of it.
"""

import json
import math

import numpy as np
import plotly.graph_objects as go

from .cylinder import Cylinder
from .registry import register_detector


@register_detector('polygon_cylinder')
class PolygonalCylinder(Cylinder):
    """Regular N-gon prism with flat end caps."""

    def __init__(self, n_sides, apothem, height, n_sensors, sensor_radius,
                 expose_height=None):
        """
        Parameters
        ----------
        n_sides : int
            Number of barrel panels (SK: 38).
        apothem : float
            Perpendicular distance from the axis to a panel — the wall itself.
        height : float
            Full height.
        n_sensors : int
            Number of photosensors to place algorithmically.
        sensor_radius : float
            Photocathode *aperture* radius (SK: 0.254 m).
        expose_height : float, optional
            How far the photocathode dome protrudes into the medium (SK: 0.18 m). When
            given, the sensor becomes a spherical cap: ``S_radius`` is promoted to the
            radius of curvature and centres are set back by ``pmt_offset``. When
            omitted the sensor stays a bare sphere of ``sensor_radius`` centred on the
            wall (the cylinder's convention).
        """
        self.n_sides = int(n_sides)
        self._set_shape(apothem, height)
        self._set_pmt_model(sensor_radius, expose_height)
        super().__init__(apothem, height, n_sensors, sensor_radius)
        # Cylinder.__init__ overwrote these via Detector.__init__ / its own body.
        self._set_shape(apothem, height)
        self._set_pmt_model(sensor_radius, expose_height)

    # ── shape / sensor-model bookkeeping ──────────────────────────────

    def _set_shape(self, apothem, height):
        self.apothem = float(apothem)
        self.r = float(apothem)              # inscribed circle: safe for inherited uses
        self.H = float(height)
        self.dphi = 2.0 * math.pi / self.n_sides
        self.circumradius = self.apothem / math.cos(math.pi / self.n_sides)
        self.panel_width = 2.0 * self.apothem * math.tan(self.dphi / 2.0)
        ang = (np.arange(self.n_sides) + 0.5) * self.dphi
        self.panel_normals = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], 1)

    def _set_pmt_model(self, aperture_radius, expose_height):
        """Set ``S_radius`` / ``aperture_radius`` / ``pmt_offset`` from the PMT model.

        ``sphere_radius = (expose^2 + aperture^2) / (2 expose)`` is WCSim's own formula
        (``WCSimConstructPMT.cc:66``); ``pmt_offset = sphere_radius - expose`` is its
        ``PMTOffset``. The two invariants worth remembering:
        ``sphere_radius - pmt_offset == expose`` and
        ``sqrt(sphere_radius^2 - pmt_offset^2) == aperture``.
        """
        self.aperture_radius = float(aperture_radius)
        self.expose_height = None if expose_height is None else float(expose_height)
        if self.expose_height is None:
            self.sphere_radius = float(aperture_radius)
            self.pmt_offset = 0.0
        else:
            e, a = self.expose_height, self.aperture_radius
            self.sphere_radius = (e * e + a * a) / (2.0 * e)
            self.pmt_offset = self.sphere_radius - e
        self.S_radius = self.sphere_radius

    # ── construction from measured PMT positions ──────────────────────

    @classmethod
    def from_pmt_file(cls, npz_file_path, n_sides, apothem, height,
                      expose_height=None, verify_tol=1.0e-6):
        """Load PMT positions from an npz and impose a declared polygon on them.

        The polygon is *declared* (from WCSim's ``WCIDDiameter``/``WCIDHeight``) and then
        *verified* against the positions — never fitted. A fit could silently converge on
        the wrong N.

        ``snap_to_wall`` is not offered: on the correct polygon the PMT centres already
        sit exactly ``pmt_offset`` outside the wall, so there is nothing to snap. If they
        do not, that is a geometry error and :meth:`verify_pmt_surface` raises.
        """
        inst = super().from_pmt_file(npz_file_path, snap_to_wall=False)
        inst.n_sides = int(n_sides)
        inst._set_shape(apothem, height)
        # Cylinder.from_pmt_file set S_radius from the npz 'sensor_radius', which is the
        # aperture radius; promote it to the radius of curvature.
        inst._set_pmt_model(inst.S_radius, expose_height)
        inst._n_u = None
        if verify_tol is not None:
            inst.verify_pmt_surface(tol=verify_tol)
        return inst

    def surface_offsets(self):
        """Signed distance of every sensor centre beyond the wall, per surface.

        Positive means outside the detector, which is where a photocathode's centre of
        curvature belongs. Returns an ``(n_sensors,)`` array.
        """
        pts = np.asarray(self.all_points)
        case = np.array([self.ID_to_case[i] for i in range(len(pts))])
        out = np.empty(len(pts))
        barrel = case == 0
        if barrel.any():
            phi = np.arctan2(pts[barrel, 1], pts[barrel, 0]) % (2.0 * np.pi)
            ang = (np.floor(phi / self.dphi) + 0.5) * self.dphi
            perp = pts[barrel, 0] * np.cos(ang) + pts[barrel, 1] * np.sin(ang)
            out[barrel] = perp - self.apothem
        out[case == 1] = pts[case == 1, 2] - self.H / 2.0
        out[case == 2] = -self.H / 2.0 - pts[case == 2, 2]
        return out

    def verify_pmt_surface(self, tol=1.0e-6):
        """Assert every sensor centre sits ``pmt_offset`` outside the wall.

        This is the cross-check that the declared polygon is the real one: for SK the
        residual is float64 noise (~1e-11 m), while a wrong side count leaves centimetres.
        """
        resid = self.surface_offsets() - self.pmt_offset
        worst = float(np.abs(resid).max())
        if worst > tol:
            raise ValueError(
                f"PolygonalCylinder: sensor centres are not on the declared "
                f"{self.n_sides}-gon (apothem {self.apothem:.6f}, offset "
                f"{self.pmt_offset:.6f} m): worst residual {worst:.3e} m > {tol:.1e}. "
                f"Check n_sides / apothem / height / expose_height.")
        return worst

    # ── grid ──────────────────────────────────────────────────────────

    def configure_grid(self, n_cap=None, n_angular=None, n_height=None,
                       max_candidates_per_ray=4):
        """Size the grid, then round ``n_angular`` up to a multiple of ``n_sides``.

        The rounding is what keeps a cell from straddling a panel edge — a straddling
        cell would break the single-flat-face assumption its sensors are assigned under.
        Sizing otherwise follows the cylinder's nearest-neighbour rule.
        """
        scale = math.sqrt(max_candidates_per_ray / 4.0)

        if self.all_points is not None and len(self.all_points) >= 2:
            from scipy.spatial import cKDTree
            pts = np.asarray(self.all_points)
            dists, _ = cKDTree(pts).query(pts, k=2)
            target = max(0.9 * float(dists[:, 1].min()) * scale,
                         self.aperture_radius * scale)
        else:
            target = 2 * self.aperture_radius * scale

        perimeter = self.n_sides * self.panel_width
        want = n_angular if n_angular is not None else max(
            self.n_sides, int(math.ceil(perimeter / target)))
        self._n_u = max(1, int(math.ceil(want / self.n_sides)))
        self._n_angular = self.n_sides * self._n_u
        self._n_height = n_height if n_height is not None else max(
            10, int(math.ceil(self.H / target)))
        # Cap cells span the circumradius so the polygon corners are covered.
        self._n_cap = n_cap if n_cap is not None else max(
            10, int(math.ceil(2 * self.circumradius / target)))

    # ── propagation interface ─────────────────────────────────────────

    def bounds_check(self, positions):
        from lucid.propagation.polygon import polygon_bounds_check
        return polygon_bounds_check(positions, self.apothem, self.H, self.n_sides)

    def boundary_signed_distance(self, positions):
        """Distance to the nearest of panel / caps, positive inside."""
        import jax.numpy as jnp
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        phi = jnp.arctan2(y, x) % (2.0 * jnp.pi)
        ang = (jnp.floor(phi / self.dphi) + 0.5) * self.dphi
        perp = x * jnp.cos(ang) + y * jnp.sin(ang)
        return jnp.minimum(self.apothem - perp, self.H / 2 - jnp.abs(z))

    def intersect_ray(self, origins, directions):
        from lucid.propagation.polygon import batch_intersect_polygon_with_grid
        (intersects, t, is_wall, is_top_cap, wall_indices, cap_indices,
         point, panel) = batch_intersect_polygon_with_grid(
            origins, directions, self.apothem, self.H,
            self.n_sides, self._n_u, self._n_height, self._n_cap)
        grid_info = (is_wall, is_top_cap, wall_indices, cap_indices)
        surface_info = (is_wall, is_top_cap, panel)
        return point, t, grid_info, surface_info

    def compute_normal(self, intersection_point, surface_info):
        from lucid.propagation.polygon import calculate_polygon_normals
        is_wall, is_top_cap, panel = surface_info
        return calculate_polygon_normals(is_wall, is_top_cap, panel, self.n_sides)

    def assign_sensor_to_cells(self, sensors, sensor_radius):
        from lucid.propagation.polygon import assign_sensors_to_polygon_grid
        return assign_sensors_to_polygon_grid(
            sensors, sensor_radius, self.apothem, self.H,
            self.n_sides, self._n_u, self._n_height, self._n_cap)

    def grid_cell_centers(self):
        from lucid.propagation.polygon import calculate_polygon_grid_centers
        return calculate_polygon_grid_centers(
            self.apothem, self.H, self.n_sides, self._n_u,
            self._n_height, self._n_cap)

    # ── algorithmic placement ─────────────────────────────────────────

    def place_photosensors(self):
        """Tile the panels and the caps, splitting sensors by surface area."""
        perimeter = self.n_sides * self.panel_width
        barrel_area = perimeter * self.H
        cap_area = 0.5 * perimeter * self.apothem            # one cap
        total = barrel_area + 2 * cap_area

        n_barrel = int(self.n_sensors * barrel_area / total)
        n_per_cap = (self.n_sensors - n_barrel) // 2

        self.barr_points = self._place_panel_sensors(n_barrel)
        self.tcap_points = self._place_cap_sensors(n_per_cap, self.H / 2)
        self.bcap_points = self._place_cap_sensors(n_per_cap, -self.H / 2)
        self.all_points = np.concatenate(
            [self.barr_points, self.tcap_points, self.bcap_points], axis=0)

        n_barr, n_tcap = len(self.barr_points), len(self.tcap_points)
        self.ID_to_position = {i: self.all_points[i] for i in range(len(self.all_points))}
        self.ID_to_case = {
            i: (0 if i < n_barr else (1 if i < n_barr + n_tcap else 2))
            for i in range(len(self.all_points))
        }

    def _place_panel_sensors(self, n_sensors):
        """Uniform grid on each flat panel, offset inward by ``pmt_offset``."""
        if n_sensors == 0:
            return np.empty((0, 3))
        height_eff = self.H - 3 * self.aperture_radius
        perimeter = self.n_sides * self.panel_width
        n_rows = max(1, int(round(math.sqrt(n_sensors * height_eff / perimeter))))
        cols_per_panel = max(1, int(round(n_sensors / (n_rows * self.n_sides))))

        z = np.linspace(-height_eff / 2, height_eff / 2, n_rows)
        w = self.panel_width
        u = -w / 2 + (np.arange(cols_per_panel) + 0.5) * (w / cols_per_panel)
        # Centres sit pmt_offset OUTSIDE the wall (centre of curvature convention).
        perp = self.apothem + self.pmt_offset

        pts = []
        for k in range(self.n_sides):
            ang = (k + 0.5) * self.dphi
            nrm = np.array([math.cos(ang), math.sin(ang)])
            tan = np.array([-math.sin(ang), math.cos(ang)])
            for uu in u:
                xy = perp * nrm + uu * tan
                for zz in z:
                    pts.append([xy[0], xy[1], zz])
        return np.asarray(pts[:n_sensors])

    def _place_cap_sensors(self, n_sensors, z_position):
        """Square lattice clipped to the polygon, offset outward by ``pmt_offset``."""
        if n_sensors == 0:
            return np.empty((0, 3))
        limit = self.apothem - 1.5 * self.aperture_radius
        if limit <= 0:
            return np.empty((0, 3))
        z = z_position + math.copysign(self.pmt_offset, z_position)

        pts = []
        n_side = int(math.ceil(math.sqrt(n_sensors * 4.0 / math.pi)))
        for _ in range(60):                                  # widen until enough fit
            step = 2 * limit / n_side
            g = (np.arange(n_side) - n_side / 2 + 0.5) * step
            xx, yy = np.meshgrid(g, g, indexing='ij')
            xx, yy = xx.reshape(-1), yy.reshape(-1)
            phi = np.arctan2(yy, xx) % (2 * np.pi)
            ang = (np.floor(phi / self.dphi) + 0.5) * self.dphi
            inside = xx * np.cos(ang) + yy * np.sin(ang) <= limit
            pts = np.stack([xx[inside], yy[inside], np.full(int(inside.sum()), z)], 1)
            if len(pts) >= n_sensors:
                break
            n_side += 1
        order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]))    # keep the innermost
        return pts[order[:n_sensors]]

    # ── visualisation ─────────────────────────────────────────────────

    def _add_detector_surface(self, fig, surface_color='gray'):
        """Faceted barrel + polygonal caps (the base class would draw a round barrel)."""
        offset = 0.995
        ang = np.arange(self.n_sides + 1) * self.dphi
        vx = offset * self.circumradius * np.cos(ang)
        vy = offset * self.circumradius * np.sin(ang)
        z = np.array([-offset * self.H / 2, offset * self.H / 2])
        fig.add_trace(go.Surface(
            x=np.tile(vx, (2, 1)), y=np.tile(vy, (2, 1)),
            z=np.repeat(z[:, None], self.n_sides + 1, axis=1),
            opacity=1.0, showscale=False,
            colorscale=[[0, surface_color], [1, surface_color]],
            name='Barrel Surface', showlegend=False, hoverinfo='skip'))
        for zc, nm in ((offset * self.H / 2, 'Top Cap'), (-offset * self.H / 2, 'Bottom Cap')):
            fig.add_trace(go.Mesh3d(
                x=np.append(vx[:-1], 0.0), y=np.append(vy[:-1], 0.0),
                z=np.full(self.n_sides + 1, zc),
                i=np.full(self.n_sides, self.n_sides),
                j=np.arange(self.n_sides),
                k=(np.arange(self.n_sides) + 1) % self.n_sides,
                color=surface_color, opacity=1.0, name=nm,
                showlegend=False, hoverinfo='skip'))

    def visualize_geometry_wireframe(self, show_sensors=True):
        fig = go.Figure()
        self._add_detector_surface(fig, 'lightblue')
        if show_sensors:
            p = np.asarray(self.all_points)
            fig.add_trace(go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers',
                marker=dict(size=2, color='blue'), name=f'Sensors ({len(p)})'))
        fig.update_layout(
            scene=dict(aspectmode='data'),
            title=f'{self.n_sides}-gon prism (apothem={self.apothem:.3f}, H={self.H:.3f})',
            height=800)
        fig.show()

    # ── config helper ─────────────────────────────────────────────────

    @classmethod
    def from_config(cls, file_path):
        """Build from a geometry JSON (npz-backed or algorithmic)."""
        import os
        with open(file_path) as f:
            cfg = json.load(f)
        g = cfg['geometry_definitions']
        if 'npz_file_path' in g:
            npz = os.path.join(os.path.dirname(os.path.abspath(file_path)),
                               g['npz_file_path'])
            return cls.from_pmt_file(
                npz, n_sides=g['n_sides'], apothem=g['apothem'], height=g['height'],
                expose_height=g.get('expose_height'),
                verify_tol=g.get('surface_tolerance', 1.0e-6))
        return cls(g['n_sides'], g['apothem'], g['height'],
                   g['n_sensors'], g['sensor_radius'], g.get('expose_height'))
