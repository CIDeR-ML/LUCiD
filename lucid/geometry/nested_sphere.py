"""Nested concentric detector geometry (JUNO-like).

An inner sphere (liquid-scintillator ↔ water optical interface) concentric with an
outer sphere (the PMT surface + reflecting wall). Sensors live on the **outer** sphere,
so all PMT placement / grid / sensor-map machinery is inherited unchanged from
:class:`~lucid.geometry.sphere.Sphere` built at the outer radius. The only extra state
is the inner radius ``r_inner`` (the interface); the photon transport reads it to know
where the two media meet.

The matching propagator lives in :mod:`lucid.propagation.nested_sphere`.
"""

import io
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .juno_pmt import (
    build_juno_pmt_dimensions,
    load_juno_pmt_layout,
)
from .juno_support import load_juno_support_layout
from .sphere import Sphere
from .registry import register_detector


@register_detector('nested_sphere')
class NestedSphere(Sphere):
    """Two-layer detector, optionally extended with an Acrylic shell.

    With ``acrylic_radius=None`` this retains the historical two-medium
    LS--Water geometry.  When supplied, the radii define three regions:
    LS, Acrylic, and Water.
    """

    def __init__(self, inner_radius, outer_radius, n_sensors, sensor_radius,
                 acrylic_radius=None):
        """
        Parameters
        ----------
        inner_radius : float
            Radius of the inner medium interface (LS ↔ water), in metres.
        outer_radius : float
            Radius of the outer PMT/wall surface, in metres. Sensors are placed here.
        n_sensors : int
            Number of photosensors on the outer surface.
        sensor_radius : float
            Radius of each sensor.
        acrylic_radius : float, optional
            Outer radius of the Acrylic shell (metres).  It must lie strictly
            between ``inner_radius`` and ``outer_radius``.
        """
        if not (inner_radius < outer_radius):
            raise ValueError(
                f"inner_radius ({inner_radius}) must be < outer_radius ({outer_radius})")
        # Sphere.__init__ places sensors on a sphere of radius=outer_radius (self.r).
        super().__init__(outer_radius, n_sensors, sensor_radius)
        self.r_inner = inner_radius
        self.r_acrylic = acrylic_radius
        self.r_outer = outer_radius
        self.pmt_geometry = None
        self.pmt_copy_numbers = None
        self.pmt_inward_axes = None
        self.pmt_type_codes = None
        self.pmt_dimensions = None
        self.support_geometry = None
        self.support_rods = None
        if acrylic_radius is not None and not (
                inner_radius < acrylic_radius < outer_radius):
            raise ValueError(
                "acrylic_radius must lie strictly between inner_radius and "
                "outer_radius")

    @classmethod
    def from_pmt_file(cls, csv_file_path, inner_radius, outer_radius,
                      sensor_radius, acrylic_radius=None,
                      pmt_type_file_path=None, pmt_geometry=None,
                      support_rod_file_path=None,
                      support_geometry=None):
        """Build a nested sphere using externally supplied PMT positions.

        The position CSV contains CopyNo, coordinates in mm, and orientation
        angles. With ``pmt_geometry="juno_20inch_front"``, the type table is
        joined by CopyNo and supplies the Hamamatsu/NNVT front-head shape.
        The returned detector stores coordinates in metres. This constructor
        preserves the regular :class:`Sphere` sensor maps while replacing
        the placement and, when requested, the PMT intersection surface.
        """
        csv_path = Path(csv_file_path)
        if csv_path.suffix.lower() != ".csv":
            raise ValueError(
                f"PMT positions must be provided as CSV, got {csv_path!r}"
            )

        if pmt_geometry is None:
            data_lines = []
            with csv_path.open(encoding="utf-8") as stream:
                for line in stream:
                    stripped = line.strip()
                    if stripped and not stripped.startswith(("#", '"')):
                        data_lines.append(stripped)
            if not data_lines:
                raise ValueError(
                    f"PMT CSV {csv_path!r} contains no data rows"
                )
            try:
                positions_mm = np.loadtxt(
                    io.StringIO("\n".join(data_lines)),
                    dtype=np.float64,
                    usecols=(1, 2, 3),
                    ndmin=2,
                )
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"Could not parse PMT coordinates from {csv_path!r}"
                ) from error
            positions = positions_mm * 0.001
            layout = None
            dimensions = None
        elif pmt_geometry == "juno_20inch_front":
            if pmt_type_file_path is None:
                raise ValueError(
                    "pmt_type_file_path is required for JUNO PMT geometry"
                )
            layout = load_juno_pmt_layout(
                csv_path, Path(pmt_type_file_path)
            )
            dimensions = build_juno_pmt_dimensions(layout.type_codes)
            positions = layout.positions_m
        else:
            raise ValueError(f"Unknown PMT geometry model {pmt_geometry!r}")

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"PMT positions must have shape (n, 3), got {positions.shape}")
        if not np.all(np.isfinite(positions)):
            raise ValueError(
                f"PMT positions in {csv_path!r} contain non-finite values"
            )

        instance = cls(
            inner_radius, outer_radius, len(positions), sensor_radius,
            acrylic_radius=acrylic_radius)
        instance.all_points = positions
        instance.ID_to_position = {
            i: positions[i] for i in range(len(positions))
        }
        instance.ID_to_case = {i: 0 for i in range(len(positions))}
        instance.csv_file_path = csv_path
        instance.pmt_geometry = pmt_geometry
        instance.pmt_copy_numbers = (
            None if layout is None else layout.copy_numbers
        )
        instance.pmt_inward_axes = (
            None if layout is None else layout.inward_axes
        )
        instance.pmt_type_codes = (
            None if layout is None else layout.type_codes
        )
        instance.pmt_dimensions = dimensions
        if support_geometry is not None:
            if support_geometry != "juno_radial_cylinders":
                raise ValueError(
                    f"Unknown support geometry {support_geometry!r}"
                )
            if support_rod_file_path is None:
                raise ValueError(
                    "support_rod_file_path is required for support geometry"
                )
            instance.support_rods = load_juno_support_layout(
                support_rod_file_path
            )
        instance.support_geometry = support_geometry
        return instance

    def bounds_check(self, positions):
        """Inside the detector ⇔ inside the outer sphere (the whole instrumented volume)."""
        return jnp.linalg.norm(positions, axis=1) <= self.r_outer

    def region_of(self, positions):
        """Return the optical medium id at each position.

        For the historical two-layer geometry, ids are ``0`` (inner) and
        ``1`` (outer).  With an Acrylic shell, ids are ``0`` (LS), ``1``
        (Acrylic), and ``2`` (Water).  Boundary points belong to the region
        immediately outside the boundary.
        """
        r = jnp.linalg.norm(positions, axis=1)
        if self.r_acrylic is None:
            return (r >= self.r_inner).astype(jnp.int32)
        return jnp.where(
            r >= self.r_acrylic,
            2,
            jnp.where(r >= self.r_inner, 1, 0),
        ).astype(jnp.int32)
