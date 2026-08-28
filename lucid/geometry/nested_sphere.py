"""Nested two-sphere detector geometry (JUNO-like).

An inner sphere (liquid-scintillator ↔ water optical interface) concentric with an
outer sphere (the PMT surface + reflecting wall). Sensors live on the **outer** sphere,
so all PMT placement / grid / sensor-map machinery is inherited unchanged from
:class:`~lucid.geometry.sphere.Sphere` built at the outer radius. The only extra state
is the inner radius ``r_inner`` (the interface); the photon transport reads it to know
where the two media meet.

The matching propagator lives in :mod:`lucid.propagation.nested_sphere`.
"""

import jax.numpy as jnp
import numpy as np

from .sphere import Sphere
from .registry import register_detector


@register_detector('nested_sphere')
class NestedSphere(Sphere):
    """Two concentric spheres: inner optical interface + outer PMT/wall surface."""

    def __init__(self, inner_radius, outer_radius, n_sensors, sensor_radius):
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
        """
        if not (inner_radius < outer_radius):
            raise ValueError(
                f"inner_radius ({inner_radius}) must be < outer_radius ({outer_radius})")
        # Sphere.__init__ places sensors on a sphere of radius=outer_radius (self.r).
        super().__init__(outer_radius, n_sensors, sensor_radius)
        self.r_inner = inner_radius
        self.r_outer = outer_radius

    @classmethod
    def from_pmt_file(cls, npz_file_path, inner_radius, outer_radius,
                      sensor_radius):
        """Build a nested sphere using externally supplied PMT positions.

        The file must contain a ``pmt_positions`` array in metres with shape
        ``(n_sensors, 3)``.  This constructor preserves the regular
        :class:`Sphere` sensor maps while replacing only the placement, so the
        nested propagator can use measured JUNO coordinates without a runtime
        monkey patch.
        """
        with np.load(npz_file_path, allow_pickle=False) as data:
            if "pmt_positions" not in data.files:
                raise KeyError(
                    f"PMT file {npz_file_path!r} must contain 'pmt_positions'")
            positions = np.asarray(data["pmt_positions"], dtype=np.float64)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"PMT positions must have shape (n, 3), got {positions.shape}")
        if not np.all(np.isfinite(positions)):
            raise ValueError(f"PMT positions in {npz_file_path!r} contain non-finite values")

        instance = cls(inner_radius, outer_radius, len(positions), sensor_radius)
        instance.all_points = positions
        instance.ID_to_position = {
            i: positions[i] for i in range(len(positions))
        }
        instance.ID_to_case = {i: 0 for i in range(len(positions))}
        instance.npz_file_path = npz_file_path
        return instance

    def bounds_check(self, positions):
        """Inside the detector ⇔ inside the outer sphere (the whole instrumented volume)."""
        return jnp.linalg.norm(positions, axis=1) <= self.r_outer

    def region_of(self, positions):
        """Per-position medium id: 0 = inner medium (r < r_inner), 1 = outer medium.

        Used to initialise each photon's ``medium_id`` from its emission point.
        """
        r = jnp.linalg.norm(positions, axis=1)
        return (r >= self.r_inner).astype(jnp.int32)
