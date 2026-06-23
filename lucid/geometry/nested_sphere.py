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

    def bounds_check(self, positions):
        """Inside the detector ⇔ inside the outer sphere (the whole instrumented volume)."""
        return jnp.linalg.norm(positions, axis=1) <= self.r_outer

    def region_of(self, positions):
        """Per-position medium id: 0 = inner medium (r < r_inner), 1 = outer medium.

        Used to initialise each photon's ``medium_id`` from its emission point.
        """
        r = jnp.linalg.norm(positions, axis=1)
        return (r >= self.r_inner).astype(jnp.int32)
