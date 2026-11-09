"""The instrumented volume of a detector, as a value: membership, a scaled region, its extent.

A detector answers "is this point inside me?" through ``detector.volume.contains``. Any other
region -- a fiducial volume, a search box -- belongs to the caller, who builds it from the volume
(``detector.volume.scaled(0.95)``) and passes it on. Nothing here takes a fraction of the detector
as a parameter: a fraction is the caller's definition of a different region, not a property of
the detector.

``contains`` works on numpy and JAX arrays alike, ``(..., 3) -> (...)`` booleans, true on the
surface. Inside a trace it is the forward kernel's own test.

``from_unit(u)`` maps ``u`` in ``[0, 1)^3`` to a point uniform in the volume; the caller owns the
random stream that draws ``u``.

``clip(points)`` is the nearest point of the volume to each point, numpy ``(..., 3)``: a point
inside is returned unchanged, one outside lands on the surface (to rounding).
"""
from dataclasses import dataclass

import numpy as np


def _is_jax(points):
    import jax
    return isinstance(points, jax.Array)


@dataclass(frozen=True)
class CylinderVolume:
    """Upright cylinder ``x² + y² <= radius²``, ``z_min <= z <= z_max`` (metres).

    Detector cylinders are centred on the origin; a string array's envelope is not.
    """
    radius: float
    z_min: float
    z_max: float

    def contains(self, points):
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        return ((x ** 2 + y ** 2) <= self.radius ** 2) & ((z >= self.z_min) & (z <= self.z_max))

    def scaled(self, factor):
        """Radius and half-height times ``factor``, about the volume's own centre."""
        zc = 0.5 * (self.z_min + self.z_max)
        if zc == 0.0:
            return CylinderVolume(self.radius * factor, self.z_min * factor, self.z_max * factor)
        h = 0.5 * (self.z_max - self.z_min) * factor
        return CylinderVolume(self.radius * factor, zc - h, zc + h)

    def from_unit(self, u):
        r = self.radius * np.sqrt(u[0])
        phi = 2.0 * np.pi * u[1]
        zc, h = 0.5 * (self.z_min + self.z_max), 0.5 * (self.z_max - self.z_min)
        return np.array([r * np.cos(phi), r * np.sin(phi), zc + (2.0 * u[2] - 1.0) * h])

    def bounding_box(self):
        """``(lo, hi)`` corners of the axis-aligned box around the volume."""
        return (np.array([-self.radius, -self.radius, self.z_min]),
                np.array([self.radius, self.radius, self.z_max]))

    def clip(self, points):
        """Radius and z clamped separately: the two constraints are independent."""
        p = np.array(points, float)
        rxy = np.hypot(p[..., 0], p[..., 1])
        s = np.where(rxy > self.radius, self.radius / np.where(rxy > self.radius, rxy, 1.0), 1.0)
        p[..., 0] *= s
        p[..., 1] *= s
        p[..., 2] = np.clip(p[..., 2], self.z_min, self.z_max)
        return p


@dataclass(frozen=True)
class SphereVolume:
    """Sphere of ``radius`` about the origin (metres)."""
    radius: float

    def contains(self, points):
        if _is_jax(points):
            import jax.numpy as jnp
            return jnp.linalg.norm(points, axis=-1) <= self.radius
        return np.sqrt(np.sum(np.asarray(points) ** 2, axis=-1)) <= self.radius

    def scaled(self, factor):
        return SphereVolume(self.radius * factor)

    def from_unit(self, u):
        r = self.radius * u[0] ** (1.0 / 3.0)
        cos_t = 2.0 * u[1] - 1.0
        sin_t = np.sqrt(1.0 - cos_t * cos_t)
        phi = 2.0 * np.pi * u[2]
        return r * np.array([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t])

    def bounding_box(self):
        r = self.radius
        return np.array([-r, -r, -r]), np.array([r, r, r])

    def clip(self, points):
        p = np.asarray(points, float)
        r = np.linalg.norm(p, axis=-1, keepdims=True)
        return p * np.where(r > self.radius, self.radius / np.where(r > self.radius, r, 1.0), 1.0)


@dataclass(frozen=True)
class BoxVolume:
    """Axis-aligned box about the origin, given by its half-extents (metres)."""
    half_x: float
    half_y: float
    half_z: float

    def contains(self, points):
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        return ((x >= -self.half_x) & (x <= self.half_x) &
                (y >= -self.half_y) & (y <= self.half_y) &
                (z >= -self.half_z) & (z <= self.half_z))

    def scaled(self, factor):
        return BoxVolume(self.half_x * factor, self.half_y * factor, self.half_z * factor)

    def from_unit(self, u):
        return (2.0 * np.asarray(u, float) - 1.0) * np.array([self.half_x, self.half_y, self.half_z])

    def bounding_box(self):
        h = np.array([self.half_x, self.half_y, self.half_z])
        return -h, h

    def clip(self, points):
        h = np.array([self.half_x, self.half_y, self.half_z])
        return np.clip(np.asarray(points, float), -h, h)
