"""Hard ray-intersection oracle for the SK 20-inch PMT surface.

This module is a NumPy reference implementation of the geometry encoded by
``SKDONUTS`` in SKDetSim.  It intentionally uses hard root selection and
Boolean surface cuts.  The production JAX propagator does not call it; its
purpose is to define and test the forward geometry before a differentiable
version is introduced.
"""

from typing import NamedTuple

import numpy as np


# SKDONUTS constants converted from centimetres to metres.
SPHERE_CENTER_Z = -0.127
SPHERE_RADIUS = 0.315
TORUS_MAJOR_RADIUS = 0.104
TORUS_MINOR_RADIUS = 0.150
GEANT_PMT_RADIUS = 0.254
SPHERE_TO_TORUS_Z = (
    TORUS_MINOR_RADIUS * -SPHERE_CENTER_Z
    / (SPHERE_RADIUS - TORUS_MINOR_RADIUS)
)
BARREL_INACTIVE_BAND_Z = 0.020

_ROOT_IMAG_TOL = 1e-7
_SURFACE_TOL = 1e-8
_CUT_TOL = 1e-10


class SK20InchPMTHit(NamedTuple):
    """Result of one hard ray/PMT query.

    ``intersects`` includes the inactive two-centimetre barrel band. ``active``
    is the photocathode decision after that band cut. Normals point from the
    PMT solid toward the surrounding water.
    """

    intersects: bool
    active: bool
    distance: float
    position: np.ndarray
    normal: np.ndarray
    local_position: np.ndarray
    local_normal: np.ndarray
    surface: str
    incidence_cosine: float


def _as_vector3(value, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _unit_vector(value, name):
    vector = _as_vector3(value, name)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{name} must have non-zero length")
    return vector / norm


def _pmt_frame(inward_axis):
    """Construct an orthonormal frame whose local +z is the PMT axis."""
    local_z = _unit_vector(inward_axis, "pmt_direction")
    # Pick the Cartesian direction least parallel to z, then project it into
    # the transverse plane. The SK surface is rotationally symmetric, so the
    # otherwise arbitrary transverse orientation cannot affect a hit.
    helper = np.eye(3)[np.argmin(np.abs(local_z))]
    local_x = helper - np.dot(helper, local_z) * local_z
    local_x /= np.linalg.norm(local_x)
    local_y = np.cross(local_z, local_x)
    return np.column_stack((local_x, local_y, local_z))


def _quadratic_roots(a, b, c):
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return []
    sqrt_discriminant = np.sqrt(max(discriminant, 0.0))
    return [
        (-b - sqrt_discriminant) / (2.0 * a),
        (-b + sqrt_discriminant) / (2.0 * a),
    ]


def _sphere_candidates(origin, direction, min_distance):
    center = np.array([0.0, 0.0, SPHERE_CENTER_Z])
    offset = origin - center
    roots = _quadratic_roots(
        float(np.dot(direction, direction)),
        2.0 * float(np.dot(offset, direction)),
        float(np.dot(offset, offset) - SPHERE_RADIUS**2),
    )
    candidates = []
    for distance in roots:
        point = origin + distance * direction
        if (distance >= min_distance
                and point[2] >= SPHERE_TO_TORUS_Z - _CUT_TOL):
            candidates.append((float(distance), "sphere", point))
    return candidates


def _torus_polynomial(origin, direction):
    """Return ascending coefficients of the ray/torus quartic."""
    # Polynomial represents x(t)^2 + y(t)^2.
    rho_squared = np.polynomial.polynomial.polyadd(
        np.polynomial.polynomial.polymul(
            [origin[0], direction[0]], [origin[0], direction[0]]),
        np.polynomial.polynomial.polymul(
            [origin[1], direction[1]], [origin[1], direction[1]]),
    )
    z_squared = np.polynomial.polynomial.polymul(
        [origin[2], direction[2]], [origin[2], direction[2]])
    radial_sum = np.polynomial.polynomial.polyadd(rho_squared, z_squared)
    radial_sum[0] += TORUS_MAJOR_RADIUS**2 - TORUS_MINOR_RADIUS**2
    quartic = np.polynomial.polynomial.polymul(radial_sum, radial_sum)
    projected = np.pad(
        4.0 * TORUS_MAJOR_RADIUS**2 * rho_squared,
        (0, len(quartic) - len(rho_squared)),
    )
    return quartic - projected


def _torus_candidates(origin, direction, min_distance):
    # The standard quartic is a squared form of the radial equation. For the
    # SK spindle torus it can contain roots from the unwanted squared branch,
    # so every numerical root is checked against SKDDONUTS itself below.
    coefficients = _torus_polynomial(origin, direction)
    roots = np.roots(coefficients[::-1])
    candidates = []
    for root in roots:
        if abs(root.imag) > _ROOT_IMAG_TOL:
            continue
        distance = float(root.real)
        if distance < min_distance:
            continue
        point = origin + distance * direction
        rho = float(np.hypot(point[0], point[1]))
        residual = ((rho - TORUS_MAJOR_RADIUS)**2 + point[2]**2
                    - TORUS_MINOR_RADIUS**2)
        if abs(residual) > _SURFACE_TOL:
            continue
        if (-_CUT_TOL <= point[2]
                and point[2] < SPHERE_TO_TORUS_Z + _CUT_TOL):
            candidates.append((distance, "torus", point))
    return candidates


def _bounded_torus_candidates(origin, direction, min_distance):
    """Solve the torus after shifting to the nearby GEANT PMT boundary.

    The detector-scale ray origin can be tens of metres from a 20-inch PMT.
    Building the torus quartic in that coordinate system is ill-conditioned.
    SKDONUTS begins its millimetre march at the PMT boundary; shifting the
    polynomial origin there leaves the physical ray unchanged and gives the
    same numerical scale.
    """
    roots = _quadratic_roots(
        float(np.dot(direction, direction)),
        2.0 * float(np.dot(origin, direction)),
        float(np.dot(origin, origin) - GEANT_PMT_RADIUS**2),
    )
    if not roots:
        return []
    lower = max(min(roots), min_distance)
    upper = max(roots)
    if upper < lower:
        return []
    shifted_origin = origin + lower * direction
    local_candidates = _torus_candidates(
        shifted_origin, direction, 0.0)
    interval = upper - lower
    return [
        (lower + distance, surface, point)
        for distance, surface, point in local_candidates
        if distance <= interval + _CUT_TOL
    ]


def _local_surface_normal(point, surface):
    if surface == "sphere":
        normal = point - np.array([0.0, 0.0, SPHERE_CENTER_Z])
    else:
        rho = float(np.hypot(point[0], point[1]))
        if rho == 0.0:
            raise RuntimeError("torus surface normal is undefined on its axis")
        radial_scale = 1.0 - TORUS_MAJOR_RADIUS / rho
        normal = np.array([
            point[0] * radial_scale,
            point[1] * radial_scale,
            point[2],
        ])
    return normal / np.linalg.norm(normal)


def _miss():
    missing = np.full(3, np.nan)
    return SK20InchPMTHit(
        intersects=False,
        active=False,
        distance=np.inf,
        position=missing.copy(),
        normal=missing.copy(),
        local_position=missing.copy(),
        local_normal=missing.copy(),
        surface="miss",
        incidence_cosine=0.0,
    )


def intersect_sk20inch_pmt_hard(
    ray_origin,
    ray_direction,
    pmt_position,
    pmt_direction,
    *,
    barrel=False,
    min_distance=1e-9,
):
    """Intersect one ray with the SKDetSim 20-inch PMT surface.

    Parameters
    ----------
    ray_origin, ray_direction : array-like, shape (3,)
        World-space ray. The direction is normalized internally, so the
        returned distance is in metres.
    pmt_position : array-like, shape (3,)
        World-space origin of the SK local PMT coordinates.
    pmt_direction : array-like, shape (3,)
        Inward-looking PMT axis; this becomes local ``+z`` as in SKDONUTS.
    barrel : bool
        Apply SKDONUTS' inactive base-band rule for a barrel PMT.
    min_distance : float
        Reject roots closer than this forward distance.

    Notes
    -----
    SKDONUTS approaches the toroidal transition in 1 mm steps. This oracle
    instead solves the same implicit surface as a quartic and therefore
    removes that stepping error while retaining the hard SK region cuts.
    It is deliberately NumPy-only and is not differentiable.
    """
    world_origin = _as_vector3(ray_origin, "ray_origin")
    world_direction = _unit_vector(ray_direction, "ray_direction")
    world_pmt_position = _as_vector3(pmt_position, "pmt_position")
    if not np.isfinite(min_distance) or min_distance < 0.0:
        raise ValueError("min_distance must be finite and non-negative")

    frame = _pmt_frame(pmt_direction)
    local_origin = frame.T @ (world_origin - world_pmt_position)
    local_direction = frame.T @ world_direction

    candidates = _sphere_candidates(
        local_origin, local_direction, min_distance)
    candidates.extend(_bounded_torus_candidates(
        local_origin, local_direction, min_distance))
    if not candidates:
        return _miss()

    distance, surface, local_position = min(candidates, key=lambda item: item[0])
    local_normal = _local_surface_normal(local_position, surface)
    position = world_pmt_position + frame @ local_position
    normal = frame @ local_normal
    incidence_cosine = float(np.clip(
        -np.dot(world_direction, normal), 0.0, 1.0))
    active = not (bool(barrel)
                  and local_position[2] <= BARREL_INACTIVE_BAND_Z + _CUT_TOL)

    return SK20InchPMTHit(
        intersects=True,
        active=active,
        distance=distance,
        position=position,
        normal=normal,
        local_position=local_position,
        local_normal=local_normal,
        surface=surface,
        incidence_cosine=incidence_cosine,
    )


__all__ = [
    "BARREL_INACTIVE_BAND_Z",
    "GEANT_PMT_RADIUS",
    "SK20InchPMTHit",
    "SPHERE_CENTER_Z",
    "SPHERE_RADIUS",
    "SPHERE_TO_TORUS_Z",
    "TORUS_MAJOR_RADIUS",
    "TORUS_MINOR_RADIUS",
    "intersect_sk20inch_pmt_hard",
]
