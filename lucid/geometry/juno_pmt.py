"""JUNO 20-inch PMT layout and detector-facing bulb geometry.

P1a models the outer and inner surfaces of the front glass head. Optical
layers and photoelectron production are deliberately outside this module.
"""

import io
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


HAMAMATSU = 0
NNVT = 1
HIGH_QE_NNVT = 2

_TYPE_CODES = {
    "Hamamatsu": HAMAMATSU,
    "NNVT": NNVT,
    "HighQENNVT": HIGH_QE_NNVT,
}

_OUTER_RADIAL_M = 0.254
_HAMAMATSU_OUTER_AXIAL_M = 0.190
_NNVT_OUTER_AXIAL_M = 0.184
_GLASS_THICKNESS_M = 0.005


class JunoPmtLayout(NamedTuple):
    """PMT placement joined to technology labels by CopyNo."""

    copy_numbers: np.ndarray
    positions_m: np.ndarray
    inward_axes: np.ndarray
    type_codes: np.ndarray


class JunoPmtDimensions(NamedTuple):
    """Per-PMT front-head dimensions in metres."""

    outer_radial_m: np.ndarray
    outer_axial_m: np.ndarray
    inner_radial_m: np.ndarray
    inner_axial_m: np.ndarray


class EllipsoidIntersection(NamedTuple):
    """First forward intersection with a detector-facing ellipsoid half."""

    distance_m: jnp.ndarray
    valid: jnp.ndarray
    points_m: jnp.ndarray
    water_normals: jnp.ndarray
    detector_outward_normals: jnp.ndarray


def _data_lines(path):
    """Return non-comment, non-empty lines from a JUNOSW table."""
    lines = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped and not stripped.startswith(("#", '"')):
                lines.append(stripped)
    if not lines:
        raise ValueError(f"JUNO PMT table {path!r} contains no data rows")
    return lines


def _orientation_vectors(theta_deg, phi_deg):
    """Convert JUNOSW spherical angles into outward unit vectors."""
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    return np.column_stack(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        )
    )


def load_juno_pmt_layout(position_path, type_path):
    """Load positions, orientations, and types using CopyNo as the key.

    Parameters
    ----------
    position_path, type_path : path-like
        JUNOSW PMT position and type tables.

    Returns
    -------
    JunoPmtLayout
        Positions are converted from mm to m. Axes point from the PMT
        equator toward the detector centre.
    """
    position_lines = _data_lines(position_path)
    try:
        position_data = np.loadtxt(
            io.StringIO("\n".join(position_lines)),
            dtype=np.float64,
            ndmin=2,
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Could not parse JUNO PMT positions from {position_path!r}"
        ) from error
    if position_data.shape[1] < 6:
        raise ValueError("JUNO PMT positions require six columns")

    copy_numbers = position_data[:, 0].astype(np.int32)
    if not np.array_equal(copy_numbers, position_data[:, 0]):
        raise ValueError("PMT CopyNo values must be integers")
    if len(np.unique(copy_numbers)) != len(copy_numbers):
        raise ValueError("PMT position table contains duplicate CopyNo values")

    type_by_copy = {}
    for line in _data_lines(type_path):
        fields = line.split()
        if len(fields) != 2 or fields[1] not in _TYPE_CODES:
            raise ValueError(f"Invalid JUNO PMT type row: {line!r}")
        copy_number = int(fields[0])
        if copy_number in type_by_copy:
            raise ValueError(f"Duplicate PMT type CopyNo {copy_number}")
        type_by_copy[copy_number] = _TYPE_CODES[fields[1]]
    missing = [
        int(value)
        for value in copy_numbers
        if int(value) not in type_by_copy
    ]
    extra = sorted(set(type_by_copy) - set(copy_numbers.tolist()))
    if missing or extra:
        raise ValueError(
            "Position/type CopyNo mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    outward_axes = _orientation_vectors(
        position_data[:, 4], position_data[:, 5]
    )
    return JunoPmtLayout(
        copy_numbers=copy_numbers,
        positions_m=position_data[:, 1:4] * 0.001,
        inward_axes=-outward_axes,
        type_codes=np.asarray(
            [type_by_copy[int(value)] for value in copy_numbers],
            dtype=np.int8,
        ),
    )


def build_juno_pmt_dimensions(type_codes):
    """Return front-head dimensions matching JUNOSW J26.3.2."""
    type_codes = np.asarray(type_codes, dtype=np.int8)
    known = np.isin(type_codes, list(_TYPE_CODES.values()))
    if not np.all(known):
        raise ValueError(f"Unknown JUNO PMT type codes: {type_codes[~known]}")
    outer_radial = np.full(type_codes.shape, _OUTER_RADIAL_M)
    outer_axial = np.where(
        type_codes == HAMAMATSU,
        _HAMAMATSU_OUTER_AXIAL_M,
        _NNVT_OUTER_AXIAL_M,
    )
    return JunoPmtDimensions(
        outer_radial_m=outer_radial,
        outer_axial_m=outer_axial,
        inner_radial_m=outer_radial - _GLASS_THICKNESS_M,
        inner_axial_m=outer_axial - _GLASS_THICKNESS_M,
    )


def intersect_ellipsoid_heads(
    ray_origins,
    ray_directions,
    centers,
    inward_axes,
    radial_radii,
    axial_radii,
):
    """Intersect rays with the detector-facing halves of PMT ellipsoids.

    The local axial coordinate is positive toward the detector centre. The
    returned normal points from glass into detector water.
    """
    origins = jnp.asarray(ray_origins)
    directions = jnp.asarray(ray_directions)
    centers = jnp.asarray(centers)
    axes = jnp.asarray(inward_axes)
    radial = jnp.broadcast_to(
        jnp.asarray(radial_radii), origins.shape[:1]
    )
    axial = jnp.broadcast_to(
        jnp.asarray(axial_radii), origins.shape[:1]
    )

    directions = directions / (
        jnp.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    )
    axes = axes / (jnp.linalg.norm(axes, axis=1, keepdims=True) + 1e-12)
    relative = origins - centers
    origin_axial = jnp.sum(relative * axes, axis=1)
    direction_axial = jnp.sum(directions * axes, axis=1)
    origin_perp = relative - origin_axial[:, None] * axes
    direction_perp = directions - direction_axial[:, None] * axes

    a = (
        jnp.sum(direction_perp**2, axis=1) / radial**2
        + direction_axial**2 / axial**2
    )
    b = 2.0 * (
        jnp.sum(origin_perp * direction_perp, axis=1) / radial**2
        + origin_axial * direction_axial / axial**2
    )
    c = (
        jnp.sum(origin_perp**2, axis=1) / radial**2
        + origin_axial**2 / axial**2
        - 1.0
    )
    discriminant = b**2 - 4.0 * a * c
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    near = (-b - root) / (2.0 * a + 1e-12)
    far = (-b + root) / (2.0 * a + 1e-12)
    near_axial = origin_axial + near * direction_axial
    far_axial = origin_axial + far * direction_axial
    epsilon = 1e-7
    near_valid = (
        (discriminant >= 0.0)
        & (near > epsilon)
        & (near_axial >= -epsilon)
    )
    far_valid = (
        (discriminant >= 0.0)
        & (far > epsilon)
        & (far_axial >= -epsilon)
    )
    valid = near_valid | far_valid
    distance = jnp.where(near_valid, near, jnp.where(far_valid, far, -1.0))
    points = origins + distance[:, None] * directions
    hit_relative = points - centers
    hit_axial = jnp.sum(hit_relative * axes, axis=1)
    hit_perp = hit_relative - hit_axial[:, None] * axes
    gradient = (
        hit_perp / radial[:, None] ** 2
        + hit_axial[:, None] * axes / axial[:, None] ** 2
    )
    water_normals = gradient / (
        jnp.linalg.norm(gradient, axis=1, keepdims=True) + 1e-12
    )
    points = jnp.where(valid[:, None], points, origins)
    water_normals = jnp.where(
        valid[:, None], water_normals, jnp.zeros_like(water_normals)
    )
    detector_outward_normals = -water_normals
    return EllipsoidIntersection(
        distance,
        valid,
        points,
        water_normals,
        detector_outward_normals,
    )
