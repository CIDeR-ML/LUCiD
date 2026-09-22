"""Effective JUNO chimney and Acrylic-bowl geometry.

The implementation keeps the main J26.3.2 positions and engineering scales
while avoiding CAD-only bolt holes and coincident Boolean surfaces.  Lengths
are expressed in metres.  Surface codes are stable simulation metadata:

``1`` lower-chimney Acrylic, ``2`` lower-chimney steel, and ``3`` Acrylic
addition bowl.  Code ``0`` remains reserved for support rods.
"""

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from lucid.propagation.base import find_closest_sensors
from lucid.propagation.sphere import calculate_sphere_grid_centers


LARGE_DISTANCE = 1e10
CHIMNEY_ACRYLIC = 1
CHIMNEY_STEEL = 2
ACRYLIC_BOWL = 3
BOWL_INTERFACE_OFFSET_M = 5e-4


class SurfaceHit(NamedTuple):
    """Nearest effective engineering-surface intersection per ray."""

    distance_m: jnp.ndarray
    valid: jnp.ndarray
    object_indices: jnp.ndarray
    surface_codes: jnp.ndarray
    positions: jnp.ndarray
    transport_normals: jnp.ndarray


class ChimneyGeometry(NamedTuple):
    """Concentric annular segments representing the lower chimney."""

    inner_radii_m: np.ndarray
    outer_radii_m: np.ndarray
    z_min_m: np.ndarray
    z_max_m: np.ndarray
    surface_codes: np.ndarray
    aperture_radius_m: float
    aperture_z_min_m: float

    @classmethod
    def junosw_j26(cls, outer_limit_m):
        """Build the J26.3.2 lower chimney clipped to LUCiD's boundary.

        JUNOSW places a 3.926 m lower-chimney volume at ``z=19.787 m``.
        Its Acrylic segment spans 17.824--18.424 m; the 5 mm steel segment
        continues upward.  LUCiD stops it at the PMT radius.
        """
        z_outer = float(outer_limit_m)
        if z_outer <= 18.424:
            raise ValueError("outer_limit_m must exceed 18.424 m")
        return cls(
            inner_radii_m=np.array([0.400, 0.400]),
            outer_radii_m=np.array([0.524, 0.405]),
            z_min_m=np.array([17.824, 18.424]),
            z_max_m=np.array([18.424, z_outer]),
            surface_codes=np.array(
                [CHIMNEY_ACRYLIC, CHIMNEY_STEEL], dtype=np.int8
            ),
            aperture_radius_m=0.400,
            aperture_z_min_m=17.7,
        )


class AcrylicBowlLayout(NamedTuple):
    """Radial truncated-cone approximations to Acrylic additions."""

    copy_numbers: np.ndarray
    attachment_points_m: np.ndarray
    outward_axes: np.ndarray
    base_radii_m: np.ndarray
    tip_radii_m: np.ndarray
    lengths_m: np.ndarray


def load_juno_acrylic_bowls(csv_file_path):
    """Load the portable J26.3.2 Acrylic-addition placement table."""
    path = Path(csv_file_path)
    data = np.genfromtxt(
        path, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    data = np.atleast_1d(data)
    required = {
        "copy_no", "x_m", "y_m", "z_m", "base_radius_m",
        "tip_radius_m", "length_m",
    }
    if data.dtype.names is None or not required.issubset(data.dtype.names):
        raise ValueError(f"Acrylic-bowl CSV {path!r} has an invalid schema")
    points = np.column_stack([
        data["x_m"], data["y_m"], data["z_m"]
    ]).astype(np.float64)
    norms = np.linalg.norm(points, axis=1)
    dimensions = np.column_stack([
        data["base_radius_m"],
        data["tip_radius_m"],
        data["length_m"],
    ]).astype(np.float64)
    if np.any(~np.isfinite(points)) or np.any(norms <= 0.0):
        raise ValueError(f"Acrylic-bowl CSV {path!r} has invalid positions")
    if np.any(~np.isfinite(dimensions)) or np.any(dimensions <= 0.0):
        raise ValueError("Acrylic-bowl dimensions must be positive")
    return AcrylicBowlLayout(
        copy_numbers=np.asarray(data["copy_no"], dtype=np.int32),
        attachment_points_m=points,
        outward_axes=points / norms[:, None],
        base_radii_m=dimensions[:, 0],
        tip_radii_m=dimensions[:, 1],
        lengths_m=dimensions[:, 2],
    )


def build_bowl_candidate_map(layout, n_divisions, max_candidates):
    """Map spherical angular cells to nearby Acrylic additions."""
    radius = float(np.mean(np.linalg.norm(
        layout.attachment_points_m, axis=1
    )))
    cell_centres = calculate_sphere_grid_centers(radius, n_divisions)
    count = len(layout.attachment_points_m)
    selected = find_closest_sensors(
        cell_centres,
        jnp.asarray(layout.attachment_points_m),
        min(max_candidates, count),
    )
    if count >= max_candidates:
        return selected
    return jnp.pad(
        selected,
        ((0, 0), (0, max_candidates - count)),
        mode="edge",
    )


def _normalise(directions):
    """Return row-wise unit directions."""
    return directions / (
        jnp.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    )


def intersect_chimney(origins, directions, chimney):
    """Intersect rays with the two annular lower-chimney wall segments.

    Origins already inside a wall segment are ignored.  This gives each
    transparent effective wall a single surface interaction rather than an
    artificial second interaction at the exit face.
    """
    origins = jnp.atleast_2d(origins)
    directions = _normalise(jnp.atleast_2d(directions))
    inner = jnp.asarray(chimney.inner_radii_m)[None, :]
    outer = jnp.asarray(chimney.outer_radii_m)[None, :]
    z_min = jnp.asarray(chimney.z_min_m)[None, :]
    z_max = jnp.asarray(chimney.z_max_m)[None, :]
    count = inner.shape[1]
    rho2 = jnp.sum(origins[:, :2] ** 2, axis=1, keepdims=True)
    inside = (
        (rho2 >= inner**2)
        & (rho2 <= outer**2)
        & (origins[:, 2:3] >= z_min)
        & (origins[:, 2:3] <= z_max)
    )

    ox = origins[:, 0:1]
    oy = origins[:, 1:2]
    dx = directions[:, 0:1]
    dy = directions[:, 1:2]
    qa = dx**2 + dy**2
    qb = 2.0 * (ox * dx + oy * dy)
    side_distances = []
    side_normals = []
    for radius, sign in ((inner, -1.0), (outer, 1.0)):
        qc = ox**2 + oy**2 - radius**2
        discriminant = qb**2 - 4.0 * qa * qc
        root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        denominator = jnp.where(qa > 1e-12, 2.0 * qa, 1.0)
        roots = jnp.stack([
            (-qb - root) / denominator,
            (-qb + root) / denominator,
        ], axis=2)
        z_hit = origins[:, 2:3, None] + roots * directions[:, 2:3, None]
        valid = (
            (discriminant[:, :, None] >= 0.0)
            & (qa[:, :, None] > 1e-12)
            & (roots > 1e-6)
            & (z_hit >= z_min[:, :, None] - 1e-7)
            & (z_hit <= z_max[:, :, None] + 1e-7)
            & (~inside[:, :, None])
        )
        distances = jnp.where(valid, roots, LARGE_DISTANCE)
        chosen = jnp.argmin(distances, axis=2)
        rows = jnp.arange(origins.shape[0])[:, None]
        columns = jnp.arange(count)[None, :]
        distance = distances[rows, columns, chosen]
        points = origins[:, None, :] + distance[:, :, None] * (
            directions[:, None, :]
        )
        radial = jnp.stack([
            points[:, :, 0],
            points[:, :, 1],
            jnp.zeros_like(points[:, :, 0]),
        ], axis=2)
        radial /= jnp.linalg.norm(radial, axis=2, keepdims=True) + 1e-12
        side_distances.append(distance)
        side_normals.append(sign * radial)

    cap_distances = []
    cap_normals = []
    for z_value, sign in ((z_min, -1.0), (z_max, 1.0)):
        distance = (z_value - origins[:, 2:3]) / jnp.where(
            jnp.abs(directions[:, 2:3]) > 1e-12,
            directions[:, 2:3],
            1.0,
        )
        points = origins[:, None, :] + distance[:, :, None] * (
            directions[:, None, :]
        )
        cap_rho2 = jnp.sum(points[:, :, :2] ** 2, axis=2)
        valid = (
            (jnp.abs(directions[:, 2:3]) > 1e-12)
            & (distance > 1e-6)
            & (cap_rho2 >= inner**2)
            & (cap_rho2 <= outer**2)
            & (~inside)
        )
        cap_distances.append(
            jnp.where(valid, distance, LARGE_DISTANCE)
        )
        normal = jnp.zeros_like(points).at[:, :, 2].set(sign)
        cap_normals.append(normal)

    distances = jnp.stack(side_distances + cap_distances, axis=2)
    normals = jnp.stack(side_normals + cap_normals, axis=2)
    face = jnp.argmin(distances, axis=2)
    rows = jnp.arange(origins.shape[0])[:, None]
    columns = jnp.arange(count)[None, :]
    per_segment = distances[rows, columns, face]
    per_normal = normals[rows, columns, face]
    segment = jnp.argmin(per_segment, axis=1)
    ray_rows = jnp.arange(origins.shape[0])
    nearest = per_segment[ray_rows, segment]
    valid = nearest < LARGE_DISTANCE
    positions = origins + nearest[:, None] * directions
    outward = per_normal[ray_rows, segment]
    codes = jnp.asarray(chimney.surface_codes)[segment]
    return SurfaceHit(
        distance_m=nearest,
        valid=valid,
        object_indices=jnp.where(valid, segment, -1),
        surface_codes=jnp.where(valid, codes, -1),
        positions=positions,
        transport_normals=-outward,
    )


def intersect_acrylic_bowls(origins, directions, layout,
                             candidate_indices,
                             surface_offset_m=0.0):
    """Intersect rays with candidate radial truncated-cone bowls."""
    origins = jnp.atleast_2d(origins)
    directions = _normalise(jnp.atleast_2d(directions))
    candidates = jnp.asarray(candidate_indices, dtype=jnp.int32)
    safe = jnp.maximum(candidates, 0)
    axes = jnp.asarray(layout.outward_axes)[safe]
    attach = jnp.asarray(layout.attachment_points_m)[safe]
    attach = attach + surface_offset_m * axes
    base = jnp.asarray(layout.base_radii_m)[safe]
    tip = jnp.asarray(layout.tip_radii_m)[safe]
    length = jnp.asarray(layout.lengths_m)[safe]
    offset = origins[:, None, :] - attach
    ray = directions[:, None, :]
    o_axial = jnp.sum(offset * axes, axis=2)
    d_axial = jnp.sum(ray * axes, axis=2)
    o_perp = offset - o_axial[:, :, None] * axes
    d_perp = ray - d_axial[:, :, None] * axes
    slope = (tip - base) / length
    radius_at_origin = base + slope * o_axial
    inside = (
        (o_axial >= 0.0)
        & (o_axial <= length)
        & (jnp.sum(o_perp**2, axis=2) <= radius_at_origin**2)
    )

    qa = jnp.sum(d_perp**2, axis=2) - (slope * d_axial) ** 2
    qb = 2.0 * (
        jnp.sum(o_perp * d_perp, axis=2)
        - radius_at_origin * slope * d_axial
    )
    qc = jnp.sum(o_perp**2, axis=2) - radius_at_origin**2
    discriminant = qb**2 - 4.0 * qa * qc
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    denominator = jnp.where(jnp.abs(qa) > 1e-12, 2.0 * qa, 1.0)
    side_roots = jnp.stack([
        (-qb - root) / denominator,
        (-qb + root) / denominator,
    ], axis=2)
    side_axial = o_axial[:, :, None] + (
        side_roots * d_axial[:, :, None]
    )
    side_valid = (
        (discriminant[:, :, None] >= 0.0)
        & (jnp.abs(qa[:, :, None]) > 1e-12)
        & (side_roots > 1e-6)
        & (side_axial >= -1e-7)
        & (side_axial <= length[:, :, None] + 1e-7)
        & (~inside[:, :, None])
    )
    side_distances = jnp.where(
        side_valid, side_roots, LARGE_DISTANCE
    )
    side_choice = jnp.argmin(side_distances, axis=2)
    rows = jnp.arange(origins.shape[0])[:, None]
    columns = jnp.arange(candidates.shape[1])[None, :]
    side_distance = side_distances[rows, columns, side_choice]
    side_point = origins[:, None, :] + side_distance[:, :, None] * ray
    relative = side_point - attach
    axial = jnp.sum(relative * axes, axis=2)
    radial = relative - axial[:, :, None] * axes
    radial /= jnp.linalg.norm(radial, axis=2, keepdims=True) + 1e-12
    side_normal = radial - slope[:, :, None] * axes
    side_normal /= (
        jnp.linalg.norm(side_normal, axis=2, keepdims=True) + 1e-12
    )

    cap_distances = []
    cap_normals = []
    for axial_value, radius, sign in (
        (jnp.zeros_like(length), base, -1.0),
        (length, tip, 1.0),
    ):
        distance = (axial_value - o_axial) / jnp.where(
            jnp.abs(d_axial) > 1e-12, d_axial, 1.0
        )
        point = offset + distance[:, :, None] * ray
        point_axial = jnp.sum(point * axes, axis=2)
        point_perp = point - point_axial[:, :, None] * axes
        valid = (
            (jnp.abs(d_axial) > 1e-12)
            & (distance > 1e-6)
            & (jnp.sum(point_perp**2, axis=2) <= radius**2)
            & (~inside)
        )
        cap_distances.append(
            jnp.where(valid, distance, LARGE_DISTANCE)
        )
        cap_normals.append(sign * axes)

    distances = jnp.stack(
        [side_distance, *cap_distances], axis=2
    )
    normals = jnp.stack(
        [side_normal, *cap_normals], axis=2
    )
    face = jnp.argmin(distances, axis=2)
    per_candidate = distances[rows, columns, face]
    per_normal = normals[rows, columns, face]
    per_candidate = jnp.where(
        candidates >= 0, per_candidate, LARGE_DISTANCE
    )
    selected = jnp.argmin(per_candidate, axis=1)
    ray_rows = jnp.arange(origins.shape[0])
    nearest = per_candidate[ray_rows, selected]
    valid = nearest < LARGE_DISTANCE
    index = jnp.where(valid, candidates[ray_rows, selected], -1)
    positions = origins + nearest[:, None] * directions
    outward = per_normal[ray_rows, selected]
    return SurfaceHit(
        distance_m=nearest,
        valid=valid,
        object_indices=index,
        surface_codes=jnp.where(valid, ACRYLIC_BOWL, -1),
        positions=positions,
        transport_normals=-outward,
    )
