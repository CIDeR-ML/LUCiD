"""Effective JUNO central-detector support-rod geometry.

Version 1 represents the two J26.3.2 ``StrutSteel`` tube families as finite
capped cylinders.  It deliberately excludes ball heads, fixtures and other
CAD details.  The checked-in CSV makes normal simulation independent of
CVMFS; ``config/scripts/extract_juno_support_rods.py`` regenerates that table.
"""

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from lucid.propagation.base import find_closest_sensors
from lucid.propagation.sphere import calculate_sphere_grid_centers


LARGE_DISTANCE = 1e10


class SupportRodLayout(NamedTuple):
    """Arrays defining finite radial support cylinders in metres."""

    copy_numbers: np.ndarray
    family_codes: np.ndarray
    centers_m: np.ndarray
    inward_axes: np.ndarray
    radii_m: np.ndarray
    half_lengths_m: np.ndarray


class SupportHit(NamedTuple):
    """Nearest selected support-cylinder intersection for each ray."""

    distance_m: jnp.ndarray
    valid: jnp.ndarray
    rod_indices: jnp.ndarray
    positions: jnp.ndarray
    transport_normals: jnp.ndarray


def load_juno_support_layout(csv_file_path):
    """Load the portable JUNOSW support table and derive radial axes.

    The local cylinder +z axis in the official geometry points toward the
    detector centre.  Since the two retained families are exactly radial,
    the unit axis is derived from the centre and avoids an Euler-convention
    dependency at simulation time.
    """
    path = Path(csv_file_path)
    data = np.genfromtxt(
        path, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    data = np.atleast_1d(data)
    required = {
        "copy_no", "family", "center_x_m", "center_y_m", "center_z_m",
        "radius_m", "half_length_m",
    }
    if data.dtype.names is None or not required.issubset(data.dtype.names):
        raise ValueError(f"Support CSV {path!r} has an invalid schema")
    centers = np.column_stack([
        data["center_x_m"], data["center_y_m"], data["center_z_m"]
    ]).astype(np.float64)
    norms = np.linalg.norm(centers, axis=1)
    if np.any(~np.isfinite(centers)) or np.any(norms <= 0.0):
        raise ValueError(f"Support CSV {path!r} has invalid centres")
    family_names = np.asarray(data["family"], dtype=str)
    known = {"main_85mm": 0, "secondary_60mm": 1, "main": 0}
    try:
        family_codes = np.array(
            [known[name] for name in family_names], dtype=np.int8
        )
    except KeyError as error:
        raise ValueError(f"Unknown support family {error.args[0]!r}") from error
    radii = np.asarray(data["radius_m"], dtype=np.float64)
    half_lengths = np.asarray(data["half_length_m"], dtype=np.float64)
    if np.any(radii <= 0.0) or np.any(half_lengths <= 0.0):
        raise ValueError("Support radii and half-lengths must be positive")
    return SupportRodLayout(
        copy_numbers=np.asarray(data["copy_no"], dtype=np.int32),
        family_codes=family_codes,
        centers_m=centers,
        inward_axes=-centers / norms[:, None],
        radii_m=radii,
        half_lengths_m=half_lengths,
    )


def build_support_candidate_map(layout, n_divisions, max_candidates):
    """Map each spherical angular cell to nearby support cylinders."""
    reference_radius = float(
        np.mean(np.linalg.norm(layout.centers_m, axis=1))
    )
    cell_centres = calculate_sphere_grid_centers(
        reference_radius, n_divisions
    )
    count = len(layout.centers_m)
    selected = find_closest_sensors(
        cell_centres, jnp.asarray(layout.centers_m),
        min(max_candidates, count),
    )
    if count >= max_candidates:
        return selected
    return jnp.pad(
        selected,
        ((0, 0), (0, max_candidates - count)),
        mode="edge",
    )


def _candidate_intersections(origins, directions, centres, axes,
                             radii, half_lengths):
    """Return candidate distances and outward-from-solid normals."""
    origins = origins[:, None, :]
    directions = directions / (
        jnp.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    )
    directions = directions[:, None, :]
    offset = origins - centres
    d_parallel = jnp.sum(directions * axes, axis=2)
    o_parallel = jnp.sum(offset * axes, axis=2)
    d_perp = directions - d_parallel[:, :, None] * axes
    o_perp = offset - o_parallel[:, :, None] * axes
    qa = jnp.sum(d_perp * d_perp, axis=2)
    qb = 2.0 * jnp.sum(o_perp * d_perp, axis=2)
    qc = jnp.sum(o_perp * o_perp, axis=2) - radii**2
    discriminant = qb**2 - 4.0 * qa * qc
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    denominator = jnp.where(qa > 1e-12, 2.0 * qa, 1.0)
    side_roots = jnp.stack(
        [(-qb - root) / denominator, (-qb + root) / denominator], axis=2
    )
    side_axial = o_parallel[:, :, None] + (
        side_roots * d_parallel[:, :, None]
    )
    side_valid = (
        (discriminant[:, :, None] >= 0.0)
        & (qa[:, :, None] > 1e-12)
        & (side_roots > 1e-6)
        & (jnp.abs(side_axial) <= half_lengths[:, :, None] + 1e-7)
    )
    side_t = jnp.min(
        jnp.where(side_valid, side_roots, LARGE_DISTANCE), axis=2
    )

    cap_signs = jnp.asarray([-1.0, 1.0])
    cap_t = (
        cap_signs[None, None, :] * half_lengths[:, :, None]
        - o_parallel[:, :, None]
    ) / jnp.where(
        jnp.abs(d_parallel[:, :, None]) > 1e-12,
        d_parallel[:, :, None],
        1.0,
    )
    cap_points = offset[:, :, None, :] + (
        cap_t[:, :, :, None] * directions[:, :, None, :]
    )
    cap_axial = jnp.sum(cap_points * axes[:, :, None, :], axis=3)
    cap_radial = cap_points - cap_axial[:, :, :, None] * axes[:, :, None, :]
    cap_valid = (
        (jnp.abs(d_parallel[:, :, None]) > 1e-12)
        & (cap_t > 1e-6)
        & (jnp.sum(cap_radial**2, axis=3) <= radii[:, :, None] ** 2)
    )
    cap_t = jnp.min(
        jnp.where(cap_valid, cap_t, LARGE_DISTANCE), axis=2
    )

    use_side = side_t <= cap_t
    distance = jnp.minimum(side_t, cap_t)
    point = origins + distance[:, :, None] * directions
    axial = jnp.sum((point - centres) * axes, axis=2)
    side_normal = point - centres - axial[:, :, None] * axes
    side_normal = side_normal / (
        jnp.linalg.norm(side_normal, axis=2, keepdims=True) + 1e-12
    )
    cap_normal = jnp.where(
        axial[:, :, None] >= 0.0, axes, -axes
    )
    outward_normal = jnp.where(
        use_side[:, :, None], side_normal, cap_normal
    )
    return distance, outward_normal


def intersect_capped_cylinders(origins, directions, layout,
                               candidate_indices):
    """Intersect rays with candidate finite cylinders and choose nearest.

    ``transport_normals`` use the LUCiD boundary convention: ``-normal``
    points back into the surrounding water after a surface interaction.
    """
    origins = jnp.atleast_2d(origins)
    directions = jnp.atleast_2d(directions)
    candidates = jnp.asarray(candidate_indices, dtype=jnp.int32)
    safe = jnp.maximum(candidates, 0)
    centres = jnp.asarray(layout.centers_m)[safe]
    axes = jnp.asarray(layout.inward_axes)[safe]
    radii = jnp.asarray(layout.radii_m)[safe]
    half_lengths = jnp.asarray(layout.half_lengths_m)[safe]
    distances, outward_normals = _candidate_intersections(
        origins, directions, centres, axes, radii, half_lengths
    )
    distances = jnp.where(candidates >= 0, distances, LARGE_DISTANCE)
    selected = jnp.argmin(distances, axis=1)
    rows = jnp.arange(origins.shape[0])
    nearest = distances[rows, selected]
    valid = nearest < LARGE_DISTANCE
    rod_indices = jnp.where(valid, candidates[rows, selected], -1)
    direction_unit = directions / (
        jnp.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    )
    positions = origins + nearest[:, None] * direction_unit
    outward = outward_normals[rows, selected]
    return SupportHit(
        distance_m=nearest,
        valid=valid,
        rod_indices=rod_indices,
        positions=positions,
        transport_normals=-outward,
    )
