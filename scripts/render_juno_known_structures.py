#!/usr/bin/env python
"""Render reproducible static views of LUCiD's JUNO structures.

The Acrylic-bowl mesh reuses the production ``acrylic_bowl_radius``
function.  The figures are explanatory views of the effective LUCiD
geometry, not CAD reconstructions.
"""

import argparse
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import (
    Line3DCollection,
    Poly3DCollection,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

from lucid.geometry.juno_structures import (  # noqa: E402
    BOWL_ARC_LENGTH_M,
    BOWL_ARC_RADIUS_M,
    acrylic_bowl_radius,
)


def _unit(vector):
    """Return one normalised three-vector."""
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _transverse_basis(axis):
    """Return two unit vectors perpendicular to ``axis``."""
    axis = _unit(axis)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, reference)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    first = _unit(np.cross(axis, reference))
    second = np.cross(axis, first)
    return first, second


def build_bowl_surface(attachment, outward_axis, axial_samples=None,
                       azimuth_samples=32):
    """Return a 3D surface mesh for one production-profile Acrylic bowl.

    Parameters
    ----------
    attachment : array-like, shape (3,)
        Point on the Acrylic shell at the bowl base, in metres.
    outward_axis : array-like, shape (3,)
        Radial outward unit-vector direction.
    axial_samples : array-like, optional
        Local outward coordinates in metres.  The default uses 33 points.
    azimuth_samples : int
        Number of samples around the rotational surface.

    Returns
    -------
    tuple of numpy.ndarray
        The x, y, and z surface coordinates.
    """
    attachment = np.asarray(attachment, dtype=np.float64)
    axis = _unit(outward_axis)
    first, second = _transverse_basis(axis)
    if axial_samples is None:
        axial_samples = np.linspace(0.0, BOWL_ARC_LENGTH_M, 33)
    axial = np.asarray(axial_samples, dtype=np.float64)
    radii = np.asarray(acrylic_bowl_radius(axial), dtype=np.float64)
    azimuth = np.linspace(
        0.0, 2.0 * np.pi, azimuth_samples, endpoint=True
    )
    radial_directions = (
        np.cos(azimuth)[:, None] * first
        + np.sin(azimuth)[:, None] * second
    )
    points = (
        attachment[None, None, :]
        + axial[:, None, None] * axis[None, None, :]
        + radii[:, None, None] * radial_directions[None, :, :]
    )
    return points[:, :, 0], points[:, :, 1], points[:, :, 2]


def _load_table(path):
    """Load one checked-in comma-separated geometry table."""
    return np.atleast_1d(np.genfromtxt(
        path, delimiter=",", names=True, dtype=None, encoding="utf-8"
    ))


def _support_segments(support):
    """Return endpoints for the checked-in radial support cylinders."""
    centres = np.column_stack([
        support["center_x_m"],
        support["center_y_m"],
        support["center_z_m"],
    ]).astype(np.float64)
    axes = centres / np.linalg.norm(centres, axis=1)[:, None]
    half_lengths = np.asarray(support["half_length_m"])[:, None]
    return centres, np.stack([
        centres - half_lengths * axes,
        centres + half_lengths * axes,
    ], axis=1)


def _bowl_layout(bowls):
    """Return Acrylic-bowl attachment points and outward axes."""
    attachments = np.column_stack([
        bowls["x_m"], bowls["y_m"], bowls["z_m"]
    ]).astype(np.float64)
    axes = attachments / np.linalg.norm(
        attachments, axis=1, keepdims=True
    )
    return attachments, axes


def _set_equal_limits(axis, limits):
    """Apply common Cartesian limits and physical axis labels."""
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_zlim(*limits[2])
    axis.set_box_aspect(tuple(hi - lo for lo, hi in limits))
    axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")


def render_global(attachments, support_segments, output_file):
    """Render the full-detector placement overview."""
    figure = plt.figure(figsize=(9.2, 7.4), constrained_layout=True)
    axis = figure.add_subplot(projection="3d")
    axis.scatter(
        attachments[:, 0], attachments[:, 1], attachments[:, 2],
        s=8, color="#7651d8", alpha=0.78, label="590 bowl anchors",
    )
    axis.add_collection3d(Line3DCollection(
        support_segments,
        colors="#d95f02",
        linewidths=0.32,
        alpha=0.18,
        label="590 support cylinders",
    ))
    polar = np.linspace(0.0, np.pi, 25)
    azimuth = np.linspace(0.0, 2.0 * np.pi, 40)
    polar, azimuth = np.meshgrid(polar, azimuth)
    for radius, colour, label in (
        (17.700, "#55a868", "LS radius 17.700 m"),
        (17.824, "#4c9ed9", "Acrylic radius 17.824 m"),
    ):
        axis.plot_wireframe(
            radius * np.sin(polar) * np.cos(azimuth),
            radius * np.sin(polar) * np.sin(azimuth),
            radius * np.cos(polar),
            rstride=3,
            cstride=4,
            color=colour,
            linewidth=0.35,
            alpha=0.23,
            label=label,
        )
    axis.set_title("LUCiD effective geometry: global layout")
    _set_equal_limits(axis, ((-20, 20), (-20, 20), (-20, 20)))
    axis.view_init(elev=19, azim=-53)
    axis.legend(loc="upper left", fontsize=8)
    figure.savefig(output_file, dpi=220)
    plt.close(figure)


def _plot_cylinder(axis, radius, z_min, z_max, colour, alpha):
    """Add one explanatory cylindrical chimney surface."""
    azimuth = np.linspace(0.0, 2.0 * np.pi, 40)
    z_axis = np.linspace(z_min, z_max, 2)
    azimuth, z_axis = np.meshgrid(azimuth, z_axis)
    axis.plot_surface(
        radius * np.cos(azimuth),
        radius * np.sin(azimuth),
        z_axis,
        color=colour,
        alpha=alpha,
        linewidth=0.0,
        shade=False,
    )


def render_top(attachments, axes, support_centres, support_segments,
               output_file):
    """Render a top-region detail with curved bowls and support rods."""
    figure = plt.figure(figsize=(10.2, 7.8), constrained_layout=True)
    axis = figure.add_subplot(projection="3d")
    bowl_mask = (
        (attachments[:, 2] > 17.45)
        & (np.abs(attachments[:, 0]) < 3.35)
        & (np.abs(attachments[:, 1]) < 3.35)
    )
    selected = np.flatnonzero(bowl_mask)
    for index in selected:
        x_axis, y_axis, z_axis = build_bowl_surface(
            attachments[index], axes[index], azimuth_samples=24
        )
        axis.plot_surface(
            x_axis,
            y_axis,
            z_axis,
            color="#7755d9",
            alpha=0.64,
            linewidth=0.0,
            shade=True,
        )
        for edge in (0, -1):
            vertices = np.column_stack([
                x_axis[edge], y_axis[edge], z_axis[edge]
            ])
            axis.add_collection3d(Poly3DCollection(
                [vertices],
                facecolors="#7755d9",
                edgecolors="none",
                alpha=0.64,
            ))
    support_mask = (
        (support_centres[:, 2] > 17.4)
        & (np.abs(support_centres[:, 0]) < 3.35)
        & (np.abs(support_centres[:, 1]) < 3.35)
    )
    axis.add_collection3d(Line3DCollection(
        support_segments[support_mask],
        colors="#e66a1f",
        linewidths=3.0,
        alpha=0.88,
    ))
    _plot_cylinder(axis, 0.524, 17.824, 18.424, "#34a8eb", 0.45)
    _plot_cylinder(axis, 0.405, 18.424, 19.55, "#60748e", 0.55)
    shell_radius = 17.824
    x_axis = np.linspace(-3.5, 3.5, 100)
    axis.plot(
        x_axis,
        np.zeros_like(x_axis),
        np.sqrt(shell_radius**2 - x_axis**2),
        color="#48b978",
        linewidth=0.9,
        alpha=0.72,
    )
    axis.text(
        -3.25, -2.7, 19.16,
        "support: capped radial cylinders",
        color="#a52a2a",
    )
    axis.text(
        0.72, -0.4, 18.12,
        f"Acrylic bowl: R={BOWL_ARC_RADIUS_M:.3f} m smooth arc",
        color="#4d32a8",
    )
    axis.text(0.72, -0.3, 18.78, "chimney steel", color="#26384c")
    axis.text(0.72, -0.3, 18.22, "chimney Acrylic", color="#1677a8")
    axis.set_title(
        "LUCiD effective geometry: curved bowls, chimney, and supports"
    )
    _set_equal_limits(
        axis, ((-3.5, 3.5), (-3.5, 3.5), (17.45, 19.60))
    )
    axis.view_init(elev=20, azim=-58)
    figure.savefig(output_file, dpi=220)
    plt.close(figure)


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("geometry_render"),
        help="Directory for known_structures_global/top.png",
    )
    return parser.parse_args()


def main():
    """Load checked-in geometry and render both static views."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bowls = _load_table(REPO_ROOT / "config/JUNO_acrylic_bowls.csv")
    support = _load_table(REPO_ROOT / "config/JUNO_support_rods.csv")
    attachments, axes = _bowl_layout(bowls)
    support_centres, support_segments = _support_segments(support)
    render_global(
        attachments,
        support_segments,
        args.output_dir / "known_structures_global.png",
    )
    render_top(
        attachments,
        axes,
        support_centres,
        support_segments,
        args.output_dir / "known_structures_top.png",
    )


if __name__ == "__main__":
    main()
