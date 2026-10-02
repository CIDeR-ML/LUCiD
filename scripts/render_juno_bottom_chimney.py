#!/usr/bin/env python
"""Render reproducible views of LUCiD's effective bottom chimney."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lucid.geometry.juno_structures import BottomChimneyGeometry  # noqa: E402


ACRYLIC_COLOR = "#25a7d9"
STEEL_COLOR = "#66717e"
LS_COLOR = "#f3c84b"


def _cylinder(axis, radius, z_min, z_max, color, alpha, linewidth=0.2):
    """Draw one cylindrical side using physical metre coordinates."""
    phi = np.linspace(0.0, 2.0 * np.pi, 80)
    z_values = np.array([z_min, z_max])
    phi, z_values = np.meshgrid(phi, z_values)
    axis.plot_surface(
        radius * np.cos(phi),
        radius * np.sin(phi),
        z_values,
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        edgecolor=color,
        shade=False,
    )


def _ring(axis, inner, outer, z_value, color, alpha):
    """Draw one annular end face."""
    phi = np.linspace(0.0, 2.0 * np.pi, 80)
    radius = np.array([inner, outer])
    phi, radius = np.meshgrid(phi, radius)
    axis.plot_surface(
        radius * np.cos(phi),
        radius * np.sin(phi),
        np.full_like(radius, z_value),
        color=color,
        alpha=alpha,
        linewidth=0.15,
        edgecolor=color,
        shade=False,
    )


def _lower_ellipsoid(axis, bottom, color, alpha):
    """Draw the outer lower-half ellipsoid of the steel enclosure."""
    radial, axial = bottom.steel_outer_axes_m
    theta = np.linspace(np.pi / 2.0, np.pi, 48)
    phi = np.linspace(0.0, 2.0 * np.pi, 96)
    theta, phi = np.meshgrid(theta, phi)
    x = radial * np.sin(theta) * np.cos(phi)
    y = radial * np.sin(theta) * np.sin(phi)
    z = bottom.steel_centre_z_m + axial * np.cos(theta)
    axis.plot_surface(
        x,
        y,
        z,
        color=color,
        alpha=alpha,
        linewidth=0.12,
        edgecolor=color,
        shade=True,
    )


def _draw_bottom_chimney(axis, bottom, alpha_scale=1.0):
    """Draw exactly the principal surfaces used by LUCiD transport."""
    acrylic = bottom.acrylic
    for index in range(len(acrylic.inner_radii_m)):
        z_min = acrylic.z_min_m[index]
        z_max = acrylic.z_max_m[index]
        _cylinder(
            axis,
            acrylic.outer_radii_m[index],
            z_min,
            z_max,
            ACRYLIC_COLOR,
            0.26 * alpha_scale,
        )
        _cylinder(
            axis,
            acrylic.inner_radii_m[index],
            z_min,
            z_max,
            LS_COLOR,
            0.18 * alpha_scale,
        )
    _ring(axis, 0.580, 0.675, -18.600, ACRYLIC_COLOR, 0.34)
    _ring(axis, 0.500, 0.675, -18.700, ACRYLIC_COLOR, 0.34)

    steel = bottom.steel_upper
    for index in range(len(steel.inner_radii_m)):
        _cylinder(
            axis,
            steel.outer_radii_m[index],
            steel.z_min_m[index],
            steel.z_max_m[index],
            STEEL_COLOR,
            0.55 * alpha_scale,
        )
        _cylinder(
            axis,
            steel.inner_radii_m[index],
            steel.z_min_m[index],
            steel.z_max_m[index],
            LS_COLOR,
            0.20 * alpha_scale,
        )
    _lower_ellipsoid(axis, bottom, STEEL_COLOR, 0.68 * alpha_scale)


def _set_equal_box(axis, x_limits, y_limits, z_limits):
    """Use physical axis ratios rather than visually inflated geometry."""
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_zlim(*z_limits)
    axis.set_box_aspect((
        x_limits[1] - x_limits[0],
        y_limits[1] - y_limits[0],
        z_limits[1] - z_limits[0],
    ))
    axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")


def render_component(bottom, output):
    """Render the standalone effective bottom-chimney component."""
    figure = plt.figure(figsize=(8.2, 8.0), constrained_layout=True)
    axis = figure.add_subplot(projection="3d")
    _draw_bottom_chimney(axis, bottom)
    axis.plot(
        [0.0, 0.0],
        [0.0, 0.0],
        [-17.70, -19.015],
        color=LS_COLOR,
        linewidth=2.0,
        label="LS bore axis",
    )
    axis.set_title("LUCiD effective JUNO bottom chimney")
    _set_equal_box(axis, (-0.75, 0.75), (-0.75, 0.75), (-19.10, -17.60))
    axis.view_init(elev=20, azim=-55)
    axis.text(0.70, 0.0, -18.15, "Acrylic wall: 80 mm")
    axis.text(0.70, 0.0, -18.66, "Acrylic flange: r = 0.675 m")
    axis.text(0.50, 0.0, -18.95, "Steel enclosure")
    axis.legend(loc="upper left")
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    plt.close(figure)


def render_south(bottom, output):
    """Render the LS/Acrylic southern hemispheres with the component."""
    figure = plt.figure(figsize=(9.0, 7.6), constrained_layout=True)
    axis = figure.add_subplot(projection="3d")
    theta = np.linspace(np.pi / 2.0, np.pi, 28)
    phi = np.linspace(0.0, 2.0 * np.pi, 52)
    theta, phi = np.meshgrid(theta, phi)
    for radius, color, label in (
        (17.700, "#58a65c", "LS boundary: 17.700 m"),
        (17.824, ACRYLIC_COLOR, "Acrylic boundary: 17.824 m"),
    ):
        axis.plot_wireframe(
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta),
            rstride=2,
            cstride=3,
            color=color,
            alpha=0.24,
            linewidth=0.38,
            label=label,
        )
    _draw_bottom_chimney(axis, bottom, alpha_scale=1.2)
    axis.scatter(
        [0.0], [0.0], [-18.70],
        s=36, color="#c0392b", label="Acrylic/steel joint",
    )
    axis.set_title("JUNO south hemisphere with effective bottom chimney")
    _set_equal_box(axis, (-18.5, 18.5), (-18.5, 18.5), (-19.3, 0.3))
    axis.view_init(elev=13, azim=-52)
    axis.legend(loc="upper left", fontsize=8)
    inset = figure.add_axes([0.68, 0.58, 0.28, 0.30], projection="3d")
    _draw_bottom_chimney(inset, bottom, alpha_scale=1.2)
    _set_equal_box(
        inset,
        (-0.75, 0.75),
        (-0.75, 0.75),
        (-19.10, -17.60),
    )
    inset.view_init(elev=20, azim=-55)
    inset.set_title("Bottom chimney (zoom)", fontsize=8)
    inset.tick_params(labelsize=6)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    plt.close(figure)


def main():
    """Render both requested PDF figures and PNG previews."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("geometry_render/bottom_chimney"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bottom = BottomChimneyGeometry.junosw_j26()
    component = args.output_dir / "bottom_chimney_3d.pdf"
    south = args.output_dir / "south_hemisphere_bottom_chimney_3d.pdf"
    render_component(bottom, component)
    render_south(bottom, south)
    print(component)
    print(south)


if __name__ == "__main__":
    main()
