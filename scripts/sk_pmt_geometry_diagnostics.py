#!/usr/bin/env python3
"""Make geometry and autodiff diagnostics for the SK 20-inch PMT model.

The plots deliberately separate three objects:

* the legacy LUCiD sphere centered on the mounting point;
* the continuous surface defined by SKDetSim's ``skdonuts.F`` constants;
* the same continuous surface and smooth coverage lookup used by new LUCiD.

The SKDetSim implementation finds torus hits in 1 mm steps.  New LUCiD solves
the same implicit surface continuously, so the two curves overlay in the shape
plot even though individual torus hit positions can differ by up to 1 mm along
a ray.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from lucid.propagation.sk_pmt import (
    BARREL_INACTIVE_BAND_Z,
    GEANT_PMT_RADIUS,
    SPHERE_CENTER_Z,
    SPHERE_RADIUS,
    SPHERE_TO_TORUS_Z,
    TORUS_MAJOR_RADIUS,
    TORUS_MINOR_RADIUS,
)
from lucid.propagation.sk_pmt_coverage import (
    create_sk20inch_surface_quadrature,
    sk20inch_projected_area,
)
from lucid.propagation.sk_pmt_jax import intersect_sk20inch_pmt_jax
from lucid.propagation.sk_pmt_lookup import (
    build_sk20inch_coverage_lookup,
    sk20inch_lookup_coverage,
)


jax.config.update("jax_enable_x64", True)

COLORS = {
    "legacy": "#D55E00",
    "sk": "#111111",
    "lucid": "#0072B2",
    "barrel": "#009E73",
    "accent": "#CC79A7",
}


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pmt_profiles():
    sphere_join_r = np.sqrt(
        SPHERE_RADIUS**2
        - (SPHERE_TO_TORUS_Z - SPHERE_CENTER_Z) ** 2
    )
    sphere_r = np.linspace(0.0, sphere_join_r, 500)
    sphere_z = SPHERE_CENTER_Z + np.sqrt(
        np.maximum(0.0, SPHERE_RADIUS**2 - sphere_r**2)
    )

    torus_join_r = TORUS_MAJOR_RADIUS + np.sqrt(
        np.maximum(0.0, TORUS_MINOR_RADIUS**2 - SPHERE_TO_TORUS_Z**2)
    )
    torus_r = np.linspace(torus_join_r, GEANT_PMT_RADIUS, 500)
    torus_z = np.sqrt(
        np.maximum(
            0.0,
            TORUS_MINOR_RADIUS**2 - (torus_r - TORUS_MAJOR_RADIUS) ** 2,
        )
    )
    return sphere_r, sphere_z, torus_r, torus_z


def plot_cross_section(output: Path) -> dict:
    sphere_r, sphere_z, torus_r, torus_z = pmt_profiles()
    legacy_r = np.linspace(-GEANT_PMT_RADIUS, GEANT_PMT_RADIUS, 800)
    legacy_z = np.sqrt(
        np.maximum(0.0, GEANT_PMT_RADIUS**2 - legacy_r**2)
    )

    fig, ax = plt.subplots(figsize=(9.0, 6.8))
    ax.plot(
        legacy_r * 100,
        legacy_z * 100,
        color=COLORS["legacy"],
        lw=2.4,
        ls="--",
        label="Legacy LUCiD: 25.4 cm sphere",
    )
    for sign in (-1.0, 1.0):
        # Draw SKDetSim first, then sparse LUCiD markers on the same curve.
        ax.plot(
            sign * sphere_r * 100,
            sphere_z * 100,
            color=COLORS["sk"],
            lw=5.0,
            label="SKDetSim skdonuts.F surface" if sign == 1 else None,
        )
        ax.plot(
            sign * torus_r * 100,
            torus_z * 100,
            color=COLORS["sk"],
            lw=5.0,
        )
        ax.plot(
            sign * sphere_r[::24] * 100,
            sphere_z[::24] * 100,
            "o",
            ms=3.6,
            mfc="white",
            mec=COLORS["lucid"],
            mew=1.2,
            label="New LUCiD continuous surface" if sign == 1 else None,
            zorder=5,
        )
        ax.plot(
            sign * torus_r[::24] * 100,
            torus_z[::24] * 100,
            "o",
            ms=3.6,
            mfc="white",
            mec=COLORS["lucid"],
            mew=1.2,
            zorder=5,
        )

    ax.axhline(0.0, color="0.45", lw=1.2, label="PMT mounting plane")
    ax.axhspan(
        0.0,
        BARREL_INACTIVE_BAND_Z * 100,
        color=COLORS["barrel"],
        alpha=0.15,
        label="Inactive only for barrel PMTs (2 cm band)",
    )
    ax.axhline(
        SPHERE_TO_TORUS_Z * 100,
        color=COLORS["accent"],
        lw=1.3,
        ls=":",
        label=f"Sphere/torus switch ({SPHERE_TO_TORUS_Z*100:.2f} cm)",
    )
    ax.annotate(
        "SK/new apex = 18.8 cm",
        xy=(0.0, (SPHERE_CENTER_Z + SPHERE_RADIUS) * 100),
        xytext=(7.5, 20.7),
        arrowprops=dict(arrowstyle="->", color=COLORS["sk"]),
        fontsize=10,
    )
    ax.annotate(
        "legacy apex = 25.4 cm",
        xy=(0.0, GEANT_PMT_RADIUS * 100),
        xytext=(-23.5, 27.0),
        arrowprops=dict(arrowstyle="->", color=COLORS["legacy"]),
        color=COLORS["legacy"],
        fontsize=10,
    )
    ax.set(
        xlabel="Distance from PMT axis [cm]",
        ylabel="Distance inward from mounting plane [cm]",
        title="PMT cross-section: SKDetSim and new LUCiD use the same bulb equation",
        xlim=(-31, 31),
        ylim=(-1.5, 29.5),
        aspect="equal",
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.29), ncol=2)
    fig.subplots_adjust(bottom=0.27)
    save(fig, output / "pmt_shape_cross_section.png")

    torus_at_switch_r = TORUS_MAJOR_RADIUS + np.sqrt(
        TORUS_MINOR_RADIUS**2 - SPHERE_TO_TORUS_Z**2
    )
    sphere_at_switch_r = np.sqrt(
        SPHERE_RADIUS**2
        - (SPHERE_TO_TORUS_Z - SPHERE_CENTER_Z) ** 2
    )
    return {
        "legacy_apex_m": GEANT_PMT_RADIUS,
        "sk_lucid_apex_m": SPHERE_CENTER_Z + SPHERE_RADIUS,
        "apex_difference_m": GEANT_PMT_RADIUS
        - (SPHERE_CENTER_Z + SPHERE_RADIUS),
        "sphere_torus_switch_z_m": SPHERE_TO_TORUS_Z,
        "rounded_constant_seam_radial_difference_m": (
            sphere_at_switch_r - torus_at_switch_r
        ),
    }


def ray_plane(angle_degrees, first, second):
    angle = np.deg2rad(angle_degrees)
    direction = np.array([np.sin(angle), 0.0, -np.cos(angle)])
    parallel = np.array([np.cos(angle), 0.0, np.sin(angle)])
    perpendicular = np.array([0.0, -1.0, 0.0])
    origins = (
        -direction[None, :]
        + first.reshape(-1, 1) * parallel[None, :]
        + second.reshape(-1, 1) * perpendicular[None, :]
    )
    return direction, origins


def plot_silhouettes(output: Path) -> None:
    angles = (0, 30, 60, 75, 90)
    n = 241
    extent = 0.34
    coordinate = np.linspace(-extent, extent, n)
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="xy")
    legacy = xx**2 + yy**2 <= GEANT_PMT_RADIUS**2

    @jax.jit
    def batch_hits(origins, direction):
        return jax.vmap(
            lambda origin: intersect_sk20inch_pmt_jax(
                origin,
                direction,
                jnp.zeros(3),
                jnp.array([0.0, 0.0, 1.0]),
            ).active
        )(origins)

    fig, axes = plt.subplots(1, len(angles), figsize=(16.0, 3.65), sharex=True, sharey=True)
    for ax, angle in zip(axes, angles):
        direction, origins = ray_plane(angle, xx.ravel(), yy.ravel())
        new_mask = np.asarray(
            batch_hits(jnp.asarray(origins), jnp.asarray(direction))
        ).reshape(n, n)
        ax.contourf(
            xx * 100,
            yy * 100,
            new_mask.astype(float),
            levels=[0.5, 1.5],
            colors=[COLORS["lucid"]],
            alpha=0.27,
        )
        ax.contour(
            xx * 100,
            yy * 100,
            new_mask.astype(float),
            levels=[0.5],
            colors=[COLORS["sk"]],
            linewidths=2.0,
        )
        ax.contour(
            xx * 100,
            yy * 100,
            legacy.astype(float),
            levels=[0.5],
            colors=[COLORS["legacy"]],
            linewidths=1.8,
            linestyles="--",
        )
        ax.set_title(rf"$\theta={angle}^\circ$")
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.set_xlabel("parallel offset [cm]")
    axes[0].set_ylabel("perpendicular offset [cm]")
    fig.suptitle(
        "Projected photocathode silhouette versus incidence angle\n"
        "black/blue = SKDetSim equation = new LUCiD; orange dashed = legacy sphere",
        y=1.08,
    )
    save(fig, output / "pmt_projected_silhouettes.png")


def plot_projected_area(output: Path) -> dict:
    angles = np.linspace(0.0, 90.0, 181)
    cap = create_sk20inch_surface_quadrature(64, 128, barrel=False)
    barrel = create_sk20inch_surface_quadrature(64, 128, barrel=True)

    def direction(angle_degrees):
        angle = jnp.deg2rad(angle_degrees)
        return jnp.array([jnp.sin(angle), 0.0, -jnp.cos(angle)])

    area_fn_cap = jax.jit(
        jax.vmap(
            lambda angle: sk20inch_projected_area(
                direction(angle), jnp.array([0.0, 0.0, 1.0]), cap
            )
        )
    )
    area_fn_barrel = jax.jit(
        jax.vmap(
            lambda angle: sk20inch_projected_area(
                direction(angle), jnp.array([0.0, 0.0, 1.0]), barrel
            )
        )
    )
    reference = np.pi * GEANT_PMT_RADIUS**2
    cap_ratio = np.asarray(area_fn_cap(jnp.asarray(angles))) / reference
    barrel_ratio = np.asarray(area_fn_barrel(jnp.asarray(angles))) / reference

    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    ax.axhline(
        1.0,
        color=COLORS["legacy"],
        lw=2.1,
        ls="--",
        label="Legacy LUCiD sphere (constant circular projection)",
    )
    ax.plot(
        angles,
        cap_ratio,
        color=COLORS["lucid"],
        lw=2.4,
        label="SK/new LUCiD cap PMT",
    )
    ax.plot(
        angles,
        barrel_ratio,
        color=COLORS["barrel"],
        lw=2.4,
        label="SK/new LUCiD barrel PMT (2 cm band removed)",
    )
    ax.set(
        xlabel="Angle between incoming ray and PMT axis [degrees]",
        ylabel=r"Projected active area / $\pi(25.4\,\mathrm{cm})^2$",
        title="Why the curved geometry changes scattered-light acceptance",
        xlim=(0, 90),
    )
    ax.grid(alpha=0.22)
    ax.legend()
    save(fig, output / "pmt_projected_area_vs_angle.png")
    return {
        f"cap_area_ratio_{int(a)}deg": float(cap_ratio[np.argmin(abs(angles - a))])
        for a in (0, 30, 60, 75, 90)
    } | {
        f"barrel_area_ratio_{int(a)}deg": float(
            barrel_ratio[np.argmin(abs(angles - a))]
        )
        for a in (0, 30, 60, 75, 90)
    }


def plot_connection_table_layout(output: Path, simulation_output: Path) -> None:
    """Plot the PMT centers and cable ordering saved by the simulation."""
    saved = np.load(simulation_output, allow_pickle=True)
    xyz = np.asarray(saved["xyz"])
    cable = np.asarray(saved["cable_id"])
    top = xyz[:, 2] > 18.0
    bottom = xyz[:, 2] < -18.0
    barrel = ~(top | bottom)
    phi = np.arctan2(xyz[barrel, 1], xyz[barrel, 0])

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.0, 4.9),
        gridspec_kw={"width_ratios": [1.0, 1.55, 1.0]},
        constrained_layout=True,
    )
    cable_norm = Normalize(vmin=float(np.min(cable)), vmax=float(np.max(cable)))
    panels = (
        (axes[0], top, "Top cap", "x [m]", "y [m]"),
        (axes[2], bottom, "Bottom cap", "x [m]", "y [m]"),
    )
    for ax, mask, title, xlabel, ylabel in panels:
        points = ax.scatter(
            xyz[mask, 0],
            xyz[mask, 1],
            c=cable[mask],
            s=7,
            cmap="viridis",
            norm=cable_norm,
            linewidths=0,
        )
        ax.set(
            title=f"{title} ({np.sum(mask):,} PMTs)",
            xlabel=xlabel,
            ylabel=ylabel,
            aspect="equal",
        )
        ax.grid(alpha=0.15)

    points = axes[1].scatter(
        np.rad2deg(phi),
        xyz[barrel, 2],
        c=cable[barrel],
        s=5,
        cmap="viridis",
        norm=cable_norm,
        linewidths=0,
    )
    axes[1].set(
        title=f"Barrel unwrapped ({np.sum(barrel):,} PMTs)",
        xlabel="azimuth [degrees]",
        ylabel="z [m]",
        xlim=(-180, 180),
    )
    axes[1].grid(alpha=0.15)
    fig.colorbar(points, ax=axes, fraction=0.022, pad=0.02, label="cable ID")
    fig.suptitle(
        "ConnectionTable_SK5 PMT layout actually used by both LUCiD runs",
        y=1.08,
    )
    save(fig, output / "connection_table_detector_layout.png")


def lookup_ray(angle_degrees, parallel_offset, perpendicular_offset=0.03):
    angle = jnp.deg2rad(angle_degrees)
    direction = jnp.array([jnp.sin(angle), 0.0, -jnp.cos(angle)])
    parallel = jnp.array([jnp.cos(angle), 0.0, jnp.sin(angle)])
    perpendicular = jnp.array([0.0, -1.0, 0.0])
    origin = (
        -direction
        + parallel_offset * parallel
        + perpendicular_offset * perpendicular
    )
    return origin, direction


def differentiability_diagnostics(output: Path) -> dict:
    # This is the production width used by temperature=0.2 and R=0.254 m.
    sigma = 0.2 * GEANT_PMT_RADIUS
    lookup = build_sk20inch_coverage_lookup(sigma)
    axis = jnp.array([0.0, 0.0, 1.0])
    pmt = jnp.zeros(3)

    def coverage(offset, angle, barrel=False):
        origin, direction = lookup_ray(angle, offset)
        return sk20inch_lookup_coverage(
            origin, direction, pmt, axis, barrel, lookup
        )

    offsets = jnp.linspace(-0.36, 0.36, 361)
    angles = (0.0, 30.0, 60.0, 75.0)
    fig, (ax_value, ax_grad) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for angle in angles:
        fn = lambda x: coverage(x, angle, False)
        values = np.asarray(jax.vmap(fn)(offsets))
        gradients = np.asarray(jax.vmap(jax.grad(fn))(offsets))
        ax_value.plot(np.asarray(offsets) * 100, values, lw=2, label=f"{angle:.0f}°")
        ax_grad.plot(np.asarray(offsets) * 100, gradients / 100, lw=2, label=f"{angle:.0f}°")
    ax_value.set(
        xlabel="parallel bundle offset [cm]",
        ylabel="smooth deposited weight",
        title="Forward value (same lookup used in simulation)",
        ylim=(-0.03, 1.03),
    )
    ax_grad.set(
        xlabel="parallel bundle offset [cm]",
        ylabel="d(weight) / d(offset in cm)",
        title="Reverse-mode gradient through that same value",
    )
    for ax in (ax_value, ax_grad):
        ax.grid(alpha=0.22)
        ax.legend(title="incidence")
    fig.suptitle("Smooth SK PMT coverage is differentiable through its edges")
    save(fig, output / "smooth_coverage_and_gradient.png")

    rng = np.random.default_rng(20260928)
    sample_angles = rng.uniform(5.0, 82.0, 100)
    sample_offsets = rng.uniform(-0.31, 0.31, 100)
    reverse = []
    forward = []
    finite = []
    step = 2.0e-5
    for angle, offset in zip(sample_angles, sample_offsets):
        fn = lambda x: coverage(x, angle, False)
        reverse.append(float(jax.grad(fn)(offset)))
        forward.append(float(jax.jvp(fn, (offset,), (1.0,))[1]))
        finite.append(float((fn(offset + step) - fn(offset - step)) / (2 * step)))
    reverse = np.asarray(reverse)
    forward = np.asarray(forward)
    finite = np.asarray(finite)

    # Test exact hit geometry away from branch transitions as well.
    exact_points = np.concatenate(
        [np.linspace(0.01, 0.19, 30), np.linspace(0.225, 0.248, 20)]
    )

    def distance(x):
        return intersect_sk20inch_pmt_jax(
            jnp.array([x, 0.0, 1.0]),
            jnp.array([0.0, 0.0, -1.0]),
            pmt,
            axis,
        ).distance

    exact_reverse = np.asarray([float(jax.grad(distance)(x)) for x in exact_points])
    exact_forward = np.asarray(
        [float(jax.jvp(distance, (x,), (1.0,))[1]) for x in exact_points]
    )
    exact_finite = np.asarray(
        [float((distance(x + step) - distance(x - step)) / (2 * step)) for x in exact_points]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    all_lookup = np.concatenate([reverse, forward, finite])
    lo, hi = np.nanpercentile(all_lookup, [1, 99])
    axes[0].plot([lo, hi], [lo, hi], color="0.5", ls="--")
    axes[0].scatter(finite, reverse, s=25, alpha=0.72, label="reverse VJP")
    axes[0].scatter(finite, forward, s=18, marker="x", label="forward JVP")
    axes[0].set(
        xlabel="finite-difference derivative",
        ylabel="autodiff derivative",
        title="Smooth coverage lookup",
        xlim=(lo, hi),
        ylim=(lo, hi),
    )

    all_exact = np.concatenate([exact_reverse, exact_forward, exact_finite])
    lo2, hi2 = np.nanmin(all_exact), np.nanmax(all_exact)
    axes[1].plot([lo2, hi2], [lo2, hi2], color="0.5", ls="--")
    axes[1].scatter(exact_finite, exact_reverse, s=28, alpha=0.75, label="reverse VJP")
    axes[1].scatter(exact_finite, exact_forward, s=20, marker="x", label="forward JVP")
    axes[1].set(
        xlabel="finite-difference derivative",
        ylabel="autodiff derivative",
        title="Exact hit distance within fixed sphere/torus branch",
        xlim=(lo2, hi2),
        ylim=(lo2, hi2),
    )
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.legend()
    fig.suptitle("Forward- and reverse-mode autodiff agree with numerical derivatives")
    save(fig, output / "autodiff_gradient_parity.png")

    return {
        "production_smooth_sigma_m": sigma,
        "lookup_reverse_vs_forward_max_abs": float(
            np.max(np.abs(reverse - forward))
        ),
        "lookup_reverse_vs_finite_max_abs": float(
            np.max(np.abs(reverse - finite))
        ),
        "lookup_reverse_vs_finite_rms": float(
            np.sqrt(np.mean((reverse - finite) ** 2))
        ),
        "exact_root_reverse_vs_forward_max_abs": float(
            np.max(np.abs(exact_reverse - exact_forward))
        ),
        "exact_root_reverse_vs_finite_max_abs": float(
            np.max(np.abs(exact_reverse - exact_finite))
        ),
        "all_gradients_finite": bool(
            np.all(np.isfinite(reverse))
            and np.all(np.isfinite(forward))
            and np.all(np.isfinite(exact_reverse))
            and np.all(np.isfinite(exact_forward))
        ),
    }


def verify_saved_geometry(old_path: Path, new_path: Path, config: Path, table: Path) -> dict:
    old = np.load(old_path, allow_pickle=True)
    new = np.load(new_path, allow_pickle=True)
    old_provenance = json.loads(str(old["provenance"]))
    new_provenance = json.loads(str(new["provenance"]))

    def recorded_hash(provenance, basename):
        matches = [value for key, value in provenance.items() if Path(key).name == basename]
        if len(matches) != 1:
            raise RuntimeError(f"expected one provenance entry for {basename}, got {matches}")
        return matches[0]

    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    table_hash = hashlib.sha256(table.read_bytes()).hexdigest()
    return {
        "old_output": str(old_path),
        "new_output": str(new_path),
        "n_pmts": int(len(new["cable_id"])),
        "n_unique_cable_ids": int(len(np.unique(new["cable_id"]))),
        "cable_id_min": int(np.min(new["cable_id"])),
        "cable_id_max": int(np.max(new["cable_id"])),
        "old_new_cable_ids_exact": bool(
            np.array_equal(old["cable_id"], new["cable_id"])
        ),
        "old_new_positions_exact": bool(np.array_equal(old["xyz"], new["xyz"])),
        "old_new_max_position_difference_m": float(
            np.max(np.abs(old["xyz"] - new["xyz"]))
        ),
        "config_path": str(config),
        "connection_table_path": str(table),
        "config_sha256": config_hash,
        "connection_table_sha256": table_hash,
        "old_recorded_config_sha256": recorded_hash(
            old_provenance, config.name
        ),
        "new_recorded_config_sha256": recorded_hash(
            new_provenance, config.name
        ),
        "old_recorded_connection_table_sha256": recorded_hash(
            old_provenance, table.name
        ),
        "new_recorded_connection_table_sha256": recorded_hash(
            new_provenance, table.name
        ),
        "all_hashes_match": bool(
            config_hash == recorded_hash(old_provenance, config.name)
            == recorded_hash(new_provenance, config.name)
            and table_hash == recorded_hash(old_provenance, table.name)
            == recorded_hash(new_provenance, table.name)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--old-output", type=Path, required=True)
    parser.add_argument("--new-output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--connection-table", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "connection_table_verification": verify_saved_geometry(
            args.old_output,
            args.new_output,
            args.config,
            args.connection_table,
        ),
        "shape": plot_cross_section(args.output_dir),
        "projected_area": plot_projected_area(args.output_dir),
        "autodiff": differentiability_diagnostics(args.output_dir),
    }
    plot_connection_table_layout(args.output_dir, args.new_output)
    plot_silhouettes(args.output_dir)
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
