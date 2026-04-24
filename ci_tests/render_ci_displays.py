"""CI visual smoke: render Q + T event displays for a run_job.py output.

Reads the sparse sensor output at ``<output-dir>/sensor/wc_sensor_0000.h5``
and produces a single PNG containing PE (charge) and first-hit time for each
event side-by-side on the barrel+caps unwrap. Intended to be archived as a
CI artifact so a human reviewer can eyeball whether anything changed.

Usage:
    python render_ci_displays.py \
        --output-dir /path/to/run_job_output \
        --geom-config /opt/LUCiD/config/SK_like_geom_config.json \
        --out-png /path/to/ci_displays.png

Fails if the sensor file has fewer events than requested with --n-events,
or if any per-event PE sum is zero (full darkness is almost certainly a
configuration bug, not a valid physics outcome for ~GeV muons).
"""

import argparse
import os
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import pdist

sys.path.insert(0, "/opt/LUCiD")
from lucid.geometry import generate_detector


def _calc_min_distance(positions):
    dists = pdist(positions)
    return float(np.min(dists)) if len(dists) > 0 else 1.0


def build_unwrap(geom_config):
    """Compute unwrapped xy coordinates for every sensor (barrel + caps)."""
    detector = generate_detector(geom_config)
    radius = detector.r
    height = detector.H
    sensor_positions = np.array(detector.all_points)
    sensor_cases = np.array([detector.ID_to_case[i] for i in range(len(sensor_positions))])
    n_sensors = len(sensor_positions)

    caps_offset = 1.05 * height / 2 + radius
    x = np.zeros(n_sensors)
    y = np.zeros(n_sensors)

    barrel = sensor_cases == 0
    theta = np.arctan2(sensor_positions[barrel, 1], sensor_positions[barrel, 0])
    theta = (theta + np.pi * 3 / 2) % (2 * np.pi) / 2
    x[barrel] = theta * radius * 2
    y[barrel] = sensor_positions[barrel, 2]

    top = sensor_cases == 1
    x[top] = sensor_positions[top, 0] + np.pi * radius
    y[top] = caps_offset + sensor_positions[top, 1]

    bot = sensor_cases == 2
    x[bot] = sensor_positions[bot, 0] + np.pi * radius
    y[bot] = -caps_offset - sensor_positions[bot, 1]

    xy = np.column_stack((x, y))
    min_d = _calc_min_distance(xy)
    return {
        "xy": xy,
        "diameter": min_d,
        "xmin": x.min() - min_d,
        "xmax": x.max() + min_d,
        "ymin": y.min() - min_d,
        "ymax": y.max() + min_d,
        "n_sensors": n_sensors,
    }


def render_panel(ax, values, hit_mask, unwrap, title, plot_time=False,
                 perc_min=1.0, perc_max=99.0):
    # hit_mask is the authoritative "has a hit" flag from sensor_idx.
    # values on hit sensors can be any sign (T in particular can be
    # negative when the event's t0 lands in the negative half of the
    # [-250, +250] ns window).
    if hit_mask.any():
        hit_vals = values[hit_mask]
        vmin = float(np.percentile(hit_vals, perc_min))
        vmax = float(np.percentile(hit_vals, perc_max))
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0
    cmap = plt.get_cmap("viridis_r" if plot_time else "viridis")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    plot_vals = np.clip(values, vmin, vmax)
    colors = sm.to_rgba(plot_vals)
    colors[~hit_mask] = np.array([0.9, 0.9, 0.9, 1.0])

    ells = EllipseCollection(
        widths=unwrap["diameter"], heights=unwrap["diameter"], angles=0,
        units="x", facecolors=colors, offsets=unwrap["xy"],
        transOffset=ax.transData, edgecolors="none")
    ax.add_collection(ells)
    ax.set_xlim(unwrap["xmin"], unwrap["xmax"])
    ax.set_ylim(unwrap["ymin"], unwrap["ymax"])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    return sm


def sparse_to_dense(idx, vals, n_sensors):
    out = np.zeros(n_sensors, dtype=np.float32)
    out[idx] = vals
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True,
                   help="run_job.py output root (contains sensor/, labl/, ...).")
    p.add_argument("--geom-config", required=True)
    p.add_argument("--out-png", required=True)
    p.add_argument("--n-events", type=int, default=3,
                   help="Number of events to render (from file index 0).")
    p.add_argument("--label", type=str, default="",
                   help="Short provenance string to stamp on the figure (e.g. git sha).")
    args = p.parse_args()

    sensor_h5 = os.path.join(args.output_dir, "sensor", "wc_sensor_0000.h5")
    unwrap = build_unwrap(args.geom_config)
    n_sensors = unwrap["n_sensors"]

    with h5py.File(sensor_h5, "r") as f:
        event_keys = sorted([k for k in f.keys() if k.startswith("event_")])
        if len(event_keys) < args.n_events:
            raise RuntimeError(
                f"Only {len(event_keys)} events in {sensor_h5}; needed {args.n_events}.")
        events = []
        for ek in event_keys[:args.n_events]:
            g = f[ek]
            idx = np.asarray(g["sensor_idx"][()], dtype=np.int64)
            pe = np.asarray(g["PE"][()], dtype=np.float32)
            t = np.asarray(g["T"][()], dtype=np.float32)
            events.append((ek, idx, pe, t))

    summary_lines = []
    fig, axes = plt.subplots(args.n_events, 2, figsize=(14, 4 * args.n_events))
    if args.n_events == 1:
        axes = np.array([axes])

    for row, (ek, idx, pe_s, t_s) in enumerate(events):
        pe = sparse_to_dense(idx, pe_s, n_sensors)
        t = sparse_to_dense(idx, t_s, n_sensors)
        hit_mask = np.zeros(n_sensors, dtype=bool)
        hit_mask[idx] = True
        total_pe = float(pe.sum())
        n_hits = int(hit_mask.sum())
        summary_lines.append(f"  {ek}: PE_sum={total_pe:.1f}  n_hits={n_hits}")
        if total_pe <= 0.0:
            raise RuntimeError(
                f"Event {ek} has PE_sum=0 — almost certainly a configuration bug "
                f"(full darkness for a GeV-range muon in SK_like is not physical).")
        sm_pe = render_panel(
            axes[row, 0], pe, hit_mask, unwrap,
            f"{ek}  PE   total={total_pe:.0f}  hits={n_hits}")
        sm_t = render_panel(
            axes[row, 1], t, hit_mask, unwrap,
            f"{ek}  first-hit T (ns)  hits={n_hits}", plot_time=True)
        for ax, sm in ((axes[row, 0], sm_pe), (axes[row, 1], sm_t)):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            plt.colorbar(sm, cax=cax)

    suptitle = "LUCiD visual smoke — dataprod_01 (SK_like, mu-)"
    if args.label:
        suptitle += f"  [{args.label}]"
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    plt.savefig(args.out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)

    print("Rendered smoke figure:")
    print(f"  -> {args.out_png}")
    print("Per-event summary:")
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
