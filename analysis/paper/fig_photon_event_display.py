#!/usr/bin/env python3
"""Figure: 3D photon event display -- Cherenkov photons coloured by creation time.

One PhotonSim event: the optical photons scattered in 3D and coloured by their
creation time, inside a cyan detector-volume box, with a red star at the primary
vertex. Migrated (core rendering only) from the old PhotonSim
tools/visualization/visualize_photons.py -- the interactive/ipywidgets machinery is
dropped and it saves a static figure.

    python fig_photon_event_display.py                       # shipped 1 GeV muon, event 0
    python fig_photon_event_display.py --event 3 --azim -72 --elev 8

Reads photons with LUCiD's shipped loader, so a bare clone reproduces it from the
bundled data/water/muon/1000MeV_100events.root.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'      # match the other paper figures
plt.rcParams['font.size'] = 10

REPO_ROOT = Path(__file__).resolve().parents[2]        # LUCiD/
sys.path.insert(0, str(REPO_ROOT))
from analysis.paper.utils import paths                  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / 'data' / 'water' / 'muon' / '1000MeV_100events.root'


def _box(ax, bounds):
    (x0, x1), (y0, y1), (z0, z1) = bounds
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    for e in edges:
        p = v[e]
        ax.plot3D(p[:, 0], p[:, 1], p[:, 2], color='cyan', alpha=0.4, lw=1)


def _bounds(pos, pad=1.2):
    lo, hi = pos.min(0), pos.max(0)
    c = (lo + hi) / 2
    r = np.maximum(np.abs(hi - c), np.abs(lo - c)) * pad
    r = np.maximum(r, np.abs(c) + r.max() * 0.05)          # keep origin in frame
    return list(zip(c - r, c + r))


def make_display(root, event, out, max_photons, n_arrows, elev, azim):
    from lucid.sources.event_io import read_photon_data_from_photonsim
    r = read_photon_data_from_photonsim(str(root), event)
    pos = np.asarray(r['photon_origins'], float)           # (N,3), cm
    t = np.asarray(r['photon_times'], float)               # ns
    dirs = np.asarray(r['photon_directions'], float)
    E = float(r['energy'])
    n = len(pos)

    # Lay the track horizontally: map the longest (beam) axis to the display x.
    order = np.argsort(pos.max(0) - pos.min(0))[::-1]
    pos, dirs = pos[:, order], dirs[:, order]

    pos_all = pos                                          # for bounds (reordered, full)
    rng = np.random.default_rng(0)
    if n > max_photons:
        sel = rng.choice(n, max_photons, replace=False)
        pos, t, dirs = pos[sel], t[sel], dirs[sel]

    fig = plt.figure(figsize=(6.5, 4), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    try:
        ax.set_computed_zorder(False)
    except Exception:
        pass
    bounds = _bounds(pos_all)
    _box(ax, bounds)

    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=t, s=1, alpha=1.0,
                    cmap='plasma', depthshade=False)
    # a sparse set of photon-direction arrows along the track
    if n_arrows:
        ai = rng.choice(len(pos), min(n_arrows, len(pos)), replace=False)
        span = min(b[1] - b[0] for b in bounds)
        s = span * 0.22
        ax.quiver(pos[ai, 0], pos[ai, 1], pos[ai, 2],
                  dirs[ai, 0] * s, dirs[ai, 1] * s, dirs[ai, 2] * s,
                  color='black', alpha=0.4, arrow_length_ratio=0.1, lw=0.6)
    ax.scatter([0], [0], [0], color='red', s=100, marker='*', zorder=5)

    ax.set_xlim(bounds[0]); ax.set_ylim(bounds[1]); ax.set_zlim(bounds[2])
    aspect = [b[1] - b[0] for b in bounds]
    try:                                                   # zoom<1 keeps the box inside the frame
        ax.set_box_aspect(aspect, zoom=0.92)
    except TypeError:                                      # older matplotlib: no zoom kwarg
        ax.set_box_aspect(aspect)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.28, aspect=10, pad=0.04)
    cbar.set_label('Time [ns]', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    top = int(round(float(t.max()) / 5.0) * 5)             # nearest multiple of 5 to the max
    cbar.set_ticks(np.arange(0, top + 1, 5))

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    base = out / 'photon_event_display'
    for ext in ('png', 'pdf'):
        fig.savefig(f'{base}.{ext}', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'wrote {base}.pdf (+png)  [event={event}, {E:.0f} MeV, {n:,} photons]')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=str(DEFAULT_ROOT))
    ap.add_argument('--event', type=int, default=0)
    ap.add_argument('--max-photons', type=int, default=15000)
    ap.add_argument('--arrows', type=int, default=500)
    ap.add_argument('--elev', type=float, default=8.0)
    ap.add_argument('--azim', type=float, default=-72.0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    make_display(a.root, a.event, Path(a.out) if a.out else paths.figure_dir(),
                 a.max_photons, a.arrows, a.elev, a.azim)


if __name__ == '__main__':
    main()
