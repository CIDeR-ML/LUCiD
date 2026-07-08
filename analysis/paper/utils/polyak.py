"""Polyak (tail-averaged iterate) analysis for the tracking fits.

The reported fit is the Polyak average of the last ``polyak_w`` optimizer iterates
(``fit_vec9 == mean(traj_win[-40:])`` in production). This computes the vertex p68
as a function of the averaging window, per ray count, from the stored trajectories.
"""
import glob
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

WINDOWS = [1, 5, 10, 20, 30, 40, 60, 80, 100, 120, 140]


def _vertex_p68_vs_window(h5p, windows):
    import h5py
    with h5py.File(h5p, 'r') as f:
        vtx_by_w = []
        for k in sorted(f['events']):
            ev = f['events'][k]
            tr = np.asarray(ev['traj_win']); th = np.asarray(ev['truth_vec9'])
            row = [np.linalg.norm((tr[-W:].mean(0)[1:4] - th[1:4])) * 100 for W in windows]
            vtx_by_w.append(row)
    return np.percentile(np.array(vtx_by_w), 68, axis=0)     # (n_windows,)


def window_scan(base, tags, out, particle, windows=None, production_w=40):
    """Vertex p68 vs Polyak window, one curve per ray tag. Writes ``<particle>_polyak_window_scan``."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    windows = windows or WINDOWS
    out = Path(out); out.mkdir(parents=True, exist_ok=True)

    curves = {}
    for t in tags:
        hits = sorted(glob.glob(f'{base}{t}/*.h5'))
        if not hits:
            continue
        curves[t] = _vertex_p68_vs_window(hits[0], windows)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for (t, p68), c in zip(curves.items(), plt.cm.viridis(np.linspace(0, 0.9, len(curves)))):
        ax.plot(windows, p68, 'o-', color=c, ms=4, label=t)
    ax.axvline(production_w, color='gray', ls='--', lw=1, label=f'production ({production_w})')
    ax.set_xlabel('Polyak window (last W iterations)'); ax.set_ylabel('Vertex p68 (cm)')
    ax.set_title(f'{particle}: vertex vs Polyak window')
    ax.grid(True, alpha=0.3); ax.legend(frameon=False, ncol=2, fontsize=8)
    base_out = out / f'{particle}_polyak_window_scan'
    for ext in ('png', 'pdf'):
        fig.savefig(f'{base_out}.{ext}', bbox_inches='tight', dpi=150)
    return base_out.with_suffix('.pdf'), curves
