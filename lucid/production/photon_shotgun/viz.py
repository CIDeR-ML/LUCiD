"""Small plotting helpers for photon-shotgun notebooks.

Two utilities:

- ``scatter_sensors_3d`` — ``(N, 3)`` sensor positions coloured by a per-sensor
  value (charge, hit count, arrival time, …). Optional log color norm.
- ``plot_hist_lin_log`` — 1D histogram on linear + log-y side by side.

No cylinder unwrapping. 3D scatter matches the physical detector and reads
directly without extra frames.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize


def scatter_sensors_3d(sensor_points: np.ndarray, values: np.ndarray, *,
                       ax=None, log: bool = False, cmap: str = 'magma',
                       title: str = '', cbar_label: str = '',
                       s: float = 4.0,
                       zero_color=(0.12, 0.12, 0.12),
                       source_origin: Optional[np.ndarray] = None):
    """3D scatter of sensors coloured by ``values``.

    Sensors with ``value == 0`` are drawn faint so the detector outline
    remains visible. ``source_origin`` (optional 3-vector) adds a red star.
    """
    pts = np.asarray(sensor_points)
    values = np.asarray(values, dtype=float)
    mask = values > 0

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    if (~mask).any():
        ax.scatter(pts[~mask, 0], pts[~mask, 1], pts[~mask, 2],
                   c=[zero_color], s=s * 0.5, marker='.',
                   linewidths=0, alpha=0.25)

    if mask.any():
        if log:
            pos = values[values > 0]
            norm = LogNorm(vmin=pos.min(), vmax=values.max())
        else:
            norm = Normalize(vmin=0.0, vmax=float(values.max()))
        sc = ax.scatter(pts[mask, 0], pts[mask, 1], pts[mask, 2],
                        c=values[mask], cmap=cmap, norm=norm, s=s,
                        linewidths=0)
        cb = fig.colorbar(sc, ax=ax, pad=0.05, shrink=0.7)
        cb.set_label(cbar_label or 'value')

    if source_origin is not None:
        o = np.asarray(source_origin).ravel()
        ax.scatter([o[0]], [o[1]], [o[2]], marker='*', s=140, color='red',
                   edgecolors='white', linewidths=0.5, label='source')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.7)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title(title)
    return ax


def plot_hist_lin_log(x, *, bins=100, title: str = '', xlabel: str = '',
                      color='steelblue', figsize=(12, 3.5)):
    """Side-by-side histogram: linear y and log y."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    for ax, yscale in zip(axes, ('linear', 'log')):
        ax.hist(x, bins=bins, color=color)
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f'counts ({yscale})')
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
