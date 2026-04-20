"""Build starter shotgun visualization notebooks.

Run once from the notebooks/ directory::

    python _build_notebooks.py

Produces:
  01_waveform_single_event.ipynb
  02_per_photon_hit_map.ipynb
  03_detection_rate_scan.ipynb
"""
import json
from pathlib import Path


def _nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def _code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


HEADER_IMPORTS = """\
import sys
sys.path.append('../../../../')  # notebooks/ → photon_shotgun/ → production/ → lucid/ → repo root

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from lucid.production.photon_shotgun.io import (
    load_shotgun_waveform, load_shotgun_per_photon,
)
from lucid.production.photon_shotgun.viz import (
    scatter_sensors_3d, plot_hist_lin_log,
)
from lucid.geometry import generate_detector

plt.rcParams['figure.dpi'] = 110
plt.rcParams['figure.figsize'] = (10, 5)


def _resolve_geom(meta):
    geom_path = meta.get('detector_config', '')
    if isinstance(geom_path, bytes):
        geom_path = geom_path.decode()
    for candidate in ['../../../../' + geom_path, geom_path]:
        if os.path.exists(candidate):
            return candidate
    return geom_path
"""


# ---------------------------------------------------------------------------
# 01 — waveform single event
# ---------------------------------------------------------------------------

nb1 = _nb([
    _md("""\
# Single-event waveform inspection

Load a shotgun waveform HDF5, pick a case, and show:

1. Detector-summed waveform (linear + log y — tail is invisible in linear).
2. **3D scatter of all sensors coloured by integrated charge** (linear + log).
3. Same 3D scatter coloured by per-sensor first-hit time — shows the
   Cherenkov / time-of-flight geometry.
4. Top-10 brightest sensor waveforms with a dashed marker at direct-light ToF.
"""),
    _code(HEADER_IMPORTS),
    _code("""\
WAVE_PATH = '../../../../runs/shotgun_SK_1k_waveform.h5'
CASE = 0

out = load_shotgun_waveform(WAVE_PATH)
meta = dict(out['meta'])
num_sensors = int(meta['num_sensors'])
n_time_bins = int(meta['n_time_bins'])
bin_w = float(meta['bin_width_ns'])

print(f"file contains {int(meta['n_cases'])} cases × {int(meta['n_photons'])} photons")
print(f"waveform shape per case: ({num_sensors}, {n_time_bins})   "
      f"window={meta['window_ns']} ns, bin={bin_w} ns")
print(f"total detected: {int(out['n_detected'].sum()):,}   "
      f"dropped: {int(out['n_dropped'].sum())}")
"""),
    _md("""\
## Reconstruct one case from COO

COO storage lets us load just the nonzero entries for a single case without
materializing the full ``(n_cases, num_sensors, n_time_bins)`` tensor.
"""),
    _code("""\
m = out['case_idx'] == CASE
sid, tb, ch = out['sensor_id'][m], out['time_bin'][m], out['charge'][m]
wf_case = np.zeros((num_sensors, n_time_bins), dtype=np.float32)
wf_case[sid, tb] = ch

charge_per_sensor = wf_case.sum(axis=1)
has_hit = wf_case > 0
first_bin = np.where(has_hit.any(axis=1), np.argmax(has_hit, axis=1), -1)
first_time = np.where(first_bin >= 0, first_bin * bin_w, np.nan)

print(f"case {CASE}: detected={int(out['n_detected'][CASE])}  "
      f"sensors hit={(charge_per_sensor > 0).sum()}  "
      f"total charge={charge_per_sensor.sum():.1f}")
"""),
    _md("""\
## Detector-summed waveform
"""),
    _code("""\
total_time = wf_case.sum(axis=0)
t_axis = np.arange(n_time_bins) * bin_w

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharex=True)
for ax, yscale in zip(axes, ('linear', 'log')):
    ax.step(t_axis, total_time, where='mid', color='steelblue')
    ax.set_yscale(yscale)
    ax.set_xlabel('time (ns)')
    ax.set_ylabel(f'total charge ({yscale})')
    ax.grid(alpha=0.3)
fig.suptitle(f'case {CASE} — detector-summed waveform')
fig.tight_layout()
plt.show()
"""),
    _md("""\
## Sensor geometry
"""),
    _code("""\
det = generate_detector(_resolve_geom(meta))
sensor_points = np.asarray(det.all_points)
print(f"{sensor_points.shape[0]} sensors; z range [{sensor_points[:, 2].min():.1f}, {sensor_points[:, 2].max():.1f}]")

src = out.get('source')
source_origin = np.asarray(src.origins)[CASE, 0] if src is not None else None
if source_origin is not None:
    print(f"source origin for case {CASE}: {source_origin}")
"""),
    _md("""\
## Integrated charge per sensor (3D) — linear + log
"""),
    _code("""\
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
scatter_sensors_3d(sensor_points, charge_per_sensor, ax=ax1, log=False,
                   title=f'case {CASE} — integrated charge (linear)',
                   cbar_label='charge (PE)',
                   source_origin=source_origin)
ax2 = fig.add_subplot(122, projection='3d')
scatter_sensors_3d(sensor_points, charge_per_sensor, ax=ax2, log=True,
                   title=f'case {CASE} — integrated charge (log)',
                   cbar_label='charge (PE)',
                   source_origin=source_origin)
plt.show()
"""),
    _md("""\
## First-hit time per sensor (3D)
"""),
    _code("""\
first_time_plot = np.nan_to_num(first_time, nan=0.0)
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
scatter_sensors_3d(sensor_points, first_time_plot, ax=ax, log=False,
                   cmap='viridis', title=f'case {CASE} — first-hit time',
                   cbar_label='time (ns)',
                   source_origin=source_origin)
plt.show()
"""),
    _md("""\
## Top-10 brightest sensor waveforms

Dashed red marker: expected direct-light time-of-flight from the source
(``|r_sensor − r_source| / c_medium``).
"""),
    _code("""\
c_med = 0.2253  # m/ns in water
top_idx = np.argsort(charge_per_sensor)[-10:][::-1]
fig, axes = plt.subplots(5, 2, figsize=(11, 10), sharex=True)
for ax, sidx in zip(axes.flat, top_idx):
    ax.step(t_axis, wf_case[sidx], where='mid')
    if source_origin is not None:
        d = float(np.linalg.norm(sensor_points[sidx] - source_origin))
        ax.axvline(d / c_med, color='red', lw=0.8, alpha=0.4, ls='--')
    ax.set_title(f'sensor {sidx}  (Q={charge_per_sensor[sidx]:.1f})')
    ax.set_ylabel('charge')
    ax.grid(alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel('time (ns)')
fig.suptitle(f'case {CASE} — top-10 brightest sensors (red dashed = direct-light ToF)')
fig.tight_layout()
plt.show()
"""),
])


# ---------------------------------------------------------------------------
# 02 — per-photon hit map
# ---------------------------------------------------------------------------

nb2 = _nb([
    _md("""\
# Per-photon hit map

Requires a per-photon run (``--output-mode per_photon``):

1. Detection-fraction distribution over cases (linear + log y).
2. Hit-time distribution (linear + log y — scatter tail).
3. 3D scatter of sensors coloured by total hits over the run (linear + log).
4. 1D distribution of hits-per-sensor.
"""),
    _code(HEADER_IMPORTS),
    _code("""\
PP_PATH = '../../../../runs/shotgun_SK_1k_per_photon.h5'

out = load_shotgun_per_photon(PP_PATH)
meta = dict(out['meta'])
print('n_cases, n_photons:', int(meta['n_cases']), int(meta['n_photons']))

detected = out['detected']
sensor_id = out['sensor_id']
hit_time = out['hit_time']

frac_per_case = detected.mean(axis=1)
print(f"detection fraction: mean={frac_per_case.mean():.4f}  "
      f"std={frac_per_case.std():.4f}  "
      f"[{frac_per_case.min():.4f}, {frac_per_case.max():.4f}]")
"""),
    _md("""\
## Detection-fraction histogram (per case)
"""),
    _code("""\
plot_hist_lin_log(frac_per_case, bins=60,
                  title='detection fraction per case',
                  xlabel='fraction', color='steelblue')
plt.show()
"""),
    _md("""\
## Detected-photon hit times
"""),
    _code("""\
det_times = hit_time[detected]
plot_hist_lin_log(det_times, bins=200,
                  title='hit times (all detected photons)',
                  xlabel='time (ns)', color='coral')
plt.show()
"""),
    _md("""\
## Per-sensor hit-count 3D scatter — linear + log
"""),
    _code("""\
det = generate_detector(_resolve_geom(meta))
sensor_points = np.asarray(det.all_points)
num_sensors = sensor_points.shape[0]

valid = detected & (sensor_id >= 0)
hits_per_sensor = np.bincount(sensor_id[valid], minlength=num_sensors).astype(np.float32)
print(f"{(hits_per_sensor > 0).sum()} / {num_sensors} sensors had ≥1 hit")

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
scatter_sensors_3d(sensor_points, hits_per_sensor, ax=ax1, log=False,
                   title='hits per sensor (linear)', cbar_label='hits')
ax2 = fig.add_subplot(122, projection='3d')
scatter_sensors_3d(sensor_points, hits_per_sensor, ax=ax2, log=True,
                   title='hits per sensor (log)', cbar_label='hits')
plt.show()
"""),
    _md("""\
## Per-sensor hit-count distribution
"""),
    _code("""\
plot_hist_lin_log(hits_per_sensor[hits_per_sensor > 0], bins=80,
                  title='per-sensor hit count (sensors with ≥1 hit)',
                  xlabel='hits', color='seagreen')
plt.show()
"""),
])


# ---------------------------------------------------------------------------
# 03 — detection rate vs source position
# ---------------------------------------------------------------------------

nb3 = _nb([
    _md("""\
# Detection rate vs source position

Requires a run saved with ``--save-source``.

1. Detection-fraction histogram (linear + log y).
2. Scatter vs ρ / z with rolling-median overlay.
3. 2D ``(ρ, z)`` binned heatmap (linear + log color).
4. Detection vs distance-to-nearest-wall.
5. 3D source-position scatter coloured by detection fraction.
"""),
    _code(HEADER_IMPORTS),
    _code("""\
PATH = '../../../../runs/shotgun_SK_1k_waveform.h5'
MODE = 'waveform'   # or 'per_photon'

if MODE == 'waveform':
    out = load_shotgun_waveform(PATH)
    n_detected = out['n_detected']
else:
    out = load_shotgun_per_photon(PATH)
    n_detected = out['detected'].sum(axis=1)

meta = dict(out['meta'])
n_cases = int(meta['n_cases'])
n_photons = int(meta.get('n_photons',
                         out['detected'].shape[1] if MODE == 'per_photon' else 100_000))
frac = n_detected / n_photons
print(f"n_cases={n_cases}, n_photons={n_photons}")
print(f"detection fraction: mean={frac.mean():.4f}  median={np.median(frac):.4f}  "
      f"[{frac.min():.4f}, {frac.max():.4f}]")
"""),
    _code("""\
plot_hist_lin_log(frac, bins=60,
                  title=f'{n_cases:,} cases × {n_photons:,} photons',
                  xlabel='detection fraction')
plt.show()
"""),
    _md("""\
## Source positions
"""),
    _code("""\
src = out.get('source')
if src is None:
    raise RuntimeError('Re-run with --save-source to persist per-case '
                       'origin / direction arrays')

pos = np.asarray(src.origins)[:, 0, :]        # (n_cases, 3)
rho = np.hypot(pos[:, 0], pos[:, 1])
z = pos[:, 2]

det = generate_detector(_resolve_geom(meta))
sensor_points = np.asarray(det.all_points)
R = float(np.hypot(sensor_points[:, 0], sensor_points[:, 1]).max())
H = float(getattr(det, 'H'))
print(f"detector: R={R:.2f} m, H={H:.2f} m")
print(f"position extent: ρ ∈ [{rho.min():.2f}, {rho.max():.2f}], "
      f"z ∈ [{z.min():.2f}, {z.max():.2f}]")
"""),
    _md("""\
## Scatter + rolling median vs ρ and z
"""),
    _code("""\
def _rolling_median(x, y, n_bins=40):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    bins = np.linspace(xs[0], xs[-1], n_bins + 1)
    idx = np.digitize(xs, bins[1:-1])
    med = np.array([np.median(ys[idx == i]) if (idx == i).any() else np.nan
                    for i in range(n_bins)])
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, med

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, x_, name in zip(axes, (rho, z), ('ρ', 'z')):
    ax.scatter(x_, frac, s=3, alpha=0.25, color='steelblue')
    cx, mx = _rolling_median(x_, frac)
    ax.plot(cx, mx, 'red', lw=2, label='rolling median')
    ax.set_xlabel(f'{name} (m)')
    ax.set_ylabel('detection fraction')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=8)
fig.suptitle('detection fraction vs source position')
fig.tight_layout()
plt.show()
"""),
    _md("""\
## 2D ``(ρ, z)`` heatmap — linear + log color
"""),
    _code("""\
from matplotlib.colors import LogNorm, Normalize

H_sum, x_edges, y_edges = np.histogram2d(rho, z, bins=[40, 40], weights=n_detected)
H_cnt, _, _ = np.histogram2d(rho, z, bins=[x_edges, y_edges])
with np.errstate(invalid='ignore'):
    mean_frac = (H_sum / np.maximum(H_cnt, 1)) / n_photons

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, scale in zip(axes, ('linear', 'log')):
    data = mean_frac.T
    if scale == 'log':
        positive = data[data > 0]
        norm = LogNorm(vmin=positive.min(), vmax=data.max()) if positive.size else Normalize()
    else:
        norm = Normalize(vmin=0.0, vmax=float(data.max()))
    mesh = ax.pcolormesh(x_edges, y_edges, data, shading='auto',
                         cmap='magma', norm=norm)
    fig.colorbar(mesh, ax=ax, label='mean detection fraction')
    ax.set_xlabel('ρ (m)')
    ax.set_ylabel('z (m)')
    ax.set_title(f'{scale} scale')
fig.suptitle('detection fraction vs (ρ, z)')
fig.tight_layout()
plt.show()
"""),
    _md("""\
## Detection vs distance-to-nearest-wall
"""),
    _code("""\
d_wall = np.minimum(R - rho, H / 2 - np.abs(z))

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, yscale in zip(axes, ('linear', 'log')):
    ax.scatter(d_wall, frac, s=3, alpha=0.25, color='coral')
    cx, mx = _rolling_median(d_wall, frac)
    ax.plot(cx, mx, 'black', lw=2, label='rolling median')
    ax.set_xlabel('distance to nearest wall (m)')
    ax.set_ylabel('detection fraction')
    ax.set_yscale(yscale)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=8)
fig.suptitle('detection fraction vs wall distance')
fig.tight_layout()
plt.show()
"""),
    _md("""\
## 3D source-position scatter
"""),
    _code("""\
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=frac, cmap='viridis',
                s=6, alpha=0.7)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('source positions coloured by detection fraction')
fig.colorbar(sc, shrink=0.6, label='detection fraction')
plt.show()
"""),
])


if __name__ == '__main__':
    here = Path(__file__).parent
    for name, nb in [
        ('01_waveform_single_event.ipynb', nb1),
        ('02_per_photon_hit_map.ipynb', nb2),
        ('03_detection_rate_scan.ipynb', nb3),
    ]:
        with open(here / name, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f'wrote {name}')
