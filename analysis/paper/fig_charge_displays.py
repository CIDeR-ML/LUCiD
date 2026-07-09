#!/usr/bin/env python3
"""Figure: data-like vs predicted charge displays across three detector geometries.

For one PhotonSim muon event, each detector (Box / Cylinder / Sphere) is shown twice:
the DATA-like charge display (real PhotonSim photons propagated with the realistic
QE/first-arrival response) next to the PREDICTION (LUCiD's differentiable forward model
for the same track). The six plotly disc renders are montaged into a 3x2 grid (detector
rows, data/prediction columns) by default, or 2x3 with --layout.

The expensive part (six GPU simulations) is split from the cheap part (rendering +
montage) via a charge cache, so plotting style can be iterated without re-simulating:

    python fig_charge_displays.py                        # simulate + cache + plot (PDF)
    python fig_charge_displays.py --generate-data         # only simulate + cache (forward-sim)
    python fig_charge_displays.py --plot-results          # re-render + montage from cache
    python fig_charge_displays.py --plot-results --light  # white-background variant
    python fig_charge_displays.py --plot-results --colormap plasma --inactive-color dimgray

Output is a PDF by default (add --png for a raster). --light writes charge_displays_light.pdf
alongside the dark charge_displays.pdf.

The prediction track is taken from the event itself: PhotonSim fires the primary from the
origin, so the vertex is (0,0,0) and the direction is the photon-origin centroid, rotated
to (--theta, --phi) so the ring shows at an angle. Photons and the SIREN model both ship
with LUCiD, so a bare clone reproduces it from data/water/muon/1000MeV_100events.root.
Plotly image export needs the kaleido env layered on; run inside the container as:
    APPTAINERENV_PYTHONUSERBASE=$LUCID_ENV_BASE apptainer exec ... \
        /opt/conda/bin/python3 analysis/paper/fig_charge_displays.py
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]        # LUCiD/
sys.path.insert(0, str(REPO_ROOT))
from analysis.paper.utils import paths                  # noqa: E402

CONFIG = REPO_ROOT / 'config'
FIGURE = 'charge_displays'
DEFAULT_ROOT = REPO_ROOT / 'data' / 'water' / 'muon' / '1000MeV_100events.root'

# (label, geom config, physics config) — one water event rendered in each geometry.
DETECTORS = [
    ('Box',      'MidBox_geom_config.json',  'MidBox_physics_config.json'),
    ('Cylinder', 'SK_like_geom_config.json', 'SK_like_physics_config.json'),
    ('Sphere',   'JUNO_geom_config.json',    'JUNO_physics_config.json'),
]
COLUMNS = ['Data', 'Prediction']

# Per-detector zoom in the montage (<1 leaves more black margin around the view). The
# cylinder is zoomed out less so it doesn't look small next to the box and sphere.
ZOOM = {'Box': 0.80, 'Cylinder': 0.92, 'Sphere': 0.80}

# Display the true particle at an angle instead of along the beam (+z): the data photons
# are rotated onto this direction and the prediction track is pointed the same way.
THETA = math.pi / 4        # polar angle from +z
PHI = math.pi / 6          # azimuth


def _panels_dir():
    return paths.data_dir(FIGURE, 'local')


# --------------------------------------------------------------------------- simulate
def _dir_from_angles(theta, phi):
    return np.array([math.sin(theta) * math.cos(phi),
                     math.sin(theta) * math.sin(phi), math.cos(theta)])


def _rotation(d0, d1):
    """(axis, angle) of the rotation taking unit vector d0 onto unit vector d1."""
    axis = np.cross(d0, d1)
    s = np.linalg.norm(axis)
    if s < 1e-9:                                            # already aligned (or antiparallel)
        return np.array([1., 0., 0.]), 0.0
    return axis / s, float(np.arccos(np.clip(np.dot(d0, d1), -1.0, 1.0)))


def _pad(raw, n_photons, rot_axis, rot_angle):
    """Data-mode photon dict. Delegates tiling/units/keys to the tracking pipeline's
    _pad_event (single source of truth); only the display rotation is added on top."""
    import types
    import jax.numpy as jnp
    from analysis.tracking.pipeline import TrackingPipeline
    pd = TrackingPipeline._pad_event(types.SimpleNamespace(cfg={'nbuf': n_photons}), raw)
    if rot_angle != 0.0:                                    # swing the true dir to (THETA, PHI)
        pd['apply_rotation'] = True
        pd['rotation_axis'] = jnp.asarray(rot_axis, jnp.float32)
        pd['rotation_angle'] = jnp.asarray(rot_angle, jnp.float32)
    return pd


def _render(det, charge, time, out_png, colormap, log_scale, inactive_color, surface_color,
            dark_theme):
    """One plotly disc panel coloured by charge; lit sensors only, no colorbar.
    The detector shell (surface_color) is a different gray from the dead PMTs
    (inactive_color) so the two don't merge."""
    charge, time = np.asarray(charge), np.asarray(time)
    idx = np.where(charge > 0)[0]
    det.visualize_event_data_plotly_discs(
        idx, charge[idx], time[idx],
        plot_time=False, log_scale=log_scale, show_all_sensors=True,
        show_colorbar=False, dark_theme=dark_theme, colorscale=colormap,
        surface_color=surface_color, inactive_color=inactive_color, figname=out_png)


# The expensive step (six forward-sim runs) writes per-sensor charge/time arrays here so
# plotting style — colormap, PMT/shell colours, layout, gaps — can be iterated with
# --plot-results alone (no re-simulation). Delete it to force a fresh simulation.
def _cache_file():
    return _panels_dir() / 'charges.npz'


def generate_data(root, entry, n_photons, K, temperature, particle, theta, phi):
    """Simulate the six charge displays and cache the per-sensor arrays (no rendering)."""
    import jax
    import jax.numpy as jnp
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator
    # Same reader the tracking pipeline uses (origins in cm — the gun-frame convention the
    # data simulator expects); do NOT use lucid.sources.root_reader, which returns metres.
    from lucid.sources.event_io import read_photon_data_from_photonsim
    from lucid.fitting import track_from_vec9
    from analysis.tracking.pipeline import truth9

    raw = read_photon_data_from_photonsim(str(root), entry)
    E = float(raw['energy'])
    origins = np.asarray(raw['photon_origins'])
    d0 = origins.mean(0); d0 = d0 / np.linalg.norm(d0)         # true dir (primary fired from origin)
    d1 = _dir_from_angles(theta, phi)                          # display it at (theta, phi)
    rot_axis, rot_angle = _rotation(d0, d1)                    # swings data photons d0 -> d1
    th9, _ = truth9(np.zeros(3), d1, E, 0.0)                   # prediction track points along d1
    track = track_from_vec9(jnp.asarray(th9))
    photon_dict = _pad(raw, n_photons, rot_axis, rot_angle)
    key = jax.random.PRNGKey(0)
    print(f'event {entry}: {E:.0f} MeV, true dir {np.round(d0, 3)} -> '
          f'display dir {np.round(d1, 3)} (theta={theta:.3f}, phi={phi:.3f}), '
          f'{n_photons:,} photons', flush=True)

    cache = {}
    for label, geom, phys in DETECTORS:
        geom_p, phys_p = str(CONFIG / geom), str(CONFIG / phys)
        common = dict(K=K, default_detector_params=True, physics_config=phys_p,
                      particle=particle, detector_type=label, hit_mode='realistic')
        data_sim = setup_event_simulator(geom_p, n_photons, temperature=None,
                                         is_data=True, **common)
        pred_sim = setup_event_simulator(geom_p, n_photons, temperature=temperature,
                                         is_data=False, **common)
        cd, td = data_sim(track, key, photon_dict)
        cp, tp = pred_sim(track, key)
        for tag, c, t in (('data', cd, td), ('pred', cp, tp)):
            cache[f'{label}_{tag}_c'] = np.asarray(c)
            cache[f'{label}_{tag}_t'] = np.asarray(t)
        print(f'  {label}: data {int((np.asarray(cd) > 0).sum())} lit / '
              f'pred {int((np.asarray(cp) > 0).sum())} lit', flush=True)
    _panels_dir().mkdir(parents=True, exist_ok=True)
    np.savez(_cache_file(), **cache)
    print(f'cached charges -> {_cache_file()}')


def render_panels(colormap, log_scale, inactive_color, surface_color, dark_theme):
    """Re-render the six plotly panels from the cached charges (cheap; no re-simulation)."""
    from lucid.geometry import generate_detector
    cache_file = _cache_file()
    if not cache_file.exists():
        print(f'[skip] no charge cache at {cache_file} — run --generate-data first'); return False
    cache = np.load(cache_file)
    d = _panels_dir()
    for label, geom, _ in DETECTORS:
        det = generate_detector(str(CONFIG / geom))
        for tag in ('data', 'pred'):
            _render(det, cache[f'{label}_{tag}_c'], cache[f'{label}_{tag}_t'],
                    str(d / f'panel_{label}_{tag}.png'), colormap, log_scale,
                    inactive_color, surface_color, dark_theme)
    return True


# --------------------------------------------------------------------------- montage
def _load_rgb(p, bg_rgb):
    from PIL import Image
    im = Image.open(p).convert('RGBA')
    bg = Image.new('RGBA', im.size, bg_rgb + (255,))          # flatten onto the theme background
    bg.alpha_composite(im)
    return bg.convert('RGB')


def _autocrop(im, bg_name, pad=8):
    """Trim the uniform background margin around a panel down to its content."""
    from PIL import Image, ImageChops
    bbox = ImageChops.difference(im, Image.new('RGB', im.size, bg_name)).getbbox()
    if bbox:
        l, t, r, b = bbox
        im = im.crop((max(0, l - pad), max(0, t - pad),
                      min(im.width, r + pad), min(im.height, b + pad)))
    return im


def plot_results(out, layout, zoom=1.0, dark_theme=True, write_png=False):
    from PIL import Image
    bg_name = 'black' if dark_theme else 'white'
    bg_rgb = (0, 0, 0) if dark_theme else (255, 255, 255)
    d = _panels_dir()
    cells = {}
    for label, *_ in DETECTORS:
        for col, tag in zip(COLUMNS, ('data', 'pred')):
            p = d / f'panel_{label}_{tag}.png'
            if not p.exists():
                print(f'[skip] missing {p} — run --generate-data first'); return
            cells[(label, col)] = _autocrop(_load_rgb(p, bg_rgb), bg_name)

    cw = max(im.width for im in cells.values())
    ch = max(im.height for im in cells.values())
    rows, colnames = (DETECTORS, COLUMNS) if layout == '3x2' else (COLUMNS, DETECTORS)
    nrow, ncol = len(rows), len(colnames)

    def _row_col(r, c):
        """(detector_label, column_name) for grid cell (r, c)."""
        if layout == '3x2':                                    # rows=detectors, cols=data/pred
            return rows[r][0], colnames[c]
        return colnames[c][0], rows[r]                         # 2x3: rows=data/pred, cols=detectors

    # No text, no gutters: cells butt together on a continuous background field.
    canvas = Image.new('RGB', (ncol * cw, nrow * ch), bg_name)
    for r in range(nrow):
        for c in range(ncol):
            det_lab, col_lab = _row_col(r, c)
            im = cells[(det_lab, col_lab)]
            sc = min(cw / im.width, ch / im.height) * ZOOM.get(det_lab, 1.0) * zoom
            nw, nh = int(im.width * sc), int(im.height * sc)
            im = im.resize((nw, nh))
            px = c * cw + (cw - nw) // 2
            py = r * ch + (ch - nh) // 2
            canvas.paste(im, (px, py))

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    base = out / ('charge_displays' if dark_theme else 'charge_displays_light')
    if write_png:
        canvas.save(f'{base}.png')
    import matplotlib.pyplot as plt                            # PDF via mpl (PIL lacks JPEG codec)
    fig = plt.figure(figsize=(canvas.width / 200, canvas.height / 200), dpi=200,
                     facecolor=bg_name)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(np.asarray(canvas)); ax.axis('off')
    fig.savefig(f'{base}.pdf', facecolor=bg_name); plt.close(fig)
    print(f'wrote {base}.pdf{" (+png)" if write_png else ""}  '
          f'({canvas.width}x{canvas.height}, {layout})')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate-data', action='store_true',
                    help='simulate the 6 charge displays and cache the arrays (forward-sim step)')
    ap.add_argument('--plot-results', action='store_true',
                    help='re-render + montage from the cache (fast; iterate on style here)')
    ap.add_argument('--root', default=str(DEFAULT_ROOT))
    ap.add_argument('--entry', type=int, default=2)
    ap.add_argument('--particle', default='muon')
    ap.add_argument('--n-photons', type=int, default=1_000_000)
    ap.add_argument('--K', type=int, default=6)
    ap.add_argument('--temperature', type=float, default=0.05)
    ap.add_argument('--theta', type=float, default=THETA, help='polar angle from +z (rad)')
    ap.add_argument('--phi', type=float, default=PHI, help='azimuth (rad)')
    ap.add_argument('--colormap', default='viridis')
    ap.add_argument('--light', action='store_true', help='white-background variant')
    ap.add_argument('--inactive-color', default=None,
                    help='colour of no-hit PMTs (default: theme-appropriate gray)')
    ap.add_argument('--surface-color', default=None,
                    help='colour of the detector shell (default: theme-appropriate gray)')
    ap.add_argument('--layout', choices=['3x2', '2x3'], default='3x2')
    ap.add_argument('--zoom', type=float, default=1.0,
                    help='global zoom multiplier on top of the per-detector ZOOM (<1 = out)')
    ap.add_argument('--no-log', action='store_true', help='linear charge scale (default: log)')
    ap.add_argument('--no-render', action='store_true',
                    help='skip the plotly re-render; just re-montage existing panels '
                         '(instant — use for --zoom/--layout tweaks)')
    ap.add_argument('--png', action='store_true', help='also write a PNG (default: PDF only)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    dark = not a.light
    # Two distinct grays per theme: shell darker/lighter than the dead PMTs.
    surface = a.surface_color or ('dimgray' if dark else 'lightgray')
    inactive = a.inactive_color or ('darkgray' if dark else 'gray')
    both = not (a.generate_data or a.plot_results)
    if a.generate_data or both:
        generate_data(a.root, a.entry, a.n_photons, a.K, a.temperature,
                      a.particle, a.theta, a.phi)
    if a.plot_results or both:
        ok = (True if a.no_render else
              render_panels(a.colormap, not a.no_log, inactive, surface, dark))
        if ok:
            plot_results(Path(a.out) if a.out else paths.figure_dir(),
                         a.layout, a.zoom, dark, a.png)


if __name__ == '__main__':
    main()
