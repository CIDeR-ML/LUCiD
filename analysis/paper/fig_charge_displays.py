#!/usr/bin/env python3
"""Figure: data-like vs predicted charge displays across three detector geometries.

For one PhotonSim muon event, each detector (Box / Cylinder / Sphere) is shown twice:
the DATA-like charge display (real PhotonSim photons propagated with the realistic
QE/first-arrival response) next to the PREDICTION (LUCiD's differentiable forward model
for the same track). The six plotly disc renders are montaged into a 3x2 grid (detector
rows, data/prediction columns).

The expensive part (six GPU simulations) is split from the cheap part (rendering +
montage) via a charge cache, so the panels can be re-rendered without re-simulating:

    python fig_charge_displays.py                        # simulate + cache + plot (PDF)
    python fig_charge_displays.py --generate-data         # only simulate + cache (forward-sim)
    python fig_charge_displays.py --plot-results          # re-render + montage from cache

Output is the paper's figure, charge_displays.pdf (add --png for a raster): charge on a log
scale in the event viewer's charge colormap (utils.plotting.CHARGE_CMAP), on white, the detector
shell black and the dead PMTs near-black.

The prediction track is taken from the event itself: the recorded primary, rotated about its
vertex to point along (THETA, PHI) so the ring shows at an angle, with the vertex at the
detector centre. Photons and the SIREN model both ship
with LUCiD, so a clone reproduces it from data/water/muon/1000MeV_5events.root
(scripts/download_data.py).
Plotly image export needs the kaleido env layered on; run inside the container as:
    APPTAINERENV_PYTHONUSERBASE=<paths.S3DF_ENV_BASE> apptainer exec ... \
        /opt/conda/bin/python3 analysis/paper/fig_charge_displays.py
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]        # LUCiD/
sys.path.insert(0, str(REPO_ROOT))
from analysis.paper.utils import paths, plotting, provenance      # noqa: E402

CONFIG = REPO_ROOT / 'config'
FIGURE = 'charge_displays'
DEFAULT_ROOT = REPO_ROOT / 'data' / 'water' / 'muon' / '1000MeV_5events.root'
# The paper's box, in place of config/MidBox_geom_config.json's dimensions and sensors.
BOX = dict(length=8.0, width=20.0, height=10.0, n_sensors=6480, sensor_radius=0.11)

# (label, geom config, physics config) — one water event rendered in each geometry.
DETECTORS = [
    ('Cylinder', 'SK_like_geom_config.json', 'SK_like_physics_config.json'),
    ('Sphere',   'JUNO_geom_config.json',    'JUNO_physics_config.json'),
    ('Box',      'MidBox_geom_config.json',  'MidBox_physics_config.json'),
]
COLUMNS = ['Data', 'Prediction']

# Per-detector zoom in the montage (<1 leaves more black margin around the view). The
# cylinder is zoomed out less so it doesn't look small next to the box and sphere.
ZOOM = {'Box': 0.80, 'Cylinder': 0.92, 'Sphere': 0.80}

# Display the true particle at an angle instead of along the beam (+z): the data photons
# are rotated onto this direction and the prediction track is pointed the same way.
THETA = math.pi / 4        # polar angle from +z
PHI = math.pi / 6          # azimuth

# The detector shell and the dead PMTs, in two shades so the two don't merge.
SHELL_COLOR = 'black'
DEAD_PMT_COLOR = '#101010'


def _panels_dir():
    return paths.data_dir(FIGURE, 'local')


# --------------------------------------------------------------------------- simulate
def _render(det, charge, time, out_png):
    """One plotly disc panel coloured by log charge; lit sensors only, no colorbar."""
    from lucid.visualization import event_display_3d
    charge, time = np.asarray(charge), np.asarray(time)
    idx = np.where(charge > 0)[0]
    event_display_3d(
        det, idx, charge[idx], time[idx],
        plot_time=False, log_scale=True, show_all_sensors=True,
        show_colorbar=False, dark_theme=False, colorscale=plotting.CHARGE_CMAP,
        surface_color=SHELL_COLOR, inactive_color=DEAD_PMT_COLOR, figname=out_png)


# The expensive step (six forward-sim runs) writes per-sensor charge/time arrays here so
# the panels can be re-rendered with --plot-results alone (no re-simulation).
def _cache_file():
    return _panels_dir() / 'charges.npz'


def _box_config():
    """config/MidBox_geom_config.json with BOX in place of its dimensions and sensors, written
    beside the cache; returns its path."""
    import json
    cfg = json.load(open(CONFIG / 'MidBox_geom_config.json'))
    cfg['geometry_definitions'].update(BOX)
    _panels_dir().mkdir(parents=True, exist_ok=True)
    out = _panels_dir() / 'MidBox_resolved.json'
    with open(out, 'w') as f:
        json.dump(cfg, f, indent=2)
    return str(out)


def generate_data(root, entry, n_photons, K, temperature, particle):
    """Simulate the charge displays and cache the per-sensor arrays (no rendering)."""
    import jax
    import jax.numpy as jnp
    from lucid.simulation import setup_event_simulator
    from lucid.fitting import track_from_vec9, vec9_from_track
    from lucid.utils import rotation_onto, spherical_to_cartesian
    from analysis.paper.utils import events

    raw, E, v0, d0 = events.load_event(root, entry)            # the pipeline's reader + primary
    d1 = spherical_to_cartesian(THETA, PHI, np)                    # display it at (THETA, PHI)
    R = rotation_onto(np.asarray(d0, float), d1, np)           # swings the primary d0 -> d1
    th9 = vec9_from_track(E, np.zeros(3), d1)                  # prediction track points along d1
    track = track_from_vec9(jnp.asarray(th9))                  # centered event (vertex = origin)
    photon_dict = events.pad_event(raw, n_photons, primary_vertex=v0, rotation=R,
                                   vertex=np.zeros(3))
    key = jax.random.PRNGKey(0)
    print(f'event {entry}: {E:.0f} MeV, true dir {np.round(d0, 3)} -> '
          f'display dir {np.round(d1, 3)} (theta={THETA:.3f}, phi={PHI:.3f}), '
          f'{n_photons:,} photons', flush=True)

    cache = {}
    for label, geom, phys in DETECTORS:
        geom_p = _box_config() if label == 'Box' else str(CONFIG / geom)
        common = dict(K=K, default_detector_params=True, physics_config=str(CONFIG / phys),
                      particle=particle, hits='first_arrival', detection='sampled')
        data_sim = setup_event_simulator(geom_p, n_photons, temperature=None,
                                         source='photons', **common)
        cd, td = data_sim(key, photon_dict)
        pred_sim = setup_event_simulator(geom_p, n_photons, temperature=temperature,
                                         source='particle', **common)
        cp, tp = pred_sim(track, key)
        cache[f'{label}_data_c'] = np.asarray(cd); cache[f'{label}_data_t'] = np.asarray(td)
        cache[f'{label}_pred_c'] = np.asarray(cp); cache[f'{label}_pred_t'] = np.asarray(tp)
        print(f'  {label}: data {int((np.asarray(cd) > 0).sum())} lit / '
              f'pred {int((np.asarray(cp) > 0).sum())} lit', flush=True)
    _panels_dir().mkdir(parents=True, exist_ok=True)
    np.savez(_cache_file(), **cache)
    print(f'cached charges -> {_cache_file()}')


def render_panels():
    """Re-render the six plotly panels from the cached charges (cheap; no re-simulation)."""
    from lucid.geometry import generate_detector
    cache_file = _cache_file()
    if not cache_file.exists():
        print(f'[skip] no charge cache at {cache_file} — run --generate-data first'); return False
    cache = np.load(cache_file)
    d = _panels_dir()
    for label, geom, _ in DETECTORS:
        det = generate_detector(_box_config() if label == 'Box' else str(CONFIG / geom))
        for tag in ('data', 'pred'):
            _render(det, cache[f'{label}_{tag}_c'], cache[f'{label}_{tag}_t'],
                    str(d / f'panel_{label}_{tag}.png'))
    return True


# --------------------------------------------------------------------------- montage
def _load_rgb(p):
    from PIL import Image
    im = Image.open(p).convert('RGBA')
    bg = Image.new('RGBA', im.size, (255, 255, 255, 255))     # flatten onto the white background
    bg.alpha_composite(im)
    return bg.convert('RGB')


def _autocrop(im, pad=8):
    """Trim the uniform white margin around a panel down to its content."""
    from PIL import Image, ImageChops
    bbox = ImageChops.difference(im, Image.new('RGB', im.size, 'white')).getbbox()
    if bbox:
        l, t, r, b = bbox
        im = im.crop((max(0, l - pad), max(0, t - pad),
                      min(im.width, r + pad), min(im.height, b + pad)))
    return im


def plot_results(out, write_png=False):
    """Montage the six panels 3x2: rows = detectors, columns = data/prediction."""
    from PIL import Image
    d = _panels_dir()
    cells = {}
    for label, *_ in DETECTORS:
        for col, tag in zip(COLUMNS, ('data', 'pred')):
            p = d / f'panel_{label}_{tag}.png'
            if not p.exists():
                print(f'[skip] missing {p} — run --generate-data first'); return
            cells[(label, col)] = _autocrop(_load_rgb(p))

    cw = max(im.width for im in cells.values())
    ch = max(im.height for im in cells.values())
    # No text, no gutters: cells butt together on a continuous background field.
    canvas = Image.new('RGB', (len(COLUMNS) * cw, len(DETECTORS) * ch), 'white')
    for r, (det_lab, *_) in enumerate(DETECTORS):
        for c, col_lab in enumerate(COLUMNS):
            im = cells[(det_lab, col_lab)]
            sc = min(cw / im.width, ch / im.height) * ZOOM.get(det_lab, 1.0)
            nw, nh = int(im.width * sc), int(im.height * sc)
            im = im.resize((nw, nh))
            canvas.paste(im, (c * cw + (cw - nw) // 2, r * ch + (ch - nh) // 2))

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    base = out / 'charge_displays'
    if write_png:
        canvas.save(f'{base}.png')
    import matplotlib.pyplot as plt                            # PDF via mpl (PIL lacks JPEG codec)
    fig = plt.figure(figsize=(canvas.width / 200, canvas.height / 200), dpi=200,
                     facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(np.asarray(canvas)); ax.axis('off')
    fig.savefig(f'{base}.pdf', facecolor='white'); plt.close(fig)
    print(f'wrote {base}.pdf{" (+png)" if write_png else ""}  ({canvas.width}x{canvas.height})')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate-data', action='store_true',
                    help='simulate the 6 charge displays and cache the arrays (forward-sim step)')
    ap.add_argument('--plot-results', action='store_true',
                    help='re-render + montage from the cache (fast)')
    ap.add_argument('--root', default=str(DEFAULT_ROOT))
    ap.add_argument('--entry', type=int, default=3)
    ap.add_argument('--particle', default='muon')
    ap.add_argument('--n-photons', type=int, default=1_000_000)
    ap.add_argument('--K', type=int, default=6)
    ap.add_argument('--temperature', type=float, default=0.05)
    ap.add_argument('--png', action='store_true', help='also write a PNG (default: PDF only)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    both = not (a.generate_data or a.plot_results)
    if a.generate_data or both:
        generate_data(a.root, a.entry, a.n_photons, a.K, a.temperature, a.particle)
        provenance.write_run_record(_panels_dir(), __file__, a)
    if (a.plot_results or both) and render_panels():
        out = Path(a.out) if a.out else paths.figure_dir()
        plot_results(out, a.png)
        provenance.write_run_record(out, __file__, a)


if __name__ == '__main__':
    main()
