#!/usr/bin/env python3
"""Figure: 2x2 gallery of the four detector geometries with per-panel scale bars.

Cylinder (SK-like), Sphere (JUNO) and Box (MidBox) are rendered with the plotly disc
helper (all sensors equal charge); the Telescope (IceCube-86) is a matplotlib render
of the DOMs inside a translucent icy prism. The four panels are montaged 2x2 and each
gets a horizontal double-headed scale bar sized from the panel's real extent, so the
labels (10 m vs 200 m) convey the true size differences.

    python fig_detector_gallery.py                 # render panels + montage
    python fig_detector_gallery.py --generate-data  # just render the four panels
    python fig_detector_gallery.py --plot-results    # just montage existing panels

Plotly image export needs the kaleido env layered on; run inside the container as:
    APPTAINERENV_PYTHONUSERBASE=$LUCID_ENV_BASE apptainer exec ... \
        /opt/conda/bin/python3 analysis/paper/fig_detector_gallery.py
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]        # LUCiD/
sys.path.insert(0, str(REPO_ROOT))
from analysis.paper.utils import paths                  # noqa: E402

CONFIG = REPO_ROOT / 'config'
FIGURE = 'detector_gallery'
SURFACE = os.environ.get('SURFACE', 'black')

# (config, scale-bar length [m], label, zoom multiplier)
DISC_DETS = [
    ('SK_like_geom_config.json', 10.0, '10 m', 1.0),   # cylinder
    ('JUNO_geom_config.json',    10.0, '10 m', 1.0),   # sphere
    ('MidBox_geom_config.json',  10.0, '10 m', 0.8),   # box (zoomed out 20%)
]
TELE_CFG = 'IceCube86_full_geom_config.json'
TELE_BAR = (200.0, '200 m', 1.0)


def _panels_dir():
    return paths.data_dir(FIGURE, 'local')


# ----------------------------------------------------------------------------- render
def _hextent(cfg):
    from lucid.geometry import generate_detector
    pts = np.asarray(generate_detector(str(CONFIG / cfg)).all_points)
    return max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))


def _disc_png(cfg, out_png):
    from lucid.geometry import generate_detector
    det = generate_detector(str(CONFIG / cfg))
    n = len(det.all_points)
    det.visualize_event_data_plotly_discs(
        np.arange(n), np.ones(n), np.zeros(n),
        show_all_sensors=True, log_scale=False, show_colorbar=False,
        dark_theme=False, plot_time=False, colorscale='plasma',
        surface_color=SURFACE, figname=out_png)
    return out_png


def _telescope_png(out_png, margin=30.0, zmargin=40.0, elev=18, azim=30, msize=10.0):
    """IceCube DOMs inside a translucent icy prism following the string footprint."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import ConvexHull
    from lucid.geometry import generate_detector

    pts = np.asarray(generate_detector(str(CONFIG / TELE_CFG)).all_points)
    xy = pts[:, :2]
    poly = xy[ConvexHull(xy).vertices]
    c = poly.mean(0)
    poly = poly + margin * (poly - c) / np.linalg.norm(poly - c, axis=1, keepdims=True)
    z_lo, z_hi = pts[:, 2].min() - zmargin, pts[:, 2].max() + zmargin
    bot = [[x, y, z_lo] for x, y in poly]
    top = [[x, y, z_hi] for x, y in poly]
    M = len(poly)
    faces = [bot, top] + [[bot[i], bot[(i + 1) % M], top[(i + 1) % M], top[i]] for i in range(M)]

    fig = plt.figure(figsize=(7, 7), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    ax.set_proj_type('persp')
    try:
        ax.set_computed_zorder(False)
    except Exception:
        pass
    prism = Poly3DCollection(faces, facecolors='#c9e0e9', edgecolors='none', zorder=0)
    prism.set_alpha(0.15)
    ax.add_collection3d(prism)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=msize,
               c=[matplotlib.colormaps['plasma'](0.5)], marker='o',
               edgecolors='black', linewidths=0.5, depthshade=False, zorder=5)
    ax.set_box_aspect([np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2]) + 2 * zmargin])
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_png, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'telescope: {len(pts)} DOMs')
    return out_png


def generate_data():
    d = _panels_dir()
    for cfg, *_ in DISC_DETS:
        name = cfg.split('_geom')[0]
        _disc_png(cfg, str(d / f'panel_{name}.png'))
        print(f'disc: {name}')
    _telescope_png(str(d / 'panel_telescope.png'))


# --------------------------------------------------------------------------- montage
def _load_rgb(p):
    from PIL import Image
    im = Image.open(p).convert('RGBA')
    bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return bg.convert('RGB')


def _autocrop(im, pad=25):
    from PIL import Image, ImageChops
    bbox = ImageChops.difference(im, Image.new('RGB', im.size, 'white')).getbbox()
    if bbox:
        l, t, r, b = bbox
        im = im.crop((max(0, l - pad), max(0, t - pad),
                      min(im.width, r + pad), min(im.height, b + pad)))
    return im


def _scalebar(canvas, xc, y, bar_px, label, font, cap_px, lw=5):
    from PIL import ImageDraw
    d = ImageDraw.Draw(canvas)
    x0, x1 = xc - bar_px / 2, xc + bar_px / 2
    d.line([(x0, y), (x1, y)], fill='black', width=lw)
    for xe in (x0, x1):
        d.line([(xe, y - cap_px), (xe, y + cap_px)], fill='black', width=lw)
    tb = d.textbbox((0, 0), label, font=font)
    d.text((xc - (tb[2] - tb[0]) / 2, y + cap_px + 8), label, fill='black', font=font)


def plot_results(out):
    import matplotlib
    from PIL import Image, ImageFont
    d = _panels_dir()
    panels = []
    for cfg, L, lab, zm in DISC_DETS:
        panels.append((d / f"panel_{cfg.split('_geom')[0]}.png", _hextent(cfg), L, lab, zm))
    panels.append((d / 'panel_telescope.png', _hextent(TELE_CFG), *TELE_BAR))

    fill = float(os.environ.get('FILL', 0.82))
    cells = [(_autocrop(_load_rgb(p)), ext, L, lab, zm) for p, ext, L, lab, zm in panels]
    cw = max(c.width for c, *_ in cells)
    ch = max(c.height for c, *_ in cells)
    font = ImageFont.truetype(f'{matplotlib.get_data_path()}/fonts/ttf/DejaVuSerif.ttf',
                              int(ch * 0.052))
    cap_px, bmar = int(0.012 * ch), int(0.13 * ch)
    canvas = Image.new('RGB', (2 * cw, 2 * ch), 'white')
    for k, (c, ext, L, lab, zm) in enumerate(cells):
        row, col = divmod(k, 2)
        sc = min(cw / c.width, (ch - bmar) / c.height) * fill * zm
        nw, nh = int(c.width * sc), int(c.height * sc)
        c = c.resize((nw, nh))
        px = col * cw + (cw - nw) // 2
        py = row * ch + (ch - bmar) - nh
        canvas.paste(c, (px, py))
        _scalebar(canvas, px + nw / 2, row * ch + (ch - bmar) + int(0.03 * ch),
                  L * nw / ext, lab, font, cap_px)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    base = out / 'detector_gallery'
    canvas.save(f'{base}.png'); canvas.save(f'{base}.pdf')
    print(f'wrote {base}.pdf (+png)  ({canvas.width}x{canvas.height})')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate-data', action='store_true', help='render the four panels')
    ap.add_argument('--plot-results', action='store_true', help='montage existing panels')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    both = not (a.generate_data or a.plot_results)
    if a.generate_data or both:
        generate_data()
    if a.plot_results or both:
        plot_results(Path(a.out) if a.out else paths.figure_dir())


if __name__ == '__main__':
    main()
