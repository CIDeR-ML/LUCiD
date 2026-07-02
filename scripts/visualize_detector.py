"""visualize_detector — render a detector geometry from its config, no event needed.

Sanity-checks sensor placement for any `*_geom_config.json` (cylinder / sphere / box / string):
prints a summary and writes a 3D view of all sensor positions (and, for cylinders, the 2D
unrolled layout). Useful when adding or editing a geometry.

Run:  python scripts/visualize_detector.py --geom config/JUNO_geom_config.json
      python scripts/visualize_detector.py --geom config/IceCube86_full_geom_config.json
"""
import argparse, os
import numpy as np
from lucid.geometry import generate_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geom', required=True, help='*_geom_config.json')
    ap.add_argument('--out', default=None, help='output HTML (default: <geom-stem>_geometry.html)')
    args = ap.parse_args()

    det = generate_detector(args.geom)
    pts = np.asarray(det.all_points)
    out = args.out or (os.path.splitext(os.path.basename(args.geom))[0] + '_geometry.html')
    dtype = type(det).__name__
    print(f'{dtype}: {len(pts)} sensors')
    print(f'  x [{pts[:,0].min():.1f}, {pts[:,0].max():.1f}]  '
          f'y [{pts[:,1].min():.1f}, {pts[:,1].max():.1f}]  '
          f'z [{pts[:,2].min():.1f}, {pts[:,2].max():.1f}]  (m)')

    # 3D scatter of all sensors (works for any geometry)
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='markers',
                                 marker=dict(size=2, color=pts[:, 2], colorscale='viridis',
                                             colorbar=dict(title='z (m)'))))
    fig.update_layout(title=f'{dtype} — {len(pts)} sensors', template='plotly_dark',
                      scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=30, b=0))
    fig.write_html(out)
    print(f'  wrote {out}')

    # 2D unrolled display is cylinder-only
    if hasattr(det, 'H') and hasattr(det, 'r'):
        from lucid.visualization import create_detector_display
        png = os.path.splitext(out)[0] + '_2d.png'
        disp = create_detector_display(args.geom, sparse=False)
        disp(np.zeros(len(pts)), np.zeros(len(pts)), file_name=png)
        print(f'  wrote {png} (2D unrolled)')
    else:
        print('  (2D unrolled view is cylinder-only; skipped for this geometry)')


if __name__ == '__main__':
    main()
