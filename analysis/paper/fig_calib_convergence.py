#!/usr/bin/env python3
"""Figure: joint 19-parameter calibration convergence (the all-parameter optical fit).

Per-wavelength {L_s, L_a, QE} at 5 SK laser lines + 4 shared specular/diffuse reflection
parameters, with ~NS per-PMT gains profiled in closed form, fitted by Gauss-Newton on a
Neyman chi-square from a perturbed start. 2x3 panels: optics (top), reflection and per-PMT
gains (bottom); one line style per seed, dotted = truth.

    python fig_calib_convergence.py                  # fit all seeds, then plot
    python fig_calib_convergence.py --generate-data  # just run the fits
    python fig_calib_convergence.py --plot-results   # just plot from existing runs

The recipe lives in ``utils/calibration.py:CALIB_RECIPE`` and nowhere else. It is translated
into typed arguments by ``utils/calib_run.py``, which fits every seed in this process through
``lucid.fitting``. Every run records the settings it used into its ``.npz``, so a saved run can be
checked without trusting this script.

Cost warning: the published run is 3 seeds x 600 Gauss-Newton iterations on the full SK-like
geometry. Use ``--steps`` to shorten it for a smoke test; the figure will not be the paper's.

Note on the readout: the trajectories are smoothed at PLOT time with a proportional Polyak
window (``--polyak-window``, default 150). The reported tail-average optimum is nearer W=200.
That is a readout choice only and needs no re-running.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # LUCiD/
from analysis.paper.utils import paths, calibration, calib_plots   # noqa: E402

FIGURE = 'calib_convergence'


def generate_data(seeds, steps, out_data):
    """Fit every seed in this process, through the library, and write one .npz each.

    The configuration comes only from ``CALIB_RECIPE`` as typed arguments, and every seed shares
    one simulator, so the forward is traced and compiled once.
    """
    import numpy as np
    from analysis.paper.utils import calib_run

    kw = calib_run.recipe_to_kwargs(calibration.CALIB_RECIPE)
    if steps is not None:
        kw['steps'] = int(steps)
    out_data.mkdir(parents=True, exist_ok=True)

    shared = dict(n_ph=kw.pop('n_ph'), k=kw.pop('k'), intensity=kw.pop('intensity', None))
    sim, sources = calib_run.build_shared(**shared)
    for s in seeds:
        print(f'[{FIGURE}] seed {s}: STEPS={kw["steps"]} -> {out_data}', flush=True)
        rec = calib_run.run_seed(s, sim=sim, sources=sources, n_ph=shared['n_ph'],
                                 k=shared['k'], **kw)
        np.savez(out_data / f'crb_part_{s}.npz', **rec)


def plot_results(out_data, out_fig, polyak_window, xzoom):
    calib_plots.convergence(out_data, out_fig, polyak_window=polyak_window, xzoom=xzoom)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate-data', action='store_true')
    ap.add_argument('--plot-results', action='store_true')
    ap.add_argument('--seeds', default=None,
                    help=f'comma list (default {",".join(map(str, calibration.CALIB_SEEDS))})')
    ap.add_argument('--steps', type=int, default=None,
                    help='override STEPS (default: the published 600)')
    ap.add_argument('--polyak-window', type=int, default=calibration.POLYAK_PLOT_WINDOW,
                    help='plot-time smoothing window')
    ap.add_argument('--xzoom', type=int, default=100,
                    help='iteration limit on the fast panels (diffuse keeps the full range)')
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--out', default=None, help='figure output dir')
    a = ap.parse_args()

    seeds = [int(x) for x in a.seeds.split(',')] if a.seeds else list(calibration.CALIB_SEEDS)
    out_data = Path(a.data_dir) if a.data_dir else paths.data_dir(FIGURE)
    out_fig = Path(a.out) if a.out else paths.figure_dir()

    both = not (a.generate_data or a.plot_results)      # no flags => do both
    if a.generate_data or both:
        generate_data(seeds, a.steps, out_data)
    if a.plot_results or both:
        plot_results(out_data, out_fig, a.polyak_window, a.xzoom)


if __name__ == '__main__':
    main()
