#!/usr/bin/env python3
"""Figure: joint 19-parameter calibration convergence (the all-parameter optical fit).

Per-wavelength {L_s, L_a, QE} at 5 SK laser lines + 4 shared specular/diffuse reflection
parameters, with ~NS per-PMT gains profiled in closed form, fitted by Gauss-Newton on a
Neyman chi-square from a perturbed start. 2x3 panels: optics (top), reflection and per-PMT
gains (bottom); one line style per seed, dotted = truth.

    python fig_calib_convergence.py                  # fit all seeds, then plot
    python fig_calib_convergence.py --generate-data  # just run the fits
    python fig_calib_convergence.py --plot-results   # just plot from existing runs

The recipe lives in ``utils/calibration.py:CALIB`` and nowhere else: it is the keyword
arguments of ``utils/calib_run.run_seed``, which fits every seed in this process through
``lucid.fitting``. Every run records that recipe, the commit, the package versions and the
device mapping into its ``.npz``, so a saved run can be checked without trusting this script.

Cost warning: the paper's run is 3 seeds x 1200 Gauss-Newton iterations on the full SK-like
geometry (~3 h per seed on one A100). Use ``--steps`` to shorten it for a smoke test; the figure will not be the paper's.

Note on the readout: the trajectories are smoothed at PLOT time with a proportional Polyak
window (``--polyak-window``, default 150). The reported tail-average optimum is nearer W=200.
That is a readout choice only and needs no re-running.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # LUCiD/
from analysis.paper.utils import paths, provenance, calibration, calib_plots   # noqa: E402

FIGURE = 'calib_convergence'
XZOOM = 100                # iteration limit of the fast panels (diffuse keeps the full range)


def generate_data(seeds, steps, out_data):
    """Fit every seed in this process, through the library, and write one .npz each.

    Every seed shares one simulator, so the forward is traced and compiled once.
    """
    import json
    import numpy as np
    from analysis.paper.utils import calib_run
    from analysis.paper.utils.provenance import run_provenance

    recipe = dict(calibration.CALIB)
    if steps is not None:
        recipe['steps'] = int(steps)
    record = dict(recipe=json.dumps(recipe), **run_provenance(), **calib_run.map_mode())
    out_data.mkdir(parents=True, exist_ok=True)

    sim = calib_run.build_shared(recipe['n_ph'], recipe['k'], recipe['grid'])
    sources = calibration._laser_sources(recipe['intensity'])
    for s in seeds:
        print(f'[{FIGURE}] seed {s}: steps={recipe["steps"]} -> {out_data}', flush=True)
        rec = calib_run.run_seed(s, sim=sim, sources=sources, **recipe)
        np.savez(out_data / f'crb_part_{s}.npz', **rec, **record)


def plot_results(out_data, out_fig, polyak_window):
    calib_plots.convergence(out_data, out_fig, polyak_window=polyak_window, xzoom=XZOOM)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate-data', action='store_true')
    ap.add_argument('--plot-results', action='store_true')
    ap.add_argument('--seeds', default=None,
                    help=f'comma list (default {",".join(map(str, calibration.CALIB_SEEDS))})')
    ap.add_argument('--steps', type=int, default=None,
                    help='override STEPS (default: the paper\'s 1200)')
    ap.add_argument('--polyak-window', type=int, default=calibration.POLYAK_PLOT_WINDOW,
                    help='plot-time smoothing window')
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
        plot_results(out_data, out_fig, a.polyak_window)
        provenance.write_run_record(out_fig, __file__, a)


if __name__ == '__main__':
    main()
