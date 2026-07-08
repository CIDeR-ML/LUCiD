#!/usr/bin/env python3
"""Figure: per-event optimizer convergence (p68/p90 vs iteration + final histograms).

Plot-only: reuses the nrays reconstruction output (the trajectories stored per event),
so run fig_nrays.py first (or point --backend at where that data lives).

    python fig_convergence.py                 # muon + electron, from the local nrays data
    python fig_convergence.py --backend s3df
"""
import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.paper.utils import paths, plotting   # noqa: E402

TAG = '250k'


def plot_results(backend, out, tag):
    ddir = paths.data_dir('nrays', backend)
    for particle, pfx in (('muon', 'mu'), ('electron', 'el')):
        hits = sorted(glob.glob(str(ddir / f'nrays_{pfx}_{tag}' / '*.h5')))
        if not hits:
            print(f'[skip] no nrays {tag} data for {particle} in {ddir} — run fig_nrays.py first')
            continue
        pdf = plotting.convergence(hits[0], out, label=particle)
        print(f'wrote {pdf}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plot-results', action='store_true', help='(default action)')
    ap.add_argument('--backend', choices=['local', 's3df'], default='local')
    ap.add_argument('--tag', default=TAG, help='ray tag to draw trajectories from')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    plot_results(a.backend, Path(a.out) if a.out else paths.figure_dir(), a.tag)


if __name__ == '__main__':
    main()
