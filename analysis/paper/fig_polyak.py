#!/usr/bin/env python3
"""Figure: vertex resolution vs Polyak averaging window, per ray count.

The reported fit is the Polyak average of the last 40 optimizer iterates; this shows
the vertex p68 as a function of that window, confirming the production choice and how
the optimum shifts with ray count.

Plot-only: reuses the nrays reconstruction trajectories, so run fig_nrays.py first.

    python fig_polyak.py                 # muon + electron, from the local nrays data
    python fig_polyak.py --backend s3df
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.paper.utils import paths, studies, polyak   # noqa: E402


def plot_results(backend, out, tags):
    ddir = paths.data_dir('nrays', backend)
    for particle, pfx in (('muon', 'mu'), ('electron', 'el')):
        pdf, curves = polyak.window_scan(base=str(ddir / f'nrays_{pfx}_'), tags=tags,
                                         out=out, particle=particle)
        if not curves:
            print(f'[skip] no nrays data for {particle} in {ddir} — run fig_nrays.py first')
        else:
            print(f'wrote {pdf}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plot-results', action='store_true', help='(default action)')
    ap.add_argument('--backend', choices=['local', 's3df'], default='local')
    ap.add_argument('--tags', default=None, help='comma list of ray tags')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    tags = a.tags.split(',') if a.tags else list(studies.NRAYS_TAGS)
    plot_results(a.backend, Path(a.out) if a.out else paths.figure_dir(), tags)


if __name__ == '__main__':
    main()
