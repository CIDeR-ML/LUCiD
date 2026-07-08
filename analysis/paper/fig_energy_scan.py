#!/usr/bin/env python3
"""Figure: tracking resolution vs energy (muons, 400-1800 MeV, TTS=2.0).

Composite time-weight policy: w=1 below 1600 MeV, w=0.75 at/above (the measured
crossover). Two reconstruction passes are generated (w=1 over the full range, w=0.75
at high energy) and the figure draws each energy point from the appropriate pass.

    python fig_energy_scan.py                       # small local run + plot
    python fig_energy_scan.py --plot-results
    python fig_energy_scan.py --generate-data --backend s3df --events 100
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.paper.utils import paths, studies, run, plotting   # noqa: E402

FIGURE = 'energy'
PARTICLE = 'muon'
LOCAL_ENERGIES = [600, 1000, 1400, 1800]


def _passes(backend, events, root_base, energies):
    n = events or (100 if backend == 's3df' else 20)
    w1 = studies.energy_configs(PARTICLE, n_events=n, energies=energies, time_weight=1.0,
                                root_base=root_base, name_prefix='escan')
    hi = [E for E in energies if E >= studies.W_CROSSOVER]
    w075 = studies.energy_configs(PARTICLE, n_events=n, energies=hi, time_weight=0.75,
                                  root_base=root_base, name_prefix='escanw75')
    return w1 + w075


def generate_data(backend, events, root_base, energies):
    energies = energies or (studies.ENERGIES if backend == 's3df' else LOCAL_ENERGIES)
    ddir = paths.data_dir(FIGURE, backend)
    for cfg in _passes(backend, events, root_base, energies):
        per = ddir / cfg['name']
        if backend == 's3df':
            run.submit_s3df(cfg, per, config_dir=ddir / 'configs', job_name=cfg['name'])
        else:
            run.run_local(cfg, per, events=list(range(cfg['n_events'])))


def plot_results(backend, out, energies):
    energies = energies or (studies.ENERGIES if backend == 's3df' else LOCAL_ENERGIES)
    ddir = paths.data_dir(FIGURE, backend)
    pdf = plotting.energy_composite(w1_dir=ddir, w075_dir=ddir, energies=energies,
                                    crossover=studies.W_CROSSOVER, out=out,
                                    max_energy=max(energies))
    print(f'wrote {pdf}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate-data', action='store_true')
    ap.add_argument('--plot-results', action='store_true')
    ap.add_argument('--backend', choices=['local', 's3df'], default='local')
    ap.add_argument('--events', type=int, default=None)
    ap.add_argument('--energies', default=None, help='comma list of energies in MeV')
    ap.add_argument('--root-base', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    energies = [int(x) for x in a.energies.split(',')] if a.energies else None
    both = not (a.generate_data or a.plot_results)
    if a.generate_data or both:
        generate_data(a.backend, a.events, a.root_base, energies)
    if a.plot_results or both:
        plot_results(a.backend, Path(a.out) if a.out else paths.figure_dir(), energies)


if __name__ == '__main__':
    main()
