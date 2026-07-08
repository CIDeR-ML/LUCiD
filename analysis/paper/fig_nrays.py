#!/usr/bin/env python3
"""Figure: tracking resolution vs number of rays (muon + electron, 1 GeV, TTS=2.0).

Position / direction / t0 / momentum resolution and per-event timing as a function
of the ray budget, for muons (navy) and electrons (red).

Reproduce end-to-end (small, local, no cluster):
    python fig_nrays.py                         # generate a few events locally, then plot
    python fig_nrays.py --generate-data         # just (re)generate the local data
    python fig_nrays.py --plot-results          # just plot from whatever data exists

Full production (500 events/point on S3DF GPUs):
    python fig_nrays.py --generate-data --backend s3df --events 500 --tags 5k,10k,25k,50k,100k,150k,250k

Notes
-----
* Local defaults are deliberately small (a subset of ray tags, few events) so the
  script runs on a laptop in minutes and demonstrates the pipeline; the published
  figure uses the S3DF backend with 500 events and all seven ray tags.
* ``--root-base`` points at the directory holding ``<mu-|e->/<E>MeV_<N>events.root``
  PhotonSim inputs (default: the S3DF ROOT area). Local reproduction needs those
  inputs present; generating them from scratch is ``data_generation/generate_data.py``.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # LUCiD/
from analysis.paper.utils import paths, studies, run, plotting   # noqa: E402

FIGURE = 'nrays'
PARTICLES = ('muon', 'electron')
LOCAL_TAGS = ['5k', '25k', '100k']            # small, illustrative subset
FULL_TAGS = list(studies.NRAYS_TAGS)          # all seven


def generate_data(backend, events, root_base, tags):
    tags = tags or (FULL_TAGS if backend == 's3df' else LOCAL_TAGS)
    n = events or (500 if backend == 's3df' else 20)
    ddir = paths.data_dir(FIGURE, backend)
    for particle in PARTICLES:
        for cfg in studies.nrays_configs(particle, n_events=n, tags=tags, root_base=root_base):
            per_cfg_dir = ddir / cfg['name']
            if backend == 's3df':
                run.submit_s3df(cfg, per_cfg_dir, config_dir=ddir / 'configs',
                                partition='ampere', job_name=cfg['name'])
            else:
                run.run_local(cfg, per_cfg_dir, events=list(range(n)))


def plot_results(backend, out):
    ddir = paths.data_dir(FIGURE, backend)
    pdf = plotting.nrays_combined(
        mu_base=ddir / 'nrays_mu_', el_base=ddir / 'nrays_el_',
        mass_mu=studies.mass('muon'), mass_el=studies.mass('electron'), out=out)
    print(f'wrote {pdf}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate-data', action='store_true')
    ap.add_argument('--plot-results', action='store_true')
    ap.add_argument('--backend', choices=['local', 's3df'], default='local')
    ap.add_argument('--events', type=int, default=None, help='events per ray tag')
    ap.add_argument('--tags', default=None, help='comma list of ray tags (e.g. 5k,50k,250k)')
    ap.add_argument('--root-base', default=None, help='dir with <particle>/<E>MeV_<N>events.root')
    ap.add_argument('--out', default=None, help='figure output dir')
    a = ap.parse_args()

    tags = a.tags.split(',') if a.tags else None
    both = not (a.generate_data or a.plot_results)      # no flags => do both
    if a.generate_data or both:
        generate_data(a.backend, a.events, a.root_base, tags)
    if a.plot_results or both:
        plot_results(a.backend, Path(a.out) if a.out else paths.figure_dir())


if __name__ == '__main__':
    main()
