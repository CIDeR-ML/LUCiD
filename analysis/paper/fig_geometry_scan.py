#!/usr/bin/env python3
"""Figure: tracking resolution vs detector granularity (2k-18k sensors, 1 GeV muons).

    python fig_geometry_scan.py                     # small local run + plot
    python fig_geometry_scan.py --plot-results
    python fig_geometry_scan.py --generate-data --backend s3df --events 100

Needs the per-count geometry configs (analysis/paper/geometries/SK_like_<N>_geom_config.json);
generate any missing ones with analysis/paper/make_geometries.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.paper.utils import paths, studies, run, plotting   # noqa: E402

FIGURE = 'geometry'
LOCAL_SENSORS = [4000, 10000, 16000]


def generate_data(backend, events, root_base, sensors):
    sensors = sensors or (studies.SENSORS if backend == 's3df' else LOCAL_SENSORS)
    n = events or (100 if backend == 's3df' else 20)
    ddir = paths.data_dir(FIGURE, backend)
    for cfg in studies.geom_configs(n_events=n, sensors=sensors, root_base=root_base):
        per = ddir / cfg['name']
        if backend == 's3df':
            run.submit_s3df(cfg, per, config_dir=ddir / 'configs', job_name=cfg['name'])
        else:
            run.run_local(cfg, per, events=list(range(n)))


def plot_results(backend, out, sensors):
    sensors = sensors or (studies.SENSORS if backend == 's3df' else LOCAL_SENSORS)
    ddir = paths.data_dir(FIGURE, backend)
    pdf = plotting.sensors(base=ddir / 'gscan_', out=out, max_sensors=max(sensors))
    print(f'wrote {pdf}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate-data', action='store_true')
    ap.add_argument('--plot-results', action='store_true')
    ap.add_argument('--backend', choices=['local', 's3df'], default='local')
    ap.add_argument('--events', type=int, default=None)
    ap.add_argument('--sensors', default=None, help='comma list of sensor counts')
    ap.add_argument('--root-base', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    sensors = [int(x) for x in a.sensors.split(',')] if a.sensors else None
    both = not (a.generate_data or a.plot_results)
    if a.generate_data or both:
        generate_data(a.backend, a.events, a.root_base, sensors)
    if a.plot_results or both:
        plot_results(a.backend, Path(a.out) if a.out else paths.figure_dir(), sensors)


if __name__ == '__main__':
    main()
