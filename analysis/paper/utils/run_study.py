#!/usr/bin/env python3
"""Run one tracking-study config: reconstruct its events and log everything to one HDF5.

    python -m analysis.paper.utils.run_study --config config_00.json --output OUT_DIR [--events 0,3,7]

The config must be complete, as ``studies`` writes them. Writes ``<OUT_DIR>/<config-name>.h5``
through ``run.write_study``:

  root attrs   : the full config JSON (``config_json``) + provenance (n_sensors, hostname,
                 timestamps, lucid git commit, dirty flag, package versions) and completion
                 (``n_events_requested``, ``n_events_done``, ``finished``).
  events/ev%04d: per-event group with the 9-vectors ``[E, x,y,z, sinθ,cosθ, sinφ,cosφ, t0]``
                 of the truth, the three seeds and the fit, the fitted start's trajectory
                 (``traj_win``, one 9-vector per optimization step) and gradient norms, the fit's
                 errors, and metadata (n_hit, q_tot, energy_true, picked seed, seconds).

A failed event raises and the job fails: the file is left without ``finished``, and the figures
refuse it.

Designed to run inside the LUCiD container (see utils/submit_job.py); on a GPU
node with the layered JAX-CUDA env the per-event fit runs on the GPU.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # LUCiD/ on path
from analysis.paper.utils.pipeline import load_config          # noqa: E402
from analysis.paper.utils.run import write_study               # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='study config JSON')
    ap.add_argument('--output', required=True, help='directory for <config-name>.h5')
    ap.add_argument('--events', default=None,
                    help='comma list of event indices (overrides event_start/n_events)')
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg['name']
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{name}.h5'

    if args.events:
        events = [int(x) for x in args.events.split(',')]
    else:
        events = list(range(cfg['event_start'], cfg['event_start'] + cfg['n_events']))

    print(f"=== tracking study '{name}' ===")
    print(f"  config : {Path(args.config).resolve()}")
    print(f"  output : {out_path}")
    print(f"  events : {len(events)}  ({events[0]}..{events[-1]})")
    print(f"  started: {datetime.now():%Y-%m-%d %H:%M:%S}\n", flush=True)
    write_study(cfg, out_path, events)
    print(f"\n=== done: {len(events)} events -> {out_path} ===", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
