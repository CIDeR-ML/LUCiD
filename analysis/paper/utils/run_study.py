#!/usr/bin/env python3
"""Run one tracking-study config: reconstruct its events and log everything to one HDF5.

    python -m analysis.paper.run_study --config config_00.json --output OUT_DIR
    python analysis/paper/utils/run_study.py  --config config_00.json --output OUT_DIR [--events 0,3,7]

Writes ``<OUT_DIR>/<config-name>.h5`` with:

  root attrs   : the full config JSON (``config_json``) + derived provenance
                 (n_sensors, n_events, hostname, timestamps, lucid git commit).
  events/ev%04d: per-event group with, for the requested 7-tuple (x,y,z,phi,theta,t0,E):
                 ``truth_phys`` (true), ``seedA_phys`` / ``seedB_phys`` (initial guesses),
                 ``traj_win_phys`` (reco at each optimization step) — plus the raw 9-vectors,
                 both seeds' full trajectories, gradient norms, per-seed errors, and metadata
                 (n_hit, q_tot, energy_true, winning seed, seconds).

Designed to run inside the LUCiD container (see utils/submit_job.py); on a GPU
node with the layered JAX-CUDA env the per-event fit runs on the GPU.
"""
import argparse
import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # LUCiD/ on path
from analysis.paper.utils.pipeline import TrackingPipeline, load_config, REPO_ROOT   # noqa: E402


# scalars stored as HDF5 group attrs; everything else in the result dict is a dataset.
_ATTR_KEYS = {'ev', 'energy_true', 'which', 'best_iterA', 'best_iterB', 'best_iterF',
              'best_iter_win', 'n_hit', 'q_tot', 'seconds'}


def _git_commit():
    try:
        import subprocess
        return subprocess.check_output(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def _write_event(grp, rec):
    for k, v in rec.items():
        if k in _ATTR_KEYS:
            grp.attrs[k] = v
        else:
            grp.create_dataset(k, data=np.asarray(v), compression='gzip')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='study config JSON')
    ap.add_argument('--output', required=True, help='directory for <config-name>.h5')
    ap.add_argument('--events', default=None,
                    help='comma list of event indices (overrides event_start/n_events)')
    args = ap.parse_args()

    import h5py

    cfg = load_config(args.config)
    name = cfg.get('name') or Path(args.config).stem
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

    pipe = TrackingPipeline(cfg)

    with h5py.File(out_path, 'w') as h5:
        h5.attrs['config_json'] = json.dumps(cfg)
        h5.attrs['name'] = name
        h5.attrs['particle'] = cfg['particle']
        h5.attrs['study'] = cfg['study']
        h5.attrs['n_sensors'] = pipe.ND
        h5.attrs['n_rays'] = cfg['n_rays']
        h5.attrs['energy_nominal_MeV'] = cfg['energy_nominal_MeV']
        h5.attrs['n_events_requested'] = len(events)
        h5.attrs['phys_param_order'] = 'x,y,z,phi,theta,t0,E'
        h5.attrs['root_file'] = str(cfg['root_file'])
        h5.attrs['hostname'] = socket.gethostname()
        h5.attrs['lucid_commit'] = _git_commit()
        h5.attrs['started'] = datetime.now().isoformat()
        egrp = h5.create_group('events')

        ok = fail = 0
        for ev in events:
            try:
                rec = pipe.reconstruct(ev)
                _write_event(egrp.create_group(f'ev{ev:04d}'), rec)
                h5.flush()
                ok += 1
                fe = rec['fit_err']
                print(f"ev{ev:04d} WIN={'ABF'[rec['which']]} vtx{fe[0]:6.1f}cm dir{fe[1]:5.2f}deg "
                      f"dE{fe[2]:+7.1f} dt0{fe[3]:+5.2f} | nhit={rec['n_hit']} "
                      f"[{rec['seconds']:.0f}s]", flush=True)
            except Exception as e:
                fail += 1
                print(f"ev{ev:04d} FAILED: {type(e).__name__}: {e}", flush=True)

        h5.attrs['n_events_done'] = ok
        h5.attrs['n_events_failed'] = fail
        h5.attrs['finished'] = datetime.now().isoformat()

    print(f"\n=== done: {ok} ok, {fail} failed -> {out_path} ===", flush=True)
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
