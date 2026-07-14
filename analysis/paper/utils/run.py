"""Run reconstructions for a figure — locally (in-process) or on S3DF (SLURM).

``run_local`` reconstructs a handful of events in the current process (no SLURM,
no site assumptions) and writes the same ``<name>.h5`` layout the plotting code
expects. ``submit_s3df`` shells out to ``analysis/paper/submit_job.py`` for
the full production. Both consume the ordinary study-config dicts from
``studies.py``.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]      # LUCiD/
sys.path.insert(0, str(REPO_ROOT))

_ATTR_KEYS = {'ev', 'energy_true', 'which', 'best_iterA', 'best_iterB', 'best_iterF',
              'best_iter_win', 'n_hit', 'q_tot', 'seconds'}


def _write_event(grp, rec):
    for k, v in rec.items():
        (grp.attrs.__setitem__ if k in _ATTR_KEYS
         else (lambda kk, vv: grp.create_dataset(kk, data=np.asarray(vv), compression='gzip')))(k, v)


def run_local(config: dict, out_dir, events=None, verbose=True):
    """Reconstruct ``config`` in-process; write ``<out_dir>/<name>.h5``. Returns the path.

    ``events`` overrides the config's event range (default: first ``n_events``).
    """
    import h5py
    from analysis.paper.utils.pipeline import TrackingPipeline  # local import: heavy (JAX)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    name = config['name']
    out_path = out_dir / f'{name}.h5'
    evs = events if events is not None else list(
        range(config['event_start'], config['event_start'] + config['n_events']))

    if verbose:
        print(f"[local] {name}: {len(evs)} events, n_rays={config['n_rays']}", flush=True)
    pipe = TrackingPipeline(config, verbose=verbose)
    with h5py.File(out_path, 'w') as h5:
        h5.attrs['config_json'] = json.dumps(config)
        h5.attrs['name'] = name
        egrp = h5.create_group('events')
        for ev in evs:
            rec = pipe.reconstruct(ev)
            _write_event(egrp.create_group(f'ev{ev:04d}'), rec)
            h5.flush()
            if verbose:
                fe = rec['fit_err']
                print(f"  ev{ev:04d} vtx {fe[0]:5.1f}cm dir {fe[1]:4.2f}deg "
                      f"dE {fe[2]:+6.1f} dt0 {fe[3]:+5.2f}", flush=True)
    return out_path


def submit_s3df(config: dict, out_dir, config_dir, partition='ampere', time='20:00:00',
                job_name=None, submit=True):
    """Write ``config`` to disk and hand it to submit_job.py for a SLURM GPU run."""
    config_dir = Path(config_dir); config_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = config_dir / f"{config['name']}.json"
    cfg_path.write_text(json.dumps(config, indent=2))
    cmd = [sys.executable, str(REPO_ROOT / 'analysis/paper/submit_job.py'),
           '--config', str(cfg_path), '--output', str(out_dir),
           '--job-name', job_name or config['name'],
           '--partition', partition, '--time', time]
    if submit:
        cmd.append('--submit')
    print('[s3df] ' + ' '.join(cmd), flush=True)
    return subprocess.run(cmd, check=True)
