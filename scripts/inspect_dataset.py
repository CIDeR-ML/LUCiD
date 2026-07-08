"""inspect_dataset — summarize a dataset batch (sensor / hits / step / labl) at the terminal.

`lucid-run-job` writes four parallel HDF5 files per batch under
`<output_dir>/{sensor,hits,step,labl}/wc_*_<F:04d>.h5`. This prints a text summary — event
count, per-event array shapes, charge/hit stats — without needing a notebook or a graphics
window. Full schema: docs/LUCID_DATASET.md.

Run:  python scripts/inspect_dataset.py --path OUTDIR              # batch 0 of a dataset dir
      python scripts/inspect_dataset.py --path OUTDIR --batch 2    # another file_index
      python scripts/inspect_dataset.py --path OUTDIR --event 0    # detail one event
"""
import argparse, os
import numpy as np
from lucid.production.verify_output import batch_paths
from lucid.sources.reader import (list_events, read_sensor_event, read_hits_event,
                                  read_step_event, read_labl_event)


def _shape_of(v):
    if isinstance(v, dict):
        return '{' + ', '.join(f'{k}:{_shape_of(x)}' for k, x in sorted(v.items())) + '}'
    a = np.asarray(v)
    return str(a.shape) if a.shape else repr(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True,
                    help='dataset directory (with sensor/hits/step/labl subdirs)')
    ap.add_argument('--batch', type=int, default=0, help='batch file_index (default 0)')
    ap.add_argument('--event', type=int, default=None, help='inspect one event in detail')
    args = ap.parse_args()

    if not os.path.isdir(args.path):
        raise SystemExit(f'dataset directory not found: {args.path}')
    paths = batch_paths(args.path, args.batch)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit('missing batch files:\n  ' + '\n  '.join(missing))

    source_idx = list_events(paths['sensor'])
    n = len(source_idx)
    print(f'dataset: {args.path} (batch {args.batch})')
    print(f'  events: {n}')
    if n == 0:
        return

    if args.event is None:
        # summary over all events
        n_hit, q_tot = [], []
        for ev in range(n):
            s = read_sensor_event(paths['sensor'], ev)
            q = np.asarray(s.get('charge', s.get('Q', [])))
            n_hit.append(int((q > 0).sum())); q_tot.append(float(q.sum()))
        n_hit, q_tot = np.array(n_hit), np.array(q_tot)
        print(f'  hit PMTs / event : mean {n_hit.mean():.0f}  min {n_hit.min()}  max {n_hit.max()}')
        print(f'  total charge     : mean {q_tot.mean():.0f}  min {q_tot.min():.0f}  max {q_tot.max():.0f}')
        print('  (use --event N for per-event detail)')
    else:
        ev = args.event
        print(f'\n--- event {ev} (source_event_idx {source_idx[ev]}) ---')
        for name, reader in [('sensor', read_sensor_event), ('hits', read_hits_event),
                             ('step', read_step_event), ('labl', read_labl_event)]:
            try:
                d = reader(paths[name], ev)
                shapes = {k: _shape_of(v) for k, v in d.items()}
                print(f'  {name:8s}: {shapes}')
            except Exception as e:
                print(f'  {name:8s}: <unavailable: {type(e).__name__}: {e}>')


if __name__ == '__main__':
    main()
