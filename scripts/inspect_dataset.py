"""inspect_dataset — summarize a v3 dataset batch (sensor / hits / step / labl) at the terminal.

`lucid-run-job` writes four parallel HDF5 files per batch. This prints a text summary — event
count, per-event array shapes, charge/hit stats, and the truth/label table — without needing a
notebook or a graphics window (the visual counterpart is
`lucid/production/visualize_particle_events.py`). Full schema: docs/LUCID_DATASET.md.

Run:  python scripts/inspect_dataset.py --path OUTDIR            # a batch directory
      python scripts/inspect_dataset.py --path OUTDIR --event 0 # detail one event
"""
import argparse, os
import numpy as np
from lucid.sources.v3_reader import (list_events_v3, read_sensor_event_v3,
                                      read_hits_event_v3, read_step_event_v3, read_labl_event_v3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True, help='v3 batch directory (or a sensor .h5 path)')
    ap.add_argument('--event', type=int, default=None, help='inspect one event in detail')
    args = ap.parse_args()

    if not os.path.exists(args.path):
        raise SystemExit(f'path not found: {args.path}')

    events = list_events_v3(args.path)
    n = len(events)
    print(f'v3 dataset: {args.path}')
    print(f'  events: {n}')
    if n == 0:
        return

    if args.event is None:
        # summary over all events
        n_hit, q_tot = [], []
        for ev in events:
            s = read_sensor_event_v3(args.path, ev)
            q = np.asarray(s.get('charge', s.get('Q', [])))
            n_hit.append(int((q > 0).sum())); q_tot.append(float(q.sum()))
        n_hit, q_tot = np.array(n_hit), np.array(q_tot)
        print(f'  hit PMTs / event : mean {n_hit.mean():.0f}  min {n_hit.min()}  max {n_hit.max()}')
        print(f'  total charge     : mean {q_tot.mean():.0f}  min {q_tot.min():.0f}  max {q_tot.max():.0f}')
        print('  (use --event N for per-event detail)')
    else:
        ev = args.event
        print(f'\n--- event {ev} ---')
        for name, reader in [('sensor', read_sensor_event_v3), ('hits', read_hits_event_v3),
                             ('step', read_step_event_v3), ('labl', read_labl_event_v3)]:
            try:
                d = reader(args.path, ev)
                shapes = {k: np.asarray(v).shape for k, v in d.items()}
                print(f'  {name:8s}: {shapes}')
            except Exception as e:
                print(f'  {name:8s}: <unavailable: {type(e).__name__}>')


if __name__ == '__main__':
    main()
