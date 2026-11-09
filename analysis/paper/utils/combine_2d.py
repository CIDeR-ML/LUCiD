"""Stage 2: combine the per-shard partial landscapes into the full npz. Each shard filled its
disjoint round-robin points and left zeros elsewhere, so summing across shards reconstructs the
full grid exactly.

    python combine_2d.py --data-dir DIR
"""
import argparse
from pathlib import Path

import numpy as np

_KEYS = ('L', 'Gx', 'Gy', 'Fxx', 'Fxy', 'Fyy')


def combine(parts):
    d0 = np.load(parts[0])
    acc = {k: np.zeros_like(d0[k]) for k in _KEYS}
    for p in parts:
        d = np.load(p)
        if str(d['recipe']) != str(d0['recipe']):
            raise ValueError(f'{p} was computed with a different recipe than {parts[0]}')
        for k in _KEYS:
            acc[k] += d[k]                                         # disjoint points -> sum = union
    return dict(X=d0['X'], Y=d0['Y'], truth_x=d0['truth_x'], truth_y=d0['truth_y'],
                wl=d0['wl'], recipe=d0['recipe'], **acc)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    a = ap.parse_args(argv)
    ddir = Path(a.data_dir)
    parts = sorted(ddir.glob('landscape2d_neyman_part*.npz'))
    if not parts:
        raise SystemExit(f'no shard partials in {ddir}')
    res = combine(parts)
    np.savez(ddir / 'landscape2d_neyman.npz', **res)
    print(f'[combine] {len(parts)} shards -> {ddir}/landscape2d_neyman.npz   '
          f'L range [{res["L"].min():.3g}, {res["L"].max():.3g}]')


if __name__ == '__main__':
    main()
