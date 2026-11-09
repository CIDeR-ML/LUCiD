"""Stage 1 of the 2D loss-geometry figure: the Neyman surface on a grid, one shard of it.

Linear grid in metres over the 405 nm scattering and absorption lengths. Each shard evaluates its
round-robin slice one point at a time (one GPU each; a point fits an 11 GB card) and saves X, Y,
L, Gx, Gy and the per-point Fisher (Fxx, Fxy, Fyy) as ``landscape2d_neyman_part<shard>.npz``.

    python compute_2d_neyman.py --shard 0 --nshards 10 --data-dir DIR
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np                                                     # noqa: E402
import jax.numpy as jnp                                                # noqa: E402

from analysis.paper.utils import calibration as C                      # noqa: E402
from analysis.paper.utils.landscape import Surface                     # noqa: E402


def surface_shard(recipe, shard, nshards):
    """This shard's points of the grid; the others are left at zero, so shards sum to the union."""
    s = Surface(recipe)
    t0 = time.time()
    _ = s.loss_and_grad(jnp.array([s.tx, s.ty])); _ = s.fisher(jnp.array([s.tx, s.ty]))
    print(f'[neyman] shard {shard}/{nshards} warm {time.time() - t0:.0f}s', flush=True)

    n, hwf = recipe['ngrid'], recipe['hwf']
    X = np.linspace(s.tx * (1 - hwf), s.tx * (1 + hwf), n)
    Y = np.linspace(s.ty * (1 - hwf), s.ty * (1 + hwf), n)
    out = {k: np.zeros((n, n)) for k in ('L', 'Gx', 'Gy', 'Fxx', 'Fxy', 'Fyy')}
    IJ = [(i, j) for i in range(n) for j in range(n)]                 # index [i=Ls, j=La]
    mine = [idx for idx in range(len(IJ)) if idx % nshards == shard]
    t0 = time.time()
    for c, idx in enumerate(mine):
        i, j = IJ[idx]; v = jnp.array([X[i], Y[j]])
        lv, g = s.loss_and_grad(v); F = np.asarray(s.fisher(v))
        out['L'][i, j] = float(lv); out['Gx'][i, j] = float(g[0]); out['Gy'][i, j] = float(g[1])
        out['Fxx'][i, j] = F[0, 0]; out['Fxy'][i, j] = 0.5 * (F[0, 1] + F[1, 0]); out['Fyy'][i, j] = F[1, 1]
        if c % 50 == 0:
            print(f'[neyman] shard{shard} {c}/{len(mine)} ({time.time() - t0:.0f}s)', flush=True)
    return dict(X=X, Y=Y, truth_x=s.tx, truth_y=s.ty, wl=C.REP_WL, recipe=json.dumps(recipe), **out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, required=True)
    ap.add_argument('--data-dir', required=True)
    a = ap.parse_args(argv)
    res = surface_shard(C.LANDSCAPE, a.shard, a.nshards)
    out = Path(a.data_dir) / f'landscape2d_neyman_part{a.shard}.npz'
    np.savez(out, **res)
    print(f'[neyman] shard {a.shard} saved -> {out}', flush=True)


if __name__ == '__main__':
    main()
