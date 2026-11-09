"""Stage 3 of the 2D loss-geometry figure: two optimization trajectories on the Neyman surface.

Built from the same ``landscape.Surface`` and recipe as the surface stage, so each iterate sits
on the plotted surface.

  - GD  (left  panel): fixed-step gradient descent  x <- x - eta grad L, run to convergence.
  - GN  (right panel): the damped Fisher step the joint fit takes (Marquardt lam*diag(F) +
                       Levenberg mu*median(diag F), plain solve), x <- x - A^-1 grad L.

Both converge to the realized minimum (the cyan X, displaced from truth by the independent-key
noise). Saves ``traj2d_neyman.npz``: starts(2,2), min(2), gd0, gd1, gn0, gn1 (each (T,2), metres).

    python traj_2d_neyman.py --data-dir DIR
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np                                                     # noqa: E402
import jax.numpy as jnp                                                # noqa: E402

from analysis.paper.utils.landscape import Surface, gauss_newton_step  # noqa: E402


def trajectories(recipe, landscape):
    """GD and GN paths from two opposite corners of the plotted box, to the realized minimum."""
    s = Surface(recipe)
    lam, mu = recipe['lam'], recipe['mu']
    t0 = time.time(); _ = s.loss_and_grad(jnp.array([s.tx, s.ty])); _ = s.fisher(jnp.array([s.tx, s.ty]))
    print(f'[traj] warm {time.time() - t0:.0f}s', flush=True)

    X, Y, Lgrid = landscape['X'], landscape['Y'], landscape['L']
    im, jm = np.unravel_index(np.argmin(Lgrid), Lgrid.shape); mx, my = float(X[im]), float(Y[jm])
    xlo, xhi, ylo, yhi = float(X.min()), float(X.max()), float(Y.min()), float(Y.max())

    # Two starts in diagonally opposite corners, top-left (low lambda_s, high lambda_a) and
    # bottom-right, so the two crawls reach the razor valley from opposite sides.
    def corner(fx, fy):
        return np.array([xlo + fx * (xhi - xlo), ylo + fy * (yhi - ylo)])
    starts = [corner(0.18, 0.88), corner(0.82, 0.12)]
    print(f'[traj] starts {[[round(float(p[0]), 1), round(float(p[1]), 1)] for p in starts]}', flush=True)

    # Plain fixed-step gradient descent. Its path is an integral curve of the -grad L field, so it
    # lies on the plotted streamlines. With eta ~ 0.9/lambda_max it clears the stiff lambda_s
    # direction in ~1 step, then crawls the soft lambda_a valley at rate (1-1/kappa), ~O(kappa)
    # steps: the ill-conditioning the Fisher preconditioner removes.
    def gd(x0, eta, maxit=500, xtol=5e-2, cap=20.0):
        x = np.asarray(x0, float); path = [x.copy()]
        for it in range(maxit):
            _, g = s.loss_and_grad(jnp.array(x)); g = np.asarray(g)
            step = eta * g; n = np.linalg.norm(step)
            if n > cap:
                step = step * (cap / n)                            # trust region for the far field
            x = x - step; path.append(x.copy())
            if np.linalg.norm(step) < xtol:
                break
            if it % 25 == 0:
                print(f'[traj]   gd it{it} x=({x[0]:.1f},{x[1]:.1f}) |step|={np.linalg.norm(step):.3g}', flush=True)
        return np.array(path)

    def gn(x0, maxit=40, cap=80.0, dtol=5e-2):
        x = np.asarray(x0, float); path = [x.copy()]
        for _ in range(maxit):
            _, g = s.loss_and_grad(jnp.array(x)); g = np.asarray(g)
            step = -np.asarray(gauss_newton_step(s.fisher(jnp.array(x)), g, lam, mu), float)
            n = np.linalg.norm(step)
            if n > cap:
                step = step * (cap / n)                            # keeps a far start in frame
            x = x + step; path.append(x.copy())
            if np.linalg.norm(step) < dtol:
                break
        return np.array(path)

    lam_max = float(np.linalg.eigvalsh(np.asarray(s.fisher(jnp.array([mx, my])))).max())   # stiff curvature at the min
    eta = 0.9 / lam_max
    print(f'[traj] lambda_max(F@min)={lam_max:.4g}  eta={eta:.4g}', flush=True)
    res = {}
    for i, p in enumerate(starts):
        t0 = time.time(); pg = gd(p, eta)
        print(f'[traj] gd{i}: {len(pg)} steps {time.time() - t0:.0f}s -> ({pg[-1, 0]:.1f},{pg[-1, 1]:.1f})', flush=True)
        t0 = time.time(); pn = gn(p)
        print(f'[traj] gn{i}: {len(pn)} steps {time.time() - t0:.0f}s -> ({pn[-1, 0]:.1f},{pn[-1, 1]:.1f})', flush=True)
        res[f'gd{i}'] = pg; res[f'gn{i}'] = pn
    return dict(starts=np.array(starts), min=np.array([mx, my]), recipe=json.dumps(recipe), **res)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    a = ap.parse_args(argv)
    ddir = Path(a.data_dir)
    landscape = np.load(ddir / 'landscape2d_neyman.npz')
    recipe = json.loads(str(landscape['recipe']))                  # the surface's own recipe
    np.savez(ddir / 'traj2d_neyman.npz', **trajectories(recipe, landscape))
    print(f'[traj] saved {ddir}/traj2d_neyman.npz', flush=True)


if __name__ == '__main__':
    main()
