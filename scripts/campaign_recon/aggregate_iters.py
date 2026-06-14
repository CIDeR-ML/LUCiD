"""Aggregate the 500-iter 1-key re-runs (out_B5/C5/D5) + the converged A baseline (out_A, 250-iter).
Reports each combo under the chosen Polyak10 readout (plus ming/final for reference), and a
convergence re-check (vtx move over the final 20% of iters) to confirm 500 iters is enough."""
import glob, numpy as np
from numpy.linalg import norm
import sys; sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_unification')

COMBOS = [('A', 'nkeys4 / nph250k  (250 it)'), ('B5', 'nkeys1 / nph250k  (500 it)'),
          ('C5', 'nkeys1 / nph500k  (500 it)'), ('D5', 'nkeys1 / nph1M    (500 it)')]
COLS = ['vtx (cm)', 'dir (deg)', 'dE (MeV)', 'dt0 (ns)']


def load(name):
    return [np.load(f) for f in sorted(glob.glob(f'out_{name}/ev*.npz'))]


def vec9_dir(x):
    p, a = np.arctan2(x[4], x[5]), np.arctan2(x[6], x[7])
    return np.array([np.sin(p) * np.cos(a), np.sin(p) * np.sin(a), np.cos(p)])


def err4(x9, th):                                              # vtx cm, dir deg, dE, dt0 vs exact truth
    dv = norm(x9[1:4] - th[1:4]) * 100
    dd = np.degrees(np.arccos(np.clip(vec9_dir(x9) @ vec9_dir(th), -1, 1)))
    return np.array([dv, dd, x9[0] - th[0], x9[8] - th[8]])


def readout(d, mode):
    tr = d['traj']
    if mode == 'ming': return tr[int(np.argmin(d['gnorm']))]
    if mode == 'final': return tr[-1]
    if mode == 'polyak10': return tr[-10:].mean(0)


def block(name, lab):
    D = load(name)
    print(f'\n===== {name}: {lab}   (N={len(D)}) =====')
    for mode in ['ming', 'final', 'polyak10']:
        E = np.array([err4(readout(d, mode), d['truth']) for d in D])
        bad = int(np.isnan(E).any(1).sum())            # diverged/NaN-tail events this readout grabbed
        E = E[~np.isnan(E).any(1)]                      # NaN-filter for the stats
        wnd = int((E[:, 0] > 100).sum())
        s = '  '.join(f'{c.split()[0]} {np.median(E[:,i]):6.1f}/{np.sqrt((E[:,i]**2).mean()):6.1f}'
                      for i, c in enumerate(COLS))      # median/RMS per metric
        print(f'  {mode:9s} (med/RMS): {s}   wand>100cm={wnd}  NaN={bad}')
    # convergence: vtx (from polyak-free traj) move over final 20% of iters
    VE = np.array([norm(d['traj'][:, 1:4] - d['truth'][1:4], axis=1) * 100 for d in D])
    n = VE.shape[1] - 1; drop = np.median(VE[:, -1] - VE[:, int(0.8 * n)])
    bi = np.array([int(d['best_iter']) for d in D])
    print(f'  CONVERGENCE: vtx move over final 20% (iter {int(0.8*n)}->{n}) median {drop:+.2f} cm   '
          f'best-‖g‖ iter median {int(np.median(bi))}/{n}')


for name, lab in COMBOS:
    if glob.glob(f'out_{name}/ev*.npz'): block(name, lab)
print('\n(med/RMS in cm,deg,MeV,ns. polyak10 = mean of last 10 iterates = the "min-g after Polyak" readout.)')
