"""Readout study (post-hoc on saved trajectories, no GPU). The min-‖g‖ readout under-serves the
1-key combos because ‖g‖ is noisy there. Test smoothing + Polyak-tail variants to recover the
LOWEST-error iterate using only observable signals (no truth). Score each vs exact truth."""
import glob, numpy as np
from numpy.linalg import norm


def load(name):
    return [np.load(f) for f in sorted(glob.glob(f'out_{name}/ev*.npz'))]


def mov(a, w):                                   # centered moving average, edge-padded
    if w <= 1: return a
    k = np.ones(w) / w; return np.convolve(np.pad(a, w // 2, mode='edge'), k, 'valid')[:len(a)]


def ve(x9, th): return norm(x9[1:4] - th[1:4]) * 100


def readouts(d):
    tr, g = d['traj'], d['gnorm']; th = d['truth']; n = len(g)
    out = {}
    out['ming'] = tr[int(np.argmin(g))]                                  # current
    out['final'] = tr[-1]
    for N in (10, 20, 40):
        out[f'polyak{N}'] = tr[-N:].mean(0)                              # tail-average
    for w in (5, 11, 21):
        i = int(np.argmin(mov(g, w)))
        out[f'smin{w}'] = tr[i]                                          # smoothed-‖g‖ argmin
        lo, hi = max(0, i - w // 2), min(n, i + w // 2 + 1)
        out[f'sminP{w}'] = tr[lo:hi].mean(0)                             # Polyak around smoothed argmin
    return {k: ve(v, th) for k, v in out.items()}


METHODS = ['ming', 'final', 'polyak10', 'polyak20', 'polyak40',
           'smin5', 'sminP5', 'smin11', 'sminP11', 'smin21', 'sminP21']
print(f'{"method":9s} | ' + ' | '.join(f'{c:>18s}' for c in ['A nk4/250k', 'B nk1/250k', 'C nk1/500k']))
print(f'{"":9s} | ' + ' | '.join('   med  mean   RMS' for _ in range(3)))
data = {n: load(n) for n in 'ABC'}
res = {n: {m: np.array([readouts(d)[m] for d in data[n]]) for m in METHODS} for n in 'ABC'}
for m in METHODS:
    cells = []
    for n in 'ABC':
        a = res[n][m]; cells.append(f'{np.median(a):6.1f}{a.mean():6.1f}{np.sqrt((a**2).mean()):6.1f}')
    print(f'{m:9s} | ' + ' | '.join(cells))
# best method per combo by median, and a single method that's robust across all three
print('\nbest-by-median:', {n: min(METHODS, key=lambda m: np.median(res[n][m])) for n in 'ABC'})
# rank each method by mean over combos of (median / best-median-in-that-combo)
bestmed = {n: min(np.median(res[n][m]) for m in METHODS) for n in 'ABC'}
score = {m: np.mean([np.median(res[n][m]) / bestmed[n] for n in 'ABC']) for m in METHODS}
print('most robust (min mean relative-median):', min(score, key=score.get),
      '->', {m: round(score[m], 3) for m in sorted(score, key=score.get)[:4]})
