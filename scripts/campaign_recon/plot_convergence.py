"""Convergence diagnostics for the keys×photons combos (winning-seed trajectories already saved):
median ‖g‖ vs iter, median vertex-error vs iter (from traj+truth), and best-iter histogram.
Answers: do the fits plateau before NITERS=250, or are they still improving at the budget edge?"""
import os, glob, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
COMBOS = [('A', 'nkeys4 / nph250k', 'C0'), ('B', 'nkeys1 / nph250k', 'C3'),
          ('C', 'nkeys1 / nph500k', 'C2')]


def load(name):
    fs = sorted(glob.glob(os.path.join(HERE, f'out_{name}', 'ev*.npz')))
    return [np.load(f) for f in fs]


fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
print(f'{"combo":18s} {"best-iter med":>13s} {"% peak in last 10%":>19s} '
      f'{"vtxErr@final":>13s} {"vtxErr@best":>12s} {"vtx drop last 20%":>17s}')
for name, lab, col in COMBOS:
    D = load(name)
    G = np.array([d['gnorm'] for d in D])                       # (N, 251) scaled ‖g‖ on winning seed
    # vertex error (cm) at every iteration from the winning-seed trajectory vs saved truth
    VE = np.array([np.linalg.norm(d['traj'][:, 1:4] - d['truth'][1:4], axis=1) * 100 for d in D])  # (N,251)
    bi = np.array([int(d['best_iter']) for d in D])
    it = np.arange(G.shape[1])
    # (1) ‖g‖ normalized to its own start, median + IQR
    Gn = G / G[:, :1]
    ax[0].plot(it, np.median(Gn, 0), col, lw=2, label=lab)
    ax[0].fill_between(it, np.percentile(Gn, 25, 0), np.percentile(Gn, 75, 0), color=col, alpha=.15)
    # (2) vertex error median + IQR
    ax[1].plot(it, np.median(VE, 0), col, lw=2, label=lab)
    ax[1].fill_between(it, np.percentile(VE, 25, 0), np.percentile(VE, 75, 0), color=col, alpha=.15)
    # (3) best-iter histogram
    ax[2].hist(bi, bins=np.linspace(0, 250, 26), histtype='step', color=col, lw=2, label=lab)
    # console summary
    last10 = (bi > 0.9 * (G.shape[1] - 1)).mean() * 100
    ve_fin = np.median(VE[:, -1]); ve_best = np.median([VE[i, bi[i]] for i in range(len(D))])
    drop = np.median((VE[:, -1] - VE[:, int(0.8 * (G.shape[1] - 1))]))   # vtx change over final 20%
    print(f'{name+" "+lab:18s} {int(np.median(bi)):>13d} {last10:>18.0f}% {ve_fin:>11.1f}cm '
          f'{ve_best:>10.1f}cm {drop:>15.2f}cm')
ax[0].set_yscale('log'); ax[0].set_xlabel('iteration'); ax[0].set_ylabel('‖g‖ / ‖g‖₀ (median, IQR)')
ax[0].set_title('Gradient-norm convergence'); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('vertex error (cm, median, IQR)')
ax[1].set_ylim(0, 90); ax[1].set_title('Vertex-error convergence (vs exact truth)')
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
ax[2].set_xlabel('best-‖g‖ iteration'); ax[2].set_ylabel('events')
ax[2].axvline(250, color='k', ls=':', lw=1); ax[2].set_title('Where ‖g‖ bottomed out (budget=250)')
ax[2].legend(fontsize=9); ax[2].grid(alpha=.3)
plt.tight_layout(); out = os.path.join(HERE, 'fig_convergence.png'); plt.savefig(out, dpi=120)
print('\nwrote', out)
