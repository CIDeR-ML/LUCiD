"""Final keys×photons×iters figure: vtx CDF, energy-bias vs photon budget (the key result:
single-key bias grows with photons, 4-key averaging cancels it), and paired vtx vs A."""
import glob, numpy as np
from numpy.linalg import norm
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt


def load(n): return {int(f[-7:-4]): np.load(f) for f in sorted(glob.glob(f'out_{n}/ev*.npz'))}
def mg(d): return d['traj'][int(np.argmin(d['gnorm']))]                  # min-‖g‖ readout
def vtx(d): return norm(mg(d)[1:4] - d['truth'][1:4]) * 100
def de(d): return mg(d)[0] - d['truth'][0]


A, B, C, D = load('A'), load('B5'), load('C5'), load('D5')
ev = sorted(set(A) & set(B) & set(C) & set(D))                          # common 50 events
SETS = [('A 4x250k (1M tot)', A, 'C0'), ('B5 1x250k', B, 'C3'),
        ('C5 1x500k', C, 'C2'), ('D5 1x1M', D, 'C1')]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
# (1) vtx CDF on common events
for lab, X, col in SETS:
    v = np.sort([vtx(X[e]) for e in ev])
    ax[0].plot(v, np.linspace(0, 1, len(v)), col, lw=2, label=f'{lab}  (med {np.median(v):.1f})')
ax[0].axvline(20, color='k', ls=':', lw=1, alpha=.5); ax[0].set_xlim(0, 50)
ax[0].set_xlabel('vertex error (cm)'); ax[0].set_ylabel('cumulative fraction')
ax[0].set_title('Vertex-error CDF (50 common events, min-‖g‖)'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
# (2) THE KEY RESULT: energy bias vs total photon budget, single-key trend + the 4-key point
sk = [('B5', B, 0.25), ('C5', C, 0.5), ('D5', D, 1.0)]                  # single-key, Mphotons
xb = [m for _, _, m in sk]; yb = [np.median([de(X[e]) for e in ev]) for _, X, _ in sk]
eb = [np.std([de(X[e]) for e in ev]) / np.sqrt(len(ev)) for _, X, _ in sk]
ax[1].errorbar(xb, yb, yerr=eb, fmt='o-', color='C3', lw=2, ms=8, label='1 key (single sample)')
aY = np.median([de(A[e]) for e in ev]); aE = np.std([de(A[e]) for e in ev]) / np.sqrt(len(ev))
ax[1].errorbar([1.0], [aY], yerr=[aE], fmt='s', color='C0', ms=12, label='4 keys × 250k (1M tot)')
ax[1].axhline(0, color='k', lw=1, alpha=.5)
ax[1].set_xlabel('total predictor photons (M)'); ax[1].set_ylabel('energy bias median (MeV)')
ax[1].set_title('Energy bias: 1-key grows with photons,\n4-key averaging cancels it')
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
# (3) paired vtx vs A
vA = np.array([vtx(A[e]) for e in ev])
for lab, X, col in SETS[1:]:
    ax[2].scatter(vA, [vtx(X[e]) for e in ev], s=16, c=col, alpha=.6, label=lab.split('(')[0])
m = 60; ax[2].plot([0, m], [0, m], 'k--', lw=1, alpha=.5); ax[2].set_xlim(0, 35); ax[2].set_ylim(0, m)
ax[2].set_xlabel('A vtx err (cm)'); ax[2].set_ylabel('B5 / C5 / D5 vtx err (cm)')
ax[2].set_title('Paired per-event vertex vs A'); ax[2].legend(fontsize=9); ax[2].grid(alpha=.3)
plt.tight_layout(); out = 'fig_keys_vs_photons.png'; plt.savefig(out, dpi=120); print('wrote', out)
