"""Plot the keys×photons study: vtx-error CDF + per-combo scatter, AD Fisher lr=1."""
import os, glob, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
COMBOS = [('A', 'nkeys4 / nph250k  (69 min)', 'C0'),
          ('B', 'nkeys1 / nph250k  (24 min)', 'C3'),
          ('C', 'nkeys1 / nph500k  (39 min)', 'C2')]


def load(name):
    fs = sorted(glob.glob(os.path.join(HERE, f'out_{name}', 'ev*.npz')))
    evs = [int(os.path.basename(f)[2:5]) for f in fs]
    return {ev: np.load(f) for ev, f in zip(evs, fs)}


data = {n: load(n) for n, _, _ in COMBOS}
common = sorted(set.intersection(*[set(d.keys()) for d in data.values()]))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
# (1) vtx-error CDF
for name, lab, col in COMBOS:
    v = np.sort(np.array([data[name][ev]['fit_err'][0] for ev in common]))
    ax[0].plot(v, np.linspace(0, 1, len(v)), col, lw=2, label=lab)
ax[0].axvline(20, color='k', ls=':', lw=1, alpha=.5)
ax[0].set_xlim(0, 80); ax[0].set_xlabel('vertex error |Δ| (cm)'); ax[0].set_ylabel('cumulative fraction')
ax[0].set_title('Vertex-error CDF (100 events, AD Fisher lr=1)'); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
# (2) paired vtx: A vs B and A vs C
vA = np.array([data['A'][ev]['fit_err'][0] for ev in common])
for name, lab, col in COMBOS[1:]:
    vX = np.array([data[name][ev]['fit_err'][0] for ev in common])
    ax[1].scatter(vA, vX, s=14, c=col, alpha=.6, label=lab.split('(')[0])
m = 90; ax[1].plot([0, m], [0, m], 'k--', lw=1, alpha=.5)
ax[1].set_xlim(0, 45); ax[1].set_ylim(0, m); ax[1].set_xlabel('combo A vtx err (cm)')
ax[1].set_ylabel('combo B / C vtx err (cm)'); ax[1].set_title('Paired per-event vertex error vs A')
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
# (3) median bars: vtx + dir
labels = [n for n, _, _ in COMBOS]; x = np.arange(3)
vmed = [float(np.median([data[n][ev]['fit_err'][0] for ev in common])) for n in labels]
dmed = [float(np.median([data[n][ev]['fit_err'][1] for ev in common])) for n in labels]
ax2 = ax[2]; b1 = ax2.bar(x - .2, vmed, .4, color='C0', label='vtx median (cm)')
ax2.set_ylabel('vtx median (cm)', color='C0'); ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_title('Median vtx / dir per combo'); ax2.bar_label(b1, fmt='%.1f', fontsize=9)
ax3 = ax2.twinx(); b2 = ax3.bar(x + .2, dmed, .4, color='C1', label='dir median (deg)')
ax3.set_ylabel('dir median (deg)', color='C1'); ax3.bar_label(b2, fmt='%.2f', fontsize=9)
plt.tight_layout(); out = os.path.join(HERE, 'fig_keysphotons.png'); plt.savefig(out, dpi=120)
print('wrote', out)
