"""Make recon plots, one figure per particle:
  (1) distributions across events  (vtx, dir, dE%, dt0, q-ratio)  -- from the optb_*.log files
  (2) optimization process         (vtx-err / dE / scaled-gnorm vs iteration) -- from optb_*.npz

Usage: python plot_recon.py [muon electron]
"""
import os, sys, re, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'plots'); os.makedirs(OUT, exist_ok=True)
PARTICLES = sys.argv[1:] or ['muon', 'electron']
ECOL = {500: 'tab:blue', 1000: 'tab:green', 1500: 'tab:red'}

EVLINE = re.compile(
    r'ev(\d+)\s+n_hit\s*\d+\s+q\s*[\d.]+\s+mq\s*[\d.]+\s+R\s*([\d.]+)\s*\|\s*'
    r'vtx\s*([\d.]+)cm\s+dir\s*([\d.]+)\s+dE\s*([+-]?\d+)\(\s*([+-]?[\d.]+)%\)\s+dt0\s*([+-]?[\d.]+)')


def parse_log(path):
    rows = []
    for line in open(path):
        m = EVLINE.search(line)
        if m:
            _, R, vtx, d, dE, dEp, dt0 = m.groups()
            rows.append((float(R), float(vtx), float(d), float(dEp), float(dt0)))
    return np.array(rows) if rows else np.empty((0, 5))


def energies_for(particle):
    out = {}
    for p in sorted(glob.glob(os.path.join(HERE, f'optb_{particle[:4]}*.log'))):
        m = re.search(r'(\d+)\.log$', p) or re.search(rf'{particle[:4]}(\d+)', os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


# ---------- (1) DISTRIBUTIONS ----------
for particle in PARTICLES:
    logs = energies_for(particle)
    if not logs:
        print(f'no logs for {particle}'); continue
    metrics = [('q_tot ratio (data/model)', 0, 14), ('vertex |Δ| (cm)', 1, 14),
               ('direction err (deg)', 2, 14), ('energy bias dE (%)', 3, 14),
               ('t0 bias (ns)', 4, 14)]
    data = {E: parse_log(logs[E]) for E in sorted(logs) if parse_log(logs[E]).size}
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
    fig.suptitle(f'{particle}: recon distributions across events (band-consistent, no norm)', fontsize=13)
    for ax, (name, col, nb) in zip(axes, metrics):
        pooled = np.concatenate([d[:, col] for d in data.values()])
        lo, hi = float(pooled.min()), float(pooled.max())
        if hi - lo < 1e-9: lo, hi = lo - 0.5, hi + 0.5
        pad = 0.04 * (hi - lo); edges = np.linspace(lo - pad, hi + pad, nb + 1)   # SHARED edges (aligned)
        for E, d in data.items():
            vals = d[:, col]
            ax.hist(vals, bins=edges, histtype='stepfilled', alpha=0.30, color=ECOL.get(E, 'gray'))
            ax.hist(vals, bins=edges, histtype='step', lw=1.8, color=ECOL.get(E, 'gray'),
                    label=f'{E} MeV (n={len(vals)}, med {np.median(vals):.2f})')
        ax.set_xlabel(name); ax.set_ylabel('events')
        ax.margins(x=0)
        if name.startswith('q_tot'): ax.axvline(1.0, color='k', ls=':', lw=1)
        if 'bias' in name: ax.axvline(0.0, color='k', ls=':', lw=1)
        ax.legend(fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fp = os.path.join(OUT, f'dist_{particle}.png'); fig.savefig(fp, dpi=110); plt.close(fig)
    print(f'wrote {fp}')

# ---------- (2) OPTIMIZATION PROCESS ----------
for particle in PARTICLES:
    npzs = sorted(glob.glob(os.path.join(OUT, f'optb_{particle}_*.npz')))
    if not npzs:
        print(f'no npz trajectories for {particle} (run opt_band_multi.py with OUT=plots)'); continue
    panels = [('vertex |Δ| vs iter (cm)', 'err', 0, 'log'),
              ('|dE| vs iter (MeV)', 'err', 2, 'log'),
              ('scaled ‖g‖ vs iter', 'gnorm', None, 'log')]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f'{particle}: optimization process (truth-seed Fisher-GN, per event)', fontsize=13)
    for ax, (name, key, col, ys) in zip(axes, panels):
        for fp in npzs:
            z = np.load(fp, allow_pickle=True); E = int(z['energy'])
            arr = z['traj_err'] if key == 'err' else z['traj_gnorm']
            for i in range(arr.shape[0]):
                y = np.abs(arr[i, :, col]) if key == 'err' else arr[i]
                ax.plot(y, color=ECOL.get(E, 'gray'), alpha=0.35, lw=0.8)
            ax.plot([], [], color=ECOL.get(E, 'gray'), label=f'{E} MeV')
        ax.set_xlabel('iteration'); ax.set_ylabel(name); ax.set_yscale(ys); ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fp = os.path.join(OUT, f'process_{particle}.png'); fig.savefig(fp, dpi=110); plt.close(fig)
    print(f'wrote {fp}')
print('PLOTS DONE')
