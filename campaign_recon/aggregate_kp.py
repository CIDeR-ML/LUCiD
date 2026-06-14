"""Aggregate the keys×photons study (out_A/out_B/out_C). All scored vs EXACT gun truth
(worker.py errs() uses the rand_tf exact (vtx,dir) truth). Reports per-combo resolution
(vtx/dir/E/t0), convergence buckets, which-seed-won, timing, and trajectory convergence
(best-‖g‖ iter, ‖g‖ reduction). Combos share event ids so deltas are paired."""
import os, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
COMBOS = [('A', 'nkeys4 / nph250k'), ('B', 'nkeys1 / nph250k'), ('C', 'nkeys1 / nph500k')]
COLS = ['vtx |Δ| (cm)', 'dir (deg)', 'dE (MeV)', 'dt0 (ns)']


def load(name):
    fs = sorted(glob.glob(os.path.join(HERE, f'out_{name}', 'ev*.npz')))
    evs = [int(os.path.basename(f)[2:5]) for f in fs]
    D = {ev: np.load(f) for ev, f in zip(evs, fs)}
    return D


def stat_block(arr, lab, n):
    print(f'\n{lab}  (N={n})')
    print(f'  {"":14s} {"median":>9s} {"mean":>9s} {"RMS":>9s} {"68%":>9s}')
    for i, c in enumerate(COLS):
        v = arr[:, i]
        print(f'  {c:14s} {np.median(v):9.2f} {v.mean():9.2f} {np.sqrt((v**2).mean()):9.2f} '
              f'{np.percentile(np.abs(v), 68):9.2f}')


data = {n: load(n) for n, _ in COMBOS}
# events present in ALL combos -> paired comparison
common = sorted(set.intersection(*[set(d.keys()) for d in data.values()]))
print(f'common events across all combos: {len(common)}')

summary = {}
for name, desc in COMBOS:
    D = data[name]
    fit = np.array([D[ev]['fit_err'] for ev in common])           # (N,4) vs exact truth
    which = np.array([int(D[ev]['which']) for ev in common])
    # trajectory convergence on the WINNING seed
    bi = np.array([int(D[ev]['best_iter']) for ev in common])
    gred = np.array([D[ev]['gnorm'][int(D[ev]['best_iter'])] / D[ev]['gnorm'][0] for ev in common])
    vtx = fit[:, 0]
    buckets = (int((vtx < 20).sum()), int(((vtx >= 20) & (vtx < 40)).sum()),
               int(((vtx >= 40) & (vtx < 100)).sum()), int((vtx >= 100).sum()))
    summary[name] = dict(fit=fit, which=which, bi=bi, gred=gred, buckets=buckets)
    print(f'\n========== combo {name}: {desc} ==========')
    stat_block(fit, 'FULL FIT (two-start, margin-selected)', len(common))
    print(f'  buckets <20/20-40/40-100/>100 cm: {buckets[0]}/{buckets[1]}/{buckets[2]}/{buckets[3]}'
          f'   time-seed won: {int((which == 1).sum())}/{len(common)}')
    print(f'  best-‖g‖ iter median {int(np.median(bi))}/250   ‖g‖ reduction median ×{np.median(gred):.4f}')

# paired side-by-side medians + A-vs-X deltas
print('\n\n=== SIDE BY SIDE (median over common events; Δ vs combo A, paired) ===')
print(f'{"metric":16s} ' + ' '.join(f'{n:>22s}' for n, _ in COMBOS))
A = summary['A']['fit']
for i, c in enumerate(COLS):
    cells = []
    for name, _ in COMBOS:
        v = summary[name]['fit'][:, i]
        med = np.median(v)
        if name == 'A':
            cells.append(f'{med:8.2f}            ')
        else:
            dmed = np.median(summary[name]['fit'][:, i] - A[:, i])  # paired delta
            cells.append(f'{med:8.2f} (Δpair {dmed:+6.2f})')
    print(f'{c:16s} ' + ' '.join(f'{x:>22s}' for x in cells))
# vtx RMS row (the tail-sensitive metric)
print(f'{"vtx RMS (cm)":16s} ' + ' '.join(
    f'{np.sqrt((summary[n]["fit"][:,0]**2).mean()):8.2f}            '[:22].rjust(22) for n, _ in COMBOS))
