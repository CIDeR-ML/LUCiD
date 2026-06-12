"""Aggregate campaign_recon/out/ev*.npz → seed/fit resolution + CONVERGENCE diagnostics."""
import os, glob, numpy as np

OUT = os.environ.get('OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))
files = sorted(glob.glob(os.path.join(OUT, 'ev*.npz')))
D = [np.load(f) for f in files]
seed = np.array([d['seed_err'] for d in D])      # (N,4): vtx cm, dir deg, dE MeV, dt0 ns
fit = np.array([d['fit_err'] for d in D])
best_iter = np.array([int(d['best_iter']) for d in D])
gn = [np.asarray(d['gnorm']) for d in D]
niters = max(len(g) for g in gn) - 1
N = len(files)


def stats(a, lab):
    print(f'\n{lab}  (N={N})')
    cols = ['vtx |Δ| (cm)', 'dir (deg)', 'dE (MeV)', 'dt0 (ns)']
    print(f'  {"":14s} {"median":>9s} {"mean":>9s} {"RMS":>9s} {"68%":>9s}')
    for i, c in enumerate(cols):
        v = a[:, i]
        print(f'  {c:14s} {np.median(v):9.2f} {v.mean():9.2f} {np.sqrt((v**2).mean()):9.2f} '
              f'{np.percentile(np.abs(v), 68):9.2f}')


print(f'=== Seeded recon campaign: {N} events (GEANT4 data, 2.5 ns TTS, {niters} iters) ===')
stats(seed, 'INITIAL GUESS (3-stage data-driven seeder)')
stats(fit, 'FULL FIT (seed -> Fisher-GN)')

# --- CONVERGENCE ---
fv = fit[:, 0]
buckets = [('converged (<20cm)', fv < 20), ('good (20-40cm)', (fv >= 20) & (fv < 40)),
           ('partial (40-100cm)', (fv >= 40) & (fv < 100)), ('wanderer (>100cm)', fv >= 100)]
print('\nCONVERGENCE — fit vertex capture:')
for lab, m in buckets:
    print(f'  {lab:20s} {int(m.sum()):3d}/{N}  ({100*m.mean():4.0f}%)')
improved = int(np.sum(fit[:, 0] < seed[:, 0]))
print(f'\n  fit improved vertex on {improved}/{N}; median vtx {np.median(seed[:,0]):.0f}cm (seed) '
      f'-> {np.median(fv):.1f}cm (fit)')
# is min-‖g‖ still at the very end? (=> not converged, needs more iters)
late = int(np.sum(best_iter > 0.9 * niters))
print(f'  best-‖g‖ iter: median {int(np.median(best_iter))}/{niters}; '
      f'{late}/{N} events had best-‖g‖ in the last 10% (would benefit from more iters)')
gred = np.array([g[bi] / max(g[0], 1e-9) for g, bi in zip(gn, best_iter)])
print(f'  ‖g‖ reduction (best/init): median {np.median(gred):.3f}  (smaller = better converged)')
