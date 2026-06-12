"""Aggregate campaign_recon/out/ev*.npz → seed-vs-truth and fit-vs-truth distributions."""
import os, glob, numpy as np

OUT = os.environ.get('OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))
files = sorted(glob.glob(os.path.join(OUT, 'ev*.npz')))
seed = np.array([np.load(f)['seed_err'] for f in files])      # (N,4): vtx cm, dir deg, dE MeV, dt0 ns
fit = np.array([np.load(f)['fit_err'] for f in files])
N = len(files)


def stats(a, lab):
    print(f'\n{lab}  (N={N})')
    cols = ['vtx |Δ| (cm)', 'dir (deg)', 'dE (MeV)', 'dt0 (ns)']
    print(f'  {"":14s} {"median":>9s} {"mean":>9s} {"std":>9s} {"68%":>9s} {"|>2σ| frac":>10s}')
    for i, c in enumerate(cols):
        v = a[:, i]; av = np.abs(v) if i in (2, 3) else v   # vtx/dir already non-neg
        med = np.median(av); p68 = np.percentile(np.abs(v), 68)
        wand = float(np.mean(np.abs(v - np.median(v)) > 2 * np.std(v))) if N > 3 else 0.0
        print(f'  {c:14s} {med:9.2f} {v.mean():9.2f} {v.std():9.2f} {p68:9.2f} {wand:10.2f}')


print(f'=== Seeded recon campaign: {N} events (GEANT4 data, 2.5 ns TTS) ===')
stats(seed, 'INITIAL GUESS (3-stage seeder, data-driven)')
stats(fit, 'FULL FIT (seed -> Fisher-GN fit_track)')
# how many events the fit IMPROVED the vertex vs the seed
better = int(np.sum(fit[:, 0] < seed[:, 0]))
print(f'\nfit improved vertex on {better}/{N} events; '
      f'median vtx seed {np.median(seed[:,0]):.0f}cm -> fit {np.median(fit[:,0]):.1f}cm')
print(f'wanderers (fit vtx > 50 cm): {int(np.sum(fit[:,0]>50))}/{N}')
