"""Aggregate the TWO-START campaign (out_ms100/) and compare to the OLD single-start (out/).
Reports convergence buckets, which-seed-won stats, and old-vs-new resolution side by side."""
import os, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.environ.get('OUT', os.path.join(HERE, 'out_ms100'))
MARGIN = float(os.environ.get('MARGIN', '0.01'))                 # pick B only if it beats A by >1% loss
nf = sorted(glob.glob(os.path.join(NEW, 'ev*.npz')))
D = [np.load(f) for f in nf]; N = len(D)
# margin-gated arbitration (recomputed from saved per-seed losses + errors; prefer A=charge seed)
L = np.array([d['losses'] for d in D]); relgap = (L[:, 0] - L[:, 1]) / np.abs(L[:, 0])
which = (relgap > MARGIN).astype(int)
fit = np.array([d['fitB_err'] if w else d['fitA_err'] for d, w in zip(D, which)])
fa = np.array([d['fitA_err'][0] for d in D]); fb = np.array([d['fitB_err'][0] for d in D])
print(f'(arbitration: 1% loss margin, prefer charge seed; MARGIN={MARGIN})')
# old single-start baseline, matched by event id
ev_ids = [int(os.path.basename(f)[2:5]) for f in nf]
old = []
for ev in ev_ids:
    of = os.path.join(HERE, 'out', f'ev{ev:03d}.npz')
    old.append(np.load(of)['fit_err'][0] if os.path.exists(of) else np.nan)
old = np.array(old)


def stats(a, lab):
    cols = ['vtx |Δ| (cm)', 'dir (deg)', 'dE (MeV)', 'dt0 (ns)']
    print(f'\n{lab}  (N={N})')
    print(f'  {"":14s} {"median":>9s} {"mean":>9s} {"RMS":>9s} {"68%":>9s}')
    for i, c in enumerate(cols):
        v = a[:, i]
        print(f'  {c:14s} {np.median(v):9.2f} {v.mean():9.2f} {np.sqrt((v**2).mean()):9.2f} '
              f'{np.percentile(np.abs(v), 68):9.2f}')


print(f'=== TWO-START recon: {N} events (GEANT4 data, 2.5 ns TTS) ===')
stats(fit, 'FULL FIT (two-start, lower-loss basin)')
fv = fit[:, 0]
buckets = [('converged (<20cm)', fv < 20), ('good (20-40cm)', (fv >= 20) & (fv < 40)),
           ('partial (40-100cm)', (fv >= 40) & (fv < 100)), ('wanderer (>100cm)', fv >= 100)]
print('\nCONVERGENCE — fit vertex capture:')
for lab, m in buckets:
    print(f'  {lab:20s} {int(m.sum()):3d}/{N}  ({100*m.mean():4.0f}%)')
print(f'\nSEED ARBITRATION: time-seed (B) won {int((which==1).sum())}/{N}; charge-seed (A) won {int((which==0).sum())}/{N}')
print(f'  when B won: median fit {np.median(fv[which==1]):.1f}cm  | when A won: median fit {np.median(fv[which==0]):.1f}cm')

# vs old single-start
print('\n=== OLD single-start vs TWO-start (matched events) ===')
m = ~np.isnan(old)
print(f'  vtx median: OLD {np.median(old[m]):.1f}cm -> TWO {np.median(fv[m]):.1f}cm   '
      f'| mean: OLD {old[m].mean():.1f} -> TWO {fv[m].mean():.1f}  | RMS: OLD {np.sqrt((old[m]**2).mean()):.1f} -> TWO {np.sqrt((fv[m]**2).mean()):.1f}')
oldtail = int(np.sum(old[m] >= 40)); newtail = int(np.sum(fv[m] >= 40))
print(f'  events >=40cm: OLD {oldtail} -> TWO {newtail}   | wanderers >100cm: OLD {int(np.sum(old[m]>=100))} -> TWO {int(np.sum(fv[m]>=100))}')
better = int(np.sum(fv[m] < old[m] - 5)); worse = int(np.sum(fv[m] > old[m] + 5))
print(f'  two-start better by >5cm on {better}/{int(m.sum())}; worse by >5cm on {worse}/{int(m.sum())}')
