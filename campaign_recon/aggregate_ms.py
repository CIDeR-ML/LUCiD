"""Aggregate the TWO-START campaign (out_ms100/) and compare to the OLD single-start (out/).
Reports convergence buckets, which-seed-won stats, and old-vs-new resolution side by side."""
import os, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.environ.get('OUT', os.path.join(HERE, 'out_ms100'))
from truth_exact import exact_truth, fit_err_exact
MARGIN = float(os.environ.get('MARGIN', '0.01'))                 # pick B only if it beats A by >1% loss
nf = sorted(glob.glob(os.path.join(NEW, 'ev*.npz')))
D = [np.load(f) for f in nf]; N = len(D)
ev_ids = [int(os.path.basename(f)[2:5]) for f in nf]
ET = exact_truth(ev_ids)                                          # exact gun truth per event
# per-seed errors vs EXACT truth, recomputed from the saved 9-vecs (truth-source-independent)
errA = np.array([fit_err_exact(d['fitA'], *ET[ev]) for d, ev in zip(D, ev_ids)])
errB = np.array([fit_err_exact(d['fitB'], *ET[ev]) for d, ev in zip(D, ev_ids)])
L = np.array([d['losses'] for d in D]); relgap = (L[:, 0] - L[:, 1]) / np.abs(L[:, 0])
which = (relgap > MARGIN).astype(int)                             # margin-gated, prefer A=charge seed
fit = np.where(which[:, None] == 1, errB, errA)                  # (N,4) margin-selected error vs exact
fa = errA[:, 0]; fb = errB[:, 0]
print(f'(arbitration: {MARGIN*100:.0f}% loss margin, prefer charge seed; scored vs EXACT gun truth)')
# old single-start baseline, matched by event id — ALSO re-scored vs exact (its 'fit' 9-vec)
old = []
for ev in ev_ids:
    of = os.path.join(HERE, 'out', f'ev{ev:03d}.npz')
    old.append(fit_err_exact(np.load(of)['fit'], *ET[ev])[0] if os.path.exists(of) else np.nan)
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
