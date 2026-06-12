"""Compare OLD single-start (charge seed, out/) vs NEW two-start (out_ms/) on the same events.
Shows per-seed fits, the loss-arbitrated winner, and the net vertex resolution."""
import os, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))


def cwall(d):
    vtx = d['truth'][1:4]; rad = np.array([vtx[0], vtx[1], 0.]); rad /= np.linalg.norm(rad) + 1e-9
    return float(d['tdir'] @ rad)


new = sorted(glob.glob(os.path.join(HERE, 'out_ms', 'ev*.npz')))
print(f'{"ev":>3} {"cWall":>6} | {"OLD":>6} | {"fitA":>6} {"fitB":>6} {"win":>3} {"TWO":>6} | {"dir":>5} {"E":>6} {"t0":>6}')
print('-' * 76)
rows = []
for f in new:
    ev = int(os.path.basename(f)[2:5]); dn = np.load(f)
    of = os.path.join(HERE, 'out', f'ev{ev:03d}.npz')
    old = np.load(of)['fit_err'][0] if os.path.exists(of) else np.nan
    fa, fb, ft = dn['fitA_err'][0], dn['fitB_err'][0], dn['fit_err'][0]
    win = 'AB'[int(dn['which'])]
    rows.append((ev, cwall(dn), old, fa, fb, ft, dn['fit_err'][1], dn['fit_err'][2], dn['fit_err'][3]))
    print(f'{ev:3d} {cwall(dn):+6.2f} | {old:6.1f} | {fa:6.1f} {fb:6.1f} {win:>3} {ft:6.1f} | '
          f'{dn["fit_err"][1]:5.2f} {dn["fit_err"][2]:+6.1f} {dn["fit_err"][3]:+6.2f}')
R = np.array([r[2:6] for r in rows])
print('-' * 76)
print(f'median fit vtx: OLD {np.nanmedian(R[:,0]):.1f}cm -> TWO-START {np.median(R[:,3]):.1f}cm   '
      f'| mean: OLD {np.nanmean(R[:,0]):.1f} -> TWO {np.mean(R[:,3]):.1f}')
inward = np.array([r[1] for r in rows]) < -0.8
if inward.any():
    print(f'inward (cWall<-0.8): OLD {np.nanmedian(R[inward,0]):.0f}cm -> TWO {np.median(R[inward,3]):.0f}cm  ({int(inward.sum())} ev)')
worse = int(np.sum(R[:, 3] > R[:, 0] + 5))
print(f'two-start worse than old by >5cm on {worse}/{len(rows)} events (should be ~0 — never-worse property)')
