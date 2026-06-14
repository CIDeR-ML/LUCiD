"""Compare OLD (charge-grid seed, out/) vs NEW (time_vertex seed, out_tv/) on the same events.
Shows seed-vtx and fit-vtx error side by side, plus cWall (track inward-ness) per event."""
import os, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
new = sorted(glob.glob(os.path.join(HERE, 'out_tv', 'ev*.npz')))


def cwall(d):
    vtx = d['truth'][1:4]; rad = np.array([vtx[0], vtx[1], 0.]); rad /= np.linalg.norm(rad) + 1e-9
    return float(d['tdir'] @ rad)


print(f'{"ev":>3} {"cWall":>6} | {"OLDseed":>7} {"NEWseed":>7} | {"OLDfit":>7} {"NEWfit":>7} | {"dir":>5} {"E":>6} {"t0":>6}')
print('-' * 78)
rows = []
for f in new:
    ev = int(os.path.basename(f)[2:5]); dn = np.load(f)
    of = os.path.join(HERE, 'out', f'ev{ev:03d}.npz')
    do = np.load(of) if os.path.exists(of) else None
    cw = cwall(dn)
    os_, ns = (do['seed_err'][0] if do is not None else np.nan), dn['seed_err'][0]
    ofit, nfit = (do['fit_err'][0] if do is not None else np.nan), dn['fit_err'][0]
    rows.append((ev, cw, os_, ns, ofit, nfit, dn['fit_err'][1], dn['fit_err'][2], dn['fit_err'][3]))
    print(f'{ev:3d} {cw:+6.2f} | {os_:7.0f} {ns:7.0f} | {ofit:7.1f} {nfit:7.1f} | '
          f'{dn["fit_err"][1]:5.2f} {dn["fit_err"][2]:+6.1f} {dn["fit_err"][3]:+6.2f}')
R = np.array([r[2:6] for r in rows])
print('-' * 78)
print(f'median seed: OLD {np.nanmedian(R[:,0]):.0f}cm -> NEW {np.median(R[:,1]):.0f}cm   '
      f'| median fit: OLD {np.nanmedian(R[:,2]):.1f}cm -> NEW {np.median(R[:,3]):.1f}cm')
inward = np.array([r[1] for r in rows]) < -0.8
if inward.any():
    print(f'inward (cWall<-0.8) fit: OLD {np.nanmedian(R[inward,2]):.0f}cm -> NEW {np.median(R[inward,3]):.0f}cm  ({int(inward.sum())} events)')
