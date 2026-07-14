#!/usr/bin/env python3
"""p68 convergence: the 68% quantile of |error| across events, per GN iteration.

Reads a run_study HDF5 and, for every event's WINNING-seed trajectory, computes at each
iteration the four errors vs truth — vertex distance (cm), direction (deg), |dE| (MeV),
|dt0| (ns) — then takes the 68% quantile over events at each iteration. Also reports p68 at
the seed (iter 0), at each event's min-||g|| readout iterate (the actual fit readout), and at
the final iterate.

    python analysis/paper/utils/p68_evolution.py <run.h5> [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def _dirs_from_traj(tr):
    """(niters+1, 9) -> (niters+1, 3) unit directions (numpy mirror of vec9_dir)."""
    st, ct, sp, cp = tr[:, 4], tr[:, 5], tr[:, 6], tr[:, 7]
    nt = np.hypot(st, ct) + 1e-12; npp = np.hypot(sp, cp) + 1e-12
    st, ct, sp, cp = st / nt, ct / nt, sp / npp, cp / npp
    return np.stack([st * cp, st * sp, ct], 1)


def event_err_curves(traj, truth, tdir):
    """(niters+1, 4) error curves for one event: [vtx cm, dir deg, |dE| MeV, |dt0| ns]."""
    tr = np.asarray(traj); truth = np.asarray(truth)
    vtx = np.linalg.norm(tr[:, 1:4] - truth[1:4], axis=1) * 100.0
    ddeg = np.degrees(np.arccos(np.clip(_dirs_from_traj(tr) @ np.asarray(tdir), -1, 1)))
    dE = np.abs(tr[:, 0] - truth[0])
    dt0 = np.abs(tr[:, 8] - truth[8])
    return np.stack([vtx, ddeg, dE, dt0], 1)


METRICS = [('vertex distance (cm)', 'vtx'), ('direction (deg)', 'dir'),
           ('|dE| (MeV)', 'E'), ('|dt0| (ns)', 't0')]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('h5file')
    ap.add_argument('--out', default=None, help='output dir for the PNG (default: next to the h5)')
    args = ap.parse_args()

    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    curves, ming_iters, ming_errs, final_errs, which = [], [], [], [], []
    with h5py.File(args.h5file, 'r') as f:
        for k in sorted(f['events']):
            ev = f['events'][k]
            c = event_err_curves(ev['traj_win'][:], ev['truth_vec9'][:], ev['tdir'][:])
            curves.append(c)
            bi = int(ev.attrs['best_iter_win'])
            ming_iters.append(bi); ming_errs.append(c[bi]); final_errs.append(c[-1])
            which.append(int(ev.attrs['which']))
        name = f.attrs.get('name', Path(args.h5file).stem)

    C = np.stack(curves)                       # (n_events, niters+1, 4)
    N, NI, _ = C.shape
    p68 = np.quantile(C, 0.68, axis=0)         # (niters+1, 4)
    ming_errs = np.stack(ming_errs); final_errs = np.stack(final_errs)
    p68_seed, p68_ming, p68_final = p68[0], np.quantile(ming_errs, 0.68, 0), np.quantile(final_errs, 0.68, 0)

    wfrac = [int(np.sum(np.array(which) == i)) for i in range(3)]
    print(f"\n=== p68 convergence — {name}  (N={N} events, {NI-1} GN iterations) ===")
    print(f"  winning seed: A {wfrac[0]}  B {wfrac[1]}  fused {wfrac[2]}")
    print(f"  median min-||g|| readout iteration: {np.median(ming_iters):.0f}")
    print(f"  {'metric':<22}{'seed (it 0)':>14}{'min-g readout':>16}{'final iter':>14}")
    for j, (lab, _) in enumerate(METRICS):
        print(f"  {lab:<22}{p68_seed[j]:>14.2f}{p68_ming[j]:>16.2f}{p68_final[j]:>14.2f}")

    it = np.arange(NI)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for j, (lab, _) in enumerate(METRICS):
        a = ax.ravel()[j]
        a.plot(it, p68[:, j], lw=1.8, color='C0')
        a.axhline(p68_ming[j], color='C3', ls='--', lw=1.1,
                  label=f'min-‖g‖ readout p68 = {p68_ming[j]:.2f}')
        a.set(ylabel=f'p68 {lab}', yscale='log'); a.grid(alpha=.3, which='both')
        a.legend(fontsize=8)
    for a in ax[1]:
        a.set_xlabel('GN iteration')
    fig.suptitle(f'p68 evolution over GN iterations — {name} (N={N}, winning seed per event)')
    fig.tight_layout()

    out_dir = Path(args.out) if args.out else Path(args.h5file).parent
    png = out_dir / f'{Path(args.h5file).stem}_p68_evolution.png'
    fig.savefig(png, dpi=130)
    print(f"\nwrote {png}")


if __name__ == '__main__':
    sys.exit(main())
