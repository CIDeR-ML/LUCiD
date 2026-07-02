#!/usr/bin/env python3
"""Standalone study of the INITIAL-GUESS (seed) performance — no Gauss-Newton.

For each event, builds the two complementary seeds (charge-grid ‖ time-multilateration, shared
energy scan, per-vertex cone direction) via :meth:`TrackingPipeline.seed_event` and scores them
against the exact truth. Reports, per component:

  vtx_cm, vtx_trans_cm, vtx_long_cm (signed, + = ahead), dir_deg, dE_MeV, dt0_ns

for seedA, seedB, the component-wise ORACLE (best-of-two per component), a vertex-oracle (pick
the seed with the smaller 3D vertex error), and the LOSS-PICK (what the margin-gated data-loss
selector in fit_track_multistart would hand to GN). Also reports how often the loss-pick agrees
with the vertex-oracle — i.e. whether the selector chooses the geometrically better seed.

    python analysis/tracking/seed_study.py --config <cfg.json> --output OUT [--events 0,1,2]

Writes ``<OUT>/<name>_seeds.h5`` (flat stacked arrays), prints a summary table, and saves PNGs.
Shares the --config/--output/--events interface with run_study.py, so submit_job.py can queue it
via ``--run-script seed_study.py``.
"""
import argparse
import json
import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.tracking.pipeline import (TrackingPipeline, load_config, SEED_ERR_NAMES,   # noqa: E402
                                        REPO_ROOT)


def _stats(errs):
    """(N, 6) -> {comp: (median, rms)} over events."""
    return {n: (float(np.median(errs[:, i])), float(np.sqrt(np.mean(errs[:, i] ** 2))))
            for i, n in enumerate(SEED_ERR_NAMES)}


def _print_summary(errA, errB, errF, pick_gated, pick3):
    N = len(errA)
    allthree = np.stack([errA, errB, errF])                       # (3, N, 6)
    oracle = np.take_along_axis(allthree, np.argmin(np.abs(allthree), 0)[None], 0)[0]  # min |err| per comp
    losspick3 = allthree[pick3, np.arange(N)]                     # the 3-way loss-selected seed per event

    cols = [('seedA', _stats(errA)), ('seedB', _stats(errB)), ('fused', _stats(errF)),
            ('loss-pick3', _stats(losspick3)), ('oracle', _stats(oracle))]
    print(f"\n{'='*104}\n  SEED PERFORMANCE  (N={N} events)   median / RMS per component\n{'='*104}")
    print(f"  {'component':<14}" + "".join(f"{c[0]:>16}" for c in cols))
    for comp in SEED_ERR_NAMES:
        row = f"  {comp:<14}"
        for _, st in cols:
            med, rms = st[comp]; row += f"  {med:6.1f}/{rms:6.1f} "
        print(row)

    fr = [float(np.mean(pick3 == i)) for i in range(3)]
    print(f"{'-'*104}")
    print(f"  3-way loss-pick chooses  A {fr[0]*100:4.0f}%  B {fr[1]*100:4.0f}%  fused {fr[2]*100:4.0f}%")
    print(f"  vertex RMS (cm):  seedA {np.sqrt(np.mean(errA[:,0]**2)):6.1f}   "
          f"seedB {np.sqrt(np.mean(errB[:,0]**2)):6.1f}   fused {np.sqrt(np.mean(errF[:,0]**2)):6.1f}   "
          f"loss-pick3 {np.sqrt(np.mean(losspick3[:,0]**2)):6.1f}   oracle {np.sqrt(np.mean(oracle[:,0]**2)):6.1f}")
    print(f"  t0 RMS (ns)   :  seedA {np.sqrt(np.mean(errA[:,5]**2)):6.1f}   "
          f"seedB {np.sqrt(np.mean(errB[:,5]**2)):6.1f}   fused {np.sqrt(np.mean(errF[:,5]**2)):6.1f}")
    print(f"{'='*104}\n")


def _plots(out_base, errA, errB, errF, lossA, lossB, lossF, pick_gated, pick3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    vtxA, vtxB = errA[:, 0], errB[:, 0]
    losspick3 = np.stack([errA, errB, errF])[pick3, np.arange(len(errA))]

    # 1. per-component error histograms. Draw derived series (loss-pick3) first, primaries last
    # (seedA topmost) so identical/overlapping step curves stay legible.
    idx = [(0, 'vertex error (cm)', 'log'), (3, 'direction error (deg)', 'log'),
           (4, 'energy error (MeV)', 'linear'), (5, 't0 error (ns)', 'linear')]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for a, (i, lab, xs) in zip(ax.ravel(), idx):
        layers = [('loss-pick3', losspick3[:, i], 'C3', ':', 1.8),
                  ('fused', errF[:, i], 'C2', '-', 2.2),
                  ('seedB', errB[:, i], 'C1', '-', 2.0),
                  ('seedA', errA[:, i], 'C0', '-', 2.0)]
        if xs == 'log':
            lo = max(1e-2, min(s.min() for _, s, *_ in layers))
            bins = np.logspace(np.log10(lo), np.log10(max(s.max() for _, s, *_ in layers) + 1e-9), 30)
            a.set_xscale('log')
        else:
            bins = 30
        for name, s, col, ls, lw in layers:
            a.hist(s, bins=bins, histtype='step', color=col, ls=ls, lw=lw, label=name)
        a.set(xlabel=lab, ylabel='events'); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle('Seed error distributions  (seedA/seedB/fused solid; loss-pick3 dotted '
                 '— energy is shared across all seeds)')
    fig.tight_layout()
    fig.savefig(f'{out_base}_hist.png', dpi=120); plt.close(fig)

    # 2. vertex transverse vs longitudinal — the A/B bias signature and where fused lands
    fig, a = plt.subplots(figsize=(7, 5.8))
    a.scatter(errA[:, 2], errA[:, 1], s=18, alpha=.6, c='C0', label='seedA (charge-grid)')
    a.scatter(errB[:, 2], errB[:, 1], s=18, alpha=.6, c='C1', label='seedB (time-multilat.)')
    a.scatter(errF[:, 2], errF[:, 1], s=22, alpha=.7, c='C2', marker='D', label='fused')
    a.axvline(0, c='k', lw=.6)
    a.set(xlabel='longitudinal vertex error (cm, + = ahead)', ylabel='transverse vertex error (cm)',
          title='Vertex seed bias: transverse vs longitudinal'); a.grid(alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig(f'{out_base}_vtx_transverse_longitudinal.png', dpi=120); plt.close(fig)

    # 3. selector diagnostic: does lower data-loss => smaller vertex error?
    dloss = np.asarray(lossB) - np.asarray(lossA); dvtx = vtxB - vtxA
    fig, a = plt.subplots(figsize=(6.5, 5.5))
    sc = a.scatter(dloss, dvtx, c=(pick_gated == 1), cmap='coolwarm', s=22, alpha=.8)
    a.axhline(0, c='k', lw=.6); a.axvline(0, c='k', lw=.6)
    a.set(xlabel='lossB - lossA', ylabel='vtxB - vtxA (cm)',
          title='Selector: data-loss gap vs vertex-error gap\n(blue=picks A, red=picks B)')
    a.grid(alpha=.3); fig.colorbar(sc, label='loss-pick == B')
    fig.tight_layout(); fig.savefig(f'{out_base}_selector.png', dpi=120); plt.close(fig)
    return [f'{out_base}_hist.png', f'{out_base}_vtx_transverse_longitudinal.png',
            f'{out_base}_selector.png']


def replot(h5_path):
    """Regenerate the summary + PNGs from an existing seeds HDF5 (no reconstruction)."""
    import h5py
    with h5py.File(h5_path, 'r') as f:
        errA, errB, errF = f['seedA_err'][:], f['seedB_err'][:], f['seedF_err'][:]
        lossA, lossB, lossF = f['lossA'][:], f['lossB'][:], f['lossF'][:]
        pickg, pick3 = f['loss_pick_gated'][:], f['loss_pick3'][:]
    _print_summary(errA, errB, errF, pickg, pick3)
    base = str(Path(h5_path).with_suffix(''))
    for p in _plots(base, errA, errB, errF, lossA, lossB, lossF, pickg, pick3):
        print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', help='study config JSON (required unless --replot)')
    ap.add_argument('--output', help='output dir (required unless --replot)')
    ap.add_argument('--events', default=None, help='comma list (overrides event_start/n_events)')
    ap.add_argument('--name', default=None)
    ap.add_argument('--replot', default=None,
                    help='regenerate summary+plots from an existing *_seeds.h5, then exit')
    args = ap.parse_args()

    if args.replot:
        replot(args.replot); return 0
    if not (args.config and args.output):
        ap.error('--config and --output are required unless --replot is given')

    import h5py

    cfg = load_config(args.config)
    name = args.name or (cfg.get('name') or Path(args.config).stem) + '_seeds'
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{name}.h5'

    events = ([int(x) for x in args.events.split(',')] if args.events
              else list(range(cfg['event_start'], cfg['event_start'] + cfg['n_events'])))

    print(f"=== seed study '{name}' ===  {len(events)} events, n_rays={cfg['n_rays']}")
    print(f"  config: {Path(args.config).resolve()}\n  output: {out_path}\n", flush=True)

    pipe = TrackingPipeline(cfg)
    rows = []
    for ev in events:
        try:
            r = pipe.seed_event(ev)
            rows.append(r)
            print(f"ev{ev:04d} A[vtx{r['seedA_err'][0]:6.1f} dir{r['seedA_err'][3]:5.2f}] "
                  f"B[vtx{r['seedB_err'][0]:6.1f} dir{r['seedB_err'][3]:5.2f}] "
                  f"F[vtx{r['seedF_err'][0]:6.1f} t0{r['seedF_err'][5]:+5.1f}] "
                  f"pick3={'ABF'[r['loss_pick3']]} [{r['seconds']:.0f}s]", flush=True)
        except Exception as e:
            print(f"ev{ev:04d} FAILED: {type(e).__name__}: {e}", flush=True)

    if not rows:
        print("No events succeeded."); return 1

    def stack(key):
        return np.stack([np.asarray(r[key]) for r in rows])
    errA, errB, errF = stack('seedA_err'), stack('seedB_err'), stack('seedF_err')
    lossA = np.array([r['lossA'] for r in rows]); lossB = np.array([r['lossB'] for r in rows])
    lossF = np.array([r['lossF'] for r in rows])
    pick = np.array([r['loss_pick'] for r in rows]); pickg = np.array([r['loss_pick_gated'] for r in rows])
    pick3 = np.array([r['loss_pick3'] for r in rows])

    with h5py.File(out_path, 'w') as h5:
        h5.attrs['config_json'] = json.dumps(cfg)
        h5.attrs['name'] = name; h5.attrs['particle'] = cfg['particle']
        h5.attrs['n_sensors'] = pipe.ND; h5.attrs['n_rays'] = cfg['n_rays']
        h5.attrs['energy_nominal_MeV'] = cfg['energy_nominal_MeV']
        h5.attrs['n_events'] = len(rows)
        h5.attrs['seed_err_names'] = ','.join(SEED_ERR_NAMES)
        h5.attrs['phys_param_order'] = 'x,y,z,phi,theta,t0,E'
        h5.attrs['hostname'] = socket.gethostname()
        h5.attrs['finished'] = datetime.now().isoformat()
        for key in ('truth_phys', 'seedA_phys', 'seedB_phys', 'seedF_phys', 'truth_vec9',
                    'seedA_vec9', 'seedB_vec9', 'seedF_vec9', 'tdir'):
            h5.create_dataset(key, data=stack(key), compression='gzip')
        h5.create_dataset('seedA_err', data=errA, compression='gzip')
        h5.create_dataset('seedB_err', data=errB, compression='gzip')
        h5.create_dataset('seedF_err', data=errF, compression='gzip')
        h5.create_dataset('lossA', data=lossA); h5.create_dataset('lossB', data=lossB)
        h5.create_dataset('lossF', data=lossF)
        h5.create_dataset('loss_pick', data=pick); h5.create_dataset('loss_pick_gated', data=pickg)
        h5.create_dataset('loss_pick3', data=pick3)
        h5.create_dataset('ev', data=np.array([r['ev'] for r in rows]))
        h5.create_dataset('energy_true', data=np.array([r['energy_true'] for r in rows]))
        h5.create_dataset('n_hit', data=np.array([r['n_hit'] for r in rows]))

    _print_summary(errA, errB, errF, pickg, pick3)
    pngs = _plots(str(out_path.with_suffix('')), errA, errB, errF, lossA, lossB, lossF, pickg, pick3)
    print(f"wrote {out_path}")
    for p in pngs:
        print(f"wrote {p}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
