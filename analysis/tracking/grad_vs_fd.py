#!/usr/bin/env python3
"""AD-gradient vs finite-difference slope of the loss, in E, at truth geometry.

For each event and each E point: geometry+t0 pinned at truth, compare

    g_AD(E) = dL/dE from reverse-mode autodiff (ReconModel.grad — what the GN follows;
              the time term carries the AMP_DETACH stop_gradients),
    g_FD(E) = [L(E+h) - L(E-h)] / 2h from the plain loss values (ReconModel.loss —
              what the scan/minimum sees), h = 20 MeV (SCALE9 FD scale is 0.4*50=20).

Both averaged over the GN nkeys. If the two disagree systematically — g_AD crossing zero
near ~1.3 E_true while g_FD crosses near E_true — the AMP_DETACH attribution is proven.

    python analysis/tracking/grad_vs_fd.py --config <cfg.json> --output OUT [--events 0,1,2]
                                            [--emin 900 --emax 1500 --estep 50 --h 20]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.tracking.pipeline import TrackingPipeline, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--events', default='0,1,2')
    ap.add_argument('--emin', type=float, default=900.0)
    ap.add_argument('--emax', type=float, default=1500.0)
    ap.add_argument('--estep', type=float, default=50.0)
    ap.add_argument('--h', type=float, default=20.0)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfg = load_config(args.config)
    cfg['fix_geometry'] = True                       # data only, no seeder
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    events = [int(x) for x in args.events.split(',')]
    E_scan = np.arange(args.emin, args.emax + 1e-9, args.estep)
    h = args.h

    pipe = TrackingPipeline(cfg)
    keys = [jax.random.PRNGKey(s) for s in range(cfg['gn']['nkeys'])]

    def loss_at(t9):
        return float(np.mean([float(pipe.model.loss(jnp.asarray(t9), oc, ot, k)) for k in keys]))

    def grad_E_at(t9):
        return float(np.mean([np.asarray(pipe.model.grad(jnp.asarray(t9), oc, ot, k))[0]
                              for k in keys]))

    def zero_crossing(E, g):
        i = np.where(np.diff(np.sign(g)) != 0)[0]
        if not len(i): return None
        j = i[0]
        return float(E[j] - g[j] * (E[j + 1] - E[j]) / (g[j + 1] - g[j]))

    print(f"=== AD-grad vs FD-slope in E: {len(events)} events, E in "
          f"[{args.emin},{args.emax}] step {args.estep}, FD h={h} MeV ===", flush=True)

    rows = []
    for evi in events:
        P = pipe._prepare_event(evi)
        th9 = P['th9']; oc = jnp.asarray(P['oc']); ot = jnp.asarray(P['ot'])
        gad = np.zeros(len(E_scan)); gfd = np.zeros(len(E_scan))
        for i, E in enumerate(E_scan):
            t9 = np.array(th9, float); t9[0] = E
            gad[i] = grad_E_at(t9)
            tp = t9.copy(); tp[0] = E + h; tm = t9.copy(); tm[0] = E - h
            gfd[i] = (loss_at(tp) - loss_at(tm)) / (2 * h)
        zad, zfd = zero_crossing(E_scan, gad), zero_crossing(E_scan, gfd)
        rows.append(dict(ev=evi, E_scan=E_scan, g_ad=gad, g_fd=gfd, zero_ad=zad, zero_fd=zfd))
        print(f"ev{evi:04d}  zero(g_AD)={zad if zad else 'none':>8}  "
              f"zero(g_FD)={zfd if zfd else 'none':>8}  (truth 1000)", flush=True)

    fig, ax = plt.subplots(1, len(rows), figsize=(5.2 * len(rows), 4.4), sharey=False)
    ax = np.atleast_1d(ax)
    for a, r in zip(ax, rows):
        a.axhline(0, c='k', lw=.8); a.axvline(1000, c='k', ls=':', lw=1)
        a.plot(r['E_scan'], r['g_ad'], 'o-', label='AD grad (GN uses)')
        a.plot(r['E_scan'], r['g_fd'], 's--', label='FD slope of loss')
        if r['zero_ad']: a.axvline(r['zero_ad'], color='C0', ls='--', lw=1)
        if r['zero_fd']: a.axvline(r['zero_fd'], color='C1', ls='--', lw=1)
        a.set(xlabel='E (MeV)', title=f"ev{r['ev']}"); a.grid(alpha=.3); a.legend(fontsize=8)
    ax[0].set_ylabel('dL/dE')
    fig.suptitle('AD gradient vs FD slope of the loss, in E (truth geometry; dotted = true E)')
    fig.tight_layout()
    png = out_dir / 'grad_vs_fd.png'; fig.savefig(png, dpi=130)

    with h5py.File(out_dir / 'grad_vs_fd.h5', 'w') as h5:
        h5.attrs['config_json'] = json.dumps(cfg); h5.attrs['h_MeV'] = h
        h5.attrs['finished'] = datetime.now().isoformat()
        for r in rows:
            g = h5.create_group(f"ev{r['ev']:04d}")
            for k in ('E_scan', 'g_ad', 'g_fd'):
                g.create_dataset(k, data=r[k])
            g.attrs['zero_ad'] = r['zero_ad'] or np.nan; g.attrs['zero_fd'] = r['zero_fd'] or np.nan
    print(f"wrote {png}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
