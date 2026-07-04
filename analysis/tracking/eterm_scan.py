#!/usr/bin/env python3
"""Loss-term decomposition vs E at TRUTH geometry: charge Poisson term vs time NLL term.

For each event: pin (vertex, direction, t0) at truth, scan E, and evaluate the two components
of the reconstruction loss separately (averaged over the GN nkeys):

    charge_i(E) = counts_loss(oc, mu(E), eps=0, normalize=False)   # sum mu - n ln mu
    time_i(E)   = sum first_arrival_window_nll(E)

Reports each term's argmin (and the total's) per event, and writes a per-event scan curve plot.

    python analysis/tracking/eterm_scan.py --config <cfg.json> --output OUT [--events 0,1,2]
                                            [--emin 800 --emax 1500 --estep 25]
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
    ap.add_argument('--events', default='0,1,2,3,4')
    ap.add_argument('--emin', type=float, default=800.0)
    ap.add_argument('--emax', type=float, default=1500.0)
    ap.add_argument('--estep', type=float, default=25.0)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lucid.losses import counts_loss

    cfg = load_config(args.config)
    cfg['fix_geometry'] = True                   # _prepare_event: data only, no seeder
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    events = [int(x) for x in args.events.split(',')]
    E_scan = np.arange(args.emin, args.emax + 1e-9, args.estep)

    pipe = TrackingPipeline(cfg)
    keys = [jax.random.PRNGKey(s) for s in range(cfg['gn']['nkeys'])]

    print(f"=== E-term scan: {len(events)} events, E in [{args.emin},{args.emax}] "
          f"step {args.estep} ({len(E_scan)} pts) ===", flush=True)

    all_rows = []
    for ev in events:
        P = pipe._prepare_event(ev)
        th9 = P['th9']; oc = jnp.asarray(P['oc']); ot = jnp.asarray(P['ot'])
        qterm = np.zeros(len(E_scan)); tterm = np.zeros(len(E_scan))
        for i, E in enumerate(E_scan):
            t9 = np.array(th9, float); t9[0] = E
            qs = ts = 0.0
            for k in keys:
                mu, tnll = pipe.model.perpmt(jnp.asarray(t9), oc, ot, k)
                qs += float(counts_loss(oc, mu, eps=0.0, normalize=False))
                ts += float(jnp.sum(tnll))
            qterm[i] = qs / len(keys); tterm[i] = ts / len(keys)
        tot = qterm + tterm
        eq, et, etot = (float(E_scan[np.argmin(x)]) for x in (qterm, tterm, tot))
        all_rows.append(dict(ev=ev, E_scan=E_scan, qterm=qterm, tterm=tterm,
                             eq=eq, et=et, etot=etot))
        print(f"ev{ev:04d}  argmin_E: charge={eq:6.0f}  time={et:6.0f}  total={etot:6.0f} MeV "
              f"(truth 1000)", flush=True)

    # plot: per-event curves, each term minus its own minimum
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    for r in all_rows:
        ax[0].plot(r['E_scan'], r['qterm'] - r['qterm'].min(), label=f"ev{r['ev']}")
        ax[1].plot(r['E_scan'], r['tterm'] - r['tterm'].min())
        ax[2].plot(r['E_scan'], (r['qterm'] + r['tterm']) - (r['qterm'] + r['tterm']).min())
    for a, t in zip(ax, ['charge Poisson term', 'time NLL term', 'total loss']):
        a.axvline(1000, c='k', ls=':', lw=1); a.set(xlabel='E (MeV)', title=t); a.grid(alpha=.3)
    ax[0].set_ylabel('term - min'); ax[0].legend(fontsize=8)
    fig.suptitle('Loss terms vs E at truth geometry (dotted = true E)')
    fig.tight_layout()
    png = out_dir / 'eterm_scan.png'; fig.savefig(png, dpi=130)

    with h5py.File(out_dir / 'eterm_scan.h5', 'w') as h5:
        h5.attrs['config_json'] = json.dumps(cfg); h5.attrs['finished'] = datetime.now().isoformat()
        for r in all_rows:
            g = h5.create_group(f"ev{r['ev']:04d}")
            for k in ('E_scan', 'qterm', 'tterm'):
                g.create_dataset(k, data=r[k])
            g.attrs.update({k: r[k] for k in ('eq', 'et', 'etot')})
    print(f"wrote {png}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
