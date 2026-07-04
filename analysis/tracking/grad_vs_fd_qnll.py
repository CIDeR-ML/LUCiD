#!/usr/bin/env python3
"""AD vs FD in E for the CHARGE POISSON NLL term only (no time term).

Same exercise as grad_vs_fd.py / grad_vs_fd_qtot.py, with the loss being exactly the charge
term of the reconstruction loss at truth geometry:

    L(E) = counts_loss(oc, mu(E), eps=0, normalize=False) = sum_i [ mu_i(E) - n_i ln mu_i(E) ]

with mu_i = max(total_charge_i * tot_n_scale, 1e-8) from the per-photon predictor (as in
ReconModel). Distinguishes the PER-PMT charge path (soft assignment weights per sensor) from
the total-charge path already shown to be AD-faithful.

    python analysis/tracking/grad_vs_fd_qnll.py --config <cfg.json> --output OUT
        [--events 0,1,2] [--emin 900 --emax 1500 --estep 50 --h 20]
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
    from lucid.fitting import track_from_vec9
    from lucid.losses import counts_loss

    cfg = load_config(args.config)
    cfg['fix_geometry'] = True
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    events = [int(x) for x in args.events.split(',')]
    E_scan = np.arange(args.emin, args.emax + 1e-9, args.estep)
    h = args.h

    pipe = TrackingPipeline(cfg)
    tns = float(cfg['gn']['tot_n_scale'])
    keys = [jax.random.PRNGKey(s) for s in range(cfg['gn']['nkeys'])]

    def make_fns(oc):
        def L(t9, key):
            _, _, _, tot = pipe.pred(track_from_vec9(t9), key)
            mu = jnp.maximum(tot * tns, 1e-8)
            return counts_loss(oc, mu, eps=0.0, normalize=False)
        return jax.jit(L), jax.jit(jax.grad(L))

    def zero_crossing(E, g):
        i = np.where(np.diff(np.sign(g)) != 0)[0]
        if not len(i): return None
        j = i[0]
        return float(E[j] - g[j] * (E[j + 1] - E[j]) / (g[j + 1] - g[j]))

    print(f"=== charge-NLL loss, AD vs FD in E: {len(events)} events, E in "
          f"[{args.emin},{args.emax}] step {args.estep}, FD h={h} MeV ===", flush=True)

    rows = []
    for evi in events:
        P = pipe._prepare_event(evi)
        th9 = P['th9']; oc = jnp.asarray(P['oc'])
        Lf, Gf = make_fns(oc)
        gad = np.zeros(len(E_scan)); gfd = np.zeros(len(E_scan)); lval = np.zeros(len(E_scan))
        for i, E in enumerate(E_scan):
            t9 = np.array(th9, float); t9[0] = E
            gad[i] = float(np.mean([np.asarray(Gf(jnp.asarray(t9), k))[0] for k in keys]))
            tp = t9.copy(); tp[0] = E + h; tm = t9.copy(); tm[0] = E - h
            lp = np.mean([float(Lf(jnp.asarray(tp), k)) for k in keys])
            lm = np.mean([float(Lf(jnp.asarray(tm), k)) for k in keys])
            gfd[i] = (lp - lm) / (2 * h)
            lval[i] = np.mean([float(Lf(jnp.asarray(t9), k)) for k in keys])
        zad, zfd = zero_crossing(E_scan, gad), zero_crossing(E_scan, gfd)
        rows.append(dict(ev=evi, E_scan=E_scan, g_ad=gad, g_fd=gfd, lval=lval,
                         zero_ad=zad, zero_fd=zfd))
        print(f"ev{evi:04d}  zero(g_AD)={zad if zad else 'none':>8}  "
              f"zero(g_FD)={zfd if zfd else 'none':>8}  (truth 1000)", flush=True)

    fig, ax = plt.subplots(2, len(rows), figsize=(5.2 * len(rows), 8), sharex=True)
    ax = np.atleast_2d(ax.reshape(2, -1))
    for c, r in enumerate(rows):
        a = ax[0, c]
        a.axhline(0, c='k', lw=.8); a.axvline(1000, c='k', ls=':', lw=1)
        a.plot(r['E_scan'], r['g_ad'], 'o-', label='AD grad')
        a.plot(r['E_scan'], r['g_fd'], 's--', label='FD slope')
        if r['zero_ad']: a.axvline(r['zero_ad'], color='C0', ls='--', lw=1)
        if r['zero_fd']: a.axvline(r['zero_fd'], color='C1', ls='--', lw=1)
        a.set(title=f"ev{r['ev']}"); a.grid(alpha=.3); a.legend(fontsize=8)
        b = ax[1, c]
        b.axvline(1000, c='k', ls=':', lw=1)
        b.plot(r['E_scan'], r['lval'] - r['lval'].min(), 'o-', color='C2')
        b.set(xlabel='E (MeV)', ylabel='L - min' if c == 0 else None); b.grid(alpha=.3)
    ax[0, 0].set_ylabel('dL/dE  (charge Poisson NLL)')
    fig.suptitle('Charge-NLL term: AD gradient vs FD slope in E (truth geometry)')
    fig.tight_layout()
    png = out_dir / 'grad_vs_fd_qnll.png'; fig.savefig(png, dpi=130)

    with h5py.File(out_dir / 'grad_vs_fd_qnll.h5', 'w') as h5:
        h5.attrs['config_json'] = json.dumps(cfg); h5.attrs['h_MeV'] = h
        h5.attrs['finished'] = datetime.now().isoformat()
        for r in rows:
            g = h5.create_group(f"ev{r['ev']:04d}")
            for k in ('E_scan', 'g_ad', 'g_fd', 'lval'):
                g.create_dataset(k, data=r[k])
            g.attrs['zero_ad'] = r['zero_ad'] or np.nan; g.attrs['zero_fd'] = r['zero_fd'] or np.nan
    print(f"wrote {png}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
