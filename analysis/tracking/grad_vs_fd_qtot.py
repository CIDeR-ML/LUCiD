#!/usr/bin/env python3
"""AD vs FD in E for the SIMPLEST loss: total predicted vs observed charge.

Same exercise as grad_vs_fd.py, but the loss is just

    L(E) = 0.5 * ( sum_i mu_i(E) - sum_i n_i )^2

with mu from the per-photon predictor at truth geometry (mu = max(total_charge *
tot_n_scale, 1e-8), as in ReconModel). No time term, no per-PMT structure — this isolates
whether d(sum mu)/dE from reverse-mode autodiff through the simulator agrees with the finite-
difference slope, i.e. whether the E-gradient pathology is already present in the predicted
TOTAL charge or only enters via the time-NLL (AMP_DETACH) term.

    python analysis/tracking/grad_vs_fd_qtot.py --config <cfg.json> --output OUT
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

    cfg = load_config(args.config)
    cfg['fix_geometry'] = True                       # data only, no seeder
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    events = [int(x) for x in args.events.split(',')]
    E_scan = np.arange(args.emin, args.emax + 1e-9, args.estep)
    h = args.h

    pipe = TrackingPipeline(cfg)
    tns = float(cfg['gn']['tot_n_scale'])
    keys = [jax.random.PRNGKey(s) for s in range(cfg['gn']['nkeys'])]

    def make_fns(q_obs):
        def L(t9, key):
            _, _, _, tot = pipe.pred(track_from_vec9(t9), key)
            mu = jnp.maximum(tot * tns, 1e-8)
            return 0.5 * (jnp.sum(mu) - q_obs) ** 2
        return jax.jit(L), jax.jit(jax.grad(L))

    def zero_crossing(E, g):
        i = np.where(np.diff(np.sign(g)) != 0)[0]
        if not len(i): return None
        j = i[0]
        return float(E[j] - g[j] * (E[j + 1] - E[j]) / (g[j + 1] - g[j]))

    print(f"=== Qtot loss, AD vs FD in E: {len(events)} events, E in "
          f"[{args.emin},{args.emax}] step {args.estep}, FD h={h} MeV ===", flush=True)

    rows = []
    for evi in events:
        P = pipe._prepare_event(evi)
        th9 = P['th9']; q_obs = float(np.sum(P['oc']))
        Lf, Gf = make_fns(q_obs)
        gad = np.zeros(len(E_scan)); gfd = np.zeros(len(E_scan)); qpred = np.zeros(len(E_scan))
        for i, E in enumerate(E_scan):
            t9 = np.array(th9, float); t9[0] = E
            gad[i] = float(np.mean([np.asarray(Gf(jnp.asarray(t9), k))[0] for k in keys]))
            tp = t9.copy(); tp[0] = E + h; tm = t9.copy(); tm[0] = E - h
            lp = np.mean([float(Lf(jnp.asarray(tp), k)) for k in keys])
            lm = np.mean([float(Lf(jnp.asarray(tm), k)) for k in keys])
            gfd[i] = (lp - lm) / (2 * h)
            # predicted total at E for context (first key)
            _, _, _, tot = pipe.pred(track_from_vec9(jnp.asarray(t9)), keys[0])
            qpred[i] = float(jnp.sum(jnp.maximum(tot * tns, 1e-8)))
        zad, zfd = zero_crossing(E_scan, gad), zero_crossing(E_scan, gfd)
        rows.append(dict(ev=evi, E_scan=E_scan, g_ad=gad, g_fd=gfd, qpred=qpred,
                         q_obs=q_obs, zero_ad=zad, zero_fd=zfd))
        print(f"ev{evi:04d}  q_obs={q_obs:7.0f}  zero(g_AD)={zad if zad else 'none':>8}  "
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
        b.plot(r['E_scan'], r['qpred'], 'o-', color='C2', label='Q_pred(E)')
        b.axhline(r['q_obs'], color='C3', ls='--', lw=1.2, label='Q_obs')
        b.set(xlabel='E (MeV)'); b.grid(alpha=.3); b.legend(fontsize=8)
    ax[0, 0].set_ylabel('dL/dE  (L = 0.5(Qpred-Qobs)^2)')
    ax[1, 0].set_ylabel('total charge')
    fig.suptitle('Total-charge loss: AD gradient vs FD slope in E (truth geometry)')
    fig.tight_layout()
    png = out_dir / 'grad_vs_fd_qtot.png'; fig.savefig(png, dpi=130)

    with h5py.File(out_dir / 'grad_vs_fd_qtot.h5', 'w') as h5:
        h5.attrs['config_json'] = json.dumps(cfg); h5.attrs['h_MeV'] = h
        h5.attrs['finished'] = datetime.now().isoformat()
        for r in rows:
            g = h5.create_group(f"ev{r['ev']:04d}")
            for k in ('E_scan', 'g_ad', 'g_fd', 'qpred'):
                g.create_dataset(k, data=r[k])
            g.attrs['q_obs'] = r['q_obs']
            g.attrs['zero_ad'] = r['zero_ad'] or np.nan; g.attrs['zero_fd'] = r['zero_fd'] or np.nan
    print(f"wrote {png}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
