#!/usr/bin/env python3
"""Residual maps of the two likelihood terms at TRUTH — hypothesis-free diagnostics.

Per event: one forward pass at the true vec9, compared to the observed event.
  CHARGE map: sum(observed)/sum(predicted) charge per cos(alpha) bin, where alpha is the
              angle of the PMT (seen from the true vertex) w.r.t. the true track direction.
              Fore/aft structure here IS the charge-term longitudinal lever.
  TIME map  : per hit PMT, observed first arrival (t0-corrected) minus the predicted
              earliest arrival (intensity-weighted 5% quantile of the ray arrival times),
              profiled vs cos(alpha) and vs PMT distance. Structure here IS the time-term pull.

    python analysis/tracking/residual_maps.py --config <cfg.json> --output OUT
        [--n-events 100] [--nkeys 2]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.tracking.pipeline import TrackingPipeline, load_config  # noqa: E402


def time_residuals_by_index(fi, ft, w, nd, oc, tobs):
    """Likelihood-consistent per-PMT time residuals vs the predicted arrival DISTRIBUTION.

    For hit PMT i with n_i observed photons and observed first arrival tobs_i:
      pit_i  = 1 - (1 - F_i(tobs_i))**n_i   (PIT of the first-arrival order statistic;
               Uniform(0,1) under a correct model, occupancy-independent)
      dtq_i  = tobs_i - F_i^{-1}(1/(n_i+1)) (ns residual vs the order-statistic-matched
               quantile - the interpretable companion)
    Returns (pit, dtq), NaN where no rays or no hit.
    """
    pit = np.full(nd, np.nan); dtq = np.full(nd, np.nan)
    order = np.lexsort((ft, fi))
    fi, ft, w = fi[order], ft[order], w[order]
    starts = np.searchsorted(fi, np.arange(nd), side='left')
    ends = np.searchsorted(fi, np.arange(nd), side='right')
    for i in np.where(oc > 0)[0]:
        s, e = starts[i], ends[i]
        if e <= s:
            continue
        ww = w[s:e]; tt = ft[s:e]
        cw = np.cumsum(ww)
        if cw[-1] <= 0:
            continue
        n = float(oc[i])
        F = np.searchsorted(tt, tobs[i], side='right')
        Ft = (cw[F - 1] / cw[-1]) if F > 0 else 0.0
        pit[i] = 1.0 - (1.0 - min(Ft, 1.0)) ** n
        q = 1.0 / (n + 1.0)
        dtq[i] = tobs[i] - tt[np.searchsorted(cw, q * cw[-1])]
    return pit, dtq


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--n-events', type=int, default=100)
    ap.add_argument('--nkeys', type=int, default=2)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lucid.fitting import track_from_vec9

    cfg = load_config(args.config)
    pipe = TrackingPipeline(cfg, verbose=True)
    POS = np.asarray(pipe.POS)                                   # (ND, 3) sensor positions (m)
    ND = pipe.ND
    E = cfg['energy_nominal_MeV']

    ca_bins = np.linspace(-1, 1, 41)
    rd_bins = np.linspace(0, 40, 21)
    q_sum = np.zeros(len(ca_bins) - 1); mu_sum = np.zeros(len(ca_bins) - 1)
    dt_ca = [[] for _ in range(len(ca_bins) - 1)]
    dt_rd = [[] for _ in range(len(rd_bins) - 1)]

    fwd = jax.jit(lambda t9, key: pipe.pred(track_from_vec9(t9), key))
    for ev in range(args.n_events):
        P = pipe._prepare_event(ev)
        th9 = jnp.asarray(P['th9'])
        oc, ot = np.asarray(P['oc']), np.asarray(P['ot'])
        vtx = np.asarray(P['th9'][1:4]); u = np.asarray(P['d']); u = u / np.linalg.norm(u)
        rel = POS - vtx; dist = np.linalg.norm(rel, axis=1)
        ca = (rel @ u) / np.maximum(dist, 1e-9)
        ci = np.clip(np.digitize(ca, ca_bins) - 1, 0, len(ca_bins) - 2)
        ri = np.clip(np.digitize(dist, rd_bins) - 1, 0, len(rd_bins) - 2)

        tobs = ot - float(P['th9'][8])
        mu_acc = np.zeros(ND)
        fi_all, ft_all, w_all = [], [], []
        for s in range(args.nkeys):
            lw, ft, fi, tot = fwd(th9, jax.random.PRNGKey(s))
            mu_acc += np.asarray(tot)
            fi_all.append(np.asarray(fi).astype(int)); ft_all.append(np.asarray(ft))
            w_all.append(np.exp(np.asarray(lw)))
        mu = mu_acc / args.nkeys
        pit, dtq = time_residuals_by_index(np.concatenate(fi_all), np.concatenate(ft_all),
                                           np.concatenate(w_all), ND, oc, tobs)

        np.add.at(q_sum, ci, oc)
        np.add.at(mu_sum, ci, mu)
        for i in np.where(np.isfinite(pit))[0]:
            dt_ca[ci[i]].append((pit[i], dtq[i])); dt_rd[ri[i]].append((pit[i], dtq[i]))
        if ev % 20 == 0:
            print(f'ev{ev:04d} done', flush=True)

    cac = 0.5 * (ca_bins[:-1] + ca_bins[1:]); rdc = 0.5 * (rd_bins[:-1] + rd_bins[1:])
    ratio = np.where(mu_sum > 0, q_sum / np.maximum(mu_sum, 1e-9), np.nan)
    pit_ca = np.array([np.mean([x[0] for x in v]) if len(v) > 30 else np.nan for v in dt_ca])
    pit_rd = np.array([np.mean([x[0] for x in v]) if len(v) > 30 else np.nan for v in dt_rd])
    dt_ca_med = np.array([np.median([x[1] for x in v]) if len(v) > 30 else np.nan for v in dt_ca])
    dt_rd_med = np.array([np.median([x[1] for x in v]) if len(v) > 30 else np.nan for v in dt_rd])

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(cac, ratio, 'o-', color='navy')
    ax[0].axhline(1, c='k', lw=.8)
    ax[0].set(xlabel=r'$\cos\alpha$ (PMT vs track dir, from true vertex)',
              ylabel='observed / predicted charge', title=f'{E} MeV — charge residual map')
    ax[0].grid(alpha=.3)
    ax[1].plot(cac, pit_ca - 0.5, 'o-', color='crimson', label='PIT mean - 0.5')
    ax[1].plot(cac, dt_ca_med, 's--', color='darkorange', label='order-stat dt (ns)', alpha=.7)
    ax[1].axhline(0, c='k', lw=.8)
    ax[1].set(xlabel=r'$\cos\alpha$', ylabel='calibration residual',
              title='time residual vs angle (PIT: >0 = light arrives LATER than model)')
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8, frameon=False)
    ax[2].plot(rdc, pit_rd - 0.5, 'o-', color='crimson', label='PIT mean - 0.5')
    ax[2].plot(rdc, dt_rd_med, 's--', color='darkorange', label='order-stat dt (ns)', alpha=.7)
    ax[2].axhline(0, c='k', lw=.8)
    ax[2].set(xlabel='PMT distance from vertex (m)', ylabel='calibration residual',
              title='time residual vs distance')
    ax[2].grid(alpha=.3); ax[2].legend(fontsize=8, frameon=False)
    fig.suptitle(f'Residual maps at truth — {cfg["name"]}')
    fig.tight_layout()
    tag = cfg['name']
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'residual_maps_{tag}.{ext}', dpi=140)

    with h5py.File(out_dir / f'residual_maps_{tag}.h5', 'w') as h5:
        h5.attrs['config_json'] = json.dumps(cfg)
        h5.attrs['finished'] = datetime.now().isoformat()
        for k, v in (('ca_centers', cac), ('rd_centers', rdc), ('charge_ratio', ratio),
                     ('dt_vs_ca', dt_ca_med), ('dt_vs_rd', dt_rd_med),
                     ('pit_vs_ca', pit_ca), ('pit_vs_rd', pit_rd),
                     ('q_sum', q_sum), ('mu_sum', mu_sum)):
            h5.create_dataset(k, data=v)
    print(f'wrote residual_maps_{tag}.png/h5', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
