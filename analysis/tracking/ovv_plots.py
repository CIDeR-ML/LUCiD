#!/usr/bin/env python3
"""Performance plots in the style of good_notebooks/optimization_vs_variables.ipynb (c603ed1),
fed by the CURRENT pipeline outputs (run_study h5 files).

Two modes:
  --mode iters --h5 <run.h5>          : median / p68 / p90 of the four metrics vs GN iteration
                                        (winning-seed trajectories; E from the trajectory, i.e.
                                        pre-E-refit — the refit only moves the final readout).
  --mode nrays --base <dir-prefix>    : final-fit bootstrap p68 (90% CI) vs number of rays over
                                        all COMPLETED nrays_scan_<tag> runs found, with the
                                        notebook's exp-like weighted fits + timing panel.

Metrics and style mirror the notebook exactly: Pos. error (cm), Dir. error (deg),
t0 error (ns), Mom. error (%) [dE -> dp via E_tot=KE+m_mu, p=sqrt(E_tot^2-m_mu^2)];
serif font, vertical panel stack, navy points, cornflowerblue fits, bootstrap
(percentile=68, n_bootstrap=1000, ci_level=90, seed=42).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

M_MU = 105.658
TAGS = ['5k', '10k', '25k', '50k', '100k', '150k', '250k', '500k', '1M']
NRAYS = {'5k': 5e3, '10k': 1e4, '25k': 2.5e4, '50k': 5e4, '100k': 1e5,
         '150k': 1.5e5, '250k': 2.5e5, '500k': 5e5, '1M': 1e6}


def mom_err_pct(dE, true_E):
    """Notebook compute_momentum_errors: dE -> momentum error %, KE convention."""
    E_tot = true_E + M_MU
    p = np.sqrt(E_tot**2 - M_MU**2)
    return (E_tot / p) * (np.abs(dE) / E_tot) * 100.0


def bootstrap_p68(data, n_bootstrap=1000, ci_level=90, rng=None):
    rng = rng or np.random.default_rng(42)
    boots = np.array([np.percentile(rng.choice(data, size=len(data), replace=True), 68)
                      for _ in range(n_bootstrap)])
    a = (100 - ci_level) / 2
    est = np.percentile(data, 68)
    return est, est - np.percentile(boots, a), np.percentile(boots, 100 - a) - est


def exp_like(x, a, b, c):
    return a * (b ** x) + c


def power_like(x, a, b, c):
    """y = a * x^b + c -- MC-statistics scaling (b ~ -0.5) with a floor."""
    return a * np.power(x, b) + c


def _style():
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.size'] = 12


def event_metric_curves(h5):
    """(n_events, niters+1, 4): pos cm, dir deg, |t0| ns, mom %  along winning trajectories."""
    from analysis.tracking.p68_evolution import _dirs_from_traj
    import h5py
    out = []
    with h5py.File(h5, 'r') as f:
        for k in sorted(f['events']):
            ev = f['events'][k]
            tr = ev['traj_win'][:]; th = ev['truth_vec9'][:]; d = ev['tdir'][:]
            pos = np.linalg.norm(tr[:, 1:4] - th[1:4], axis=1) * 100
            ddeg = np.degrees(np.arccos(np.clip(_dirs_from_traj(tr) @ d, -1, 1)))
            t0 = np.abs(tr[:, 8] - th[8])
            mom = mom_err_pct(tr[:, 0] - th[0], float(th[0]))
            out.append(np.stack([pos, ddeg, t0, mom], 1))
    return np.stack(out)


def mode_iters(args):
    """create_convergence_figure from track_optimization_visualization.ipynb (c603ed1 cell 3):
    p68 solid navy + p90 dashed cornflowerblue vs iteration, LINEAR y with fixed per-metric
    ranges, and a right-hand horizontal histogram of the final-iteration values (pink, black
    edges) with p68/p90/worst-event lines."""
    _style()
    import matplotlib
    matplotlib.rcParams['font.size'] = 10
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    C = event_metric_curves(args.h5)                     # (N, T, 4); pos in cm
    C[:, :, 0] /= 100.0                                  # notebook plots Pos in m
    it = np.arange(C.shape[1])

    metrics = [(0, 'Pos. Error (m)', 1.5), (1, 'Dir. Error (°)', 15.0),
               (2, 't0 Error (ns)', 5.0), (3, 'Mom. Error (%)', 10.0)]
    col_p68, col_p90 = 'navy', 'cornflowerblue'
    lw = 1.5

    fig, axs = plt.subplots(4, 1, figsize=(4, 8), sharex=True)
    hist_axes = []
    for ax in axs:
        bbox = ax.get_position()
        hist_axes.append(fig.add_axes([bbox.x1 + 0.001, bbox.y0, 0.22, bbox.height]))

    for (j, ylabel, max_y), ax, hist_ax in zip(metrics, axs, hist_axes):
        p68 = np.percentile(C[:, :, j], 68, axis=0)
        p90 = np.percentile(C[:, :, j], 90, axis=0)
        worst = np.percentile(C[:, :, j], 99.99, axis=0)
        ax.plot(it, p68, color=col_p68, lw=lw, linestyle='-', label='68%')
        ax.plot(it, p90, color=col_p90, lw=lw, linestyle='--', label='90%')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, len(it) - 1)
        ax.set_ylim(0, max_y)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=False, ncol=3)
        if j in (0, 2):
            ax.yaxis.set_major_formatter(FormatStrFormatter('%6.2f'))
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%6.1f'))

        last = C[:, -1, j]
        hist_ax.hist(last, bins=18, range=[0, max_y], orientation='horizontal',
                     color='pink', alpha=0.4, edgecolor='black', linewidth=0.5)
        hist_ax.axhline(y=p68[-1], color=col_p68, lw=lw, linestyle='-')
        hist_ax.axhline(y=p90[-1], color=col_p90, lw=lw, linestyle='--')
        hist_ax.axhline(y=worst[-1], color='gray', lw=1, linestyle='-')
        hist_ax.text(0.9, worst[-1], 'worst event', color='gray', ha='right', va='top',
                     fontsize=8, transform=hist_ax.get_yaxis_transform())
        hist_ax.set_yticks([]); hist_ax.set_xticks([])
        hist_ax.set_ylim(ax.get_ylim())
        try:
            import seaborn as sns
            sns.despine(ax=hist_ax, left=True, bottom=False, right=False, top=False)
        except ImportError:
            hist_ax.spines['left'].set_visible(False)

    axs[-1].set_xlabel('Iteration')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base = out / 'convergence_all_metrics'
    for ext in ('pdf', 'png'):
        plt.savefig(f'{base}.{ext}', bbox_inches='tight', dpi=150)
    print(f'Saved figure to: {base}.pdf (+png)  (N={C.shape[0]} events)')


def mode_nrays(args):
    _style()
    import h5py
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    from scipy.optimize import curve_fit

    rows = []                     # per completed config
    for tag in TAGS:
        if args.max_rays is not None and NRAYS[tag] > args.max_rays:
            continue
        h5p = Path(f'{args.base}{tag}') / f'nrays_scan_{tag}.h5'
        logd = Path(f'{args.base}{tag}') / 'logs'
        if not h5p.exists():
            continue
        try:
            f = h5py.File(h5p, 'r')
        except (BlockingIOError, OSError):
            print(f'{tag}: h5 locked (job still running) - skipped')
            continue
        with f:
            fe = np.stack([f['events'][k]['fit_err'][:] for k in sorted(f['events'])])
            tE = np.array([float(f['events'][k]['truth_vec9'][0]) for k in sorted(f['events'])])
            secs = np.array([float(f['events'][k].attrs['seconds']) for k in sorted(f['events'])])
        rows.append(dict(tag=tag, x=NRAYS[tag], pos=fe[:, 0], dir=fe[:, 1],
                         t0=np.abs(fe[:, 3]), mom=mom_err_pct(fe[:, 2], tE.mean()),
                         t_mean=float(np.mean(secs[1:])), t_std=float(np.std(secs[1:]))))  # drop ev0 (JIT compile)
        print(f'{tag}: n={len(fe)} events loaded')
    if not rows:
        print('no completed configs found'); return

    metrics = [('pos', 'Pos. error (cm)'), ('dir', 'Dir. error (°)'),
               ('t0', 't$_0$ error (ns)'), ('mom', 'Mom. error (%)')]
    rng = np.random.default_rng(42)
    x = np.array([r['x'] for r in rows]) / 1000.0

    fig, axes = plt.subplots(5, 1, figsize=(6, 12.5), sharex=True)
    plt.subplots_adjust(hspace=0.05)
    for ax, (key, lab) in zip(axes[:4], metrics):
        y, lo, hi = [], [], []
        for r in rows:
            e, el, eh = bootstrap_p68(r[key], rng=rng)
            y.append(e); lo.append(el); hi.append(eh)
        y = np.array(y); lo = np.array(lo); hi = np.array(hi)
        ax.errorbar(x, y, yerr=[lo, hi], fmt='o', capsize=4, color='navy', label=f'{lab} p68')
        sigma = np.clip(0.5 * (lo + hi), 1e-12, None)
        try:
            popt, _ = curve_fit(power_like, x, y, sigma=sigma, absolute_sigma=True,
                                p0=[(np.nanmax(y) - np.nanmin(y)) * np.sqrt(x.min()), -0.5,
                                    np.nanmin(y)],
                                bounds=([0, -2.0, 0], [np.inf, 0, np.inf]), maxfev=20000)
            xf = np.linspace(x.min(), x.max(), 200)
            ax.plot(xf, power_like(xf, *popt), '-', color='cornflowerblue', lw=2)
            a, b, c = popt
            ax.text(0.5, 0.9, f'$y = {a:.1f}\\,x^{{{b:.2f}}} + {c:.2f}$',
                    transform=ax.transAxes, fontsize=10, color='cornflowerblue',
                    ha='left', va='top')
        except RuntimeError:
            print(f'fit failed for {key}')
        ax.set_ylabel(lab); ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ax_t = axes[-1]
    ax_t.errorbar(x, [r['t_mean'] for r in rows], yerr=[r['t_std'] for r in rows],
                  fmt='o-', capsize=4, color='tab:red', label='Total')
    ax_t.set_ylabel('Time per event (s)'); ax_t.grid(True, alpha=0.3)
    ax_t.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax_t.legend(frameon=False, loc='lower right')
    ax_t.set_ylim(0); ax_t.set_xlim(0)
    axes[-1].set_xlabel('Number of Rays ($\\times10^3$)')
    fig.align_ylabels(axes)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base = out / 'tracking_performance_vs_nrays'
    for ext in ('pdf', 'png'):
        plt.savefig(f'{base}.{ext}', bbox_inches='tight', dpi=150)
    print(f'Saved figure to: {base}.pdf (+png)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', required=True, choices=['iters', 'nrays'])
    ap.add_argument('--h5', help='run_study h5 (iters mode)')
    ap.add_argument('--base', help='nrays_scan dir prefix, e.g. .../nrays_scan_ (nrays mode)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-rays', type=float, default=None,
                    help='exclude configs above this ray count (nrays mode)')
    args = ap.parse_args()
    if args.mode == 'iters':
        mode_iters(args)
    else:
        mode_nrays(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
