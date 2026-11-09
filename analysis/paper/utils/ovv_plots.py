#!/usr/bin/env python3
"""Performance plots in the style of good_notebooks/optimization_vs_variables.ipynb (c603ed1),
fed by the CURRENT pipeline outputs (run_study h5 files).

The plot modes, called by the paper figures through ``plotting.py``:
  mode_iters           median / p68 / p90 of the four metrics vs GN iteration (winning-seed
                       trajectories; E from the trajectory, i.e. pre-E-refit -- the refit only
                       moves the final readout);
  mode_nrays_combined  muon + electron final-fit bootstrap p68 (90% CI) vs number of rays, with
                       the notebook's weighted fits and a timing panel;
  mode_energy, mode_sensors  the same metrics vs energy and vs number of sensors.

Metrics and style mirror the notebook exactly: Pos. error (cm), Dir. error (deg),
t0 error (ns), Mom. error (%) [dE -> dp via E_tot=KE+m_mu, p=sqrt(E_tot^2-m_mu^2)];
serif font, vertical panel stack, navy points, cornflowerblue fits, bootstrap
(percentile=68, n_bootstrap=1000, ci_level=90, seed=42).
"""
from pathlib import Path

import numpy as np

from lucid.fitting import bootstrap_ci
from lucid.pdg import PDG_MASS_MEV

M_MU = PDG_MASS_MEV[13]
NRAYS = {'5k': 5e3, '10k': 1e4, '25k': 2.5e4, '50k': 5e4, '100k': 1e5,
         '150k': 1.5e5, '250k': 2.5e5, '500k': 5e5, '1M': 1e6}


def mom_err_pct(dE, true_E, mass=M_MU):
    """Momentum error % from a kinetic-energy error ``dE`` at kinetic energy ``true_E`` (MeV):
    p^2 = E_tot^2 - m^2 gives dp = E_tot dE / p, so dp/p = E_tot |dE| / p^2."""
    E_tot = true_E + mass
    p2 = E_tot**2 - mass**2
    return E_tot * np.abs(dE) / p2 * 100.0


def p68_errorbar(data, ci_level):
    """The 68% containment of ``data`` with its bootstrap ``ci_level``% interval as error-bar
    lengths ``(p68, p68 - lo, hi - p68)``: 1000 resamples, seeded 42 for every point."""
    def p68(a):
        return np.percentile(a, 68)
    lo, hi = bootstrap_ci(data, p68, ci=ci_level, n_boot=1000, seed=42)
    est = p68(data)
    return est, est - lo, hi - est


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


def find_study(run_dir):
    """The study file of a run dir: ``<dir>/<dir name>.h5``, else the dir's only ``.h5``."""
    d = Path(run_dir)
    named = d / f'{d.name}.h5'
    if named.exists():
        return named
    cands = sorted(d.glob('*.h5'))
    if len(cands) != 1:
        raise FileNotFoundError(f'{d}: expected one study .h5, found {len(cands)}')
    return cands[0]


def open_study(path):
    """An open, finished study file. Raises if it is missing, still being written (locked), never
    finished, or holds fewer events than it was asked for: a figure is never drawn from a
    partial run."""
    import h5py
    f = h5py.File(path, 'r')
    a = f.attrs
    n_req = int(a['n_events_requested']) if 'n_events_requested' in a else None
    complete = ('finished' in a and n_req is not None and int(a['n_events_done']) == n_req
                and len(f['events']) == n_req
                and not ('n_events_failed' in a and int(a['n_events_failed'])))
    if not complete:
        msg = (f'{path}: not a finished, complete study run (finished={"finished" in a}, '
               f'events {len(f["events"])}/{n_req})')
        f.close()
        raise ValueError(msg)
    return f


def event_metric_curves(h5):
    """(n_events, niters+1, 4): pos cm, dir deg, |t0| ns, mom %  along winning trajectories."""
    from lucid.fitting import track_errors
    out = []
    from analysis.paper.utils.studies import mass as particle_mass
    with open_study(h5) as f:
        mass = particle_mass(str(f.attrs['particle']))
        for k in sorted(f['events']):
            ev = f['events'][k]
            th = ev['truth_vec9'][:]
            pos, ddeg, dE, dt0 = track_errors(ev['traj_win'][:], th, ev['tdir'][:]).T
            out.append(np.stack([pos, ddeg, np.abs(dt0), mom_err_pct(dE, float(th[0]), mass)], 1))
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
        # The worst event can exceed the panel's fixed range (e.g. position and direction on
        # the muon sample), and a line drawn outside the axes silently vanishes. The ranges are
        # deliberate: widening them would compress the region the figure exists to show. So the
        # marker is clamped to the top edge and its label carries the true value.
        _w = float(worst[-1])
        _off = _w > max_y
        hist_ax.axhline(y=min(_w, max_y), color='gray', lw=1,
                        linestyle=(0, (2, 2)) if _off else '-')
        hist_ax.text(0.9, min(_w, max_y),
                     f'worst event {_w:.3g} (off scale)' if _off else 'worst event',
                     color='gray', ha='right', va='top', fontsize=8,
                     transform=hist_ax.get_yaxis_transform())
        if _off:
            # An overflow arrow, so the clamp cannot be mistaken for the value itself.
            hist_ax.plot([0.5], [max_y], marker='^', color='gray', ms=5, clip_on=False,
                         transform=hist_ax.get_yaxis_transform())
        hist_ax.set_yticks([]); hist_ax.set_xticks([])
        hist_ax.set_ylim(ax.get_ylim())
        hist_ax.spines['left'].set_visible(False)
        hist_ax.yaxis.set_ticks_position('right')

    axs[-1].set_xlabel('Iteration')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base = out / 'convergence_all_metrics'
    for ext in ('pdf', 'png'):
        plt.savefig(f'{base}.{ext}', bbox_inches='tight', dpi=150)
    print(f'Saved figure to: {base}.pdf (+png)  (N={C.shape[0]} events)')


def _load_nrays_rows(base, tags, max_events, mass):
    """Load per-config resolution + timing rows for one nrays base prefix."""
    rows = []
    for tag in tags:
        with open_study(find_study(f'{base}{tag}')) as f:
            _ks = sorted(f['events'])
            if max_events is not None:
                _ks = _ks[:max_events]
            fe = np.stack([f['events'][k]['fit_err'][:] for k in _ks])
            tE = np.array([float(f['events'][k]['truth_vec9'][0]) for k in _ks])
            secs = np.array([float(f['events'][k].attrs['seconds']) for k in _ks])
        rows.append(dict(tag=tag, x=NRAYS[tag], pos=fe[:, 0], dir=fe[:, 1],
                         t0=np.abs(fe[:, 3]), mom=mom_err_pct(fe[:, 2], tE.mean(), mass=mass),
                         t_mean=float(np.mean(secs[1:])), t_std=float(np.std(secs[1:]))))
        print(f'{tag}: n={len(fe)} events loaded')
    return rows


def mode_nrays_combined(args):
    """Muon + electron overlaid vs nrays. Muon = navy/cornflowerblue, electron = red/orange.
    Timing panel = MUON only, dark gray. --base is muon, --base-el is electron."""
    _style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    from scipy.optimize import curve_fit

    mu_rows = _load_nrays_rows(args.base, args.tags, args.max_events, args.mass)
    el_rows = _load_nrays_rows(args.base_el, args.tags, args.max_events, args.mass_el)

    metrics = [('pos', 'Pos. error (cm)'), ('dir', 'Dir. error (°)'),
               ('t0', 't$_0$ error (ns)'), ('mom', 'Mom. error (%)')]
    fig, axes = plt.subplots(5, 1, figsize=(6, 12.5), sharex=True)
    plt.subplots_adjust(hspace=0.05)

    def draw(ax, rows, key, pt_c, fit_c, fit='power'):
        x = np.array([r['x'] for r in rows]) / 1000.0
        y, lo, hi = [], [], []
        for r in rows:
            e, el_, eh = p68_errorbar(r[key], 90)
            y.append(e); lo.append(el_); hi.append(eh)
        y = np.array(y); lo = np.array(lo); hi = np.array(hi)
        ax.errorbar(x, y, yerr=[lo, hi], fmt='o', capsize=4, color=pt_c)
        sigma = np.clip(0.5 * (lo + hi), 1e-12, None)
        xf = np.linspace(x.min(), x.max(), 200)
        try:
            if fit == 'const':
                c = float(np.sum(y / sigma**2) / np.sum(1.0 / sigma**2))  # weighted mean
                ax.plot(xf, np.full_like(xf, c), '-', color=fit_c, lw=2)
                return ('const', c)
            popt, _ = curve_fit(power_like, x, y, sigma=sigma, absolute_sigma=True,
                                p0=[(np.nanmax(y) - np.nanmin(y)) * np.sqrt(x.min()), -0.5, np.nanmin(y)],
                                bounds=([0, -2.0, 0], [np.inf, 0, np.inf]), maxfev=20000)
            ax.plot(xf, power_like(xf, *popt), '-', color=fit_c, lw=2)
            return ('power', popt)
        except RuntimeError:
            print(f'fit failed for {key}'); return None

    def eqtext(res):
        if res is None:
            return None
        kind, p = res
        if kind == 'const':
            return f'{p:.2f}'
        a, b, c = p
        return f'{a:.1f}\\,x^{{{b:.2f}}} + {c:.2f}'

    FIT = {'pos': 'power', 'dir': 'power', 't0': 'power', 'mom': 'const'}
    YLIM = {'pos': (9, 65), 'dir': (0, 6.5), 't0': (0, 1.8), 'mom': (0, 4.5)}
    for ax, (key, lab) in zip(axes[:4], metrics):
        pm = draw(ax, mu_rows, key, 'navy', 'cornflowerblue', fit=FIT[key])
        pe = draw(ax, el_rows, key, 'red', 'orange', fit=FIT[key])
        tm, te = eqtext(pm), eqtext(pe)
        if tm is not None:
            ax.text(0.62, 0.92, f'$\\mu:\\ {tm}$',
                    transform=ax.transAxes, fontsize=9, color='cornflowerblue', ha='left', va='top')
        if te is not None:
            ax.text(0.62, 0.80, f'$e:\\ {te}$',
                    transform=ax.transAxes, fontsize=9, color='orange', ha='left', va='top')
        ax.set_ylabel(lab); ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_ylim(*YLIM[key])

    # timing: muon cornflowerblue, electron orange (same orange as fit lines above)
    ax_t = axes[-1]
    xm = np.array([r['x'] for r in mu_rows]) / 1000.0
    xe = np.array([r['x'] for r in el_rows]) / 1000.0
    tm = np.array([r['t_mean'] for r in mu_rows]); te = np.array([r['t_mean'] for r in el_rows])
    ax_t.errorbar(xm, tm, yerr=[r['t_std'] for r in mu_rows],
                  fmt='o', capsize=4, color='navy', label='Muon')
    ax_t.errorbar(xe, te, yerr=[r['t_std'] for r in el_rows],
                  fmt='o', capsize=4, color='red', label='Electron')
    xf = np.linspace(0, max(xm.max(), xe.max()), 200)
    mm, bm = np.polyfit(xm, tm, 1); me, be = np.polyfit(xe, te, 1)
    ax_t.plot(xf, mm * xf + bm, '-', color='cornflowerblue', lw=2)
    ax_t.plot(xf, me * xf + be, '-', color='orange', lw=2)
    ax_t.text(0.03, 0.92, f'$\\mu:\\ {mm:.3f}\\,x + {bm:.2f}$',
              transform=ax_t.transAxes, fontsize=9, color='cornflowerblue', ha='left', va='top')
    ax_t.text(0.03, 0.80, f'$e:\\ {me:.3f}\\,x + {be:.2f}$',
              transform=ax_t.transAxes, fontsize=9, color='orange', ha='left', va='top')
    ax_t.set_ylabel('Time per event (s)'); ax_t.grid(True, alpha=0.3)
    ax_t.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax_t.legend(frameon=False, loc='lower right')
    ax_t.set_ylim(0); ax_t.set_xlim(0)
    axes[-1].set_xlabel('Number of Rays ($\\times10^3$)')

    fig.align_ylabels(axes)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base = out / 'tracking_performance_vs_nrays_combined'
    for ext in ('pdf', 'png'):
        plt.savefig(f'{base}.{ext}', bbox_inches='tight', dpi=150)
    print(f'Saved figure to: {base}.pdf (+png)')


def mode_energy(args):
    """Notebook cell 5: tracking_performance_vs_energy — x = E (MeV), ci_level=68,
    exp fits for pos/dir, CONSTANT fits for t0/mom, the notebook's ylims, no timing panel."""
    _style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    from scipy.optimize import curve_fit

    def constant(x, c):
        return np.full_like(np.asarray(x, dtype=float), c)

    rows = []
    for E in args.energies:
        with open_study(find_study(f'{args.base}{E}')) as f:
            fe = np.stack([f['events'][k]['fit_err'][:] for k in sorted(f['events'])])
        rows.append(dict(x=E, pos=fe[:, 0], dir=fe[:, 1], t0=np.abs(fe[:, 3]),
                         mom=mom_err_pct(fe[:, 2], float(E))))
        print(f'{E}: n={len(fe)}')

    metrics = [('pos', 'Pos. error (cm)', 'exp'), ('dir', 'Dir. error (°)', 'exp'),
               ('t0', 't$_0$ error (ns)', 'const'), ('mom', 'Mom. error (%)', 'const')]
    ylims = {'pos': (10, 35), 'dir': (0., 2.9), 't0': (0, 1.6), 'mom': (0, 4.4)}
    x = np.array([r['x'] for r in rows], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(6, 10), sharex=True)
    plt.subplots_adjust(hspace=0.05)
    for ax, (key, lab, ftype) in zip(axes, metrics):
        y, lo, hi = [], [], []
        for r in rows:
            e, el, eh = p68_errorbar(r[key], 68)
            y.append(e); lo.append(el); hi.append(eh)
        y = np.array(y); lo = np.array(lo); hi = np.array(hi)
        ax.errorbar(x, y, yerr=[lo, hi], fmt='o', capsize=4, color='navy', label=f'{lab} p68')
        sigma = np.clip(0.5 * (lo + hi), 1e-12, None)
        xf = np.linspace(x.min(), x.max(), 200)
        try:
            if ftype == 'const':
                popt, _ = curve_fit(constant, x, y, sigma=sigma, absolute_sigma=True,
                                    p0=[np.nanmean(y)], maxfev=10000)
                ax.plot(xf, constant(xf, *popt), '-', color='cornflowerblue', lw=2)
                ax.text(0.55, 0.9, f'$y = {popt[0]:.2f}$', transform=ax.transAxes,
                        fontsize=10, color='cornflowerblue', ha='left', va='top')
            else:
                popt, _ = curve_fit(exp_like, x, y, sigma=sigma, absolute_sigma=True,
                                    p0=[np.nanmax(y) - np.nanmin(y), 0.998, np.nanmin(y)],
                                    bounds=([-np.inf, 0.0, 0.0], [np.inf, 1.0, np.inf]),
                                    maxfev=20000)
                ax.plot(xf, exp_like(xf, *popt), '-', color='cornflowerblue', lw=2)
                a, b, c = popt
                ax.text(0.5, 0.9, f'$y = {a:.2f}\\times\\,{b:.4f}^x + {c:.2f}$',
                        transform=ax.transAxes, fontsize=10, color='cornflowerblue',
                        ha='left', va='top')
        except RuntimeError:
            print(f'fit failed for {key}')
        yl = ylims.get(key)
        if yl:
            top = max(yl[1], float(np.nanmax(y + hi)) * 1.08)   # never clip a data point
            ax.set_ylim(yl[0] if float(np.nanmin(y - lo)) > yl[0] else 0, top)
        ax.set_ylabel(lab); ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axes[-1].set_xlabel('Energy (MeV)')
    fig.align_ylabels(axes)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base = out / 'tracking_performance_vs_energy'
    for ext in ('pdf', 'png'):
        plt.savefig(f'{base}.{ext}', bbox_inches='tight', dpi=150)
    print(f'Saved figure to: {base}.pdf (+png)')


def mode_sensors(args):
    """Notebook cell 6: detector_perf_vs_num_sensors — x = PLACED sensors (x10^3),
    ci_level=68, no timing panel, the notebook's exact ylims."""
    _style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    from scipy.optimize import curve_fit

    rows = []
    for N in args.sensors:
        with open_study(find_study(f'{args.base}{N}')) as f:
            fe = np.stack([f['events'][k]['fit_err'][:] for k in sorted(f['events'])])
            tE = np.array([float(f['events'][k]['truth_vec9'][0]) for k in sorted(f['events'])])
            placed = int(f.attrs['n_sensors'])
        rows.append(dict(x=placed, pos=fe[:, 0], dir=fe[:, 1], t0=np.abs(fe[:, 3]),
                         mom=mom_err_pct(fe[:, 2], tE.mean())))
        print(f'{N}: placed={placed}, n={len(fe)}')

    metrics = [('pos', 'Pos. error (cm)'), ('dir', 'Dir. error (°)'),
               ('t0', 't$_0$ error (ns)'), ('mom', 'Mom. error (%)')]
    ylims = {'pos': (10, 40), 'dir': (0.4, 1.8), 't0': (0, 1.2), 'mom': (0, 4.5)}
    x = np.array([r['x'] for r in rows]) / 1000.0

    fig, axes = plt.subplots(4, 1, figsize=(6, 10), sharex=True)
    plt.subplots_adjust(hspace=0.05)
    for ax, (key, lab) in zip(axes, metrics):
        y, lo, hi = [], [], []
        for r in rows:
            e, el, eh = p68_errorbar(r[key], 68)
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
        if key in ylims:
            ax.set_ylim(ylims[key])
        ax.set_ylabel(lab); ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axes[-1].set_xlabel('Number of Sensors ($\\times10^3$)')
    fig.align_ylabels(axes)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base = out / 'detector_perf_vs_num_sensors'
    for ext in ('pdf', 'png'):
        plt.savefig(f'{base}.{ext}', bbox_inches='tight', dpi=150)
    print(f'Saved figure to: {base}.pdf (+png)')
