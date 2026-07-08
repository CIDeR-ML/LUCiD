#!/usr/bin/env python3
"""Measure the two photon-level quantities the ray model does NOT simulate, vs energy.

Per energy (PhotonSim muon files, first --n-events each), per photon:
  rho = transverse distance of the emission point from the track axis (m).
        SIREN rays start ON the axis (rho = 0 exactly) -> the full data distribution
        is unmodeled. Suspect for the charge-term longitudinal bias sweep.
  dt  = t_photon - predict_t0(s, E): emission time relative to the ray model's own
        fitted baseline (d/c + stretched-exp delay). Residual tail = unmodeled late
        light. Suspect for the time-term backward pull growing with E.

Reports quantiles + tail fractions vs E, early/late-track splits, and writes full
histograms to h5 + a 4-panel summary figure.

    python analysis/tracking/measure_raygaps.py --output OUT [--n-events 100]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATA_DIR = Path('/sdf/data/neutrino/cjesus/CIDER/ROOT_files/LARGE_files/water/mu-')
ENERGIES = list(range(400, 2001, 100))
C_MM_NS = 299.792


def find_t0_params(d, path=''):
    """Recursively locate the stretched_exp_delay t0 block ({'A','lambda','beta'})."""
    if isinstance(d, dict):
        if {'A', 'lambda', 'beta'} <= set(d.keys()):
            return d, path
        for k, v in d.items():
            r = find_t0_params(v, f'{path}/{k}')
            if r is not None:
                return r
    return None


def cubic_log10(coeffs, E):
    x = np.log10(E)
    return coeffs[0] + coeffs[1] * x + coeffs[2] * x**2 + coeffs[3] * x**3


def predict_t0_np(d_mm, E, t0p):
    A = 10.0 ** cubic_log10(np.asarray(t0p['A']['log10_poly_logE'], float), E)
    lam = 10.0 ** cubic_log10(np.asarray(t0p['lambda']['log10_poly_logE'], float), E)
    beta = cubic_log10(np.asarray(t0p['beta']['poly_logE'], float), E)
    arg = np.power(np.clip(d_mm / lam, 1e-12, None), beta)
    return d_mm / C_MM_NS + A * (np.expm1(arg))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', required=True)
    ap.add_argument('--n-events', type=int, default=100)
    args = ap.parse_args()

    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lucid.sources.event_io import read_photon_data_from_photonsim

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    from lucid.utils import unpack_t0_params
    a_c, l_c, b_c = unpack_t0_params('muon', 'water')
    t0p = {'A': {'log10_poly_logE': list(a_c)},
           'lambda': {'log10_poly_logE': list(l_c)},
           'beta': {'poly_logE': list(b_c)}}
    t0path = 'lucid.utils.unpack_t0_params(muon, water)'
    print(f't0 baseline coeffs loaded via {t0path}', flush=True)

    rho_bins = np.linspace(0, 1.0, 101)          # m
    dt_bins = np.linspace(-2.0, 8.0, 201)        # ns
    summary = []
    H_rho, H_dt = [], []
    for E in ENERGIES:
        root = DATA_DIR / f'{E}MeV_500events.root'
        if not root.exists():
            print(f'{E}: no file', flush=True); continue
        rho_l, dt_l, s_l = [], [], []
        for ev in range(args.n_events):
            r = read_photon_data_from_photonsim(str(root), ev)
            O = np.asarray(r['photon_origins'], float)           # cm, gun frame (+z track)
            t = np.asarray(r['photon_times'], float)             # ns
            s_mm = O[:, 2] * 10.0
            rho = np.hypot(O[:, 0], O[:, 1]) / 100.0             # m
            dt = t - predict_t0_np(np.clip(s_mm, 0, None), float(E), t0p)
            rho_l.append(rho.astype(np.float32))
            dt_l.append(dt.astype(np.float32))
            s_l.append((O[:, 2] / 100.0).astype(np.float32))
        rho = np.concatenate(rho_l); dt = np.concatenate(dt_l); s = np.concatenate(s_l)
        smax = np.quantile(s, 0.999)
        early = s < 0.25 * smax; late = s > 0.75 * smax
        row = dict(
            E=E, n_phot=len(rho),
            rho_med=np.median(rho), rho_p68=np.quantile(rho, 0.68),
            rho_p95=np.quantile(rho, 0.95),
            rho_f10=float((rho > 0.10).mean()), rho_f25=float((rho > 0.25).mean()),
            rho_early=float(np.median(rho[early])), rho_late=float(np.median(rho[late])),
            dt_p05=np.quantile(dt, 0.05), dt_med=np.median(dt),
            dt_p68=np.quantile(dt, 0.68), dt_p95=np.quantile(dt, 0.95),
            dt_f05=float((dt > 0.5).mean()), dt_f1=float((dt > 1.0).mean()),
            dt_early=float(np.median(dt[early])), dt_late=float(np.median(dt[late])),
        )
        summary.append(row)
        H_rho.append(np.histogram(rho, rho_bins)[0] / args.n_events)
        H_dt.append(np.histogram(dt, dt_bins)[0] / args.n_events)
        print(f"{E:5d}: rho med/p68/p95 {row['rho_med']*100:5.1f}/{row['rho_p68']*100:5.1f}/"
              f"{row['rho_p95']*100:6.1f} cm  f(>10cm)={row['rho_f10']:.3f} f(>25cm)={row['rho_f25']:.3f} "
              f"early/late {row['rho_early']*100:.1f}/{row['rho_late']*100:.1f} | "
              f"dt p05/med/p68/p95 {row['dt_p05']:+5.2f}/{row['dt_med']:+5.2f}/{row['dt_p68']:+5.2f}/"
              f"{row['dt_p95']:+6.2f} ns  f(>1ns)={row['dt_f1']:.3f} early/late {row['dt_early']:+.2f}/"
              f"{row['dt_late']:+.2f}", flush=True)

    Ev = np.array([r['E'] for r in summary], float)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    a = ax[0, 0]
    a.plot(Ev, [r['rho_med'] * 100 for r in summary], 'o-', label='median')
    a.plot(Ev, [r['rho_p68'] * 100 for r in summary], 's-', label='p68')
    a.plot(Ev, [r['rho_p95'] * 100 for r in summary], '^-', label='p95')
    a.set(xlabel='E (MeV)', ylabel=r'$\rho$ transverse offset (cm)',
          title='data emission transverse offset (SIREN rays: 0)')
    a.grid(alpha=.3); a.legend(frameon=False)
    a = ax[0, 1]
    a.plot(Ev, [r['rho_f10'] for r in summary], 'o-', label=r'$\rho>10$ cm')
    a.plot(Ev, [r['rho_f25'] for r in summary], 's-', label=r'$\rho>25$ cm')
    a.set(xlabel='E (MeV)', ylabel='photon fraction', title='off-axis tail fractions')
    a.grid(alpha=.3); a.legend(frameon=False)
    a = ax[1, 0]
    a.plot(Ev, [r['dt_p05'] for r in summary], 'v-', label='p05 (early edge)')
    a.plot(Ev, [r['dt_med'] for r in summary], 'o-', label='median')
    a.plot(Ev, [r['dt_p68'] for r in summary], 's-', label='p68')
    a.plot(Ev, [r['dt_p95'] for r in summary], '^-', label='p95')
    a.axhline(0, c='k', lw=.8)
    a.set(xlabel='E (MeV)', ylabel=r'$\Delta t$ vs predict_t0 (ns)',
          title='emission-time residual vs the ray model baseline')
    a.grid(alpha=.3); a.legend(frameon=False)
    a = ax[1, 1]
    a.plot(Ev, [r['dt_f1'] for r in summary], 'o-', label=r'$\Delta t>1$ ns')
    a.plot(Ev, [r['dt_f05'] for r in summary], 's-', label=r'$\Delta t>0.5$ ns')
    a.plot(Ev, [r['dt_late'] for r in summary], 'd-', label='median, last quarter of track')
    a.axhline(0, c='k', lw=.8)
    a.set(xlabel='E (MeV)', ylabel='fraction / ns', title='late-light tail vs E')
    a.grid(alpha=.3); a.legend(frameon=False)
    fig.suptitle('Unmodeled photon-level physics vs E (data vs ray-model assumptions)')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'raygaps.{ext}', dpi=140)

    with h5py.File(out_dir / 'raygaps.h5', 'w') as h5:
        h5.attrs['n_events'] = args.n_events
        h5.attrs['t0_params_path'] = t0path
        h5.attrs['t0_params_json'] = json.dumps(t0p)
        h5.attrs['finished'] = datetime.now().isoformat()
        h5.create_dataset('rho_bins', data=rho_bins)
        h5.create_dataset('dt_bins', data=dt_bins)
        h5.create_dataset('H_rho', data=np.stack(H_rho))
        h5.create_dataset('H_dt', data=np.stack(H_dt))
        for key in summary[0]:
            h5.create_dataset(key, data=np.array([r[key] for r in summary]))
    print(f'wrote {out_dir}/raygaps.png and raygaps.h5', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
