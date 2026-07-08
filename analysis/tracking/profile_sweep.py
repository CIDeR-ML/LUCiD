#!/usr/bin/env python3
"""Emission-calibration sweep vs energy: is the dE bias + longitudinal vertex bias SIREN-side?

Per energy E (PhotonSim 500-event muon files, first --n-events each):
  DATA : mean total emitted photons/event, longitudinal centroid <s>_data (m).
  SIREN: N_photons(E) from the emitter context, <s>_SIREN from intensity-weighted rays.

Predictions tested against the measured escan2 biases:
  dE_pred(E)   = (r(E) - 1) * nphot(E)/nphot'(E),  r = data_total / N_photons(E)
                 (E the fit must move to reproduce the data total through the model yield)
  vtx_pred(E)  = <s>_data - <s>_SIREN  (fit slides vertex forward when model centroid is upstream)

Measured references are recomputed from the escan2_<E> h5s (dEm and mean longitudinal
vertex error along the true direction).

    python analysis/tracking/profile_sweep.py --output OUT [--n-events 100] [--n-rays 100000]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATA_DIR = Path('/sdf/data/neutrino/cjesus/CIDER/ROOT_files/LARGE_files/water/mu-')
ESCAN_DIR = Path('/sdf/data/neutrino/cjesus/CIDER/LUCiD_tracking/muon')
ENERGIES = list(range(400, 2001, 100))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', required=True)
    ap.add_argument('--n-events', type=int, default=100)
    ap.add_argument('--n-rays', type=int, default=100_000)
    ap.add_argument('--nkeys', type=int, default=4)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lucid.sources.event_io import read_photon_data_from_photonsim
    from lucid.utils import unpack_siren_params
    from lucid.siren.core import build_cherenkov_context
    from lucid.siren.training.inference import SIRENPredictor
    from lucid.sources.siren_rays import make_cherenkov_surrogate_fn

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    siren_cfg = unpack_siren_params('muon', 'water')
    pred = SIRENPredictor(siren_cfg['siren_model_path'])
    ctx = build_cherenkov_context(pred, dict(siren_cfg['ray_sampling']))
    get_rays = make_cherenkov_surrogate_fn(ctx)
    origin = jnp.zeros(3); zhat = jnp.array([0., 0., 1.])
    keys = [jax.random.PRNGKey(s) for s in range(args.nkeys)]

    rows = []
    for E in ENERGIES:
        root = DATA_DIR / f'{E}MeV_500events.root'
        if not root.exists():
            print(f'{E}: no data file, skipped', flush=True); continue
        tot, s_sum, s_n = [], 0.0, 0
        for ev in range(args.n_events):
            r = read_photon_data_from_photonsim(str(root), ev)
            s = np.asarray(r['photon_origins'], float)[:, 2] / 100.0   # cm -> m
            tot.append(len(s)); s_sum += s.sum(); s_n += len(s)
        data_total = float(np.mean(tot)); s_data = s_sum / s_n

        nphot = float(ctx.n_photons_fn(float(E)))
        dE_num = 1.0  # MeV step for local nphot slope
        dnphot = (float(ctx.n_photons_fn(float(E) + dE_num))
                  - float(ctx.n_photons_fn(float(E) - dE_num))) / (2 * dE_num)
        s_w, w_w = 0.0, 0.0
        for k in keys:
            vec, org, inten = get_rays(origin, zhat, jnp.asarray(float(E)),
                                       args.n_rays, pred.params, k)
            w = np.asarray(inten); s_w += float((np.asarray(org)[:, 2] * w).sum())
            w_w += float(w.sum())
        s_sir = s_w / w_w

        r_tot = data_total / nphot
        dE_pred = (r_tot - 1.0) * nphot / dnphot
        vtx_pred_cm = (s_data - s_sir) * 100.0
        rows.append(dict(E=E, data_total=data_total, nphot=nphot, r=r_tot,
                         s_data=s_data, s_sir=s_sir,
                         dE_pred=dE_pred, vtx_pred=vtx_pred_cm))
        print(f'{E:5d}: data {data_total:9,.0f}  siren {nphot:9,.0f}  r={r_tot:.4f}  '
              f'<s>_data {s_data:.3f} m  <s>_siren {s_sir:.3f} m  '
              f'-> dE_pred {dE_pred:+6.1f} MeV  vtx_pred {vtx_pred_cm:+6.1f} cm', flush=True)

    # ---------------- measured biases from escan2 -------------------------------------------
    meas = []
    for E in ENERGIES:
        h5p = ESCAN_DIR / f'escan2_{E}' / f'escan2_{E}.h5'
        if not h5p.exists():
            continue
        with h5py.File(h5p, 'r') as f:
            dE, lon = [], []
            for k in sorted(f['events']):
                g = f['events'][k]
                dE.append(float(g['fit_vec9'][0] - g['truth_vec9'][0]))
                dv = (g['fit_vec9'][1:4] - g['truth_vec9'][1:4]) * 100.0
                u = g['tdir'][:]; u = u / np.linalg.norm(u)
                lon.append(float(np.dot(dv, u)))
        meas.append(dict(E=E, dEm=float(np.mean(dE)), lonm=float(np.mean(lon))))
        print(f'{E:5d}: measured dEm {np.mean(dE):+6.1f} MeV  long bias {np.mean(lon):+6.1f} cm',
              flush=True)

    Ep = np.array([r['E'] for r in rows]); Em = np.array([m['E'] for m in meas])
    fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    plt.subplots_adjust(hspace=0.08)
    ax[0].plot(Ep, [r['dE_pred'] for r in rows], 'o-', color='cornflowerblue',
               label='predicted from yield ratio r(E)')
    ax[0].plot(Em, [m['dEm'] for m in meas], 's', color='crimson', label='measured dE mean (escan2)')
    ax[0].axhline(0, c='k', lw=.8)
    ax[0].set_ylabel('dE bias (MeV)'); ax[0].grid(alpha=.3); ax[0].legend(frameon=False)
    ax[1].plot(Ep, [r['vtx_pred'] for r in rows], 'o-', color='cornflowerblue',
               label=r'predicted $\langle s\rangle_{data}-\langle s\rangle_{SIREN}$')
    ax[1].plot(Em, [m['lonm'] for m in meas], 's', color='crimson',
               label='measured longitudinal vtx bias (escan2)')
    ax[1].axhline(0, c='k', lw=.8)
    ax[1].set_xlabel('Energy (MeV)'); ax[1].set_ylabel('longitudinal bias (cm)')
    ax[1].grid(alpha=.3); ax[1].legend(frameon=False)
    fig.suptitle('SIREN emission calibration residuals vs measured reconstruction biases')
    png = out_dir / 'profile_sweep.png'
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'profile_sweep.{ext}', dpi=140, bbox_inches='tight')

    with h5py.File(out_dir / 'profile_sweep.h5', 'w') as h5:
        h5.attrs['n_events'] = args.n_events; h5.attrs['n_rays'] = args.n_rays
        h5.attrs['finished'] = datetime.now().isoformat()
        for key in ('E', 'data_total', 'nphot', 'r', 's_data', 's_sir', 'dE_pred', 'vtx_pred'):
            h5.create_dataset(key, data=np.array([r[key] for r in rows]))
        h5.create_dataset('meas_E', data=Em)
        h5.create_dataset('meas_dEm', data=np.array([m['dEm'] for m in meas]))
        h5.create_dataset('meas_lon', data=np.array([m['lonm'] for m in meas]))
    print(f'wrote {png}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
