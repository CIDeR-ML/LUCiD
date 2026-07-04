#!/usr/bin/env python3
"""Average light EMISSION of N data events vs the SIREN emitter's prediction.

Pure emission-level comparison — no detector, no propagation, no QE. The gun fires from the
origin along +z, so no placement transform is applied; the comparison is in gun coordinates.

  DATA : PhotonSim ROOT photons of N events — emission point s = origin_z (m, projection on
         the track axis) and emission angle cos(theta) = direction_z, raw counts / event.
  SIREN: cherenkov_get_rays(origin, +z, E, n_rays, params, key) — the EXACT emitter the
         reconstruction predictor uses — weighted by its intensities (sum = N_photons(E)),
         averaged over nkeys keys.

Outputs: total emitted photons/event (data vs N_photons(E)), dN/ds and dN/dcos(theta)
overlays + data/SIREN ratios, written as PNG + h5.

    python analysis/tracking/siren_emission_check.py --config <cfg.json> --output OUT
        [--n-events 100] [--n-rays 250000]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.tracking.pipeline import load_config, _resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--n-events', type=int, default=100)
    ap.add_argument('--n-rays', type=int, default=250_000)
    ap.add_argument('--threshold', type=float, default=None,
                    help='override ray_sampling.threshold (default: value in siren_params.json)')
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

    cfg = load_config(args.config)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    root = str(_resolve(cfg['root_file']))
    part = cfg['particle']

    # ---------------- DATA: N events, gun frame (origin, +z) --------------------------------
    s_all, ct_all, totals, energies = [], [], [], []
    for ev in range(args.n_events):
        r = read_photon_data_from_photonsim(root, ev)
        O = np.asarray(r['photon_origins'], float) / 100.0        # cm -> m
        D = np.asarray(r['photon_directions'], float)
        s_all.append(O[:, 2]); ct_all.append(D[:, 2])
        totals.append(len(O)); energies.append(float(r['energy']))
    s_data = np.concatenate(s_all); ct_data = np.concatenate(ct_all)
    n_ev = len(totals); E = float(np.mean(energies))
    print(f"DATA : {n_ev} events @ {E:.0f} MeV — emitted photons/event "
          f"mean {np.mean(totals):,.0f} std {np.std(totals):,.0f}", flush=True)

    # ---------------- SIREN: the reconstruction emitter at the same track -------------------
    siren_cfg = unpack_siren_params(part, 'water')
    pred = SIRENPredictor(siren_cfg['siren_model_path'])
    ray_sampling = dict(siren_cfg['ray_sampling'])
    if args.threshold is not None:
        ray_sampling['threshold'] = args.threshold
    print(f"ray_sampling: {ray_sampling}", flush=True)
    ctx = build_cherenkov_context(pred, ray_sampling)
    get_rays = make_cherenkov_surrogate_fn(ctx)
    nphot_E = float(ctx.n_photons_fn(E))
    origin = jnp.zeros(3); zhat = jnp.array([0., 0., 1.])
    keys = [jax.random.PRNGKey(s) for s in range(cfg['gn']['nkeys'])]

    s_sir, ct_sir, w_sir = [], [], []
    for k in keys:
        vec, org, inten = get_rays(origin, zhat, jnp.asarray(E), args.n_rays, pred.params, k)
        s_sir.append(np.asarray(org)[:, 2]); ct_sir.append(np.asarray(vec)[:, 2])
        w_sir.append(np.asarray(inten))
    s_sir = np.concatenate(s_sir); ct_sir = np.concatenate(ct_sir)
    w_sir = np.concatenate(w_sir) / len(keys)                     # sum == N_photons(E)
    print(f"SIREN: N_photons({E:.0f}) = {nphot_E:,.0f}   (data/SIREN total ratio "
          f"{np.mean(totals)/nphot_E:.4f})", flush=True)

    # ---------------- histograms (per-event normalisation) ----------------------------------
    smax_m = float(ctx.s_max_fn(E)) / 1000.0
    s_bins = np.linspace(0.0, max(smax_m * 1.15, s_data.max()), 60)
    ct_bins = np.linspace(-1.0, 1.0, 120)
    hd_s, _ = np.histogram(s_data, s_bins); hs_s, _ = np.histogram(s_sir, s_bins, weights=w_sir)
    hd_c, _ = np.histogram(ct_data, ct_bins); hs_c, _ = np.histogram(ct_sir, ct_bins, weights=w_sir)
    hd_s = hd_s / n_ev; hd_c = hd_c / n_ev                         # data: photons/event
    sc = 0.5 * (s_bins[:-1] + s_bins[1:]); cc = 0.5 * (ct_bins[:-1] + ct_bins[1:])

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw=dict(height_ratios=[2, 1]))
    for col, (x, hd, hs, lab) in enumerate([(sc, hd_s, hs_s, 'emission distance s (m)'),
                                            (cc, hd_c, hs_c, 'emission cos(theta) wrt track')]):
        a = ax[0, col]
        a.step(x, hd, where='mid', label=f'data ({n_ev} evts avg)')
        a.step(x, hs, where='mid', label='SIREN emitter')
        a.set(ylabel='photons / event / bin', title=lab, yscale='log'); a.grid(alpha=.3)
        a.legend(fontsize=9)
        r = ax[1, col]
        with np.errstate(divide='ignore', invalid='ignore'):
            rr = np.where(hs > 0, hd / hs, np.nan)
        r.axhline(1, c='k', lw=.8)
        r.plot(x, rr, '.', ms=4)
        r.set(xlabel=lab, ylabel='data / SIREN', ylim=(0, 2)); r.grid(alpha=.3)
    if smax_m:
        ax[0, 0].axvline(smax_m, c='k', ls=':', lw=1)
    fig.suptitle(f'{part} {E:.0f} MeV — average light emission: data vs SIREN '
                 f'(totals: data {np.mean(totals):,.0f}, SIREN {nphot_E:,.0f})')
    fig.tight_layout()
    tag = f"_thr{ray_sampling['threshold']}" if args.threshold is not None else ''
    png = out_dir / f'siren_emission_check{tag}.png'; fig.savefig(png, dpi=130)

    with h5py.File(out_dir / f'siren_emission_check{tag}.h5', 'w') as h5:
        h5.attrs['config_json'] = json.dumps(cfg); h5.attrs['n_events'] = n_ev
        h5.attrs['energy'] = E; h5.attrs['nphot_siren'] = nphot_E
        h5.attrs['data_total_mean'] = float(np.mean(totals))
        h5.attrs['data_total_std'] = float(np.std(totals))
        h5.attrs['finished'] = datetime.now().isoformat()
        for k, v in (('s_bins', s_bins), ('ct_bins', ct_bins), ('data_s', hd_s),
                     ('siren_s', hs_s), ('data_ct', hd_c), ('siren_ct', hs_c),
                     ('data_totals', np.array(totals))):
            h5.create_dataset(k, data=v)
    print(f"wrote {png}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
