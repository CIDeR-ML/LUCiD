#!/usr/bin/env python3
"""Charge-vs-time term tension along the track axis, around the CONVERGED joint fit.

For each event: take the fitted vec9 from a completed study h5, regenerate the exact same
observed event (same seeds via the pipeline), and scan ONLY the longitudinal coordinate:
vertex -> vertex + delta * u_true, everything else frozen at the fit.

  L_Q(delta)  charge Poisson NLL (t0-independent)
  L_T(delta)  first-arrival window NLL, minimized over t0 at each delta
  L_J(delta)  = L_Q + min_t0 L_T   (the joint objective restricted to the scan line)

Each term's argmin gives its preferred longitudinal position relative to the joint fit
(delta*_Q, delta*_T). If they coincide, both terms want the same (biased) vertex -> the
bias is a shared model displacement, not term competition. If they disagree, the joint
optimum is an information-weighted blend and the competition is measured in cm.

    python analysis/tracking/term_split_scan.py --config <escan2_E.json> \
        --h5 <escan2_E.h5> --output OUT [--n-events 50] [--nkeys 2]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.tracking.pipeline import TrackingPipeline, load_config  # noqa: E402


def parabolic_min(x, y):
    """Refine grid argmin with a 3-point parabola (falls back to the grid point)."""
    i = int(np.argmin(y))
    if i == 0 or i == len(x) - 1:
        return float(x[i])
    x0, x1, x2 = x[i - 1], x[i], x[i + 1]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = (y0 - 2 * y1 + y2)
    if den <= 0:
        return float(x1)
    return float(x1 + 0.5 * (y0 - y2) / den * (x2 - x1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--h5', required=True, help='completed study h5 with fit_vec9 per event')
    ap.add_argument('--output', required=True)
    ap.add_argument('--n-events', type=int, default=50)
    ap.add_argument('--nkeys', type=int, default=2)
    ap.add_argument('--delta-cm', type=float, default=40.0, help='scan half-range (cm)')
    ap.add_argument('--n-delta', type=int, default=41)
    ap.add_argument('--t0-ns', type=float, default=2.5, help='t0 half-range (ns)')
    ap.add_argument('--n-t0', type=int, default=101)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import h5py
    from lucid.fitting import track_from_vec9
    from lucid.losses import counts_loss, first_arrival_window_nll

    cfg = load_config(args.config)
    pipe = TrackingPipeline(cfg, verbose=True)
    gn = cfg['gn']
    sigma, delta_k = float(gn['sigma']), float(gn['delta'])
    ND = pipe.ND

    deltas_m = np.linspace(-args.delta_cm, args.delta_cm, args.n_delta) / 100.0
    t0_grid = jnp.linspace(-args.t0_ns, args.t0_ns, args.n_t0)

    def _terms(t9, oc, ot, key):
        lw, ft, fi, tot = pipe.pred(track_from_vec9(t9), key)
        mu = jnp.maximum(tot, 1e-8)
        LQ = counts_loss(oc, mu, eps=0.0, normalize=False)
        def tn(t0):
            return jnp.sum(first_arrival_window_nll(lw, ft, fi, ot - t0, mu, oc, ND,
                                                    sigma=sigma, delta=delta_k))
        LT = jax.vmap(tn)(t0_grid)                    # (n_t0,)
        return LQ, LT
    terms = jax.jit(_terms)

    fh = h5py.File(args.h5, 'r')
    keys_h5 = sorted(fh['events'])[:args.n_events]
    out_rows = []
    for ki, k in enumerate(keys_h5):
        g = fh['events'][k]
        ev = int(g.attrs['ev']) if 'ev' in g.attrs else ki
        t9f = np.asarray(g['fit_vec9'], float)
        th9 = np.asarray(g['truth_vec9'], float)
        u = np.asarray(g['tdir'], float); u = u / np.linalg.norm(u)
        lon_fit = float(np.dot((t9f[1:4] - th9[1:4]), u)) * 100.0   # cm, fit vs truth

        P = pipe._prepare_event(ev)
        oc, ot = jnp.asarray(P['oc']), jnp.asarray(P['ot'])

        LQ = np.zeros(len(deltas_m)); LT = np.zeros(len(deltas_m))
        for di, dm in enumerate(deltas_m):
            t9d = t9f.copy(); t9d[1:4] = t9f[1:4] + dm * u
            lq_acc, lt_acc = 0.0, None
            for s in range(args.nkeys):
                lq, lt = terms(jnp.asarray(t9d), oc, ot, jax.random.PRNGKey(s))
                lq_acc += float(lq)
                lt_acc = np.asarray(lt) if lt_acc is None else lt_acc + np.asarray(lt)
            LQ[di] = lq_acc / args.nkeys
            LT[di] = float(np.min(lt_acc / args.nkeys))             # min over t0
        dcm = deltas_m * 100.0
        dQ = parabolic_min(dcm, LQ); dT = parabolic_min(dcm, LT)
        dJ = parabolic_min(dcm, LQ + LT)
        out_rows.append(dict(ev=ev, lon_fit=lon_fit, dQ=dQ, dT=dT, dJ=dJ,
                             LQ=LQ, LT=LT, dcm=dcm))
        print(f'ev{ev:04d}: lon_fit {lon_fit:+7.1f} cm | dQ* {dQ:+6.1f}  dT* {dT:+6.1f}  '
              f'dJ* {dJ:+6.1f} cm | absQ {lon_fit+dQ:+7.1f}  absT {lon_fit+dT:+7.1f}', flush=True)
    fh.close()

    dQm = np.mean([r['dQ'] for r in out_rows]); dTm = np.mean([r['dT'] for r in out_rows])
    dJm = np.mean([r['dJ'] for r in out_rows]); lonm = np.mean([r['lon_fit'] for r in out_rows])
    print(f'SUMMARY n={len(out_rows)}: mean lon_fit {lonm:+.1f} cm | '
          f'dQ* {dQm:+.1f}  dT* {dTm:+.1f}  dJ* {dJm:+.1f} cm | '
          f'ABS charge-pref {lonm+dQm:+.1f}  time-pref {lonm+dTm:+.1f} cm | '
          f'tension(Q-T) {dQm-dTm:+.1f} cm', flush=True)

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.config).stem
    with h5py.File(out_dir / f'term_split_{tag}.h5', 'w') as h5:
        h5.attrs['config_json'] = json.dumps(cfg)
        h5.attrs['finished'] = datetime.now().isoformat()
        h5.create_dataset('delta_cm', data=out_rows[0]['dcm'])
        for name in ('ev', 'lon_fit', 'dQ', 'dT', 'dJ'):
            h5.create_dataset(name, data=np.array([r[name] for r in out_rows]))
        h5.create_dataset('LQ', data=np.stack([r['LQ'] for r in out_rows]))
        h5.create_dataset('LT', data=np.stack([r['LT'] for r in out_rows]))
    print(f'wrote {out_dir}/term_split_{tag}.h5', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
