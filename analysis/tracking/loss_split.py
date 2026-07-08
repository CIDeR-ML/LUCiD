#!/usr/bin/env python3
"""Report the separate charge (Q) and time (T) NLL contributions at the converged fits.

For each event of a completed study: regenerate the observed event, evaluate the fit's
vec9 in one forward pass, and print L_Q (Poisson charge NLL) and L_T (first-arrival
window NLL at the fit's own t0) separately, plus per-config means.

    python analysis/tracking/loss_split.py --config <cfg.json> --h5 <study.h5> \
        [--n-events 50] [--nkeys 2]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.tracking.pipeline import TrackingPipeline, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--n-events', type=int, default=50)
    ap.add_argument('--nkeys', type=int, default=2)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import h5py
    from lucid.fitting import track_from_vec9
    from lucid.losses import counts_loss, first_arrival_window_nll

    cfg = load_config(args.config)
    pipe = TrackingPipeline(cfg, verbose=False)
    gn = cfg['gn']
    sigma, delta_k = float(gn['sigma']), float(gn['delta'])
    ND = pipe.ND

    def _terms(t9, oc, ot, key):
        lw, ft, fi, tot = pipe.pred(track_from_vec9(t9), key)
        mu = jnp.maximum(tot, 1e-8)
        LQ = counts_loss(oc, mu, eps=0.0, normalize=False)
        LT = jnp.sum(first_arrival_window_nll(lw, ft, fi, ot - t9[8], mu, oc, ND,
                                              sigma=sigma, delta=delta_k))
        return LQ, LT
    terms = jax.jit(_terms)

    LQs, LTs, nh = [], [], []
    with h5py.File(args.h5, 'r') as fh:
        keys_h5 = sorted(fh['events'])[:args.n_events]
        for ki, k in enumerate(keys_h5):
            g = fh['events'][k]
            ev = int(g.attrs['ev']) if 'ev' in g.attrs else ki
            t9f = jnp.asarray(np.asarray(g['fit_vec9'], float))
            P = pipe._prepare_event(ev)
            oc, ot = jnp.asarray(P['oc']), jnp.asarray(P['ot'])
            lq, lt = 0.0, 0.0
            for s in range(args.nkeys):
                a, b = terms(t9f, oc, ot, jax.random.PRNGKey(s))
                lq += float(a); lt += float(b)
            LQs.append(lq / args.nkeys); LTs.append(lt / args.nkeys)
            nh.append(int(P['n_hit']))
    LQs = np.array(LQs); LTs = np.array(LTs); nh = np.array(nh)
    tw = float(gn.get('time_weight', 1.0))
    print(f"{cfg['name']}: n={len(LQs)}  tw={tw}")
    print(f"  L_Q  mean {LQs.mean():12.1f}  median {np.median(LQs):12.1f}")
    print(f"  L_T  mean {LTs.mean():12.1f}  median {np.median(LTs):12.1f}   (x tw = {tw*LTs.mean():.1f} in objective)")
    print(f"  ratio tw*L_T / L_Q (means): {tw*LTs.mean()/LQs.mean():+.4f}   n_hit mean {nh.mean():.0f}")
    print(f"  per-hit: L_T/n_hit {np.mean(LTs/nh):.3f}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
