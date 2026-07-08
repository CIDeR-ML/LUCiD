#!/usr/bin/env python3
"""Per-term score/curvature decomposition in the track frame — what does time_weight balance?

For each event, with L_Q (charge Poisson NLL) and L_T (time window NLL) as separate
functions of vec9:
  score at TRUTH   s_k = <dL/dtheta_k>|truth   (zero for a well-specified term)
  curvature        H_kk per term (Hessian diagonal at truth)
  tension at FIT   g_k per term at the converged vec9 (g_Q = -w g_T at convergence)

All rotated into the physical frame [E, longitudinal, trans1, trans2, |dir4|, t0]
(position block rotated onto the true direction; the 4 sin/cos direction params are
reported as a block magnitude). Also projects on the empirical soft mode
v = (0.285 m along track, +1 ns t0) and reports the zero-bias weight prediction
w* = -s_Q.v / s_T.v per energy.

    python analysis/tracking/score_decomp.py --config <cfg.json> --h5 <study.h5> \
        [--n-events 100] [--nkeys 4]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.tracking.pipeline import TrackingPipeline, load_config  # noqa: E402

SOFT_SLOPE_M_PER_NS = 0.285      # measured vertex-t0 degeneracy: 28.5 cm/ns


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--n-events', type=int, default=100)
    ap.add_argument('--nkeys', type=int, default=4)
    ap.add_argument('--nkeys-hess', type=int, default=1)
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

    def LQ(t9, oc, ot, key):
        _, _, _, tot = pipe.pred(track_from_vec9(t9), key)
        return counts_loss(oc, jnp.maximum(tot, 1e-8), eps=0.0, normalize=False)

    def LT(t9, oc, ot, key):
        lw, ft, fi, tot = pipe.pred(track_from_vec9(t9), key)
        mu = jnp.maximum(tot, 1e-8)
        return jnp.sum(first_arrival_window_nll(lw, ft, fi, ot - t9[8], mu, oc, ND,
                                                sigma=sigma, delta=delta_k))

    gQ_f = jax.jit(jax.grad(LQ)); gT_f = jax.jit(jax.grad(LT))

    def perpmt(t9, oc, ot, key):
        lw, ft, fi, tot = pipe.pred(track_from_vec9(t9), key)
        mu = jnp.maximum(tot, 1e-8)
        tnll = first_arrival_window_nll(lw, ft, fi, ot - t9[8], mu, oc, ND,
                                        sigma=sigma, delta=delta_k)
        return mu, tnll
    pjac_f = jax.jit(jax.jacfwd(perpmt, argnums=0))
    perpmt_f = jax.jit(perpmt)

    def rot(vec9_grad, u, t1, t2):
        """vec9-space gradient -> [E, long, tr1, tr2, |dir4|, t0] (chain rule for position)."""
        gp = np.asarray(vec9_grad)
        return np.array([gp[0], gp[1:4] @ u, gp[1:4] @ t1, gp[1:4] @ t2,
                         np.linalg.norm(gp[4:8]), gp[8]])

    S_Q, S_T = [], []
    W_GRID = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    LON_PRED, US, GQ9, GT9, FQ9, FT9 = [], [], [], [], [], []
    with h5py.File(args.h5, 'r') as fh:
        keys_h5 = sorted(fh['events'])[:args.n_events]
        for ki, k in enumerate(keys_h5):
            g = fh['events'][k]
            ev = int(g.attrs['ev']) if 'ev' in g.attrs else ki
            P = pipe._prepare_event(ev)
            th9 = np.asarray(P['th9'], float)
            oc, ot = jnp.asarray(P['oc']), jnp.asarray(P['ot'])
            u = np.asarray(P['d'], float); u = u / np.linalg.norm(u)
            a = np.array([1., 0., 0.]) if abs(u[0]) < 0.9 else np.array([0., 1., 0.])
            t1 = np.cross(u, a); t1 /= np.linalg.norm(t1); t2 = np.cross(u, t1)

            gq = np.mean([np.asarray(gQ_f(jnp.asarray(th9), oc, ot, jax.random.PRNGKey(s)))
                          for s in range(args.nkeys)], 0)
            gt = np.mean([np.asarray(gT_f(jnp.asarray(th9), oc, ot, jax.random.PRNGKey(s)))
                          for s in range(args.nkeys)], 0)
            FQ = np.zeros((9, 9)); FT = np.zeros((9, 9))
            for sk in range(args.nkeys_hess):
                Jmu, Jl = pjac_f(jnp.asarray(th9), oc, ot, jax.random.PRNGKey(sk))
                mu, _ = perpmt_f(jnp.asarray(th9), oc, ot, jax.random.PRNGKey(sk))
                Jmu = np.asarray(Jmu); Jl = np.asarray(Jl); mu = np.asarray(mu)
                FQ += (Jmu / np.clip(mu, 1e-8, None)[:, None]).T @ Jmu
                FT += Jl.T @ Jl
            FQ /= args.nkeys_hess; FT /= args.nkeys_hess

            S_Q.append(rot(gq, u, t1, t2)); S_T.append(rot(gt, u, t1, t2))
            # full 9D first-order displacement per event: delta(w) = -(FQ + w FT)^-1 (sq + w st)
            row = []
            for w in W_GRID:
                A = FQ + w * FT + 1e-6 * np.eye(9)
                try:
                    d9 = -np.linalg.solve(A, gq + w * gt)
                    row.append(100.0 * float(d9[1:4] @ u))       # predicted lon shift (cm)
                except np.linalg.LinAlgError:
                    row.append(np.nan)
            LON_PRED.append(row)
            US.append(u); GQ9.append(gq); GT9.append(gt); FQ9.append(FQ); FT9.append(FT)
            if ki % 20 == 0:
                print(f'ev{ev:04d} done', flush=True)

    import h5py
    names = ['E(MeV)', 'long(m)', 'tr1(m)', 'tr2(m)', '|dir4|', 't0(ns)']
    S_Q = np.array(S_Q); S_T = np.array(S_T); LON_PRED = np.array(LON_PRED)
    n = len(S_Q)
    print(f"\n=== {cfg['name']}  (n={n}) ===")
    print('component |  score_Q@truth (SE) |  score_T@truth (SE)')
    for i, nm in enumerate(names):
        sq, st = S_Q[:, i], S_T[:, i]
        print(f'{nm:>9} | {sq.mean():+12.2f} ({sq.std()/np.sqrt(n):.2f}) | '
              f'{st.mean():+12.2f} ({st.std()/np.sqrt(n):.2f})')
    print('\nPREDICTED b(w) from per-event 9D solve  [mean +- SE, cm]:')
    for wi, w in enumerate(W_GRID):
        col = LON_PRED[:, wi]
        m = np.isfinite(col)
        print(f'  w={w:4.2f}: {col[m].mean():+7.2f} +- {col[m].std()/np.sqrt(m.sum()):.2f}')
    mb = np.nanmean(LON_PRED, axis=0)
    cross = np.where(np.diff(np.sign(mb)))[0]
    if len(cross):
        i = cross[0]
        wstar = W_GRID[i] + (W_GRID[i+1]-W_GRID[i]) * (0 - mb[i]) / (mb[i+1]-mb[i])
        print(f'  predicted w* (b=0): {wstar:.2f}')
    else:
        print('  predicted w*: no crossing in grid')
    out = Path(args.h5).parent.parent / 'score_decomp' / f"scores_{cfg['name']}.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, 'w') as h5:
        h5.create_dataset('w_grid', data=W_GRID)
        h5.create_dataset('lon_pred', data=LON_PRED)
        h5.create_dataset('gq', data=np.array(GQ9)); h5.create_dataset('gt', data=np.array(GT9))
        h5.create_dataset('FQ', data=np.array(FQ9)); h5.create_dataset('FT', data=np.array(FT9))
        h5.create_dataset('u', data=np.array(US))
    print(f'wrote {out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
