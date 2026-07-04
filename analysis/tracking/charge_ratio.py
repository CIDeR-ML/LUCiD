#!/usr/bin/env python3
"""Truth-charge ratio diagnostic: predicted vs observed total charge at the EXACT truth track.

For each event: make the realistic data exactly as run_study does (same placement + t0 seeds),
then evaluate the per-photon predictor AT TRUTH (no seeding, no GN) and record

    R = sum_i mu_i(truth) / sum_i q_i(obs)        (averaged over the GN nkeys PRNG keys)

R directly measures the predictor-vs-data brightness mismatch that the charge-Poisson term
converts into an energy bias (dE/E ~ 1/R - 1 to first order, charge ~ linear in E). This is
how the historical ``tot_n_scale = 0.982`` was derived. Also logs the ratio over LIT PMTs only
and the predicted/observed hit multiplicity for context.

    python analysis/tracking/charge_ratio.py --config <cfg.json> --output OUT [--events 0,1,...]

Shares the submit_job.py interface (--run-script charge_ratio.py).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.tracking.pipeline import TrackingPipeline, load_config, rand_tf, truth9, _resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--events', default=None)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import h5py
    from lucid.sources.event_io import read_photon_data_from_photonsim
    from lucid.fitting import track_from_vec9

    cfg = load_config(args.config)
    name = (cfg.get('name') or Path(args.config).stem) + '_chargeratio'
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    events = ([int(x) for x in args.events.split(',')] if args.events
              else list(range(cfg['event_start'], cfg['event_start'] + cfg['n_events'])))

    pipe = TrackingPipeline(cfg)
    gn = cfg['gn']
    keys = [jax.random.PRNGKey(s) for s in range(gn['nkeys'])]

    print(f"=== truth-charge ratio '{name}' — {len(events)} events, n_rays={cfg['n_rays']}, "
          f"tot_n_scale={gn['tot_n_scale']} ===", flush=True)

    rows = []
    for ev in events:
        try:
            raw = read_photon_data_from_photonsim(str(_resolve(cfg['root_file'])), ev)
            track_len = pipe.smax_m(raw['energy']) if pipe.smax_m is not None else None
            raw, vtx, d = rand_tf(raw, ev, cfg['fidr'], cfg['fidz'], cfg['placement_seed_base'],
                                  track_len_m=track_len, bounds=pipe.bounds,
                                  margin=cfg.get('containment_margin'))
            lo, hi = cfg['true_t0_range']
            t0t = float(np.random.default_rng(cfg['placement_seed_base'] + 7919 + ev)
                        .uniform(lo, hi)) if hi > lo else float(lo)
            if t0t != 0.0:
                raw = dict(raw); raw['photon_times'] = np.asarray(raw['photon_times'], float) + t0t
            th9, _ = truth9(vtx, d, raw['energy'], t0=t0t)
            pd = pipe._pad_event(raw)
            c, t = jax.lax.stop_gradient(pipe.data_sim(
                track_from_vec9(jnp.asarray(th9)),
                jax.random.PRNGKey(cfg['data_seed_base'] + ev), pd))
            oc = np.asarray(c); ot = np.where(oc > 0, np.asarray(t), 0.)

            # predictor charge AT TRUTH: mu from ReconModel.perpmt (includes tot_n_scale)
            mus = [np.asarray(pipe.model.perpmt(jnp.asarray(th9), jnp.asarray(oc),
                                                jnp.asarray(ot), k)[0]) for k in keys]
            mu = np.mean(mus, 0)
            lit = oc > 0
            r_all = float(mu.sum() / oc.sum())
            r_lit = float(mu[lit].sum() / oc[lit].sum())
            rows.append(dict(ev=ev, E=float(raw['energy']), r_all=r_all, r_lit=r_lit,
                             q_obs=float(oc.sum()), q_pred=float(mu.sum()),
                             nhit_obs=int(lit.sum()), nhit_pred=int((mu > 0.5).sum())))
            print(f"ev{ev:04d} R_all={r_all:.4f} R_lit={r_lit:.4f} "
                  f"q_obs={oc.sum():7.0f} q_pred={mu.sum():7.0f} nhit={lit.sum()}", flush=True)
        except Exception as e:
            print(f"ev{ev:04d} FAILED: {type(e).__name__}: {e}", flush=True)

    if not rows:
        print("no events succeeded"); return 1
    ra = np.array([r['r_all'] for r in rows]); rl = np.array([r['r_lit'] for r in rows])
    print(f"\n=== SUMMARY (N={len(rows)}) ===")
    print(f"  R_all (pred/obs total charge): mean {ra.mean():.4f}  median {np.median(ra):.4f}  "
          f"std {ra.std():.4f}")
    print(f"  R_lit (lit PMTs only)        : mean {rl.mean():.4f}  median {np.median(rl):.4f}")
    print(f"  first-order implied dE at 1 GeV: {(1/np.median(ra)-1)*1000:+.0f} MeV "
          f"(observed GN bias ~ +90)")
    print(f"  equivalent tot_n_scale to null it: {np.median(ra):.4f}")

    out = out_dir / f'{name}.h5'
    with h5py.File(out, 'w') as h5:
        h5.attrs['config_json'] = json.dumps(cfg); h5.attrs['finished'] = datetime.now().isoformat()
        for k in rows[0]:
            h5.create_dataset(k, data=np.array([r[k] for r in rows]))
    print(f"wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
