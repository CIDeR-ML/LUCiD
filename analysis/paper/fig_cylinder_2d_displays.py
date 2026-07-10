#!/usr/bin/env python3
"""Figure: 2D unfolded-cylinder charge (Q) and first-arrival-time (T) displays.

For one PhotonSim muon event in the SK-like cylinder, four unfolded-cylinder panels laid
out 2x2 (rows: data / prediction; columns: Q / T):

  * DATA  Q, T  — the realistic per-PMT (charge, first-arrival time) from data mode.
  * PRED  Q     — the model's expected per-PMT charge (predicted mu).
  * PRED  T     — the "most likely first arrival" time per PMT, read out from the SAME
                  first-hit likelihood the tracker uses. Per PMT it is the time that
                  minimizes lucid.losses.first_arrival_window_nll (the first-arrival
                  order statistic of mu predicted photons) — i.e. the time term the fitter
                  optimizes, shown in colour instead of summed into a loss.

    python fig_cylinder_2d_displays.py               # simulate + cache + plot
    python fig_cylinder_2d_displays.py --generate-data
    python fig_cylinder_2d_displays.py --plot-results

Reuses the pipeline's event_io reader + _pad_event (units-correct) and
lucid.visualization.create_detector_display (the 2D unfold). The expensive forward-sim is
cached, so plotting style can be iterated with --plot-results. Run inside the container.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]        # LUCiD/
sys.path.insert(0, str(REPO_ROOT))
from analysis.paper.utils import paths, events          # noqa: E402

GEOM = str(REPO_ROOT / 'config' / 'SK_like_geom_config.json')
PHYS = str(REPO_ROOT / 'config' / 'SK_like_physics_config.json')
FIGURE = 'cylinder_2d_displays'
DEFAULT_ROOT = REPO_ROOT / 'data' / 'water' / 'muon' / '1000MeV_100events.root'
SIGMA, DELTA = 2.5, 1.0                                 # tracking time-term defaults (gn)
DIRECTION = [1.0, 0.0, 0.0]                             # transverse track -> classic barrel ring


def _cache_file():
    return paths.data_dir(FIGURE, 'local') / 'displays.npz'


# ------------------------------------------------------------ predicted first-arrival time
def _pool_runs(runs):
    """Concatenate the per-photon (time, weight, sensor) collections from independent
    prediction passes into one bigger collection — N passes of M photons -> N*M photons."""
    import jax.numpy as jnp
    lw = jnp.concatenate([r[0] for r in runs])
    ft = jnp.concatenate([r[1] for r in runs])
    fi = jnp.concatenate([r[2] for r in runs])
    q = jnp.mean(jnp.stack([r[3] for r in runs]), axis=0)       # per-PMT charge (a.u.) for Q
    return lw, ft, fi, q


def _predicted_time(lw, ft, fi, ND):
    """Per-PMT predicted arrival time = weighted mean of that PMT's photon arrival times,
    ``sum(w*t)/sum(w)``, in one pass over the collection. No grid. PMTs with no photons -> 0."""
    import jax
    import jax.numpy as jnp
    w = jnp.exp(jnp.clip(lw, -60.0, 20.0))
    ws = jax.ops.segment_sum(w, fi, num_segments=ND)
    wts = jax.ops.segment_sum(w * ft, fi, num_segments=ND)
    return np.asarray(jnp.where(ws > 0, wts / jnp.maximum(ws, 1e-30), 0.0))


# ------------------------------------------------------------------------------- simulate
def generate_data(root, entry, n_photons, pred_photons, pred_runs, K, temperature, push_back,
                  data_seed=0, reuse_pred=False):
    import jax
    import jax.numpy as jnp
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator

    det = generate_detector(GEOM); ND = len(det.all_points)
    raw, E, d0 = events.load_event(root, entry)
    d1 = np.array(DIRECTION, float); d1 /= np.linalg.norm(d1)
    axis, ang = events.rotation(d0, d1)                 # swing data photons d0 -> d1
    vertex = -push_back * d1                             # push the vertex back along the track
    _, track = events.build_track(vertex, d1, E)
    photon_dict = events.pad_event(raw, n_photons, axis, ang, translation=vertex)
    common = dict(K=K, default_detector_params=True, physics_config=PHYS,
                  particle='muon', detector_type='Cylinder')
    print(f'event {entry}: {E:.0f} MeV, {ND} PMTs, dir {np.round(d1, 2)}, vertex '
          f'{np.round(vertex, 1)}, data {n_photons:,} rays (seed {data_seed}) / '
          f'pred {pred_runs} x {pred_photons:,} rays', flush=True)

    data_sim = setup_event_simulator(GEOM, n_photons, temperature=None, is_data=True,
                                     hit_mode='realistic', **common)
    cd, td = (np.asarray(x) for x in data_sim(track, jax.random.PRNGKey(data_seed), photon_dict))

    if reuse_pred and _cache_file().exists():           # re-roll only the data; keep the pred
        cached = np.load(_cache_file())
        cq, time_pred = cached['pred_q'], cached['pred_t']
        print('  reusing cached prediction', flush=True)
    else:
        pred = setup_event_simulator(GEOM, pred_photons, temperature=temperature, is_data=False,
                                     hit_mode='per_photon', **common)
        # Independent prediction passes (different seeds); each pred_photons fits in memory. Pool
        # their per-PMT (time, weight) collections BEFORE the likelihood -> pred_runs x pred_photons
        # effective statistics without ever holding a grid x N_photons array.
        runs = [pred(track, jax.random.PRNGKey(100 + i)) for i in range(pred_runs)]
        lw, ft, fi, mu = _pool_runs(runs)
        time_pred = _predicted_time(lw, ft, fi, ND)
        cq = np.asarray(mu)

    d = paths.data_dir(FIGURE, 'local'); d.mkdir(parents=True, exist_ok=True)
    np.savez(_cache_file(), data_q=cd, data_t=td, pred_q=cq, pred_t=time_pred, energy=E)
    print(f'  data {int((cd > 0).sum())} lit / pred {int((cq > 0.01 * cq.max()).sum())} lit; '
          f'cached -> {_cache_file()}')


# ----------------------------------------------------------------------------------- plot
def _zero_at_first_hit(t, first=1.0):
    """Reference the time to the event's first hit: subtract the earliest lit time so the
    first PMT to fire sits at ``first`` ns (1 by default, so a log colour scale has no
    log(0) problem and all lit times are >= 1). Unlit PMTs (t<=0) stay at 0."""
    t = np.asarray(t, float)
    pos = t[t > 0]
    if pos.size == 0:
        return t
    return np.where(t > 0, t - pos.min() + first, 0.0)


def _scale(vals_list):
    """(vmin, vmax) = full min/max of the positive (lit) values — no percentile clipping.
    Shared across the data/pred pair per column so the colour means the same in both."""
    pos = np.concatenate([v[v > 0] for v in vals_list])
    return float(pos.min()), float(pos.max())


def _panel(disp, charges, values, plot_time, log_scale, vmin, vmax, base, label=None):
    """One unfolded-cylinder display saved as its own PDF (+PNG)."""
    for ext in ('pdf', 'png'):
        disp(np.asarray(charges), np.asarray(values), plot_time=plot_time, log_scale=log_scale,
             vmin=vmin, vmax=vmax, show_colorbar=True, colorbar_width='2.5%',
             colorbar_label=label, file_name=f'{base}.{ext}')
    import matplotlib.pyplot as plt
    plt.close('all')
    print(f'wrote {base}.pdf (+png)  [vmin={vmin:.3g} vmax={vmax:.3g}]')


def plot_results(out, log_q, pred_threshold):
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    from lucid.visualization import create_detector_display
    cf = _cache_file()
    if not cf.exists():
        print(f'[skip] no cache at {cf} — run --generate-data first'); return
    c = np.load(cf)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    disp = create_detector_display(GEOM, sparse=False)

    # The model predicts a little charge on every PMT; PMTs whose predicted charge is below
    # pred_threshold (expected PE) count as "no hit" and render gray in BOTH prediction panels,
    # matching how the data shows unlit PMTs. Default 0.5 PE ~ the data's lit fraction.
    pred_hit = c['pred_q'] >= pred_threshold
    pred_q = np.where(pred_hit, c['pred_q'], 0.0)
    pred_t = np.where(pred_hit, c['pred_t'], 0.0)
    # Times are referenced to the first hit (t=1 ns at the earliest arrival) per panel.
    arrays = {'data_q': c['data_q'], 'data_t': _zero_at_first_hit(c['data_t']),
              'pred_q': pred_q, 'pred_t': _zero_at_first_hit(pred_t)}

    # Per-panel colour scale (full min/max of that panel's own values), so each colorbar
    # matches its own hits — data T reaches ~450 ns (late scattered light), pred T only the
    # early direct-light region it is masked to; a shared scale would mislead one of them.
    # (name, charge_key, value_key, plot_time, log_scale, colorbar_label). Time panels use a
    # log scale (first hit referenced to 1 ns, so no log(0)); charge follows --log.
    Q = 'Photoelectron Count'
    cases = [
        ('data_Q', 'data_q', 'data_q', False, log_q, Q),
        ('data_T', 'data_q', 'data_t', True,  True,  None),
        ('pred_Q', 'pred_q', 'pred_q', False, log_q, Q),
        ('pred_T', 'pred_q', 'pred_t', True,  True,  None),
    ]
    for name, ck, vk, plot_time, log_s, label in cases:
        vals = arrays[vk]
        vmin, vmax = _scale([vals])                     # time: min = 1 (first hit), max = full
        _panel(disp, arrays[ck], vals, plot_time, log_s, vmin, vmax,
               str(out / f'cylinder_2d_{name}'), label=label)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate-data', action='store_true')
    ap.add_argument('--plot-results', action='store_true')
    ap.add_argument('--root', default=str(DEFAULT_ROOT))
    ap.add_argument('--entry', type=int, default=2)
    ap.add_argument('--n-photons', type=int, default=1_000_000, help='data photon buffer')
    ap.add_argument('--pred-photons', type=int, default=1_000_000,
                    help='prediction rays PER pass (kept at a size that fits in memory)')
    ap.add_argument('--pred-runs', type=int, default=3,
                    help='independent prediction passes to pool (3 x 1M = 3M effective stats)')
    ap.add_argument('--K', type=int, default=6)
    ap.add_argument('--temperature', type=float, default=0.05)
    ap.add_argument('--push-back', type=float, default=10.0,
                    help='metres to move the vertex back along the track (default: 10)')
    ap.add_argument('--data-seed', type=int, default=0,
                    help='RNG seed for the realistic data QE sampling (re-roll stray hits)')
    ap.add_argument('--reuse-pred', action='store_true',
                    help='re-run only the data sim and keep the cached prediction (fast)')
    ap.add_argument('--log', action='store_true', help='log charge scale (default: linear)')
    ap.add_argument('--pred-threshold', type=float, default=0.0,
                    help='PMTs with predicted charge (expected PE) below this render as no-hit '
                         '(gray) in the prediction panels; 0 = show every PMT (default)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    both = not (a.generate_data or a.plot_results)
    if a.generate_data or both:
        generate_data(a.root, a.entry, a.n_photons, a.pred_photons, a.pred_runs, a.K,
                      a.temperature, a.push_back, a.data_seed, a.reuse_pred)
    if a.plot_results or both:
        plot_results(Path(a.out) if a.out else paths.figure_dir(), a.log, a.pred_threshold)


if __name__ == '__main__':
    main()
