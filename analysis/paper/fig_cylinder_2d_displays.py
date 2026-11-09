#!/usr/bin/env python3
"""Figure: 2D unfolded-cylinder charge (Q) and first-arrival-time (T) displays.

For one PhotonSim muon event in the SK-like cylinder, four unfolded-cylinder panels laid
out 2x2 (rows: data / prediction; columns: Q / T):

  * DATA  Q, T  — the realistic per-PMT (charge, first-arrival time) from data mode.
  * PRED  Q     — the model's expected per-PMT charge (predicted mu).
  * PRED  T     — the predicted per-PMT arrival time: the weighted mean arrival of that
                  PMT's predicted photons (_predicted_time).

    python fig_cylinder_2d_displays.py               # simulate + cache + plot
    python fig_cylinder_2d_displays.py --generate-data
    python fig_cylinder_2d_displays.py --plot-results

The paper's figure is the two data panels (cylinder_2d_data_Q, cylinder_2d_data_T) of entry 5 of
data/water/muon/1500MeV_6events.root, a 1.5 GeV muon (scripts/download_data.py). The event
viewer's style: charge on a log scale over Q_RANGE in utils.plotting.CHARGE_CMAP, time
histogram-equalised in utils.plotting.TIME_CMAP.

Reuses the pipeline's root_reader reader + data_event.pad_photon_data (units-correct) and
lucid.visualization.create_detector_display (the 2D unfold). The expensive forward-sim is
cached, so the panels can be re-drawn with --plot-results. Run inside the container.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]        # LUCiD/
sys.path.insert(0, str(REPO_ROOT))
from analysis.paper.utils import paths, events, plotting, provenance   # noqa: E402

GEOM = str(REPO_ROOT / 'config' / 'SK_like_geom_config.json')
PHYS = str(REPO_ROOT / 'config' / 'SK_like_physics_config.json')
FIGURE = 'cylinder_2d_displays'
DEFAULT_ROOT = REPO_ROOT / 'data' / 'water' / 'muon' / '1500MeV_6events.root'
SIGMA, DELTA = 2.5, 1.0                                 # tracking time-term defaults (gn)
DIRECTION = [1.0, 0.0, 0.0]                             # transverse track -> classic barrel ring

Q_RANGE = (1.0, 6.0)                                    # charge colour bounds (PE), log scale


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
                  data_seed=0):
    import jax
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator
    from lucid.utils import rotation_onto

    det = generate_detector(GEOM); ND = len(det.all_points)
    raw, E, v0, d0 = events.load_event(root, entry)
    d1 = np.array(DIRECTION, float)
    d1 /= np.linalg.norm(d1)
    R = rotation_onto(np.asarray(d0, float), d1, np)     # swing the event's primary d0 -> d1
    vertex = -push_back * d1                             # push the vertex back along the track
    _, track = events.build_track(vertex, d1, E)
    photon_dict = events.pad_event(raw, n_photons, primary_vertex=v0, rotation=R, vertex=vertex)
    common = dict(K=K, default_detector_params=True, physics_config=PHYS,
                  particle='muon')
    print(f'event {entry}: {E:.0f} MeV, {ND} PMTs, dir {np.round(d1, 2)}, vertex '
          f'{np.round(vertex, 1)}, data {n_photons:,} rays (seed {data_seed}) / '
          f'pred {pred_runs} x {pred_photons:,} rays', flush=True)

    data_sim = setup_event_simulator(GEOM, n_photons, source='photons', temperature=None, **common)
    cd, td = (np.asarray(x) for x in data_sim(jax.random.PRNGKey(data_seed), photon_dict))

    pred = setup_event_simulator(GEOM, pred_photons, source='particle', temperature=temperature,
                                 **common)
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


def _equalised(vals):
    """Histogram-equalising norm over the lit (positive) times: each time maps to its midpoint
    quantile in the empirical CDF, so colour contrast is uniform per hit (the dense prompt peak
    gets most of the range, the sparse late tail little). Colorbar ticks at the 0/25/50/75/100th
    percentiles."""
    from matplotlib.colors import FuncNorm
    lit = np.asarray(vals)[np.asarray(vals) > 0]
    uniq, cnts = np.unique(lit, return_counts=True)
    cdf = np.cumsum(cnts) / cnts.sum()
    qmid = 0.5 * (np.concatenate([[0.0], cdf[:-1]]) + cdf)
    norm = FuncNorm((lambda v: np.interp(v, uniq, qmid), lambda p: np.interp(p, qmid, uniq)),
                    vmin=float(uniq[0]), vmax=float(uniq[-1]))
    ticks = sorted({round(float(x)) for x in np.percentile(lit, [0, 25, 50, 75, 100])})
    return norm, ticks


def _panel(disp, charges, values, plot_time, log_scale, vmin, vmax, base, label=None,
           colormap=None, colormap_norm=None, colorbar_ticks=None):
    """One unfolded-cylinder display saved as its own PDF (+PNG)."""
    for ext in ('pdf', 'png'):
        disp(np.asarray(charges), np.asarray(values), plot_time=plot_time, log_scale=log_scale,
             vmin=vmin, vmax=vmax, show_colorbar=True, colorbar_width='2.5%',
             colorbar_label=label, file_name=f'{base}.{ext}',
             colormap=colormap, colormap_norm=colormap_norm, colorbar_ticks=colorbar_ticks)
    import matplotlib.pyplot as plt
    plt.close('all')
    print(f'wrote {base}.pdf (+png)  [vmin={vmin:.3g} vmax={vmax:.3g}]')


def plot_results(out):
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    from lucid.visualization import create_detector_display
    cf = _cache_file()
    if not cf.exists():
        print(f'[skip] no cache at {cf} — run --generate-data first'); return
    c = np.load(cf)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    disp = create_detector_display(GEOM, sparse=False)
    qmap, tmap = plt.get_cmap(plotting.CHARGE_CMAP), plt.get_cmap(plotting.TIME_CMAP)
    qticks = sorted({int(round(x)) for x in np.geomspace(*Q_RANGE, 4)})
    # Times are referenced to the first hit (t=1 ns at the earliest arrival) per panel.
    for tag in ('data', 'pred'):
        q, t = c[f'{tag}_q'], _zero_at_first_hit(c[f'{tag}_t'])
        _panel(disp, q, q, False, True, *Q_RANGE, str(out / f'cylinder_2d_{tag}_Q'),
               label='Charge', colormap=qmap, colorbar_ticks=qticks)
        tnorm, tticks = _equalised(t)
        _panel(disp, q, t, True, False, tnorm.vmin, tnorm.vmax, str(out / f'cylinder_2d_{tag}_T'),
               label='Time (ns)', colormap=tmap, colormap_norm=tnorm, colorbar_ticks=tticks)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate-data', action='store_true')
    ap.add_argument('--plot-results', action='store_true')
    ap.add_argument('--root', default=str(DEFAULT_ROOT))
    ap.add_argument('--entry', type=int, default=5)
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
                    help='RNG seed for the realistic data QE sampling')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    both = not (a.generate_data or a.plot_results)
    if a.generate_data or both:
        generate_data(a.root, a.entry, a.n_photons, a.pred_photons, a.pred_runs, a.K,
                      a.temperature, a.push_back, a.data_seed)
        provenance.write_run_record(_cache_file().parent, __file__, a)
    if a.plot_results or both:
        out = Path(a.out) if a.out else paths.figure_dir()
        plot_results(out)
        provenance.write_run_record(out, __file__, a)


if __name__ == '__main__':
    main()
