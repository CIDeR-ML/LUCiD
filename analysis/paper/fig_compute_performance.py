#!/usr/bin/env python3
"""Figure: computational performance -- timing vs number of rays, per K.

Two independent statements, one figure each:

  raytracing_timing        forward only, data-like output. The SIREN ray tracer with
                           ``hit_mode='realistic'`` (Bernoulli QE, hard-min first-arrival,
                           TTS smearing, Abe_2013 charge model) -- i.e. what data mode
                           produces, driven by the SIREN emitter instead of photon-by-photon
                           injection. How fast we predict per-PMT Q and T.

  optimization_step_timing one full reconstruction step: the reverse-mode gradient of the
                           recon NLL, the forward-mode Gauss-Newton/Fisher metric
                           ``Jmu^T diag(1/mu) Jmu + Jl^T Jl`` (jacfwd of ReconModel._perpmt,
                           ONE key), and the SCALE9-preconditioned damped 9x9 solve --
                           exactly what fit_track pays per iteration when the metric refreshes.

Both use the published tracking working point (SK_like 11k sensors, sigma = TTS = 2.1 ns,
energy_scale_mode='nphot'); see analysis/paper/utils/studies.py.

This is a live GPU benchmark, so it follows the generate/plot split:

    python fig_compute_performance.py                        # minimal local benchmark + plot
    python fig_compute_performance.py --plot-results         # plot existing results
    python fig_compute_performance.py --generate-data --backend s3df --full   # paper grid on a GPU

Defaults are deliberately small (K=1,3; N up to 200k; few timing runs). ``--full`` does the
paper grid (K=1..9 step 2, N up to 1M) and ``--backend s3df`` submits it to a GPU partition.
Every timed call is stored, so the per-point estimator (``--estimator mode|median|min|mean``)
is a plotting choice and never needs a rerun (``mean`` by default; ``mode`` is the robust
alternative when the run count is low, since wall-clock outliers are one-sided slow).
Any point that exhausts GPU memory is recorded as null and reported, not fatal.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]        # LUCiD/
sys.path.insert(0, str(REPO_ROOT))
from analysis.paper.utils import paths, studies         # noqa: E402

GEOM = str(REPO_ROOT / 'config' / 'SK_like_geom_config.json')
PHYS = str(REPO_ROOT / 'config' / 'SK_like_physics_config.json')
GRID = {'n_cap': 80, 'n_angular': 120, 'n_height': 80}
BAND = (274.91, 673.83)
PARTICLE = 'muon'

DEF_K, DEF_N = [1, 3], [10000, 100000, 200000]
# The grid stops at 1M: the optimization step pushes 9 forward-mode tangents through
# per-photon arrays of length K*max_candidates*N, and a single allocation exceeds a 40 GB
# A100 above 1M rays (measured: OOM at 1.5M for every K>=3, at 2M for K=1). Both panels
# therefore share one N range.
PLOT_MAX_N = 1_000_000
FULL_K = [1, 3, 5, 7, 9]
FULL_N = [10000, 100000, 250000, 500000, 750000, 1000000]

IMAGE = '/sdf/data/neutrino/cjesus/software/images/lucid.sif'
BINDS = '/sdf,/fs,/sdf/scratch,/lscratch,/cvmfs'
ENV_BASE = '/sdf/data/neutrino/cjesus/python_envs/lucid'   # CUDA jaxlib layered over the image

# fit_track's step knobs at the published working point (lucid/fitting/recon.py).
LR, LAM, RIDGE_I, TRUST = 4.0, 0.01, 0.1, 3.0


def _track9(jnp):
    """A 1 GeV muon at the centre: theta9 = (E, x, y, z, sin/cos theta, sin/cos phi, t0)."""
    th, ph = np.pi / 3, np.pi / 4
    return jnp.asarray([1000.0, 0.0, 0.0, 0.0,
                        np.sin(th), np.cos(th), np.sin(ph), np.cos(ph), 0.0], dtype=float)


def _detector_params(jnp, ND):
    from lucid.detector_params import load_detector_params
    dp = load_detector_params(PHYS, num_sensors=ND)
    return dp._replace(response=dp.response._replace(tts=jnp.asarray(studies.TTS)))


def half_sample_mode(xs):
    """Robust mode of a continuous sample (Bickel/Robertson-Cryer half-sample mode).

    Wall-clock timings are right-skewed: interference only ever makes a call SLOWER, so
    the mean is biased upward by outliers while the mode sits at the true cost. The bare
    mode is undefined for continuous data (no value repeats), so recurse on the densest
    half -- the shortest window containing ceil(n/2) sorted points -- until two remain.
    """
    xs = np.sort(np.asarray(xs, float))
    while len(xs) > 2:
        h = (len(xs) + 1) // 2
        widths = xs[h - 1:] - xs[:len(xs) - h + 1]
        i = int(np.argmin(widths))
        xs = xs[i:i + h]
    return float(np.mean(xs))


def _time_it(call, warmup, runs):
    """Per-call wall-clock times for ``runs`` calls after ``warmup`` untimed ones.

    Returns the raw samples so the estimator (mode / median / min / mean) is a PLOTTING
    choice, not a measurement one -- no rerun is needed to change it.
    """
    ts = []
    for i in range(warmup + runs):
        t0 = time.time()
        call(i)
        if i >= warmup:
            ts.append(time.time() - t0)
    return ts


# ----------------------------------------------------------------- the three measurements
PANELS = ('pred', 'grad', 'predgrad', 'fisher')


def _bench_recon(K_values, N_values, warmup, runs, want=PANELS):
    """Time the three reconstruction primitives on ONE ReconModel per (K, N).

    All three share the same predictor and the same primal, so their costs decompose
    cleanly rather than being three unrelated measurements:

      pred   -- ReconModel.perpmt: the per-PMT prediction (mean charge mu, time NLL).
      grad   -- ReconModel.grad: reverse-mode gradient of the recon NLL w.r.t. theta9.
                NOTE reverse-mode must evaluate the primal to build the tape, so the forward
                pass is already inside this number; jax.grad merely discards the loss value.
      predgrad -- jax.value_and_grad of the same loss: the gradient AND the loss value, i.e.
                what an optimizer that also wants the objective pays.
      fisher -- ReconModel.fisher_ad: the Fisher/Gauss-Newton metric
                Jmu^T diag(1/mu) Jmu + Jl^T Jl, by jacfwd of perpmt (9 tangents), one key.

    Building the model once per grid point also means one set of compilations covers all
    three, and every measurement sees identical detector, data and machine conditions.
    """
    import jax
    import jax.numpy as jnp
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator
    from lucid.fitting import ReconModel, track_from_vec9
    from lucid.utils import unpack_siren_params
    from lucid.siren.core import build_cherenkov_context
    from lucid.siren.training.inference import SIRENPredictor

    ND = len(generate_detector(GEOM).all_points)
    dp = _detector_params(jnp, ND)
    t9 = _track9(jnp)

    # nphot(E) from the emitter context -- the campaign's single-pass energy scale.
    scfg = unpack_siren_params(PARTICLE, 'water')
    ctx = build_cherenkov_context(SIRENPredictor(scfg['siren_model_path']), scfg['ray_sampling'])

    # Fixed pseudo-data (per-PMT Q, T) from one realistic forward -- the observed event the
    # model is evaluated against. Independent of the benchmark's N, so every point sees one
    # dataset, and the data-like sim never enters the timings.
    data_sim = setup_event_simulator(
        GEOM, 250_000, temperature=None, K=8, hit_mode='realistic', physics_config=PHYS,
        default_detector_params=dp, particle=PARTICLE, wavelength_mode=True,
        charge_resolution=studies.CHARGE_RESOLUTION, cherenkov_emission_band=BAND, **GRID)
    oc, ot = data_sim(track_from_vec9(t9), jax.random.PRNGKey(0))
    oc, ot = jnp.asarray(oc), jnp.asarray(ot)

    res = {p: {K: {'N': [], 'runs': [], 'mode': []} for K in K_values} for p in want}
    for K in K_values:
        for N in N_values:
            pred = setup_event_simulator(
                GEOM, N, temperature=0.1, K=K, hit_mode='per_photon', physics_config=PHYS,
                default_detector_params=True, particle=PARTICLE, wavelength_mode=True,
                pos_grad_threshold=K, n_grad_iters=K, cherenkov_emission_band=BAND, **GRID)
            model = ReconModel(pred, ND, sigma=studies.GN['sigma'], delta=1.0, tot_n_scale=1.0,
                               energy_from_scale=True, energy_scale_mode='nphot',
                               nphot_fn=ctx.n_photons_fn)

            def _pred(i):
                out = model.perpmt(t9, oc, ot, jax.random.PRNGKey(i))
                jax.tree.map(lambda x: x.block_until_ready(), out)

            def _grad(i):
                g = model.grad(t9, oc, ot, jax.random.PRNGKey(i))
                jax.tree.map(lambda x: x.block_until_ready(), g)

            vg = jax.jit(jax.value_and_grad(model._loss))

            def _predgrad(i):
                out = vg(t9, oc, ot, jax.random.PRNGKey(i))
                jax.tree.map(lambda x: x.block_until_ready(), out)

            def _fisher(i):
                # returns a numpy 9x9, so the host transfer already synchronises
                model.fisher_ad(t9, oc, ot, [jax.random.PRNGKey(i)], None)

            calls = {'pred': _pred, 'grad': _grad, 'predgrad': _predgrad, 'fisher': _fisher}
            for name in want:
                try:
                    ts = _time_it(calls[name], warmup, runs)
                    m = half_sample_mode(ts)
                except Exception as e:            # OOM: 9 tangents over K*max_candidates*N
                    print(f"  {name:6s} K={K} N={N:>8}: FAILED ({type(e).__name__})", flush=True)
                    ts, m = None, None
                r = res[name][K]
                r['N'].append(N); r['runs'].append(ts); r['mode'].append(m)
                if m is not None:
                    print(f"  {name:6s} K={K} N={N:>8}: {m * 1000:8.2f} ms", flush=True)
    return res


def generate_data(backend, full, warmup, runs, out_data, account='mli:cider-ml', panels=PANELS):
    K, N = (FULL_K, FULL_N) if full else (DEF_K, DEF_N)
    if backend == 's3df':
        script = Path(__file__).resolve()
        cmd = (f"APPTAINERENV_PYTHONUSERBASE={ENV_BASE} APPTAINERENV_PYTHONPATH='' "
               f"apptainer exec --nv -B {BINDS} {IMAGE} /opt/conda/bin/python3 {script} "
               f"--generate-data --backend local {'--full' if full else ''} "
               f"--warmup {warmup} --runs {runs} --out-data {out_data}")
        sb = out_data.parent / 'compute_benchmark.sh'
        # Timing must come from ONE GPU type, so the partition is fixed to ampere (A100) and
        # only the account is variable — the mli tree's node cap is often saturated by sibling
        # accounts, which blocks mli:cider-ml at zero usage.
        sb.write_text("#!/bin/bash\n#SBATCH -p ampere\n#SBATCH --gpus 1\n"
                      "#SBATCH --cpus-per-task 4\n#SBATCH --mem 64G\n"
                      f"#SBATCH -t 08:00:00\n#SBATCH -A {account}\n"
                      f"#SBATCH -o {out_data.parent}/benchmark_%j.log\n\n"
                      "nvidia-smi -L || true\n"
                      f"{cmd}\n")
        # sbatch lives outside the container; run this branch on the login node.
        print(f'[s3df] wrote {sb}; submitting')
        subprocess.run(['sbatch', str(sb)], check=True)
        return
    print(f'[local] benchmarking {list(panels)} K={K} N={N} (warmup {warmup}, runs {runs})',
          flush=True)
    # Merge into any existing results so a single panel can be re-measured on its own.
    results = json.loads(out_data.read_text()) if out_data.exists() else {}
    results.update({'K': K, 'N': N, 'tts': studies.TTS, 'sigma': studies.GN['sigma'],
                    'charge_resolution': studies.CHARGE_RESOLUTION, 'geom': Path(GEOM).name})
    results.setdefault('runs_per_point', {})
    measured = _bench_recon(K, N, warmup, runs, want=tuple(panels))
    for name, r in measured.items():
        results[name] = r
        results['runs_per_point'][name] = runs
    out_data.parent.mkdir(parents=True, exist_ok=True)
    out_data.write_text(json.dumps(results))
    print(f'wrote {out_data}')


# ----------------------------------------------------------------------------- plot
ESTIMATORS = {'mode': half_sample_mode, 'median': np.median,
              'min': np.min, 'mean': np.mean}


def bootstrap_se(xs, est, n_boot=400, seed=0):
    """Standard error of the point estimator, by resampling the timed calls.

    The mode has no closed-form standard error (sigma/sqrt(n) is the SE of the MEAN), so
    resample with replacement and take the spread of the recomputed estimator.
    """
    xs = np.asarray(xs, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(xs), size=(n_boot, len(xs)))
    return float(np.std([ESTIMATORS[est](xs[i]) for i in idx]))


def _estimate_se(d, est):
    """Per-point standard error in ms (NaN where raw samples were not stored)."""
    if 'runs' not in d:
        return np.full(len(d['N']), np.nan)
    return np.array([np.nan if r is None else bootstrap_se(r, est) * 1000
                     for r in d['runs']])


def _estimate(d, est):
    """Per-point statistic in ms, from the stored raw samples where available.

    ``runs`` holds every timed call, so the estimator is a plotting choice. Results
    written before raw samples were stored fall back to whatever summary they carry.
    """
    if 'runs' in d:
        return np.array([np.nan if r is None else ESTIMATORS[est](r) * 1000
                         for r in d['runs']])
    key = 'mode' if 'mode' in d else 'mean'
    return np.array([np.nan if x is None else x * 1000 for x in d[key]])


def _panel(ax, res, K_values, title, max_n=None, est='mean', fit=True):
    import matplotlib.pyplot as plt
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(K_values)))
    dropped = 0
    fits = {}
    for c, K in zip(colors, K_values):
        d = res[str(K)] if str(K) in res else res[K]
        N = np.array(d['N'], float)
        m = _estimate(d, est)
        se = _estimate_se(d, est)
        if max_n is not None:                     # both panels share one N range
            keep = N <= max_n
            N, m, se = N[keep], m[keep], se[keep]
        dropped += int(np.isnan(m).sum())
        ok = ~np.isnan(m)
        if fit and ok.sum() >= 2:
            # time = slope*N + intercept: the slope is the marginal cost per ray (throughput),
            # the intercept the N-independent per-call overhead.
            slope, icept = np.polyfit(N[ok], m[ok], 1)
            fits[K] = (slope, icept)
            xf = np.linspace(0.0, N[ok].max(), 100)
            ax.plot(xf, slope * xf + icept, '-', color=c, lw=1.4, alpha=0.85, zorder=1)
            label = f'K={K}  ({slope * 1e6:.0f} ms/10$^6$, {icept:+.1f} ms)'
            ax.errorbar(N[ok], m[ok], yerr=se[ok], fmt='o', color=c, label=label,
                        capsize=3, elinewidth=1.2, zorder=2)
        else:
            ax.errorbar(N[ok], m[ok], yerr=se[ok], fmt='o-', color=c, label=f'K={K}',
                        capsize=3, elinewidth=1.2)
    ax.set_xlabel('Number of Rays (N)')
    ax.set_ylabel('Time (ms)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return dropped, fits


def plot_results(out_data, out, max_n=PLOT_MAX_N, est='mean', fit=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    if not out_data.exists():
        print(f'[skip] no benchmark results at {out_data} -- run --generate-data first')
        return
    r = json.loads(out_data.read_text())
    K = r['K']
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    # Reverse-mode must evaluate the primal to build the tape, and jacfwd carries it along
    # with the tangents, so the derivative panels include the forward pass by construction.
    panels = (('pred', 'Prediction Only', 'prediction_timing'),
              ('grad', 'Prediction and Gradient', 'gradient_timing'),
              ('fisher', 'Prediction and Fisher Matrix', 'fisher_timing'))
    for key, title, name in panels:
        if key not in r:
            print(f'[skip] no {key!r} results in {out_data}'); continue
        fig, ax = plt.subplots(figsize=(6, 4))
        dropped, fits = _panel(ax, r[key], K, title, max_n=max_n, est=est, fit=fit)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'{name}.{ext}', dpi=200, bbox_inches='tight')
        plt.close(fig)
        note = f'  ({dropped} point(s) missing -- see the benchmark log)' if dropped else ''
        print(f'wrote {out / name}.pdf (+png){note}')
        for k_fit, (slope, icept) in fits.items():
            print(f'    K={k_fit}: {slope * 1e6:7.1f} ms per 10^6 rays, {icept:+6.2f} ms offset')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate-data', action='store_true')
    ap.add_argument('--plot-results', action='store_true')
    ap.add_argument('--backend', choices=['local', 's3df'], default='local')
    ap.add_argument('--full', action='store_true', help='paper grid (K=1..9 step 2, N up to 1M)')
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--runs', type=int, default=30)
    ap.add_argument('--out', default=None, help='figure output dir')
    ap.add_argument('--out-data', default=None, help='benchmark results JSON')
    ap.add_argument('--max-n', type=float, default=PLOT_MAX_N,
                    help='upper N limit for both panels (default 1e6); 0 = no limit')
    ap.add_argument('--no-fit', action='store_true', help='markers only, no linear fits')
    ap.add_argument('--panels', nargs='+', choices=list(PANELS), default=list(PANELS),
                    help='which panel(s) to (re)measure; results are merged into the JSON')
    ap.add_argument('--estimator', choices=list(ESTIMATORS), default='mean',
                    help='per-point statistic over the timed runs. With 500 runs/point the '
                         'mean and the mode agree to <1%; at low run counts prefer mode, '
                         'which rejects the one-sided slow outliers.')
    ap.add_argument('--account', default='mli:cider-ml',
                    help='SLURM account for --backend s3df (the ampere partition is fixed: '
                         'timing must come from one GPU type)')
    a = ap.parse_args()
    out_fig = Path(a.out) if a.out else paths.figure_dir()
    out_data = (Path(a.out_data) if a.out_data else
                paths.data_dir('compute_performance', 'local') / 'timing_recon.json')
    both = not (a.generate_data or a.plot_results)
    if a.generate_data or both:
        generate_data(a.backend, a.full, a.warmup, a.runs, out_data, a.account, tuple(a.panels))
    if (a.plot_results or both) and a.backend != 's3df':
        plot_results(out_data, out_fig, max_n=(a.max_n or None), est=a.estimator,
                     fit=not a.no_fit)


if __name__ == '__main__':
    main()
