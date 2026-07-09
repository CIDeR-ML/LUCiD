#!/usr/bin/env python3
"""Figure: computational performance -- simulator timing vs number of rays, per K.

Times the JAX event simulator over a grid of N (rays) x K (per-ray propagation
steps), for prediction-only and prediction+gradient, and plots time-vs-N with one
curve per K. This is a live GPU benchmark (it runs the simulator), so it follows
the generate/plot split:

    python fig_compute_performance.py                       # minimal local benchmark + plot
    python fig_compute_performance.py --plot-results         # plot existing results
    python fig_compute_performance.py --generate-data --backend s3df --full   # full grid on a GPU

Defaults are deliberately small (K=1,3; N up to 200k; few timing runs) so it runs
on a laptop/CPU in a couple of minutes; --full does the paper grid (K=1..8, N up to
2M) and --backend s3df submits it to a GPU partition.
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
from analysis.paper.utils import paths                  # noqa: E402

GEOM = str(REPO_ROOT / 'config' / 'SK_geom_config.json')
PHYS = str(REPO_ROOT / 'config' / 'SK_physics_config.json')
DEF_K, DEF_N = [1, 3], [10000, 100000, 200000]
FULL_K = [1, 2, 3, 4, 5, 6, 7, 8]
FULL_N = [10000, 100000, 250000, 500000, 750000, 1000000, 1500000, 2000000]
IMAGE = '/sdf/data/neutrino/cjesus/software/images/lucid.sif'
BINDS = '/sdf,/fs,/sdf/scratch,/lscratch,/cvmfs'
ENV_BASE = '/sdf/data/neutrino/cjesus/python_envs/lucid'   # CUDA jaxlib layered over the image


# ------------------------------------------------------------------------ benchmark
def _benchmark(K_values, N_values, warmup, runs, gradient):
    import jax
    import jax.numpy as jnp
    from jax import jit, value_and_grad
    from lucid.geometry import generate_detector
    from lucid.simulation import setup_event_simulator
    from lucid.utils import generate_random_point_inside_cylinder, generate_random_params
    from lucid.detector_params import ParticleParams, isotropic_source
    from lucid.losses import compute_simplified_loss

    detpts = jnp.array(generate_detector(GEOM).all_points)
    track = ParticleParams(energy=jnp.array(800.0, jnp.float32),
                           position=jnp.zeros(3, jnp.float32),
                           theta=jnp.array(jnp.pi / 3, jnp.float32),
                           phi=jnp.array(jnp.pi / 4, jnp.float32),
                           t0=jnp.array(0.0, jnp.float32))

    def mk_sim(N, K):
        return setup_event_simulator(GEOM, N, temperature=None, K=K, physics_config=PHYS,
                                     default_detector_params=True, hit_mode='aggregated',
                                     max_candidates_per_ray=4)

    key = jax.random.PRNGKey(0)
    true_data = None
    if gradient:                                          # fixed reference event for the loss
        st = mk_sim(100000, 2)
        key, sk = jax.random.split(key)
        true_data = jax.lax.stop_gradient(st(track, sk))

    res = {K: {'N': [], 'mean': [], 'std': []} for K in K_values}
    for K in K_values:
        for N in N_values:
            sim = mk_sim(N, K)
            if gradient:
                @jit
                def call(p, _k):
                    def lf(pp):
                        return compute_simplified_loss(detpts, *true_data, *sim(pp, _k),
                                                       lambda_time=0.0)
                    return value_and_grad(lf)(p)
                gen = lambda k: (generate_random_params(k), k)
            else:
                sim_j = jit(sim)
                call = lambda p, _k: sim_j(p, _k)
                gen = lambda k: (track, k)

            for _ in range(warmup + runs):
                key, sk = jax.random.split(key)
            key = jax.random.PRNGKey(0)                   # reset the stream for the timed loop
            ts = []
            for i in range(warmup + runs):
                key, sk = jax.random.split(key)
                p, kk = gen(sk)
                t0 = time.time()
                r = call(p, kk)
                jax.tree.map(lambda x: x.block_until_ready(), r)
                if i >= warmup:
                    ts.append(time.time() - t0)
            res[K]['N'].append(N)
            res[K]['mean'].append(float(np.mean(ts)))
            res[K]['std'].append(float(np.std(ts)))
            print(f"  {'grad' if gradient else 'sim '} K={K} N={N:>8}: "
                  f"{np.mean(ts) * 1000:8.2f} ms", flush=True)
    return res


def generate_data(backend, full, warmup, runs, out_data):
    K, N = (FULL_K, FULL_N) if full else (DEF_K, DEF_N)
    if backend == 's3df':
        script = Path(__file__).resolve()
        cmd = (f"APPTAINERENV_PYTHONUSERBASE={ENV_BASE} APPTAINERENV_PYTHONPATH='' "
               f"apptainer exec --nv -B {BINDS} {IMAGE} /opt/conda/bin/python3 {script} "
               f"--generate-data --backend local {'--full' if full else ''} "
               f"--warmup {warmup} --runs {runs} --out {out_data.parent}")
        sb = out_data.parent / 'compute_benchmark.sh'
        sb.write_text("#!/bin/bash\n#SBATCH -p ampere\n#SBATCH --gpus 1\n"
                      "#SBATCH --cpus-per-task 4\n#SBATCH --mem 32G\n"
                      "#SBATCH -t 08:00:00\n#SBATCH -A mli:cider-ml\n"
                      f"#SBATCH -o {out_data.parent}/benchmark_%j.log\n\n"
                      "nvidia-smi -L || true\n"
                      f"{cmd}\n")
        # sbatch lives outside the container; run this branch on the login node.
        print(f'[s3df] wrote {sb}; submitting')
        subprocess.run(['sbatch', str(sb)], check=True)
        return
    print(f'[local] benchmarking K={K} N={N} (warmup {warmup}, runs {runs})', flush=True)
    results = {'K': K, 'N': N,
               'sim': _benchmark(K, N, warmup, runs, gradient=False),
               'grad': _benchmark(K, N, warmup, runs, gradient=True)}
    out_data.write_text(json.dumps(results))
    print(f'wrote {out_data}')


# ----------------------------------------------------------------------------- plot
def _panel(ax, res, K_values, title):
    import matplotlib.pyplot as plt
    colors = plt.cm.viridis(np.linspace(0, 1, len(K_values)))
    for c, K in zip(colors, K_values):
        d = res[str(K)] if str(K) in res else res[K]
        N = np.array(d['N']); m = np.array(d['mean']) * 1000; s = np.array(d['std']) * 1000
        ax.plot(N, m, 'o-', color=c, label=f'K={K}')
        ax.fill_between(N, m - s, m + s, alpha=0.3, color=c)
    ax.set_xlabel('Number of Rays (N)'); ax.set_ylabel('Time (ms)')
    ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend()


def plot_results(out_data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    if not out_data.exists():
        print(f'[skip] no benchmark results at {out_data} -- run --generate-data first'); return
    r = json.loads(out_data.read_text())
    K = r['K']
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for key, title, name in (('sim', 'Prediction Only', 'simulation_timing'),
                             ('grad', 'Prediction and Gradient Calculation', 'gradient_timing')):
        fig, ax = plt.subplots(figsize=(6, 4))
        _panel(ax, r[key], K, title)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'{name}.{ext}', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'wrote {out / name}.pdf (+png)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate-data', action='store_true')
    ap.add_argument('--plot-results', action='store_true')
    ap.add_argument('--backend', choices=['local', 's3df'], default='local')
    ap.add_argument('--full', action='store_true', help='paper grid (K=1..8, N up to 2M)')
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out_fig = Path(a.out) if a.out else paths.figure_dir()
    out_data = paths.data_dir('compute_performance', 'local') / 'timing.json'
    both = not (a.generate_data or a.plot_results)
    if a.generate_data or both:
        generate_data(a.backend, a.full, a.warmup, a.runs, out_data)
    if (a.plot_results or both) and a.backend != 's3df':
        plot_results(out_data, out_fig)


if __name__ == '__main__':
    main()
