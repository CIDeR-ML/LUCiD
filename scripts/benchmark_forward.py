"""benchmark_forward — warm wall time of what the fits run, vs photon count N and scatter steps K.

``--what track``: the reconstruction model, ``lucid.fitting.ReconModel.build`` at the published
``MODEL_RECIPE`` with N and K replaced: its per-PMT forward (``perpmt``), its ``loss`` and its
``grad``, against pseudo-data from one sampled forward of the same track.

``--what calibration``: the calibration forward, a ``CalibrationForward`` over a light-source
simulator (one laser and two isotropic sources, one wavelength, the physics config's optics), and
its Jacobian (``CalibrationForward.jacobian``), the cost of a Gauss-Newton refresh.

Each time is the best of ``--repeats`` calls after one untimed call, so JIT is excluded. The
paper's timing figure is ``analysis/paper/fig_compute_performance.py``, not this script.

Run:  python scripts/benchmark_forward.py                 # defaults: SK_like, both
      python scripts/benchmark_forward.py --what track --n 100000 500000 --k 4 8 --repeats 3
"""
import argparse
import time
from dataclasses import replace

import jax
import jax.numpy as jnp

from lucid.detector_params import load_detector_params
from lucid.fitting import (MODEL_RECIPE, CalibrationForward, CalibrationParams, ReconModel,
                           track_from_vec9, vec9_from_track)
from lucid.simulation import setup_event_simulator
from lucid.sources import isotropic_source, laser_source

# The paper's propagation grid (analysis/paper/utils/studies.py PIPELINE['grid']).
GRID = dict(n_cap=80, n_angular=120, n_height=80, max_candidates_per_ray=8)
TTS = 2.1               # the paper's data TTS and likelihood sigma (ns)


def best_ms(fn, repeats):
    """Best wall time of ``fn()`` in ms over ``repeats`` calls, after one untimed (compiling) call."""
    jax.block_until_ready(fn())
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    return 1e3 * min(ts)


def bench_track(args, N, K):
    model = ReconModel.build(args.geom, args.physics, 'muon', sigma=TTS,
                             recipe=replace(MODEL_RECIPE, n_rays=N, K=K), grid=GRID)
    # a 1 GeV muon at the centre, off every symmetry axis; the data are one sampled forward of it
    t9 = jnp.asarray(vec9_from_track(1000.0, [0.0, 0.0, 0.0], [0.5, 0.5, 0.7071]))
    dp = load_detector_params(args.physics, num_sensors=model.ND)
    data_sim = setup_event_simulator(
        args.geom, N, source='particle', detection='sampled', hits='first_arrival',
        temperature=None, K=K, physics_config=args.physics, particle='muon', wavelength_mode=True,
        default_detector_params=dp._replace(response=dp.response._replace(tts=jnp.asarray(TTS))),
        **GRID)
    oc, ot = data_sim(track_from_vec9(t9), jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(1)
    return dict(forward=best_ms(lambda: model.perpmt(t9, oc, ot, key), args.repeats),
                loss=best_ms(lambda: model.loss(t9, oc, ot, key), args.repeats),
                grad=best_ms(lambda: model.grad(t9, oc, ot, key), args.repeats))


def bench_calibration(args, N, K):
    sim = setup_event_simulator(args.geom, N, source='light_source', temperature=None, K=K,
                                physics_config=args.physics, wavelength_mode=False, **GRID)
    dp = load_detector_params(args.physics)
    params = CalibrationParams(n_wavelengths=1)
    theta = jnp.asarray(params.theta_from_physical(
        [dict(scatter_length=float(dp.scattering.scatter_length),
              absorption_length=float(dp.absorption.absorption_length),
              qe=float(dp.response.qe))],
        float(dp.reflection.wall_reflection_rate), float(dp.reflection.wall_fspec),
        float(dp.reflection.sensor_reflection_rate), float(dp.reflection.sensor_fspec)))
    sources = [laser_source(position=[0.0, 0.0, 0.0], direction=[0.0, 0.0, -1.0], intensity=1e8),
               isotropic_source(position=[1.0, 0.0, 0.0], intensity=1e8),
               isotropic_source(position=[-1.0, 0.0, 0.0], intensity=1e8)]
    forward = CalibrationForward(sim, sources, params)
    gains = jnp.ones(forward.NS)
    data_key, forward_key, jacobian_key = jax.random.split(jax.random.PRNGKey(0), 3)
    data = forward(theta, data_key, gains)
    q_floor = 0.01 * float(jnp.mean(data))
    lk = jnp.zeros(forward.NS)
    return dict(forward=best_ms(lambda: forward(theta, forward_key, gains), args.repeats),
                jacobian=best_ms(lambda: forward.jacobian(theta, lk, jacobian_key, data, q_floor),
                                 args.repeats))


def main():
    ap = argparse.ArgumentParser(description='Warm wall time of the track and calibration fits.')
    ap.add_argument('--what', choices=('track', 'calibration', 'both'), default='both')
    ap.add_argument('--geom', default='config/SK_like_geom_config.json')
    ap.add_argument('--physics', default='config/SK_like_physics_config.json')
    ap.add_argument('--n', type=int, nargs='+', default=[100_000, 250_000])
    ap.add_argument('--k', type=int, nargs='+', default=[4, 8])
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()

    benches = {'track': bench_track, 'calibration': bench_calibration}
    what = list(benches) if args.what == 'both' else [args.what]
    print(f'backend: {jax.default_backend()} | {args.geom} | {args.physics} | '
          f'warm, best of {args.repeats}, ms')
    for name in what:
        for N in args.n:
            for K in args.k:
                times = benches[name](args, N, K)
                print(f'{name:12s} N={N:>9d} K={K:>2d}  '
                      + '  '.join(f'{k}={v:9.1f}' for k, v in times.items()), flush=True)


if __name__ == '__main__':
    main()
