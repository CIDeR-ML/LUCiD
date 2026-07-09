"""benchmark_forward — wall-time of the differentiable forward (and its gradient) vs (N, K).

Times the JIT-compiled forward pass, and — for the calibration loss — the value_and_grad, as a
function of photon count N and scatter iterations K. Reports warm timings (JIT excluded) so you
can size runs and check the gradient overhead. A reproducible driver for LUCiD's efficiency
claims (the notebook `computational_performance_evaluation` did this ad hoc).

Run:  python scripts/benchmark_forward.py                 # defaults
      python scripts/benchmark_forward.py --n 100000 500000 --k 4 8 --repeats 3
"""
import argparse, time
import jax, jax.numpy as jnp, numpy as np
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.detector_params import ParticleParams, DetectorParams
from lucid.losses import WC_smooth_loss
from lucid.sources import laser_source

GRID = dict(n_cap=80, n_angular=120, n_height=80)


def _time(fn, repeats):
    fn()  # warm up (JIT compile)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); jax.block_until_ready(fn()); ts.append(time.perf_counter() - t0)
    return 1e3 * min(ts)  # ms, best of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geom', default='config/SK_like_geom_config.json')
    ap.add_argument('--physics', default='config/SK_like_physics_config.json')
    ap.add_argument('--n', type=int, nargs='+', default=[100_000, 250_000])
    ap.add_argument('--k', type=int, nargs='+', default=[4, 8])
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()

    det = generate_detector(args.geom); NS = len(det.all_points)
    print(f'backend: {jax.default_backend()} | {NS} sensors | grid {GRID}')
    print(f'{"mode":12s}{"N":>10s}{"K":>4s}{"forward(ms)":>13s}{"fwd+grad(ms)":>14s}')

    track = ParticleParams.from_cartesian(energy=1000., position=[0., 0., 0.], direction=[1., 0., 0.], t0=0.)
    dp = DetectorParams.from_flat(scatter_length=70., mie_scatter_length=3000., g=0.9,
                                  wall_reflection_rate=.2, sensor_reflection_rate=.2,
                                  absorption_length=60., qe=0.07, qe_corrections=jnp.ones(NS))
    src = laser_source(position=[0, 0, det.H/2 - .1], direction=[0, 0, -1], intensity=1e6)
    pts = jnp.asarray(det.all_points)

    for N in args.n:
        for K in args.k:
            # track forward (per-photon mode)
            trk = setup_event_simulator(args.geom, N, K=K, hit_mode='per_photon', physics_config=args.physics,
                                        default_detector_params=True, particle='muon', wavelength_mode=True,
                                        pos_grad_threshold=K, n_grad_iters=K, **GRID)
            f_trk = _time(lambda: trk(track, jax.random.PRNGKey(0)), args.repeats)
            print(f'{"track":12s}{N:>10d}{K:>4d}{f_trk:>13.1f}{"—":>14s}')

            # calibration forward + gradient (value_and_grad of the calibration loss over DetectorParams)
            cal = setup_event_simulator(args.geom, N, temperature=None, K=K, is_calibration=True,
                                        hit_mode='aggregated', physics_config=args.physics,
                                        wavelength_mode=False, **GRID)
            cal_d = setup_event_simulator(args.geom, N, temperature=None, K=K, is_calibration=True,
                                          hit_mode='aggregated', physics_config=args.physics,
                                          default_detector_params=dp, wavelength_mode=False, **GRID)
            truth = jax.lax.stop_gradient(cal_d(src, jax.random.PRNGKey(1)))
            def loss(x):
                return WC_smooth_loss(pts, *truth, *cal(src, x, jax.random.PRNGKey(2)),
                                      lambda_poisson=1.0, lambda_time=0.0, tau=0.5)
            vg = jax.jit(jax.value_and_grad(loss))
            f_cal = _time(lambda: cal(src, dp, jax.random.PRNGKey(2)), args.repeats)
            g_cal = _time(lambda: vg(dp), args.repeats)
            print(f'{"calib":12s}{N:>10d}{K:>4d}{f_cal:>13.1f}{g_cal:>14.1f}')


if __name__ == '__main__':
    main()
