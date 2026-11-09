"""hello_calibrate_own_optimiser — one calibration problem, four solvers, one bound.

The problem is the published estimator: the Neyman residual with every per-PMT gain profiled in
closed form, on a forward redrawn every step. Any optimiser can run it:

1. the default, damped Gauss-Newton: ``calibrate(problem, start)``;
2. any optax rule, one argument away: ``calibrate(problem, start, tx=optax.adam(...))``, which
   keeps the step clip, the Polyak readout and ``fix``;
3. your own loop: ``problem.grad_metric_loss(theta, step)`` hands you the published gradient
   ``Jᵀr`` (gains profiled, Jacobian on a stream independent of the residual's), here under a
   plain optax update;
4. scipy's ``least_squares`` on one fixed draw, from ``problem.residuals`` and ``problem.jacobian``.

Two isotropic flashers at very different path lengths to the wall fit the absorption length and
the QE, 20% off at the start, with a 5% per-PMT gain spread in the data. The Cramer-Rao bound is
printed next to the fits: with this light, the absorption-QE valley is shallow, and the bound says
how far along it any solver may wander. Adam has no curvature and crawls along such directions,
which is why Gauss-Newton is the default.

Run:  python examples/hello_calibrate_own_optimiser.py [--n-photons N] [--steps N]   (one GPU;
      slower on CPU)
"""
import argparse

import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import least_squares

from lucid.detector_params import DetectorParams
from lucid.fitting import (FieldParams, CalibrationForward, CalibrationProblem, calibrate,
                           closure_data, crb)
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources import isotropic_source

GEOM = 'config/SK_like_geom_config.json'
FIELDS = ['absorption_length', 'qe']


def main(n_photons=1_000_000, steps=60):
    det = generate_detector(GEOM); NS = len(det.all_points); R = det.r
    dp = DetectorParams.from_flat(scatter_length=70., absorption_length=60., qe=0.07,
                                  qe_corrections=jnp.ones(NS))
    # Two flasher positions with very different path lengths to the wall: absorption depends on the
    # path, qe is a pure scale, so the pair separates them. One flasher alone could not.
    sources = [isotropic_source(position=[0, 0, 0], intensity=1e6),
               isotropic_source(position=[0.85 * R, 0, 0], intensity=1e6)]
    sim = setup_event_simulator(GEOM, n_photons, source='light_source', temperature=None, K=8,
                                wavelength_mode=False)

    params = FieldParams(dp, FIELDS)
    model = CalibrationForward(sim, sources, params)
    theta_true = params.theta_from_physical()
    rng = np.random.default_rng(0)
    k_true = np.exp(rng.normal(0.0, 0.05, NS)); k_true /= np.exp(np.mean(np.log(k_true)))
    problem = CalibrationProblem(model, closure_data(model, theta_true, k_true), jacobian_draws=2)
    start = theta_true + np.log([0.8, 1.2])

    # 1. The default: damped Gauss-Newton.
    # --8<-- [start:gn]
    gn = calibrate(problem, start, steps=steps, refresh=3, polyak=20)
    # --8<-- [end:gn]
    # 2. Adam on the same problem, through the same driver.
    # --8<-- [start:adam]
    adam = calibrate(problem, start, tx=optax.adam(2e-2), steps=steps, refresh=3, polyak=20)
    # --8<-- [end:adam]
    # 3. Your own loop over the problem's gradient. A Jacobian refresh every step here; the driver
    #    above reuses it for `refresh` steps.
    # --8<-- [start:own]
    opt = optax.adam(2e-2)
    theta = jnp.asarray(start, jnp.float32); state = opt.init(theta)
    for step in range(steps):
        g, _, loss = problem.grad_metric_loss(theta, step)
        updates, state = opt.update(g, state)
        theta = optax.apply_updates(theta, updates)
    # --8<-- [end:own]
    # 4. scipy, on one fixed draw (common random numbers, so the objective is deterministic).
    # --8<-- [start:scipy]
    sol = least_squares(lambda t: np.ravel(problem.residuals(t, step=0)), start,
                        jac=lambda t: np.reshape(problem.jacobian(t, step=0), (-1, params.P)),
                        max_nfev=steps)
    # --8<-- [end:scipy]

    sigma = crb(model, theta_true)['sigma']
    truth = params.to_real(theta_true)
    fits = (gn['real'], adam['real'], params.to_real(theta), params.to_real(sol.x))
    print(f'{"param":20s}{"truth":>8s}{"GN":>8s}{"Adam":>8s}{"own":>8s}{"scipy":>8s}{"±CRB":>8s}')
    for i, f in enumerate(FIELDS):
        errs = [r[i] / truth[i] - 1 for r in fits]
        print(f'{f:20s}{truth[i]:8.3f}' + ''.join(f'{e:+8.1%}' for e in errs) + f'{sigma[i]:8.1%}')
    print(f'final loss: GN {gn["loss"][-1]:.1f}  Adam {adam["loss"][-1]:.1f}  own {loss:.1f}  '
          f'(at truth {problem.loss(theta_true, steps):.1f})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-photons', type=int, default=1_000_000, help='photons per source [1000000]')
    ap.add_argument('--steps', type=int, default=60,
                    help='steps of each solver (scipy: evaluations) [60]')
    main(**vars(ap.parse_args()))
