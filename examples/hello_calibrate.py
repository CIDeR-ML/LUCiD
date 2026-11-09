"""hello_calibrate — recover the detector's optical parameters from calibration light.

The published estimator (Neyman chi-square, every per-PMT gain profiled in closed form, damped
Gauss-Newton) through the library's four steps: a model, closure data, a problem, ``calibrate``;
then the Cramer-Rao bound from the model. Seven global optical parameters are recovered from a
perturbed start while all 10764 per-PMT gains (a 5% spread baked into the data) are profiled at
every step. The knobs are this example's, chosen for speed, not the paper's recipe.

About 4 min on one GPU at the defaults; much slower on CPU (set JAX_PLATFORMS=cpu to force it).
Run:  python examples/hello_calibrate.py [--n-photons N] [--steps N]
"""
import argparse

import jax.numpy as jnp
import numpy as np

from lucid.detector_params import DetectorParams
from lucid.fitting import (FieldParams, CalibrationForward, CalibrationProblem, calibrate,
                           closure_data, crb)
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources import laser_source, isotropic_source

GEOM = 'config/SK_like_geom_config.json'
FIELDS = ['g', 'scatter_length', 'mie_scatter_length', 'absorption_length',
          'wall_reflection_rate', 'sensor_reflection_rate', 'qe']


def main(n_photons=1_000_000, steps=100):
    # --8<-- [start:calibrate]
    det = generate_detector(GEOM); NS = len(det.all_points)
    top, bot, R = det.H/2 - .1, -det.H/2 + .1, det.r

    # truth optics; per-PMT qe_corrections is the gain map (the fit profiles it)
    dp = DetectorParams.from_flat(scatter_length=70., mie_scatter_length=3000., g=0.9,
                                  wall_reflection_rate=.2, sensor_reflection_rate=.2,
                                  absorption_length=60., qe=0.07, qe_corrections=jnp.ones(NS))
    # source diversity is the lever: lasers from the caps and the wall, and an isotropic flasher
    sources = [laser_source(position=[0, 0, top], direction=[0, 0, -1], intensity=1e6),
               laser_source(position=[0, 0, bot], direction=[0, 0, 1], intensity=1e6),
               laser_source(position=[R-.1, 0, 0], direction=[-1, 0, 0], intensity=1e6),
               isotropic_source(position=[0, 0, 0], intensity=1e6)]
    sim = setup_event_simulator(GEOM, n_photons, source='light_source', temperature=None, K=8,
                                wavelength_mode=False,
                                n_cap=100, n_angular=150, n_height=100)

    params = FieldParams(dp, FIELDS)                      # fit these leaves of dp, in log space
    model = CalibrationForward(sim, sources, params)      # predicted charge, every source
    theta_true = params.theta_from_physical()

    # Closure data: the model at truth with a 5% per-PMT gain spread baked in, averaged over 8
    # draws (closure_data's default, as the paper's calibration does). With real data: the
    # observed (n_sources, n_sensors) charge.
    rng = np.random.default_rng(0)
    k_true = np.exp(rng.normal(0.0, 0.05, NS)); k_true /= np.exp(np.mean(np.log(k_true)))
    data = closure_data(model, theta_true, k_true)

    problem = CalibrationProblem(model, data, jacobian_draws=2)
    start = theta_true + rng.uniform(-.15, .15, params.P)
    res = calibrate(problem, start, steps=steps, refresh=15, max_step=0.08, polyak=0)
    sigma = crb(model, theta_true)['sigma']               # Cramer-Rao bound, fractional, per field
    # res['real']: the fitted globals (compare params.to_real(theta_true)); res['gains']: the gains
    # --8<-- [end:calibrate]

    truth, fit = params.to_real(theta_true), res['real']
    print(f'{"param":22s}{"truth":>9s}{"start":>9s}{"fit":>9s}{"err":>8s}{"±CRB":>8s}')
    for i, f in enumerate(FIELDS):
        print(f'{f:22s}{truth[i]:9.3f}{np.exp(start[i]):9.3f}{fit[i]:9.3f}'
              f'{fit[i]/truth[i]-1:+8.1%}{sigma[i]:8.1%}')
    print(f'per-PMT gains: RMS error {np.std(res["gains"] / k_true - 1):.2%} over {NS} sensors')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-photons', type=int, default=1_000_000, help='photons per source [1000000]')
    ap.add_argument('--steps', type=int, default=100, help='Gauss-Newton steps [100]')
    main(**vars(ap.parse_args()))
