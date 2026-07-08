"""calibrate_campaign — run a calibration campaign (multi-start recovery vs the CRB).

Portable version of the calibration campaign (scripts/campaign, which is SK-hardcoded and
driven by ~15 env vars): from a geometry + physics + a diverse set of calibration sources, run
the Gauss-Newton + per-PMT-Schur fit from several random starts and report each global optical
parameter's recovery (mean +/- std over starts, fraction within the Cramer-Rao bound). The
campaign-scale companion to `examples/hello_calibrate.py`.

Run:  python scripts/calibrate_campaign.py --n-init 5 --out calib_out
"""
import argparse, os
import jax, jax.numpy as jnp, numpy as np
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources import laser_source, isotropic_source
from lucid.detector_params import DetectorParams
from lucid.fitting import build_calibration_problem, fit, crb

FIELDS = ['g', 'scatter_length', 'mie_scatter_length', 'absorption_length',
          'wall_reflection_rate', 'sensor_reflection_rate', 'qe']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geom', default='config/SK_like_geom_config.json')
    ap.add_argument('--physics', default='config/SK_like_physics_config.json')
    ap.add_argument('--n-init', type=int, default=5, help='number of random starts')
    ap.add_argument('--nphot', type=int, default=1_000_000)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--perturb', type=float, default=0.15, help='+/- start perturbation in log-param')
    ap.add_argument('--out', default=None, help='optional output dir for the results npz')
    args = ap.parse_args()

    det = generate_detector(args.geom); NS = len(det.all_points)
    top, bot, R = det.H/2 - .1, -det.H/2 + .1, det.r
    dp = DetectorParams.from_flat(scatter_length=70., mie_scatter_length=3000., g=0.9,
                                  wall_reflection_rate=.2, sensor_reflection_rate=.2,
                                  absorption_length=60., qe=0.07, qe_corrections=jnp.ones(NS))
    sources = [laser_source(position=[0, 0, top], direction=[0, 0, -1], intensity=1e6),
               laser_source(position=[0, 0, bot], direction=[0, 0,  1], intensity=1e6),
               laser_source(position=[R-.1, 0, 0], direction=[-1, 0, 0], intensity=1e6),
               isotropic_source(position=[0, 0, 0], intensity=1e6)]
    sim = setup_event_simulator(args.geom, args.nphot, temperature=None, K=8, is_calibration=True,
                                hit_mode='aggregated', physics_config=args.physics,
                                wavelength_mode=False, n_cap=100, n_angular=150, n_height=100)
    prob = build_calibration_problem(sim, sources, dp, FIELDS, key=jax.random.PRNGKey(1))
    sigma = crb(prob['source_models'], prob['theta_true'], NS)['sigma']
    truth = np.exp(prob['theta0'])
    print(f'{NS} sensors | {len(sources)} sources | {args.n_init} starts | {args.nphot:,} photons')

    fits = []
    for i in range(args.n_init):
        start = prob['theta0'] + np.random.default_rng(i).uniform(-args.perturb, args.perturb,
                                                                  prob['theta0'].shape)
        res = fit(prob['source_models'], prob['truth_charge'], start, NS,
                  steps=args.steps, refresh=15, nb_h=2)
        fits.append(res['theta'])
        print(f'  start {i+1}/{args.n_init} done', flush=True)
    fits = np.array(fits)                                   # (n_init, n_param)
    rel = fits / truth - 1.0                                # fractional error per start

    print(f'\n{"param":22s}{"truth":>10s}{"mean err":>10s}{"std":>8s}{"CRB":>8s}{"in CRB":>8s}')
    for j, f in enumerate(FIELDS):
        me, sd = rel[:, j].mean(), rel[:, j].std()
        frac = float(np.mean(np.abs(rel[:, j]) <= sigma[j]))
        print(f'{f:22s}{truth[j]:10.3f}{me:+9.1%}{sd:8.1%}{sigma[j]:8.1%}{frac:7.0%}')

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        np.savez(os.path.join(args.out, 'calibration_campaign.npz'),
                 fields=FIELDS, truth=truth, fits=fits, sigma=sigma)
        print(f'\nwrote {args.out}/calibration_campaign.npz')


if __name__ == '__main__':
    main()
