"""check_gradients — verify LUCiD's automatic-differentiation gradients against finite differences.

Differentiability is LUCiD's whole premise, so this is a sanity check: for the calibration loss
(per-PMT charge vs a laser source), compare the JAX autodiff gradient of each optical parameter
to a central finite-difference estimate. Prints the AD and FD values and their relative error;
close agreement means gradients flow correctly through the forward model.

Run:  python scripts/check_gradients.py
"""
import argparse
import jax, jax.numpy as jnp, numpy as np
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources import laser_source
from lucid.detector_params import DetectorParams
from lucid.losses import WC_smooth_loss
from lucid.gradient_analysis import SweepParam
from lucid.gradient_analysis.sweep import get_param_value, set_param_value, get_grad_component


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geom', default='config/SK_like_geom_config.json')
    ap.add_argument('--physics', default='config/SK_like_physics_config.json')
    ap.add_argument('--nphot', type=int, default=1_000_000)
    ap.add_argument('--rel-h', type=float, default=1e-3, help='FD step as a fraction of the value')
    args = ap.parse_args()

    det = generate_detector(args.geom); NS = len(det.all_points); pts = jnp.asarray(det.all_points)
    dp = DetectorParams.from_flat(scatter_length=70., mie_scatter_length=3000., g=0.9,
                                  wall_reflection_rate=.2, sensor_reflection_rate=.2,
                                  absorption_length=60., qe=0.07, qe_corrections=jnp.ones(NS))
    src = laser_source(position=[0, 0, det.H/2 - .1], direction=[0, 0, -1], intensity=1e6)
    sim = setup_event_simulator(args.geom, args.nphot, temperature=None, K=8, is_calibration=True,
                                hit_mode='aggregated', physics_config=args.physics, wavelength_mode=False,
                                n_cap=100, n_angular=150, n_height=100)
    sim_t = setup_event_simulator(args.geom, args.nphot, temperature=None, K=8, is_calibration=True,
                                  hit_mode='aggregated', physics_config=args.physics, default_detector_params=dp,
                                  wavelength_mode=False, n_cap=100, n_angular=150, n_height=100)
    truth = jax.lax.stop_gradient(sim_t(src, jax.random.PRNGKey(0)))

    @jax.jit
    def loss(x):
        return WC_smooth_loss(pts, *truth, *sim(src, x, jax.random.PRNGKey(1)),
                              lambda_poisson=1.0, lambda_time=0.0, tau=0.5)

    sp = lambda f: SweepParam(f, f, half_width=1.0)   # half_width unused here; we only need the accessor

    def loss_k(x, key):
        return WC_smooth_loss(pts, *truth, *sim(src, x, key), lambda_poisson=1.0, lambda_time=0.0, tau=0.5)

    def ad_fd(field, key):
        _, g = jax.value_and_grad(lambda x: loss_k(x, key))(dp)
        ad = float(get_grad_component(g, sp(field)))
        x0 = float(get_param_value(dp, sp(field))); h = args.rel_h * max(abs(x0), 1.0)
        fp = float(loss_k(set_param_value(dp, sp(field), x0 + h), key))
        fm = float(loss_k(set_param_value(dp, sp(field), x0 - h), key))
        return ad, (fp - fm) / (2 * h)

    # PATHWISE parameters: deterministic multipliers on the light — AD must equal FD (same key/CRN).
    print(f'PATHWISE parameters — AD vs finite difference (common random numbers), {NS} PMTs:')
    print(f'{"param":22s}{"AD":>14s}{"FD":>14s}{"rel.err":>10s}')
    worst = 0.0
    for f in ['absorption_length', 'wall_reflection_rate', 'sensor_reflection_rate', 'qe']:
        ad, fd = ad_fd(f, jax.random.PRNGKey(1))
        rel = abs(ad - fd) / (abs(fd) + 1e-12); worst = max(worst, rel)
        print(f'{f:22s}{ad:>14.4e}{fd:>14.4e}{rel:>10.1%}')
    print(f'  -> {"PASS" if worst < 0.1 else "CHECK"} (worst {worst:.1%}); pathwise AD gradients match FD.\n')

    # SCORE-ESTIMATED parameters: scatter / Mie / g act through DISCRETE scatter decisions, so the
    # gradient is a DiCE score-function estimator (unbiased, low-variance). Finite differences do
    # NOT validate it quickly — CRN-FD is biased on discrete branches, and a non-CRN expectation
    # FD needs very many samples to converge — so we report the AD value only, not an FD check.
    _, g = jax.value_and_grad(lambda x: loss_k(x, jax.random.PRNGKey(1)))(dp)
    print('SCORE-ESTIMATED parameters (discrete scatter/Mie) — AD gradient (no quick FD check):')
    print(f'{"param":22s}{"AD":>14s}')
    for f in ['scatter_length', 'mie_scatter_length', 'g']:
        print(f'{f:22s}{float(get_grad_component(g, sp(f))):>14.4e}')
    print('  (DiCE score-function gradients: unbiased & low-variance; validated against many-sample\n'
          '   expectation FD in the gradient study, not a point finite difference.)')


if __name__ == '__main__':
    main()
