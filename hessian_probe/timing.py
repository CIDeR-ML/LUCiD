"""Are AD Jacobian/Hessian properly jitted? Separate COMPILE (1st call) from STEADY-STATE, with the
key as a TRACED argument (changing keys must NOT retrace). Compare forward / jacfwd / FD / hessian."""
import os, sys, time
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian'); os.chdir('/sdf/group/neutrino/omara/LUCiD_hessian')
import lucid.simulation.simulator as SIM
import lucid.simulation.photon_step as PS
def plain_factory(reflection_fn=PS.scalar_reflection):
    def step(p, d, t, sd, n, sl, ml, g, rp, al, hs, lam, rk, c):
        return PS.photon_iteration_update_factors(p, d, t, sd, n, sl, ml, g, rp, al, hs, lam, rk, c,
                                                  reflection_fn=reflection_fn)
    return step
SIM.make_photon_iteration_update_factors_safe = plain_factory
from lucid.geometry import generate_detector
from lucid.detector_params import load_detector_params
from lucid.sources.calibration_sources import laser_source
from lucid.fitting import build_calibration_problem

GEOM, PHYS = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json'
NPHOT = 30000; K = 8
det = generate_detector(GEOM); ND = len(det.all_points); Hd = float(getattr(det, 'H', 36.0))
dp = load_detector_params(PHYS, num_sensors=ND)
src = laser_source(position=[0.0, 0.0, Hd / 2 - 0.1], intensity=NPHOT)
sim = SIM.setup_event_simulator(GEOM, NPHOT, temperature=0.2, K=K, is_calibration=True,
                                physics_config=PHYS, hit_mode='aggregated', wavelength_mode=False)
GLOB = ['scatter_length', 'mie_scatter_length', 'g', 'absorption_length',
        'wall_reflection_rate', 'sensor_reflection_rate', 'qe']
prob = build_calibration_problem(sim, [src], dp, GLOB, key=jax.random.PRNGKey(0))
fwd = prob['source_models'][0].forward; th0 = jnp.asarray(prob['theta0'], float); P = th0.shape[0]
c = jnp.asarray(np.random.default_rng(0).standard_normal(ND))

# key is a TRACED ARGUMENT in every jit (so different keys reuse the same compiled program)
f_fwd = jax.jit(lambda th, k: fwd(th, k, k))
f_jac = jax.jit(lambda th, k: jax.jacfwd(lambda t: fwd(t, k, k))(th))
f_grd = jax.jit(lambda th, k: jax.grad(lambda t: jnp.sum(c * fwd(t, k, k)))(th))
f_hes = jax.jit(lambda th, k: jax.hessian(lambda t: jnp.sum(c * fwd(t, k, k)))(th))


def bench(fn, label, reps=10):
    k = jax.random.PRNGKey(0)
    t = time.perf_counter(); jax.block_until_ready(fn(th0, k)); compile_s = time.perf_counter() - t
    ts = []
    for i in range(reps):                       # DIFFERENT key each call → tests retrace
        k = jax.random.PRNGKey(100 + i)
        t = time.perf_counter(); jax.block_until_ready(fn(th0, k)); ts.append(time.perf_counter() - t)
    print(f'  {label:24s} compile={compile_s:6.2f}s   steady={1000*np.median(ts):7.1f} ms   '
          f'(min {1000*min(ts):.1f}, max {1000*max(ts):.1f})')


print(f'ND={ND} P={P} K={K} Nphot={NPHOT}\n')
bench(f_fwd, 'forward (1 pass)')
bench(f_jac, f'jacfwd Jacobian ({P} passes)')
bench(f_grd, 'reverse grad (scalar)')
bench(f_hes, f'jax.hessian ({P}x{P})')


# FD Jacobian (the current SourceModel recipe): P forward evals, forward-diff, jitted forward
def fd_jac(th, k, h=1e-3):
    base = f_fwd(th, k)
    return jnp.stack([(f_fwd(th.at[d].add(h), k) - base) / h for d in range(P)], 1)
fd_j = jax.jit(fd_jac)
bench(fd_j, f'FD Jacobian ({P} fwd evals)')
print(f'\n  steady jacfwd/FD ratio is the apples-to-apples AD-vs-FD Jacobian cost.')
