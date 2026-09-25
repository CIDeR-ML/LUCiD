"""hello_calibrate_adam — the smallest calibration you can set up yourself: a gradient and Adam.

The simulator is differentiable, so a calibration is just a loss on the per-sensor charge and an
optimizer. Here: two isotropic flashers in an SK-like tank, absorption length and QE fitted in
log space by ``optax.adam`` on a squared-charge loss, starting 20% off. The loop is plain JAX
around the simulator; ``build_calibration_problem`` is used only to get a forward that takes the
parameters as a log-vector.

It is a CLOSURE test: the model reuses the photon draw that made the "data", so the truth has
zero loss and whatever error remains is the optimizer's. With an independent draw the Monte-Carlo
noise sets a loss floor and a plain squared loss fits that noise; handling it (data averaged over
draws, a data-weighted residual) is what ``hello_calibrate.py`` does.

The gradient is FORWARD-mode (``jax.jacfwd``): with two parameters that is two JVPs, the
cheapest and most robust route through the simulator.

Why these two. Their gradients are pathwise and exact. Scattering is not: it changes photon
paths by DISCRETE decisions, and forward-mode AD through the simulator currently returns about a
third of its finite-difference gradient, so a first-order fit of the scattering length drifts.

When to move to ``hello_calibrate.py`` instead: scattering, more parameters, free per-PMT gains, or
the stiff directions (Mie, ``g``, the reflectivities). Adam has no curvature, so it crawls along
those; the recipe there uses Gauss-Newton with profiled gains, the one the paper runs.

Run:  python examples/hello_calibrate_adam.py      (one GPU)
"""
import jax, jax.numpy as jnp, numpy as np, optax
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources import isotropic_source
from lucid.detector_params import DetectorParams
from lucid.fitting import build_calibration_problem

GEOM = 'config/SK_like_geom_config.json'
det = generate_detector(GEOM); NS = len(det.all_points); R = det.r
dp = DetectorParams.from_flat(scatter_length=70., absorption_length=60., qe=0.07,
                              qe_corrections=jnp.ones(NS))
# Two flasher positions with very different path lengths to the wall: absorption depends on the
# path, qe is a pure scale, so the pair separates them. One flasher alone could not.
sources = [isotropic_source(position=[0, 0, 0]), isotropic_source(position=[0.85 * R, 0, 0])]
sim = setup_event_simulator(GEOM, 1_000_000, temperature=None, K=8, is_calibration=True,
                            hit_mode='aggregated', wavelength_mode=False)

FIELDS = ['absorption_length', 'qe']
prob = build_calibration_problem(sim, sources, dp, FIELDS, key=jax.random.PRNGKey(1))
forwards = [m.forward for m in prob['source_models']]
data = [jnp.asarray(q) for q in prob['truth_charge']]        # "observed" charge
model_key = jax.random.PRNGKey(1)                             # the SAME draw: a closure test


def loss(theta):
    """Relative squared charge mismatch, summed over sources."""
    return sum(jnp.sum((f(theta, model_key, model_key) - q) ** 2) / jnp.sum(q ** 2)
               for f, q in zip(forwards, data))


grad = jax.jit(lambda th: jax.jacfwd(loss)(th))
truth = np.exp(prob['theta0'])
theta = jnp.asarray(prob['theta0'] + np.log([0.8, 1.2]))   # start 20% off
opt = optax.adam(0.02)
state = opt.init(theta)
for step in range(200):
    updates, state = opt.update(grad(theta), state)
    theta = optax.apply_updates(theta, updates)
    if step % 50 == 0 or step == 199:
        print(f'step {step:3d}  loss {float(loss(theta)):.3e}  ' +
              '  '.join(f'{f}={v:.3f}' for f, v in zip(FIELDS, np.exp(theta))))

print(f'\n{"param":20s}{"truth":>9s}{"fit":>9s}{"err":>8s}')
for f, t, v in zip(FIELDS, truth, np.exp(theta)):
    print(f'{f:20s}{t:9.3f}{v:9.3f}{v / t - 1:+8.1%}')
