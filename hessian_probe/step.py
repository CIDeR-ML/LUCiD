"""Bisect the AD break at the SINGLE-STEP level: jax.grad of the per-photon step outputs w.r.t.
optical params, on (a) the PLAIN photon_iteration_update_factors and (b) the custom_vjp-wrapped
'safe' step. Tells us if the step itself is differentiable and whether the custom_vjp/nan_to_num
zeros it. FD reference with the same key."""
import sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/sdf/group/neutrino/omara/LUCiD_hessian')
from lucid.simulation.photon_step import (photon_iteration_update_factors,
                                           make_photon_iteration_update_factors_safe)
from lucid.simulation.reflection import scalar_reflection, ScalarReflection

jax.config.update('jax_platform_name', 'cpu')

# one photon, generic geometry
pos = jnp.array([0., 0., 0.]); direction = jnp.array([0., 0., 1.]); t = jnp.array(0.)
Dd = jnp.array(3.0)                       # surface distance
normal = jnp.array([0., 0., -1.]); hit_sensor = jnp.array(1.0)
mie = jnp.array(3000.); g = jnp.array(0.9); lam = jnp.array(400.); c = jnp.array(0.2998)
rk = jax.random.PRNGKey(3)
refl = ScalarReflection(wall_rate=jnp.array(0.2), sensor_rate=jnp.array(0.2))

safe_step = make_photon_iteration_update_factors_safe(scalar_reflection)
OUT = ['new_pos', 'new_dir', 'new_time', 'detect_prob', 'refl_atten', 'continuing', 'logp_inc']


def call_plain(sl, al):
    return photon_iteration_update_factors(pos, direction, t, Dd, normal, sl, mie, g, refl, al,
                                           hit_sensor, lam, rk, c, reflection_fn=scalar_reflection)


def call_safe(sl, al):
    return safe_step(pos, direction, t, Dd, normal, sl, mie, g, refl, al, hit_sensor, lam, rk, c)


SL0, AL0 = jnp.array(50.0), jnp.array(400.0)
for oi, name in [(3, 'detect_prob'), (5, 'continuing_factor'), (6, 'logp_increment'), (2, 'new_time')]:
    for pname, idx in [('scatter_length', 0), ('absorption_length', 1)]:
        def fp(x, fn=call_plain, oi=oi, idx=idx):
            args = [SL0, AL0]; args[idx] = x
            return jnp.sum(fn(*args)[oi])
        gp = float(jax.grad(fp)(SL0 if idx == 0 else AL0))
        gs = float(jax.grad(lambda x, **k: fp(x, fn=call_safe))(SL0 if idx == 0 else AL0))
        x0 = float(SL0 if idx == 0 else AL0); h = 1e-3 * x0
        fd = float((fp(jnp.array(x0 + h)) - fp(jnp.array(x0 - h))) / (2 * h))
        flag = '' if abs(gp - fd) < 1e-6 + 0.05 * abs(fd) else '  <-- AD!=FD'
        print(f'{name:18s} d/d{pname:18s}: AD_plain={gp:+.5e}  AD_safe={gs:+.5e}  FD={fd:+.5e}{flag}')
