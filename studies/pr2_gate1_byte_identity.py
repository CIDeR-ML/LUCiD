"""GATE 1 — single-medium byte-identity of the unified factory step.

make_photon_step('sample'|'update_factors', has_interface=False) must be array_equal to the
legacy photon_iteration_sample / photon_iteration_update_factors; has_interface=True must
match the legacy *_nested. Tested per-photon and vmapped, eager and jit, plus gradient.
"""
import jax, jax.numpy as jnp
import numpy as np
from lucid.simulation.reflection import ScalarReflection
from lucid.simulation import photon_step as P
from lucid.simulation.photon_step_factory import make_photon_step

REFL = ScalarReflection(wall_rate=jnp.float32(0.2), sensor_rate=jnp.float32(0.2))


def rand_inputs(n, seed):
    r = np.random.RandomState(seed)
    def unit(a):
        a = a / np.linalg.norm(a, axis=-1, keepdims=True)
        return jnp.asarray(a, jnp.float32)
    pos = jnp.asarray(r.uniform(-15, 15, (n, 3)), jnp.float32)
    direction = unit(r.normal(size=(n, 3)))
    normal = unit(r.normal(size=(n, 3)))
    time = jnp.asarray(r.uniform(0, 50, n), jnp.float32)
    sdist = jnp.asarray(r.uniform(0.1, 30, n), jnp.float32)
    scat = jnp.asarray(r.uniform(20, 300, n), jnp.float32)
    mie = jnp.asarray(r.uniform(1e3, 1e4, n), jnp.float32)
    g = jnp.full(n, 0.95, jnp.float32)
    absl = jnp.asarray(r.uniform(20, 300, n), jnp.float32)
    hit_sensor = jnp.asarray(r.rand(n) < 0.4)
    lam = jnp.full(n, 420.0, jnp.float32)
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    sol = jnp.full(n, 0.2254, jnp.float32)
    hit_iface = jnp.asarray(r.rand(n) < 0.3)
    mid = jnp.asarray((r.rand(n) < 0.5).astype(np.int32))
    return (pos, direction, time, sdist, normal, scat, mie, g, absl,
            hit_sensor, lam, keys, sol, hit_iface, mid)


def vrun(fn, args, with_iface, ufmode=False):
    # in_axes: per-photon over everything except g(None? no, g per-photon here), refl_params(None)
    base_axes = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # 13 vmapped + refl_params(None)
    pos, dirn, time, sdist, normal, scat, mie, g, absl, hs, lam, keys, sol, hiface, mid = args
    if with_iface:
        f = lambda *a: fn(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], REFL, a[8],
                          a[9], a[10], a[11], a[12], a[13], a[14], 1.48, 1.33)
        ax = (0,)*13 + (0, 0)
        return jax.vmap(f, in_axes=ax)(pos, dirn, time, sdist, normal, scat, mie, g, absl,
                                       hs, lam, keys, sol, hiface, mid)
    else:
        f = lambda *a: fn(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], REFL, a[8],
                          a[9], a[10], a[11], a[12])
        ax = (0,)*13
        return jax.vmap(f, in_axes=ax)(pos, dirn, time, sdist, normal, scat, mie, g, absl,
                                       hs, lam, keys, sol)


def eq(a, b):
    return all(bool(jnp.array_equal(x, y)) for x, y in zip(a, b))


def main():
    print("backend:", jax.default_backend())
    nested_sample = P.photon_iteration_sample_nested
    nested_uf = P.make_photon_iteration_update_factors_nested_safe()
    legacy_uf = P.make_photon_iteration_update_factors_safe()

    fac_s0 = make_photon_step('sample', False)
    fac_s1 = make_photon_step('sample', True)
    fac_u0 = make_photon_step('update_factors', False)
    fac_u1 = make_photon_step('update_factors', True)

    ok = True
    for seed in [0, 1, 7, 42, 123]:
        a = rand_inputs(256, seed)
        # SAMPLE no-interface
        r_leg = vrun(P.photon_iteration_sample, a, False)
        r_fac = vrun(fac_s0, a, False)
        t1 = eq(r_leg, r_fac)
        # SAMPLE interface
        r_legn = vrun(nested_sample, a, True)
        r_facn = vrun(fac_s1, a, True)
        t2 = eq(r_legn, r_facn)
        # UPDATE no-interface
        r_legu = vrun(legacy_uf, a, False)
        r_facu = vrun(fac_u0, a, False)
        t3 = eq(r_legu, r_facu)
        # UPDATE interface
        r_legun = vrun(nested_uf, a, True)
        r_facun = vrun(fac_u1, a, True)
        t4 = eq(r_legun, r_facun)
        # jit
        t5 = eq(jax.jit(lambda aa: vrun(P.photon_iteration_sample, aa, False))(a),
                jax.jit(lambda aa: vrun(fac_s0, aa, False))(a))
        print(f"seed {seed:3d}: sample={t1} sample_iface={t2} update={t3} update_iface={t4} jit={t5}")
        ok = ok and t1 and t2 and t3 and t4 and t5

    # gradient byte-identity (expected-value step, no interface) wrt scatter_length
    a = rand_inputs(128, 5)
    def loss(fn, scat):
        args = a[:5] + (scat,) + a[6:]
        out = vrun(fn, args, False)
        return jnp.sum(out[3]) + jnp.sum(out[5])   # detect_prob + continuing_factor
    g_leg = jax.grad(lambda s: loss(legacy_uf, s))(a[5])
    g_fac = jax.grad(lambda s: loss(fac_u0, s))(a[5])
    tg = bool(jnp.array_equal(g_leg, g_fac))
    print(f"gradient byte-identical (update, no-iface): {tg}")
    ok = ok and tg

    print("\nGATE 1:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
