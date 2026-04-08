"""
Test B: Multi-Bounce Jacobian Growth (Self-Contained)
=====================================================
Self-contained toy with ZERO dependency on tools.*. Reimplements only the
essential numerical operations. Should run in under a minute.

Section 1: Pure specular bouncing — Jacobian d(dir_K)/d(dir_0).
Section 2: Sensor tangent gradient profile (Mechanism 1).
Section 3: Multi-bounce + sensor detection (both mechanisms combined).

Run:  python3 tests/test_multibounce_jacobian.py
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np
import time as time_module

# Ensure float32 for speed
jax.config.update("jax_enable_x64", False)


# =============================================================================
# Primitives
# =============================================================================

def normalize(v, eps=1e-6):
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.maximum(norm, eps)


def intersect_cylinder(pos, d, r, h):
    """Single-ray cylinder intersection. Returns (t, part). 0=wall,1=top,2=bot."""
    LARGE = 1e10

    # Wall
    a_w = d[0]**2 + d[1]**2
    b_w = 2.0 * (pos[0]*d[0] + pos[1]*d[1])
    c_w = pos[0]**2 + pos[1]**2 - r**2
    disc_w = b_w**2 - 4*a_w*c_w
    sqrt_d = jnp.sqrt(jnp.maximum(0.0, disc_w))
    denom_w = 2*a_w + 1e-12
    t1 = (-b_w - sqrt_d) / denom_w
    t2 = (-b_w + sqrt_d) / denom_w
    t1, t2 = jnp.minimum(t1, t2), jnp.maximum(t1, t2)
    t_cand = jnp.where(t1 > 1e-6, t1, t2)
    z_hit = pos[2] + t_cand * d[2]
    wall_ok = (disc_w >= 0) & (t_cand > 1e-6) & (jnp.abs(z_hit) <= h/2) & (a_w > 1e-12)
    t_wall = jnp.where(wall_ok, t_cand, LARGE)

    # Caps
    def cap_t(z_cap):
        t_c = jnp.where(jnp.abs(d[2]) > 1e-12, (z_cap - pos[2]) / d[2], LARGE)
        r2 = (pos[0] + t_c*d[0])**2 + (pos[1] + t_c*d[1])**2
        ok = (t_c > 1e-6) & (r2 <= r**2)
        return jnp.where(ok, t_c, LARGE)

    t_top = cap_t(h/2)
    t_bot = cap_t(-h/2)

    ts = jnp.array([t_wall, t_top, t_bot])
    return jnp.min(ts), jnp.argmin(ts)


def cyl_normal(hit, part):
    xy = jnp.sqrt(hit[0]**2 + hit[1]**2) + 1e-10
    wall_n = jnp.array([-hit[0]/xy, -hit[1]/xy, 0.0])
    top_n = jnp.array([0.0, 0.0, -1.0])
    bot_n = jnp.array([0.0, 0.0,  1.0])
    return jnp.where(part == 0, wall_n, jnp.where(part == 1, top_n, bot_n))


def reflect(d, n):
    d = normalize(d)
    n = normalize(n)
    return normalize(d - 2.0 * jnp.sum(d*n) * n)


def sensor_intersect(ray_pos, ray_dir, center, radius, use_fix=False):
    """
    Ray-sphere (sensor) intersection. Returns closest_approach_distance.
    use_fix: if True, sqrt(max(0,disc)+1e-6); if False, sqrt(max(1e-10,disc)).
    """
    d = normalize(ray_dir)
    oc = ray_pos - center
    t_ca = -jnp.sum(oc * d)
    closest = ray_pos + t_ca * d
    dist = jnp.linalg.norm(closest - center)

    a = jnp.sum(d*d)
    b = 2.0 * jnp.sum(oc*d)
    c = jnp.sum(oc*oc) - radius**2
    disc = b**2 - 4*a*c

    # THE CRITICAL SQRT
    sqrt_term = jnp.where(
        use_fix,
        jnp.sqrt(jnp.maximum(0.0, disc) + 1e-6),
        jnp.sqrt(jnp.maximum(1e-10, disc))
    )

    q = jnp.where(b > 0, -0.5*(b + sqrt_term), -0.5*(b - sqrt_term))
    t1 = q / (a + 1e-10)
    t2 = c / (q + jnp.sign(q)*1e-10)
    t_int = jnp.where((t1>0)&(t2>0), jnp.minimum(t1,t2),
                       jnp.where(t1>0, t1, jnp.where(t2>0, t2, -1.0)))

    return dist, t_int, disc


# =============================================================================
# Section 1: Pure bouncing Jacobian
# =============================================================================

def run_section_1():
    print("=" * 72)
    print("SECTION 1: Pure Specular Bouncing — Jacobian d(dir_K)/d(dir_0)")
    print("=" * 72)

    K_values = [1, 2, 3, 4, 5, 6, 7]

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def bounce_K_dir(initial_dir, K, r, h):
        pos = jnp.zeros(3)
        d = normalize(initial_dir)
        def step(carry, _):
            pos, d = carry
            t, part = intersect_cylinder(pos, d, r, h)
            hit = pos + t * d
            n = cyl_normal(hit, part)
            return (hit + 1e-4*n, reflect(d, n)), None
        (_, final_dir), _ = lax.scan(step, (pos, d), jnp.arange(K))
        return final_dir

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def bounce_K_pos(initial_dir, K, r, h):
        pos = jnp.zeros(3)
        d = normalize(initial_dir)
        def step(carry, _):
            pos, d = carry
            t, part = intersect_cylinder(pos, d, r, h)
            hit = pos + t * d
            n = cyl_normal(hit, part)
            return (hit + 1e-4*n, reflect(d, n)), t
        (final_pos, _), ts = lax.scan(step, (pos, d), jnp.arange(K))
        return final_pos

    # Get t values (forward only, no jacobian)
    @partial(jax.jit, static_argnums=(1, 2, 3))
    def get_ts(initial_dir, K, r, h):
        pos = jnp.zeros(3)
        d = normalize(initial_dir)
        def step(carry, _):
            pos, d = carry
            t, part = intersect_cylinder(pos, d, r, h)
            hit = pos + t * d
            n = cyl_normal(hit, part)
            return (hit + 1e-4*n, reflect(d, n)), (t, part)
        _, (ts, parts) = lax.scan(step, (pos, d), jnp.arange(K))
        return ts, parts

    test_dirs = {
        'generic':  jnp.array([0.8, 0.5, 0.3]),
        'radial':   jnp.array([1.0, 0.01, 0.01]),
        'vertical': jnp.array([0.01, 0.01, 1.0]),
        'shallow':  jnp.array([0.7, 0.7, 0.1]),
    }

    for r, h, det_label in [(4.0, 8.0, 'Small r=4m'), (35.0, 70.0, 'HK r=35m')]:
        print(f"\n  --- {det_label} ---")
        for dname, init_dir in test_dirs.items():
            print(f"\n  dir={dname}  {np.array(init_dir)}")
            print(f"    {'K':>3}  {'spec_radius':>12}  {'||J||_F':>10}  "
                  f"{'||J_pos||_F':>12}  {'t_values'}")
            prev_spec = None
            for K in K_values:
                try:
                    t0 = time_module.time()
                    J = jax.jacobian(bounce_K_dir, argnums=0)(init_dir, K, r, h)
                    J_pos = jax.jacobian(bounce_K_pos, argnums=0)(init_dir, K, r, h)
                    ts, parts = get_ts(init_dir, K, r, h)
                    dt = time_module.time() - t0

                    ev = jnp.linalg.eigvals(J)
                    spec_r = float(jnp.max(jnp.abs(ev)))
                    frob = float(jnp.linalg.norm(J, 'fro'))
                    frob_pos = float(jnp.linalg.norm(J_pos, 'fro'))

                    ts_str = ','.join(f'{float(t):.1f}' for t in ts)
                    parts_str = ','.join(['W','T','B'][int(p)] for p in parts)
                    ratio_str = f"(x{spec_r/prev_spec:.1f})" if prev_spec else ""
                    prev_spec = spec_r

                    print(f"    {K:>3}  {spec_r:12.4f}  {frob:10.4f}  "
                          f"{frob_pos:12.2f}  t=[{ts_str}] {parts_str} {ratio_str} "
                          f"({dt:.1f}s)")
                except Exception as e:
                    print(f"    {K:>3}  FAILED: {e}")
                    prev_spec = None


# =============================================================================
# Section 2: Sensor tangent gradient
# =============================================================================

def run_section_2():
    print(f"\n{'=' * 72}")
    print("SECTION 2: Sensor Tangent Gradient (Mechanism 1)")
    print("=" * 72)

    center = jnp.array([4.0, 0.0, 0.0])
    R = 0.25
    origin = jnp.zeros(3)

    alpha_tan = float(jnp.arcsin(R / 4.0))
    print(f"  Sensor at (4,0,0), R={R}. Tangent angle={alpha_tan:.6f} rad")

    n = 300
    alphas = jnp.linspace(0.0, 2.0 * alpha_tan, n)

    for use_fix, label in [(False, 'ORIGINAL sqrt(max(1e-10,disc))'),
                            (True,  'FIX_A    sqrt(max(0,disc)+1e-6)')]:
        print(f"\n  --- {label} ---")

        @jax.jit
        def grad_t(alpha):
            def f(a):
                d = jnp.array([jnp.cos(a), jnp.sin(a), 0.0])
                _, t, _ = sensor_intersect(origin, d, center, R, use_fix=use_fix)
                return t
            return jax.grad(f)(alpha)

        @jax.jit
        def get_disc(alpha):
            d = jnp.array([jnp.cos(alpha), jnp.sin(alpha), 0.0])
            _, _, disc = sensor_intersect(origin, d, center, R, use_fix=use_fix)
            return disc

        grads_list = []
        discs_list = []
        for a in alphas:
            grads_list.append(float(grad_t(a)))
            discs_list.append(float(get_disc(a)))

        grads = np.abs(np.array(grads_list))
        discs = np.array(discs_list)
        finite = np.isfinite(grads)

        print(f"    max|grad| = {np.max(grads[finite]):.1f}, "
              f"NaN/Inf = {np.sum(~finite)}")

        # Key points
        print(f"    {'region':>14}  {'disc':>12}  {'|grad|':>12}")
        for lbl, target in [('direct hit', 0.1), ('near-tangent', 1e-4),
                             ('tangent', 0.0), ('miss', -0.01)]:
            idx = np.argmin(np.abs(discs - target))
            print(f"    {lbl:>14}  {discs[idx]:12.2e}  {grads[idx]:12.2f}")


# =============================================================================
# Section 3: Multi-bounce + sensors
# =============================================================================

def run_section_3():
    print(f"\n{'=' * 72}")
    print("SECTION 3: Multi-Bounce + Sensor Detection (Combined)")
    print("=" * 72)

    K_values = [1, 2, 3, 4, 5, 6, 7]

    for r, h, det_label in [(4.0, 8.0, 'Small r=4m'), (35.0, 70.0, 'HK r=35m')]:
        R_sensor = 0.25
        # Small number of sensors to keep vmap fast
        n_ang, n_ht = 8, 4
        sensors = []
        for ang in np.linspace(0, 2*np.pi, n_ang, endpoint=False):
            for ht in np.linspace(-h/2+1, h/2-1, n_ht):
                sensors.append([r*np.cos(ang), r*np.sin(ang), ht])
        sensor_centers = jnp.array(sensors)
        n_sensors = len(sensors)

        print(f"\n  --- {det_label}, {n_sensors} sensors ---")

        for use_fix, fix_label in [(False, 'original'), (True, 'fix_a')]:

            @partial(jax.jit, static_argnums=(1,))
            def loss(initial_dir, K):
                pos = jnp.zeros(3)
                d = normalize(initial_dir)

                def step(carry, _):
                    pos, d, surv, total = carry

                    t, part = intersect_cylinder(pos, d, r, h)
                    hit = pos + t * d
                    n = cyl_normal(hit, part)

                    # Check sensors
                    def check(c):
                        dist, _, _ = sensor_intersect(pos, d, c, R_sensor, use_fix=use_fix)
                        return jax.nn.sigmoid(20.0 * (R_sensor - dist) / R_sensor)

                    weights = jax.vmap(check)(sensor_centers)
                    detect = jnp.sum(weights)
                    total = total + surv * detect

                    new_d = reflect(d, n)
                    new_pos = hit + 1e-4 * n
                    surv = surv * 0.5

                    return (new_pos, new_d, surv, total), None

                (_, _, _, total), _ = lax.scan(
                    step, (pos, d, 1.0, 0.0), jnp.arange(K))
                return total

            init_dir = jnp.array([0.8, 0.5, 0.3])
            print(f"\n    sqrt: {fix_label}")
            print(f"    {'K':>3}  {'|grad|':>14}  {'ratio':>8}  {'nan':>5}  {'time':>6}")
            prev = None
            for K in K_values:
                try:
                    t0 = time_module.time()
                    g = jax.grad(loss)(init_dir, K)
                    g.block_until_ready()
                    dt = time_module.time() - t0

                    gn = float(jnp.linalg.norm(g))
                    has_nan = bool(jnp.any(jnp.isnan(g)))
                    ratio = gn / prev if prev and prev > 1e-12 else float('nan')
                    prev = gn

                    print(f"    {K:>3}  {gn:14.4e}  {ratio:8.2f}  "
                          f"{'Y' if has_nan else 'N':>5}  {dt:5.1f}s")
                except Exception as e:
                    print(f"    {K:>3}  FAILED: {e}")
                    prev = None


# =============================================================================
if __name__ == '__main__':
    run_section_1()
    run_section_2()
    run_section_3()
