"""Comprehensive test: reproduce all 3 problems, then test all fixes.

Problem 1: Lever arm (position Jacobian growth)
Problem 2: sqrt singularity at tangent sensors
Problem 3: Scatter mixing norm amplification

Fix A: sqrt(max(0,disc) + 1e-6) for sensor intersection
Fix B: normalize_safe custom_vjp for mixing
"""
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np
import time

jax.config.update("jax_enable_x64", False)

# ══════════════════════════════════════════════════════════════
# Primitives
# ══════════════════════════════════════════════════════════════

def normalize(v, eps=1e-6):
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.maximum(norm, eps)

@jax.custom_vjp
def normalize_safe(v):
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.maximum(norm, 1e-6)

def normalize_safe_fwd(v):
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    y = v / jnp.maximum(norm, 1e-6)
    return y, y

def normalize_safe_bwd(y, g):
    g_tangent = g - jnp.sum(y * g, axis=-1, keepdims=True) * y
    return (g_tangent,)

normalize_safe.defvjp(normalize_safe_fwd, normalize_safe_bwd)

def intersect_cylinder(pos, d, r, h):
    LARGE = 1e10
    a_w = d[0]**2+d[1]**2
    b_w = 2.0*(pos[0]*d[0]+pos[1]*d[1])
    c_w = pos[0]**2+pos[1]**2-r**2
    disc_w = b_w**2-4*a_w*c_w
    sqrt_d = jnp.sqrt(jnp.maximum(0.0, disc_w))
    t1 = (-b_w-sqrt_d)/(2*a_w+1e-12)
    t2 = (-b_w+sqrt_d)/(2*a_w+1e-12)
    t1, t2 = jnp.minimum(t1,t2), jnp.maximum(t1,t2)
    t_cand = jnp.where(t1>1e-6, t1, t2)
    z_hit = pos[2]+t_cand*d[2]
    wall_ok = (disc_w>=0)&(t_cand>1e-6)&(jnp.abs(z_hit)<=h/2)&(a_w>1e-12)
    t_wall = jnp.where(wall_ok, t_cand, LARGE)
    t_top = jnp.where(jnp.abs(d[2])>1e-12, (h/2-pos[2])/d[2], LARGE)
    r2_t = (pos[0]+t_top*d[0])**2+(pos[1]+t_top*d[1])**2
    t_top = jnp.where((t_top>1e-6)&(r2_t<=r**2), t_top, LARGE)
    t_bot = jnp.where(jnp.abs(d[2])>1e-12, (-h/2-pos[2])/d[2], LARGE)
    r2_b = (pos[0]+t_bot*d[0])**2+(pos[1]+t_bot*d[1])**2
    t_bot = jnp.where((t_bot>1e-6)&(r2_b<=r**2), t_bot, LARGE)
    ts = jnp.array([t_wall, t_top, t_bot])
    return jnp.min(ts), jnp.argmin(ts)

def cyl_normal(hit, part):
    xy = jnp.sqrt(hit[0]**2+hit[1]**2)+1e-10
    wall_n = jnp.array([-hit[0]/xy,-hit[1]/xy,0.0])
    top_n = jnp.array([0.0,0.0,-1.0])
    bot_n = jnp.array([0.0,0.0,1.0])
    return jnp.where(part==0,wall_n,jnp.where(part==1,top_n,bot_n))

def reflect(d, n):
    d = normalize(d); n = normalize(n)
    return normalize(d - 2.0*jnp.sum(d*n)*n)


# ══════════════════════════════════════════════════════════════
# PROBLEM 2: sqrt singularity — reproduce with fine sampling
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("PROBLEM 2: sqrt singularity at tangent sensor rays")
print("=" * 60)

def sensor_intersect_original(ray_pos, ray_dir, center, radius):
    """Original: sqrt(max(1e-10, disc))"""
    d = normalize(ray_dir)
    oc = ray_pos - center
    a = jnp.sum(d*d)
    b = 2.0*jnp.sum(oc*d)
    c = jnp.sum(oc*oc) - radius**2
    disc = b**2 - 4*a*c
    sqrt_term = jnp.sqrt(jnp.maximum(1e-10, disc))
    q = jnp.where(b > 0, -0.5*(b+sqrt_term), -0.5*(b-sqrt_term))
    t1 = q / (a+1e-10)
    t2 = c / (q + jnp.sign(q)*1e-10)
    t_int = jnp.where((t1>0)&(t2>0), jnp.minimum(t1,t2),
                       jnp.where(t1>0, t1, jnp.where(t2>0, t2, -1.0)))
    return t_int, disc

def sensor_intersect_fix(ray_pos, ray_dir, center, radius):
    """Fix A: sqrt(max(0, disc) + 1e-6)"""
    d = normalize(ray_dir)
    oc = ray_pos - center
    a = jnp.sum(d*d)
    b = 2.0*jnp.sum(oc*d)
    c = jnp.sum(oc*oc) - radius**2
    disc = b**2 - 4*a*c
    sqrt_term = jnp.sqrt(jnp.maximum(0.0, disc) + 1e-6)
    q = jnp.where(b > 0, -0.5*(b+sqrt_term), -0.5*(b-sqrt_term))
    t1 = q / (a+1e-10)
    t2 = c / (q + jnp.sign(q)*1e-10)
    t_int = jnp.where((t1>0)&(t2>0), jnp.minimum(t1,t2),
                       jnp.where(t1>0, t1, jnp.where(t2>0, t2, -1.0)))
    return t_int, disc

center = jnp.array([4.0, 0.0, 0.0])
R = 0.25
origin = jnp.zeros(3)
alpha_tan = float(jnp.arcsin(R / 4.0))

# Fine sampling around the tangent angle
n_pts = 2000
# Sample very tightly around the tangent angle
alphas = jnp.linspace(alpha_tan - 0.001, alpha_tan + 0.001, n_pts)

print(f"Sensor at (4,0,0), R={R}m. Tangent angle={alpha_tan:.6f} rad")
print(f"Sampling {n_pts} points in [{alpha_tan-0.001:.6f}, {alpha_tan+0.001:.6f}]")

for variant_name, intersect_fn in [('ORIGINAL', sensor_intersect_original),
                                     ('FIX_A', sensor_intersect_fix)]:
    print(f"\n--- {variant_name} ---")

    @jax.jit
    def grad_t(alpha, _fn=intersect_fn):
        def f(a):
            d = jnp.array([jnp.cos(a), jnp.sin(a), 0.0])
            t, _ = _fn(origin, d, center, R)
            return t
        return jax.grad(f)(alpha)

    @jax.jit
    def get_disc(alpha, _fn=intersect_fn):
        d = jnp.array([jnp.cos(alpha), jnp.sin(alpha), 0.0])
        _, disc = _fn(origin, d, center, R)
        return disc

    # Warm up
    print("  Compiling...", end=" ", flush=True)
    t0 = time.time()
    _ = grad_t(alphas[0])
    _ = get_disc(alphas[0])
    print(f"done ({time.time()-t0:.1f}s)")

    print(f"  Evaluating {n_pts} points...", end=" ", flush=True)
    t0 = time.time()
    grads = np.array([float(grad_t(a)) for a in alphas])
    discs = np.array([float(get_disc(a)) for a in alphas])
    print(f"done ({time.time()-t0:.1f}s)")

    abs_grads = np.abs(grads)
    finite = np.isfinite(abs_grads)
    nan_count = np.sum(~finite)
    max_grad = np.max(abs_grads[finite]) if np.any(finite) else float('nan')

    print(f"  max|grad| = {max_grad:.1f}, NaN/Inf = {nan_count}/{n_pts}")

    # Show gradient near disc=0
    print(f"  {'disc':>12}  {'|grad|':>12}  {'alpha-alpha_tan':>16}")
    # Find points closest to disc = various values
    for target_disc in [0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, -1e-4]:
        idx = np.argmin(np.abs(discs - target_disc))
        g_val = abs_grads[idx] if finite[idx] else float('nan')
        print(f"  {discs[idx]:12.2e}  {g_val:12.2f}  {float(alphas[idx])-alpha_tan:16.8f}")

    # Show the 5 worst gradients
    print(f"  Top 5 worst:")
    sorted_idx = np.argsort(-abs_grads * finite)[:5]
    for rank, idx in enumerate(sorted_idx):
        print(f"    #{rank+1}: |grad|={abs_grads[idx]:.1f}, disc={discs[idx]:.2e}, "
              f"alpha-tan={float(alphas[idx])-alpha_tan:.8f}")


# ══════════════════════════════════════════════════════════════
# PROBLEM 3 + FIX B: Scatter mixing (recap with more w values)
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("PROBLEM 3: Scatter mixing — ORIGINAL vs FIX (normalize_safe)")
print("=" * 60)

init_dir = jnp.array([0.8, 0.5, 0.3])
K_values = [1, 2, 3, 5, 7]
r, h = 35.0, 70.0

for w in [0.9, 0.8, 0.7]:
    for fix_label, norm_fn in [("ORIGINAL", normalize), ("FIX_B", normalize_safe)]:
        @partial(jax.jit, static_argnums=(1,))
        def bounce_mixed(initial_dir, K, _r=r, _h=h, _w=w, _norm=norm_fn):
            pos = jnp.zeros(3)
            d = normalize(initial_dir)
            scatter_base = jnp.array([0.3, -0.7, 0.5])
            def step(carry, _):
                pos, d = carry
                t, part = intersect_cylinder(pos, d, _r, _h)
                hit = pos + t*d
                n = cyl_normal(hit, part)
                refl_dir = reflect(d, n)
                scatter_dir = normalize(d + 0.3 * scatter_base)
                new_d = _norm(_w * refl_dir + (1-_w) * scatter_dir)
                new_pos = hit + 1e-4*n
                return (new_pos, new_d), None
            (_, fdir), _ = lax.scan(step, (pos, d), jnp.arange(K))
            return fdir

        print(f"\n  w={w}, {fix_label}:")
        print(f"  {'K':>3}  {'spec_r':>10}  {'||J||_F':>10}")
        for K in K_values:
            t0 = time.time()
            print(f"  {K:>3}  compiling...", end="", flush=True)
            J = jax.jacobian(bounce_mixed)(init_dir, K)
            jax.block_until_ready(J)
            dt = time.time()-t0
            ev = jnp.linalg.eigvals(J)
            spec = float(jnp.max(jnp.abs(ev)))
            frob = float(jnp.linalg.norm(J, 'fro'))
            print(f"\r  {K:>3}  {spec:10.4f}  {frob:10.4f}  ({dt:.1f}s)")


# ══════════════════════════════════════════════════════════════
# COMBINED: Multi-bounce + sensors + mixing with BOTH fixes
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("COMBINED: Multi-bounce + sensors + scatter mixing")
print("  Testing: no fix, fix_A only, fix_B only, both fixes")
print("=" * 60)

r, h = 35.0, 70.0
R_s = 0.25
sigma = 5.0 * R_s  # Gaussian kernel width

n_ang, n_ht = 8, 4
sensors = []
for ang in np.linspace(0, 2*np.pi, n_ang, endpoint=False):
    for ht in np.linspace(-h/2+1, h/2-1, n_ht):
        sensors.append([r*np.cos(ang), r*np.sin(ang), ht])
sensor_centers = jnp.array(sensors)

init_dir = jnp.array([0.8, 0.5, 0.3])
K_values = [1, 2, 3, 5, 7]
w_mix = 0.8  # The problematic STE weight

for combo_label, use_sqrt_fix, use_norm_fix in [
    ("NO_FIX",   False, False),
    ("FIX_A",    True,  False),   # sqrt fix only
    ("FIX_B",    False, True),    # normalize fix only
    ("BOTH",     True,  True),    # both fixes
]:
    _sc = sensor_centers
    _sigma = sigma
    _norm_fn = normalize_safe if use_norm_fix else normalize
    _sqrt_fix = use_sqrt_fix

    @partial(jax.jit, static_argnums=(1,))
    def loss(initial_dir, K, _r=r, _h=h, _w=w_mix,
             _norm=_norm_fn, _sfix=_sqrt_fix,
             _sensors=_sc, _sig=_sigma, _Rs=R_s):
        pos = jnp.zeros(3)
        d = normalize(initial_dir)
        scatter_base = jnp.array([0.3, -0.7, 0.5])

        def step(carry, _):
            pos, d, surv, total = carry
            t, part = intersect_cylinder(pos, d, _r, _h)
            hit = pos + t*d
            n = cyl_normal(hit, part)

            # Sensor detection with Gaussian kernel
            def check(center):
                oc = pos - center
                d_norm = normalize(d)
                t_ca = -jnp.sum(oc * d_norm)
                closest = pos + t_ca * d_norm
                dist = jnp.linalg.norm(closest - center)
                return jnp.exp(-0.5 * (dist / _sig)**2)

            weights = jax.vmap(check)(_sensors)
            detect = jnp.sum(weights)
            total = total + surv * detect

            # Scatter mixing (simulating STE behavior)
            refl_dir = reflect(d, n)
            scatter_dir = normalize(d + 0.3 * scatter_base)
            new_d = _norm(_w * refl_dir + (1-_w) * scatter_dir)
            new_pos = hit + 1e-4*n
            surv = surv * 0.5
            return (new_pos, new_d, surv, total), None

        (_, _, _, total), _ = lax.scan(step, (pos, d, 1.0, 0.0), jnp.arange(K))
        return total

    print(f"\n  --- {combo_label} (w={w_mix}) ---")
    print(f"  {'K':>3}  {'loss':>10}  {'|grad|':>14}  {'ratio':>8}  {'nan':>5}  {'time':>6}")
    prev = None
    for K in K_values:
        t0 = time.time()
        print(f"  {K:>3}  compiling...", end="", flush=True)
        try:
            val = loss(init_dir, K)
            g = jax.grad(loss)(init_dir, K)
            jax.block_until_ready(g)
            dt = time.time() - t0
            gn = float(jnp.linalg.norm(g))
            has_nan = bool(jnp.any(jnp.isnan(g)))
            ratio = gn/prev if prev and prev > 1e-12 else float('nan')
            prev = gn
            print(f"\r  {K:>3}  {float(val):10.4f}  {gn:14.4e}  {ratio:8.2f}  "
                  f"{'Y' if has_nan else 'N':>5}  {dt:5.1f}s")
        except Exception as e:
            print(f"\r  {K:>3}  FAILED: {e}")
            prev = None

print("\nAll tests complete.")
