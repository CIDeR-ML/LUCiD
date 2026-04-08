"""Combined test with REAL sensor intersection (not Gaussian).

This tests whether the sqrt tangent singularity (problem 2) actually matters
in the multi-bounce pipeline, and whether fixes A+B together are sufficient.
"""
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np
import time

jax.config.update("jax_enable_x64", False)

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


# Real sensor intersection — ray-sphere with the actual sqrt
def sensor_check_original(ray_pos, ray_dir, center, radius):
    """Original sqrt: sqrt(max(1e-10, disc)). Returns sigmoid detection weight."""
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
    # Closest approach distance
    t_ca = -jnp.sum(oc * d)
    closest = ray_pos + t_ca * d
    dist = jnp.linalg.norm(closest - center)
    # Sigmoid detection (like the real code)
    return jax.nn.sigmoid(20.0 * (radius - dist) / radius)

def sensor_check_fix(ray_pos, ray_dir, center, radius):
    """Fixed sqrt: sqrt(max(0, disc) + 1e-6). Returns sigmoid detection weight."""
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
    t_ca = -jnp.sum(oc * d)
    closest = ray_pos + t_ca * d
    dist = jnp.linalg.norm(closest - center)
    return jax.nn.sigmoid(20.0 * (radius - dist) / radius)

def sensor_check_gaussian(ray_pos, ray_dir, center, radius, sigma):
    """Gaussian kernel — always smooth gradient, no sqrt issue."""
    d = normalize(ray_dir)
    oc = ray_pos - center
    t_ca = -jnp.sum(oc * d)
    closest = ray_pos + t_ca * d
    dist = jnp.linalg.norm(closest - center)
    return jnp.exp(-0.5 * (dist / sigma)**2)


print("=" * 60)
print("Combined: bouncing + scatter mixing + sensor detection")
print("  Comparing sensor detection methods x normalize variants")
print("=" * 60)

r, h = 35.0, 70.0
R_s = 0.25
sigma = 5.0 * R_s

# Place sensors densely enough that photons will pass near some
n_ang, n_ht = 16, 8
sensors = []
for ang in np.linspace(0, 2*np.pi, n_ang, endpoint=False):
    for ht in np.linspace(-h/2+1, h/2-1, n_ht):
        sensors.append([r*np.cos(ang), r*np.sin(ang), ht])
sensor_centers = jnp.array(sensors)
print(f"  {len(sensors)} sensors, r={r}, h={h}")

K_values = [1, 2, 3, 5, 7]
w_mix = 0.8

for sensor_label, sensor_fn_maker in [
    ("REAL_ORIG", lambda: lambda pos, d, c, R: sensor_check_original(pos, d, c, R)),
    ("REAL_FIX",  lambda: lambda pos, d, c, R: sensor_check_fix(pos, d, c, R)),
    ("GAUSSIAN",  lambda: lambda pos, d, c, R: sensor_check_gaussian(pos, d, c, R, sigma)),
]:
    for norm_label, norm_fn in [("normalize", normalize), ("norm_safe", normalize_safe)]:
        _sc = sensor_centers
        _sensor_fn = sensor_fn_maker()
        _norm = norm_fn

        @partial(jax.jit, static_argnums=(1,))
        def loss(initial_dir, K, _r=r, _h=h, _w=w_mix,
                 _nf=_norm, _sf=_sensor_fn, _sensors=_sc):
            pos = jnp.zeros(3)
            d = normalize(initial_dir)
            scatter_base = jnp.array([0.3, -0.7, 0.5])

            def step(carry, _):
                pos, d, surv, total = carry
                t, part = intersect_cylinder(pos, d, _r, _h)
                hit = pos + t*d
                n = cyl_normal(hit, part)

                # Sensor detection
                def check(center):
                    return _sf(pos, d, center, 0.25)
                weights = jax.vmap(check)(_sensors)
                detect = jnp.sum(weights)
                total = total + surv * detect

                # Scatter mixing
                refl_dir = reflect(d, n)
                scatter_dir = normalize(d + 0.3 * scatter_base)
                new_d = _nf(_w * refl_dir + (1-_w) * scatter_dir)
                new_pos = hit + 1e-4*n
                surv = surv * 0.5
                return (new_pos, new_d, surv, total), None

            (_, _, _, total), _ = lax.scan(step, (pos, d, 1.0, 0.0), jnp.arange(K))
            return total

        init_dir = jnp.array([0.8, 0.5, 0.3])
        label = f"{sensor_label} + {norm_label}"
        print(f"\n  --- {label} (w={w_mix}) ---")
        print(f"  {'K':>3}  {'loss':>10}  {'|grad|':>14}  {'ratio':>8}  {'nan':>5}")
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
                      f"{'Y' if has_nan else 'N':>5}  ({dt:.1f}s)")
            except Exception as e:
                print(f"\r  {K:>3}  FAILED: {e}")
                prev = None

# Also test with multiple initial directions to see if any hit tangent cases
print(f"\n{'=' * 60}")
print("Stress test: multiple initial directions (tangent-prone)")
print("=" * 60)

# Directions chosen to increase chance of tangent sensor hits
stress_dirs = {
    'generic':  jnp.array([0.8, 0.5, 0.3]),
    'radial':   jnp.array([1.0, 0.01, 0.01]),
    'shallow':  jnp.array([0.7, 0.7, 0.1]),
    'grazing':  jnp.array([0.99, 0.1, 0.05]),
}

K = 7

for dname, init_dir in stress_dirs.items():
    for sensor_label, sensor_fn_maker in [
        ("REAL_ORIG", lambda: lambda pos, d, c, R: sensor_check_original(pos, d, c, R)),
        ("REAL_FIX",  lambda: lambda pos, d, c, R: sensor_check_fix(pos, d, c, R)),
    ]:
        for norm_label, norm_fn in [("normalize", normalize), ("norm_safe", normalize_safe)]:
            _sensor_fn = sensor_fn_maker()
            _norm = norm_fn

            @partial(jax.jit, static_argnums=(1,))
            def loss_stress(initial_dir, K, _r=r, _h=h, _w=w_mix,
                     _nf=_norm, _sf=_sensor_fn, _sensors=sensor_centers):
                pos = jnp.zeros(3)
                d = normalize(initial_dir)
                scatter_base = jnp.array([0.3, -0.7, 0.5])
                def step(carry, _):
                    pos, d, surv, total = carry
                    t, part = intersect_cylinder(pos, d, _r, _h)
                    hit = pos + t*d
                    n = cyl_normal(hit, part)
                    def check(center):
                        return _sf(pos, d, center, 0.25)
                    weights = jax.vmap(check)(_sensors)
                    detect = jnp.sum(weights)
                    total = total + surv * detect
                    refl_dir = reflect(d, n)
                    scatter_dir = normalize(d + 0.3 * scatter_base)
                    new_d = _nf(_w * refl_dir + (1-_w) * scatter_dir)
                    new_pos = hit + 1e-4*n
                    surv = surv * 0.5
                    return (new_pos, new_d, surv, total), None
                (_, _, _, total), _ = lax.scan(step, (pos, d, 1.0, 0.0), jnp.arange(K))
                return total

            try:
                g = jax.grad(loss_stress)(init_dir, K)
                jax.block_until_ready(g)
                gn = float(jnp.linalg.norm(g))
                has_nan = bool(jnp.any(jnp.isnan(g)))
                print(f"  dir={dname:>8}  {sensor_label}+{norm_label:>10}  "
                      f"|grad|={gn:12.4e}  nan={'Y' if has_nan else 'N'}")
            except Exception as e:
                print(f"  dir={dname:>8}  {sensor_label}+{norm_label:>10}  FAILED: {e}")

print("\nDone.")
