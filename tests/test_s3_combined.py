"""Section 3: Multi-bounce + sensor detection — redesigned.

Uses a Gaussian detection kernel so there's always a non-zero gradient,
even when the photon doesn't pass directly through a sensor. This captures
the real gradient chain: dir_0 -> pos_k -> closest_approach_dist -> weight.
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

def intersect_cylinder(pos, d, r, h):
    LARGE = 1e10
    a_w = d[0]**2+d[1]**2
    b_w = 2.0*(pos[0]*d[0]+pos[1]*d[1])
    c_w = pos[0]**2+pos[1]**2 - r**2
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


print("=" * 60)
print("Section 3: Multi-Bounce + Sensor Detection (Gaussian kernel)")
print("=" * 60)

K_values = [1, 2, 3, 4, 5, 6, 7]
init_dir = jnp.array([0.8, 0.5, 0.3])

for r, h, det_label in [(4.0, 8.0, 'Small r=4m'), (35.0, 70.0, 'HK r=35m')]:
    R_s = 0.25
    # Gaussian width: sigma = 5*R so sensors at dist ~1m still contribute
    sigma = 5.0 * R_s

    # Place sensors
    n_ang, n_ht = 8, 4
    sensors = []
    for ang in np.linspace(0, 2*np.pi, n_ang, endpoint=False):
        for ht in np.linspace(-h/2+1, h/2-1, n_ht):
            sensors.append([r*np.cos(ang), r*np.sin(ang), ht])
    sensor_centers = jnp.array(sensors)

    print(f"\n--- {det_label}, {len(sensors)} sensors, sigma={sigma:.2f}m ---")

    # Closures need to be captured explicitly for JAX tracing
    _r, _h, _sc, _Rs, _sigma = r, h, sensor_centers, R_s, sigma

    @partial(jax.jit, static_argnums=(1,))
    def loss(initial_dir, K):
        pos = jnp.zeros(3)
        d = normalize(initial_dir)

        def step(carry, _):
            pos, d, surv, total = carry
            t, part = intersect_cylinder(pos, d, _r, _h)
            hit = pos + t*d
            n = cyl_normal(hit, part)

            # For each sensor: closest approach distance, then Gaussian weight
            def check(center):
                oc = pos - center
                d_norm = normalize(d)
                t_ca = -jnp.sum(oc * d_norm)
                closest = pos + t_ca * d_norm
                dist = jnp.linalg.norm(closest - center)
                # Gaussian kernel — always non-zero gradient
                return jnp.exp(-0.5 * (dist / _sigma)**2)

            weights = jax.vmap(check)(_sc)
            detect = jnp.sum(weights)
            total = total + surv * detect

            new_d = reflect(d, n)
            new_pos = hit + 1e-4*n
            surv = surv * 0.5
            return (new_pos, new_d, surv, total), None

        (_, _, _, total), _ = lax.scan(step, (pos, d, 1.0, 0.0), jnp.arange(K))
        return total

    # Warm up
    print(f"  Compiling K=1...", end=" ", flush=True)
    t0 = time.time()
    val = loss(init_dir, 1)
    jax.block_until_ready(val)
    print(f"fwd done ({time.time()-t0:.1f}s), loss={float(val):.4f}", flush=True)

    print(f"  Compiling K=1 grad...", end=" ", flush=True)
    t0 = time.time()
    g = jax.grad(loss)(init_dir, 1)
    jax.block_until_ready(g)
    print(f"done ({time.time()-t0:.1f}s), |grad|={float(jnp.linalg.norm(g)):.4e}")

    # Run all K
    print(f"\n  {'K':>3}  {'loss':>10}  {'|grad|':>14}  {'ratio':>8}  {'nan':>5}  {'time':>6}")
    prev = None
    for K in K_values:
        t0 = time.time()
        try:
            val = loss(init_dir, K)
            g = jax.grad(loss)(init_dir, K)
            jax.block_until_ready(g)
            dt = time.time() - t0
            gn = float(jnp.linalg.norm(g))
            has_nan = bool(jnp.any(jnp.isnan(g)))
            ratio = gn/prev if prev and prev > 1e-12 else float('nan')
            prev = gn
            print(f"  {K:>3}  {float(val):10.4f}  {gn:14.4e}  {ratio:8.2f}  "
                  f"{'Y' if has_nan else 'N':>5}  {dt:5.1f}s")
        except Exception as e:
            print(f"  {K:>3}  FAILED: {e}")
            prev = None

print("\nSection 3 complete.")
