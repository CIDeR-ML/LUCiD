"""Section 1b: Same as Section 1 but for HK-scale (r=35m) + adds scattering."""
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

init_dir = jnp.array([0.8, 0.5, 0.3])
K_values = [1, 2, 3, 5, 7]

# ── Part A: HK-scale position Jacobian ──
print("=" * 60)
print("Part A: Position Jacobian at HK scale (r=35m)")
print("=" * 60)

for r, h, label in [(4.0, 8.0, 'r=4'), (35.0, 70.0, 'r=35')]:
    @partial(jax.jit, static_argnums=(1,))
    def bounce_pos(initial_dir, K, _r=r, _h=h):
        pos = jnp.zeros(3)
        d = normalize(initial_dir)
        def step(carry, _):
            pos, d = carry
            t, part = intersect_cylinder(pos, d, _r, _h)
            hit = pos + t*d
            n = cyl_normal(hit, part)
            return (hit+1e-4*n, reflect(d,n)), t
        (fpos, _), ts = lax.scan(step, (pos, d), jnp.arange(K))
        return fpos

    @partial(jax.jit, static_argnums=(1,))
    def bounce_dir(initial_dir, K, _r=r, _h=h):
        pos = jnp.zeros(3)
        d = normalize(initial_dir)
        def step(carry, _):
            pos, d = carry
            t, part = intersect_cylinder(pos, d, _r, _h)
            hit = pos + t*d
            n = cyl_normal(hit, part)
            return (hit+1e-4*n, reflect(d,n)), None
        (_, fdir), _ = lax.scan(step, (pos, d), jnp.arange(K))
        return fdir

    print(f"\n  {label}:")
    print(f"  {'K':>3}  {'spec_r(dir)':>12}  {'||J_pos||':>10}  {'time':>6}")
    for K in K_values:
        t0 = time.time()
        print(f"  {K:>3}  compiling...", end="", flush=True)
        J_d = jax.jacobian(bounce_dir)(init_dir, K)
        J_p = jax.jacobian(bounce_pos)(init_dir, K)
        jax.block_until_ready(J_d); jax.block_until_ready(J_p)
        dt = time.time()-t0
        ev = jnp.linalg.eigvals(J_d)
        spec = float(jnp.max(jnp.abs(ev)))
        frob_p = float(jnp.linalg.norm(J_p, 'fro'))
        print(f"\r  {K:>3}  {spec:12.4f}  {frob_p:10.2f}  {dt:5.1f}s")


# ── Part B: Scattering test ──
# Add a DETERMINISTIC "scatter" that mixes reflection with a fixed offset direction.
# This mimics what the STE does: new_dir = w_reflect * reflect_dir + w_scatter * scatter_dir
# We use a FIXED scatter direction (no randomness) and a FIXED mixture weight
# to see if the mixing operation itself causes gradient growth.

print(f"\n{'=' * 60}")
print("Part B: Reflection + deterministic scatter mixing")
print("  new_dir = w * reflect_dir + (1-w) * scatter_dir")
print("=" * 60)

for r, h, label in [(4.0, 8.0, 'r=4'), (35.0, 70.0, 'r=35')]:
    for w in [1.0, 0.8, 0.5]:
        @partial(jax.jit, static_argnums=(1,))
        def bounce_mixed(initial_dir, K, _r=r, _h=h, _w=w):
            pos = jnp.zeros(3)
            d = normalize(initial_dir)
            # Fixed scatter offset direction (arbitrary but fixed)
            scatter_base = jnp.array([0.3, -0.7, 0.5])

            def step(carry, _):
                pos, d = carry
                t, part = intersect_cylinder(pos, d, _r, _h)
                hit = pos + t*d
                n = cyl_normal(hit, part)

                refl_dir = reflect(d, n)
                # Scatter direction: forward-scatter with fixed deviation
                scatter_dir = normalize(d + 0.3 * scatter_base)

                # Mix (like STE would, but deterministic)
                new_d = normalize(_w * refl_dir + (1-_w) * scatter_dir)
                new_pos = hit + 1e-4*n
                return (new_pos, new_d), None

            (_, fdir), _ = lax.scan(step, (pos, d), jnp.arange(K))
            return fdir

        print(f"\n  {label}, w_reflect={w}:")
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

print("\nDone.")
