"""Section 1: Pure bouncing Jacobian — run standalone with progress prints."""
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
    a_w = d[0]**2 + d[1]**2
    b_w = 2.0*(pos[0]*d[0] + pos[1]*d[1])
    c_w = pos[0]**2 + pos[1]**2 - r**2
    disc_w = b_w**2 - 4*a_w*c_w
    sqrt_d = jnp.sqrt(jnp.maximum(0.0, disc_w))
    t1 = (-b_w - sqrt_d) / (2*a_w + 1e-12)
    t2 = (-b_w + sqrt_d) / (2*a_w + 1e-12)
    t1, t2 = jnp.minimum(t1, t2), jnp.maximum(t1, t2)
    t_cand = jnp.where(t1 > 1e-6, t1, t2)
    z_hit = pos[2] + t_cand*d[2]
    wall_ok = (disc_w >= 0) & (t_cand > 1e-6) & (jnp.abs(z_hit) <= h/2) & (a_w > 1e-12)
    t_wall = jnp.where(wall_ok, t_cand, LARGE)

    t_top = jnp.where(jnp.abs(d[2]) > 1e-12, (h/2 - pos[2])/d[2], LARGE)
    r2_top = (pos[0]+t_top*d[0])**2 + (pos[1]+t_top*d[1])**2
    t_top = jnp.where((t_top > 1e-6) & (r2_top <= r**2), t_top, LARGE)

    t_bot = jnp.where(jnp.abs(d[2]) > 1e-12, (-h/2 - pos[2])/d[2], LARGE)
    r2_bot = (pos[0]+t_bot*d[0])**2 + (pos[1]+t_bot*d[1])**2
    t_bot = jnp.where((t_bot > 1e-6) & (r2_bot <= r**2), t_bot, LARGE)

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
    return normalize(d - 2.0*jnp.sum(d*n)*n)

# ── Test ──

print("=" * 60)
print("Section 1: Pure specular bouncing Jacobian")
print("=" * 60)

# Step A: can we even run the forward pass?
print("\n[A] Testing forward pass...")
init_dir = jnp.array([0.8, 0.5, 0.3])
r, h = 4.0, 8.0

@partial(jax.jit, static_argnums=(1,))
def bounce_fwd(initial_dir, K):
    pos = jnp.zeros(3)
    d = normalize(initial_dir)
    def step(carry, _):
        pos, d = carry
        t, part = intersect_cylinder(pos, d, r, h)
        hit = pos + t * d
        n = cyl_normal(hit, part)
        return (hit + 1e-4*n, reflect(d, n)), (t, part)
    (fpos, fdir), (ts, parts) = lax.scan(step, (pos, d), jnp.arange(K))
    return fpos, fdir, ts, parts

for K in [1, 2, 3]:
    t0 = time.time()
    print(f"  compiling K={K}...", end=" ", flush=True)
    fpos, fdir, ts, parts = bounce_fwd(init_dir, K)
    jax.block_until_ready(fpos)
    dt = time.time() - t0
    ts_str = ','.join(f'{float(v):.2f}' for v in ts)
    ps_str = ','.join(['W','T','B'][int(p)] for p in parts)
    print(f"done ({dt:.1f}s). t=[{ts_str}] parts=[{ps_str}]")

# Step B: can we compute a scalar gradient?
print("\n[B] Testing scalar gradient (grad of ||final_dir||^2)...")

@partial(jax.jit, static_argnums=(1,))
def scalar_loss(initial_dir, K):
    pos = jnp.zeros(3)
    d = normalize(initial_dir)
    def step(carry, _):
        pos, d = carry
        t, part = intersect_cylinder(pos, d, r, h)
        hit = pos + t * d
        n = cyl_normal(hit, part)
        return (hit + 1e-4*n, reflect(d, n)), None
    (_, fdir), _ = lax.scan(step, (pos, d), jnp.arange(K))
    return jnp.sum(fdir**2)

for K in [1, 2, 3, 5, 7]:
    t0 = time.time()
    print(f"  compiling grad K={K}...", end=" ", flush=True)
    g = jax.grad(scalar_loss)(init_dir, K)
    jax.block_until_ready(g)
    dt = time.time() - t0
    print(f"done ({dt:.1f}s). |grad|={float(jnp.linalg.norm(g)):.4e}  "
          f"grad={np.array(g)}")

# Step C: full 3x3 Jacobian
print("\n[C] Testing full 3x3 Jacobian d(dir_K)/d(dir_0)...")

@partial(jax.jit, static_argnums=(1,))
def bounce_dir(initial_dir, K):
    pos = jnp.zeros(3)
    d = normalize(initial_dir)
    def step(carry, _):
        pos, d = carry
        t, part = intersect_cylinder(pos, d, r, h)
        hit = pos + t * d
        n = cyl_normal(hit, part)
        return (hit + 1e-4*n, reflect(d, n)), None
    (_, fdir), _ = lax.scan(step, (pos, d), jnp.arange(K))
    return fdir

for K in [1, 2, 3, 5, 7]:
    t0 = time.time()
    print(f"  compiling jacobian K={K}...", end=" ", flush=True)
    J = jax.jacobian(bounce_dir)(init_dir, K)
    jax.block_until_ready(J)
    dt = time.time() - t0
    ev = jnp.linalg.eigvals(J)
    spec = float(jnp.max(jnp.abs(ev)))
    frob = float(jnp.linalg.norm(J, 'fro'))
    print(f"done ({dt:.1f}s). spec_radius={spec:.4f}  ||J||_F={frob:.4f}")

# Step D: position Jacobian (lever arm)
print("\n[D] Testing d(pos_K)/d(dir_0) — lever arm...")

@partial(jax.jit, static_argnums=(1,))
def bounce_pos(initial_dir, K):
    pos = jnp.zeros(3)
    d = normalize(initial_dir)
    def step(carry, _):
        pos, d = carry
        t, part = intersect_cylinder(pos, d, r, h)
        hit = pos + t * d
        n = cyl_normal(hit, part)
        return (hit + 1e-4*n, reflect(d, n)), None
    (fpos, _), _ = lax.scan(step, (pos, d), jnp.arange(K))
    return fpos

for K in [1, 2, 3, 5, 7]:
    t0 = time.time()
    print(f"  compiling pos-jacobian K={K}...", end=" ", flush=True)
    J = jax.jacobian(bounce_pos)(init_dir, K)
    jax.block_until_ready(J)
    dt = time.time() - t0
    frob = float(jnp.linalg.norm(J, 'fro'))
    print(f"done ({dt:.1f}s). ||J_pos||_F={frob:.2f}")

print("\nSection 1 complete.")
