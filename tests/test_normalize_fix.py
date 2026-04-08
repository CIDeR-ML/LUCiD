"""Test the normalize_safe_mix custom_vjp fix.

Part 1: Verify forward pass matches standard normalize.
Part 2: Verify backward pass is bounded (no 1/||v|| amplification).
Part 3: Integrate into scatter mixing and check spectral radius stays <= 1.
"""
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np
import time

jax.config.update("jax_enable_x64", False)

# ── Standard normalize ──
def normalize(v, eps=1e-6):
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.maximum(norm, eps)

# ── Fixed normalize with custom backward ──
@jax.custom_vjp
def normalize_safe(v):
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.maximum(norm, 1e-6)

def normalize_safe_fwd(v):
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    y = v / jnp.maximum(norm, 1e-6)
    return y, y  # residual is just the output unit vector

def normalize_safe_bwd(y, g):
    # Pure tangent-space projection: remove the component along y
    # This is equivalent to (1/||v||)(I - y y^T) g with the 1/||v|| dropped
    # So the backward Jacobian is just (I - y y^T), which has spectral radius = 1
    g_tangent = g - jnp.sum(y * g, axis=-1, keepdims=True) * y
    return (g_tangent,)

normalize_safe.defvjp(normalize_safe_fwd, normalize_safe_bwd)


# ═══════════════════════════════════════════════════════════════
# Part 1: Forward correctness
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("Part 1: Forward correctness")
print("=" * 60)

test_vectors = [
    jnp.array([1.0, 0.0, 0.0]),
    jnp.array([0.8, 0.5, 0.3]),
    jnp.array([0.01, 0.01, 0.01]),
    jnp.array([3.0, -2.0, 1.5]),
    # Mixed vector like w*reflect + (1-w)*scatter with ||v|| < 1
    jnp.array([0.6, 0.3, 0.1]),
]

all_ok = True
for v in test_vectors:
    y_std = normalize(v)
    y_fix = normalize_safe(v)
    diff = float(jnp.linalg.norm(y_std - y_fix))
    ok = diff < 1e-6
    all_ok = all_ok and ok
    print(f"  v={np.array(v)}  ||std-fix||={diff:.2e}  {'OK' if ok else 'FAIL'}")

print(f"  Forward: {'ALL OK' if all_ok else 'SOME FAILED'}")


# ═══════════════════════════════════════════════════════════════
# Part 2: Backward boundedness
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("Part 2: Backward Jacobian comparison")
print("=" * 60)

print(f"\n  {'||v||':>8}  {'||J_std||':>10}  {'||J_fix||':>10}  {'spec_std':>10}  {'spec_fix':>10}")

for v in [jnp.array([0.3, 0.2, 0.1]),   # ||v|| ~ 0.37
          jnp.array([0.6, 0.4, 0.2]),   # ||v|| ~ 0.75
          jnp.array([0.9, 0.3, 0.1]),   # ||v|| ~ 0.95
          jnp.array([1.0, 0.0, 0.0]),   # ||v|| = 1.0
          jnp.array([2.0, 1.0, 0.5]),   # ||v|| ~ 2.29
          ]:
    J_std = jax.jacobian(normalize)(v)
    J_fix = jax.jacobian(normalize_safe)(v)

    frob_std = float(jnp.linalg.norm(J_std, 'fro'))
    frob_fix = float(jnp.linalg.norm(J_fix, 'fro'))

    ev_std = jnp.linalg.eigvals(J_std)
    ev_fix = jnp.linalg.eigvals(J_fix)
    spec_std = float(jnp.max(jnp.abs(ev_std)))
    spec_fix = float(jnp.max(jnp.abs(ev_fix)))

    vnorm = float(jnp.linalg.norm(v))
    print(f"  {vnorm:8.3f}  {frob_std:10.4f}  {frob_fix:10.4f}  {spec_std:10.4f}  {spec_fix:10.4f}")

print("  (spec_fix should always be <= 1.0, spec_std can exceed 1 when ||v|| < 1)")


# ═══════════════════════════════════════════════════════════════
# Part 3: Scatter mixing spectral radius
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("Part 3: Scatter mixing — spectral radius with fix vs original")
print("=" * 60)

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
r, h = 35.0, 70.0

for w in [1.0, 0.8, 0.5]:
    for fix_label, norm_fn in [("ORIGINAL", normalize), ("FIX", normalize_safe)]:
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

                # Use _norm for the final mixing normalization
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

print("\nDone.")
