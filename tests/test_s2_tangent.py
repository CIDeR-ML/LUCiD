"""Section 2: Sensor tangent gradient — standalone with progress prints."""
import jax
import jax.numpy as jnp
import numpy as np
import time

jax.config.update("jax_enable_x64", False)

def normalize(v, eps=1e-6):
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.maximum(norm, eps)

def sensor_intersect_original(ray_pos, ray_dir, center, radius):
    """sqrt(max(1e-10, disc))"""
    d = normalize(ray_dir)
    oc = ray_pos - center
    t_ca = -jnp.sum(oc * d)
    closest = ray_pos + t_ca * d
    dist = jnp.linalg.norm(closest - center)
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
    return dist, t_int, disc

def sensor_intersect_fix(ray_pos, ray_dir, center, radius):
    """sqrt(max(0, disc) + 1e-6)"""
    d = normalize(ray_dir)
    oc = ray_pos - center
    t_ca = -jnp.sum(oc * d)
    closest = ray_pos + t_ca * d
    dist = jnp.linalg.norm(closest - center)
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
    return dist, t_int, disc

print("=" * 60)
print("Section 2: Sensor Tangent Gradient (Mechanism 1)")
print("=" * 60)

center = jnp.array([4.0, 0.0, 0.0])
R = 0.25
origin = jnp.zeros(3)
alpha_tan = float(jnp.arcsin(R / 4.0))
print(f"Sensor at (4,0,0), R={R}m. Tangent angle={alpha_tan:.6f} rad")

n = 300
alphas = jnp.linspace(0.0, 2.0*alpha_tan, n)

for variant_name, intersect_fn in [('ORIGINAL', sensor_intersect_original),
                                     ('FIX_A', sensor_intersect_fix)]:
    print(f"\n--- {variant_name} ---")

    print("  Compiling grad...", end=" ", flush=True)
    t0 = time.time()

    @jax.jit
    def grad_t(alpha):
        def f(a):
            d = jnp.array([jnp.cos(a), jnp.sin(a), 0.0])
            _, t, _ = intersect_fn(origin, d, center, R)
            return t
        return jax.grad(f)(alpha)

    @jax.jit
    def get_disc(alpha):
        d = jnp.array([jnp.cos(alpha), jnp.sin(alpha), 0.0])
        _, _, disc = intersect_fn(origin, d, center, R)
        return disc

    # Warm up (compile)
    _ = grad_t(alphas[0])
    _ = get_disc(alphas[0])
    print(f"done ({time.time()-t0:.1f}s)")

    print(f"  Evaluating {n} points...", end=" ", flush=True)
    t0 = time.time()
    grads = np.array([float(grad_t(a)) for a in alphas])
    discs = np.array([float(get_disc(a)) for a in alphas])
    print(f"done ({time.time()-t0:.1f}s)")

    abs_grads = np.abs(grads)
    finite = np.isfinite(abs_grads)
    print(f"  max|grad| = {np.max(abs_grads[finite]):.1f}, NaN/Inf = {np.sum(~finite)}")

    print(f"  {'region':>14}  {'disc':>12}  {'|grad|':>12}")
    for lbl, target in [('direct hit', 0.1), ('near-tangent', 1e-4),
                         ('tangent~0', 1e-8), ('miss', -0.01)]:
        idx = np.argmin(np.abs(discs - target))
        print(f"  {lbl:>14}  {discs[idx]:12.2e}  {abs_grads[idx]:12.2f}")

    # Top 5 worst
    sorted_idx = np.argsort(-abs_grads * finite)[:5]
    print(f"  Top 5 worst:")
    for rank, idx in enumerate(sorted_idx):
        print(f"    #{rank+1}: |grad|={abs_grads[idx]:.1f}, disc={discs[idx]:.2e}")

print("\nSection 2 complete.")
