"""
Test A: Sensor Intersection Gradient Profile
=============================================
Diagnoses Mechanism 1 — the sqrt(discriminant) singularity at near-tangent ray-sensor
intersections in compute_sensor_intersections_base.

What this test does:
  1. Places a single sensor sphere on a cylinder wall.
  2. Sweeps ray directions from "direct hit" through "tangent" to "clean miss".
  3. Computes the gradient of intersection outputs (time, normal, position) w.r.t.
     a scalar angle parameter that rotates the ray direction.
  4. Plots gradient magnitude vs discriminant value — reveals the gradient cliff.

After diagnosing, applies two candidate fixes and re-measures:
  Fix A: sqrt(max(0, disc) + eps)     — smooth additive epsilon inside sqrt
  Fix B: custom_vjp with clamped backward — caps the 1/sqrt gradient explicitly

Run from the LUCiD/ directory:
    python tests/test_tangent_gradients.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from functools import partial
import numpy as np

# ── Use the real overlap function from the codebase ──────────────────────────
from lucid.overlap import create_overlap_prob


# =============================================================================
# Part 1: Isolated quadratic-solver gradient profile
# =============================================================================

def ray_sensor_discriminant_and_sqrt(alpha, origin, sensor_center, sensor_radius):
    """
    Compute the discriminant and sqrt_term for a ray-sphere intersection,
    exactly as done in compute_sensor_intersections_base (base.py:184-193).

    alpha: scalar angle parameterizing the ray direction in the xy-plane.
    Returns: (discriminant, sqrt_term, t_intersect)
    """
    direction = jnp.array([jnp.cos(alpha), jnp.sin(alpha), 0.0])
    d = direction / (jnp.linalg.norm(direction) + 1e-10)
    oc = origin - sensor_center

    a = jnp.sum(d * d)
    b = 2.0 * jnp.sum(oc * d)
    c = jnp.sum(oc * oc) - sensor_radius ** 2

    discriminant = b ** 2 - 4 * a * c

    # === ORIGINAL CODE (base.py:193) ===
    sqrt_term = jnp.sqrt(jnp.maximum(1e-10, discriminant))

    # Stable quadratic formula (base.py:196-202)
    q = jnp.where(b > 0, -0.5 * (b + sqrt_term), -0.5 * (b - sqrt_term))
    t1 = q / (a + 1e-10)
    t2 = c / (q + jnp.sign(q) * 1e-10)
    t_intersect = jnp.where(
        (t1 > 0) & (t2 > 0), jnp.minimum(t1, t2),
        jnp.where(t1 > 0, t1, jnp.where(t2 > 0, t2, -1.0))
    )

    return discriminant, sqrt_term, t_intersect


def ray_sensor_discriminant_and_sqrt_fix_a(alpha, origin, sensor_center, sensor_radius):
    """Fix A: additive epsilon inside sqrt — sqrt(max(0, disc) + eps)."""
    direction = jnp.array([jnp.cos(alpha), jnp.sin(alpha), 0.0])
    d = direction / (jnp.linalg.norm(direction) + 1e-10)
    oc = origin - sensor_center

    a = jnp.sum(d * d)
    b = 2.0 * jnp.sum(oc * d)
    c = jnp.sum(oc * oc) - sensor_radius ** 2

    discriminant = b ** 2 - 4 * a * c

    # === FIX A: smooth epsilon ===
    eps_sqrt = 1e-6
    sqrt_term = jnp.sqrt(jnp.maximum(0.0, discriminant) + eps_sqrt)

    q = jnp.where(b > 0, -0.5 * (b + sqrt_term), -0.5 * (b - sqrt_term))
    t1 = q / (a + 1e-10)
    t2 = c / (q + jnp.sign(q) * 1e-10)
    t_intersect = jnp.where(
        (t1 > 0) & (t2 > 0), jnp.minimum(t1, t2),
        jnp.where(t1 > 0, t1, jnp.where(t2 > 0, t2, -1.0))
    )

    return discriminant, sqrt_term, t_intersect


# Fix B: custom_vjp that caps the backward gradient of sqrt_term
@jax.custom_vjp
def _safe_sqrt(x):
    return jnp.sqrt(jnp.maximum(1e-10, x))

def _safe_sqrt_fwd(x):
    y = jnp.sqrt(jnp.maximum(1e-10, x))
    return y, (x, y)

def _safe_sqrt_bwd(res, g):
    x, y = res
    # Original gradient: g / (2*y), but clip the magnitude
    max_grad = 100.0  # cap: sqrt gradient ≤ 100
    raw = g / (2.0 * y)
    clipped = jnp.clip(raw, -max_grad, max_grad)
    # Zero gradient where x was clamped (disc < 0)
    clipped = jnp.where(x > 1e-10, clipped, 0.0)
    return (clipped,)

_safe_sqrt.defvjp(_safe_sqrt_fwd, _safe_sqrt_bwd)


def ray_sensor_discriminant_and_sqrt_fix_b(alpha, origin, sensor_center, sensor_radius):
    """Fix B: custom_vjp with clamped backward sqrt gradient."""
    direction = jnp.array([jnp.cos(alpha), jnp.sin(alpha), 0.0])
    d = direction / (jnp.linalg.norm(direction) + 1e-10)
    oc = origin - sensor_center

    a = jnp.sum(d * d)
    b = 2.0 * jnp.sum(oc * d)
    c = jnp.sum(oc * oc) - sensor_radius ** 2

    discriminant = b ** 2 - 4 * a * c

    # === FIX B: clamped backward ===
    sqrt_term = _safe_sqrt(discriminant)

    q = jnp.where(b > 0, -0.5 * (b + sqrt_term), -0.5 * (b - sqrt_term))
    t1 = q / (a + 1e-10)
    t2 = c / (q + jnp.sign(q) * 1e-10)
    t_intersect = jnp.where(
        (t1 > 0) & (t2 > 0), jnp.minimum(t1, t2),
        jnp.where(t1 > 0, t1, jnp.where(t2 > 0, t2, -1.0))
    )

    return discriminant, sqrt_term, t_intersect


# =============================================================================
# Part 2: Full compute_sensor_intersections_base gradient test
# =============================================================================

def full_sensor_intersection_gradient(alpha, origin, sensor_center, sensor_radius,
                                       overlap_prob_fn, variant='original'):
    """
    Run the full sensor intersection computation (mirroring base.py:140-244)
    for a single ray and single sensor parameterized by angle alpha.

    Returns a scalar loss = time_output (so gradient captures the full chain).
    """
    direction = jnp.array([jnp.cos(alpha), jnp.sin(alpha), 0.0])
    ray_d = direction / (jnp.linalg.norm(direction) + 1e-10)

    # Expand to batch dims expected by the real code (1 ray)
    ray_origins = origin[None, :]
    ray_directions = direction[None, :]
    ray_d_batch = ray_d[None, :]

    oc = ray_origins - sensor_center[None, :]

    # Closest approach (base.py:176-179)
    t_closest = -jnp.sum(oc * ray_d_batch, axis=1, keepdims=True)
    closest = ray_origins + t_closest * ray_d_batch
    to_sensor = closest - sensor_center[None, :]
    distance = jnp.linalg.norm(to_sensor, axis=1)

    # Quadratic solver (base.py:184-207)
    a = jnp.sum(ray_d_batch * ray_d_batch, axis=1)
    b = 2.0 * jnp.sum(oc * ray_d_batch, axis=1)
    c_coeff = jnp.sum(oc * oc, axis=1) - sensor_radius ** 2
    discriminant = b ** 2 - 4 * a * c_coeff

    if variant == 'original':
        sqrt_term = jnp.sqrt(jnp.maximum(1e-10, discriminant))
    elif variant == 'fix_a':
        sqrt_term = jnp.sqrt(jnp.maximum(0.0, discriminant) + 1e-6)
    elif variant == 'fix_b':
        sqrt_term = _safe_sqrt(discriminant)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    q = jnp.where(b > 0, -0.5 * (b + sqrt_term), -0.5 * (b - sqrt_term))
    t1 = q / (a + 1e-10)
    t2 = c_coeff / (q + jnp.sign(q) * 1e-10)
    t_intersect = jnp.where(
        (t1 > 0) & (t2 > 0), jnp.minimum(t1, t2),
        jnp.where(t1 > 0, t1, jnp.where(t2 > 0, t2, -1.0))
    )

    # Intersection point and normal (base.py:210-214)
    intersection_points = ray_origins + t_intersect[:, None] * ray_d_batch
    to_intersection = intersection_points - sensor_center[None, :]
    normals_intersect = to_intersection / (jnp.linalg.norm(to_intersection, axis=1, keepdims=True) + 1e-10)

    # Overlap weight (base.py:220)
    weight = overlap_prob_fn(distance[0])

    # Combine: return a scalar that depends on time, normal, and weight
    # This captures all three gradient paths
    intersects = (discriminant[0] > 1e-6) & (t_intersect[0] > 0)
    time_out = jnp.where(intersects, t_intersect[0], t_closest[0, 0])
    normal_magnitude = jnp.linalg.norm(normals_intersect[0])

    # Loss = weighted time + normal contribution (arbitrary but captures both paths)
    return weight * time_out + 0.1 * normal_magnitude


# =============================================================================
# Part 3: Run diagnostics
# =============================================================================

def run_test():
    print("=" * 80)
    print("TEST A: Sensor Intersection Gradient Profile")
    print("=" * 80)

    # ── Configuration ──
    sensor_radius = 0.25  # meters (typical HK PMT)
    sensor_center = jnp.array([4.0, 0.0, 0.0])  # on cylinder wall at r=4m
    origin = jnp.array([0.0, 0.0, 0.0])  # ray starts at center

    # The tangent angle: alpha where discriminant = 0
    # disc = 4*(R^2 - d_perp^2), d_perp = 4*sin(alpha)
    # disc = 0 => sin(alpha) = R/4 = 0.0625 => alpha ~ 0.0625 rad
    alpha_tangent = jnp.arcsin(sensor_radius / jnp.linalg.norm(sensor_center))
    print(f"\nGeometry: sensor at {sensor_center}, radius={sensor_radius}m")
    print(f"Ray origin at center. Tangent angle = {float(alpha_tangent):.6f} rad "
          f"({float(jnp.degrees(alpha_tangent)):.3f} deg)")

    # Sweep alpha from 0 (direct hit) through tangent to 2x tangent (miss)
    n_points = 2000
    alpha_values = jnp.linspace(0.0, 2.0 * alpha_tangent, n_points)

    overlap_prob_fn = create_overlap_prob(0.2 * sensor_radius, sensor_radius)

    # ── Part 1: Isolated quadratic solver gradient ──
    print("\n--- Part 1: Quadratic solver gradient profile ---")

    variants = {
        'original': ray_sensor_discriminant_and_sqrt,
        'fix_a':    ray_sensor_discriminant_and_sqrt_fix_a,
        'fix_b':    ray_sensor_discriminant_and_sqrt_fix_b,
    }

    results = {}
    for name, fn in variants.items():
        # Gradient of t_intersect w.r.t. alpha
        def scalar_t(alpha):
            _, _, t = fn(alpha, origin, sensor_center, sensor_radius)
            return t

        grad_fn = jit(grad(scalar_t))
        disc_fn = jit(lambda a: fn(a, origin, sensor_center, sensor_radius)[0])

        grads = []
        discs = []
        for alpha in alpha_values:
            g = grad_fn(alpha)
            d = disc_fn(alpha)
            grads.append(float(g))
            discs.append(float(d))

        results[name] = {
            'grads': np.array(grads),
            'discs': np.array(discs),
        }

        abs_grads = np.abs(results[name]['grads'])
        finite_mask = np.isfinite(abs_grads)
        max_grad = np.max(abs_grads[finite_mask]) if finite_mask.any() else float('inf')
        nan_count = np.sum(~np.isfinite(abs_grads))

        print(f"  [{name:>10}]  max |grad| = {max_grad:12.2f},  NaN/Inf count = {nan_count}")

    # ── Part 2: Full intersection gradient ──
    print("\n--- Part 2: Full sensor intersection gradient (weight × time + normal) ---")

    for variant_name in ['original', 'fix_a', 'fix_b']:
        def loss(alpha):
            return full_sensor_intersection_gradient(
                alpha, origin, sensor_center, sensor_radius,
                overlap_prob_fn, variant=variant_name)

        grad_fn = jit(grad(loss))

        grads = []
        for alpha in alpha_values:
            g = grad_fn(alpha)
            grads.append(float(g))

        abs_grads = np.abs(np.array(grads))
        finite_mask = np.isfinite(abs_grads)
        max_grad = np.max(abs_grads[finite_mask]) if finite_mask.any() else float('inf')
        nan_count = np.sum(~np.isfinite(abs_grads))

        # Find the worst 5 points
        sorted_idx = np.argsort(-abs_grads * finite_mask)[:5]
        print(f"  [{variant_name:>10}]  max |grad| = {max_grad:12.2f},  NaN/Inf = {nan_count}")
        for idx in sorted_idx:
            disc_val = results['original']['discs'][idx]
            print(f"      alpha={float(alpha_values[idx]):.6f}, "
                  f"disc={disc_val:.2e}, |grad|={abs_grads[idx]:.2f}")

    # ── Part 3: Discriminant vs gradient table ──
    print("\n--- Part 3: Gradient magnitude at specific discriminant values ---")
    print(f"  {'discriminant':>14}  {'|grad| original':>16}  {'|grad| fix_a':>14}  {'|grad| fix_b':>14}")
    print("  " + "-" * 66)

    # Find alpha values that give specific discriminant values
    discs_original = results['original']['discs']
    target_discs = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0]

    for target in target_discs:
        # Find the closest alpha to this discriminant
        idx = np.argmin(np.abs(discs_original - target))
        actual_disc = discs_original[idx]
        g_orig = np.abs(results['original']['grads'][idx])
        g_a = np.abs(results['fix_a']['grads'][idx])
        g_b = np.abs(results['fix_b']['grads'][idx])
        print(f"  {actual_disc:14.2e}  {g_orig:16.2f}  {g_a:14.2f}  {g_b:14.2f}")

    # ── Part 4: Theoretical prediction ──
    print("\n--- Part 4: Theoretical sqrt gradient at exact discriminant values ---")
    print("  (This is d(sqrt(max(eps, disc)))/d(disc) — the raw amplification factor)")
    for disc in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0]:
        original_grad = 1.0 / (2.0 * np.sqrt(max(1e-10, disc)))
        fix_a_grad = 1.0 / (2.0 * np.sqrt(max(0, disc) + 1e-6))
        fix_b_grad = min(100.0, 1.0 / (2.0 * np.sqrt(max(1e-10, disc))))
        print(f"  disc={disc:.0e}:  original={original_grad:12.1f},  "
              f"fix_a={fix_a_grad:10.1f},  fix_b={fix_b_grad:10.1f}")

    # ── Save results for plotting ──
    try:
        save_path = os.path.join(os.path.dirname(__file__), 'tangent_gradient_results.npz')
        np.savez(save_path,
                 alpha_values=np.array(alpha_values),
                 discs_original=results['original']['discs'],
                 grads_original=results['original']['grads'],
                 grads_fix_a=results['fix_a']['grads'],
                 grads_fix_b=results['fix_b']['grads'],
                 alpha_tangent=float(alpha_tangent))
        print(f"\nResults saved to {save_path}")
        print("To plot: load with np.load() and plot |grads| vs alpha or vs disc")
    except Exception as e:
        print(f"\nCould not save results: {e}")


if __name__ == '__main__':
    run_test()
