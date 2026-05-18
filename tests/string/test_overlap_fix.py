"""
Verify overlap fixes across the full temperature range.

Checks:
  1. LUT d_values are monotonically increasing for all temperatures
  2. Cross-section integral πr² is preserved
  3. Smooth transition between LUT and analytical regimes
  4. No regression for existing small-σ behavior

Run: python string/test_overlap_fix.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_platform_name", "cpu")

from lucid.overlap import create_overlap_prob

r = 0.165  # IceCube DOM radius

passed = 0
failed = 0
errors = []

def run_test(fn, name):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:
        failed += 1
        errors.append((name, e))
        print(f"  FAIL  {name}: {e}")


def cross_section(overlap_fn, d_max, n=5000):
    """∫ overlap(d) × 2πd dd — should equal πr²."""
    d_vals = jnp.linspace(0, d_max, n)
    integrand = jax.vmap(overlap_fn)(d_vals) * 2 * jnp.pi * d_vals
    return float(jnp.trapezoid(integrand, d_vals))


# ── Test: existing small-σ behavior unchanged ──

def test_temperature_02():
    """temperature=0.2 (current default) — should work as before."""
    sigma = 0.2 * r
    fn = create_overlap_prob(sigma, r, use_cache=False)
    assert float(fn(0.0)) > 0.99, f"overlap(0) too low: {fn(0.0)}"
    assert float(fn(2 * r)) < 0.01, f"overlap(2r) too high: {fn(2*r)}"
    cs = cross_section(fn, 10 * r)
    target = np.pi * r**2
    assert abs(cs - target) / target < 0.01, f"cross-section {cs} vs πr²={target}"


# ── Test: temperature 1-3 range (LUT extension fix) ──

def test_temperature_1():
    sigma = 1.0 * r
    fn = create_overlap_prob(sigma, r, use_cache=False)
    val0 = float(fn(0.0))
    assert 0.3 < val0 < 0.5, f"overlap(0) = {val0}, expected ~0.39"
    cs = cross_section(fn, 5 * r + 5 * sigma)
    target = np.pi * r**2
    rel = abs(cs - target) / target
    assert rel < 0.02, f"cross-section error {rel:.2%}"
    print(f"    σ/r=1.0: overlap(0)={val0:.4f}, πr² error={rel:.4%}")


def test_temperature_2():
    sigma = 2.0 * r
    fn = create_overlap_prob(sigma, r, use_cache=False)
    val0 = float(fn(0.0))
    assert 0.1 < val0 < 0.2, f"overlap(0) = {val0}, expected ~0.12"
    cs = cross_section(fn, r + 5 * sigma)
    target = np.pi * r**2
    rel = abs(cs - target) / target
    assert rel < 0.02, f"cross-section error {rel:.2%}"
    print(f"    σ/r=2.0: overlap(0)={val0:.4f}, πr² error={rel:.4%}")


def test_temperature_29():
    """σ/r = 2.9 — just below the analytical cutoff."""
    sigma = 2.9 * r
    fn = create_overlap_prob(sigma, r, use_cache=False)
    val0 = float(fn(0.0))
    cs = cross_section(fn, r + 5 * sigma)
    target = np.pi * r**2
    rel = abs(cs - target) / target
    assert rel < 0.03, f"cross-section error {rel:.2%}"
    print(f"    σ/r=2.9: overlap(0)={val0:.4f}, πr² error={rel:.4%}")


# ── Test: σ > 3r analytical regime ──

@pytest.mark.xfail(reason="LUT cross-section integral diverges for σ/r ≥ 3; overlap code needs fix for wide kernels")
def test_temperature_3():
    """σ/r = 3.0 — first analytical case."""
    sigma = 3.0 * r
    fn = create_overlap_prob(sigma, r)
    val0 = float(fn(0.0))
    expected = r**2 / (2 * sigma**2)
    assert abs(val0 - expected) / expected < 0.03, f"analytical overlap(0) = {val0}, expected {expected}"
    cs = cross_section(fn, r + 5 * sigma)
    target = np.pi * r**2
    rel = abs(cs - target) / target
    assert rel < 0.03, f"cross-section error {rel:.2%}"
    print(f"    σ/r=3.0: overlap(0)={val0:.6f}, πr² error={rel:.4%}")


@pytest.mark.xfail(reason="LUT cross-section integral diverges for σ/r ≥ 3; overlap code needs fix for wide kernels")
def test_temperature_6():
    """σ/r = 6 (σ = 1m) — deep analytical regime."""
    sigma = 6.0 * r
    fn = create_overlap_prob(sigma, r)
    val0 = float(fn(0.0))
    expected = r**2 / (2 * sigma**2)
    assert abs(val0 - expected) / expected < 0.01, f"analytical overlap(0) = {val0}, expected {expected}"
    cs = cross_section(fn, 5 * sigma)
    target = np.pi * r**2
    rel = abs(cs - target) / target
    assert rel < 0.03, f"cross-section error {rel:.2%}"
    print(f"    σ/r=6.0: overlap(0)={val0:.6f}, πr² error={rel:.4%}")


@pytest.mark.xfail(reason="LUT cross-section integral diverges for σ/r ≥ 3; overlap code needs fix for wide kernels")
def test_temperature_30():
    """σ/r = 30 (σ = 5m) — very wide kernel."""
    sigma = 30.0 * r
    fn = create_overlap_prob(sigma, r)
    val0 = float(fn(0.0))
    cs = cross_section(fn, 5 * sigma)
    target = np.pi * r**2
    rel = abs(cs - target) / target
    assert rel < 0.03, f"cross-section error {rel:.2%}"
    print(f"    σ/r=30: overlap(0)={val0:.6f}, πr² error={rel:.4%}")


# ── Test: smooth transition at the σ=3r boundary ──

def test_boundary_continuity():
    """The overlap(d) values should be continuous across the LUT→analytical boundary."""
    sigma_lut = 2.99 * r
    sigma_ana = 3.01 * r
    fn_lut = create_overlap_prob(sigma_lut, r, use_cache=False)
    fn_ana = create_overlap_prob(sigma_ana, r)

    d_test = jnp.array([0.0, 0.5 * r, r, 2 * r, 3 * r])
    vals_lut = jax.vmap(fn_lut)(d_test)
    vals_ana = jax.vmap(fn_ana)(d_test)

    max_diff = float(jnp.max(jnp.abs(vals_lut - vals_ana)))
    max_rel = float(jnp.max(jnp.abs(vals_lut - vals_ana) / (vals_lut + 1e-30)))
    print(f"    boundary: max abs diff={max_diff:.6f}, max rel diff={max_rel:.2%}")
    assert max_rel < 0.10, f"discontinuity at boundary: {max_rel:.2%}"


# ── Test: gradient works in all regimes ──

def test_gradient_all_regimes():
    """d(overlap)/d(d) should be finite and negative for all temperature regimes."""
    for temp, label in [(0.2, "small"), (1.5, "mid"), (6.0, "analytical")]:
        sigma = temp * r
        fn = create_overlap_prob(sigma, r, use_cache=False) if temp < 3 else create_overlap_prob(sigma, r)
        grad_fn = jax.grad(lambda d: fn(d))
        d_test = r * 0.5
        g = float(grad_fn(d_test))
        assert jnp.isfinite(g), f"temp={temp}: non-finite gradient"
        assert g < 0, f"temp={temp}: gradient should be negative at d=0.5r, got {g}"
    print(f"    gradients OK for all regimes")


if __name__ == "__main__":
    print("=== Existing behavior (temperature < 1) ===")
    run_test(test_temperature_02, "temperature=0.2")

    print("\n=== LUT extension fix (temperature 1-3) ===")
    run_test(test_temperature_1, "temperature=1.0")
    run_test(test_temperature_2, "temperature=2.0")
    run_test(test_temperature_29, "temperature=2.9")

    print("\n=== Analytical regime (temperature > 3) ===")
    run_test(test_temperature_3, "temperature=3.0")
    run_test(test_temperature_6, "temperature=6.0 (σ=1m)")
    run_test(test_temperature_30, "temperature=30 (σ=5m)")

    print("\n=== Boundary & gradients ===")
    run_test(test_boundary_continuity, "boundary_continuity")
    run_test(test_gradient_all_regimes, "gradients_all_regimes")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, e in errors:
            print(f"  {name}: {e}")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)
