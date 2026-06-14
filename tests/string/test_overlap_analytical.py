"""
Compare the exact disk-integral overlap with the analytical approximation
across the full range of σ/r ratios.

The analytical formula: overlap(d) ≈ (r²/2σ²) × exp(-d²/2σ²)
is exact for σ >> r (DOM disk is a point relative to the Gaussian).
This script finds where it breaks down.

Run: python string/test_overlap_analytical.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")

from lucid.overlap import create_overlap_prob, precompute_lookup


def analytical_overlap(d, r, sigma):
    """Point-source approximation: disk is negligible compared to Gaussian width."""
    return (r**2 / (2 * sigma**2)) * np.exp(-d**2 / (2 * sigma**2))


def compute_exact_overlap(d_values, r, sigma):
    """Compute exact disk-integral overlap using the LUT machinery."""
    n_theta, n_rho = 2000, 2000
    theta_vals = jnp.linspace(0, 2 * jnp.pi, n_theta)
    rho_vals = jnp.linspace(0, r, n_rho)

    from lucid.overlap import integral_f_of_d

    results = []
    for d in d_values:
        val = integral_f_of_d(float(d), r, sigma, theta_vals, rho_vals)
        results.append(float(val))
    return np.array(results)


def cross_section_integral(overlap_fn, d_max, n_points=5000):
    """Compute ∫₀^d_max overlap(d) × 2πd dd — should equal πr² for unbiased kernel."""
    d_vals = np.linspace(0, d_max, n_points)
    integrand = np.array([overlap_fn(d) * 2 * np.pi * d for d in d_vals])
    return np.trapz(integrand, d_vals)


if __name__ == "__main__":
    r = 0.165  # IceCube DOM radius

    # Test distances: from 0 to 5r, dense near r
    d_test = np.concatenate([
        np.linspace(0, 0.5 * r, 10),
        np.linspace(0.5 * r, 2 * r, 30),
        np.linspace(2 * r, 5 * r, 10),
    ])

    sigma_over_r = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]

    print(f"r = {r} m (IceCube DOM radius)")
    print(f"\n{'σ/r':>6s}  {'σ (m)':>8s}  {'exact(0)':>10s}  {'analyt(0)':>10s}  "
          f"{'rel_err(0)':>10s}  {'max_rel_err':>12s}  {'max_err_at':>10s}  "
          f"{'πr² exact':>10s}  {'πr² analyt':>10s}")
    print("-" * 115)

    target_cross = np.pi * r**2

    for ratio in sigma_over_r:
        sigma = ratio * r

        # Exact overlap via numerical integration
        exact_vals = compute_exact_overlap(d_test, r, sigma)

        # Analytical approximation
        analyt_vals = analytical_overlap(d_test, r, sigma)

        # Relative error (avoid division by tiny numbers)
        mask = exact_vals > 1e-10
        if mask.any():
            rel_err = np.abs(exact_vals[mask] - analyt_vals[mask]) / exact_vals[mask]
            max_rel_err = rel_err.max()
            max_err_idx = np.where(mask)[0][rel_err.argmax()]
            max_err_d = d_test[max_err_idx]
        else:
            max_rel_err = 0.0
            max_err_d = 0.0

        # Relative error at d=0
        if exact_vals[0] > 1e-10:
            rel_err_0 = abs(exact_vals[0] - analyt_vals[0]) / exact_vals[0]
        else:
            rel_err_0 = 0.0

        # Cross-section integrals (mean preservation check)
        def exact_fn(d):
            return float(compute_exact_overlap(np.array([d]), r, sigma)[0])

        def analyt_fn(d):
            return analytical_overlap(d, r, sigma)

        d_max_int = max(10 * r, 5 * sigma)
        cs_exact = cross_section_integral(exact_fn, d_max_int, n_points=2000)
        cs_analyt = cross_section_integral(analyt_fn, d_max_int, n_points=2000)

        print(f"{ratio:6.1f}  {sigma:8.4f}  {exact_vals[0]:10.6f}  {analyt_vals[0]:10.6f}  "
              f"{rel_err_0:10.2%}  {max_rel_err:12.2%}  d={max_err_d:7.4f}m  "
              f"{cs_exact:10.6f}  {cs_analyt:10.6f}")

    print(f"\nTarget cross-section πr² = {target_cross:.6f}")

    # Detailed profile comparison for key σ values
    print(f"\n{'='*80}")
    print("Detailed profiles at selected σ/r ratios")
    print(f"{'='*80}")

    for ratio in [0.2, 0.5, 1.0, 3.0, 10.0]:
        sigma = ratio * r
        d_fine = np.linspace(0, max(3 * r, 3 * sigma), 50)
        exact = compute_exact_overlap(d_fine, r, sigma)
        analyt = analytical_overlap(d_fine, r, sigma)

        print(f"\nσ/r = {ratio}, σ = {sigma:.4f}m:")
        print(f"  {'d/r':>6s}  {'d (m)':>8s}  {'exact':>10s}  {'analyt':>10s}  {'rel_err':>10s}")
        for i in range(0, len(d_fine), max(1, len(d_fine) // 12)):
            d = d_fine[i]
            e = exact[i]
            a = analyt[i]
            re = abs(e - a) / (e + 1e-30) if e > 1e-10 else 0.0
            print(f"  {d/r:6.2f}  {d:8.4f}  {e:10.6f}  {a:10.6f}  {re:10.2%}")
