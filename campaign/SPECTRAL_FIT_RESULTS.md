# Spectral λ-curve FIT recovery — unified GN recipe (gap 2)

SK_like NS=10764, N_photons=3e+05, K=8, grid={'n_cap': 100, 'n_angular': 150, 'n_height': 100}, sources=5 lasers+5 iso @ control λ, steps=100, start perturbation ±15%.
Truth abs_dev=[np.float64(1.2), np.float64(1.1), np.float64(1.0), np.float64(0.95), np.float64(0.9)], rayleigh_dev=[np.float64(0.9), np.float64(0.95), np.float64(1.0), np.float64(1.05), np.float64(1.1)] (deviated ≠1).

loss-free GN fit of ['abs_dev', 'rayleigh_dev'] (10 control points), start ±15% off truth …
  fit done (3369s)

| curve | 337nm | 375nm | 398nm | 405nm | 445nm |
|---|---|---|---|---|---|
| **L_abs(λ) dev** truth | 1.200 | 1.100 | 1.000 | 0.950 | 0.900 |
| recovered | 1.251 | 1.027 | 0.873 | 0.823 | 0.987 |
| frac err | 4.3% | 6.6% | 12.7% | 13.4% | 9.7% |
| CRB σ | 1.96% | 2.05% | 1.72% | 1.62% | 0.82% |
| **L_R(λ) dev** truth | 0.900 | 0.950 | 1.000 | 1.050 | 1.100 |
| recovered | 1.019 | 0.981 | 1.071 | 1.066 | 1.250 |
| frac err | 13.2% | 3.3% | 7.1% | 1.5% | 13.6% |
| CRB σ | 0.00% | 0.00% | 0.00% | 0.01% | 0.01% |

**Max per-point fractional recovery error = 13.6%** over all 10 control points (joint abs+Rayleigh λ-deviation fit). Recovers the deviated curves from a ±15% start — the flexible-curve FIT (not just CRB), on the unified recipe.

_Finished in 61.2 min._
