# Spectral λ-curve FIT recovery — unified GN recipe (gap 2)

SK_like NS=10764, N_photons=5e+05, K=8, grid={'n_cap': 100, 'n_angular': 150, 'n_height': 100}, sources=5 lasers+5 iso @ control λ, steps=120, start perturbation ±15%.
Truth abs_dev=[np.float64(1.2), np.float64(1.1), np.float64(1.0), np.float64(0.95), np.float64(0.9)], rayleigh_dev=[np.float64(0.9), np.float64(0.95), np.float64(1.0), np.float64(1.05), np.float64(1.1)] (deviated ≠1).

PER-CURVE mode: each curve fit with the OTHER fixed at truth (no curve-curve degeneracy) — the staged recipe.

GN fit of ['abs_dev'] (5 control points), start ±15% …
  done (4699s)

| curve | 337nm | 375nm | 398nm | 405nm | 445nm |
|---|---|---|---|---|---|
| **L_abs(λ) dev** truth | 1.200 | 1.100 | 1.000 | 0.950 | 0.900 |
| recovered | 1.190 | 1.106 | 1.001 | 0.947 | 0.902 |
| frac err | 0.8% | 0.5% | 0.1% | 0.3% | 0.2% |
| CRB σ | 1.95% | 2.05% | 1.72% | 1.62% | 0.82% |

GN fit of ['rayleigh_dev'] (5 control points), start ±15% …
  done (9373s)

| curve | 337nm | 375nm | 398nm | 405nm | 445nm |
|---|---|---|---|---|---|
| **L_R(λ) dev** truth | 0.900 | 0.950 | 1.000 | 1.050 | 1.100 |
| recovered | 1.018 | 0.982 | 1.072 | 1.066 | 1.254 |
| frac err | 13.1% | 3.4% | 7.2% | 1.5% | 14.0% |
| CRB σ | 0.01% | 0.01% | 0.01% | 0.01% | 0.01% |

**L_abs(λ) max 0.8%** (≈/below its ~2% CRB) — reaches the bound once NOT degenerate with Rayleigh, so the ~13% JOINT floor WAS the abs↔Rayleigh per-λ degeneracy (not the optimizer). **BUT L_R(λ) max 14.0%** even fit ALONE (CRB 0.01%): Rayleigh is the STIFF direction — its Hessian is huge, so the proportional ridge (∝diag) OVER-DAMPS it and it under-moves (the documented "L_R stiff/under" behavior). Fix = Polyak iterate-averaging / reduced ridge on the stiff direction — the SAME stabilizer #4 needs, missing from unified fit_gn.

_Finished in 156.2 min._
