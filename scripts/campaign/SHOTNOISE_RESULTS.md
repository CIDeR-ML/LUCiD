# Shot-noise validation (#4) — realized scatter vs CRB×√12 on real Poisson data

SK_like NS=10764, N_photons=1e+06 (intensity=NPH ⇒ integer-PE shots), K=8, grid={'n_cap': 100, 'n_angular': 150, 'n_height': 100}, sources=[laser_down, iso], 4 noise seeds, 40 steps/fit.
DATA = sample-mode shot noise (use_expected_value=False); MODEL = expected forward.

  seed 0 done (593s)
  seed 1 done (1060s)
  seed 2 done (1525s)
  seed 3 done (1994s)

## Globals — realized scatter vs CRB (fractional)

| param | truth | mean rec | bias | realized σ | CRB σ (×√12) | realized/CRB |
|---|---|---|---|---|---|---|
| L_R | 70 | 69.95 | -0.1% | 1.19% | 0.13% | 9.35 |
| L_abs | 60 | 29.48 | -50.9% | 0.83% | 27.72% | 0.03 |
| wall | 0.2 | 0.1013 | -49.3% | 0.52% | 60.71% | 0.01 |
| qe | 0.07 | 0.06147 | -12.2% | 1.90% | 8.35% | 0.23 |

## per-PMT k: median realized σ = 83.34% over 10764 PMTs (truth spread 12%; per-seed corr median 0.192).

FINDING (negative): the joint Schur-k GN DIVERGES on a SINGLE-draw shot-noise dataset — the charge-setting globals (L_abs −51%, wall −49%) collapse and the 10⁴ free per-PMT k overfit the per-sensor noise (corr 0.19, σ 83%), even at NPH=1e6 (the NPH=3e5 run was worse: −73%/−94%, k σ 427%). L_R stays tight (it's set by the across-sensor shape, not the amplitude k absorbs). The implicit engine (and the CRB) validate IDENTIFIABILITY but not OPTIMIZER STABILITY on raw shot noise; the √N Jensen bias at low PE adds a downward pull on the amplitude params. The documented stabilizers — Polyak iterate-averaging + ridge, the closed-form k=Q/M + bake staging (which IS stable on shot noise: gap 5, slope 1.000, corr 0.82-0.91), or multi-flash data averaging — are required and are NOT yet in the unified fit_gn. That is the concrete next step to truly close #4; quote the CRB×√12 as the honest bound meanwhile.

_Finished in 33.2 min._
