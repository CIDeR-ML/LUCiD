# Calibration campaign — unified framework vs mie_hunter

SK_like, NS=10764, N_photons=5e+05, intensity=5e+09, QUICK=True.
Truth: L_R=70 L_M=30 g=0.9 L_abs=60 wall=sensor=0.20 qe=0.07.
CRB = fractional σ on log-params, ×√12 honesty factor (the implicit engine is √12 quieter than Poisson).

## 1-2. CRB of the 7 optical+reflection globals vs source diversity

| param | single_laser | laser+iso | multi_source | mie_hunter |
|---|---|---|---|---|
> built CRB[single_laser] in 43s
> built CRB[laser+iso] in 82s
> built CRB[multi_source] in 160s
| g | 163.39% | 0.00% | 0.00% | loose |
| L_R | >1e4% | >1e4% | 8701.15% | tight |
| L_M | >1e4% | >1e4% | 8701.15% | ~5% |
| L_abs | >1e4% | >1e4% | 8701.15% | corr w/ qe -0.99 |
| wall_refl | >1e4% | 0.34% | 0.20% | ~0.6% (tight) |
| sensor_refl | >1e4% | 0.74% | 0.39% | flat (info-limited) |
| qe | 7359.21% | 0.07% | 0.05% | ~1% (single-λ) |

Expected (mie_hunter): L_M and sensor_refl tighten with source diversity; sensor_refl stays the loosest (few photons hit sensors); single_laser leaves L_M↔k and scatter↔abs degenerate (huge σ).

## 3. GN recovery (multi_source), from a 20%-perturbed start

fit 50 steps in 1130s

| param | truth | recovered | frac err |
|---|---|---|---|
| g | 0.9 | 0.9507 | 5.6% |
| L_R | 70 | 63.84 | 8.8% |
| L_M | 30 | 24.97 | 16.8% |
| L_abs | 60 | 49.45 | 17.6% |
| wall_refl | 0.2 | 0.1974 | 1.3% |
| sensor_refl | 0.2 | 0.2083 | 4.2% |
| qe | 0.07 | 0.06984 | 0.2% |

## 4. Charge variance (moments mode): Fano v/m = g·(1+w²) breaks QE↔gain

measured Fano v/m median = 1.1228  (expect g·(1+w²) ≈ 1.1225 at median g≈1)
→ implied w from Fano (median g=1): w ≈ 0.350  (truth w=0.35)
→ per-PMT gain recoverable from v/m and mean (g=v/m/(1+w²), then k=mean/(g·μ)); SPE width w from the population Fano. This is the QE↔gain degeneracy break.


_Campaign finished in 24.5 min._
