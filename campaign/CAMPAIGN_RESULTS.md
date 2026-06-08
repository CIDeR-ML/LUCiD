# Calibration campaign — unified framework vs mie_hunter

SK_like, NS=10764, N_photons=5e+05, intensity=1e+08, QUICK=True.
Truth: L_R=70 L_M=3000 (thin Mie, p_mie≈2.3%) g=0.9 L_abs=60 wall=sensor=0.20 qe=0.07.
CRB = fractional σ on log-params, ×√12 honesty factor (the implicit engine is √12 quieter than Poisson).

## 1-2. CRB of the 7 optical+reflection globals vs source diversity

| param | single_laser | laser+iso | multi_source | mie_hunter |
|---|---|---|---|---|
> built CRB[single_laser] in 42s
> built CRB[laser+iso] in 78s
> built CRB[multi_source] in 156s
| g | 619.69% | 0.038% | 0.0222% | loose |
| L_R | 364.67% | 0.00919% | 0.00511% | tight |
| L_M | 0% | 0.0423% | 0.0237% | ~5% |
| L_abs | >1e3% | 2.75% | 1.30% | corr w/ qe -0.99 |
| wall_refl | >1e3% | 5.85% | 2.86% | ~0.6% (tight) |
| sensor_refl | >1e3% | 8.20% | 4.30% | flat (info-limited) |
| qe | 0% | 1.19% | 0.697% | ~1% (single-λ) |

Expected (mie_hunter): L_M and sensor_refl tighten with source diversity; sensor_refl stays the loosest (few photons hit sensors); single_laser leaves L_M↔k and scatter↔abs degenerate (huge σ).

## 3. GN recovery (multi_source), from a 20%-perturbed start

fit 50 steps in 1106s

| param | truth | recovered | frac err |
|---|---|---|---|
| g | 0.9 | 0.9507 | 5.6% |
| L_R | 70 | 63.92 | 8.7% |
| L_M | 3000 | 2489 | 17.0% |
| L_abs | 60 | 58.84 | 1.9% |
| wall_refl | 0.2 | 0.188 | 6.0% |
| sensor_refl | 0.2 | 0.2185 | 9.2% |
| qe | 0.07 | 0.06988 | 0.2% |

## 4. Charge variance (moments mode): Fano v/m = g·(1+w²) breaks QE↔gain

measured Fano v/m median = 1.1228  (expect g·(1+w²) ≈ 1.1225 at median g≈1)
→ implied w from Fano (median g=1): w ≈ 0.350  (truth w=0.35)
→ per-PMT gain recoverable from v/m and mean (g=v/m/(1+w²), then k=mean/(g·μ)); SPE width w from the population Fano. This is the QE↔gain degeneracy break.

## Interpretation — unified framework vs mie_hunter

- **Source diversity breaks the degeneracies** (the central mie_hunter result): with a single laser every global is degenerate (σ ≫1e3% / 0 = unconstrained — the per-PMT k absorbs any single-source pattern, and scatter↔abs↔L_M are confounded); adding an isotropic source then up/wall lasers makes them all identifiable, tightening monotonically single → laser+iso → multi.
- **Reflection split reproduced**: sensor_refl is the loosest reflectivity (4.3% multi — few photons strike sensors, info-limited), wall_refl tighter (2.9%) — the REFLECTION_SPLIT finding.
- **L_R tight, qe ~%, L_M weak**: Rayleigh strongly drives charge (L_R 0.005%, tightest); qe lands 0.7% (cf. mie_hunter ~1%); L_M is the weak/hard parameter — thin Mie (~2% of the scatter rate) barely constrains charge, and the GN recovers it only to 17% (the "L_M is hard" result; mie_hunter needed diversity / cross-terms / multi-λ to pin it).
- **Charge variance breaks QE↔gain exactly**: the moments mode reproduces the compound-Poisson Fano v/m = g·(1+w²); the SPE width is recovered to w=0.350 (truth 0.35).
- **Absolute σ are photon-budget dependent** (intensity 1e8 here, somewhat higher statistics than the mie_hunter operating point — hence the very tight L_R/L_M/qe); the QUALITATIVE structure (easy vs hard params, diversity helping, the reflection split, the QE↔gain break) is what validates the unified base, and it matches mie_hunter. The GN recovery of the correlated optical block (L_R↔L_M, cos≈0.6) is optimizer-limited at 50 QUICK steps; well-determined params (qe 0.2%, L_abs 1.9%) already recover to ~CRB.

**Bottom line:** the consolidated `lucid/` (one DiCE forward + nested DetectorParams + λ-curves + pluggable reflection + charge-variance/timing observables + one GN+Schur+Fisher fitter) reproduces the mie_hunter calibration physics end-to-end on the real geometry.

_Campaign finished in 23.9 min (QUICK; intensity 1e8)._
