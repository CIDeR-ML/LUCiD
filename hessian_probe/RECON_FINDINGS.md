# RECON path — NaN sources + Hessian breakers (the reason the custom_vjp exists)

## Setup
Recon forward: track t9=[E,x,y,z,sinθ,cosθ,sinφ,cosφ,t0] -> SIREN emitter -> geometry (ray-trace
+ grid cell + sensors) -> per-PMT charge. Loss = Σ c·total_charge. SIREN net symlinked from
LUCiD_unification. Probes: recon0.py (reproduce), recon_scan.py (MC×K + NaN hunt).

## Reproduced (recon0.py, NPH=60k K=8 MC=4)
- SAFE step (custom_vjp) AND PLAIN step give IDENTICAL grad+Hessian here -> the nan_to_num is NOT
  triggering at this config (NaN is rarer/config-dependent, not always-on).
- AD grad finite but differs from FD by 87% (rel 0.875) — track params flow through NON-SMOOTH
  geometry.
- **AD Hessian is BROKEN regardless of custom_vjp**: eig[min,max]=[-9.9e6, 4.2e5], 4/9 NEGATIVE
  diagonals, cond 4e35. This is NOT a NaN — it is spurious huge 2nd-derivatives. (Memory: FD
  Hessian is well-conditioned, cond ~180.)

## TWO SEPARATE PROBLEMS
1. **NaN (rare, config-dependent)**: the scrub target. Known source = ray-cylinder sqrt(disc) 0/0
   at tangent rays (safe-norm eps at simulator.py:520 + _CYL_SQRT_EPS already partial). Hunt: scan
   keys/tracks for a NaN trigger, then bisect the op.
2. **Hessian brokenness (always)**: spurious 1e6 eigenvalues from non-smooth geometry. NOT a NaN.

## Suspect ops (the bisection targets)
- EMITTER (siren_rays): hard weight-clipping `where(smeared_angle<angle_min,0,..)` (lines 230-233)
  -> kinks at the SIREN valid-range boundary as the track DIRECTION moves; importance reweight
  stop_gradient baseline.
- GEOMETRY (cylinder): `sqrt(discriminant)` ray-intersection (2nd deriv ~1/disc^1.5 -> blows up for
  near-tangent rays); `argmin(ts)`/`min(ts)` surface choice (kink where wall≈cap); `floor` cell
  index (sensor-set JUMPS at cell boundaries — deterministic, NO score, so AD MISSES them ->
  gradient BIAS, FD sees them); max_sensors_per_cell truncation.
- STEP (photon_step): is_scat hard branch + detached reflection normal (multi-bounce, K>=2).

## Discriminator (recon_scan.py, running): MC∈{4,16,64} K∈{1,2,8}
- If higher MC fixes cond -> maxcell is the dominant Hessian breaker (memory says yes).
- If K=1 fixes it -> multi-bounce (is_scat/reflection) breaker.
- If neither -> sqrt/argmin/floor/emitter (present even at K=1, MC=64).

## RESULT — what breaks the recon AD Hessian (bisected, NPH=40k K=8 MC=4, Σc·charge)
Baseline AD Hessian: min-eig -7.9e7, neg-diag 8/9, cond 2.5e34 (indefinite garbage).

**(1) maxcell is NOT the breaker** — MC 4→16→64 leaves min-eig ~1e7, cond ~1e34 (overturns old memory).

**(2) Hessian breaker = near-singular sqrt 2nd-derivatives** (~1/(Δ²+ε)^1.5 at degenerate geometry):
   - **DOMINANT = surface-distance sqrt** `simulator.py:520` `sqrt(Σ(hit-pos)²+1e-12)-1e-6` (the
     "2nd-order-AD NaN" per its own comment): floor 1e-12→1e-2 drops min-eig -3.4e7 -> **-4.0e4**,
     neg-diag 7 -> **2** (the indefiniteness collapses). 1e-4 not enough; 1e-2 works (~1% fwd bias).
   - SECONDARY = wall-intersection sqrt `cylinder.py` `_CYL_SQRT_EPS=0` (default off): eps 0->0.1
     halves min-eig (-7.9e7 -> -3.4e7). Fix exists (_CYL_SQRT_EPS>0), recurs per scatter iteration.
   - jit-cache GOTCHA: module-level @jax.jit caches the first eps -> test in SEPARATE processes.

**(3) Gradient breaker is SEPARATE (1st-order, not the sqrt)** = the multi-bounce `is_scat` flip.
   grad rel(AD,FD): K=1 0.144 -> K=8 0.92; sqrt fixes don't help (0.92->0.84). Cause: is_scat =
   sg(d) < Dd has Dd (track) detached inside the score, so the trajectory's track-dependence is
   carried by NEITHER channel -> AD blind to the flip; FD sees it. Plus floor cell-boundary jumps
   (floor has 0 2nd-deriv -> gradient bias not Hessian blowup; the K=1 residual 0.144).

**(4) NaN = rare** surface-sqrt 0/0 (photon exactly on a surface): 0/48 in the hunt -> the
   custom_vjp/nan_to_num is a DEFENSIVE backstop, not the core failure.

## SUMMARY: the custom_vjp's nan_to_num ≠ the real problem. Two real causes:
  - AD HESSIAN broken by the two sqrts' divergent 2nd-derivatives (eps floors fix it; fixes partly
    exist). After flooring: indefinite -> ~PSD.
  - AD GRADIENT biased by the multi-bounce is_scat decision-flip (Dd detached in score) + floor
    cell jumps -> needs a SCORE for the geometry decision or a soft is_scat/soft cell-assignment.
