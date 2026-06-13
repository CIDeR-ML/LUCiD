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
