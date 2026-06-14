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
  gradient BIAS, FD sees them); max_candidates_per_ray truncation.
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

---

# ⚠️ CORRECTION (many-key test, recon_manykey.py) — sections (2)(3) above were SINGLE-KEY ARTIFACTS

The conclusions "AD Hessian broken" and "AD gradient biased" were drawn from a SINGLE key, comparing
AD to FD/eigenvalues. That repeats the exact calibration-arc mistake: FD-CRN through discrete branches
is the NOISY estimator, and a random-sign loss Σc·charge has NO reason to be PSD. The rigorous test —
AD-mean vs FD-mean over MANY keys (z-score), and AD-Hessian vs GT = d/dθ E[∇L] — settles it:

**GRADIENT (200 keys, all 9 params, K=1 AND K=8): AD == FD in expectation. Every z < 1.2 → AGREE.**
  - The single-key `grad_rel` 0.14→0.92 was pure FD VARIANCE. At K=8, std(FD) ≈ 12× std(AD)
    (x: 2060 vs 173; cosθ: 2872 vs 821). AD (DiCE score) is the LOW-VARIANCE UNBIASED estimator;
    FD-CRN variance explodes through the multi-bounce decision-flips. **AD is NOT biased.** There is
    no is_scat-flip gradient bug — same Rao-Blackwell story as the calibration discrete params.

**HESSIAN DIAG (200 keys, K=1 AND K=8): E[AD H_jj] == GT = d/dθ E[∇L]. Every z ≤ 1.1 → AGREE.**
  - The indefinite "8/9 neg-diag, −6e5" spectrum is the CORRECT 2nd derivative of the random-c probe
    loss — NOT a pathology. AD reproduces the true Hessian. (A physical loss — Poisson NLL / √-MSE —
    is locally PSD; Σc·charge with c~N(0,1) is indefinite by construction.) The sqrt eps floors change
    the MAGNITUDE/conditioning of genuine grazing/on-surface curvature; they do not fix a "bug".

**So nothing in the recon AD path is wrong.** The eps-toggle factorial measured real geometric
curvature (correctly reported by AD), not numerical breakage.

## NaN backstop is unnecessary (recon_nan_stress.py, PLAIN step, no custom_vjp/nan_to_num)
300 keys × 5 STRESS geometries (near-wall grazing, on-cap, axis-aligned, near-tangent-to-wall — the
exact on-surface / 0/0 firing conditions) at K=8: **NaN val 0, grad 0, Hessian 0 everywhere.**
The eps-INSIDE-sqrt floors (surface-dist `+1e-12`; discriminant `maximum(·,1e-6)`) remove the 0/0 at
source. custom_vjp+nan_to_num provably never needs to fire.

## FINAL — the correct fix (NO SIREN replacement)
1. **Drop the custom_vjp wrapper** (`make_photon_iteration_update_factors_safe` → plain step). It only
   ever scrubbed measure-zero sqrt-at-zero cotangents that the eps-inside-sqrt floors already prevent.
   This is the whole point: removing it UNBLOCKS forward-mode (jacfwd/jvp), the correct memory-light
   mode for the few-input→many-output recon/calibration Jacobian.
2. **Keep the eps-inside-sqrt floors** (surface-distance `+1e-12`; ray discriminant). Optionally switch
   the discriminant default to the C∞ additive floor (`_CYL_SQRT_EPS>0`) for better Hessian
   CONDITIONING at grazing rays — a conditioning nicety, not a correctness fix.
3. AD gradient and Hessian are unbiased and LOWER variance than FD (≈12× at K=8) → prefer AD over the
   current FD Jacobian in the fitter. SIREN emitter stays exactly as is — it was never the problem.
