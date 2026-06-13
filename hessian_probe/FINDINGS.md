# AD vs FD for the calibration forward — systematic findings

Goal: obtain AD gradient == FD gradient and AD Hessian == FD Hessian for the calibration forward.
Probes in `hessian_probe/`. Forward = `build_calibration_problem` (real calib forward); FD = CRN
central (the SourceModel recipe). SK_like, laser source, K=8.

## Bisection chain (where the gradient lives / dies)
1. **Single step** (`step.py`): `photon_iteration_update_factors` is AD-clean —
   d(detect_prob)/d(scatter,absorption) AD == FD exactly; the DiCE `logp_increment` score AD == FD;
   the `custom_vjp` "safe" wrapper gives IDENTICAL grads to plain (so **nan_to_num is NOT the killer**).
2. **Assembly** (scan + make_hits): weights = depositions·survival·detect_probs·dice_dep; make_hits =
   `segment_sum` — all differentiable. NOT the killer.
3. **`_get_optical_arrays` / `evaluate_optical_model`** = THE first killer:
   - **`wavelength_mode=True` (DEFAULT)**: scatter/absorption come from the MEDIUM(λ) model, so the
     trainable DetectorParams scalars `scatter_length`/`absorption_length` are disconnected from AD →
     **AD grad == 0** for all sensors (FD shows a tiny key-noise signal). To calibrate the SCALAR
     optical params, use `wavelength_mode=False` (or calibrate the medium/deviation-curve params).

## With `wavelength_mode=False` (scalar path) — AD flows; the split is score-vs-reparam
- **absorption_length: AD == FD** — pure PATHWISE (`exp(-d/L_abs)`), AD-exact. ✓
- **scatter_length: AD ≠ FD** — scatter enters via the DISCRETE free-path decision `is_scat = d<Dd`
  with `d = stop_gradient(d_live)` + the DiCE **score** `lf`. AD computes the score-function gradient;
  CRN-FD (finite h) moves the decision boundary (reparam/pathwise-through-the-flip). Same expectation,
  different per-realization value AND different variance (score is higher-variance). 6-key avg does
  NOT yet converge (AD -55 vs FD -94).

## What needs to change (hypotheses to validate)
- (A) **wavelength_mode**: document/route — scalar-optical calibration must use the scalar path, or the
  trainable set in λ-mode is the medium curves (different params).
- (B) **scatter (and any discrete-decision param)**: to get AD==FD, the decision must flow PATHWISE,
  not via the score — i.e. reparameterize the free-path so `d` is LIVE in the rate (Option C already
  does this for TIME), and/or SOFTEN the hard `is_scat` branch (temperature) so the branch selection
  is differentiable. Alternatively accept score==FD only in expectation (needs many keys) — measure.
- The Hessian inherits the gradient split: absorption block AD==FD; scatter block AD≠FD.

## Per-parameter categorization (wavelength_mode=False, mse loss) — MEASURED
| param | channel | AD vs FD per-key |
|---|---|---|
| absorption_length | pathwise exp(-d/L_abs) | == EXACT |
| wall_reflection_rate | pathwise (mult. refl_prob) | == EXACT |
| sensor_reflection_rate | pathwise | == EXACT |
| qe | pathwise (linear in make_hits) | == EXACT |
| scatter_length | DISCRETE free-path is_scat → DiCE score | != per-key; FD->AD as keys↑ |
| mie_scatter_length | DISCRETE Mie/Rayleigh + free-path → score | != per-key; FD->AD as keys↑ |
| g (HG asymmetry) | DISCRETE HG-angle sample → score | != per-key |

KEY INSIGHT: FD-CRN converges TOWARD AD as #keys grows (scatter FD -190→-91 → AD -72; mie
-78→-3.4 → AD -0.3). So **AD (score) is the unbiased LOW-variance gradient; FD-CRN is the noisy
SAME-expectation one**. For 4/7 params they're already byte-equal; the 3 discrete-decision params
agree only in expectation.

## What needs to change (the deliverable)
1. **wavelength_mode** (usage/routing): scalar-optical calibration MUST run `wavelength_mode=False`
   (λ-mode disconnects the scalar params → AD grad 0; the λ-mode trainables are medium curves, which
   need their own AD audit). Add a guard/doc so a λ-mode scalar-param fit can't silently give 0 AD.
2. **Discrete-decision params (scatter, mie, g) — to get per-key AD==FD, REPARAMETERIZE pathwise:**
   - free-path `d`: currently `d = stop_gradient(d_live)` for the trajectory; the deposit detect_prob
     is ALREADY Rao-Blackwellized (reach=exp(-μ·Dd) is pathwise-clean — step-level AD==FD). The
     residual mismatch is the TRAJECTORY: the hard `is_scat`/`is_mie` branches (detached) carried by
     the score. Fix = make the trajectory's scatter dependence pathwise: `d` LIVE for the next
     position AND soften/Rao-Blackwellize the hard branch (temperature on is_scat, or sum both
     branches weighted by prob) so AD = the reparam gradient = FD per-key.
   - g: reparameterize the HG inverse-CDF angle sample so g flows pathwise (not sg(cmie)+score).
   - ALTERNATIVE (cheaper): keep the score but treat AD as the REFERENCE (it's lower-variance &
     unbiased) — then "match" = drive FD to AD with many keys / antithetic CRN. The user wants AD==FD,
     so reparam is the real fix.
3. **Hessian**: inherits the split. Pathwise params → AD Hessian (rev-over-rev) should match FD-of-grad
   once gradients match. NOTE: `jax.hessian` (jacfwd∘jacrev) RAN in the scalar-path probe without the
   custom_vjp forward-mode error — re-verify whether B1's forward-mode block actually bites here.

## RIGOROUS RESOLUTION (decisive experiments: deep.py, bias.py, fwdmode.py, fwdmode2.py)

### The clean two-category truth (per PHYSICAL CHANNEL, mode-independent)
- **PATHWISE channels — AD == FD EXACT** (truncation-limited; h-scan: FD==AD at h=1e-2/1e-3,
  degrades only by roundoff at h<=1e-4): **absorption_length, wall_reflection_rate,
  sensor_reflection_rate, qe** (mono) and **abs_dev, qe_dev** (λ-mode deviation curves).
- **DISCRETE-decision channels — AD (score) and FD (CRN) share the SAME EXPECTATION but FD is
  ~19,000x higher variance**: **scatter_length, mie_scatter_length, g** (mono) and **rayleigh_dev,
  mie_dev** (λ-mode). DECISIVE (bias.py, linear functional, n=2048): AD=-22.86±0.18 (std 8.1) vs
  FD=-30.9±25 (std 1135) -> |AD-FD|/sigma_FD<1.5 (consistent => both UNBIASED). FD-CRN DIVERGES as
  h->0 (variance ∝ 1/h); the SourceModel "CRN-FD is clean and low-noise" claim is FALSE here. **AD
  is the correct LOW-variance gradient; FD is unbiased but catastrophically noisy.**

### wavelength_mode is ORTHOGONAL
λ-mode scalars are inert (ΔM~5e-7 under ×e pert = numerical noise); the λ-mode trainables are the
deviation curves, which carry the SAME pathwise/discrete split. Mode is a routing detail, not the
AD issue. (Add a guard: fitting an inert field for the mode should warn, not silently give 0 grad.)

### Why FD is used + THE FIX (B1, validated)
- The GN needs the (ND x P) Jacobian. **jacfwd** (P forward passes) is the efficient way -> BLOCKED
  by the custom_vjp ("can't apply forward-mode autodiff (jvp) to a custom_vjp function"; confirmed).
  **jacrev** over ND sensors OOMs. So FD is the only efficient fallback -> noisy for discrete params.
- The earlier "jax.hessian WORKED" was an ARTIFACT of the λ-mode disconnected (zero) gradient.
- **FIX = drop the custom_vjp** (it only scrubs NaN cotangents; the safe-norm eps at simulator.py:520
  already drives the known NaN to 0). VALIDATED (fwdmode2.py, plain step): jacfwd(L)==jacrev(L)
  exactly, finite; **jacfwd FULL (10764,2) Jacobian runs, finite, NO OOM**. => the efficient
  low-variance AD Jacobian is available for ALL params, retiring FD.

### What this means for "AD == FD" and the Hessian
1. PATHWISE params: AD == FD already (gradient AND Hessian).
2. DISCRETE params: AD (score) and FD (reparam) agree only IN EXPECTATION (FD ~140x noisier in std).
   Per-key AD==FD is NOT achievable without REPARAMETERIZING the discrete decisions to pathwise
   (live `d` + soft `is_scat`/`is_mie` + reparam HG-angle in `g`) -- a forward change with a
   temperature-bias tradeoff. The BETTER move: use AD (it's the correct low-variance estimator);
   the "match" to assert is jacfwd == jacrev (holds exactly), not AD == noisy-FD.
3. HESSIAN: post-custom_vjp-removal, jax.hessian (jacfwd∘jacrev) works -> AD Hessian available.
   GN metric JᵀWJ from the AD Jacobian is self-consistent. True-Hessian discrete blocks inherit the
   same score-vs-reparam expectation-only agreement (worse at 2nd order).

### Recommendation
Drop the custom_vjp (drive NaNs to zero at source) -> jacfwd/jax.hessian work -> switch the
calibration Jacobian/Hessian from FD to AD (jacfwd). AD==FD for the 4 pathwise channels; AD is the
correct low-variance reference for the 3 discrete channels (FD there is noise, agreeing only in
expectation). For literal per-key AD==FD on the discrete channels, reparameterize the decisions
pathwise (separate, bigger change). The custom_vjp removal is the single highest-value fix.

## MECHANISM — WHY each param behaves as it does (derived from the step + CONFIRMED by kscan.py)

Every optical param enters the per-PMT charge through one or both of two channels:
- **DEPOSIT** = `detect_prob = reach·(1-refl_prob)·atten_surf`, `reach=exp(-mu_tot·Dd)` — the
  RAO-BLACKWELLISED expected deposit (Dd is geometry, fixed; the free-path decision is integrated
  out). PURE PATHWISE in its params. -> AD==FD EXACT at ALL K.
- **TRAJECTORY** = the SAMPLED hard path (`is_scat`, `is_mie`, scatter direction). Optical params
  enter it via the DiCE SCORE `lf/la` (decision vars `d`,`cmie` detached). A step's decision only
  affects a LATER step's deposit (via `dice_dep=exp(log_p_{<k})`), so the trajectory channel is
  ZERO at K=1 and GROWS with K. AD computes the score (low variance); FD-CRN computes the
  reparam-through-the-branch (rare O(1) decision flips amplified by 1/h -> variance ∝ 1/h, blows up).

Per parameter:
| param | channels | K=1 | K>=2 |
|---|---|---|---|
| absorption_length | DEPOSIT only (atten_surf) | AD==FD (var ratio 1.0) | AD==FD (1.0 at EVERY K) |
| wall/sensor_reflection_rate | DEPOSIT only (1-refl_prob; scalar model lr=0) | AD==FD | AD==FD |
| qe | DEPOSIT only (linear in make_hits) | AD==FD | AD==FD |
| scatter_length | DEPOSIT (reach via mu_tot) + TRAJECTORY (free-path lf) | AD==FD var 1.0 | var(FD)/var(AD) 1e4-2.4e4 |
| mie_scatter_length | + Mie/Rayleigh split (la); THIN channel 1/mie<<1/scatter | AD==FD var 1.0 | var ratio 6e3-1.2e5 |
| g (HG asym) | TRAJECTORY ONLY (enters only hg_logpdf; sampler detaches g) | grad == 0 | appears; var ratio ~3e2 |

KSCAN (Nphot=20k, 48 keys): scatter K=1 AD=-10.47±1.96 FD=-10.46±1.96 (ratio 1.0) -> K=8 FD std 1001
(ratio 24000); mie K=1 ratio 1.0 -> K=8 ratio 1.2e5; g K=1 grad=0 -> K=8 AD=+1.29±12.9 FD=-50.8±239;
absorption ratio 1.0 at K=1,2,4,8. PREDICTION CONFIRMED.

## Why "AD==FD" is the WRONG goal for scatter/mie/g
The score (current AD) is the LOW-variance UNBIASED Rao-Blackwellised estimator. Forcing AD==FD
per-key means computing the reparam-through-the-branch gradient = the HIGH-variance one; done with a
HARD is_scat it also MISSES the boundary-delta term -> BIASED; done with a SOFT branch it carries an
O(tau) temperature bias. So matching FD degrades AD. The right framing: AD (score) and FD (reparam)
are two unbiased estimators of the SAME gradient; AD is ~140x lower std; the ONLY reason calibration
uses FD is the custom_vjp blocking jacfwd. FIX = drop custom_vjp, use jacfwd (the low-var AD score
Jacobian) for ALL params. AD==FD exact on the 4 deposit-only channels; AD is the SUPERIOR reference
on the 3 trajectory channels. (Per-key AD==FD there is neither achievable nor desirable.)
