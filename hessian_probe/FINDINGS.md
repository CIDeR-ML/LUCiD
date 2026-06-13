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
