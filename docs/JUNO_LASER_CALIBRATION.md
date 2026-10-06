# JUNO 532 nm laser calibration prototype

## Current observables

The repository-facing calibration entry point now uses every accepted PE
timestamp in a continuous-time Poisson point-process (PPP)
likelihood.  It does not use a first-hit statistic or the old optional charge
penalty.  Transport still produces PMT-by-time-cell masses, but the likelihood
evaluates the continuous rate at each observed PE time.

The optical time masses are now convolved with measured campaign IRFs.  The
available HDF5 file covers only part of the detector, so the current prototype
selects the highest-load record per serial number, peak-aligns and normalizes
each waveform, and averages it within Hamamatsu and NNVT families.  A future
complete IRF table can replace this two-family bank with one kernel per PMT
without changing the likelihood.  The remaining narrow Gaussian width is
only continuous interpolation, not an additional physical TTS.

All calibration paths share one validated `TimeGrid`.  Its finite observation
window must contain an integer number of uniform cells, and cell `b` is
centered at `time_min_ns + (b + 0.5) * bin_width_ns`.  The forward soft
histogram, synthetic sampler and event-wise likelihood consume this same
object, so they cannot independently choose `ceil`, `floor` or `round` for the
time dimension.

The synthetic full-geometry closure additionally retains each generated
`event_id`, `pmt_id` and `pe_time_ns`.  Its continuous-time model converts
each transport-bin mass into a fixed-width Gaussian component:

```text
lambda_p(t) = sum_b m[p,b] K_sigma(t - t_b)
q[p,b] = Phi((T_max - t_b) / sigma_p)
       - Phi((T_min - t_b) / sigma_p)
NLL = sum_e exposure_e * sum_p,b(m[p,b] q[p,b])
    - sum_PE log(exposure_event(PE) * lambda_p(t_PE))
```

With an event timing offset `delta_t[e]`, the actual model is

```text
lambda_e,p(t) = exposure_e * sum_b m[p,b]
                * K_sigma(t - t_b - delta_t[e]) + background_p(t)
```

`event_id` therefore selects both exposure and timing state.  The event term
evaluates the base optical intensity at `t_PE - delta_t[event_id]`; the CDF
integral uses window edges shifted by the same amount.  Applying an offset to
PE timestamps without changing the finite-window integral is not allowed.

For several source positions, the objective first adds every source PPP NLL
and then divides once by the parameter-independent global PE count:

```text
loss = sum_k NLL_k / max(sum_k N_PE,k, 1).
```

It does not normalize each source separately and then average sources.  The
division changes only numerical scale, not the optimum of the joint PPP.

No artificial `rate_floor` is applied.  If an observed PE has zero predicted
rate, the strict PPP NLL is infinite.  Synthetic closure performs an explicit
support check and records the minimum PE rate and number below a diagnostic
threshold.  Real-data calibration may add a uniform per-PMT background rate:

```text
lambda_ep(t) = exposure_e * lambda_signal,p(t)
             + lambda_background,p(t)
integral = sum_e(exposure_e) * sum_p,b(m[p,b] q[p,b])
         + N_event * sum_p(r_background,p) * window
```

The background therefore enters both the observed-event term and the exposure
integral.  A zero configured background keeps the strict signal-only model;
it does not silently hide a support failure.

## FSMP event and posterior-step contract

`eid` identifies a physical laser trigger and determines the PPP exposure.
`step` labels an FSMP posterior sample inside that event, so `(eid, step)` is
not treated as a new independent trigger.  The current prototype chooses one
prescribed `--fsmp-step` for every retained event.  A missing step is a hard
input error.  The loader requires the FSMP full-step table to contain one row
per channel, including empty `PEt` lists.  This makes the physical event ledger
auditable: zero-PE events contribute to the PPP integral even though they add
no log-rate term.

This fixed-step rule is intentionally simpler than the future posterior
marginalization

```text
log L_e = log[(1 / S) * sum_s L({t_ejs} | theta)].
```

The loader does not claim that marginalization has already been implemented.

The event term evaluates the original full-axis Gaussian density; it is not
renormalized when a component lies near an observation-window edge.  The PPP
integral uses the Gaussian CDF difference `q[p,b]`, so kernel probability
outside the observation window is not counted as observed signal.  The
configured `truncate_sigma` limits only the local numerical evaluation
stencil and does not renormalize the retained density.  Closure output records
the accepted mass fractions in both the configured window and an expanded
window for a window-stability check.

The default interpolation `sigma = 0.25 ns` is a narrow numerical kernel
applied after the measured IRF convolution.  Exact PE times affect the
likelihood continuously, making this a genuine unbinned closure test.  The
obsolete binned source-averaged objective and piecewise-constant diagnostic
mode have been removed so the forward closure, gradient audit and calibration
entry point have one statistical definition.

The repository physics target uses a `0.5 ns` numerical time basis, matching
the native campaign-IRF sampling and keeping the interpolation width at half
a cell.  The current 100-event full-geometry gradient qualification uses a
`2 ns` basis to keep the first end-to-end GPU test tractable.  It validates
gradient execution, not final time-grid convergence.  A physics closure must
still complete the documented `2, 1, 0.5 ns` convergence study before the
`0.5 ns` target can be called production-ready.

Every observation carries a positive `event_exposure`.  Synthetic closure
uses an explicit fixed value.  The real-data CLI requires one repeated
`--source-exposure` value for each `--source`; it neither infers exposure from
FSMP `nu_lc` nor silently substitutes the event count.  Until an absolute
Monitor-PMT normalization and prior are available, these values remain fixed
and no unconstrained exposure nuisance is fitted alongside `L_abs`.
Every observation also carries a finite `event_time_offsets_ns` array.  The
data loader accepts either scalars or arrays aligned with the retained
physical-event ledger, so Monitor-derived event timing can be connected later
without changing the likelihood interface.

## Restricted fit vector

The first full-geometry closure tests fit only

```text
log(scat_ls), log(abs_ls), z_theta,0, z_theta,1, z_theta,2
```

The LS and water refractive indices, all water optical parameters, Acrylic
properties and engineering-surface parameters remain fixed at the configured
forward-model baseline.  This prevents invalid refractive-index gradients or
water nuisances from contaminating the first LS closure test.

The three `z_theta` coordinates span an orthonormal centered-logit basis around
the configured JUNOSW nominal curve.  The origin is the nominal LSPhase
shape.  Every forward call maps the coordinates to four positive normalized
Bernstein weights.  Output files additionally convert those weights back to
the four JUNOSW `pol3` coefficients `c0,c1,c2,c3`, using the nominal integral
as a fixed gauge.  Optimizing four unconstrained coefficients would be wrong:
their common scale cancels in angular sampling and would create an exact zero
mode in the Hessian and Fisher matrix.

## Gradient safety

The calibration forward follows the production reflection-model selection and
uses the configured 532 nm wavelength.  Engineering-interface indices are
mapped per photon, and the scan state is explicitly returned in float32 even
when `JAX_ENABLE_X64=true`.

Adam raises `FloatingPointError` on a non-finite objective, gradient or
validation objective.  It must never replace NaN or infinity with zero.

Optimizer histories include the initial state at step zero.  Each row of
`coordinate_history`, `history`, `loss_history` and `grad_norm_history`
describes the same evaluated state.  The fixed-key `train_objective` supplies
the optimization gradient.  `mc_validation_objective` uses the same data with
independent transport keys and therefore tests MC-key overfitting, not data
generalization.  `heldout_validation_objective` may additionally use held-out
source positions.  Early-stopping patience counts
validation evaluations and follows held-out validation when present,
otherwise MC validation.  Stage handoff priority is held-out best, MC best,
then train best.  The final iterate remains diagnostic only.

With `validation_every = 10`, the default patience of 5 means five stale
validation checks, corresponding to roughly 50 Adam updates.  Step zero is
evaluated and can remain the selected checkpoint.

## Synthetic-closure credibility audits

Three separate checks prevent a plumbing smoke test from being mistaken for
a statistically credible closure.

First, `objective_ad_fd_comparison` compares JAX automatic differentiation
with centered finite differences for all five identifiable coordinates.  The
plus and minus evaluations reuse exactly the same transport keys.  Otherwise
the finite difference would be dominated by Monte Carlo variation rather
than the local objective derivative.  A finite-difference step scan is still
required before accepting a physics result: hard PMT assignment and path
topology can make a low-photon Monte Carlo objective non-smooth even when AD
returns a finite pathwise gradient.

Second, the diffuser source supports two importance estimators.  With a
uniform proposal in `cos(theta)`, the analytic estimator uses the known
normalized target density directly:

```text
w_i = intensity / N * f(mu_i) / q(mu_i).
```

The historical self-normalized estimator divides these weights by their
sample mean.  It fixes every finite batch's total light exactly, but introduces
an `O(1/N)` ratio-estimator bias and couples all photons.  The paired audit
reuses the same proposal rays for both modes and scans photon count and seed.
The configuration intentionally remains `self_normalized` until a full
detector A/B closure, not just a source-level test, justifies changing the
baseline.

Third, same-key and independent-key closures answer different questions.
Same keys are a common-random-number computational-graph test.  Independent
keys expose finite-proposal coverage and variance.  In a detector with about
17,600 PMTs, a few hundred independent proposal photons cannot provide
strictly positive signal support at every PMT containing a synthetic PE.  A
pure-signal independent-key test must therefore fail fast rather than apply a
rate floor.  The runnable independent-key diagnostic uses an explicit uniform
PMT background, sampled in the synthetic data and included in both PPP terms:

```text
lambda_ep(t) = exposure_e * lambda_signal,p(t) + r_background,p
```

This background makes the two probability models share support; it does not
claim that 512 photons are a production transport budget.  The final study
must still increase photons and ensemble size and demonstrate stability as
the explicit background is reduced toward its measured value.

Run the source-level importance audit with:

```bash
make -C synthetic_closure importance-audit \
  IMPORTANCE_OUT=/tmp/lucid_diffuser_importance.json
```

Run the paired key diagnostic with:

```bash
make -C synthetic_closure key-comparison \
  KEY_OUTDIR=/tmp/lucid_key_comparison \
  N_PHOTONS=512 K=4 STEPS=1 START_FRACTION=0
```

The paired JSON records scaled parameter differences, loss values, key
offsets and the common explicit background rate.  These low-budget commands
are diagnostic examples, not the final closure acceptance configuration.

### Photon, ensemble and transport-seed scan

The scan keeps the synthetic-data seed, data photon budget and data ensemble
fixed.  It varies the fit-side photon count, fit ensemble size and independent
transport-key offset.  This separates fit-forward Monte Carlo stability from
a simultaneous change of the sampled PE data:

```bash
make -C synthetic_closure credibility-scan \
  SCAN_OUTDIR=/tmp/lucid_closure_scan \
  SCAN_DATA_N_PHOTONS=4096 \
  PHOTON_SCAN="128 512" \
  ENSEMBLE_SCAN="1 2" \
  FIT_SEED_OFFSET_SCAN="1000003 2000003 3000003"
```

Every cell writes a normal closure directory.  `summary.json` reports the
initial five-coordinate gradient mean, standard deviation, relative standard
deviation and pairwise direction cosine over transport seeds.  The default
grid is intentionally a plumbing-scale example.  Physics
acceptance requires increasing the budgets until both gradient direction and
parameter recovery are stable.

### Per-source response Jacobian and Fisher information

For source `k`, the diagnostic differentiates the IRF-convolved numerical
response masses with respect to all five optimizer coordinates:

```text
J[k,p,b,a] = d m[k,p,b] / d theta[a].
```

For event exposure `E_e` and explicit background mass `beta[p,b]`, it builds
the expected numerical-basis Poisson Fisher matrix

```text
F[k,a,c] = sum_e,p,b
  (E_e J[k,p,b,a]) (E_e J[k,p,b,c])
  / (E_e m[k,p,b] + beta[p,b]).
```

No denominator floor is used.  A zero expected cell carrying a nonzero
derivative is a non-regular-support error.  The JSON stores each source's
full `5 x 5` Fisher matrix and a compact Jacobian summary: L2 norm, maximum
absolute derivative, mean absolute derivative and nonzero fraction.  It also
checks exactly that

```text
F_total = sum_k F_k.
```

The raw `PMT x time x 5` Jacobian is deliberately not written by default;
at the production 0.5 ns grid it can occupy hundreds of MB per source.  Run:

```bash
make -C synthetic_closure fisher \
  SOURCES="0,0,0 0,0,16.1 0,0,17.1" \
  FISHER_OUT=/tmp/lucid_per_source_fisher.json \
  N_PHOTONS=512 FIT_ENSEMBLE=1
```

This first Fisher is a numerical-time-basis identifiability diagnostic, not
yet the exact continuous-time PPP Fisher.  Its eigenvalues, condition number,
correlations and CRB must be checked for photon, ensemble, seed and time-grid
convergence before receiving a physics interpretation.

### Held-out source validation

`--heldout-source-index` now removes the selected source from every training
objective.  Its event-wise PPP loss is evaluated only as held-out validation.
Every optimizer stage selects its handoff checkpoint in this order:

```text
held-out-source best -> MC-validation best -> train best.
```

Absolute source indices are retained when folding JAX keys, so a source uses
the same random-key identity before and after the train/held-out split.  A
minimal full-geometry interface check is:

```bash
make -C synthetic_closure heldout-smoke
```

This is source-position validation, not event or PMT-region validation.  A
formal closure must use more than one training source and must not infer
generalization from a one-step smoke test.

Run a reproducible seed-variance audit with, for example:

```bash
source run/env.sh
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true

"$LUCID_PY" scripts/check_juno_calibration_gradients.py \
  --source \
  0,0,17.1:/path/to/z1710/simulation.parquet \
  --source-exposure 1 \
  --source-time-offset-ns 0 \
  --output /tmp/juno_full_gradient_audit.json \
  --n-photons 512 \
  --k 64 \
  --ensemble-size 1 \
  --seeds 42 43 44
```

The JSON records the input paths, random seeds, raw gradients, timing, JAX
backend, x64 state, Git commit and dirty flag.  Increase photons and ensemble
size only after checking the gradient mean-to-standard-deviation ratio.

The closure CLI accepts arbitrary repeated `--source-position` options.  It
does not define a built-in source manifest.  Synthetic truth and fitting use
separate transport budgets, and a third independent key ensemble evaluates
MC-key validation.  For a direct computational-graph smoke test use:

```bash
source run/env.sh
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true

"$LUCID_PY" scripts/run_juno_full_geometry_closure.py \
  --source-position 0,0,17.1 \
  --outdir /tmp/lucid_eventwise_closure \
  --n-events 1 \
  --data-n-photons 128 --fit-n-photons 128 \
  --data-ensemble-size 1 --fit-ensemble-size 1 \
  --mc-validation-ensemble-size 1 \
  --fit-seed-offset 1000003 \
  --mc-validation-seed-offset 2000003 \
  --k 4 --intensity 100000 --background-rate-hz 30000 \
  --time-min-ns 0 --time-max-ns 300 --time-bin-ns 50 \
  --optimization-plan joint-only --joint-steps 1 \
  --source-chunk-size 1 --checkpoint-every 1 \
  --start-fraction 0.1
```

This command is a smoke test, not a parameter-resolution claim.  It writes
`closure_observations.h5`, `closure_result.json`, `closure.log` and periodic
`checkpoint_STAGE.npz` recovery snapshots.  A checkpoint restart restores
the five coordinates but intentionally starts fresh Adam moments; it is not
a bitwise continuation.  Zero
predicted mass, zero sampled PEs, non-finite derivatives and any zero initial
parameter gradient are hard failures.  At 128 proposal photons a perturbed
independent-key fit can have insufficient signal support, hence the explicit
known background in this plumbing example.  A pure-signal recovery test must
instead raise the photon/ensemble budget until support is stable.

`--optimization-plan joint-only` activates all five identifiable coordinates
in one stage.  The historical `staged` plan remains an explicit option, not a
property of the PPP likelihood.  `--source-chunk-size` partitions only the
forward/objective graph and preserves absolute source indices and the global
PE normalization.  It does not choose source positions.

For a GPU qualification job, set a new output path and any one or more source
positions explicitly:

```bash
make -C synthetic_closure submit-one \
  OUTDIR=/junofs/C/liukainan/LUCiD/calibration/qual_1source
```

The Make target follows the collaboration convention used by
`compare_lucid_sim/Makefile`: it submits through `sudo -u#35905 sbatch`, uses
`sbatch --wrap` to preserve the frozen configuration, and does not use
`srun`.  `submit-five` uses the documented five-source set.  Both targets
refuse to overwrite an existing `closure_result.json`.

The closure worker opts out of XLA command-buffer replay through
`LUCID_DISABLE_XLA_COMMAND_BUFFER=1`.  This is required because the current
compute-node Python 3.13 + ROCm 7 stack can report malformed AQL packets when
the data, fit and MC-validation forward/VJP graphs execute in one process.
The worker pins `/opt/gentoo/usr/bin/python3.13` and records Python, JAX,
JAXLIB, device and XLA flag metadata before starting the closure.
The qualification workflow stacks all sources into one pytree and evaluates
them with a sequential `jax.lax.map`, rather than compiling one static closure
per source.  Source NLLs are summed before the single global PE normalization,
so this execution policy does not change the joint PPP objective.  After the
synthetic-data predictions are materialized on the host, their compiled cache
is released before the differently shaped fit forward is built.

## Numerical time-grid convergence

The observation remains unbinned: exact PE timestamps enter every event term.
The internal PMT-by-time array is only a numerical basis for the continuous
intensity.  A production result must therefore be stable when that basis is
refined.  The standard scan is

```text
Delta t = 2 ns, 1 ns, 0.5 ns.
```

Run the opt-in three-grid workflow with identical source, random seeds,
photon budget, event count, transport order and optimizer settings:

```bash
make -C synthetic_closure grid-convergence \
  GRID_OUTDIR=/tmp/lucid_time_grid_convergence
```

Each grid writes a normal closure directory.  The final
`time_grid_convergence.json` compares the fitted parameters, final loss per PE
and final gradient with the finest grid, and judges convergence using the two
finest grids.  The summarizer rejects inputs whose non-grid metadata differ.
The default tolerances are diagnostic starting points, not final scientific
requirements; production tolerances must be frozen before examining data.

## Fixed PMT-type numerical widths

The campaign IRF file can provide a first PMT-type sensitivity test without
claiming a final detector-response model.  Reproduce the summary with:

```bash
source run/env.sh
"$LUCID_PY" scripts/summarize_juno_irf_widths.py \
  --output /tmp/juno_irf_type_widths.json
```

The file contains 170 rows and 146 unique serial numbers.  Selecting the
highest-load row for each serial and joining to the J25.7.3 CD PMT map gives
110 Hamamatsu and 15 NNVT characterized PMTs.  Converting the median FWHM with
`sigma = FWHM / 2.354820045` gives:

```text
Hamamatsu: median FWHM = 2.45 ns  -> sigma = 1.040419 ns
NNVT:      median FWHM = 24.60 ns -> sigma = 10.446658 ns
```

These values are population-level numerical kernels.  The NNVT estimate is
based on only 15 tubes, and a Gaussian cannot reproduce the full measured IRF
shape.  Run the PMT-type closure and evaluate a common 1 ns kernel on the same
synthetic PE observations with:

```bash
make -C synthetic_closure smoke \
  OUTDIR=/tmp/lucid_pmt_type_sigma_closure \
  SIGMA_MODE=pmt_type COMPARE_COMMON_SIGMA_NS=1
```

## Scope boundary

This is a full-geometry gradient and synthetic unbinned closure-test
interface, not yet the final production likelihood described in `proposal.md`
and `diffsim.tex`.  The fixed Gaussian kernel only establishes that the
continuous-time objective and its gradients execute normally.  Real-data
calibration already has an event-wise loader and explicit fixed exposure
input, but still requires Monitor-PMT timing, a validated absolute exposure
normalization (or constrained nuisance), measured laser/TTS/electronics
response, channel offsets and background estimates.  The numerical kernel
must eventually be replaced by those validated physical response components.
