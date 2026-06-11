# MAIN_BRANCH_PLAN.md — the simplified, general LUCiD foundation (v3)

**Purpose.** This is the consolidation plan refined for *simplicity + generality* — the shape
LUCiD's **main branch** should take. It supersedes the heavier abstraction in
`RECON_CONSOLIDATION.md` (v2 + §14); v2 remains the detailed **port mechanics** (line-anchored
invariants, the order-stat time term, de-env), this is the **architecture + scope** layer.
Grounded in: a 3-agent verification of v2 (§14), a survey of all `good_notebooks/`, and a
fitting-interface simplification pass.

---

## 0. The reframe (verified)
`merge-base(reconstruction-mie, unify-calibration) = recon HEAD`; unify's first commit
`41096e3` IS the recon-forward patch. Substrate byte-identical (SIREN net, geometry, configs
same md5); `simulation/types.py`, `losses.py`, `wavelength/scattering.py` byte-identical
between worktrees. **The forward is already consolidated. `LUCiD_unify` is the main base** (only
it has `reflection.py`, `optical_model.py`, `fitting/`). The work is **simplify + migrate
drivers + 3 decisions + delete the sprawl** — not merge transport physics.

---

## 1. Fitting interface — the leanest general form (REPLACES v2 §1-2 dataclasses)
The user rejected role-tables, marker pytrees, Const/Curve, and an over-wrapped builder. v2's
`Problem`/`OptConfig`/`LossConfig`/`Nuisance` dataclasses are the same smell under new names.
**Keep the two collapses that remove real duplicated CODE; drop the wrappers around ARGUMENTS.**

**ONE fit, no config objects:**
```python
def fit(residual_and_jac, theta0, *, scale=None, nuisance=(),
        steps=300, refresh=15, ridge=0.02, mu=0.3, eigen_clip=True,
        lr=1.0, clip=0.08, readout='last', polyak=0, bake_k=False, seed=0):
    # consistent fixed-dataset GN: recompute g+resid every step; refresh J/H/Pinv every `refresh`;
    # assemble_normal -> (optional precond) -> ridge_inverse -> step -> nuisance back-sub -> readout
```
- `residual_and_jac(theta, keys) -> [(r, J, W), ...]` — the ONE per-problem closure. The
  residual *form* (√-MSE vs Poisson) and `W` (lit-mask | 1/μ | I) are PRIVATE to it; `fit` only
  sees `(r,J,W)` blocks. This is the irreducible 10%.
- `assemble_normal(blocks, nuisance)` — **KEEP** (v2 §1.3). Collapses the 4 duplicated metric
  assemblers (`gauss_newton.py` fit + fit_charge_time, `fisher.crb`, recon) into
  `H=Σ(J·W)ᵀJ`; the per-PMT Schur is a gated no-op when `nuisance=()`. Real dedup.
- `nuisance` = a list of **two tiny closures**, NOT a 5-field record: `Nuisance_k` (Jk=0.5·m,
  multiplicative gauge), `Nuisance_t0` (Jk=1, additive gauge). `bake_k` stays a **kwarg pre-pass**
  (it replaces the Schur block with a closed form — structurally not a back-sub).
- the recipe = **two module-level kwarg dicts**, not classmethods:
  `CALIB_GN = dict(scale=None, eigen_clip=True, lr=1., clip=.08, ridge=.02, mu=.3, readout='last')`
  `RECON_GN = dict(scale=particle_scale(), nuisance=(), eigen_clip=False, lr=8., clip=0, readout='ming', refresh=8)`
  → `fit(resid, dp0, nuisance=[Nuisance_k()], **CALIB_GN)` / `fit(resid, th9, **RECON_GN)`.

**Collapses:** `fit` + `fit_charge_time` → one loop (`nuisance ∈ {(),(k,),(k,t0)}`); 4 metric
assemblers → `assemble_normal`; 3 FD-Jacobian loops → one `fd_jacobian(fn,θ,h,keys,central=)`
(pick central); 3 readouts → one `readout(history,gnorms,mode)`. **Stay separate:** `crb()`
(not a loop — evaluates at truth, ×4Nphot×√12; reuses `assemble_normal`), `calibrate_timing()`
(closed-form moment estimator), and the residual *form*.
**Concept count: 1 fit + 1 assembler + 2 nuisance closures + 2 dicts** (vs v2's 4 dataclasses).
The one flag that can't default-share: `eigen_clip` (True calib indefinite-FD / False recon PSD).

---

## 2. Param system — keep nested, add two things (REFINES v2 §2.2, §9)
- **Keep nested `DetectorParams` (5 sub-tuples).** Nesting costs ZERO concepts: `make_optimization_mask`
  and `normalize_params`/`default_bounds` are generic tree-walks. ⚠️ ~16/24 leaf fields are inert
  in the default path (angular-refl, dev curves, spe_width/tts/gain/t0/walk) — **document a
  live-field matrix** (which fields are active per mode/reflection_model/wavelength_mode/hit_mode).
- **Keep the flat-JSON shim** (`from_flat`/`_nest_flat_kwargs`) as PRIVATE wire format; JSON authors
  shouldn't know the physics grouping. Don't export `_FLAT_FIELDS`.
- **ADD `particle_bounds()`** mirroring `default_bounds` → derive `particle_scale()=(hi−lo)` via
  the existing normalize machinery. Retires the TWO inconsistent magic SCALE9 vectors
  (`gn_fisher_recon.py:40` 9-vec vs `recon_harness.py:22` 7-vec).
- **ADD trivial `JointParams(detector, particle)` pytree now** (works with the generic mask/normalize
  for free); **defer the joint FIT** (blocked by the cross-block metric + √-MSE↔Poisson residual
  conflict — physics decision, not this scope).

---

## 3. The 3 decisions (code, not flags)
1. **flat→nested `DetectorParams`** (headline): ripples into recon's `simulator.py`/`photon_step.py`
   + ~210 recon driver scripts. unify ships `from_flat` as the bridge; migration is mechanical/broad.
2. **`photon_step` signature**: adopt unify's `refl_params`+`lam`+`reflection_fn=` FACTORY form
   (`make_photon_iteration_update_factors_safe`) over recon's scalar-rate module-level `@custom_vjp`.
   Output 7-tuple identical (no DiCE reconciliation).
3. **SIREN emitter** ⚠️ the riskiest: recon default = DiCE-SCORE live-resample; unify default =
   fixed-proposal IMPORTANCE (lower-variance, pathwise-exact). **Main default = importance**; keep
   **score as an explicit `emitter='score'` factory** (mirror `reflection_model=`), NOT an env var.
   **Gate the flip behind re-validating the §8 recon floor** (recon's basin/Hessian/13cm-floor record
   was measured on the score path). Delete `SIREN_SMOOTH/MARGINAL/CONTINUOUS` experiments.

---

## 4. Notebook-driven simplifications (the real usage; from the good_notebooks survey)
The notebooks are the API spec. They reveal the boilerplate to KILL and one hard rename:
- ⚠️ **`max_sensors_per_cell` (unify) vs `max_candidates_per_ray` (recon)** — a half-finished rename;
  recon notebooks pass the new name but it falls unused into `**grid_params`. **Pick ONE name**, make
  it an explicit arg (or alias both). 7 recon notebooks affected.
- **`make_sim_pair(geom, physics, ...)`** — ~12 notebooks copy-paste paired prediction(temp .1,K7)/
  data(temp 0,K20) simulators. One helper.
- **`run_calibration(...)` entry point** — calibration notebooks hand-roll the
  `optax.multi_transform({'train':adam,'freeze':set_to_zero()})`+normalize/mask dance; mirror the
  recon `run_*` so calibration has a one-call fit too.
- **ONE `likelihood_loss(...)` in `lucid.losses`** — every likelihood notebook redefines the same
  3-term (poisson_nll × first_arrival_nll × geomean, TAU=0.15) recipe + a local `poisson_nll`
  (which already exists as an alias). Promote it.
- **distinct `track_simulator`/`calibration_simulator`** (or a clear `mode=`) — the simulator's two
  call signatures (`sim(track,key)` vs `sim(source,params,key)`) force out-of-band mode knowledge.
- recon-only notebooks import non-existent scratch modules → **exclude from the curated set**.
**Public API the notebooks depend on (keep stable or replace cleanly):** `setup_event_simulator`,
`ParticleParams(.from_cartesian)`, `DetectorParams(from_flat)`, `isotropic_source`/`laser_source`,
`load/save_detector_params`, `normalize/denormalize/default_bounds/make_optimization_mask`,
`WC_loss`/`WC_smooth_loss` + the recon losses, `lucid.optimization.{run,grid_search,utils}`,
`lucid.gradient_analysis` (`SweepParam`/`sweep_1d/2d`), `lucid.visualization`.

---

## 5. Deletion / archive list (clean main; snapshot first)
⚠️ **Pin a recon commit + tarball the untracked drivers BEFORE deleting** (the exact numerical
reproductions live only in these scripts; recon worktree is mid-Mie-dev with dirty optical files).
- **`LUCiD_recon/*.py` (210 untracked)** → ~3 become CODE (`gn_fisher_recon`+`recon_harness`+
  `recon_ps_setup` ported into `lucid/fitting/` + `losses.py` + a runner shim); the rest
  (51 `loss_*`, 13 `recon_*`, `align_*`/`endtoend_*`/`hess_*`/`crb_*`/`plot_*`/…) → **archive** to a
  `recon-scratch` branch/tarball, delete from tree. (Findings already in memory/`.md`.)
- **`lucid/optimization/pipeline.py` (~600 lines Adam)** → **DELETE** the 5-stage Adam +
  pre-finalization loss; **SALVAGE** the cone-direction search (the one seed Fisher-GN still needs) +
  `generate_event_data` (PhotonSim ROOT load) into the recon runner; `run.py` → thin shim preserving
  `lucid-optimize`.
- **env reduction 64→~25**: DELETE (don't field-ify) the diagnostic modes (ESWEEP/ZESWEEP/LOSSCHK/
  GRADCHK/CMP/HIST/…), rejected experiments (EDETACH/TROBUST/OCC_GATE/T0_PRIOR/non-poisson CLOSS),
  PSTEP_DBG×7, T0_LIVE, NORM_GRAD_ITERS, DATA_CHG/LIK_CHG_NOTIME, OVERLAP_ANALYTIC=erf/logistic,
  ADAPT/LAM*, STEPCLIP, OVERLAP_RENORM, + the ~26 dead constants.
- **`mie_hunter/` (268 .py/63 .md)** → **archive wholesale**; the salvage (implicit step,
  charge-moments) is already in-tree (`photon_step.py`, `sensor_response.py`).
- **`campaign/` (12 .py + result .md)** → keep as `scripts/` (NOT `lucid/`); fold `run_grid.py`→a
  shared `run_pool()` launcher + JSON sidecar (retires 4 drifted stdout regexes); archive `*_RESULTS.md`.
- **`lucid/generate.py`** → verify vestigial (moved to `sources/`), delete.

---

## 6. The one forward bug to fix in the clean tree (PREREQUISITE)
`sensor_response.py:11,214` — a live import-time `os.environ` TTS leak: `_TTS_PERPHOTON_NS=
float(os.environ.get('TTS_NS','0'))` then `eff_tts=max(tts,_TTS_PERPHOTON_NS)`. Floors every
forward's TTS from the env; blocks two-`Problem` coexistence in one interpreter (every test needs
it). Delete the global + `max(tts,env)`; route TTS through `dp.response.tts`/`SimConfig`. Pure
refactor (defaults preserve values). **This is v2 §5/§6-step-1 and the first thing to do.**

---

## 7. Public surface (small, general)
`lucid/__init__` (lazy) exposes only: `setup_event_simulator`, `generate_detector`, `DetectorParams`,
`ParticleParams`, `JointParams`, `isotropic_source`, `laser_source`, `load_physics_config`. Fitting
optimizers, losses, hit-mode internals stay behind `from lucid.fitting import …` / `lucid.losses`.

---

## 8. Sequenced plan (each gated by the test net, v2 §11)
1. **De-env** (`sensor_response.py` TTS + recon import-env) — §6. Pure refactor, unblocks coexistence.
2. **Notebook simplifiers, low-risk first:** the `max_sensors_per_cell` rename; `make_sim_pair`;
   one `likelihood_loss`; `counts_loss(normalize=)` flag (it does NOT exist — must be ADDED, §14.3).
3. **`particle_bounds()` + `JointParams`** (additive, near-zero risk).
4. **`fit` refactor** (§1): `assemble_normal` + one loop + shared `fd_jacobian`/`readout` +
   `ridge_inverse(eigen_clip=)`; extract calib/recon residual closures. **Gate behind byte-pins** (v2 §11).
5. **Port the order-stat time term** (`first_arrival_order_stat_nll`, v2 §3) + the SIREN `emitter`
   factory + the recon runner shim; **re-validate the §8 floor under importance** before retiring score.
6. **`pipeline.py` deletion + salvage** (§5) + harness/JSON-sidecar merge.
7. **Archive the sprawl** (§5, after snapshot).

**Test net (v2 §11, REQUIRED before step 4):** byte-pin `ridge_inverse` on a fixed indefinite H;
pin toy-fit `log_theta`/`k` to 1e-6 (not 0.05); one e2e calib fit on WCTE fixture; order-stat `tnll`
value-pin (capture reference from recon FIRST); amp_detach gradient-routing test; `counts_loss(normalize=False)`
pin; an `@slow` recon floor smoke (bias-limited vtx~13cm/dir~1.8°/E~2.3%, NOT matched-SIREN best).

---

## 9. Extensibility — what it takes to add physics / change the loss / etc.
(Audited against the code. Classes: **LOCAL** = one place via a seam · **CHECKLIST** = 2-3 known spots · **HARD** = no seam, ripples through the `custom_vjp`.)

| Extension | Class | What you touch |
|---|---|---|
| **Change/add a LOSS term** (new charge/time residual) | **LOCAL** | edit the `residual_and_jac` closure → add/replace one `(r,J,W)` block. `fit`/`assemble_normal` never see the form. (Notebook API: add to `lucid.losses` + the one `likelihood_loss`.) **The design's best seam.** |
| **Add a per-PMT NUISANCE** (e.g. per-PMT gain, 3rd Schur block) | **LOCAL** (under v3) | append a nuisance closure (`Jk` + gauge) to the `nuisance` list; `assemble_normal` loops it. (Today it's a 2-block hardcode in `fit_charge_time` — v3 turns it into the loop.) |
| **Add a SOURCE / GEOMETRY** | **LOCAL** | source = callable-NamedTuple + factory (duck-typed by `sim`); geometry = `@register_detector`. (Wart: geometry *construction* still has an `if cls is Cylinder/...` chain — a 2nd spot.) |
| **New reflection model** | **LOCAL** | `reflection.py`: new fn + params-NamedTuple + 1 registry line + `build_refl_params` line. The reference seam. |
| **Add a fitted PARAMETER** (`DetectorParams`/`ParticleParams` leaf) | **CHECKLIST** | auto-flows through `normalize`/`make_optimization_mask`/`from_flat` (generic tree-walks); but hand-edit ~5 explicit per-field dicts (`default_bounds`, `default_gradient_scales`, `create_default_*`, `from_flat` defaults, JSON-loader `_NEUTRAL_DEFAULTS`) **+ the forward consumer** — each fails loudly (`_nest_flat_kwargs` requires every field; a missed loader default NaN-poisons the forward). |
| **Add an OBSERVABLE** (waveform / new time stat) | **CHECKLIST** | `make_hits` IS a factory (new `make_hits_*` + builder + 1 dict entry + `_VALID_HIT_MODES`); but the per-photon-vs-aggregated output shape has no contract — if the observable needs more than `(flat_weights,indices,times)` + the `response` bundle, you touch `_common_propagation`. Timing semantics live *inside* each `make_hits` (no separable "time model"). |
| **Add NEW TRANSPORT PHYSICS** (new scatter type / WLS / 2nd absorption) | **HARD** | no seam. Reflection is seamed but scatter/absorption are hard-coded inline in TWO step twins (`photon_iteration_update_factors` + `_sample`), behind a fixed-arg `custom_vjp` `_core` whose arglist + the hand-written `vmap in_axes` literal + `OpticalArrays` + the 5-tuple threading all reshape together. Only the DiCE `log_p` folding is already general. |

### The 3 seams to ADD (so the hard/checklist cases become local) — adopt into the plan:
1. **An `OpticalProcess`/`ScatterModel` registry mirroring `reflection.py`** — the highest-value addition and the real answer to "add new physics." Generalize reflection's pattern (single **packed-pytree** params arg + a `process_fn(...)->(factors, logp_increment)` closed statically into the step factory + the DiCE-score channel) to the medium step, so Mie/WLS/new-absorption is a one-file change instead of editing the `custom_vjp` core + both step twins + the `vmap in_axes` + `OpticalArrays`. This is the one place the forward is NOT yet "one engine with plugins."
2. **A single per-field metadata table for `DetectorParams`** (`name → min,max,scale,default,live-in-mode`) that the ~5 generator dicts read — collapses "edit 5-7 places or NaN-poison" into one table edit, and doubles as the live-field matrix (§2). Add `particle_bounds()`/`particle_scale()` so `ParticleParams` matches.
3. **Written CONTRACTS for the closure surfaces** — because v3 trades typed dataclasses for closures+kwargs, the contract of `residual_and_jac` (shapes, `W` convention), the nuisance closure (`Jk`+gauge), and the `make_hits` output shapes live only in docstrings. Write them down (a `fitting/CONTRACTS.md` or rich docstrings). This is the mitigation for the one place simplification *hurts*: extensibility stays high, **discoverability** drops without types.

### Verdict
The closures+kwargs simplification **helps** extensibility for the whole fitting/loss layer (loss term, nuisance, observable-block are all LOCAL closures; `assemble_normal` dedups real code) and is **orthogonal** to the one HARD case (transport physics is a `custom_vjp` problem, not a fitting one). It mildly **hurts discoverability** (no types → read the source), mitigated by seam #3. So: keep the simplified fitting design; add seam #1 (process registry) to make "new physics" local; add seams #2-3 (metadata table + written contracts) to de-risk the checklists. With these, every common extension is one local edit.

---

## 10. The general extension architecture (think bigger than per-case seams)

Don't enumerate seams ad hoc — derive them. LUCiD is **two halves joined by one contract**:

- **MODEL** (forward): `source → transport[optical processes] → surfaces[reflection] → sensors[response]`, over a `geometry`, under a `spectrum`/`medium`, producing **observables**. A pipeline of differentiable stages.
- **INFERENCE** (fitting): consumes a differentiable `loss(params)` (or `residual_and_jac → (r,J,W) blocks`) and produces **estimates + uncertainties**.
- **THE CONTRACT joining them:** `params` (a pytree: `DetectorParams ⊕ ParticleParams`) + a differentiable `forward(params) → observables` + `observation` ⇒ a differentiable scalar. **The model never knows the optimizer; the optimizer never knows the physics.** Every inference backend (GN today; Adam/L-BFGS/CRB/MCMC tomorrow) is just a consumer of "a differentiable scalar of a params pytree." This decoupling IS the generality — keep it sacred.

### 10.1 The ONE universal seam pattern (apply it everywhere on the MODEL side)
Reflection already discovered it; name it and reuse it. A **Component** =
`(params_pytree, fn(state, params, lam, key) -> (effect, logp_increment), gradient_channel, registry_key)`:
1. fittable params in a **single packed pytree** (so the `custom_vjp` core arglist never grows),
2. a differentiable `fn` with a fixed signature, selected by a **registry/factory**,
3. a declared **gradient channel** — pathwise (reparam) or DiCE-`logp_increment` (score), folded into `log_p`,
4. params that **auto-flow into `DetectorParams`** (via the §9-seam-2 metadata table) and thus into the fitter for free.
`reflection.py` is the reference instance. Making **optical-process, sensor-response, and source/emitter** all instances of this one pattern is what turns "add physics" from HARD into LOCAL — *and* makes the codebase teachable (one pattern, many plugins) instead of N bespoke factories.

### 10.2 The full extension-axis map (each axis = a seam; status now → what to add)
| Axis | Seam now | Add for main |
|---|---|---|
| geometry | `@register_detector` ✓ | fold the residual `if cls is …` construction chain into the registry |
| source / **emitter** | duck-typed callable ✓ | `emitter=` factory (importance default, score opt-in) under the Component pattern |
| **optical process** (scatter/abs/WLS) | **HARD** | `OpticalProcess` registry = the Component pattern on the medium step (§9 seam-1) |
| surface / reflection | factory ✓ (**the reference**) | — |
| sensor / **observable** | `hit_mode` factory ✓ | a written output-shape contract; timing as a declared `time_model`, not buried in each `make_hits` |
| spectrum / **medium** | `Spectrum` obj ✓ | a `medium` registry (water/ice/scint) mirroring reflection; `config/materials` is the data side |
| **parameters** | nested pytree + generic tree-walks ✓ | per-field metadata table (§9 seam-2); `particle_bounds()`; a **reparameterization seam** (`unpack` owns log/sin-cos) |
| loss / residual | closure ✓ (**the best seam**) | written contract (§9 seam-3) |
| **prior / regularization** | **MISSING** | a first-class additive term — see 10.3 |
| nuisance (marginalized block) | closure list (proposed) ✓ | `Jk`+gauge contract |
| **inference backend** | GN only | make the `loss(params)`/`residual_and_jac` contract explicit so Adam/L-BFGS/CRB(/MCMC) are interchangeable consumers — no model change |
| **gradient estimator** (per stochastic decision) | DiCE score vs pathwise, ad hoc | declare it per-decision via the Component's `gradient_channel` — see 10.3 |
| **batch / event axis** | implicit | guarantee `forward`+`loss` are `vmap`-able over events (recon = many events; calib = pooled); state it as an invariant |
| data / IO | `is_data` path + flat-JSON + save/load ✓ | keep; one loader contract |
| study / orchestration | sweeps + `run_pool` (proposed) ✓ | JSON sidecar; one `summarize()` |

### 10.3 The genuinely NEW seams worth adding (not just "make existing things pluggable")
- **Priors / regularization — the biggest real gap, and it lives in the INFERENCE layer, NOT the model.** A prior is a function of params ALONE (no forward), so it is purely an objective/inference concern — do NOT weave it into the model's `residual_and_jac` (that closure must stay forward-only). Specific seam: a `priors=()` argument on the fitter, each prior a **param-only** term (most generally `log_prior(params) -> scalar`; in practice a `gaussian_prior(field, μ, σ)` helper). Each backend consumes it its own way — GN linearizes a Gaussian to a trivial forward-free `(r,J,W)` block (`r=(θ−θ₀)/σ`, `J=I`), Adam autodiffs it, CRB adds it to the Fisher — so the prior spec is backend-agnostic. Payoff: (a) calibration regularization + Bayesian-MAP become first-class, (b) **the stiff-param ridge IS a Gaussian prior** — express it as one and the "ridge hack" and "prior" unify, (c) the model stays pure.
- **Inference-backend agnosticism — formalize, don't build.** The user already saw GN-vs-optax. The lesson: the model exposes `forward(params)→obs`; GN, Adam, CRB, and a future sampler are all consumers of "a differentiable scalar of params." Write the contract so swapping the backend is a consumer choice, not a model rewrite. This is why priors (and the residual *form*) are inference-side, never baked into the model.
- **Per-decision gradient estimator — design-for, defer.** The deepest generality of a differentiable MC: each random choice (emission bin, scatter type/angle, reflection branch) declares pathwise/score/reparam. LUCiD has instances (emitter importance-vs-score; the `lf/la/lr` DiCE scores). A general `gradient_channel` on the Component (10.1) is the hook; a full estimator-registry is the DTRAX frontier — design the hook now, defer the registry.
- **Multi-physics / cross-detector — the boundary, out of scope.** Scintillator/TPC/charged-particle transport (the DTRAX line) is a different forward. State it as the explicit generality boundary so main stays focused on optical photons; the Component + contract design is what would *let* a sibling forward reuse the inference half later.

### 10.4 Build-now vs design-for-but-defer
- **Build into main now (cheap, foundational):** the Component pattern (unifies optical-process/reflection/sensor/source); the explicit model↔inference contract; **priors as `(r,J,W)` closures** (fills the real gap, zero new machinery); the param metadata table + `particle_bounds` + the `unpack` reparameterization seam; the batch-axis invariant.
- **Design-for, defer:** the per-decision estimator registry; MCMC/posterior UQ; cross-detector multi-physics. Each has a *hook* in the above (gradient_channel; the loss-contract; the Component+contract decoupling) so they're additive later, never a rewrite.

**The general principle:** one Component pattern on the model side, one `(r,J,W)`/`loss(params)` contract on the inference side, and a metadata table binding params to both. Every axis above is then either a registry entry, a closure, or a pytree field — and "add physics / change the loss / add a prior / swap the optimizer" are all *the same kind of one-place edit*.

### 10.5 The crisp model/inference split (the specific recommendation)
Putting priors in inference forces the cleanest cut, and it is sharper than v3's hand-written `residual_and_jac`. Make the **model a single pure function** and let the **fitter COMPOSE the objective** from small inference-side pieces:

- **MODEL (pure, the only thing you write per forward):** `forward(params, key) -> Observables` where `Observables` is a NamedTuple (`mean_charge, var_charge, time, ...`). Knows nothing about loss/prior/optimizer. Built from the Component plugins (§10.1).
- **INFERENCE pieces (all small, composable, passed to the fitter — none in the model):**
  - `residual` — the loss FORM, a pluggable choice: `residual_fn(pred_Obs, obs) -> (r, W)` (registry: `sqrt_mse | poisson | anscombe | order_stat_time`). This is "change the loss" = a one-arg swap, NOT rewriting a closure. The fitter chains it with the forward-FD to make `J = (∂r/∂Obs)·(∂Obs/∂θ)` — and the expensive `∂Obs/∂θ` is shared across residual forms (charge √-MSE and charge-variance reuse one forward-FD).
  - `priors` — param-only terms (10.3); forward-free.
  - `nuisance` — marginalized per-PMT blocks (closures: k, t0, gain).
  - `reparam` — the `unpack`/scale map (log for positives, sin/cos for angles); defaults to identity.
- **THE FITTER** composes `objective = Σ residual(forward(θ), obs) + Σ priors(θ)`, marginalizes `nuisance`, in `reparam` coords, and hands it to a backend (GN default; the same objective feeds Adam/CRB).

So the entire per-problem spec collapses to ONE small call — no per-problem `residual_and_jac` to hand-write:
```python
# calibration
dp_hat = fit(forward_calib, obs, dp0, residual='sqrt_mse',  nuisance=[k],  priors=[ridge_prior], **CALIB_GN)
# reconstruction
trk_hat = fit(forward_track, evt, th9, residual=['poisson','order_stat_time'], nuisance=[], reparam=sincos, **RECON_GN)
```
This is *more general AND simpler* than v3 §1: the forward is pure and reused; "change the loss," "add a prior," "add a nuisance," "swap the optimizer," "reparameterize" are five independent one-arg edits; and the √-MSE↔Poisson choice (the irreducible 10%) is now a `residual=` value, not a rewritten closure. **Recommendation: adopt this split — pure `forward`, a `residual=` registry, `priors=`/`nuisance=`/`reparam=` lists — as the v3 fitting interface, superseding the single `residual_and_jac` closure.** It keeps every algorithm byte-for-byte (the fitter still builds the same `(r,J,W)` blocks) while making the model/inference boundary exact.
