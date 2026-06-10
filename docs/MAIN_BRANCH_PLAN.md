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
