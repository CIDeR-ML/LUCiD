# RECONCILIATION_PLAN.md — `unification` (base) + `refactor-v2` (features) → one `main`

> ## CONSENSUS VERDICT (two independent code-grounded agent reviews, 2026-06-14)
> **Direction APPROVED with caveats — both reviewers: feasible.** The base-side review largely
> REFUTED the main fear (unification independently carries every substantive water-mode fix
> refactor-v2 made — outward normal, stop_gradient reflection normal, per-photon QE, per-photon
> TTS-before-min, key-split, QE-Cherenkov — and is AHEAD on Mie/AD-cleanliness). The features-side
> review GREEN-LIT differentiability of the gating item (volume scatter uses no `custom_vjp`, no
> non-diff ops; direct `vmap` beside the surface step). **⚠️ REFINED (code read, 2026-06-14): "AD-clean"
> ≠ "DiCE/fitting-correct" for ice.** `photon_step_volume.py` is FORWARD-ready and even pathwise-correct
> for λ_scat/λ_abs (`sample_scatter_distance` is an inverse-CDF REPARAMETERIZATION → `∂d/∂λ` is pathwise,
> no score needed, cleaner than the surface DiCE-score). BUT it is **NOT calibration/recon-ready for ice**:
> (i) `compute_scatter_direction` is a FIXED Rayleigh phase function — **no Mie, no anisotropy `g`** (ice
> is strongly forward-peaked `g≈0.9` → physically wrong; and adding `g` is where the real DiCE-score/reparam
> work appears); (ii) **no hard-sample shot-noise mode** (the surface step's exact-Poisson engine that the
> CRB honesty rests on has no volume analog); (iii) **no `log_p`/score carry** (estimator-family + PhotonState
> mismatch vs the surface step). So volume scatter is **two deliverables**: (B-fwd) port forward-only now;
> (B-fit) a SEPARATE, intent-gated DiCE+Mie+`g`+sample-mode rewrite ONLY if ice fitting is a goal — built
> against unification's DiCE/log_p/sample machinery (another reason unification is the right base).
> Required corrections, now folded in below:
>
> 1. **"Byte-identical by construction" → "by tripwire + discipline."** Safety is procedural, not
>    structural: it rests on a fixed-seed reference that DOES NOT EXIST yet (`reference_values.json`
>    pins only unit primitives, no e2e `_common_propagation` tensor). And a forward-only tensor can't
>    protect the base's real value (AD-cleanliness) — **Phase 0 must also pin gradient + AD-Fisher
>    reference tensors + a NaN-free-under-jacfwd assertion** (the custom_vjp backstop is gone).
> 2. **TWO "additive" labels are WRONG — they are a fork/swap (this is where the real schedule risk
>    lives, NOT the param tree or volume scatter):** (a) the **wavelength/medium/optics layer** —
>    refactor-v2 DELETED `optical_model.py` (unification-only, forward-critical) and drives optics via
>    `make_medium`+`spectrum` with `MediumProperties` carrying `emission_processes`/scint/
>    `cherenkov_fraction` that unification's `MediumProperties` lacks; scintillation emission lives
>    there → "reconcile a forked layer," not "add a sub-tuple." (b) **`sources/event_io`** — refactor-v2
>    collapsed the 3384-line monolith (event_io −3285/+99) into a shim over new modules; unification
>    imports `pad_photon_data` which DOES NOT EXIST on refactor-v2 → a SWAP that must preserve
>    unification-only symbols. Insert a **sources-reconcile sub-phase in Phase 3, before production**.
> 3. **t0 emission-time model decision (explicit, was a silent omission):** refactor-v2 has a newer
>    cubic stretched-exponential `predict_t0` + re-fit `t0.json`; unification keeps the OLDER LINEAR
>    form (`siren_rays.py:246`) — and the recon dir/t0 floor was validated on the linear one. Decide:
>    port the cubic schema post-merge behind the tripwire + re-validate timing, OR keep linear and
>    document why. The SIREN net assets (and the importance emitter) travel WITH unification.
> 4. **Inventory was substantially incomplete — added L–P below:** v3 dataset format
>    (`v3_writer/reader` + 13 `tests/io/` byte-identity tests), the refactored sources split, SIREN
>    training-table tooling + 2 CLIs, trained SIREN nets for ice/wbls/electron + t0.json, new geom
>    configs (IceCube86, SK_like_wbls), electron particle support, the full ops-docs corpus, the full
>    `tests/` reorg scope, `jax.remat` on the step.
> 5. **Gate fixes:** add a DetectorParams `tree_flatten` leaf-order byte-pin to Phase 1 (a wrong
>    insertion point silently corrupts every optimizer while the forward tensor still passes); make
>    Phase 6 NUMERIC (fixed seed set, ≥100 events, tolerance band derived from the validated campaign's
>    key-spread — the floor is bias-limited + key-noisy, NKEYS≥48, so a small campaign false-passes);
>    pull a minimal CI workflow forward to Phase 1 (unification has zero CI today).
> 6. **CONFIRMED SAFE:** K (keep `pipeline.py` deletion — no production importer); the registry-based
>    geometry/propagator dispatch makes string genuinely additive. The keystone invariant to PIN: the
>    `photon_update_fn` 14-arg signature + 7-tuple return (`simulator.py:533-542`, `photon_step.py:138/277`)
>    — every new forward branch must conform or it forces a change to the shared `vmap` (the one
>    realistic way the "additive isolation" claim breaks).


> **DIRECTION (decided 2026-06-14):** the **groundwork is `unification`**; **all features from
> `refactor-v2` are added on top.** This is the *opposite* base-choice from the two-agent strategy
> review (which recommended refactor-v2 as base). It is chosen deliberately:
>
> - unification's forward is **AD-clean** (`custom_vjp` already dropped → `jacfwd` works) and is the
>   groundwork the whole calibration+reconstruction stack (`lucid/fitting/`) is built on. refactor-v2
>   still uses `custom_vjp` (`simulator.py:283 → jax.remat(photon_iteration_update_factors_safe)`,
>   `photon_step.py:259`), which blocks AD-Fisher. Building **on** unification means we **never
>   re-introduce custom_vjp** and never re-do that removal.
> - The **recon floor** (vtx ~12 cm, dir ~1°, energy unbiased, t0 ~0.5 ns) and **calibration CRB**
>   were validated on **this exact forward**. Keeping it as the base makes the **water-mode physics
>   byte-identical (enforced by the Phase-0 tripwire, not structurally guaranteed)** — the single highest-severity risk (silent forward drift) is
>   largely *designed out*, because refactor-v2's features are added as **new, non-disturbing
>   branches** (string/ice/volume/scint) that do not touch the validated water-mode path.
>
> **The tradeoff (the "be careful"):** we must re-absorb refactor-v2's ~302 feature commits
> (production, multi-geometry, volume scatter, scintillation, viewer, container, CI) **onto**
> unification. That is more feature-porting volume than the reverse direction — but each piece is
> mostly *additive*, and the validated groundwork stays intact. The discipline that makes this safe:
> **every refactor-v2 feature lands as an additive branch/module that leaves the water-mode forward
> output bit-identical** (enforced by a fixed-seed reference-tensor tripwire, Phase 0/2).

---

## 0. Ground truth (verified against the repo)

- `merge-base(unification, refactor-v2) = 3a14e3c`; unification = 213 unique commits, refactor-v2 =
  341 unique commits, **disjoint histories** → a symmetric `git merge` is rejected (semantically
  wrong conflicts in the diverged forward). We integrate by **curated, additive commits** on a branch
  cut from **unification**.
- `lucid/fitting/` (8 files), `lucid/simulation/reflection.py`, `lucid/wavelength/optical_model.py`,
  `examples/` exist **only** on unification. They stay put — they are the base.
- refactor-v2's net-new surface (to be ADDED): `lucid/geometry/{string.py,string_sizing.py}` (+ geom
  type-hint refactor), `lucid/simulation/photon_step_volume.py`, scintillation params + ice/electron/
  WbLS materials, `lucid/production/*` (cluster_common, 18 `dataprod_*.json`, `run_job`, GENIE, jobs),
  `viewer/`, `container/`, `ci_tests/`, `.github/workflows/`, the richer `pyproject.toml`.
- **Geometry is a superset, not a fork** — both branches have the decorator `registry.py`; refactor-v2
  added `string.py` + type hints. So geometry merges by taking refactor-v2's additions.

---

## 1. Feature inventory — what to port from refactor-v2, classified

| # | Feature | Files (refactor-v2) | Class | Notes / care |
|---|---|---|---|---|
| A | **String geometry** | `lucid/geometry/string.py`, `string_sizing.py`, geom type-hints | additive | new `@register_detector`; verify Cylinder path (what fitting uses) unchanged |
| B-fwd | **Volume scattering — FORWARD** (ice) | `lucid/simulation/photon_step_volume.py` | forward port | reparam-pathwise for λ_scat/λ_abs; port forward-only, label "fitting-not-validated"; NaN-stress new geom |
| B-fit | **Volume scattering — FITTING-correct** (ice) | rewrite of above | **scoped sub-project, intent-gated** | ONLY if ice calib/recon wanted: add Mie + anisotropic `g` phase (→ `g`-gradient via reparam/score), hard-sample shot-noise mode, `log_p`/PhotonState carry; gate AD==FD on ice optical params + sample-engine cross-check. Build on unification's DiCE machinery |
| C | **Scintillation + materials** | scint surrogate; ice/electron/WbLS material JSON + loaders | **FORK — careful** | scint scalars → `ScintillationParams` sub-tuple is mechanical (D); BUT scint EMISSION lives in refactor-v2's `MediumProperties` (`emission_processes`/`cherenkov_fraction`/scint λ-range) which unification lacks, and refactor-v2 DELETED `optical_model.py` (unification's optics seam) — must RECONCILE the forked `wavelength/{medium,spectrum,optical_model}` layer, not drop a branch |
| D | **DetectorParams: add scintillation** | `lucid/detector_params.py` | **keystone — careful** | KEEP unification's nested tree; ADD `ScintillationParams`; route refactor-v2 material loaders through `from_flat` |
| E | **`setup_event_simulator` union args** | `lucid/simulation/simulator.py` | union signature | ADD refactor-v2's `medium_override` + string/scint modes; KEEP unification's `reflection_model/reflection_wavelength/spectrum/overlap_*` |
| F | **String track mode + scint emission** | `lucid/sources/*` (`run_string_siren_tracks`, scint) | additive | KEEP unification's fixed-proposal IMPORTANCE SIREN emitter (the validated one); add string/scint as additional source paths |
| G | **Production chain** | `lucid/production/*` (cluster_common, configs, run_job, run_genie, verify, jobs) | additive (large) | take wholesale; reconcile unification's `voxelize.py`/`particle_data_utils.py`/`generate_events.py` (refactor-v2 removed some) |
| H | **Tooling/infra** | `viewer/`, `container/`, `ci_tests/`, `.github/workflows/` | additive (top-level) | no conflict with `lucid/` |
| I | **Packaging** | `pyproject.toml` | union | union `[project.scripts]` (prod CLIs + `lucid-optimize` + `lucid-train-siren`), extras, package-data |
| J | **Tests** | refactor-v2 `tests/` subdir layout + string/prod/geom tests | union | adopt subdir layout; slot unification's fitting/recon/reflection/calibration tests in; hand-merge `conftest.py`/`baselines.npz`/`reference_values.json` |
| K | **`optimization/pipeline.py`** | kept on refactor-v2; deleted on unification | decision → **KEEP DELETION** | confirmed safe: no production importer (`git grep` clean); keep unification's lean Fisher-GN `run.py` |
| **L** | **v3 dataset format** | `lucid/sources/{v3_writer,v3_reader}.py`, `docs/LUCID_DATASET.md`, 13 `tests/io/` byte-identity tests | **SWAP — careful** | the production output schema; scint emission couples to `EMISSION_PROCESS_SCINTILLATION` from v3_writer |
| **M** | **Refactored sources split** | `sources/{cascade,event_builder,event_generation,root_reader,particle_physics,particle_categorization,segment_grouping,seed_utils,legacy_io,track}.py` | **SWAP — careful** | `event_io` −3285/+99 → shim over these; MUST preserve unification-only `pad_photon_data` etc. (recon/forward import it; absent on refactor-v2) |
| **N** | **SIREN training-table tooling** | `lucid/siren/training/photonsim_data/{build_photon_table,build_dedx_table,...}.py` + CLIs `lucid-build-{photon,dedx}-table` | additive | add to pyproject scripts (I) |
| **O** | **Trained nets for new media** | `data/{ice,wbls,water}/{electron,muon}/siren_params.json` + `t0.json` | additive (data assets) | required to run ice/wbls/electron modes; coordinate with §3.t0 decision |
| **P** | **New geom/material configs + electron** | `config/IceCube86_*`, `config/SK_like_wbls_*`, `*.npz`; electron emitter + dataprod_03/06_e | additive | needed to exercise string/ice/wbls/electron |
| **Q** | **Ops docs corpus** | `docs/{QUICKSTART_DOCKER,LOCAL,LXPLUS,NERSC,S3DF,CLUSTER_ABSTRACTION,LUCID_DATASET,WBLS_INFORMATION,SIREN_TRAINING_INPUTS,LUCID_MIGRATION*}.md` | additive | Phase 7 docs union |

---

## 2. Phases (each independently testable; gates explicit)

**Phase 0 — Provenance + tripwire + integration branch (safe, no forward change).**
- Tag `provenance/unification-<sha>` and `provenance/refactor-v2-49905d6`; freeze both worktrees.
- **Build the tripwire (it does NOT exist yet** — `reference_values.json` pins only unit primitives):
  capture **fixed-seed water-mode (SK_like) reference tensors** for (i) the forward `_common_propagation`
  charges+times, (ii) the **AD/jacfwd gradient**, and (iii) the **AD-Fisher** — plus a **NaN-free-under-
  jacfwd** assertion (the custom_vjp backstop is gone). A forward-only tensor can't protect the base's
  real value (AD-cleanliness), so all three are required.
- Pin a **DetectorParams `tree_flatten` leaf-name/order reference** (structural, not just round-trip).
- Snapshot the recon-floor + calibration-CRB reference numbers + the validated campaign's seed set /
  event count / key-spread (for the Phase-6 tolerance band).
- Cut `integration/unify-main` **from `unification`** (HEAD); pull a **minimal CI workflow** forward now
  so Phases 1–4 get automated regression (unification has zero CI today).
- *Gate:* unification's fast suite (384) green on the branch; tripwire + leaf-order pin committed.

**Phase 1 — DetectorParams keystone (add scintillation; keep nested).**
- Add `ScintillationParams` sub-tuple (refactor-v2's scint scalars `S,kB,C,tau_*,moyal_*`) to the
  nested `DetectorParams`; extend `from_flat`/`_nest_flat_kwargs`/`default_bounds`/grad-scales (neutral
  defaults → forward byte-identical). Route refactor-v2's ice/WbLS/electron material JSON loaders
  through the existing `from_flat` flat-wire shim.
- *Gate:* both loaders round-trip (refactor-v2 material JSON **and** unification `from_flat`/save/load);
  `normalize/denormalize/default_bounds/make_optimization_mask/particle_bounds` tree-walk the new tree;
  water-mode forward **byte-identical** vs the Phase-0 tripwire; fitting `problem.py` unaffected.

**Phase 2 — Forward features onto the AD-clean step (the careful core).**
- Port `photon_step_volume.py` (volume scatter) onto unification's `photon_step.py` as an **additive,
  differentiable** branch — NO `custom_vjp`; reuse the eps-inside-sqrt NaN floors. Add string
  intersection + scint emission to the forward as new branches gated by detector_type/source.
- *Gate (critical):* water-mode forward **byte-identical** vs Phase-0 tripwire (the validated path is
  untouched); **AD==FD** on water AND ice/string/volume modes (extend `test_multibounce_jacobian` +
  `hessian_probe`); NaN stress (`recon_nan_thorough` pattern) over the NEW geometry/volume surface.

**Phase 3 — Simulator union signature + SOURCES RECONCILE (must precede production).**
- 3a **Sources swap (M/L):** replace unification's monolithic `event_io.py` with refactor-v2's split
  modules (`root_reader, event_builder, event_generation, v3_writer/reader, seed_utils, …`) **while
  preserving unification-only symbols** the recon/forward import (`pad_photon_data`, etc.). Add string-
  track + scintillation + electron source paths; KEEP the importance SIREN emitter + its nets.
- 3b **Simulator union signature:** one `setup_event_simulator` with the union arg-set — ADD
  `medium_override` + string/scint modes, KEEP unification's `reflection_model/reflection_wavelength/
  spectrum/overlap_*` (the overlap/reflection knobs are load-bearing for the recon floor). Update
  `lucid/__init__` lazy exports.
- *Gate:* water-mode tripwire holds; `lucid-optimize` + recon scripts still import (pad_photon_data
  preserved); string/ice e2e forward produces nonzero sane charge; examples + fitting calls unchanged.

**Phase 4 — Production + tooling (additive bulk).**
- Bring `lucid/production/*` wholesale from refactor-v2; reconcile the few unification-only production
  files. Add `viewer/`, `container/`, `ci_tests/`, `.github/workflows/`. Union `pyproject.toml`.
- *Gate:* production CLIs import; container builds; `pip install -e .[all]` resolves.

**Phase 5 — Test-suite union + CI.**
- Adopt refactor-v2's `tests/` subdir layout; slot in unification's fitting/recon/reflection/optical/
  calibration/timing tests; hand-merge `conftest.py`/`baselines.npz`/`reference_values.json`. Extend CI
  + `ci_tests/` to run the union.
- *Gate:* full unioned suite green in CI; container build green; perf checks pass.

**Phase 6 — Re-validation (the real acceptance gate — NUMERIC, not qualitative).**
- Re-run the recon campaign at the **validated scale** (the same ~100-event GEANT4 seed set, NKEYS≥48
  per the bias-limited/key-noisy floor) + a calibration CRB check on `integration/unify-main`. Gate on a
  **tolerance band derived from the original campaign's key-spread**, not eyeballed (a small/low-key
  campaign false-passes — single-key FD noise ≈12×AD at K=8). Confirm vtx ~12 cm / dir ~1° / E-unbiased
  and the calibration CRB within band. Check the numbers into `reference_values.json` as `@slow` asserts.

**Phase 7 — Docs + cutover.**
- Reconcile the two near-disjoint doc corpora into ONE `ARCHITECTURE.md` (fold `MAIN_BRANCH_PLAN.md`)
  + keep refactor-v2's `QUICKSTART_*`/`LUCID_DATASET`/`CLUSTER_ABSTRACTION` as the ops authority; add a
  top-level `CLAUDE.md`. Resolve K (pipeline.py) per the production audit. Fast-forward `main`.

---

## 3. The "be careful" list (invariants that must hold every phase)

1. **Water-mode forward output stays bit-identical** to Phase-0 reference tensors — this is what
   preserves the validated recon floor + CRB. Any feature that perturbs it is a bug, not a feature.
2. **No `custom_vjp` re-introduction.** Volume/string/scint forward branches must be plain-AD +
   eps-floor NaN-safe, or the AD-Fisher (calibration CRB, recon `fisher_mode='ad'`) breaks.
3. **DetectorParams pytree shape**: adding `ScintillationParams` must not shift any existing leaf or
   break refactor-v2's material loaders or fitting's tree-walks (byte-pin both).
4. **NaN-safety on NEW geometry**: refactor-v2's string/volume surface was never tested without the
   custom_vjp backstop — NaN-stress it under plain AD.
5. **Coordinate with the refactor-v2 author** on the merged-param shape and the production-import
   surface before Phases 1–2; agree a known cut commit (origin keeps moving).
6. **Re-validate, don't assume**: green unit tests ≠ floor survived. Phase 6 is mandatory.

## 4. Effort / risk

Medium-large; concentrated in Phase 1 (param keystone) + Phase 2 (forward feature-porting, the schedule
driver) + Phase 4 (production bulk, mechanical) + Phase 5 (test/CI union, mechanical). Lower forward-drift
risk than the reverse direction (water-mode preserved by construction); higher feature-porting volume.
**Fallback** if a refactor-v2 forward feature (e.g. volume scatter) cannot be made AD-clean cheaply:
land it as an explicitly-selected non-AD mode (production/data-gen only) behind a `forward_backend=`
switch, while fitting targets the AD-clean water/ice path — converge later.
