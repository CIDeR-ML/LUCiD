# Unifying calibration + reconstruction on one differentiable forward

## The core realization
**Calibration and reconstruction are the SAME problem** — a Gauss-Newton fit on the SAME differentiable photon
forward — with the free-parameter set swapped:
- **Calibration** fits the DETECTOR params (optical λ-curves, per-PMT QE/gain/t0/walk, reflectivity, SPE width, TTS)
  given KNOWN sources (lasers/iso/Cherenkov). Lives in `LUCiD/mie_hunter/`.
- **Reconstruction** fits the TRACK params (E, x, y, z, dir, t0) given a KNOWN detector, with a SIREN track source.
  Lives in `LUCiD_recon/` (a git fork of the base sim, recon layer UNCOMMITTED in the worktree).
- **Joint / self-calibration** = fit BOTH blocks of the same likelihood (freeze either) — the ultimate generalization.

They are not two systems to merge; they are one framework with two param configs. **They already share the same
engine**: the recon fork ported its DiCE/implicit-capture `photon_step.py` FROM the mie_hunter calibration engine
(`implicit_engine_lik.py`). Same `log_p` DiCE carry, `hg_logpdf`/`rayleigh_logpdf`, `dice_dep`, Rayleigh+Mie-by-rates.

## The fragmentation today: THREE copies of the forward
1. **Base `LUCiD/lucid/`** (committed) — the OLD STE / transport-mean-field expected step. The DiCE upgrade is NOT here.
2. **`LUCiD_recon/lucid/`** (uncommitted worktree edits to 10 base files) — the DiCE engine rewrite of `photon_step.py`
   + recon built on `setup_event_simulator`. The real-geometry forward, but the engine changes aren't committed.
3. **`LUCiD/mie_hunter/`** — STANDALONE toy engines (`wl_engine2`, `wlt_engine`, `refl_engine2`) that REIMPLEMENT the
   forward self-contained (SK cylinder HARDCODED, no geometry registry) — fast to iterate, but a 3rd copy of the physics.

So the same differentiable forward exists in three diverging forms. That is the thing to consolidate.

## What each side already built (reusable)

**Shared / generic (belongs in the base sim, used by BOTH):**
- The **DiCE/implicit-capture forward** (`photon_step.py` DiCE rewrite = the mie_hunter engine): two-channel Rayleigh+Mie
  by rates, reparam live free-path for the optical→TIME gradient, per-step DiCE score, detached reflection normal.
- **Observables**: charge Poisson NLL + first-arrival timing (`make_hits_*`, `first_arrival_nll`, `poisson_nll`).
- **Optimizer**: GN + FD-Hessian-recomputed + median-ridge + **positive-eigen-clip** (recon `gnrec`/`gn_recon.py` ==
  calibration `gn_fast`/Schur family). FD/HVP Fisher (AD Hessian is BROKEN for geometry; jacfwd blocked by the DiCE
  custom_vjp).
- **Overlap + OVERLAP_RENORM** (soft-overlap ~1% under-count fix), `cubic`/`erf` analytic overlap (correct AD Hessian).
- **2nd-order-AD NaN fixes**: CYL_SQRT_EPS, SAFE sqrt-norm, mie/g missing-scalar loader projection, first_arrival_nll
  empty-segment floor — all benefit calibration too.
- **Per-photon TTS** smearing + per-sensor time gate (`sensor_response.py`).
- **DiCE plumbing**: `PhotonState.log_p` carry, `PhotonStepResult` 7-tuple, `custom_vjp` extension.

**Reconstruction-specific (additive, no conflict with calibration):**
- SIREN track emitter (`siren_rays.py` ∝density inverse-CDF + DiCE, `SIREN_IMPORTANCE` pathwise energy).
- Track `ParticleParams` 9-vec `[E,x,y,z,sinθ,cosθ,sinφ,cosφ,t0]` (sin/cos coords).
- **AMP_DETACH** time↔charge factorization (energy = charge term's job; geometry/dir/t0 = time term's).
- Whole-tank basin seeding (grid-vertex search + `gnrec` 15 m descent).

**Calibration-specific (additive):**
- Laser/iso/Cherenkov sources at λ/position/intensity (+ multi-intensity ladder for the TQ map).
- Detector param registry: optical λ-curves, per-PMT QE/gain/t0/walk, reflectivity (Schlick + multilayer Fresnel), w, TTS.
- The **charge-VARIANCE observable** (compound-Poisson — breaks QE↔gain, measures SPE width) — recon does NOT have this; it's new and belongs in the shared observable library.
- Per-PMT Schur marginalization + gauges (mean-log-k=0).

## The unified architecture (target)
ONE base forward + ONE fit framework + thin application layers, all committed in `LUCiD/lucid/`:

```
lucid/simulation/        # the ONE DiCE/implicit-capture forward (commit recon's rewrite = the convergence point)
lucid/geometry/          # registry (any detector — already exists)
lucid/sources/           # source registry: laser / isotropic / Cherenkov / SIREN-track
lucid/fitting/   (NEW)   # SHARED: GN+FD-Hessian+pos-clip+ridge, Fisher/CRB, Schur per-nuisance marginalization,
                         #         declarative PARAM REGISTRY, gauges, priors (curvature), observable+noise models
                         #         (charge-mean Poisson, charge-VARIANCE compound-Poisson, first-arrival timing)
lucid/calibration/ (NEW) # free = DETECTOR params; thin config over lucid/fitting
lucid/optimization/      # free = TRACK params (recon); thin config over lucid/fitting  (already exists, refactor onto fitting)
# joint = free both blocks -> self-calibration (the generalization)
```
"Add a calibrated quantity" or "add a track param" both become one row in the param registry; the GN/Schur/Fisher
machinery is written ONCE.

## What's NEEDED (consolidation steps, in order)
1. **Commit the unified DiCE forward to base `lucid/`.** Reconcile the recon `photon_step.py` rewrite with the
   mie_hunter engine into ONE (they're the same family; recon's is real-geometry-integrated → adopt it). Resolve the
   merge-critical surface the agent flagged: the `siren_rays` default emission change (re-validate), the `PhotonState`/
   `PhotonStepResult` tuple-shape change (positional constructors break), and **isolate the import-time env writes**
   (recon sets `MAXCELL`/`OVERLAP_RENORM=1.009`/`TTS_NS=2.5` at import — must not leak into calibration runs → make
   them explicit args, not process-global env).
2. **Re-base the calibration onto the real forward.** The mie_hunter toy engines reimplement physics on a hardcoded SK
   cylinder; re-express the calibration fit on `setup_event_simulator` + the geometry registry so it runs on ANY
   detector (HK/WCTE/IceCube) and shares the exact forward recon uses. Keep the toys as a fast unit-test oracle.
3. **Factor the shared fit machinery into `lucid/fitting/`.** Merge `gnrec`(recon) + `gn_fast`(calib) → one GN module;
   `crb_*`(recon) + `fisher_*`(calib) → one CRB module; add the declarative param registry + the charge-variance
   observable (new, from calibration) + first-arrival timing (from recon). One gauge/Schur implementation.
4. **Express calibration & recon as param-configs** on `lucid/fitting`. Verify each reproduces its validated results
   (calibration: 6 curves + per-PMT QE/gain/t0/walk + w/TTS; recon: vtx 7-8 cm / dir 0.26° / E 0.26%).
5. **Add the JOINT fit** (detector + track free together) — the self-calibration capability that neither side has.
6. **Adopt the settled cross-decisions**: matched-MAXCELL (=4 fine, not ≥16 — `MAXCELL_PLAN.md`); FD/HVP Fisher (not AD
   Hessian); positive-eigen-clip essential; OVERLAP_RENORM per coverage; quote CRB-vs-realized with the √12 caveat.

## Decisions to make
- **Home**: promote both into the committed base `lucid/` (reusable, needs tests) vs keep `mie_hunter/` sandbox + recon
  worktree. Recommendation: commit the forward to base, build `lucid/fitting/` + `lucid/calibration/` there, retire the
  duplicates.
- **Scope/order**: (1) commit+unify the forward is the unlock for everything else; do it first.
- The 3 copies must become 1 before any clean consolidation — that's the load-bearing step.

## One-line takeaway
Calibration (`mie_hunter`) and reconstruction (`LUCiD_recon`) are two parameter-configs of the SAME GN fit on the SAME
DiCE forward — which currently exists as three diverging copies (base STE, recon worktree, mie_hunter toys). Consolidate
by committing ONE DiCE forward to base `lucid/`, factoring ONE `lucid/fitting/` (GN+Fisher+Schur+param-registry+
charge/variance/timing observables), and making calibration, recon, and the joint self-calibration thin configs on top.
