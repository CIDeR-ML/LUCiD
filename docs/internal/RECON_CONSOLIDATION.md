# RECON_CONSOLIDATION.md  (v2)

**Status:** design / port plan. The **calibration** half (`lucid/fitting/`: GN, Schur, ridge, CRB, `build_calibration_problem`) is **already built and green** under `tests/test_fitting*.py` — this plan *extends* it, it does not design it. The **recon** half (single-stage Fisher-GN, order-statistic time term, the loss-quality fixes) is what is missing and gets ported.
**Authoritative source for the recon pipeline:** `/sdf/group/neutrino/omara/LUCiD_recon/RECO_PIPELINE.md` + `gn_fisher_recon.py` (+ `recon_ps_setup.py`, `recon_harness.py`).
**Provenance:** 9-agent audit + 4-agent design + 5-agent adversarial review. Every claim is anchored to `file:line`; verify before editing — line numbers drift.

> v2 changelog (from review): metric is argument-ified not a callback; `Problem` interface trimmed to 4 members; reuse `SimConfig`/`build_calibration_problem`/`recon_ps_setup`; SCALE9 derived from a new `particle_bounds()`; env-leak fix extended to unify's own `sensor_response.py`; sequencing reordered (de-env first); added `pipeline.py` disposition (§10), test net (§11), and joint-fit frontier (§9). ~26 dead constants dropped, not promoted.

---

## 0. TL;DR

- Forward **substrate is byte-identical** (same SIREN md5, geometry, configs). The recon floor (vtx ~13 cm / dir 1.77° / E ~2.3%) reproduces in unify because it's the same emitter.
- Missing in unify: the recon *driver* — single-stage Fisher-GN, the order-statistic time likelihood, and the loss fixes (Poisson-unnormalized, AMP_DETACH, TOT_N_SCALE).
- Present in unify: the **calibration** GN (`lucid/fitting/`) — same optimizer family, different problem, fully tested.
- The recon/calibration divergence is **~90% arguments**. The 10% that isn't is the *residual form* (`residual_and_jac`) and the *per-PMT nuisance* (Schur). **The metric is NOT in that 10%** — it argument-ifies (§1.3).

---

## 1. Target architecture

One Gauss-Newton engine. recon and calibration are two `Problem` instances + an options bag. The shared `fit()` body owns: refresh scheduling, the FD-Jacobian/CRN loop, **the normal-matrix assembler**, precondition, ridge solve, step clip, LR, nuisance back-substitution (no-op when the list is empty), readout.

### 1.1 The `Problem` — 4 members, closure style (not a heavy Protocol)

`pack` has **zero call sites** on either side (calib's `theta0` comes raveled from `build_calibration_problem:79-80`; recon's `th9` from `derive_truth:56-61`). `param_space` is redundant with `unpack` (calib `unravel` does `exp`; recon `sc2` doesn't). `n_sensors` is Schur bookkeeping — fold into the nuisance block. The `ResidualBundle`/`ResidualBlock` records buy nothing over the `list[SourceModel]` + `Σ/S` that calib `fit` already loops (`gauss_newton.py:200-205`). So:

```python
@dataclass
class Problem:
    unpack:           Callable          # theta -> pytree   (sc2 | unravel)   -- owns sin/cos & log parameterization
    residual_and_jac: Callable          # theta,data,keys -> [(r, J, W), ...] blocks via CRN-FD  (the ONE polymorphic method)
    scale:            np.ndarray         # SCALE9 | ones      (preconditioner; see §2.2 — derived, not magic)
    nuisance:         tuple = ()         # () recon | (Nuisance('k'),) | (Nuisance('k'), Nuisance('t0')) calib
```

`Nuisance(name, couples_to, gauge='mean', step_max=0.3, bake=False)`. The calibration `Problem` IS `build_calibration_problem` (`problem.py:42`) lightly refactored to return this (the `unravel` closure → `unpack`; the `SourceModel` list → `residual_and_jac`; one `Nuisance('k')`). The recon `Problem` is a symmetric `build_recon_problem` wrapping `gn_fisher_recon.perpmt` (`:77-89`) as `residual_and_jac`, `scale=particle_scale()`, `nuisance=()`.

### 1.2 Mapping current code → interface

| piece | Calibration (`lucid/fitting/...`) | Recon (`LUCiD_recon/...`) |
|---|---|---|
| `unpack` | `unravel(theta_log,k=1)` `problem.py:84-91` | `sc2(t9)` `gn_fisher_recon.py:41-43` |
| `scale` | `ones` (log-space) | from `particle_bounds()` (§2.2), NOT literal SCALE9 |
| `residual_and_jac` charge | `sqrt_residual` `gauss_newton.py:31-33,105-107`; FD `:112-121`; `W`=lit-mask | `perpmt`→`mu`,`chargeloss` `gn_fisher_recon.py:78-97`; `Jm` FD `:130-132`; `W`=`1/mu` `:133` |
| `residual_and_jac` time | `rtA=(T+t0)−tt` `gauss_newton.py:316`; `Jt` FD `:264-274`; `W`=`w_time·lit` | **`perpmt`→`tnll` erf-survival `gn_fisher_recon.py:78-89`** (NOT `losses.first_arrival_nll`; that is the legacy soft-min — §3); `Jl` FD score `:131-132`; `W`=`I` |
| `nuisance` | `(Nuisance('k'),)`; +`('t0',)` in `fit_charge_time` `gauss_newton.py:340-341` | `()` |

> Correction vs v1: the recon time row points at the **to-be-ported** erf-survival term, not `losses.py:419`. The v1 table contradicted §3.

### 1.3 The metric argument-ifies — one assembler, four copies collapse

Every normal-matrix block on **both** sides is `Jᵀ diag(W) J`:
- calib charge `Htt += (J·W)ᵀJ`, `W`=lit-mask `gauss_newton.py:202,333`
- recon charge `Fc=(Jm/√μ)ᵀ(Jm/√μ)` = `Jᵀdiag(1/μ)J` `gn_fisher_recon.py:133`
- recon time `Ft=JlᵀJl` = `Jᵀ·I·J` (score-cov) `:133`
- calib time `Htt += w_time(Jt·Wt)ᵀJt` `:334`

The only variation is `W` (carried on each `(r,J,W)` block) and the **Schur reduction** (gated by non-empty `nuisance`). So the metric is **not** a per-Problem callback — it is one shared function:

```python
def assemble_normal(blocks, nuisance):
    H = sum((J*W[:,None]).T @ J for (r,J,W) in blocks)      # Σ Jᵀdiag(W)J
    g = sum((J*W[:,None]).T @ r for (r,J,W) in blocks)
    for nb in nuisance:                                      # empty for recon -> no-op
        Htk, Hkk, gk = nb.couple(blocks)                    # k: Jk=0.5·m ; t0: Jk=1
        Minv = make_constrained_schur(Hkk)                  # gauge mean()=0
        H -= Htk @ Minv @ Htk.T ;  g -= Htk @ Minv @ gk
    return H, g
```

This collapses **four near-duplicate assemblers**: `gauss_newton.py:198-218` (`fit`), `:328-351` (`fit_charge_time` — the 2-nuisance case), `fisher.py:62-76` (CRB — same assembly ×4·Nphot + honesty), and `gn_fisher_recon.py:125-136` (recon). It also collapses `fit` and `fit_charge_time` into **one loop** with `nuisance ∈ {(), (k,), (k,t0)}`.

### 1.4 Also collapse: the CRN-FD Jacobian loop (3 copies) and readout

- **FD-Jacobian** is triplicated: `SourceModel.fd_jacobian` `gauss_newton.py:112-121` (forward diff), `ChargeTimeModel.fd` `:264-274`, recon `fisher` `gn_fisher_recon.py:128-132` (**central** diff — unify to one; pick central). One `fd_jacobian(fn, theta, h, keys, n_avg, central)` with `n_avg` absorbing both calib `nb_h` (`:74-77`) and recon `NKEYS` (`:130-131`) key-averaging.
- **Readout** three-way (last/polyak/ming): calib `gauss_newton.py:231-237` already has Polyak; recon `:179-188` adds ming. One `readout(history, gnorms, mode)`.

---

## 2. The argument layer

### 2.1 Optimizer — `OptConfig` (new; nothing equivalent exists)

Unified `fit(...)` kwargs; **every default = current calibration literal**.

| arg | default = calib | recon | lines (`gauss_newton.py`) |
|---|---|---|---|
| `scale` | `None`→ones | `particle_scale()` | new; identity when ones |
| `precond` | `'auto'`→off when `scale is None` | on | solve `:211,221` |
| `ridge_mode` | `both` | `both` | `ridge_inverse:62` |
| `ridge` / `mu` | `0.02` / `0.3` | `0.01` / `0.3` | `ridge_inverse:60-62,64` |
| `eigen_clip` | **`True`** | **`False`** | `ridge_inverse:63-64` |
| `lr` / `noclip` | `1.0` / False | `8` / True | `:221,224` |
| `readout` / `refresh` | `last` / `15` | `ming` / `8` | `:231-237,195` |

### 2.2 SCALE9 is derived, not magic — add `particle_bounds()`

There is **no `ParticleParams` bounds/scale** in unify today (`default_bounds:620` is DetectorParams-only), and the recon repo itself carries **two inconsistent magic vectors**: `gn_fisher_recon.py:37` SCALE9 = `[50, .2,.2,.2, .02,.02,.02,.02, .2]` (sin/cos) vs `recon_harness.py:22` PARAM_SCALE = `[50, .2,.2,.2, .02,.02, .2]` (raw angles). Promoting "SCALE9" verbatim freezes that duplication.

`[50 MeV, 0.2 m, 0.02, 0.2 ns]` is exactly a per-field characteristic scale = `(hi−lo)`. **Add `particle_bounds()` to `detector_params.py`** mirroring `default_bounds`, derive the preconditioner as `(hi−lo)` via the existing `normalize_params`/`denormalize_params` (`:604-617`, type-agnostic). Put the sin/cos packing in `Problem.unpack` (keeps `pack∘unpack≠id` local — §9). This retires both magic vectors and reuses the normalize machinery.

### 2.3 Loss — slim `LossConfig` (most of it already exists)

The charge "enum" mostly maps to existing functions — map, don't invent:

| value | existing impl | note |
|---|---|---|
| `poisson` | `gn_fisher_recon.chargeloss:97` `sum(mu−oc·log mu)` | the finalized form |
| `poisson_norm` | `counts_loss`/`poisson_nll` `losses.py:292-312` (`÷Σtrue`) | = `poisson` + `charge_normalize=True`; **drop as a separate enum value** |
| `sqmse` | `sqrt_residual` `gauss_newton.py:31` AND `WC_smooth_loss` `losses.py:176` | reuse `sqrt_residual` so GN-residual and scalar-loss share one √ |
| `anscombe` | `gn_fisher_recon.chargeloss:95` | 1 place |

So `charge_loss ∈ {poisson, sqmse, anscombe}` × `charge_normalize: bool`. Other fields: `amp_detach`, `tot_n_scale`, `sigma`, `delta`, `w_charge`, `w_time`. **`time_model` and `hit_mode` move to the extended `SimConfig`** (they select the forward — §4.8), NOT `LossConfig`. `overlap_renorm` is finalized OFF (1.0) and is a forward kwarg — not a `LossConfig` field at all.

> **`amp_detach` already exists in legacy form:** `recon_ps_setup.first_arrival_shape` (`recon_ps_setup.py:96-117`) drops `-log_w_total` and `stop_gradient`s `N_s` — the count-detach sibling. Fold it in; don't write a third time term (§3).

### 2.4 Configs: extend `SimConfig`, 2 new dataclasses, drop ~26 constants

- **`SimConfig` already exists** (`config.py:5`, built at `simulator.py:216`). It holds `n_photons,K,mode,...`. **Extend it** to absorb `temperature`, `MAXCELL`, `TTS`, `hit_mode`/`time_model`, `reflection_model` — currently loose kwargs on the 30-arg `setup_event_simulator` (`simulator.py:46-74`). v1's "they're already kwargs" was only half-true.
- **`OptConfig`** (new, ~10 fields), **`LossConfig`** (new, ~7 fields, slimmed per §2.3).
- **`StudyConfig`**: reconcile with the existing `lucid/optimization` JSON-config loader (`run.py:148-172`) — make it the typed deserialization target, not a 4th config dialect. Deferrable (gates nothing in steps 1,3,4).
- **DROP ~26 dead constants** instead of promoting them (the real count is 64 env keys, not ~40):
  - 13 diagnostic modes (`ESWEEP/ZESWEEP/LOSSCHK/GRADCHK/CMP/HIST/TRACE/SELFDATA/DATA_NKEY/...`) → standalone scripts, not fit API.
  - ~10 rejected experiments (`EDETACH/EDET_W`, `TROBUST/TCAP/TW`, `OCC_GATE/OCC_MIN/CGATE`, `T0_PRIOR`, non-`poisson` `CLOSS`) — proven inert (`RECO_PIPELINE.md §3.5`); **delete, don't field-ify**.
  - redundant: `OVERLAP_RENORM` (≡TOT_N_SCALE, finalized 1.0), `ADAPT/LAM0/LAM_HI/LAM_DECAY` (anneal unused — fixed LAM), `STEPCLIP` (NOCLIP=1).
  - 64 → ~25 supported fields.

Provide `OptConfig.finalized()` / `LossConfig.recon()` classmethods so the proven recipe is one import.

---

## 3. The one PORT: order-statistic time term

`lucid/losses.py` has `first_arrival_nll:419-469` = legacy **soft-min mixture-density** (`log_sigmoid` kernel + `segment_logsumexp`), a different construction. The finalized **erf-survival window** must be lifted from `gn_fisher_recon.py:76-89`. **Build it on existing scaffold** — `segment_logsumexp` (`losses.py:406`) and the `LOG_FLOOR=-60` empty-segment guard (`:453-467`) are reusable; `log1mexp` (`gn_fisher_recon.py:76`) lifts verbatim. The new function should **subsume `recon_ps_setup.first_arrival_shape`** (the amp_detach path), not become a 4th time loss.

```python
def first_arrival_order_stat_nll(log_w, flat_times, flat_idx, t_obs, mu_survival, occ,
                                 num_detectors, sigma=2.5, delta=1.0, amp_detach=True):
    """Mean-field-exact survival-window order-statistic first-arrival NLL.
    R(t)=Σ wᵢ Φ((t−fᵢ)/σ); S=(mu_survival−R)/mu_survival; nll=−log(S(tlo)ⁿ−S(thi)ⁿ).
    amp_detach detaches BOTH wᵢ AND mu_survival."""
```

**Optional leaner form (§5-review):** a single `first_arrival(..., kernel∈{'erf_window','sigmoid_softmin'}, amp_detach)` hosting both kernels removes ~40 lines of duplicated survival/floor scaffold — but keep legacy `first_arrival_nll`'s signature intact (imported by `recon_harness`, `recon_ps_setup`, `tau_scan`) as a thin wrapper.

Consumes the `(log_w, flat_times, flat_idx, total_charge)` tuple `make_hits_likelihood` already returns (`sensor_response.py:244-291`) — no forward change.

---

## 4. Load-bearing invariants (silent regressions if violated) — all CONFIRMED at cited lines

1. **`eigen_clip=True` default.** Calib Schur-Hessian can be indefinite under FD noise; `clip(ev,0.5·m,None)` (`gauss_newton.py:64`) absorbs it. Recon score-cov is PSD, omits it. Default-False blows up calib.
2. **`ridge_mode='both'` default.** The additive `mu·median·I` (`:62`) sets the `0.5·m` floor; Marquardt-only un-floors the clip.
3. **`precond` off when `scale is None`; keep the unscaled `ridge_inverse(H)` as the default path** (don't route calib through the `S·F·S` round-trip — breaks byte-identity + per-param damping).
4. **`tot_n_scale` scales `mu` (charge) NOT `muS` (survival).** Confirmed `gn_fisher_recon.py:79` (`mu=max(tot·TOT_N_SCALE,…)` vs `muS=stop_gradient(max(tot,…))`). Apply inside the charge branch only; pass raw `tot` as `mu_survival`. Violating it collapsed the 10 m basin.
5. **`amp_detach` detaches BOTH `ww` and `muS`.** Confirmed `:79` (both `stop_gradient` on one line). `mu`-detach alone → +214 MeV.
6. **`counts_loss` `normalize=False` default.** Confirmed `losses.py:310-311` divides by `Σtrue`, and `poisson_nll IS counts_loss` (`:511`) so the division is global. `recon_harness.py:97` `jnp.sum(poisson_nll(...))` sums a scalar → silently runs the energy-killing normalized loss. (Note: the finalized `gn_fisher_recon` bypasses `recon_harness.make_loss` entirely — the fix matters for the legacy `pipeline.py` path, §10.)
7. **`order_stat` uses mean-field `R(t)` (`segment_sum`+`erf`), not `min(flat_times)`.** Confirmed `:80-81`.
8. **`time_model` ↔ `hit_mode` coupling is real.** `order_stat` needs the per-photon 4-tuple (`make_hits_likelihood`); `moment_t0` needs aggregated `(c,var,t)` (`make_hits_moments:126`). Incompatible shapes → `time_model` lives on `SimConfig`, selects the factory, asserts shape.

**Loop-state hazards to pin as explicit invariants** (review §1):
- **Gradient/residual recompute EVERY step; J/H/Pinv/Minv cached for `refresh` steps.** Don't cache both together.
- **Back-substitution consumes the NATURAL-unit step** (`raw=S·du`), never the scaled `du`. Kill `du` before it escapes the solve, else preconditioned calib mis-scales every nuisance step by `1/S`.
- **`ming` ranks by `‖scale⊙g‖`** (preconditioned metric), not raw `‖g‖` — else energy-blind pick.
- **`bake_k` needs a pre-pass** (`lkv=ΣQ/ΣM` before residuals, `Minv=None`, skip free step — `gauss_newton.py:178-184,206-207,225`), not just a back-sub no-op.

---

## 5. De-env (PREREQUISITE — see §6 sequencing)

`gn_fisher_recon.py:2` does `os.environ.setdefault('TTS_NS','2.5'/'MAXCELL'/'AMP_DETACH'/'OVERLAP_RENORM'='1.009')` at **import**, process-global. **AND unify's own `sensor_response.py:11`** does `_TTS_PERPHOTON_NS=float(os.environ.get('TTS_NS','0'))` at import, then `eff_tts=max(tts,_TTS_PERPHOTON_NS)` (`:214`) — so even after recon stops *setting* it, any `TTS_NS` in the env floors every forward and `dp.response.tts` cannot go below it.

Two `Problem`s in one process collide. **Fix BOTH:** move recon's import-env to construction args (`LossConfig`/`SimConfig`), AND delete `sensor_response.py:11,214`'s `_TTS_PERPHOTON_NS` global + `max(tts,env)`, routing TTS solely through `dp.response.tts` / a `SimConfig` field. Destinations are already specified in `DETECTOR_PARAMS_VS_ARGS.md:70-77`. `OVERLAP_RENORM` is already a real kwarg (`simulator.py:69`) — done; `TTS_NS` is the residual leak.

This is a pure refactor (defaults preserve values) and **unblocks two-Problem coexistence**, which every later test needs (pytest collects all test modules into one interpreter).

---

## 6. Sequenced port plan (reordered — de-env first)

1. **De-env (was step 4).** §5. Zero behavioral change; removes the coexistence blocker that invalidates every later test.
2. **Port `first_arrival_order_stat_nll` + `counts_loss(normalize=)` flag.** §3, §4.6. Now testable beside calibration in one process. Capture the reference `tnll` from `LUCiD_recon` **before** porting (§11).
3. **Config dataclasses:** extend `SimConfig`; add `OptConfig`, `LossConfig` (§2). Drop the ~26 dead constants.
4. **`Problem`/`fit` refactor:** `assemble_normal` (§1.3) collapsing the 4 assemblers + `fit`/`fit_charge_time` into one loop; shared `fd_jacobian`/`readout` (§1.4); `ridge_inverse` gains `eigen_clip`+`ridge_mode`. Extract `CalibProblem` from `build_calibration_problem`, add `build_recon_problem`. **Gate behind the §11 byte-pins** (current toy tests are too loose).
5. **`pipeline.py` disposition** (§10) + **launcher merge** (§7) + doc fixes (§8).

Each step is gated by the calibration regression net (§11).

---

## 7. Harness merge — extend `run_grid.py`, don't author `launch.py`

`run_grid.py` is unify's **only** GPU-pool dispatcher (the other `campaign/*.py` are single-GPU in-process; `lucid/optimization/run.py` is single-process). The genuinely shared pool body is **~12 lines** (`mgpu.py:56-67` ≈ `run_grid.py:80-104`), not 50 — the job builders (range-split+cases vs `grid(PHASE)`) and aggregation share nothing. So:

- Extract a ~15-line `run_pool(jobs, gpus, worker, log_fn)` from `run_grid.py` (hoist its `e.pop('JAX_PLATFORM_NAME')` guard `:90` — absent from `mgpu.py`, a latent CPU-fallback bug). Add a recon job-source alongside `grid(PHASE)`. **Extend `run_grid.py`** (rename to `launch.py` only for the neutral name); do not author a parallel file.
- **JSON sidecar = the real win, do it FIRST (highest value/line).** `gridfit.py:208` already writes JSON; promote `gn_fisher_recon.py:288`'s `np.savetxt` → `json.dump({ev,E,vtx,lon,tra,dir,t0,pol,soff})`. This retires **four already-drifted** stdout regexes (`mgpu.py:30`, `plot_study.py:6` — which already expects fields `mgpu` omits, `plot_dist1s.py`, `plot_recon.py`) for one `json.load`. ~3-line change.
- Fold the triplicated `rms/median/p68/wanderer` stats (`plot_study.py:13-18`, `mgpu.py:33-34`, `plot_dist1s.py`) into one `summarize(rows)`. Merge the **I/O** with `campaign/aggregate.py`; keep the stat bodies separate (CRB σ-ratios vs resolution RMS differ).
- **"Studies as modes" applies only at the launcher layer.** The 1000-event distribution and HIST are env-parameterized invocations + plotters — fold them. The worker's **7 `if MODE/continue` branches** (`gn_fisher_recon.py:224-276`: GRADCHK/CMP/ZESWEEP/ESWEEP/LOSSCHK/...) are distinct programs with per-schema npz + matching plotters — leave as-is, don't churn.

---

## 8. Documentation corrections in this worktree

- `docs/UNIFY_CALIB_RECON.md` / `PLAN_UNIFY.md` carry the reversed **"NO SCALE9"** recipe; `RECO_PIPELINE.md §4` (authoritative) makes SCALE9 preconditioning MANDATORY (raw F → energy freeze, "the session bug"). Record the reversal.
- Recon acceptance target is wrong: unify cites **7-8 cm / 0.26° / 0.26%** (matched-SIREN best case). The realistic bias-limited floor a port must hit is **vtx ~13 cm (lon 12.5/tra 9.2), dir 1.77°, E ~0±24 MeV (2.3%)** (`RECO_PIPELINE.md §8`). Make this a **test assertion** (§11), not just prose.

---

## 9. The joint-fit frontier (the thesis's endgame — scoped OUT of this port)

The unify thesis (UNIFY_CALIB_RECON.md:3-10) is *calib=free DetectorParams, recon=free ParticleParams, joint=free both, over ONE fit*. This design delivers a shared **loop + metric assembler**, which is more than v1 promised. The **param layer is already joint-ready**: `make_optimization_mask` (`detector_params.py:672-708`) recurses any nested NamedTuple by leaf-name — a `JointParams(detector, particle)` pytree + union `trainable_fields` works today with no code change; `normalize_params` is type-agnostic.

What blocks a **literal** joint fit (not this port's goal):
1. **Cross-block metric** `F_{detector,particle}` — needs `assemble_normal` over the *concatenated* Jacobian `[J_det | J_part]`. §1.3's assembler is the right shape but each Problem currently builds only its own column block. A joint Problem would supply the union `residual_and_jac`; the assembler already handles it.
2. **Charge-residual conflict** — calib √-MSE vs recon Poisson on the shared sensors (`RECO_PIPELINE.md §3.1`: √-MSE is *worse* for single-event recon). A joint objective must pick one or run mixed.

**Recommend:** build the trivial `JointParams` pytree now (cheap, documents intent, enables joint masks); defer the joint *fit* until the cross-block metric + residual-conflict are designed. The param substrate is ready; only these two remain.

---

## 10. `pipeline.py` disposition (the #1 v1 gap)

`lucid/optimization/pipeline.py` (the live `lucid-optimize` target via `run.py`) is the legacy **5-stage Adam** recon path running a **pre-finalization loss** (3-term geometric-mean + legacy soft-min `:102` + normalized `counts_loss`). Its fate must be decided — **retire the fit, salvage the seed:**

- **DELETE** `create_combined_loss_function` (`pipeline.py:54-136`) and `run_complete_optimization_adam` (`:139-465`) — superseded by the Poisson+order-stat loss and Fisher-GN. (~600 lines; the biggest single concept removal — Rank 1.)
- **SALVAGE**: the **cone direction search** (stage 2) — the *one* seed the finalized optimizer still needs (`RECO_PIPELINE.md §4/§9.11`); the grid utilities; and `generate_event_data` (`:468-612`, the PhotonSim ROOT load + truth-injection) → these become the recon `StudyConfig`/launcher inputs. (Stages 0/1/3 — energy/position/escan — become dead weight; the Fisher-GN captures those from the whole fiducial.)
- `run.py` becomes a thin shim: build `ReconProblem`, optionally pre-seed direction with the salvaged cone search, call shared `fit()`. Preserves the `lucid-optimize` entry point.

Do NOT "run both alongside" — that institutionalizes the two-loss divergence the unification exists to kill.

---

## 11. Test net (REQUIRED before §6-step-4)

Current calibration tests are **toy analytic forwards + loose tolerance** (`test_fitting.py` recover-to-0.05) — they pin *mechanism*, not the §4 invariants. `test_ridge_inverse_spd` (`:52-56`) only asserts SPD-ness with hardcoded args; adding `eigen_clip`/`ridge_mode` could flip the floor and every toy test still passes. `reference_values.json` touches nothing in `lucid/fitting/`. There is **no recon test anywhere**.

Mandatory additions:
1. **Byte-pin `ridge_inverse(H)`** for a fixed indefinite `H` into `reference_values.json` (`atol=1e-12`) — the function §4.1/§4.2 say silently regresses.
2. **Pin the toy-fit `log_theta`/`k`** to ~1e-6 (not `frac<0.05`) so the default-args path stays byte-stable across the refactor.
3. **One e2e calibration fit** (`@slow`) on the WCTE_like `conftest.py` fixture — the only test that catches the `build_calibration_problem` bridge breaking under nesting.
4. **Order-stat `tnll` value-pin**: `first_arrival_order_stat_nll` reproduces `gn_fisher_recon.perpmt` `tnll` to ~1e-5 on a fixed tuple. **Capture the reference from LUCiD_recon BEFORE porting.**
5. **AMP_DETACH gradient-routing test**: `jax.grad` zero w.r.t. amplitude, nonzero through times (pins §4.5/§4.7 mechanically).
6. **`counts_loss(normalize=False)` pin** both ways (extend `test_losses.py:24`) — guards the §4.6 energy-freeze.
7. **`@slow` floor smoke test**: one full Fisher-GN fit from a 1σ start lands near the **bias-limited** floor (vtx ~13cm/dir ~1.8°/E ~2.3%), NOT the matched-SIREN best case (§8).

---

## 12. Ranked reductions (lines × inverse-risk)

1. **Retire `pipeline.py` loss+Adam** (§10) — ~600 lines, one whole pre-finalization path gone. Risk LOW-MED (only `run.py` imports it; entry point preserved via shim). **Biggest concept removal.**
2. **`assemble_normal` collapses 4 metric assemblers + `fit`/`fit_charge_time`** (§1.3) — ~80+ lines. Risk MED (byte-identity-critical calib core; gate behind §11 pins).
3. **Harness merge + JSON sidecar** (§7) — ~12-15 shared lines + retires 4 drifted regexes. Risk LOW (pure IO). JSON sidecar first.
4. **Shared `fd_jacobian` (3 copies) + `readout`** (§1.4) — ~40 lines. Risk LOW-MED.
5. **`JointParams` pytree** (§9) — near-zero lines now; enables the thesis endgame. Risk LOW (pytree) / HIGH (the actual joint fit — deferred).
6. **`charge_loss` flag** (§2.3) — NOT a reduction (a flag; both √-MSE and Poisson must survive per `RECO_PIPELINE.md §3.1`). Default = calib √-MSE, recon overrides to `poisson`.

---

## 13. Deliberately NOT unified (and why)

- **The residual *form*** (√-MSE vs Poisson) stays inside `residual_and_jac`; `W` is `Problem`-private. `fit()` sees only `(r,J,W)` blocks. (The *metric* IS unified — §1.3 — that's the v1→v2 correction.)
- **recon-only loss knobs** (`amp_detach`, `tot_n_scale`, `time_model`) live on the recon `Problem`/`SimConfig` construction, NOT the shared `LossConfig` top level (they're meaningless for calib; exposing them invites nonsense settings).
- **`pack∘unpack ≠ id` for recon** (sin/cos renorm in `sc2:42`) — the contract must not assert round-trip identity; `fit()` only calls `unpack`.
- **t0 lives in two slots** — global `theta` column (recon) vs per-PMT Schur nuisance (calib). Declared by the `Problem`, never assumed by `fit()`.

---

## 14. Verification addendum (3-agent recon/unify diff, 2026-06-09)

The plan was re-checked against the live worktrees. **Verdict: substantively trustworthy as a port spec** — every load-bearing physics/algorithm claim CONFIRMED in code. Corrections below; the headline is a REFRAME of the work.

### 14.1 THE REFRAME — the forward is ALREADY consolidated
`git merge-base reconstruction-mie unify-calibration` = **`19a282a` = recon's own HEAD**, and unify's first commit **`41096e3`** is exactly the recon-forward patch (same 10 lucid/ files). **Substrate byte-identical**: SIREN net + weights, PhotonSim tables, geometry npz, `config/*.json` all share md5; `config/` is at the same commit `76d5cbe` in both, clean. `simulation/types.py`, `losses.py`, `wavelength/scattering.py` are **byte-identical** between the worktrees (the DiCE `log_p` 6-tuple / `logp_increment` 7-tuple, the `LOG_FLOOR=-60` guard, the HG/Rayleigh score densities are all already in unify). ⇒ **"port the forward" is DONE.** unify MUST be the base (it alone has `simulation/reflection.py`, `wavelength/optical_model.py`, `lucid/fitting/` that the consolidated tree imports). The real consolidation work is **migrating recon's DRIVERS + three decisions**, not merging transport physics.

### 14.2 The 3 genuine forward divergences needing CODE reconciliation (not flags)
1. **`DetectorParams` flat(8-field) → nested(5 sub-tuples) — the headline blocker.** Ripples into every forward access in recon's `simulator.py`/`photon_step.py` AND recon's ~200 untracked driver scripts (`gn_fisher_recon.py`, `recon_*.py`, `loss_*.py`) that build/read flat params. unify ships `from_flat`/`_nest_flat_kwargs` as the bridge — the migration is mechanical but broad.
2. **`photon_step` signature + custom_vjp factory.** recon `(…, wall_rate, sensor_rate, absorption_length, hit_sensor, rng_key, c)` + module-level `@custom_vjp`; unify `(…, refl_params, absorption_length, hit_sensor, lam, rng_key, c, reflection_fn=)` via `make_photon_iteration_update_factors_safe` factory. The 7-tuple OUTPUT is identical (no DiCE reconciliation); the INPUT arglist + vmap `in_axes` + vjp construction differ. Adopt unify's factory form.
3. **SIREN default emitter MISMATCH (a physics-of-gradient decision, NOT a refactor).** recon default = DiCE-score live-resample (`weights=topk·dice_score`, bins re-sampled at the LIVE energy, score-function gradient); unify default = fixed-proposal IMPORTANCE (`weights=topk·p_emit/sg(p_cal)`, bins frozen at E_CAL, pathwise gradient). Same forward at E_CAL, but **different off-calibration emission AND different gradient mechanism**. unify deleted the score path (kept only importance). ⚠️ Much of the recon memory (DiCE sampler, energy↔geometry gradient, basin/Hessian findings) was MEASURED on the score path → consolidation must either re-validate recon under unify's importance default OR re-introduce the score emitter as an opt-in source-mode (not the deleted env path).

Plus: recon's env-gated diagnostics (`PSTEP_DBG` ×7, `T0_LIVE`, `NORM_GRAD_ITERS`, `DATA_CHG_NOTIME`/`LIK_CHG_NOTIME`, `OVERLAP_ANALYTIC=erf/logistic`, `SIREN_*` experiments) are byte-identical-when-default but must be **dropped, not merged** (unify already dropped them at `41096e3`; recon's working tree re-accreted them).

### 14.3 Citation corrections (verified line numbers)
- **DRIFT** `SCALE9` is `gn_fisher_recon.py:40` (doc says :37). Import-time `os.environ.setdefault(...)` block is `:11` (doc says :2) — keys/values (`TTS_NS=2.5,MAXCELL=4,AMP_DETACH=1,OVERLAP_RENORM=1.009`) confirmed.
- **DRIFT** unify fitting lines moved (as the doc warned, from my recent polyak/bake_k/fit_charge_time edits): `fit` assembler now `:198-205`, polyak `:231-237`, bake_k pre-pass `:178-184`/`:206-208`, `fit_charge_time` Schur `:340-341`. `sqrt_residual:31-33`, FD `:112-121`/`:264-274`, `fisher.py:62-76`, `problem.py:42/84-91` are dead-on.
- **REFUTE (matters for §6-step-2 / §11.6):** `counts_loss`/`poisson_nll` have **NO `normalize` parameter** — they ALWAYS divide by Σtrue (`losses.py:310-311`, `poisson_nll=counts_loss` `:511`). The flag must be **ADDED**, not toggled.
- **REFUTE/clarify §5:** the `eff_tts=max(tts,_TTS_PERPHOTON_NS)` form is in **unify**'s `sensor_response.py:214` (confirmed) but **recon**'s (dirty mie-branch) version puts the leak at `:11`+`:67-68` (`if _TTS_PERPHOTON_NS>0: photon_times += normal()·_TTS_PERPHOTON_NS`). De-env recommendation stands; the two files differ — fix both, by location.
- **OVERSTATEMENT:** `setup_event_simulator` is **26 args (25 named + `**grid_params`), not ~30**; there is no `TTS`/`time_model` kwarg on it (TTS enters via `dp.response.tts` + the env leak only).

### 14.4 The biggest UNMENTIONED risk
The recon worktree is **mid-Mie-development**: `reconstruction-mie` HEAD is Mie work, and `lucid/{losses,sensor_response,simulator,detector_params,wavelength/scattering}.py` carry **uncommitted edits**, while the authoritative drivers (`gn_fisher_recon.py`, `RECO_PIPELINE.md`, `recon_*.py`) are **git-untracked**. So §0's "substrate byte-identical" is true for the COMMITTED config/SIREN/geometry but the recon working tree's optical-path files are in flux. **Pin a recon commit (or snapshot the untracked drivers) before porting**, and decide whether the in-flight Mie edits to `losses.py`/`sensor_response.py` must fold into the port.
