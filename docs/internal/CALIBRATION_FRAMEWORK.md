# Calibration framework — inventory + consolidation + generalization

## 1. What we have (268 .py, ~60 .md, 219 truth.npz)

The files fall into three tiers:

**A. CORE calibration stack (~25, keep) — the validated 06-06/06-07 lineage:**
- Engines: `wlt_engine.py` (SUPERSET: per-photon-λ + complex Fresnel reflection + timing + charge),
  `wl_engine2.py` (curves+reflection, no timing), `refl_engine2.py` (Fresnel helpers), `sample_engine.py`
  (exact shot-noise), `src_engine.py` (source nuisances). `implicit_engine.py`/`wl_engine.py` = earlier rungs.
- Generators: `gen_wl2.py`, `gen_refl2.py`, `gen_src.py`, `chargedata.py`, `make_noisy.py`, `make_chargemean.py`.
- Fitters: `gn_wl2.py` (flexible curves+k Schur), `gn_refl2.py`, `gn_src.py`, `charge_split.py` (variance→QE/gain/w),
  `timingcal.py` (t0/TTS/walk), `fullcal.py` (all per-PMT vs scale), `gn_fast.py` (cached-J Schur recipe).
- CRB: `fisher_wl2.py`, `fisher_timing2.py`, `fisher_multi.py`, `crb_vs_n.py`.
- Plots/diag: `wl2_curves_plot.py`, `refl2_plot.py`, `charge_split_plot.py`, `timing_break_plot.py`,
  `fitvsn_plot.py`, `fullcal.py` fig, `t1_diag.py`, `profile_ct.py`, `refl_check2.py`.

**B. Superseded fitters/fisher (~32, archive):** gn_LM/comb/cross/full/k/multi/prof/ratio/schur/seq/simple/xfix,
fisher_LM/cross/full/obs/occupancy/firstarrival, stage_a/c — the earlier GN/CRB variants, now subsumed by the spec above.

**C. Exploratory investigation (~86 + ~100, archive):** step1–12, score_*, sk_score_*, prototype_* (the DiCE/score
methodology study — CONCLUSIONS now in memory + DESIGN_GENERAL.md/GENERALIZATION.md), diag_500/480/400/7p* (the
single-λ landscape diagnostics), multi_joint*, optim_newton*, toy_*, bench_*, sweep*, test_*, validate_*, params_*.

Plus **219 truth.npz** (mostly stale) and **~60 docs** (findings already distilled into the memory files).

## 2. The structural problem: feature-accretion duplication

The core stack grew by COPYING the previous engine/fitter and adding ONE feature:
`implicit → wl(λ amplitudes) → wl2(λ-curves + complex reflection) → wlt(+ timing + charge)`. Likewise ~6 fitters
(`gn_wl/gn_wl2/gn_refl2/gn_src/charge_split/timingcal/fullcal`) are the SAME GN+constrained-Schur recipe with a
DIFFERENT hardcoded `asm()` param layout + a different observable. Same for ~3 fisher modules. So every new physics
piece spawned a near-copy. `wlt_engine.py` is already the engine superset; the rest is recoverable by configuration.

## 3. The generalization: a config-driven calibration package

Factor the common machinery into ONE framework, mirroring LUCiD's own architecture (composable physics configs +
registry dispatch). Proposed `lucid/calibration/` (promote out of the mie_hunter sandbox):

- **`engine.py`** — ONE composable differentiable forward (= `wlt_engine` generalized): toggle wavelength /
  reflection-model / timing / charge-smearing; outputs a dict `{charge, charge_var, t_first}`; detector via LUCiD's
  **geometry registry** (sphere/box/string/cylinder + measured-PMT `.npz`), NOT a hardcoded SK cylinder.
- **`params.py`** — a DECLARATIVE parameter registry (THE key abstraction). Each parameter declares:
  `kind` (global-scalar | global-curve | per-PMT), `transform` (log/linear), `bounds`, `gauge`
  (mean-log-zero for k/QE/gain, mean-zero for t0), `prior` (curvature for curves), and which `observable(s)`
  constrain it. This replaces every bespoke `asm()`. Example row:
  `Param('L_abs', curve, lams=[380..460], ref='pure_water', log, prior='curvature', obs=['charge'])`,
  `Param('m_M', curve, obs=['charge','timing'])`, `Param('gain', per_pmt, gauge='mean_log0', obs=['charge_var'], marginalize=True)`,
  `Param('walk', per_pmt, obs=['timing_multilevel'])`.
- **`observables.py`** — charge-mean (Poisson ∝Nphot), charge-variance (compound-Poisson ∝flashes, the QE↔gain
  splitter), timing-first-arrival (Gaussian σ=TTS/√occ, the scattering/reflection lever). Each declares its residual +
  noise model + Fisher weight.
- **`fit.py`** — ONE GN + constrained-Schur fitter: reads the param-spec + active observables, Schur-marginalizes the
  declared per-PMT nuisances, applies priors/gauges, cached-J refresh. Replaces ALL `gn_*`.
- **`fisher.py`** — ONE CRB/identifiability module (per-param σ, constrained-DOF, vs-scale, the √12 honesty factor).
  Replaces ALL `fisher_*`.
- **`sources.py`** — declarative source layouts (lasers/iso at given λ/position/intensity; the multi-intensity ladder
  for the TQ map). Adopt SK's actual λ set (337/375/398/405/445).
- **`config.py` + JSON** — a calibration config (detector + sources + which params free/fixed + observables + budget);
  `run_calibration(cfg)` does generate→fit→CRB→report. Mirrors `setup_event_simulator(json)`.

This turns "add a new calibrated quantity" into "add one row to `params.py`" — no new engine, fitter, or fisher copy.

## 4. What's NEEDED to be truly general / production

1. **Detector-agnostic** — replace the hardcoded SK cylinder with LUCiD's geometry registry (already exists: sphere/
   box/string + `Cylinder.from_pmt_file`). The calibration should run on HK/WCTE/IceCube geometries unchanged.
2. **Any param global-OR-per-PMT** — w and TTS are global today; SK has per-PMT TTS. The spec should allow promoting
   any scalar to per-PMT (and the Schur handles it).
3. **Unified JOINT fit** — the capstone was STAGED (closed-form mean→var-split→timing). The general fitter should do
   ONE joint GN over all params + all observables (captures cross-correlations the staging ignores; the Fisher already
   shows charge+timing must be solved together to break degeneracies).
4. **TQ-map / per-PMT timing made first-class** — currently bolted on; declare `t0`, `walk`, per-PMT `TTS` as per-PMT
   timing params with the multi-intensity observable. Keep the LOW-occupancy requirement for the walk.
5. **Real-data adapter** — truth = simulated today. Add ingestion of real PMT (charge, time) hits from LUCiD's
   `hits/`/`sensor/` HDF5 so the same framework calibrates REAL detector data, not just self-consistency.
6. **Time-/position-binned calibration** — the dominant real systematic is time-varying & z-dependent optics (water
   convection; SK's 20–40% absorption drift). Support binning the calibration by run-time and z so θ(t,z) is tracked.
7. **Always report CRB-vs-realized** — every run emits the Fisher bound AND the realized fit with the √12 caveat
   (the engine is ~√12 quieter than real Poisson — quote the bound, not the toy-MC scatter).
8. **Wavelength-dependence audit** — fix only first-principles shapes (Frank-Tamm 1/λ², leading Rayleigh 1/λ⁴, glass
   dispersion, n_imag); free the rest as laser-anchored curves (already the principle; make it the default in the spec).

## 5. Housekeeping
- `mie_hunter/archive/` for tiers B+C (~240 files); keep tier A active. Conclusions already in memory + docs.
- Prune 219 truth.npz to a handful of canonical ones (regenerable from configs).
- Collapse ~60 docs → this file + `refs/` + the memory index; the rest are session logs.

## One-line takeaway
We have a complete, validated set of calibration CAPABILITIES (optical λ-curves, complex reflection, per-PMT QE/gain,
SPE width, t0/TTS, TQ map; charge-mean + charge-variance + timing observables; CRB + vs-scale) spread across ~25 core
files that are 5× duplicated by feature-accretion. The generalization is a single config-driven `lucid/calibration/`
package: composable engine + declarative param registry + observable models + one GN/Schur fitter + one Fisher module.
The gaps to production are detector-agnosticism (use LUCiD's registry), a unified joint fit, real-data ingestion, and
time/position-binning for the convection systematic.
