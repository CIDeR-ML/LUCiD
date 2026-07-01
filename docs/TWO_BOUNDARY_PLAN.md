# Two-boundary / two-medium extension plan (JUNO-like: scintillator inside, water outside)

Status: **IMPLEMENTED on branch `feature/two-boundary-nested-sphere`** (2026-06-22).
Scoping + decisions below; implementation summary at the very bottom (§7).

This document scopes extending LUCiD from a single homogeneous medium bounded by one
outer surface to a **nested two-medium detector**: an inner liquid-scintillator (LS)
sphere surrounded by a water buffer, with PMTs on the outer surface — i.e. JUNO.

---

## Decisions locked (2026-06-22)

These override the generic discussion below where they conflict; the rest of the plan is
written generically and these narrow it.

1. **Goals: forward / data-gen + calibration + reconstruction** (all three). The interface
   must therefore work in **both** the sampling path (data oracle) and the expected-value /
   DiCE path (unbiased grads for calib + recon).
2. **Fidelity: generic engine, WbLS-first → LAB-LS.** Build in (n_in, n_out, per-medium
   optics); validate on the benign WbLS≈water interface first, then raise contrast to real
   JUNO LAB-LS (n≈1.48).
3. **Absorption: pure absorption only** (no re-emission). Accepted limitation: for LAB-LS a
   fitted `L_abs(LS)` is an **effective** attenuation length (absorption+re-emission folded),
   **not** microscopic absorption — label it as such; it will disagree with a microscopic
   Geant4 absorption length. Fine for WbLS (re-emission negligible) and for effective-
   attenuation calibration.
4. **Geometry: nested spheres only** (hard-coded JUNO case; reuse Sphere machinery). No
   general multi-region framework.
5. **Fit scope: bulk optics fittable, interface n a fixed input.** Two `L_abs` + two
   scattering sets are calibration targets; n_in, n_out are **fixed (not fit targets)** —
   but keep them as a possibly **λ-dependent input** `n(λ)` (reuse each medium's
   `refractive_index_curve`), since a Cherenkov-spectrum source sees dispersion at the
   boundary. Don't hard-code a scalar n. ⇒ Snell/Fresnel need **not** be differentiable
   w.r.t. n — only forward-correct and differentiable w.r.t. the photon trajectory (for
   source-position recon) and a clean pass-through for the bulk-optics grads.
6. **Acrylic: lumped into the interface** (two media, one LS→water boundary; acrylic
   thickness/absorption ignored for now).
7. **Outer boundary: idealized.** PMTs on the outer sphere at R_out + a single tunable
   `wall_reflection_rate` (reuse the current Sphere sensor model). No steel-truss buffer
   offset, no inward-facing acceptance, single PMT type for now.
8. **Sources: point sources only (no SIREN for now).** Use the existing isotropic / laser
   calibration sources (`sources/calibration_sources.py`), born inside R_in. This removes the
   Cherenkov-cone-in-LS medium-dependence and the LAB-LS SIREN-retraining dependency
   entirely. SIREN track sources are a later, separate add-on.

**Net effect on scope:** Phase 5 (sources) collapses to "init `medium_id` on existing point
sources." §4.4's Cherenkov/SIREN scaling concerns are **deferred** (no tracks). The interface
(Phase 3) stays the core, but with n fixed it need not be differentiable through n — a real
de-risk. The biggest physics caveat is the accepted pure-absorption / effective-L_abs framing.

---

## 0. What JUNO actually is (sets the fidelity target)

Real JUNO:
- **~20 kton LAB-based liquid scintillator** (linear alkylbenzene), n ≈ **1.48–1.49**,
  in an acrylic sphere of inner diameter 35.4 m (**R_in ≈ 17.7 m** — the existing
  `config/JUNO_wbls_geom_config.json` radius of 17.5 m is already in the right place).
- **Ultrapure water buffer**, n ≈ **1.33**, outside the acrylic.
- ~17.6k 20-inch + ~25.6k 3-inch PMTs mounted in the water at **R_out ≈ 19.5 m**.
- A 12 cm **acrylic** vessel (n ≈ 1.49) between LS and water — really a *third* medium.

Implications:
- **JUNO is the high-contrast case.** Δn ≈ 0.15 ⇒ critical angle ≈ 64° ⇒ strong Fresnel
  reflection and total internal reflection (TIR) at the interface. Interface optics
  (Phase 3 below) is the **main event**, not a finishing touch.
- The repo's `JUNO_wbls` config is **JUNO geometry with WbLS material** — a physical
  misnomer. WbLS (n ≈ 1.34, ~10³ ph/MeV) is a THEIA/R&D target, **not JUNO**. WbLS is
  the *benign-interface* case (Δn ≈ 1%); useful as a first validation milestone but not
  the physics goal.
- Acrylic: lump into the interface for a two-boundary first cut; add as a thin third
  shell later if needed.

**Recommendation:** build the engine generically in (n_in, n_out, L_abs per medium, …),
validate first on the WbLS≈water benign case (interface nearly transparent — isolates
the geometry/region plumbing), then turn the contrast up to LAB-LS values.

---

## 1. Why this is architectural, not a config tweak

The transport engine assumes **one homogeneous medium bounded by one outer surface.**
That assumption is load-bearing in four independent places:

| Assumption | Where (file:line) | What a 2nd medium breaks |
|---|---|---|
| One ray cast → the **outer** boundary per step | `propagation/sphere.py:57` (`t=max(t1,t2)`, "exit point from inside going outward"); `propagate_fn(pos,dir)` called once per step in `simulator.py:543` | a photon in the water shell must be able to hit the **inner** interface; a photon in LS must stop at the interface, not the PMT surface |
| Optical lengths **per-photon, computed once before the loop** from one medium | `_get_optical_arrays`→`evaluate_optical_model` (`wavelength/optical_model.py:65`); arrays passed into the scan as constants (`simulator.py:500`) | scatter/abs lengths must switch when the photon changes region |
| **One** `SPEED_OF_LIGHT_MATERIAL` scalar drives every time update | `simulator.py:227, 624, 638` | time-of-flight differs in LS vs water (different n) |
| Leaving the outer boundary **kills** the photon; only boundary events are wall-reflect / sensor-detect | `get_inside_detector_flag` (`simulator.py:626`); no transmission channel in `simulation/reflection.py` | the inner interface must **refract + Fresnel-split**, not kill or wall-reflect |

Reusable as-is (no structural change): SIREN/scint **sources** (`sources/siren_rays.py`,
`sources/scintillation_photons.py` — they just emit origin/dir/λ/weight), the DiCE-forward
**differentiability** machinery, the **sensor-response / hit-mode** layer
(`simulation/sensor_response.py`), and per-step time accumulation in `PhotonState`
(already supports a per-step speed; it just needs the speed to become medium-dependent).

Work concentrates in **geometry, the per-step kernel, and the param/config
representation** — not sources or readout.

---

## 2. Target geometry

Nested sphere:
- **R_in** — inner sphere = LS↔water interface (no sensors).
- **R_out** — outer sphere = PMT surface + reflecting wall.
- Inner medium = LS (Cherenkov + scintillation, n_LS); outer = water (Cherenkov only, n_water).
- Photons born inside R_in must cross the interface to reach PMTs at R_out.

---

## 3. Design — phased

### Phase 0 — Regression tripwire (do first)
Lock single-medium behavior with a byte-identical fixture (forward hits **and** an
AD-grad / Fisher tensor on the `JUNO_wbls` single-sphere config). Every phase below must
keep the degenerate case (R_in→0, or inner≡outer material + equal n) **bit-identical**.
This gate proves the extension is a strict superset.

### Phase 1 — Geometry: a `NestedSphere` detector
- New registry entry `@register_detector('nested_sphere')` (`geometry/registry.py`),
  mirroring `geometry/sphere.py` but carrying `inner_radius` + `outer_radius`. PMT
  placement / grid / sensor-map reuse the existing Sphere machinery **at R_out** (already
  radius-parameterized: `sphere.py:487`).
- **The one new geometry kernel:** `intersect_nested_spheres(origin, dir, R_in, R_out)`
  → `(t_hit, surface_id∈{INNER,OUTER}, normal)`. Two quadratics (reuse
  `intersect_sphere`, `sphere.py:18`): take the smallest forward `t` across both spheres'
  positive roots, tag the surface, emit the inward/outward normal. Differentiable, ~30 lines.
- Geom config gains `inner_radius`, `outer_radius`, `inner_material`, `outer_material`
  (vs today's single `radius`/`material`).

### Phase 2 — Per-photon region state + per-medium optics
- Add `medium_id` (int) to `PhotonState` (`simulation/types.py`); initialize from
  `‖origin‖ < R_in` in the source-wiring branch (`simulator.py:836+`).
- Precompute optical-length arrays **once per medium** (call the optical model twice —
  inner, outer) and select per step: `scatter_len = jnp.where(medium_id==0, sl_in, sl_out)`,
  same for mie/abs and `speed_of_light` (length-2 lookup by `medium_id`). Wavelength
  evaluation stays outside the loop — *unless* re-emission is on (see §4.2).

### Phase 3 — The interface step (physics core; dominant for JUNO)
Add an "interface" outcome to the surface kernel (`simulation/photon_step.py`), parallel
to detect/reflect/scatter/absorb. When a step's nearest surface is `INNER`:
- Incidence cosine → **Snell** refracted direction, **Fresnel** reflectance `R(θ,λ)`, and
  **TIR** when LS→water past the critical angle (~64° at LAB-LS). Reuse the real→real
  Fresnel helper `fresnel_rr` already in `reflection.py`. **The interface must work both
  directions** (LS→water *and* a back-scattered photon water→LS — no TIR going into the
  denser medium).
- **Decision: SAMPLE the branch** — `Bernoulli(T)` per hit, follow the chosen ray
  (transmit: switch `medium_id`, refracted dir; reflect: keep `medium_id`, specular dir) at
  full weight. This is the only single-ray scheme that handles **TIR** correctly (a
  deterministic transmit-and-drop-reflection would delete all light at/beyond the critical
  angle). The choice rides the **same DiCE `logp_increment` score channel** the existing
  scatter/reflect choices use (`simulator.py:653`) ⇒ unbiased grads.
- **Consequence:** the expected-value path becomes **stochastic at interfaces** (a departure
  from its otherwise-deterministic low-variance weights — accepted). Bulk-optics gradients
  (the calibration targets L_abs, scatter) still flow **pathwise and exact** for each sampled
  trajectory — the interface sampling only adds *trajectory* variance, exactly like the
  existing scatter sampling; it does not bias the L_abs gradient. Use **both** engines'
  interface as sampled (unifies the sampling and expected-value paths at the boundary).
- Interface deposits **no charge**, consumes one scan iteration; TIR can trap a photon into
  many bounces inside the LS sphere ⇒ **K must grow** beyond the single-medium 7 (budget
  ~12–16, set by the Phase-5/gate-5 K-convergence sweep, which now must include TIR-trapped
  trajectories).

### Phase 4 — Params / config representation
- Extend `DetectorParams` (`detector_params.py`) to hold **two named optical sub-bundles
  (inner / outer)** — *not* a general N-media region axis (decision: keep it concrete; the
  deferred acrylic 3rd shell will touch this structure later, accepted). Both bundles are
  **fittable leaves** (calibration must optimize either medium's bulk optics). Interface
  **n_in/n_out are fixed** (decision 5) — carried in the static geometry/medium config,
  **not** as `DetectorParams` leaves, so no grad path through n.
- Physics config carries two `medium_model` paths; loader (`detector_params.py:611`)
  resolves both.

### Phase 5 — Sources & readout (point sources — nearly free)
- **Point sources only** (decision 8): reuse `IsotropicSource` / `LaserSource`
  (`sources/calibration_sources.py:241`), placed inside R_in; init `medium_id` from the
  origin radius. No SIREN, no Cherenkov cone, no scintillation surrogate in the engine
  bring-up ⇒ the source side is medium-independent and trivial.
- **Spectrum is a pluggable knob, not a fixed choice.** The isotropic/laser point source
  takes a wavelength sampler that can be **monochromatic** (single λ) **or a Cherenkov 1/λ²
  spectrum** — and the seam stays open for arbitrary spectra later (scintillation Moyal,
  measured lamp spectra, …). Do **not** hard-code monochromatic. Concretely: run in
  `wavelength_mode`, let the source emit per-photon λ from its chosen sampler (reuse
  `sample_cherenkov_wavelengths` for the 1/λ² case), and ensure that per-photon λ is carried
  through **both** media's λ-dependent optics (Phase 2 must select per-medium *coefficients*
  but evaluate them at the photon's own λ). This is the main reason Phase 2 keeps λ in the
  per-photon arrays rather than collapsing to scalar lengths.
- **Calibration source-position scan:** to break the new L_LS↔L_water degeneracy (§4.4),
  place the point source at **several radii** inside R_in (and optionally in the water shell)
  so the d_LS:d_water path ratio varies — directly analogous to the source-diversity result
  in `mie_hunter/`. Monochromatic lasers anchor λ-points; a 1/λ² source probes the
  band-integrated response — both should be supported.
- Sensor response / hit modes unchanged — timing already accumulates per-step, so
  medium-dependent speed flows through once Phase 2 lands.

---

## 4. Absorption, weights, and scaling — what must be right

This is the part most likely to silently bias results. Four distinct issues.

### 4.1 Piecewise attenuation — automatic if optics are per-medium-per-step
Survival is multiplicative (`survival = state.survival * safe_continuing`,
`simulator.py:629`), so a photon going d_LS in LS then d_water in water gets
`exp(−d_LS/L_LS)·exp(−d_water/L_water)` **for free** — *provided* each step selects its
medium's `L_abs` (Phase 2). No special handling. Easy.

### 4.2 The big one: in LAB-LS, absorption ≠ removal — it is absorption + RE-EMISSION
JUNO's >20 m effective attenuation length exists *because* most absorbed photons are
**re-emitted** (Stokes-shifted to longer λ, isotropic, time-delayed). LUCiD today models
absorption as pure weight decay toward zero, and treats λ as terminal
(`siren_rays.py`, `scintillation_photons.py`). Consequences for **the weight model**:
- Re-emission **regenerates** weight rather than destroying it ⇒ modeling LS as pure
  absorption will **systematically under-light the PMTs and bias L_abs(LS) calibration.**
- It **changes λ mid-flight**, and optical lengths are λ-dependent and currently
  precomputed once *outside* the scan loop. Re-emission breaks that optimization ⇒ λ must
  live in `PhotonState` and optics get re-interpolated after a re-emission.
- Faithful model: with prob p_reemit, instead of killing weight, resample λ from the LS
  emission spectrum, randomize direction (isotropic), add an emission-time delay, continue
  with weight retained; the (1−p_reemit) fraction is true absorption (weight lost). This is
  a new discrete event ⇒ another DiCE score term.
- Cheap fallback (loses λ-shift, direction randomization, timing tail): fold re-emission
  into a longer **effective** L_abs. Acceptable for a first WbLS milestone; **not**
  acceptable for quantitative JUNO energy/vertex/time reconstruction.
- **This is a separate architectural item from the geometry work** and is the single
  biggest physics fidelity gap for JUNO.

### 4.3 Fresnel weight split must conserve flux — and do NOT add an n² factor
At the interface, weight splits transmit×T / reflect×R. **R+T=1 must hold exactly** or the
engine creates/destroys charge ⇒ biased. Classic ray-tracer trap: with a per-ray weight
scheme, **Snell (direction) + Fresnel T (flux) already conserve everything** — adding an
explicit n²/radiance/étendue scaling on top **double-counts and is wrong.** Conserve flux
via T; let Snell handle direction; no separate n² factor. (Call this out in the kernel.)

### 4.4 Scaling
- **Cherenkov / SIREN scaling — DEFERRED** (decision 8: point sources, no tracks). The
  medium-dependent Cherenkov angle/threshold/yield and the LS-trained-SIREN requirement only
  apply when track sources are added later. Point sources emit independently of n, so none of
  this is in scope for the engine bring-up. (Kept here as a flag for the future SIREN add-on.)
- **New calibration degeneracy (in scope):** PMT charge constrains `d_LS/L_LS + d_water/L_water`
  jointly ⇒ L_abs(LS) ↔ L_abs(water) **degenerate**, broken only by **source-position
  diversity** (different source radii ⇒ different d_LS:d_water ratios). Same structure as the
  k↔L_M / source-diversity findings in `mie_hunter/`. Characterize this CRB before trusting a
  joint two-medium absorption fit; the Phase-5 source-radius scan is the lever.

---

## 5. Validation / regression gates

**Decision: internal consistency only — no external Geant4 truth.** Accepted risk: a
shared assumption between forward and "truth" can't be caught, so the **analytic
Snell/Fresnel/TIR + flux gates (3) do the heavy lifting for interface-physics correctness**
and are non-negotiable; the degenerate/invisible-interface gates only validate the *plumbing*
(that two-media reduces to one-media), not the physics.

1. **Degeneracy (byte-identical):** inner≡outer material, n_in=n_out ⇒ reproduce
   single-sphere forward + grads exactly (Phase 0 fixture).
2. **Invisible-interface:** equal n + equal optics, R_in<R_out ⇒ photon/charge
   distribution matches a single sphere of radius R_out.
3. **Snell / Fresnel / TIR** unit tests vs closed form; **R+T=1** flux conservation across
   incidence angle. *(Primary physics-correctness gate, since there's no external truth.)*
4. **AD vs FD gradients** for both media's optical lengths — including a DiCE-unbiased check
   on the new sampled transmit/reflect interface choice. (This repo has repeatedly found
   discrete-choice gradient bugs — treat as first-class.)
5. **K-convergence** sweep (charge/time vs K), **including TIR-trapped trajectories**, to set
   the new step budget.
6. **Two-medium absorption-split CRB** (§4.4) — confirm source-radius diversity breaks the
   L_LS↔L_water degeneracy before quoting a joint fit.
7. *(Out of scope by decision — external Geant4/PhotonSim cross-check deferred. Revisit only
   if internal gates prove insufficient.)*

---

## 6. Effort / risk
- **Low risk, high reuse:** geometry (P1), region-state + per-medium optics (P2),
  params/config (P4), sources/readout (P5) — extensions of existing parameterized code.
- **Real engineering + physics risk:** Phase 3 (interface refraction/Fresnel/TIR inside
  the differentiable scan with unbiased grads + K budget) **and** §4.2 re-emission (weight
  regeneration + mid-loop λ-dependent optics re-eval). For JUNO both are essential.
- **Adjacent fidelity items (flag, not strictly "two boundaries"):** LS absorption/
  re-emission (§4.2), an LS-specific scattering model, realistic `cherenkov_fraction`,
  LS-trained SIREN, and (later) the 12 cm acrylic as a third thin shell.

---

## 7. Implementation summary (2026-06-22, branch `feature/two-boundary-nested-sphere`)

All phases landed and tested. Commits: Phase 1 (021b30e), Phases 2/3/5 (d7552b4),
Phase 4 + Phase 0 (aebee12).

**Phase 1 — geometry.** `lucid/geometry/nested_sphere.py::NestedSphere(Sphere)` (inner +
outer radii, sensors on the outer sphere, `region_of()` for medium init).
`lucid/propagation/nested_sphere.py`: `intersect_two_spheres_forward` (nearest forward
surface among the two spheres, both directions) + `create_nested_sphere_propagator` (wraps
the validated single-sphere sensor lookup at R_out, overrides positions/normals and masks
sensor weights for interface-hit rays, adds a per-ray `hit_interface` flag).
`DetectorGeometry.from_config` builds inner+outer media and the nested propagator.

**Phases 2/3/5 — transport.** `PhotonState` gains per-photon `medium_id` (None for
single-medium → carried unchanged). `photon_step.py`: `_interface_refract_reflect`
(Snell + Fresnel via `fresnel_rr` + TIR, transmit/reflect **sampled** Bernoulli(T),
DiCE-scored) and dedicated nested steps (sample + expected-value) returning the updated
medium id; single-medium steps untouched. `simulator.py`: an `_IS_NESTED`-gated branch
selects per-photon optics + speed by `medium_id`, threads `hit_interface`, evaluates both
media's optics at the same per-photon λ; calibration point sources init `medium_id` from
the emission radius. SIREN config is now loaded only in track mode (calibration-only
materials need no `siren_params.json`).

**Phase 4 — per-medium fittable optics.** `DetectorParams.outer_optics: OuterOptics | None`
(last field, default None → zero leaves → single-medium byte-identical; not in
`_SUBTUPLES` so flat save/load/normalize ignore it). `dp.with_outer_optics()` exposes the
outer medium's scatter/absorption leaves; gradients verified to flow per-medium. Follow-up:
wire these leaves into the optimizer driver's bounds/normalize helpers.

**Materials.** `config/materials/labls.json` (real JUNO LAB-LS) with SOURCED optics —
Rayleigh L=28.2 m @430 nm (arXiv:1504.01001, pure λ⁻⁴), n=1.48, Mie=0 (purified,
Rayleigh-dominated), effective L_abs=68.8 m (JUNO attenuation ≥20 m ⊖ Rayleigh 28.2 m;
the pure-absorption fit-target prior, NOT a fabricated spectrum). Configs:
`config/JUNO_nested_geom_config.json` (WbLS/water) + `JUNO_nested_labls_*` (LAB/water).

**Validation (`tests/test_nested_sphere.py`, 22 tests).** Two-sphere intersection (both
directions); analytic Snell/Fresnel R₀/TIR + flux; invisible-interface (wbls/wbls) matches
a single sphere to 0.02%; high-contrast LAB/water shows TIR trapping ~21% of the light +
later arrivals; Phase-0 leaf-count + propagator-key separation. Single-medium regression:
92-test targeted run + params suite green (byte-identity preserved).

**Known limitations (accepted decisions / flagged).** Pure absorption — no re-emission
(LAB `L_abs` is effective); LAB absorption spectrum is a flat prior (no literature curve);
nested spheres only; idealized outer wall; point sources only (no SIREN tracks);
interface n fixed (not fit). The soft-`temperature` sphere overlap deposits ~0 charge for
the JUNO sensor geometry — pre-existing, orthogonal; calibration uses `temperature=None`.
