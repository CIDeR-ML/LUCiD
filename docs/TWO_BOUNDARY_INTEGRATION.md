# Two-boundary → main: integration & generalization assessment

Status: assessment / plan (2026-06-24). Branch `feature/two-boundary-nested-sphere`.
Companion to `docs/TWO_BOUNDARY_PLAN.md` (the build) — this is about **merging it into main
generally and systematically**.

## 0. Key framing: the branch holds TWO orthogonal changes

They got entangled during development but are independent and should be generalized and
landed **separately**:

| | A. Sensor angular acceptance (cosθ) | B. Two-medium transport (nested) |
|---|---|---|
| Nature | base-model **physics correctness fix** | new **feature** |
| Byte-identical to main? | **No** — changes sphere forward at oblique incidence | **Yes** for single-medium (gated by `_IS_NESTED`) |
| Touches | `propagation/base.py`, `shared.py`, `sphere.py`, `geometry/sphere.py` | `geometry/nested_sphere.py`, `propagation/nested_sphere.py`, `photon_step.py`, `simulator.py`, `detector_params.py`, `types.py`, `detector_geometry.py` |
| Orthogonal to | media | sensors |

Treating them as one merge couples a cross-cutting correctness change to a research feature.
**Recommendation: split into PR1 (A) and PR2 (B), A first.**

---

## 1. Change A — the cosθ angular-acceptance fix

### What it is
`compute_sensor_intersections_base` modeled a PMT as a 3-D sphere with a θ-independent π·r²
capture cross-section; the physical photocathode cap is π·r²·cosθ. The fix multiplies the
capture weight by cosθ (ray vs the sensor's outward normal). It restores detection
conservation (detected fraction = sensor coverage, independent of source position; was
inflating to 1.7× near a sphere wall, 1.4× near a cylinder wall).

### The problem with how it landed: it's **inconsistent**
- It is wired ON for **sphere / nested_sphere only** (via `Sphere.sensors_radial=True`, a
  radial-normal shortcut). It is **OFF for cylinder and box**.
- ⇒ After this branch, **two PMT models coexist** in main: spheres have the physical
  angular acceptance, cylinders/boxes don't.
- **Production is cylinders** (SK / HK / WCTE). So the fix does **not** fix the production
  geometry — those still over-detect grazing rays by ~1.4×. It only fixes the JUNO research
  sphere.

### The systematic version
Make it **geometry-agnostic and physical**:
1. Every `Detector` exposes **per-sensor outward normals** (`sensor_normals`, shape
   `(n_sensors, 3)`). Sphere = radial; cylinder = radial-in-xy on the barrel, ±ẑ on the
   caps; box = face normals. (Cylinder/box normals already exist at the geometry level —
   `propagation/cylinder.py`, `box.py` — they just need to be exposed per sensor.)
2. The capture applies a **pluggable angular-acceptance function** `accept(cosθ)` uniformly
   (default `cosθ`; leaves room for a measured angular-QE curve later). The current
   `apply_radial_cos` flag becomes a special case (`sensor_normals = radial`).
3. Gate on the **conservation test for every geometry** (detected fraction flat = coverage).

This **subsumes** the sphere-only hack, fixes the production cylinders, and is the right
place for real PMT angular response.

### Implications to manage
- **Not byte-identical**: sphere (and, once generalized, cylinder/box) absolute charge
  drops at oblique incidence. ⇒ **QE / energy-scale calibration shifts** — any fit that
  absorbed the over-detection into QE or energy will move. **Re-validate the calibration/
  recon floors** after. (Good news: no test asserts golden sphere/cylinder charge
  magnitudes, so the suite won't silently lock in the old bias.)
- It is **inert at normal incidence**, so central-source results barely move.
- Regression gate already exists: `studies/sphere_detection_conservation.py` (extend to
  cylinder/box).

---

## 2. Change B — two-medium transport, and its generalization gaps

The feature works and is validated, but is **hardcoded to the binary nested-sphere case**.
Generalization seams (file/function level):

1. **Binary `medium_id` (0/1).** `_interface_refract_reflect` (`photon_step.py`) selects
   `n_from/n_to` via `where(medium_id==0, n_inner, n_outer)` and flips `1 - medium_id`;
   `_common_propagation` selects per-photon optics via `where(medium_id==0, …)`. → generalize
   to an **N-region index** with `gather` (`sl[medium_id]`) and a `next_medium` lookup.
2. **Nested-sphere-only geometry.** `intersect_two_spheres_forward`,
   `create_nested_sphere_propagator`, the `isinstance(detector, NestedSphere)` branch in
   `DetectorGeometry.from_config`, and `is_nested/medium_outer/r_inner/r_outer` fields. →
   an **N-region geometry abstraction** (`region_of(pos)→[0,N)`, N media, N−1 interface
   radii, multi-interface intersection) that also covers **nested cylinder** and the
   deferred **acrylic 3rd shell** (LS / acrylic / water).
3. **Scalar, monochromatic interface n.** `_interface_refract_reflect` takes scalar
   `n_inner/n_outer`; the `lam` per-photon wavelength is **not** used at the interface. →
   thread **n(λ)** so a Cherenkov-spectrum source sees correct dispersion at the boundary.
4. **`DetectorParams.outer_optics` is NOT integrated** with the param machinery: it is a
   live, fittable pytree leaf, but it is **excluded** from `_SUBTUPLES`, so
   `default_bounds`, `make_optimization_mask`, and `save/load_detector_params` ignore it. ⇒
   **a calibration fit's optimization mask and JSON persistence silently skip the outer
   medium.** This is a real correctness gap for the *calibration* goal and must be fixed
   before the two-medium calibration is trusted. → register per-region optics in `_SUBTUPLES`
   (or an equivalent bounds/mask/save path).
5. **Only calibration mode is fully wired.** `_common_propagation`'s nested branch handles
   all modes, but only `_simulation_sensor_calibration_impl` initialises `medium_id` (from
   the emission point) and calls `_get_optical_arrays_nested`. **Track and data modes would
   run with un-inited medium_id / single-medium optics.** → wire `medium_id` init + nested
   optics for track/data (or explicitly reject nested+track until SIREN-in-LS exists).
6. **Config schema** hardcodes `inner_material/outer_material` + `inner_radius/outer_radius`.
   → a **`regions: [{material, outer_radius}, …]`** list, looped in `generate_detector` /
   `from_config`; `DetectorGeometry` stores `media: list`, `radii: (N−1,)`.

---

## 3. Recommended systematic architecture (two independent generalizations)

**(I) Detection model — per-sensor angular acceptance (Change A, generalized).** Orthogonal
to media. Benefits *every* geometry and the existing SK/HK production work.

**(II) Transport model — N-region media + N−1 interfaces (Change B, generalized).** Orthogonal
to sensors. Single-medium = N=1 (the `if n_regions>1:` branches never trace → byte-identical).
Two-medium = N=2. Acrylic = N=3. Region index everywhere instead of an `is_nested` bool.

Keep the existing invariants: DiCE-forward differentiability, the sampled-interface DiCE
score, the byte-identical degenerate limit (now N=1), the conservation gate.

---

## 4. Merge plan

- **PR1 — generalized angular acceptance.** Per-sensor normals on all geometries + pluggable
  `accept(cosθ)`; conservation gate for sphere/cylinder/box; re-validate calibration/recon
  floors. Standalone, high value (fixes production cylinders). Land first.
- **PR2 — two-medium transport.** Refactor the binary hardcoding to the N-region index
  design (even if only N=2 ships first — cheap insurance against a second rewrite); integrate
  region optics into the param/bounds/mask/save machinery; wire (or gate) track/data; regions
  config schema. Depends on PR1.
- **Studies / configs / materials.** Keep `studies/` as studies (trim the iteration-artifact
  plots); the `labls.json` / `water_n148.json` / `JUNO_nested_*` configs go with PR2;
  `water_n148.json` is a **control material** (label it clearly, not a physical medium).
- **Docs.** `TWO_BOUNDARY_PLAN.md` (build) + this file (integration); update `CLAUDE.md`
  architecture notes when the abstractions land.

### Re-validation checklist (gates both PRs)
1. Conservation flat = coverage on sphere **and** cylinder **and** box (PR1).
2. Full fast suite green; **calibration/recon floors re-run and re-recorded** (PR1 changes them).
3. Single-medium N=1 byte-identical to pre-feature (PR2).
4. Two-medium: invisible-interface == single sphere; analytic Snell/Fresnel/TIR; forward TIR
   threshold at r*; DiCE-unbiased grad incl. the sampled interface (PR2).
5. `outer_optics`/region-optics respected by the optimization mask + JSON round-trip (PR2).

---

## 4b. Decisions locked (2026-06-24)

1. **Split merge:** PR1 = generalized cosθ fix, PR2 = two-medium (depends on PR1).
2. **cosθ: generalize to ALL geometries** via per-sensor outward normals + pluggable
   `accept(cosθ)`. Subsumes/removes the sphere-only `apply_radial_cos`/`sensors_radial` hack.
3. **Region model: N-region INDEX design, ship N=2.** Refactor the binary hardcoding to a
   general region-index abstraction; only exercise N=2 now (acrylic/nested-cylinder = config later).
4. **Integrate per-region optics now** into the DetectorParams bounds/mask/save machinery.

### PR1 — generalized angular acceptance (land first)
1. Add `sensor_normals` (per-sensor outward unit normals) to every surface `Detector`:
   sphere = radial; cylinder = barrel radial-in-xy + caps ±ẑ (reuse the wall/cap split);
   box = face normal. (String/volume detector is out of scope — different capture path.)
2. Generalize `compute_sensor_intersections_base`: replace `apply_radial_cos` with the
   per-candidate `sensor_normals`; apply `accept(cosθ)`, `cosθ = |ray_d·n_sensor|`, `accept`
   pluggable (default identity-cosθ; room for a measured angular-QE curve).
3. Thread `sensor_normals` from the detector through `create_propagator` (shared) + the
   sphere/nested propagators; delete the `sensors_radial` shortcut.
4. Extend `studies/sphere_detection_conservation.py` to **sphere + cylinder + box**; gate
   each at flat-detected-fraction = coverage.
5. Full fast suite green; **re-run + re-record the calibration/recon floors** (they shift).

## 4c. PR2 generic design — VALIDATED by prototype (2026-06-24)

Reordered: **PR2 (two-medium) first, PR1 (cosθ) later.** Two agents prototyped + ran the
generic structure; both pass. PR2 is **decoupled from cosθ** (orthogonal sensor-model
concern) and built so **single-medium is the degenerate `R=1, I=0` case, not a special branch.**

**Transport — one factory-built step (kills the `_nested` duplication + `_IS_NESTED`).**
`make_photon_step(mode, has_interface, reflection_fn)` statically specializes on
`has_interface`: False ⇒ **bit-identical** to today's single-medium step (verified
`array_equal`, sample + expected-value), with a **conditional key-split** (6/8 vs 7/9) so the
interface code + extra key are never traced at I=0 (byte-identity independent of JAX's
split-prefix property, smaller jaxpr). Per-photon optics by **gather `optics_stack[medium_id]`**
(R=1 ⇒ the one array, differentiable). Two-arity factory so the R=1 vmap keeps the 14-arg/
7-tuple signature and `medium_id=None` (unchanged scan pytree); R≥2 adds the interface args +
`new_medium_id`. Simulator parameterized by `(R, I)`; `_get_optical_arrays` unified (resolve λ
once, evaluate over R region-bundles → `(R, n)` arrays); the calibration/`_common_propagation`/
`_final_speed` branches all collapse. Gotchas: `update_factors` uses non-contiguous key indices
(k[0..5]+k[8]) — keep the same slots; prefer the conditional split over always-7.

**Geometry — sensor-free interface-surface list (replaces `intersect_two_spheres_forward`).**
A `Surface` list (tagged-tuple, JAX-traceable; each: `intersect_forward(o,d)->(t,valid,normal)`,
`inside(pos)->bool`, metadata = (region_inner, region_outer) + n-pair). Per step:
`nearest_forward_surface([interfaces] + [outer sensor surface])` → `(t, which, normal)`,
`hit_interface = which∈interfaces`; `region_of` = `inside()` priority scan. **Single-medium =
empty interface list** ⇒ plain outer-surface lookup. Verified to reproduce the current
concentric-sphere behavior with **0 mismatches (4000 rays + 5000 region pts)**, and to handle
inner≠outer shape (sphere-in-cylinder) and non-concentric inner. The interface physics
(`_interface_refract_reflect`) is already normal-sign-agnostic (`abs(cos)`), so only a
consistent outward normal is needed. **Key structural split:** the OUTER instrumented surface
stays the existing per-shape sensor propagator (sphere/cylinder/box) + its grid/sensor-map/
acceptance; the INNER region geometry is the generic sensor-free surface list in front of it.
(So sphere-in-cylinder later = cylinder sensor propagator + a spherical interface surface; out
of scope now, but the abstraction already supports it.)

Prototypes: scratchpad `proto.py` (step/optics), `proto_surfaces.py` (geometry).

### PR2 — N-region two-medium (depends on PR1)  [SUPERSEDED by §4c for the generic structure]
1. **Region abstraction:** `Detector.region_of(pos)→[0,N)`, `n_regions`, interface radii;
   `DetectorGeometry` stores `media: list[MediumProperties]`, `radii: (N−1,)`.
2. **medium_id as N-index:** gather optics `sl[medium_id]` (not `where==0`); `next_medium`
   lookup (concentric spheres: outward→+1, inward→−1).
3. **Interface:** `_interface_refract_reflect` takes `n_from/n_to` (caller indexes by region),
   made **λ-capable**; multi-interface aware.
4. **Propagator:** generalize `intersect_two_spheres_forward`→N spheres, return
   `hit_interface_idx`.
5. **Simulator:** `_IS_NESTED`→`N_REGIONS`; `_get_optical_arrays_nested`→N-region; **wire
   `medium_id` init + nested optics for track/data** (or explicitly reject nested+track).
6. **DetectorParams:** register per-region optics in `_SUBTUPLES` (or equivalent) so
   `default_bounds` / `make_optimization_mask` / `save+load` all respect them.
7. **Config:** `regions: [{material, outer_radius}, …]` schema; loop in `generate_detector`
   / `from_config`. Keep an inner/outer compat shim for the existing nested JSONs.
8. **Re-validate:** N=1 byte-identical; N=2 invisible-interface + analytic Snell/Fresnel/TIR +
   forward TIR threshold; DiCE-unbiased grad incl. sampled interface; mask + JSON round-trip
   for region optics.

## 4d. PR2 acceptance checklist (DRAFT — for sign-off before implementing)

Scope: generic two-medium **transport** — `(R,I)` parameterization, factory photon step,
surface-list inner geometry, `optics_stack[medium_id]` gather, symmetric `DetectorParams`
(inner/outer `MediumParams`), **N=2**, **no cosθ** (PR1), **calibration/point-source only**
(track+data nested raise a clear `NotImplementedError`).

Acceptance gates (all must pass; numbers recorded in the PR):
1. **Single-medium byte-identity (R=1, I=0).** Forward hits **and** an AD gradient are
   `array_equal` to current `main` for sphere, cylinder, box (the factory's no-interface
   specialization + empty interface list + unchanged scan pytree). This is THE gate that
   proves single-medium isn't perturbed.
2. **Surface-list reproduces the old geometry.** `nearest_forward_surface` + `region_of` on
   two concentric spheres == the retired `intersect_two_spheres_forward` (0 mismatches over a
   ray + region-point sweep). jit+vmap clean.
3. **Invisible interface.** Nested with inner≡outer material and n_inner=n_outer reproduces a
   single sphere at R_out (ratio ≈ 1) — valid here because *both* use the same (no-cosθ)
   detection.
4. **Analytic interface physics.** Snell + Fresnel R₀ + TIR + R+T=1 unit tests on the
   interface kernel (independent of detection).
5. **Forward TIR onset.** Contrast/matched ratio is ≈1 below r* and drops above (the geometric
   threshold emerges). NOTE: the ratio *magnitude* is biased by the detection over-count until
   PR1 — gate the threshold/onset, not the absolute %.
6. **DiCE-unbiased gradient** through the sampled interface (AD vs FD on an interface-sensitive
   loss).
7. **Param machinery.** `DetectorParams` save/load round-trips inner+outer; `make_optimization_mask`
   and `default_bounds` cover both media; normalize/denormalize descend correctly. Documented
   migration for any existing saved params.
8. **Mode guard.** Nested + track and nested + data raise a clear `NotImplementedError`.
9. **Full fast suite green** (`pytest tests/`), incl. the updated `test_nested_sphere.py`.

Explicitly NOT gated in PR2 (deferred to PR1): detection-fraction = coverage (needs cosθ);
clean absolute interface-loss percentages; cylinder/box angular acceptance.

## 4e. PR2 building blocks VALIDATED + reviewed (2026-06-24)

Worktree `LUCiD_pr2`, branch `pr2-two-medium-generic`. Gates 1–4 pass (unit + end-to-end
byte-identity incl. gradient; surface-list 0-mismatch; invisible-interface; analytic
Snell/Fresnel/TIR). 3-agent pre-integration review done; design sound (byte-identity robust at
N=100k / extremes / rbg PRNG). Hardening landed (`47b8a94`): NaN-grad fix in
`surfaces.sphere_forward_t` (eps-inside-sqrt), `region_of_spheres` S=0 guard, `nearest_interface`
`t_outer` fold-in, ported physics comments.

**Final param layout DECISION:** primary medium stays top-level (`dp.scattering`/`dp.absorption`
unchanged — zero churn to `optical_model.py`/`event_io.py`/scripts); surrounding shells =
`outer_media: tuple[MediumParams, ...]` (N-native: N=2→1 entry, N=3→2, no refactor), registered
into `default_bounds`/`make_optimization_mask`/`save-load`. `optics_stack = [primary]+outer_media`,
gathered by `medium_id`. NOT the symmetric `dp.media[i]` rename (high churn, no N=2 benefit).

**cosθ:** strip `apply_radial_cos` from the nested propagator → PR2 genuinely no-cosθ; keeps
gate 3 self-consistent (matches the no-cosθ decision). The general cosθ fix stays PR1.

### Integration roadmap (ordered; from the integration review)
1. **Setup guard** (`simulator.py` ~L219): compute I; `raise NotImplementedError` for
   `I>0 and mode in {track,data}` (nested = calibration/point-source only).
2. **Geometry plumbing** (`detector_geometry.py`): expose `n_regions`/interface centers+radii/
   `speed_stack` (keep inner/outer compat for the existing nested JSONs).
3. **Propagator** (`nested_sphere.py`): swap `intersect_two_spheres_forward` → `surfaces.nearest_interface`
   (+ the `t_outer` outer-surface comparison — net-new, needs fresh validation); strip cosθ.
4. **Factory wiring** (`simulator.py` L35-38, L257-282): one import; one `photon_update_fn` via
   `make_photon_step`; drop `photon_update_fn_nested`. Re-run gate 1 against the REAL import.
5. **Optics unify** (`simulator.py` L454-557): merge `_get_optical_arrays*` → R-region stack
   resolver. Validate R=1 stack == legacy single output.
6. **Scan-body collapse** (`simulator.py` L684-739): gather `optics_stack[mid]`/`speed_stack[mid]`;
   keep ONE `if has_interface` only around the vmap arity (7- vs 8-tuple is static in JAX).
   Keep `medium_id=None` for R=1 (preserves the unchanged scan pytree → gate-1 byte-identity).
7. **Calibration impl** (`simulator.py` L1027-1082): route through the unified resolver + `region_of`.
8. **DetectorParams** (`detector_params.py`): `outer_media` tuple + bounds/mask/save; migration =
   absent ⇒ `outer_media=()` (existing params load unchanged).
9. **Retire** `*_nested` step fns (`photon_step.py` L375-554); update the 2 study scripts +
   `tests/test_nested_sphere.py`. Keep single-medium `photon_iteration_*` (wide import surface).
10. **Tests:** promote gate 1/2 to pytest; add split-prefix invariant + DiCE-unbiased
    interface-grad (gate 6) + outer_media round-trip (gate 7).

Gotchas to respect: don't "tidy" the non-contiguous `update_factors` key slots (k[0..5]+k[8]);
`_interface_refract_reflect` stays binary (fine for N=2, not N≥3); interface n is scalar
(monochromatic-exact; a Cherenkov-spectrum source is an approximation until n(λ) is threaded).

## 5. Accepted-by-decision (carry forward, don't silently ship as "done")
Pure absorption (no LS re-emission); point sources only (no SIREN-in-LS); interface n fixed
(not a fit target); acrylic lumped (until the N=3 region path lands); idealized outer wall.
These are fine *if labelled*; they bound what the two-medium numbers mean.
