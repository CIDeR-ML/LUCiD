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

## 5. Accepted-by-decision (carry forward, don't silently ship as "done")
Pure absorption (no LS re-emission); point sources only (no SIREN-in-LS); interface n fixed
(not a fit target); acrylic lumped (until the N=3 region path lands); idealized outer wall.
These are fine *if labelled*; they bound what the two-medium numbers mean.
