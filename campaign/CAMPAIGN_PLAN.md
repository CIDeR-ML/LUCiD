# 12-hour analysis campaign — plan & running log

Goal: thoroughly analyze each calibration part, the **N-scaling** of both the CRB
(uncertainty) and the **fit** (recovery), then **source/location combinations** (laser vs
isotropic, positions). Run on all 5 GPUs. Document as we go.

## Infrastructure
- `gridfit.py` — ONE config worker: env `NPH`, `SRC` (source-combo key), `RECOVER`,
  `SHOT`, `EPS`, `BAKE_K`, `POLYAK`, `K`, `GRID`, `TAG`. Computes CRB of the 7 globals and
  (optionally) a recovery fit; writes `grid_out/<TAG>.json`. Self-contained, GPU.
- `run_grid.py` — dispatcher: enumerates a job grid, runs across the 5 GPUs (one process
  per GPU, round-robin), each with `CUDA_VISIBLE_DEVICES=<g>`. Writes per-job JSON.
- `aggregate.py` — reads `grid_out/*.json` → markdown tables.

## Source-combo registry (keys for `SRC`)
Single: `laser_down`, `laser_up`, `laser_wall`, `laser_diag`, `iso_center`,
`iso_off` (R/2,0,0), `iso_top`. Combos: `laser_iso` (down+center), `multi_laser`
(down+up+wall), `multi_laser_iso`, `iso_ring` (4 iso around), `all`.

## Phases
- **P0** — finish + analyze the stabilized shot-noise #4 (Anscombe+bake_k+polyak); full suite.
- **P1 N-scaling of CRB** — `SRC=laser_iso`, NPH ∈ {1e5,3e5,1e6,3e6,1e7(if it fits 11GB)}.
  Expect 1/√N for photon-limited (qe, L_R), FLAT for degeneracy/systematics-limited.
- **P2 N-scaling of the FIT** — recovery (implicit) + shot-noise scatter vs N for the same
  grid; compare realized σ to CRB×√12 vs N.
- **P3 source/location combos** — CRB of all 7 globals for every `SRC` key at fixed N;
  map which configuration constrains which parameter (reflection split, L_M, qe↔L_abs).
- **P4 synthesis** — best source design per parameter; N needed to reach a target σ.

## Running log
(appended as runs complete)

### P1 CRB N-scaling (DONE) — corrected (vary INTENS, NPH=1e6 fixed)
CRB ∝ 1/√budget EXACTLY for all 7 params (√budget check = 1.00 across the board) — the
Fisher bound is purely photon-limited (no systematics in the bound). Hierarchy (fixed
budget): L_R≈g≈L_M (tightest) ≪ qe < L_abs < wall < sensor (loosest, info-limited).
qe: 43%@1e5 → 0.43%@1e9; L_abs 94%→0.94%; sensor 328%→3.3%. So qe hits ~1% near 1.7e8.

### P3 source/location CRB scan (DONE, budget 1e7)
SINGLE source → amplitude params (L_abs/qe/wall/sensor) DEGENERATE (σ~1e5%, per-PMT k
absorbs them); SHAPE params (L_R/g/L_M) stay constrained. Multi-source breaks the
degeneracy; tightest = `all` (3 laser + 2 iso): L_abs 3.8%/qe 2.0%/wall 9.3%/sensor 15%.
multi_laser_iso(4) close behind; iso_ring(4) looser on reflection (lasers > iso for
reflection geometry). sensor_refl always the loosest reflectivity.

### P2 N-scaling of the FIT (DONE) — recovery + shot-noise, stabilized (bake_k+polyak+Anscombe)
Realized error = max(bias, scatter), and the two halves scale differently:
- SHAPE (L_R/g/L_M): ~UNBIASED on shot noise (|bias|≤ few %); realized σ FLOORS at ~0.1-2%,
  ABOVE the ultra-tight CRB (σ/CRB rises 0.4→3-8 as budget 1e5→3e6) ⇒ FIT/forward-limited,
  ~FLAT in N, not photon-limited.
- qe: shot bias −96%@1e5 → −0.2%@3e6 (laser_iso) — monotone de-biasing with occupancy.
  VALIDATES #4: the stabilized recipe recovers qe to ~unbiased at adequate occupancy (vs the
  −50% divergence with no stabilizers). So #4 closes with stabilizers + high N.
- L_abs/wall/sensor: BIAS-limited at low N (low occupancy + √N Jensen + reflection
  degeneracy), bias shrinks with N (L_abs −85→−20%, wall/sensor →~+18% by 3e6) — the hardest,
  occupancy/degeneracy-limited. Need higher budget than 3e6 to de-bias fully.
- multi_laser_iso tightens CRB + somewhat reduces biases vs laser_iso, but amplitude biases
  are occupancy-driven (similar). BOTTOM LINE: CRB is 1/√N for all; the FIT reaches it only
  where neither the fit-floor (shape) nor the occupancy bias (amplitude) dominates.

### loc_scan source-LOCATION sweep (DONE, budget 1e7) + P4 SYNTHESIS
Location findings:
- SINGLE isotropic constrains nothing well (uniform light → even L_R degenerate ~1e4%);
  SINGLE laser pins the SHAPE params. ⇒ lasers carry geometric/shape info, iso does not.
- iso toward the wall (r0→r95) OR laser tilt (15°→60°) IMPROVES sensor_refl (32.8%→~22%)
  and qe (4.3%→2.8%): wider range of incidence angles on PMTs/walls constrains the
  angle-dependent reflection + response.
- L_abs (~9.4%) and L_R (~0.05%) are LOCATION-INSENSITIVE (bulk/shape, set by total light).

## P4 SYNTHESIS — calibration taxonomy on the unified recipe
PER-PARAMETER verdict (charge-only, the validated GN recipe):
- L_R, g, L_M  — SHAPE params. Tightest CRB (≪0.1% at high budget). Constrained by ANY
  laser (single source OK). FIT-LIMITED in practice (~0.1-2% realized floor, flat in N) —
  the optimizer/forward noise, not photons, sets the floor. Location-insensitive.
- qe          — needs SOURCE DIVERSITY (single→degenerate). CRB~1%@budget 1.7e8. De-biases
  with occupancy (−96%@1e5→−0.2%@3e6 budget on shot noise). Off-center iso helps a little.
- L_abs       — diversity-gated, CRB~3-9% (multi). Location-insensitive. Bias-limited on
  shot noise at low occupancy; the hardest BULK param.
- wall, sensor— the REFLECTION split: sensor_refl ALWAYS loosest (few photons hit sensors).
  Helped MOST by wide-incidence-angle illumination (iso-at-wall / tilted laser) — sensor
  32.8%→22% by moving the iso to r95. Bias-limited on shot noise, occupancy-driven.
SCALING LAWS: CRB ∝ 1/√(physical budget) for ALL params (P1). The FIT reaches the CRB only
where neither (a) the shape fit-floor (~0.1-2%, flat) nor (b) the amplitude occupancy-bias
dominates. Best single config = `all` (3 laser + 2 iso); for sensor_refl specifically add
a wall-region iso or tilt the laser.
N-TARGETS (budget to hit ~1% CRB, laser_iso): qe ~1.7e8, L_abs ~1e9, wall ~4e9, sensor ~1e10
(sensor needs ~60× qe's budget — or better geometry, not just photons).
