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
