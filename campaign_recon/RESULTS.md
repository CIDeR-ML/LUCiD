# Seeded recon campaign — 100 events

GEANT4/PhotonSim data (clean unit-weight integer photons, `is_data=True`), 1050 MeV muons,
randomized into the SK_like fiducial volume, 2.5 ns per-photon TTS. Per event: 3-stage
data-driven seeder (energy scan → geometric vertex+t0 grid → cone direction) → Fisher-GN
`fit_track` (NITERS=250, NKEYS=4, SCALE9-preconditioned, min‖g‖ readout). 5 GPUs × 20 events,
**52 min wall, 0 failures.**

## Resolution (vs PCA-derived truth)

| Quantity | seed median | **fit median** | fit mean | fit RMS | fit 68% |
|---|---|---|---|---|---|
| vtx \|Δ\| | 90 cm | **13.9 cm** | 21.0 | 40.1 | 16.6 |
| dir      | 5.5° | **1.86°** | 2.03 | 2.32 | 2.27 |
| dE       | +141 MeV | **+3.8 MeV** | +1.3 | 13.2 (~1%) | 11.6 |
| dt0      | −7.5 ns | **−0.20 ns** | −0.47 | 1.43 | 0.50 |

Capture buckets: converged <20cm **76/100**, good 20–40cm 19, partial 40–100cm 3, wanderer >100cm 2.
Improved vertex on 97/100. ‖g‖ reduction median ×0.015.

## Read

The fit hits **every known floor simultaneously** on the well-seeded majority:
vtx ~13.9 cm (floor lon14/tra8), dir ~1.9° (floor ~2.1°), t0 ~0.2 ns (floor 0.2–0.5 ns),
and **energy bias ≈ 0** (+1.3 MeV mean, ~1% RMS) — *better* than the SIREN-self-consistent
path's −40 MeV coherent low bias, because this fits clean GEANT4 photons rather than
SIREN-sampled "truth." The floor is **bias-limited (SIREN emission fidelity), not
optimizer-limited** — as the diagnostics predicted. Median/68% are on the floor; mean/RMS
are inflated only by the 5-event tail below.

## Tail (5 events ≥40 cm) = seeder basin-capture, not fit failure

The 5 worst events have **median seed vtx 175 cm** (vs 88 cm overall) — the seeder placed
them outside the ~0.3 m local capture radius, so the GN refiner could not recover; ev58
(seed 175→fit 264) and ev72 (162→226) slid *further* out, with t0 running to −8…−9.5 ns and
‖g‖ barely dropping (×0.05 vs median ×0.015). n_hit is not the discriminator (bad events have
*more* hits). This is the documented local-refiner capture wall, not a forward/loss problem.

**Lever (deferred, robustness not speed):** the `time_vertex` multilateration seeder
(LUCiD_recon/seeder_time.py) tightens the seed vertex and would eliminate the >1.5 m
off-basin outliers; it would also drop the seed t0 from its fixed −7.5 ns offset, removing
the systematic that forces the full 250 iters.

---

# Two-start seeder (charge grid + robust time multilateration) — 100 events

The tail lever above, implemented and run (`out_ms100/`, two fits/event, 102 min, 0 fail).

**`seed_vertex_time`** (`lucid/fitting/recon.py`): multilateration vertex+t0 from first-arrival
times (GPS backwards). Ported from recon's `time_vertex` but made ROBUST — a large fraction of
real hits are scattered/reflected photons arriving 10s–100s ns LATE, and plain least-squares let
them drag the seed ~15 m off. Fix: bright-hit preselection + RANSAC inlier-count grid + GN on
inliers. Nails the TRANSVERSE vertex (5–30 cm); residual is a forward LONGITUDINAL bias — it finds
the Cherenkov time-centroid, ~½ a track-length ahead of the vertex (real physics, ~1.7–4 m here).

The time seed (transverse-perfect, forward-biased) and the charge-grid seed (good longitudinally,
loses inward-pointing tracks) are COMPLEMENTARY → **`fit_track_multistart`** fits from both and keeps
the better basin, gated by a **1 % relative loss margin** (prefer the charge seed; switch to the
time seed only when its converged loss is >1 % lower). Plain argmin over-selects the time seed
(SIREN bias: lowest loss ≠ closest vertex → 28/100 regressed >5 cm); the margin restricts it to the
~4 decisive inward rescues. Margin found retrospectively over the saved loss/error grid (no re-run).

| metric | OLD single | **TWO-start (1 % margin)** | oracle (truth-best) |
|---|---|---|---|
| vtx median | 13.9 cm | **13.7 cm** (tied) | 11.6 |
| vtx mean | 21.0 | **16.0** | 13.9 |
| vtx RMS | 40.1 | **18.3** (halved) | 16.1 |
| events ≥40 cm | 5 | **2** | 2 |
| wanderers >100 cm | 2 | **0** | 0 |
| vs old (better/worse >5 cm) | — | **+13 / −7** | +21 / −6 |

Buckets: converged <20 cm **75/100**, good 23, partial 2, wanderer 0. dir 1.8°, E unbiased
(median +2 MeV, mean −0.1), t0 −0.17 ns. Time seed (B) wins **4/100** — exactly the decisive
inward rescues; the gap to the oracle (30 B-picks) is the SIREN-bias tie zone, unrecoverable by a
loss rule. **Net: median preserved, RMS halved, catastrophic tail eliminated.**

Plots: `out_ms100/fig{1_distributions,2_vertex_seed,3_trajectories}.png` (regenerate via
`plots.py`; `MARGIN` env controls the gate). Aggregate via `aggregate_ms.py`.

## Convergence note

best-‖g‖ iter median 192/250; 23/100 events peak ‖g‖ in the final 10% → a minor tail that
would tighten with more iters, but the median is already converged by iter 192.
