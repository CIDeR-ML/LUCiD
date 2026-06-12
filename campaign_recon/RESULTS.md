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

## Convergence note

best-‖g‖ iter median 192/250; 23/100 events peak ‖g‖ in the final 10% → a minor tail that
would tighten with more iters, but the median is already converged by iter 192.
