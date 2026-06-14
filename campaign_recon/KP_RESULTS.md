# keys × photons study — AD-Fisher reconstruction (100 events)

Same campaign as `RESULTS.md` (GEANT4/PhotonSim clean data, 1050 MeV muons, randomized into the
SK_like fiducial volume, 2.5 ns TTS, two-start seeder → Fisher-GN `fit_track`, scored vs the
**exact gun truth**), but with the **new AD Fisher metric** (`fisher_mode='ad'`, one `jacfwd` pass
replaces 9×2 FD evals; `custom_vjp`/`nan_to_num` dropped from the forward to unblock forward-mode AD)
at `lr=1`. Three combos vary the two cost knobs:

- **NKEYS** = PRNG keys averaged into the gradient *and* the Fisher metric (each key = an independent
  photon sample of the soft predictor).
- **NPH** = predictor photons (sets the per-key Monte-Carlo shot-noise on μ and ∇μ).

| combo | NKEYS | NPH | wall (5 GPU) | vtx median | vtx mean / RMS | dir median | dE median | dt0 median | <20 cm | wanderers >100 cm |
|---|---|---|---|---|---|---|---|---|---|---|
| **A** | 4 | 250 k | **69 min** | **12.4 cm** | 14.3 / 16.2 | **0.92°** | +0.1 MeV | +0.02 ns | **83/100** | 0 |
| **B** | 1 | 250 k | 24 min | 56.0 cm | 61.9 / 73.8 | 1.93° | +16.6 MeV | −1.70 ns | 8/100 | 10 |
| **C** | 1 | 500 k | 39 min | 19.9 cm | 23.4 / 27.6 | 1.07° | +2.9 MeV | −0.23 ns | 50/100 | 0 |

All three use `FISHER_MODE=ad LR=1 NITERS=250`, the two-start (charge-grid ‖ time-multilateration)
seeder, 1 % loss-margin arbitration. 0 failures, 100/100 events each.

## Read

**1. AD reproduces the validated FD recipe (combo A).** A (AD, `lr=1`) lands at vtx **12.4 cm /
dir 0.92° / E unbiased / t0 ~0** — statistically identical to the FD two-start baseline
(`RESULTS.md`: 11.9 cm / 0.94° / unbiased), at 2.8× lower Fisher-build cost. The AD Hessian is
**not** the bottleneck and `lr=1` is the right pairing for it (confirms `recon-ad-hessian-not-broken`).

**2. NKEYS=1 is NOT viable at fixed photons (combo B).** Dropping 4→1 key at 250 k photons
**quadruples** the vtx median (12→56 cm), opens **10 wanderers >100 cm**, and biases energy
(+16.6 MeV) and t0 (−1.7 ns). The AD Hessian only de-noises the *metric build*; the **gradient
itself is still a single-key photon sample**, and at 1 key its Monte-Carlo variance dominates the
GN step — the optimizer chases noise and the 1 %-margin arbitration mis-picks the time seed far
more often (31/100 vs 3/100 for A). Faster (24 min) but unusable.

**3. Doubling photons largely buys back the lost key (combo C).** At 1 key, 500 k photons recovers
vtx to **19.9 cm**, kills all wanderers (10→0), and restores E/t0 nearly to A's level. This is the
key physics: per-key gradient variance ∝ 1/NPH, so 2× photons ≈ √2 lower gradient noise — most of
the damage from `NKEYS 4→1` is single-key **shot noise**, not a structural problem with one key.
C reaches A-class direction (1.07°) but is still ~1.6× worse on the vertex and runs 39 min.

## Cost / resolution trade

A is the **resolution winner** (12.4 cm, all floors hit) at the highest cost. C is the **efficiency
point** (≈20 cm, no tail) at ~half A's wall-clock — useful when 20 cm suffices or for large sweeps.
B (1 key / 250 k) should not be used. Equivalently: **keys and photons are interchangeable against
the gradient shot-noise floor, but you cannot simply remove keys for free — you must replace each
dropped key's variance with more photons.** 4 keys × 250 k slightly beats 1 key × 500 k (4× vs 2×
the total photon statistics into the gradient), as expected.

## Trajectories

Every `ev###.npz` saves the winning-seed trajectory (`traj` (251,9), `gnorm`, `best_iter`) **and**
both per-seed trajectories (`trajA/B`, `gnormA/B`, `best_iterA/B`) plus combo provenance
(`nkeys/nph/fisher_mode/lr`). Best-‖g‖ iter median: A 207, B 218, C 218 (all converge within the
250-iter budget); ‖g‖ reduction median ×0.015 (A) / ×0.031 (B, noisier so stalls higher) / ×0.019 (C).

Figure: `fig_keysphotons.png` (vtx-error CDF, paired per-event scatter vs A, median vtx/dir bars).
Aggregate via `aggregate_kp.py`; runs via `run_keysphotons.sh` (calls `worker.py` with the env knobs).

---

# Follow-up: convergence, readout, and the 1M test (50 events, NITERS=500)

The 250-iter combos B/C were diagnosed as **under-converged** (`plot_convergence.py`: still descending
in the final 20% of iters, best-‖g‖ peaking near the 250 budget edge). Re-ran the 1-key combos at
**NITERS=500** and added **D5 = 1 key × 1M photons** (the equal-budget partner of A=4×250k). All scored
vs exact truth, min-‖g‖ readout, 50 common events. (`run_iters.sh`, `aggregate_iters.py`, `plot_iters.py`.)

| combo | keys × photons | total phot | iters | vtx median | dir | **dE median** | converged |
|---|---|---|---|---|---|---|---|
| **A** | 4 × 250k | 1M | 250 | **11.7 cm** | 0.9° | **−0.5 MeV** | yes (best-it 207) |
| B5 | 1 × 250k | 250k | 500 | 21.0 (ming) / 15.8 (final) | 1.3° | +4.4 | marginal |
| **C5** | 1 × 500k | 500k | 500 | **12.5 cm** | 0.9° | −2.4 | yes (best-it 371) |
| D5 | 1 × 1M | 1M | 500 | 14.0 cm | 0.9° | −8.3 | yes (best-it 392) |

(paired on the same 50 events; A recomputed on that subset → 11.7 cm.)

## Finding 1 — more iters fixed the under-convergence; min-‖g‖ is the right readout

500 iters converges the 1-key combos (best-‖g‖ iter ≪ budget; vtx flat in the final 20%). Once
converged, **`min-‖g‖` is correct again** — and it's the only NaN-robust readout: 1/50 B5 fits
diverged to NaN at iter 236 (after its best point); min-‖g‖ caught the good iterate, `final`/`polyak10`
grabbed the NaN tail. The `final`/`polyak` *advantage seen at 250 iters was purely under-convergence*
(the trajectory hadn't bottomed, so the end was best). Polyak adds nothing once converged. **Use
min-‖g‖ + enough iters.** (B5 at 250k is the lone exception — so photon-starved that ‖g‖ is too noisy
for min-‖g‖ even at 500 it, wanting `final`=15.8; another reason 250k is below the usable 1-key budget.)

## Finding 2 — the ~12 cm vertex floor is bias-limited; 1M does NOT beat 500k

C5 (1×500k) reaches **12.5 cm ≈ A's 11.7**, and D5 (1×1M) is **no better — 14.0 cm**. Beyond ~500k
photons the vertex is pinned by SIREN-emission fidelity, not statistics, so more photons can't help.
**1M photons is overkill for the vertex.**

## Finding 3 — keys are NOT fully interchangeable with photons (the decisive 1M result)

The equal-budget test **A (4×250k) vs D5 (1×1M)** — same 1M total photons, split into 4 independent
keys vs one big sample — comes out **clearly in A's favor**: vertex 11.7 vs 14.0 cm (paired +2.8 cm,
25/50 events worse), and energy bias **−0.5 vs −8.3 MeV**. Moreover the single-key energy bias grows
*monotonically more negative with photons*: B5 +4.4 → C5 −2.4 → D5 −8.3, while the 4-key fit stays
unbiased. **Averaging independent photon realizations (keys) cancels a nonlinear / Jensen-type bias
in the single-key estimator that more photons cannot remove — one large sample just locks onto it
harder.** So "keys ≈ photons" holds only for the *vertex near 500k* (C5 within ~1 cm of A); for
*energy*, and at *1M*, independent keys are required.

## Recommendation

- **Best resolution + unbiased energy:** keep **A = 4 keys × 250k, min-‖g‖** (vertex 11.7 cm, energy unbiased).
- **Cheapest floor-quality vertex:** **C5 = 1 key × 500k × 500 iters** (12.5 cm) — but carries a small
  (−2.4 MeV) energy bias; fine if energy isn't the target.
- **Do not** use 1 key × 250k (photon-starved, 21 cm) or 1 key × 1M (no vertex gain, −8 MeV energy bias).
- The lever for the *energy bias* is **key count (independent realizations)**, not photons; the lever
  for the *vertex floor* is **emission-model fidelity (SIREN)**, not photons or keys past 500k.

Figure: `fig_keys_vs_photons.png` (vtx CDF; energy-bias-vs-photon-budget showing the 1-key trend vs the
4-key point; paired vtx vs A). Convergence diagnostics: `fig_convergence.png`. Readout study:
`readout_study.py`.
