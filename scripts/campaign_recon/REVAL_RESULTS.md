# Recon re-validation on the merged engine (new s/s_max SIREN emitter)

GEANT4-data recon (`worker.py`, 1050 MeV muons, SK_like fiducial, 2.5 ns TTS, 3-stage
seeder + two-start Fisher-GN). Compares the **new** emitter (merged branch) against the
documented **old**-emitter baseline.

## Headline finding — the new net is 1.66× LOW vs GEANT4

`pred` q_tot @ truth E=1050 = **3276** but the GEANT4 data q_tot = **5375** →
ratio **1.66 ± 0.02**, flat across events ⇒ pure absolute-normalization (the new net's
`nphot` block; the old net was GEANT4-matched). Uncorrected, the energy fit compensates to
E≈1617 (+57%) and the vertex regresses via energy↔vertex coupling (ev0 = 83.9 cm).

Fix: `setup_event_simulator(cherenkov_photon_norm=...)` — emitter charge calibration,
default 1.0 = byte-identical no-op.

## Resolution (20 GEANT4 events, vs the old-emitter baseline)

| Quantity | new emitter, **uncorrected** | new emitter, **×1.66** (median) | old-emitter **baseline** |
|---|---|---|---|
| vtx \|Δ\| | 83.9 cm (ev0) | **22.0 cm** (mean 21, RMS 23) | 13.9 cm |
| dir | 0.70° (ev0) | **1.05°** (mean 1.13) | 1.86° |
| dE | +566 MeV (ev0) | **+70 MeV** (mean +70, coherent) | +3.8 MeV |
| dt0 | −3.76 ns (ev0) | **−1.6 ns** (mean −1.63) | −0.20 ns |

vtx buckets (×1.66): <20 cm 9/20, 20–40 cm 10, >40 cm 1.

## Read

The 1.66× correction **dramatically improves** recon (vtx 83.9→22, dE +566→+70) and
**direction is actually better** than baseline (1.05° vs 1.86°), but it does **not fully
restore** the vtx/energy/t0 floor:

- **Residual +70 MeV** energy bias (coherent, mean≈median). Note the SIGN: at ×1.66,
  pred(truth)=5438 slightly OVER-predicts the data's 5375, so a pure total-charge argument
  would push energy *down* — yet dE is +70 *up*. So this is **not** a scale-magnitude
  miss; it's a charge-**shape** + time-term difference (the per-PMT distribution the new
  net paints vs GEANT4) that a single scalar normalization cannot absorb.
- **t0 offset −1.6 ns** (vs −0.20 baseline) — independent of charge norm; the new net's
  emission-time model (`predict_t0` cubic from the new `t0.json`) is shifted vs the old.

**Conclusion.** The merge is sound — the emitter is wired correctly (q_tot scales, AD==FD)
and the new `cherenkov_photon_norm` knob is byte-identical at 1.0. But the **new net is less
GEANT4-faithful than the old** for reconstruction: absolute normalization (1.66×, knob-
fixable), plus residual charge-shape and t0-model differences. To recover the recon floor
the **new net needs re-calibration / retraining to GEANT4** (charge yield + shape + t0) —
the "recreate the n_photons function as needed" path (yield AND per-PMT shape AND t0).
Interim: `cherenkov_photon_norm=1.66` gets vtx ~22 cm / dir ~1° (dir already beats
baseline); the energy/shape and the −1.6 ns t0 offset need the new net + `t0.json`
revisited, not a scalar.

---

## Exact cause decomposition (old vs new emitter, same recon code)

Apples-to-apples baseline is `out_ms` (OLD emitter, SAME merged recon worker), not the
single-start RESULTS.md (13.9cm). Comparing on overlapping events:

| | vtx | dir | dE | dt0 |
|---|---|---|---|---|
| out_ms (OLD emitter) | 23.5 cm | 1.88° | **−1 MeV** | **+0.20 ns** |
| reval166 (NEW, ×1.66) | 22.0 cm | 1.05° | **+70 MeV** | **−1.6 ns** |

**Vertex & direction MATCH (new ≈ or better).** Only energy + t0 regressed. Decomposed by
direct old-vs-new-vs-GEANT4 forward comparison at truth geometry:

1. **Photon YIELD −1.66×** — GEANT4 ROOT stores 357k photons @1050 MeV; the new net's
   `nphot(1050)=215k` (old net matched GEANT4). The old net's `tot_n_photons` (~11.8k) was
   an importance-reweight normalization; the new net's `nphot` is a physical-ish count that
   under-fits the true Cherenkov yield by 1.66. → `cherenkov_photon_norm=1.66` (committed).

2. **Charge SHAPE — not the problem.** Both emitters are equally diffuse (the DiCE soft
   forward + SIREN smoothness): old nlit>0 = 10,681 / 35% charge on GEANT4-dark PMTs /
   bright-ring ratio 0.673; new 10,683 / 27% / 0.748. The new emitter's charge is actually
   slightly *better*. This is why vtx/dir are fine.

3. **Emission TIMING +1.16 ns** — new emitter first-arrival vs GEANT4 = **+0.74 ns** (late);
   old = **−0.42 ns** (early). The `predict_t0` cubics are nearly identical (≤0.04 ns over
   d=2–5 m), so the shift is in the photon **emission geometry** (longitudinal `s/s_max ×
   s_max(E)` placement / time distribution), not the t0 model. It drives recon **t0 −1.6 ns**
   (absorbed) and, via the time↔energy Fisher coupling, **E +70…+150** (the fit drifts there
   even from the truth seed → genuine joint-loss-minimum bias, not optimizer noise).

## Verdict — reachable

The merge is correct (emitter wired right; q_tot scales, AD==FD; calibration CRB intact —
hello_calibrate recovers all 7 params within bound). The recon delta is **two net-
calibration offsets** in the new s/s_max net vs the old/GEANT4: **(1) yield 1.66×** and
**(2) emission timing ~1.16 ns**. Charge shape, vertex, direction already match. To reach the
old result: re-calibrate the new net's `nphot` (×1.66 → matches GEANT4) and its emission-time
/ longitudinal placement to GEANT4 (the t0/timing scan), OR carry the `cherenkov_photon_norm`
+ a small emission-time offset as interim constants. Both are net/emitter calibration, not
recon-pipeline regressions.

---

## SIREN emission vs GEANT4 — pre-propagation, multi-event (the energy mismatch)

Comparing the EMITTED photons directly (no propagation; raw ROOT frame = vertex 0 / +z;
origins cm→m), new SIREN vs 20-GEANT4-event average (7.1M photons):

**Longitudinal emission quantiles (the ENERGY mismatch):**

| | 50% | 90% | 99% | track end |
|---|---|---|---|---|
| GEANT4 (20 ev) | 2.16 | 4.08 | 4.70 m | 4.86±0.15 m |
| new SIREN ×1.00 | 1.77 | 3.46 | 4.03 m | 4.17 m |
| ratio G4/new | 1.22 | 1.18 | 1.17 | — |

The new SIREN's longitudinal emission is **uniformly compressed ~0.85×** (front-loaded) — a
1050 MeV muon's light looks ~15% short ⇒ reconstructs LOW in energy. This is the SIREN
**energy mismatch**, and it is ORTHOGONAL to the `nphot` yield knob (which rescales magnitude,
not extent). Decomposition of the 1.66 yield gap: length 1.13× × per-meter-density 1.47×.

**Fix = `s_max(E)` (longitudinal scale).** `s_max ×1.18` puts the emitted longitudinal onto
GEANT4 (50/90/99% → 2.09/4.08/4.76 m) and in recon pulls dE +87→+50 MeV (mean +80→+59) — the
predicted direction. Added `setup_event_simulator(cherenkov_smax_norm=1.0)` knob (byte-
identical at 1.0). A single scalar isn't a complete fix (the longitudinal SHAPE differs and
couples to vertex), but it confirms the diagnosis.

**Angular ring:** new is too sharp — 81% in 39–43° vs GEANT4's 54%, missing the 30–36° and
43–46° shoulders (dispersion + multiple scattering + delta-ray Cherenkov the straight-track
SIREN doesn't model). Affects vtx/dir-level, not energy.

## Full SIREN-vs-GEANT4 verdict

The new s/s_max net does NOT exactly match GEANT4 in four separable ways: **(1) yield 1.66×
low** (nphot under-fit), **(2) longitudinal ~15% short** (s_max under-fit / front-loaded
density — THE energy mismatch), **(3) emission timing +1.16 ns late** (drives recon t0), **(4)
ring too narrow** (delta-ray/scattering not modeled). The old net matched on (1)+(2). Proper
fix = re-calibrate/retrain the new net (nphot + longitudinal density/s_max + t0); interim
levers `cherenkov_photon_norm` + `cherenkov_smax_norm` (committed, byte-identical at 1.0).

---

## WHY the old net matched (the structural root cause)

OLD net inputs = `(energy, angle, DISTANCE)` with **distance in physical mm, range 10–9990 mm**
(≈ up to 10 m). It predicts density over physical distance DIRECTLY and places photons there.
NEW net inputs = `(energy, angle, s/s_max)` with distance normalized to **[0.001, 0.999]**, then
physical distance is rebuilt via a SEPARATE `s_max(E)` power-law fit.

Emitted longitudinal (no propagation), new vs old vs 20-event GEANT4:

| quantile | OLD emitter | GEANT4 | NEW emitter |
|---|---|---|---|
| 50% | 2.20 | 2.16 | 1.77 m |
| 90% | 4.14 | 4.08 | 3.46 m |
| 99% | 4.82 | 4.70 | 4.03 m |

**The OLD net's longitudinal matches GEANT4 to ~1–3%; the NEW net is ~15% short.** The whole
energy mismatch is the new net's `s/s_max × s_max(E)` reparametrization (introduced for
energy generalization) compressing the track — the old direct-physical-distance net had it
right. **Wavelengths: NEITHER net stores/uses them** (both predict λ-integrated density; the
emitter samples λ + applies QE identically), so wavelength handling is the same for both —
the new net's λ-sensitivity in recon is a downstream symptom of the compressed longitudinal,
not a saved-wavelength difference. New net is also better-FIT to its data (val loss 0.018 vs
0.075) — the defect is the parametrization/training target, not fit quality. Revisit fix:
correct the new net's `s_max(E)` / longitudinal density + `nphot` + t0 to GEANT4 (the
`cherenkov_smax_norm` / `cherenkov_photon_norm` knobs target exactly these, byte-identical@1.0).
