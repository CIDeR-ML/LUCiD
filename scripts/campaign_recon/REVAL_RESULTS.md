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

> **⚠️ SUPERSEDED — see "CORRECTED DIAGNOSIS" at the bottom of this file.** The
> conclusions below ("new net is less GEANT4-faithful / needs retraining / s_max is
> ~15% short") were drawn against the *stale 1050 MeV recon ROOT* and *with the default
> `ray_sampling.threshold=0.05`*. Re-checked against the freshly-downloaded GEANT4 (same
> production the net was trained on), the new net is GEANT4-faithful in yield, longitudinal,
> AND angle. The two recon-affecting effects are a **data wavelength-band mismatch** and an
> **emitter sampling-threshold knob** — neither is a net defect. Read the bottom section.

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

---

# CORRECTED DIAGNOSIS — the new net IS GEANT4-faithful (downloaded-GEANT4 re-check)

Re-ran `download_data.sh` into a fresh worktree (`LUCiD_dlcheck/`) to compare the new net
against the **exact GEANT4 production it was trained on** (`1000MeV_100events.root`, 100 ev),
which the old 1050 MeV recon ROOT is NOT. The downloaded ROOT carries a **`PhotonWavelength`**
branch (the old recon ROOT does not), so the wavelength cutoffs are now directly measurable.
This overturns the "net needs retraining" read above. **Both** recon-affecting effects are
non-net.

## 1. YIELD 1.66× — it's a WAVELENGTH-CUTOFF difference in the DATA, not the net

| | photons/event | per-MeV |
|---|---|---|
| downloaded GEANT4 @1000 MeV (`NOpticalPhotons`) | **202,254** | 202 |
| old recon ROOT @1050 MeV (`NOpticalPhotons`) | **358,049** | 341 |
| new net `nphot(1000)` | **203,673** | — |
| new net `nphot(1050)` | 215,422 | — |

- **new net nphot(1000) = 203,673 vs downloaded GEANT4 202,254 → ratio 1.007.** The net
  reproduces the new-production yield to **0.7 %**.
- The old recon ROOT has **1.68× more photons per MeV** than the new production. Cause =
  **different Cherenkov wavelength band.** The downloaded ROOT's `PhotonWavelength` cutoffs
  are **[275, 674] nm** (sharp). Cherenkov `dN/dλ ∝ 1/λ²`, so yield `∝ 1/λ₁ − 1/λ₂`:
  - new band [275,674]: integral 0.002153
  - old band **[200,700]**: integral 0.003571 → **ratio 1.659** ≈ the observed 1.66×.
  The old recon ROOT was generated with a **wider (bluer) band ≈[200,700] nm**; the extra blue
  photons (1/λ² is steep there) are the entire 1.66×. The net is correct for the **new**
  [275,674] band; the recon "data" is from a different (older) PhotonSim wavelength config.

## 2. LONGITUDINAL "15 % short" + RING "too sharp" — it's the EMITTER THRESHOLD KNOB, not s_max

The net's **raw density** marginal (grid-evaluate the SIREN, integrate over angle) matches the
downloaded GEANT4 to **<0.5 %** at the same energy:

| longitudinal (m), E=1000 | 50% | 90% | 99% |
|---|---|---|---|
| downloaded GEANT4 | 2.094 | 3.916 | 4.502 |
| new net **raw density** | 2.083 | 3.903 | 4.492 |

The compression appears only in the **emitter's sampled rays**, and it is caused by
`ray_sampling.threshold` (siren_params.json, default **0.05**): the seed pass keeps only grid
bins with density ≥ 5 % of peak, discarding the low-density Cherenkov **tail** (and the angular
**shoulders**). Sweep at E=1000:

| emitter `threshold` | 50% | 90% | 99% | z_max (m) | ring frac[39–43°] |
|---|---|---|---|---|---|
| **0.05** (default) | 1.690 | 3.297 | 3.853 | 3.999 | 0.803 (too sharp) |
| 0.01 | 2.072 | 3.843 | 4.393 | 4.549 | — |
| **0.001** | 2.077 | 3.901 | 4.483 | 4.739 | 0.589 |
| 0.0 | 2.082 | 3.904 | 4.488 | 5.288 | — |
| GEANT4 target | 2.094 | 3.916 | 4.502 | — | 0.561 |

At `threshold≈0.001` the emitter matches GEANT4 in **both** longitudinal (<0.5 %) and ring
width (0.589 vs 0.561). The default 0.05 truncates the tail → ~15 % longitudinal compression
(→ recon E +70) **and** over-sharpens the ring. **One config knob, not a net retrain, and not
s_max.** (s_max(1000)=5.29 m is fine; the density inside [0,0.999]·s_max is correct — 0.05 just
refuses to sample where the net says "few photons.")

## Bottom line

The merged engine + new s/s_max net is **GEANT4-faithful** (yield 0.7 %, longitudinal <0.5 %,
ring matched). The REVAL recon delta vs the old baseline is **not** a net defect:

1. **Yield 1.66×** = the recon truth ROOT (1050 MeV) was generated with a **wider Cherenkov
   band ≈[200,700] nm**; the new GEANT4/net use **[275,674] nm**. To reproduce the old recon
   numbers, either compare against GEANT4 made with the matching band, or accept the new band
   as the production standard (then the `cherenkov_photon_norm` knob is unnecessary).
2. **Longitudinal/ring** = `ray_sampling.threshold=0.05` truncates the emitted tail/shoulders.
   Set `threshold≈0.001` (siren_params.json) to recover the GEANT4 longitudinal + ring. This is
   the real fix behind the `cherenkov_smax_norm` interim knob — and it's free (just samples the
   tail the net already models). Tradeoff: more seed bins → marginally more rays; verify recon
   cost.

Repro scripts/data: `LUCiD_dlcheck/data/water/muon/1000MeV_100events.root` (downloaded,
`OpticalPhotonsRaw` tree with `PhotonWavelength`); evaluation snippets in this session.

---

# WAVELENGTH-CONSISTENT RE-TEST — the norm is NOT a real fix (it was stale-data bookkeeping)

Pulling on "why do we need cherenkov_photon_norm": traced the full λ → QE → detect chain and
re-ran recon on the **new GEANT4 ROOT** (carries true `PhotonWavelength`) so the data path
QE-weights by TRUE λ. Scripts: `band_consistent_test.py` (+ loader for the OpticalPhotonsRaw
chunked schema), `opt_multi_truthseed.py` (old-ROOT corrected-vs-uncorrected).

## The 1.66× is a DATA-PATH ARTIFACT, not a model scale

The detected charge = (emitted count) × ⟨QE⟩ × geometry; data and model sample λ ~ 1/λ² over the
**same** band and apply the **same** qe_fn, so ⟨QE⟩ CANCELS — the 1.66× is purely the emitted
COUNT ratio. QE-weighted integrals show why the count gap is spurious:

| band | emitted ∫1/λ² | **QE-detected ∫1/λ²·QE** |
|---|---|---|
| NEW [275,674] | 0.002152 | **0.000238** |
| OLD [200,700] | 0.003571 | **0.000238** |

Emitted ratio OLD/NEW = **1.659**; **QE-detected ratio = 1.000**. The QE curve only spans
[294,648] nm, so the old band's extra photons in [200,294]+[648,700] are QE-dead — physically
undetectable. Both bands deliver the SAME detectable charge. The old recon ROOT has **no
`PhotonWavelength` branch**, so the data path (simulator.py:740, `wavelengths=None`) gives every
one of its 358k photons a FRESH λ over [300,648] + a detectable ⟨QE⟩ — laundering the QE-dead
blue/red into the detection band → data charge 1.66× too high. `cherenkov_photon_norm=1.66` just
rescales the model to the laundered charge.

## Empirical confirmation (5 ev each)

| config | data | norm | thr | data/model q | TRUTH-seed vtx / dE / dt0 |
|---|---|---|---|---|---|
| uncorrected | old ROOT | 1.0 | 0.05 | 1.66 | 32.5 cm / **+586** / −1.86 |
| "corrected" | old ROOT | 1.66 | 0.001 | ~1.0 | 15.2 cm / **−18** / −0.33 |
| band-consistent | NEW ROOT | **1.0** | 0.001 | **0.84** | 17.1 cm / **−135 (−13.5%)** / +0.26 |

The 1.66× COLLAPSES to 0.84 on band-consistent data with NO norm — the artifact is resolved.
The old-ROOT "corrected" dE≈0 was partly two errors cancelling (laundered-high data × norm-up).

## The residual 0.84 is a SEPARATE, genuine model band-clamp bug

Model samples λ over **[300,648]** (water-medium floor 300 ∩ QE support 294–648), but real
Cherenkov spans **[275,674]**. **14.3% of real photons are <300 nm** (+2.7% >648) — physically
lost (UV absorption + QE≈0), correctly dropped in the data — but the net's `nphot=203k` counts
the full [275,674] emission and the model samples them all into [300,648], so it **over-detects
~17%**. Predicted `⟨QE⟩_data/⟨QE⟩_model = 0.1101/0.1326 = 0.831` ≈ observed **0.837**. This drives
dE −13.5%. Honest fix = a ~0.83 model-side normalization (or sample the true emission band so the
medium/QE fringe self-cancels) — NOT a +1.66 scalar, and it points the OPPOSITE way.

## Verdict

- **`cherenkov_photon_norm`: DROP IT.** The 1.66× was λ-bookkeeping on stale data (old ROOT has
  no per-photon wavelengths). Use ROOTs with `PhotonWavelength`.
- **Real residual:** model λ-sampling band [300,648] vs net emission band [275,674] → ~0.83
  over-count (the QE/medium-floor fringe). Small, physical, model-side. The proper normalization
  to chase, if any.
- **`ray_sampling.threshold=0.001`: KEEP.** Genuine, independent longitudinal/ring fix.

---

# RESOLUTION — emission-band fix removes the residual with NO scalar norm

Root cause of the residual 0.84 over-count (dE −13.5%): the simulator sampled model λ over
[300,648] (water floor ∩ QE) while the net's nphot counts the GEANT4 EMISSION band [275,674],
so the QE-dead fringe was never represented. FIX (committed, byte-identical default): new
`setup_event_simulator(cherenkov_emission_band=(lo,hi))` param samples Method-A λ over the net's
emission band; QE(λ)=0 (out-of-knot) + the medium interp-clamp make the fringe photons INERT, so
the detected charge uses the correct in-band fraction. Default `None` = byte-identical (84
targeted tests + unification pins pass).

5-event truth-seed fit, NEW ROOT (true λ), NO norm, threshold=0.001:

| config | data/model q (median) | vtx | dir | dE | dt0 |
|---|---|---|---|---|---|
| sample [300,648] (clamp) | 0.84 | 17.1 cm | 0.91° | **−135 (−13.5%)** | +0.26 |
| sample [275,674] (**fix**) | **0.998** | **14.6 cm** | 0.78° | **−20 (−2.0%)** | +0.08 |

The emission-band fix drives q_tot → 1.0 and dE → −2% with NO `cherenkov_photon_norm`. This is the
old-emitter baseline floor (13.9 cm / dE≈0 / dt0 −0.20). Final recipe for band-consistent recon:
**(1)** data ROOT with `PhotonWavelength` (so QE applies to true λ); **(2)** `cherenkov_emission_band`
= the net's GEANT4 emission band [275,674]; **(3)** `ray_sampling.threshold≈0.001`. No scalar norm.

---

# EXPLICIT emission band — it's an ENERGY cutoff [1.84, 4.51] eV (not a round nm)

The PhotonSim Cherenkov generation cutoff is set in photon ENERGY (Geant4 RINDEX optical range),
not wavelength. Exact per-photon extremes (muon AND electron, 5-decimal agreement):
  min E = 1.84000 eV  -> 673.83 nm  (long-λ cutoff)
  max E = 4.51000 eV  -> 274.91 nm  (short-λ cutoff)
matches the ROOT `PhotonHist_Wavelength` nonzero span [274,674] (1 nm bins). So the authoritative
band is **[1.84, 4.51] eV = [274.91, 673.83] nm**, identical for muon/electron at all energies
(scaling_wl_check: wl=[274.9,673.8] for all 6 files; net nphot/G4 = muon 1.015/1.007/1.003,
electron 1.001/1.000/0.999 @ 500/1000/1500). The eyeballed (275,674) is ~0.1 nm off at each edge
(<0.1% in-band fraction, below shot noise) — use cherenkov_emission_band=(274.91, 673.83) for the
exact value. Constant: hc = 1239.841984 eV·nm.
