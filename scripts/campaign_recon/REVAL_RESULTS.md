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
