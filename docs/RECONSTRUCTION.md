# Reconstruction — recipe & findings

How LUCiD reconstructs a track `(energy, vertex, direction, t0)` from per-PMT charge + first
hit times, and the **band-consistent forward recipe** established in the merged-engine
re-validation.

## Pipeline

`lucid.fitting` fits a 9-vector `t9 = [E, x, y, z, sinθ, cosθ, sinφ, cosφ, t0]`:
- **Forward**: `setup_event_simulator(..., hit_mode='per_photon')` → per-PMT `(μ charge, first-arrival time-NLL)`.
- **Loss**: `ReconModel` = Poisson charge NLL + first-arrival-window time NLL.
- **Optimizer**: `fit_track` / `fit_track_multistart` — consistent **Fisher-Gauss-Newton**,
  SCALE9-preconditioned, FD Fisher metric (`fisher_mode='fd'`, `lr=8`), additive Levenberg
  floor, min‖g‖ readout. Seeders: charge-grid + time-multilateration two-start.

## Band-consistent forward recipe (USE THIS)

The recon forward must be **wavelength-consistent** with the data. No scalar photon-yield norm
is needed — the SIREN net is GEANT4-faithful (see below).

1. **Data ROOT with per-photon `PhotonWavelength`.** The data path QE-weights each photon by its
   true λ; without it (old ROOTs) the QE-dead UV/IR photons get laundered into the detection
   band and inflate charge ~1.66× (a stale-data artifact, NOT a model defect).
2. **`cherenkov_emission_band=(274.91, 673.83)`** on `setup_event_simulator`. The net's `nphot(E)`
   counts photons over the PhotonSim Cherenkov emission band **[1.84, 4.51] eV = [274.91, 673.83] nm**;
   sampling the model λ over that band lets `qe_fn(λ)=0` (out of QE knots [294,648]) + the medium
   interp-clamp make the out-of-band fringe inert, so detected charge uses the correct in-band
   fraction. Default `None` samples only [300,648] → over-counts ~17% (energy bias ≈ −13%).
3. **`ray_sampling.threshold ≈ 0.001`** (siren_params.json). The default 0.05 truncates the
   low-density Cherenkov longitudinal tail + angular shoulders (energy-biasing); 0.001 matches
   GEANT4 to <0.5%.
4. `wavelength_mode=True`, per-photon `DetectorParams.response.tts` (e.g. 2.5 ns), K≥8.

> The legacy `cherenkov_photon_norm` / `cherenkov_smax_norm` scalars were **removed** — they
> compensated the stale-data artifact (1) and the threshold (3); the physical fixes above
> supersede them. `cherenkov_emission_band` is the λ band, not a scaling fudge.

## Why no yield norm: the net is GEANT4-faithful

`scaling_wl_check.py` over muon+electron at 500/1000/1500 MeV: net `nphot(E)` matches GEANT4
`NOpticalPhotons` to **≤1.5%** (muon 1.015/1.007/1.003; electron 1.001/1.000/0.999), and the
emission band is universally [274.91, 673.83] nm. The old "1.66× low" reading came entirely from
fitting a wide-band ([200,700] nm) stale ROOT with a narrow-band model.

## Gradient / Fisher / optimizer (validated)

- AD gradient unbiased for t0/x/y (AD/profile 0.90–1.07); longitudinal z ~1.5× soft (documented);
  angles kinky → Fisher-driven, not raw-gradient-driven.
- AD Fisher is 0.001–0.5× the FD diagonal (low-variance, near-singular cond 8e13); **FD Fisher
  is well-conditioned (cond ~330)** → production uses `fisher_mode='fd'` + `lr=8`.
- `fit_track` converges from the truth seed AND a 40 cm / 5° / +120 MeV / +1.5 ns perturbation to
  the same basin.

## Resolution (band-consistent, no norm; truth-seed, SK_like, 2.5 ns TTS, K=8)

20-event truth-seed fits (`opt_band_multi.py`); data = full GEANT4 yield, model = 150k SIREN rays.

| particle | E (MeV) | q_tot ratio | vertex (median) | direction | energy bias |
|---|---|---|---|---|---|
| muon | 500 | ~1.00 | ~18 cm | ~1.5° | ~−1.5% |
| muon | 1000 | ~1.00 | ~10 cm | ~0.9° | ~−2% |
| muon | 1500 | ~1.01 | ~9 cm | ~0.7° | ~−3% |
| electron | 500 | ~1.02 | ~13 cm | ~1.0° | ~+5% |
| electron | 1000 | ~1.01 | ~11 cm | ~1.2° | ~+2% |

q_tot ratio ≈ 1.0 everywhere with **no scalar norm**. Plots: `scripts/campaign_recon/plots/`
(`dist_<particle>.png` distributions, `process_<particle>.png` optimization trajectories).

## Open items

- **Electron +2–5% energy bias** (model slightly under-predicts electron charge, q-ratio ≳1) — a
  small electron-emitter/shower residual, separate from the wavelength work; worth a look at the
  electron Cherenkov surrogate.
- Longitudinal vertex is charge-shape (SIREN-fidelity) limited; transverse/direction/t0 are
  timing-limited (see the recon-CRB memory series).
- Store the emission band in the net metadata so `cherenkov_emission_band` auto-wires per particle.
