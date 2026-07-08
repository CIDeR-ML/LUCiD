# Reconstruction

How LUCiD reconstructs a track `(energy, vertex, direction, t0)` from per-PMT charge + first
hit times. Runnable entry points: `examples/hello_reconstruct.py`, `examples/seed_reconstruct.py`,
the `track_optimization` tutorial, and the `lucid-optimize` CLI.

## Pipeline

`lucid.fitting` fits a 9-vector `t9 = [E, x, y, z, sinθ, cosθ, sinφ, cosφ, t0]`:

- **Forward**: `setup_event_simulator(..., hit_mode='per_photon')` → per-PMT
  `(μ charge, first-arrival time NLL)`.
- **Loss**: `ReconModel` = Poisson charge NLL + first-arrival-window time NLL. By default
  (`energy_from_scale=True`) the energy gradient of the charge term routes through the **total
  predicted charge**, which decouples the energy scale from small shape mismatch between the
  emission model and the data.
- **Optimizer**: `fit_track` / `fit_track_multistart` — a Fisher-Gauss-Newton loop,
  SCALE9-preconditioned, with an autodiff Fisher metric (`fisher_mode='ad'`, the default), a
  learning-rate schedule (`lr=4.0` → `lr_final=1.5`), an additive Levenberg floor, and a
  Polyak-averaged readout (`readout='polyak'`).
- **Seeding**: `lucid.optimization.grid_search` — a charge-weighted position grid plus a
  time-multilateration vertex/t0 seed; `fit_track_multistart` runs both starts and keeps the
  winner by a loss-margin gate.

## Wavelength-consistent forward (use this)

The recon forward must be **wavelength-consistent** with the data; with that in place, no
scalar photon-yield normalization is needed.

1. **Data ROOT with per-photon `PhotonWavelength`.** The data path QE-weights each photon by
   its true λ; without it (older ROOT files), photons outside the QE-sensitive band get counted
   as detectable and inflate the apparent charge.
2. **`cherenkov_emission_band=(274.91, 673.83)`** on `setup_event_simulator`. The SIREN net's
   `nphot(E)` counts photons over the PhotonSim Cherenkov emission band
   [1.84, 4.51] eV = [274.91, 673.83] nm; sampling the model λ over that same band lets
   `qe_fn(λ)=0` outside the QE knots plus the medium interp-clamp make the out-of-band fringe
   inert, so detected charge uses the correct in-band fraction.
3. **Emitter sampling**: the default SIREN emitter uses importance seed-sampling over the full
   emission profile, so the low-density longitudinal tail and wide-angle shoulders are kept
   without any tuning. (A `ray_sampling.threshold` knob exists only for the legacy
   `seed_mode='uniform'` path and has no effect under the default.)
4. `wavelength_mode=True`, a realistic per-photon transit-time spread
   (`DetectorParams.response.tts`, e.g. 2.5 ns), and `K≥8` scatter iterations.

> Legacy `cherenkov_photon_norm` / `cherenkov_smax_norm` scalars were **removed** — they
> compensated for stale wide-band data and emitter truncation; the physical settings above
> supersede them. `cherenkov_emission_band` is a wavelength band, not a scaling fudge.

## What constrains what

- **Charge** carries the energy scale and the longitudinal vertex information; its quality is
  limited by the fidelity of the emission model's charge shape.
- **Timing** carries the transverse vertex, direction, and `t0`; its quality is set by the
  per-PMT time resolution (TTS).
- Angles enter through direction cosines whose raw gradients can be kinky — this is why the
  optimizer is Fisher-driven rather than raw-gradient-driven, and why multistart seeding
  matters for events far from the seed.

## Open items

- The electron emission surrogate slightly under-predicts charge relative to muons, giving a
  small positive electron energy bias; a look at the electron Cherenkov surrogate is the
  natural next step.
- Store the emission band in the SIREN net metadata so `cherenkov_emission_band` auto-wires
  per particle instead of being passed by hand.
