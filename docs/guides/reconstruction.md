# Reconstruction

How LUCiD reconstructs a track `(energy, vertex, direction, t0)` from per-PMT charge + first
hit times. Runnable entry points: `examples/hello_reconstruct.py`, `examples/seed_reconstruct.py`,
the `track_optimization` tutorial, and the `lucid-optimize` CLI.

Because the forward model is differentiable, the refinement step optimizes against **exact
autodiff derivatives** of the loss — not finite differences, and not a library of precomputed
templates — so all nine parameters move together under Gauss-Newton curvature (the coarse grid
below is only a *seed* for this gradient fit, not the fit itself).

## Minimal fit

The whole flow — seed from the data, refine with Fisher-Gauss-Newton, read out a track. This is
the condensed shape of `examples/seed_reconstruct.py` (the full runnable script, including the
cone-direction search that fills in each seed's direction):

```python
import jax, jax.numpy as jnp, numpy as np
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.fitting import (ReconModel, fit_track_multistart, seed_vertex_time,
                           vec9_from_track, track_from_vec9)
from lucid.optimization.grid_search import hierarchical_position_grid_search, get_detector_bounds

GEOM, PHYS, K = 'config/SK_like_geom_config.json', 'config/SK_like_physics_config.json', 8
det = generate_detector(GEOM); ND = len(det.all_points)
POS = np.asarray(det.all_points); bounds = get_detector_bounds(det)

# the differentiable forward the fit calls: per-photon, wavelength-consistent, K>=8
pred = setup_event_simulator(GEOM, 250_000, temperature=0.1, K=K, hit_mode='per_photon',
                             physics_config=PHYS, default_detector_params=True, particle='muon',
                             wavelength_mode=True, pos_grad_threshold=K, n_grad_iters=K)
model = ReconModel(pred, ND, sigma=2.5, delta=1.0)   # energy_from_scale=True (default)

# observed event = per-PMT (charge oc, first-hit time ot); on real data, read from a ROOT file
oc, ot = ...                                          # (ND,) arrays

# two complementary 9-vector seeds, built from the data alone (E0/dirA/dirB from an energy
# scan + cone-direction search — see seed_reconstruct.py):
grid = hierarchical_position_grid_search(jnp.asarray(POS), jnp.asarray(ot), jnp.asarray(oc),
                                         jnp.zeros(3), 0., 0., bounds, levels=6, verbosity=0)
seedA = vec9_from_track(E0, np.asarray(grid['best_position']), dirA, t0=float(grid['best_t0']))
vtxB, t0B = seed_vertex_time(POS, oc, ot)             # arrival-time multilateration
seedB = vec9_from_track(E0, vtxB, dirB, t0=t0B)

# refine: Fisher-Gauss-Newton from both seeds, keep the better basin
res, info = fit_track_multistart(model, oc, ot, [seedA, seedB],
                                 nkeys=8, niters=150, margin=0.01)
track = track_from_vec9(jnp.asarray(res))            # 9-vector [E, x,y,z, dir, t0] -> ParticleParams
```

What each knob configures (all defaults live in `fit_track`, which `fit_track_multistart` calls):

- **`energy_from_scale=True`** (`ReconModel` default) — routes the charge term's *energy*
  gradient through the **total predicted charge**, decoupling the energy scale from small
  emission-shape mismatch.
- **SCALE9 preconditioning** — the GN step is solved in per-parameter units
  `[50 m, 0.2 m ×3, 0.02 ×4 dir cosines, 0.2 ns]` (the `SCALE9` vector), so one step can move
  the MeV-scale energy and the direction cosines together.
- **`fisher_mode='ad'`** (default) — the PSD Gauss-Newton / Fisher metric is built by
  forward-mode autodiff; it is the truer metric and ~2.8× faster than the finite-difference one.
- **lr schedule `lr=4.0` → `lr_final=1.5`** — big early steps settle to small late ones
  (converges ~2× faster than a constant step and avoids late divergence). `ridge_i` is an
  additive Levenberg floor (`ridge_i=0` is forbidden — it runs `t0` away).
- **`readout='polyak'`** (default) — returns the mean of the last `polyak_w=40` iterates, robust
  to the noise-floor wandering (‖g‖ never fully vanishes at the biased minimum).
- **`margin=0.01`** (`fit_track_multistart`) — a loss-margin gate: keep the charge-grid seed
  unless the time seed beats it by 1% of the loss, killing the inward-track "wanderer" tail
  without over-selecting the forward-biased time seed.
- **`nkeys` / `niters`** — PRNG keys averaged per gradient step / Gauss-Newton iterations.

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
- **Seeding**: a charge-weighted position grid (`lucid.optimization.grid_search`) plus a
  time-multilateration vertex/t0 seed (`lucid.fitting.seed_vertex_time`);
  `fit_track_multistart` runs both starts and keeps the winner by a loss-margin gate.

## Wavelength-consistent forward (use this)

The recon forward must be **wavelength-consistent** with the data; with that in place, no
scalar photon-yield normalization is needed.

1. **Data ROOT with per-photon `PhotonWavelength`.** The data path QE-weights each photon by
   its true λ; without it (older ROOT files), photons outside the QE-sensitive band get counted
   as detectable and inflate the apparent charge.
2. **`cherenkov_emission_band=(274.91, 673.83)`** on `setup_event_simulator`. The SIREN net's
   `nphot(E)` counts photons over the PhotonSim Cherenkov emission band
   [274.91, 673.83] nm (4.51 down to 1.84 eV); sampling the model λ over that same band makes the
   out-of-band fringe inert — `qe_fn(λ)` is zero outside the QE knots and the medium curves
   clamp — so detected charge uses the correct in-band fraction.
3. **Emitter sampling**: the default SIREN emitter uses importance seed-sampling over the full
   emission profile, so the low-density longitudinal tail and wide-angle shoulders are kept
   without any tuning. (A `ray_sampling.threshold` knob exists only for the legacy
   `seed_mode='uniform'` path and has no effect under the default.)
4. `wavelength_mode=True`, a realistic per-photon transit-time spread
   (`DetectorParams.response.tts`, e.g. 2.5 ns), and `K≥8` scatter iterations.

!!! note "History: no scaling fudges"
    Legacy `cherenkov_photon_norm` / `cherenkov_smax_norm` scalars were **removed** — they
    compensated for stale wide-band data and emitter truncation; the physical settings above
    supersede them. `cherenkov_emission_band` is a wavelength band, not a scaling factor.

## What constrains what

- **Charge** carries the energy scale and the longitudinal vertex information; its quality is
  limited by the fidelity of the emission model's charge shape.
- **Timing** carries the transverse vertex, direction, and `t0`; its quality is set by the
  per-PMT time resolution (TTS).
- Angles enter through direction cosines whose raw gradients can be kinky — this is why the
  optimizer is Fisher-driven rather than raw-gradient-driven, and why multistart seeding
  matters for events far from the seed.

## If the fit misbehaves

- **Diverging or wandering vertex** — a bad single seed (a mis-multilaterated `t0` dropping the
  fit into the wrong basin) is the usual cause. Use `fit_track_multistart` with both seeds; the
  `margin` gate keeps the safe charge-grid seed unless the time seed decisively wins, which is
  what rescues inward-pointing tracks.
- **Energy frozen or runs away** — a preconditioning problem. SCALE9 scaling is required (an
  unscaled step cannot move the MeV-scale energy against a metre-scale position), and `ridge_i=0`
  is forbidden. Also confirm **both** the predictor and the data forward were given
  `physics_config` — a dropped one mis-normalizes the wavelength QE (data ≫ model) and the energy
  runs away.
- **Biased energy** — use the wavelength-consistent forward below (`cherenkov_emission_band` +
  `wavelength_mode=True`); out-of-band photons otherwise count as detectable, inflate the apparent
  charge, and bias the energy scale.
- **Too slow** — lower `nkeys`/`niters`, coarsen the seed grid (`levels`, the grid params on
  `setup_event_simulator`), or run on GPU (JAX uses it automatically; the fit is ~2–3 min on GPU).

## Open items

- The electron emission surrogate slightly under-predicts charge relative to muons, giving a
  small positive electron energy bias; a look at the electron Cherenkov surrogate is the
  natural next step.
- Store the emission band in the SIREN net metadata so `cherenkov_emission_band` auto-wires
  per particle instead of being passed by hand.
