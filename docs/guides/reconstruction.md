# Reconstruction

How LUCiD reconstructs a track `(energy, vertex, direction, t0)` from per-PMT charge + first
hit times. Runnable entry point: `examples/seed_reconstruct.py`; many events:
`analysis/paper/utils/run_study.py` (the paper's studies) or the loop below.

Because the forward model is differentiable, the refinement step optimizes against **exact
autodiff derivatives** of the loss — not finite differences, and not a library of precomputed
templates — so all nine parameters move together under Gauss-Newton curvature (the coarse grid
below is only a *seed* for this gradient fit, not the fit itself).

## Minimal fit

The whole flow — seed from the data, refine with Fisher-Gauss-Newton, read out a track. The
model and the reconstruction, included from `examples/seed_reconstruct.py`, where `GEOM`, `PHYS`
and `GRID` are SK_like's configs and propagation grid, `TTS = 2.1`, `PARTICLE[hypothesis]` is
`'muon'` or `'electron'`, `seed` is `SEED_RECIPE` and `fit` is `FIT_RECIPE` (`--quick` and
`--steps` change them for a smoke run), and `oc`, `ot` are the observed event:

```python
--8<-- "examples/seed_reconstruct.py:model"
```

```python
--8<-- "examples/seed_reconstruct.py:reconstruct"
```

The detector (sensors, volume, light speed) is the predictor's own, `model.pred.det_geom`. The
example fits a SIREN-sampled event by default, or with `--geant4` a GEANT4 event of the bundled
1 GeV muon slice; `--particle` and `--hypothesis` (`mu` or `e`) choose the event's particle (a GEANT4
event is a muon) and the model.
Its last line is the data loss at the fit, so one event fitted under both hypotheses is a μ/e
test (the lower loss names the particle):

```bash
for hyp in e mu; do
  python examples/seed_reconstruct.py --geant4 --hypothesis "$hyp" --event 0
done
```

Many events are a loop over the observables, each event's keys folded from its index, so no
two events share draws and an event reconstructs the same way whatever else is in the loop:

```python
for i, (oc, ot) in enumerate(events):                # (counts, first-hit times) per event
    res, info = reconstruct(model, oc, ot, seed=SEED_RECIPE, fit=FIT_RECIPE,
                            seed_key=jax.random.fold_in(jax.random.PRNGKey(1), i),
                            fit_key=jax.random.fold_in(jax.random.PRNGKey(2), i))
```

`reconstruct` is the published method: seeds A, B and F from the data, a pre-fit pick, one
`fit_track`. To fit from a start of your own, call
`fit_track(model, oc, ot, start, FIT_RECIPE, key=fit_key)`;
it returns `(theta, history)`, the trajectory included. Change one knob with
`dataclasses.replace`, e.g. `replace(FIT_RECIPE, nkeys=4)`.

What each knob configures (the published values are `FIT_RECIPE`, `MODEL_RECIPE` and
`SEED_RECIPE` in `lucid.fitting`; `ReconModel(pred, sigma=..., nphot_fn=...)` takes a
predictor of your own):

- **Energy from the charge scale** (`ReconModel`) — the charge term's *energy* gradient flows
  through the **total predicted charge**, decoupling the energy scale from small emission-shape
  mismatch.
- **SCALE9 preconditioning** — the GN step is solved in per-parameter units
  `[50 MeV, 0.2 m ×3, 0.02 ×4 dir cosines, 0.2 ns]` (the `SCALE9` vector), so one step can move
  the MeV-scale energy and the direction cosines together.
- **Fisher metric** — the PSD Gauss-Newton / Fisher metric is built by forward-mode autodiff.
  `max_step=3.0` clips each scaled step.
- **lr schedule `lr=4.0` → `lr_final=1.5`** — big early steps settle to small late ones
  (converges ~2× faster than a constant step and avoids late divergence). `mu` is an
  additive Levenberg floor (`mu=0` is forbidden — it runs `t0` away).
- **Polyak readout** — the fit returns the mean of the last `polyak=40` iterates, robust
  to the noise-floor wandering (‖g‖ never fully vanishes at the biased minimum).
- **`nkeys` / `steps`** — PRNG keys averaged per gradient step / Gauss-Newton iterations; the
  keys are `FIT_RECIPE.keys(fit_key)`, split from the event's own fit key, and the pre-fit pick
  scores the starts with the same keys.
- **The seed recipe** (`SEED_RECIPE`, every value stated; nothing has a default) — the energy
  window, the first-light grid's t0 range and levels, the multilateration settings, the cone
  search, and `pick_margin`: a start replaces seed A only if its pre-fit data loss is lower by
  that fraction. Widen `t0_min`/`t0_max` if your events' true t0 lies outside ±20 ns; the energy
  window covers 200–3000 MeV.
- **`seed_key` / `fit_key`** — the event's two random streams: the energy scan and the cone
  searches at A and at B each take an independent key split from `seed_key`, the fit its `nkeys`
  keys from `fit_key`. The paper takes them from the event's `derive_event_keys(...)`
  (`'seed_key'`, `'fit_key'`).

## Pipeline

`lucid.fitting` fits a 9-vector `t9 = [E, x, y, z, sinθ, cosθ, sinφ, cosφ, t0]`:

- **Forward**: `setup_event_simulator(..., source='particle')` (`hits='photon_list'`) →
  per-photon log-weights, times and sensor indices, and the per-PMT expected charge; `ReconModel`
  turns them into the loss, a Poisson charge NLL plus a first-arrival-window time NLL.
- **Fit**: `fit_track`, the Fisher-Gauss-Newton loop whose knobs are listed above.
- **Seeding** (`lucid.fitting.seed`): an energy scan along a probe track; seed A, the
  **first-light grid** (a hierarchical vertex/t0 grid on arrival-time residuals,
  `first_light_grid`); seed B, time multilateration (`seed_vertex_time`); a cone direction
  search for each (`cone_direction`); seed F, their fusion (`fuse_seeds`: B's transverse vertex,
  A's longitudinal vertex and direction, the mean t0). `reconstruct` scores A, B and F by the
  data loss, keeps A unless another wins by `pick_margin` (`pick_by_margin`), and fits once.

## Wavelength-consistent forward (use this)

The recon forward must be **wavelength-consistent** with the data; with that in place, no
scalar photon-yield normalization is needed.

1. **Data ROOT with per-photon `PhotonWavelength`.** The data path QE-weights each photon by
   its true λ; without it (older ROOT files), photons outside the QE-sensitive band get counted
   as detectable and inflate the apparent charge.
2. **The emitter's emission band.** The SIREN emitter's `nphot(E)` counts photons over the
   PhotonSim Cherenkov emission band [274.91, 673.83] nm (4.51 down to 1.84 eV), recorded as
   `emission_band_nm` in its `emitter.json`; the track mode samples the model λ over that band
   by default (`cherenkov_emission_band` on `setup_event_simulator` overrides it). The
   out-of-band fringe is then inert — `qe_fn(λ)` is zero outside the QE knots and the medium
   curves clamp — so detected charge uses the correct in-band fraction.
3. **Emitter sampling**: the default SIREN emitter uses importance seed-sampling over the full
   emission profile, so the low-density longitudinal tail and wide-angle shoulders are kept
   without any tuning.
4. `wavelength_mode=True`, a realistic per-photon transit-time spread
   (`DetectorParams.response.tts`; the paper simulates 2.1 ns), and `K≥8` scatter iterations.

`cherenkov_emission_band` is a wavelength band, not a scaling factor: the forward has no photon-
yield or track-length normalization to tune.

## What constrains what

- **Charge** carries the energy scale and the longitudinal vertex information; its quality is
  limited by the fidelity of the emission model's charge shape.
- **Timing** carries the transverse vertex, direction, and `t0`; its quality is set by the
  per-PMT time resolution (TTS).
- Angles enter through direction cosines whose raw gradients can be kinky — this is why the
  optimizer is Fisher-driven rather than raw-gradient-driven, and why the start matters for
  events far from the seed.

## If the fit misbehaves

- **Diverging or wandering vertex** — a bad start (a mis-multilaterated `t0` dropping the fit
  into the wrong basin) is the usual cause. Use `reconstruct` rather than a single hand-made
  seed: the fused seed F carries the time seed's transverse vertex, which is what rescues
  inward-pointing tracks, and `info['seeds']` shows whether the seeds agree.
- **Energy frozen or runs away** — a preconditioning problem. SCALE9 scaling is required (an
  unscaled step cannot move the MeV-scale energy against a metre-scale position), and `mu=0`
  is forbidden. Also confirm **both** the predictor and the data forward were given
  `physics_config` — a dropped one mis-normalizes the wavelength QE (data ≫ model) and the energy
  runs away.
- **Biased energy** — use the wavelength-consistent forward above (the emitter's emission band,
  `wavelength_mode=True`); a band narrower than the emitter's makes out-of-band photons count as
  detectable, inflates the apparent charge, and biases the energy scale.
- **Too slow** — lower `nkeys`/`steps`, coarsen the seed search (`grid_levels`, `cone_levels`
  in the seed recipe; the grid params on `setup_event_simulator`), or run on GPU (JAX uses it automatically). At the defaults the fit takes about 7–8 min on an
  RTX 2080 Ti, and an 11 GB card needs `XLA_PYTHON_CLIENT_MEM_FRACTION=.95`.

## Known limitation

The electron emission surrogate slightly under-predicts charge relative to muons, giving a small
positive electron energy bias.
