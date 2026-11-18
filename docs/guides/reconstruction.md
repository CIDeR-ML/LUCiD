# Reconstruction

LUCiD reconstructs a track (energy, vertex, direction, t0) from the per-PMT charge and
first-hit time of one event. Seeds are built from the data, then a Gauss-Newton fit on the
differentiable forward model refines all nine parameters together, using exact autodiff
derivatives.

Runnable: `examples/seed_reconstruct.py`. It is meant for a GPU: about 8 minutes on an RTX
2080 Ti, which needs `XLA_PYTHON_CLIENT_MEM_FRACTION=.95` to fit in its 11 GB.

## Minimal fit

From `examples/seed_reconstruct.py`, where `GEOM` and `PHYS` are the `SK_like` configs, `GRID`
the propagation grid (`n_cap=80, n_angular=120, n_height=80, max_candidates_per_ray=8`),
`TTS = 2.1` ns, `PARTICLE[hypothesis]` is `'muon'` or `'electron'`, `seed` is `SEED_RECIPE`
and `fit` is `FIT_RECIPE`:

```python
--8<-- "examples/seed_reconstruct.py:model"
```

```python
--8<-- "examples/seed_reconstruct.py:reconstruct"
```

`oc` and `ot` are the observed charge and first-hit time per PMT (zero time on an unhit PMT),
arrays of length `model.pred.det_geom.num_sensors`. The example makes them from a
SIREN-sampled track by default, or from a GEANT4 event of the bundled 1 GeV muon file with
`--geant4` (`--event 0..4`). `--particle` sets the event's particle and `--hypothesis` the model
(`mu` or `e`); `--quick` and `--steps` make a smoke run.

The last line printed is the data loss at the fit. Fitting one event under both hypotheses is a
muon/electron test, since the lower loss names the particle:

```bash
for hyp in e mu; do
  python examples/seed_reconstruct.py --geant4 --hypothesis "$hyp" --event 0
done
```

For many events, give each event its own keys so no two events share random draws:

```python
for i, (oc, ot) in enumerate(events):
    res, info = reconstruct(model, oc, ot, seed=SEED_RECIPE, fit=FIT_RECIPE,
                            seed_key=jax.random.fold_in(jax.random.PRNGKey(1), i),
                            fit_key=jax.random.fold_in(jax.random.PRNGKey(2), i))
```

`res` is the 9-vector `[E, x, y, z, sinθ, cosθ, sinφ, cosφ, t0]`; `track_from_vec9(res)` turns
it into a `ParticleParams` and `vec9_dir(res)` into a unit direction. `info` holds the three
seeds (`info['seeds']`), their pre-fit losses, which one was fitted (`info['which']`) and the fit
history. To fit from your own start, call `fit_track(model, oc, ot, start, FIT_RECIPE,
key=fit_key)`, which returns `(theta, history)`.

The data side must match the model: the same detector, `wavelength_mode=True`, the physics
config given to both, and a TTS equal to the model's `sigma`.

## The method

All names are in `lucid.fitting`.

1. **Energy seed**: a scan of the total predicted charge along a probe track through the
   detector centre.
2. **Seed A**, first-light grid: a hierarchical vertex and t0 grid on arrival-time residuals,
   then a cone search for the direction.
3. **Seed B**, time multilateration (`seed_vertex_time`), then a cone search.
4. **Seed F** (`fuse_seeds`): B's transverse vertex, A's longitudinal vertex and direction, and
   their mean t0.
5. **Pick**: A, unless B or F has a data loss lower by `pick_margin` (`pick_by_margin`).
6. **Fit** (`fit_track`): Fisher-Gauss-Newton on the loss, a Poisson charge likelihood plus a
   first-arrival time likelihood.

The model (`ReconModel`) predicts with `hits='photon_list'` and takes the energy scale from the
emitter's photon yield, so the energy gradient acts through the total charge only. There is no
charge-normalisation factor to tune.

## Settings

The published values are `MODEL_RECIPE`, `SEED_RECIPE` and `FIT_RECIPE`. Change one with
`dataclasses.replace`, for example `replace(FIT_RECIPE, nkeys=4)`.

`MODEL_RECIPE`: `n_rays=250_000`, `K=8`, `temperature=0.1`, first-arrival window `delta=1.0` ns,
`time_weight=1.0`, `energy_scale='nphot'`.

`FIT_RECIPE`:

| field | value | meaning |
|---|---|---|
| `steps` | 150 | Gauss-Newton iterations |
| `nkeys` | 8 | random keys averaged per step, fixed through the fit, split from `fit_key` |
| `lr`, `lr_final` | 4.0, 1.5 | step scale, annealed linearly |
| `lam` | 0.01 | Marquardt damping on the Fisher diagonal |
| `mu` | 0.1 | Levenberg floor; `mu=0` lets t0 run away |
| `refresh` | 8 | steps between Fisher-matrix rebuilds |
| `max_step` | 3.0 | clip on each component of the scaled step |
| `polyak` | 40 | the result is the mean of the last 40 iterates |

Steps are taken in units of `SCALE9 = [50 MeV, 0.2 m ×3, 0.02 ×4, 0.2 ns]`, so one step can
move the energy, the vertex and the direction cosines together.

`SEED_RECIPE` sets the energy scan (`energy_center=1600`, `energy_delta=1400` MeV, so 200 to
3000 MeV), the first-light grid (`t0_min=-20`, `t0_max=20` ns and its levels), the
multilateration and the cone search. Every field must be given. Widen `t0_min` and `t0_max` if
your events' true t0 can fall outside ±20 ns.

## What constrains what

- Charge sets the energy and the vertex along the track.
- Timing sets the transverse vertex, the direction and t0; its precision is set by the TTS.

## If the fit misbehaves

- The vertex wanders or diverges: usually a bad start. Use `reconstruct` rather than one
  hand-made start, and check whether `info['seeds']` agree.
- The energy runs away: check that both the model and the data simulator were given the physics
  config, and that `mu > 0`.
- The energy is biased: the data must carry per-photon wavelengths (PhotonSim's
  `PhotonWavelength`) and be simulated with `wavelength_mode=True`. Do not narrow
  `cherenkov_emission_band` below the emitter's band.
- It is too slow: lower `nkeys` or `steps`, coarsen the seed search (`grid_levels`,
  `cone_levels`), or use a GPU.

## Known limitation

The electron emitter slightly under-predicts charge relative to muons, which gives a small
positive energy bias for electrons.
