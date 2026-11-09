# The photon pipeline

LUCiD's forward model is a single differentiable pipeline, JIT-compiled and vectorized
(`jax.jit` / `vmap` / `lax.scan`) so gradients flow end to end:

```
sources  →  propagation  →  photon_step (× K)  →  hits
(emit rays)  (ray–geometry)  (scatter/reflect/    (what the sensors report:
              intersection)   absorb, soft weights)  e.g. charge & first-arrival time)
```

- **`lucid/sources/`** emits photon rays — from the SIREN track/cascade emitter, or from
  calibration sources (laser, isotropic), or from PhotonSim data.
- **`lucid/propagation/`** intersects rays with the detector geometry and iterates `K` scatter
  bounces. In the closed surface detectors a photon emitted outside the detector volume, for
  instance by a track that leaves it, carries no weight: the inner detector is optically sealed.
  String telescopes are open media and keep every photon.
  On a surface detector a sensor is weighted by the photon's closest approach to it over the leg
  the photon travels to the wall, not over its infinite ray line.
- **`lucid/simulation/photon_step.py`** does the per-step physics — scatter, reflect, absorb —
  with *soft* weights (a `temperature` controls the softness) so the whole thing is
  differentiable rather than a hard Monte-Carlo branch. Where a branch stays genuinely
  discrete, its gradient is carried by a DiCE score correction — the same technique the
  architecture page calls "DiCE soft weights".
- **`lucid/simulation/hits.py`** turns the photon deposits into what the sensors report (per-PMT
  charge and first-arrival time, a per-photon list, waveforms, ...): one function per kind of hits
  and per detection statistics, in the registry `HITS`.

For **data production** there is one more, non-differentiable stage after the simulator
kernel: **`lucid/simulation/digitizer.py`** decomposes the per-PMT signal into digits (SPE
charge sampling, per-sensor time-window integration, dark noise) and
**`lucid/simulation/trigger.py`** applies an optional sliding-window readout trigger. These
model detector electronics for dataset realism; the differentiable inference path stops at
the hits. See the [dataset schema](../reference/dataset-schema.md) for what they write.

## The one entry point: `setup_event_simulator`

`lucid.simulation.setup_event_simulator(json_filename, ...)` is the hub. It reads the geometry
JSON, builds the detector, picks the propagator, wires the photon source, and returns a
`Simulator`: a callable around the JIT-compiled forward, which also records the `det_geom` it was
built from (and, for a particle, the Cherenkov `emitter`). Four choices define it.

**`source=`** (required) says where the photons come from, and sets the call signature:

| `source` | photons | default `hits` | callable signature |
|----------|---------|----------------|--------------------|
| `'particle'` (reconstruction) | a track through its SIREN emitter | `photon_list` | `(particle_params, detector_params, key)` |
| `'light_source'` (calibration) | a source object: laser, isotropic, cascade, shotgun | `first_arrival` | `(source, detector_params, key)` |
| `'photons'` (data-like) | given photons, e.g. PhotonSim | `first_arrival` | `(detector_params, key, photon_data)` |

With `default_detector_params` set (`True`, or a `DetectorParams`), the params are baked in and
the callable drops that argument: `(particle_params, key)`, `(source, key)`, `(key, photon_data)`.

**`transport=`** `'expected' | 'sampled'` is the propagation: `'expected'` runs the *mean-field*
forward, each photon carrying a deterministic soft weight through every scatter, absorption and
reflection, so the output is the expected charge/time with **no shot noise**; `'sampled'` draws
each per-step branch (Monte Carlo), so the output fluctuates event-to-event like real data. The
default is `'sampled'` for `source='photons'`, which supports only that (given photons are one
realisation), and `'expected'` otherwise. If you are generating a *truth* dataset that should
look shot-noisy, sample it; the mean-field forward is for the differentiable inference path, where
you want a smooth loss surface, not fluctuations.

**`detection=`** `'expected' | 'sampled'` is how QE, the TTS and the charge resolution enter the
hits: as expectations (smooth, differentiable) or as draws. Same defaults as `transport`. The two
are independent: e.g. the JUNO figure's data side transports its given photons sampled and reads
them out expected (`source='photons', detection='expected'`).

**`hits=`** is what the sensors report; with `detection` it picks a function from
`lucid.simulation.hits.HITS` (an unsupported pair raises, naming the supported ones), and
`hits_config=` holds that function's setup-time options (e.g. `window_ns` for a waveform; a key
it does not take raises). Each returns a `NamedTuple`:

| `hits` | `detection` | returns |
|--------|-------------|---------|
| `first_arrival` | `expected` | `FirstArrival(charge, time)` per sensor: expected charge, mean of the predicted first-arrival distribution |
| `first_arrival` | `sampled` | `FirstArrival(charge, time)` per sensor: Bernoulli-QE-sampled charge (smeared by `charge_resolution`), hard first-arrival time with the TTS |
| `photon_list` | `expected` | `PhotonList(log_w, times, indices, charge)`: per-deposit log-weights, arrival times and sensor indices, and the per-sensor charge |
| `segment_truth` | `sampled` | `SegmentTruth`: per-sensor charge and true and smeared times, plus the per-deposit weights, times, sensor and segment indices (dataset production) |
| `waveform` | `expected`, `sampled` | `Waveform(waveform, n_dropped, n_detected)`: time-binned per-sensor waveforms |
| `photon_hits` | `sampled` | `PhotonHits(detected, sensor_id, hit_time, indirect)`: per-photon records for diagnostics |

A new kind of hits is one function with the signature `hits.py`'s docstring states and one `HITS`
entry.

Other important arguments: `n_photons`, `K` (scatter iterations), `temperature` (soft→hard),
`wavelength_mode` (per-photon λ physics vs scalar), `default_detector_params` (bake the detector
params in so the callable drops that argument). The detector class comes from the geometry JSON's
`detector_type`.

Two more settings are easy to miss but change the numbers you get out:

- **Sensor candidates per ray** (`max_candidates_per_ray`, default 4) — the per-cell sensor
  slots of the spatial-hash grid the propagator uses to find which PMTs a ray can hit. It is a grid
  keyword: `setup_event_simulator(..., **grid_params)` passes it, with any explicit cell counts
  (`n_cap`, `n_angular`, `n_height` for a cylinder; `n_divisions` for a sphere; `n_x`, `n_y`, `n_z`
  for a box), through `DetectorGeometry.from_config` to `create_propagator`. Without explicit cell
  counts the grid is sized from it (more candidates, larger cells). When geometry puts more sensors
  in one cell, `create_propagator` raises it to that count, with a warning, so no sensor is dropped
  from a cell it overlaps. The paper's reconstruction studies pass 8 (`PIPELINE['grid']` in
  `analysis/paper/utils/studies.py`): the soft model's deposit can reach a sensor of a
  neighbouring cell, and 4 loses vertex and t0 accuracy at low ray counts.
- **`charge_resolution`** (default `None`) — the per-sensor charge smearing of the sampled
  `first_arrival` and `segment_truth` hits: `None` keeps the raw sampled counts, `'Abe_2013'` is
  SK's fractional Gaussian, `'Bellamy_94'` a per-photoelectron SPE spectrum. Time resolution comes
  only from the per-photon TTS (`DetectorParams.response.tts`). (`apply_smearing` is not a setup
  argument: it is a production dataset-config option, `lucid_options.apply_smearing`, read by
  `lucid-run-job`.)

### Choosing `n_photons`, `K`, `temperature`

These three are performance-vs-fidelity knobs. Honest rules of thumb from how the repo's own
workflows set them:

- **`n_photons`** — enough that the per-PMT statistics are not the limiting noise. In-tree
  examples run from ~200–250k (display/quickstart) up to ~1M (`setup_event_simulator`'s own
  default); the paper's reconstruction studies use 250k (`MODEL_RECIPE.n_rays`). More photons
  is strictly smoother and slower; scale it
  up until the quantity you care about stops moving.
- **`K`** (scatter iterations) — the number of scatter bounces the transport unrolls. Displays and
  quick looks use `K=4`–`6` (`hello_simulate.py` uses 4); reconstruction and calibration use
  `K≥8` (`MODEL_RECIPE.K = 8`) so the late-scattered tail is faithful. `K` is
  a hard truncation: photons that would scatter more than `K` times are simply cut, so under-setting
  it biases the long-time / wide-angle tail. For string (telescope) geometries, `compute_K_min`
  (`lucid/geometry/string_sizing.py`) picks the smallest `K` with `p_scat**K ≤ eps_K` — i.e. `K`
  large enough that the residual un-terminated weight is below a floor — which for a long ~20-string
  telescope tends to land higher than the water-tank display value.
- **`temperature`** — the soft-assignment softness. `temperature=None` gives **hard** (step-function)
  photon→sensor assignment and hard first-arrival timing — the non-differentiable *truth* readout;
  in-tree data/truth simulators pass `temperature=None`. A small positive value (the recon sweeps use
  `0.1`) softens the assignment just enough for gradients to flow. Lower is more faithful but the
  gradient gets noisier/stiffer; `0.1`–`0.2` is the working range. Use `None` for the forward you
  compare against, a small float for the differentiable prediction you fit.

### The expected-value first-arrival time

The differentiable first-arrival hits (`first_arrival_expected` in `hits.py`) report the
**mean of the predicted first-arrival distribution**, the photon model of the reconstruction's
time likelihood (`lucid.losses.first_arrival_window_nll`). Each deposit i carries Poisson(w_i)
photo-electrons at its arrival time t_i, each smeared by the transit-time spread σ
(`DetectorParams.response.tts`); the expected count before t is `R(t) = Σ_i w_i Φ((t − t_i)/σ)`, and
`P(first > t | at least one) = (e^(−R(t)) − e^(−μ)) / (1 − e^(−μ))`. The reported time is its mean,
smooth in the weights and the times. A dim PMT reports the weight-averaged arrival; a bright one its
earliest light, less the spread's order-statistic shift. The sampled detection
(`first_arrival_sampled`) instead samples the photo-electrons and reports the **hard** first
arrival (`segment_min`) of one event.

## Why differentiable?

Because every step is differentiable, `jax.grad` of a loss over the per-PMT output gives exact
gradients with respect to **particle** parameters (energy, vertex, direction, t₀ →
reconstruction) *and* **detector** parameters (scattering, absorption, reflection, QE →
calibration). That is what turns both problems into gradient-based optimization — see
[Reconstruction](../guides/reconstruction.md) and [Calibration](../guides/calibration.md).
