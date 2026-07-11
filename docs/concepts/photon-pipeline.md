# The photon pipeline

LUCiD's forward model is a single differentiable pipeline, JIT-compiled and vectorized
(`jax.jit` / `vmap` / `lax.scan`) so gradients flow end to end:

```
sources  →  propagation  →  photon_step (× K)  →  sensor_response
(emit rays)  (ray–geometry)  (scatter/reflect/    (aggregate → per-PMT
              intersection)   absorb, soft weights)  charge & first-arrival time)
```

- **`lucid/sources/`** emits photon rays — from the SIREN track/cascade emitter, or from
  calibration sources (laser, isotropic), or from PhotonSim data.
- **`lucid/propagation/`** intersects rays with the detector geometry and iterates `K` scatter
  bounces.
- **`lucid/simulation/photon_step.py`** does the per-step physics — scatter, reflect, absorb —
  with *soft* weights (a `temperature` controls the softness) so the whole thing is
  differentiable rather than a hard Monte-Carlo branch. Where a branch stays genuinely
  discrete, its gradient is carried by a DiCE score correction — the same technique the
  architecture page calls "DiCE soft weights".
- **`lucid/simulation/sensor_response.py`** aggregates photon weights into per-PMT charge and a
  soft-min first-arrival time.

For **data production** there is one more, non-differentiable stage after the simulator
kernel: **`lucid/simulation/digitizer.py`** decomposes the per-PMT signal into digits (SPE
charge sampling, per-sensor time-window integration, dark noise) and
**`lucid/simulation/trigger.py`** applies an optional sliding-window readout trigger. These
model detector electronics for dataset realism; the differentiable inference path stops at
`sensor_response`. See the [dataset schema](../reference/dataset-schema.md) for what they write.

## The one entry point: `setup_event_simulator`

`lucid.simulation.setup_event_simulator(json_filename, ...)` is the hub. It reads the geometry
JSON, builds the detector, picks the propagator, wires the photon source, and returns a
**JIT-compiled callable**. Its mode flags determine the behaviour and the call signature:

| Mode | flag | callable signature |
|------|------|--------------------|
| Track (reconstruction) | `hit_mode='per_photon'` | `(particle_params, detector_params, key) → (charge, time)` |
| Calibration | `is_calibration=True`, `hit_mode='aggregated'` | `(source, detector_params, key) → (charge, time)` |
| Data-like | `is_data=True`, `hit_mode='realistic'` | `(particle_params, detector_params, key, photon_data) → (charge, time)` |

The table shows the three everyday modes; `hit_mode` has more values for special outputs —
`moments` (charge moments per PMT), `per_segment` (per-(track-segment, sensor) attribution for
dataset production), `waveform`/`waveform_expected` (time-binned waveforms), and
`shotgun_per_photon` (per-photon records for diagnostics).

Other important arguments: `n_photons`, `K` (scatter iterations), `temperature` (soft→hard),
`wavelength_mode` (per-photon λ physics vs scalar), `default_detector_params` (bake the detector
params in so the callable drops that argument), and `detector_type`.

Three more arguments are easy to miss but change the numbers you get out:

- **`use_expected_value`** (default `True`) — `True` runs the *mean-field* forward: each photon
  carries a deterministic soft weight, so the output is the expected charge/time with **no shot
  noise**. `False` switches the per-step scatter/absorb branches to Monte-Carlo *sampling*, so
  the output fluctuates event-to-event like real data. Data mode (`is_data=True`) always samples.
  If you are generating a *truth* dataset that should look shot-noisy, sample it — set
  `use_expected_value=False` (or use `is_data=True`); the mean-field forward is for the
  differentiable inference path, where you want a smooth loss surface, not fluctuations.
- **`max_candidates_per_ray`** (default `4`) — the per-cell sensor cap in the spatial-hash grid the
  propagator uses to find which PMTs a ray can hit. If a grid cell ends up holding more sensors
  than this cap, the extras are **silently dropped** from that cell's candidate list (the geometry
  validator emits a warning when this happens). This is the second, quieter cause of a
  "why are no PMTs lit?" surprise — the first being photons that never reach a sensor. Densely
  packed layouts (mPMT domes) may need a larger cap; the geometry builders scale the grid cell size
  with this value.
- **`apply_smearing`** (default `True`, data mode only) — toggles the SK-like per-PMT charge and
  time smearing applied inside the realistic (`is_data=True`) hit-maker. `False` returns the raw
  hard first-arrival time and summed charge (Bernoulli-QE-sampled but un-smeared) — useful when you
  want data-mode sampling without the electronics resolution folded in.

### Choosing `n_photons`, `K`, `temperature`

These three are performance-vs-fidelity knobs. Honest rules of thumb from how the repo's own
workflows set them:

- **`n_photons`** — enough that the per-PMT statistics are not the limiting noise. In-tree
  examples run from ~200–250k (display/quickstart) up to ~1M (`setup_event_simulator`'s own
  default); reconstruction sweeps use ~600k. More photons is strictly smoother and slower; scale it
  up until the quantity you care about stops moving.
- **`K`** (scatter iterations) — the number of scatter bounces the transport unrolls. Display and
  quick looks use `K=6`; reconstruction uses `K≥8` so the late-scattered tail is faithful. `K` is
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

### The soft-min first-arrival caveat

The differentiable aggregated hit-maker (`make_hits_simulation` in `sensor_response.py`) reports a
**soft-min** first-arrival time, `t_soft = t_min − T·log(Σ_i exp(−(t_i − t_min)/T))`. Because every
term in the sum is ≥ 1 for the earliest photon, `t_soft ≤ t_min`: the soft-min **underestimates the
hard first arrival**, and the gap grows roughly like `T·log(number of contributing photons)` — more
light on a PMT pulls the reported time slightly earlier. This is a differentiability artifact, not
physics. The realistic (data) and moments hit-makers instead read the **hard** geometric
first-arrival (`segment_min`, no soft-min), so use those when you need the true first-photon time.

## Why differentiable?

Because every step is differentiable, `jax.grad` of a loss over the per-PMT output gives exact
gradients with respect to **particle** parameters (energy, vertex, direction, t₀ →
reconstruction) *and* **detector** parameters (scattering, absorption, reflection, QE →
calibration). That is what turns both problems into gradient-based optimization — see
[Reconstruction](../guides/reconstruction.md) and [Calibration](../guides/calibration.md).
