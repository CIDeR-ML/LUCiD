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

## Why differentiable?

Because every step is differentiable, `jax.grad` of a loss over the per-PMT output gives exact
gradients with respect to **particle** parameters (energy, vertex, direction, t₀ →
reconstruction) *and* **detector** parameters (scattering, absorption, reflection, QE →
calibration). That is what turns both problems into gradient-based optimization — see
[Reconstruction](../guides/reconstruction.md) and [Calibration](../guides/calibration.md).
