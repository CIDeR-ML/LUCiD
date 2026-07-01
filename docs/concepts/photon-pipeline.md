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
  differentiable rather than a hard Monte-Carlo branch.
- **`lucid/simulation/sensor_response.py`** aggregates photon weights into per-PMT charge and a
  soft-min first-arrival time.

## The one entry point: `setup_event_simulator`

`lucid.simulation.setup_event_simulator(json_filename, ...)` is the hub. It reads the geometry
JSON, builds the detector, picks the propagator, wires the photon source, and returns a
**JIT-compiled callable**. Its mode flags determine the behaviour and the call signature:

| Mode | flag | callable signature |
|------|------|--------------------|
| Track (reconstruction) | `hit_mode='per_photon'` | `(particle_params, detector_params, key) → (charge, time)` |
| Calibration | `is_calibration=True`, `hit_mode='aggregated'` | `(source, detector_params, key) → (charge, time)` |
| Data-like | `is_data=True`, `hit_mode='realistic'` | `(particle_params, detector_params, key, photon_data) → (charge, time)` |

Other important arguments: `n_photons`, `K` (scatter iterations), `temperature` (soft→hard),
`wavelength_mode` (per-photon λ physics vs scalar), `default_detector_params` (bake the detector
params in so the callable drops that argument), and `detector_type`.

## Why differentiable?

Because every step is differentiable, `jax.grad` of a loss over the per-PMT output gives exact
gradients with respect to **particle** parameters (energy, vertex, direction, t₀ →
reconstruction) *and* **detector** parameters (scattering, absorption, reflection, QE →
calibration). That is what turns both problems into gradient-based optimization — see
[Reconstruction](../RECONSTRUCTION.md) and [Calibration](../CALIBRATION.md).
