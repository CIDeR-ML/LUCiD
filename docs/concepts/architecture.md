# How the simulator works

One forward model serves simulation, reconstruction, calibration and dataset production. It is
a chain of JAX functions, compiled with `jax.jit` and vectorised over photons, so a gradient
taken on its output reaches every input:

```
source ──► propagation ──► photon step (× K) ──► hits
emit rays   ray-geometry     scatter, reflect,     what the sensors report
            intersection     absorb                (charge, first-arrival time, ...)
```

`lucid.simulation.setup_event_simulator` assembles the chain from a geometry config and a
physics config and returns it as one callable. Its arguments are in the
[simulator reference](../reference/simulator.md).

## The stages

**Source.** Where the photons start:

- a particle track, whose Cherenkov light is drawn from a SIREN emitter. The emitter gives the
  photon density over energy, opening angle and normalised track position `s/s_max(E)`, and
  carries the track range `s_max(E)`, the photon yield `N_photons(E)` and an emission-time
  model. Trained emitters exist for muons and electrons in water.
- scintillation, for a scintillating medium (WbLS): a dE/dx SIREN emitter drives isotropic
  light with quenched yield, a rise/fall emission delay and a sampled spectrum
  ([optics](optics.md#wbls-scintillation)).
- a calibration source passed at call time: a laser, an isotropic flasher, a cascade, or a
  "shotgun" of given rays.
- photons given at call time, for example read from a PhotonSim ROOT file
  (`lucid.sources.root_reader`, positions converted to metres on read).

**Propagation.** Each ray is intersected with the detector. Cylinder, sphere and box detectors
have a closed wall that reflects and absorbs; light emitted outside the detector volume carries
no weight. String detectors (neutrino telescopes) have no wall: each ray is matched to nearby
strings and DOMs in an open medium.

**Photon step.** `K` iterations of scattering (Rayleigh and Mie), absorption and reflection.
`K` is a hard cut: light that would scatter more than `K` times is dropped.

**Hits.** The photon deposits become what the sensors report: per-sensor charge and
first-arrival time, a per-photon list for the reconstruction likelihood, waveforms, or
per-segment truth for datasets.

## Expected and sampled

Two switches decide whether the output fluctuates like data:

- `transport='expected'` carries each photon's expected weight through every scatter,
  absorption and reflection. The output is the mean, with no shot noise, and is smooth in the
  parameters. `transport='sampled'` draws each branch, like a Monte Carlo.
- `detection` does the same for QE, transit-time spread and charge resolution at the sensor.

Fits use the expected forward as the model and compare it with sampled (or real) data.

Which sensor a photon reaches is a soft assignment: a Gaussian kernel of width
`temperature × sensor_radius` around the photon's closest approach. `temperature=None` makes it
a hard step, which is exact but has no gradient with respect to geometry. Discrete choices that
remain, such as specular against diffuse reflection, get their gradient from a score-function
(DiCE) term.

## Datasets

Dataset production (`lucid-run-job`) runs the same chain with given PhotonSim photons and
`hits='segment_truth'`, then two NumPy stages with no gradients: a digitizer (charge
integration windows, SPE charge, time jitter, dark noise) and an optional sliding-window
trigger. See [digitizer and trigger](../reference/digitizer-and-trigger.md) and the
[dataset schema](../reference/dataset-schema.md).

## Package map

| package | contents |
|---|---|
| `lucid.simulation` | `setup_event_simulator`, the photon step, the hits functions (`HITS`), digitizer and trigger |
| `lucid.geometry` | the detector classes (`cylinder`, `sphere`, `box`, `string`), `generate_detector(path)` |
| `lucid.propagation` | ray-geometry intersection and sensor lookup |
| `lucid.wavelength` | media, spectra and the per-photon optical model |
| `lucid.siren` | SIREN emitters and their training (`lucid.siren.training`) |
| `lucid.sources` | track, cascade and calibration sources; the PhotonSim ROOT reader; event placement |
| `lucid.fitting` | calibration, reconstruction, the Gauss-Newton loop, the Cramér-Rao bound |
| `lucid.gradient_analysis` | 1D and 2D loss-landscape sweeps |
| `lucid.dataset` | the HDF5 dataset format: writer, reader, layout |
| `lucid.production` | the PhotonSim and GENIE dataset chain and its cluster tools |
| `lucid.detector_params` | `DetectorParams`, `ParticleParams`, the physics-config loader |
| `lucid.losses`, `lucid.visualization` | likelihoods; event displays |

## Precision on GPU

LUCiD runs in float32 throughout and never enables `jax_enable_x64`. On an NVIDIA GPU, JAX's
default float32 matrix product uses TF32 (reduced-precision accumulation); on a CPU it is exact.
The published numbers use JAX's default. Measured on one L40S, switching to full precision moved
the per-PMT prediction sum by 1.3e-3, the loss gradient by a factor 2.3 (with sign flips), the
autodiff Fisher matrix by 5.8x and a fitted vertex by 11.2 cm. To run at full float32 precision,
set `JAX_DEFAULT_MATMUL_PRECISION=highest` before JAX initialises, and say so with the results.
