# LUCiD architecture

**One differentiable forward engine with regime modes**, serving two co-equal lines —
inference/calibration (gradient-descent reconstruction, detector calibration) and
production (PhotonSim → datasets) — across two detector families: water/WbLS
Cherenkov tanks and ice/water string telescopes (see the
[bundled detectors](../getting-started/detectors.md)).

Units are **meters, nanoseconds, MeV** throughout.

## Package layout

- **`lucid/simulation/`** — `setup_event_simulator` (the JIT-compiled forward hub),
  `photon_step`, `hits` (what the sensors report, the registry `HITS`).
- **`lucid/geometry/`** — registry-dispatched detectors (`cylinder`, `sphere`, `box`, `string`);
  `Cylinder.from_pmt_file(npz)` for measured SK/HK/WCTE arrays.
- **`lucid/propagation/`** — differentiable ray–geometry intersection and multi-bounce transport.
- **`lucid/wavelength/`** — wavelength-dependent optics (`medium`, `spectrum`, `optical_model`).
- **`lucid/siren/`** — SIREN surrogate for Cherenkov/dE-dx emission (trained on PhotonSim tables).
- **`lucid/sources/`** — track/cascade/calibration emitters, the PhotonSim ROOT reader and event
  placement.
- **`lucid/fitting/`** — one damped Gauss-Newton loop serving both problems: calibration
  ([guide](../guides/calibration.md)) and reconstruction
  ([guide](../guides/reconstruction.md)); `crb(model, theta)` gives the Cramer-Rao bound.
- **`lucid/gradient_analysis/`** — 1D/2D loss-landscape sweeps.
- **`lucid/dataset/`** — the HDF5 dataset format: schema, writer, reader, batch layout, labels
  ([dataset schema](../reference/dataset-schema.md)).
- **`lucid/production/`** — PhotonSim/GENIE → HDF5 dataset chain and cluster deployment.
- Top-level modules: `detector_params.py` (`DetectorParams`, `ParticleParams`), `losses.py`,
  `overlap.py` (the soft sensor-overlap tables), `visualization.py` (event displays), `pdg.py`
  (particle masses), `utils.py`.

The console entry points are in the [CLI reference](../reference/cli.md).

## The one engine: `setup_event_simulator`

`lucid/simulation/simulator.py::setup_event_simulator(json_filename, ...)` is the central
assembly point. It reads a geometry JSON + a physics config, builds the detector geometry,
medium, propagator, photon source, and per-PMT response, and returns a **`Simulator`**: a frozen
record of the JIT-compiled function (its call signature and return per `source` and `hits` are in
the [photon pipeline](photon-pipeline.md#the-one-entry-point-setup_event_simulator)) and what it
was built from (`.det_geom`, whose `.medium` is the material physics, `.reflection_model`,
`.default_detector_params`, and for a particle `.emitter`, the Cherenkov emitter). Calling the
record calls the function. Everything is
differentiable end-to-end (`lax.scan`/`vmap`/`jit`, DiCE soft weights).

Passing `default_detector_params=True` (as the [quickstart](../getting-started/quickstart.md)
does) bakes the detector params into the returned function and drops them from the
signature, so the callable is invoked with two args — `(particle_params, key)` —
and records the baked tuple as `.default_detector_params` (None otherwise). Leave it `False` (the
default) when you need to differentiate w.r.t. `DetectorParams` (calibration).

Behaviour is selected by **regime flags / config**, not separate engines:

| Axis | Modes | Selected by |
|------|-------|-------------|
| **Geometry** | cylinder · sphere · box · **string (telescope)** | geom JSON `detector_type` (registry dispatch) |
| **Propagation** | surface (reflection) · **volume (per-DOM, no wall)** | `is_volume = isinstance(detector, StringTelescope)` |
| **Emission** | Cherenkov · **+ scintillation** | `medium.emission_processes` (+ `cherenkov_fraction` split) |
| **Optics** | scalar · wavelength-dependent (Rayleigh + Mie + g) | `wavelength_mode`, physics config |
| **Source** | SIREN track · light source · given (ROOT/PhotonSim) photons | `source` |
| **Transport** | expected (mean-field) · sampled (Monte Carlo) | `transport` |
| **Hits** | first_arrival · photon_list · **segment_truth** · waveform · photon_hits, each expected or sampled where supported | `hits`, `detection` (registry `HITS`) |

Every regime addition is gated so it is never even traced for the surface (cylinder/sphere/box,
Cherenkov-only, water) path. The suite pins that path's numbers in one place,
`tests/physics/test_forward_golden.py`: the tiny detector's calibration forward and its gradient, at
a relative 1e-3 (not byte-identical: float32 reductions move across CPUs and jax versions). The
exact before/after comparison of a change is the per-change oracle's job, not a committed reference.

## Differentiable parameters

`lucid/detector_params.py` holds `DetectorParams`, the fittable detector state, and
`ParticleParams`, the track; their layout and what is a parameter rather than an argument are on
[Parameters (DetectorParams)](detector-params-vs-args.md).

## Geometry (registry-dispatched)

`lucid/geometry/registry.py` holds `@register_detector(name)`; `generate_detector(path)` reads
a geometry JSON, looks up the class from its `detector_type` and calls its
`from_config(geometry_definitions, config_dir)`. A new geometry is a `Detector` subclass that is
registered and implements `from_config`; `generate_detector` needs no edit. Each geometry has a
matching propagator in `lucid/propagation/`. The four types and the detectors they fit are listed
in [model your own detector](../guides/model-your-own-detector.md#1-choose-your-geometry-family).
`DetectorGeometry.from_config` builds geometry + medium
(`load_medium(material)`, or the physics config's `medium_model`) + propagator:

- **cylinder / sphere / box** → `lucid/propagation/*` surface propagators (grid-celled
  sensor lookup, soft overlap, reflection).
- **string** (`lucid/geometry/string.py::StringTelescope`, DOM positions from an NPZ) →
  `lucid/propagation/string/string_propagator.py`: each ray is matched to its nearest
  strings, then snapped to the nearest DOMs (IceCube-style optical modules) along each,
  with a fallback for rays that exit the instrumented volume. Implementation details live
  in the module docstrings.

## Emission sources

- **SIREN Cherenkov** (`lucid/sources/siren_rays.py` + `lucid/siren/net.py`): a
  sinusoidal net trained on PhotonSim photon tables over `(E, angle, s/s_max)`. The
  emitter's `emitter.json` carries the count+range model — `s_max(E)` (track range) and
  `N_photons(E)` (photon budget) — so each ray's intensity is `pmf × n_photons(E)` directly,
  with no manual reweighting. Emission time `t0(d,E)` is a stretched-exponential cubic.
- **Scintillation** (`make_scintillation_surrogate_fn`): a dE/dx SIREN drives isotropic
  rays with Chou-quenched weights (`S/(1+kB·d+C·d²)`), a biexponential emission delay
  (`tau_rise/tau_fall`), and Moyal-sampled wavelengths. Built only when the medium
  scintillates; concatenated with Cherenkov rays.
- **Calibration sources** (`calibration_sources.py`): laser / isotropic, source passed at
  call time.
- **Data** (`root_reader.py` + `data_event.py` recon path, `interaction.py` production path): photons read
  from PhotonSim ROOT files, in metres. An event is placed by one rigid motion about its recorded
  vertex (`lucid.sources.placement.rigid_motion`, applied by `data_event.place_photons` and
  `interaction.place_raw_entry`; production draws the vertex with `placement.sample_vertex`), and
  its photons are zero-padded to a fixed buffer (`data_event.pad_photon_data`) whose padding the
  simulator masks out.

## Config system

Two JSONs per detector in `config/`: `*_geom_config.json` (shape, sensors) and
`*_physics_config.json` (optical properties, composable — each property independently scalar
or λ-dependent, referencing `config/materials/*.json` + `config/pmt/*` QE curves). Material
JSONs carry the medium spec, including the `scintillation` block (light yield, timing,
Moyal spectrum) that WbLS inherits.

## The two lines on one engine

- **Inference / calibration** (`lucid/fitting/`): build the
  simulator, take gradients of a loss (Poisson NLL, likelihood, factored) w.r.t.
  `DetectorParams` / `ParticleParams`, solve with a damped Gauss-Newton (per-PMT gains profiled
  in closed form) or any optax optimizer such as Adam. Fisher/CRB from autodiff. The data path turns ROOT photons into observed `(charge, time)` with `lucid.sources.data_event.simulate_data_event` (origins in **metres**, as
  `lucid.sources.root_reader` reads them; one mm-to-m conversion, in the reader).
- **Production** (`lucid/production/`): `lucid-run-job` drives GENIE → Geant4 macro →
  PhotonSim subprocess (`$PHOTONSIM_BIN`) → the LUCiD writer. Its loops
  (`lucid/production/dataprod/drivers.py`) run the library half of event generation,
  `lucid.sources.interaction`: place a PhotonSim entry, propagate it with
  `hits='segment_truth'` (per-(segment, sensor) deposits), and digitize. The **digitizer**
  (`lucid/simulation/digitizer.py` — SPE charge sampling, per-sensor time-window integration
  into digits, dark noise) turns the photon deposits into readout, and, where the dataset
  config asks for it, a **readout trigger** (`lucid/simulation/trigger.py`, sliding-window)
  gates the event; `lucid/simulation/readout.py` holds the two with the selection. The result
  is four parallel HDF5 files (`sensor/ hits/ step/ labl/`) in the format of `lucid.dataset`
  (schema, writer, reader, layout, labels). Cluster scheduling lives in
  `lucid/production/platform/`.

## Numerical precision on GPU

The stack is float32 throughout, the fitting iterate included, with `jax_enable_x64` off; no
`lucid` module sets `jax_default_matmul_precision` or `jax_enable_x64`. On an NVIDIA GPU, JAX's
default float32 `dot`/`einsum` is TF32: float32 inputs and outputs, truncated accumulation. On CPU
the same setting is exact. Published numbers were captured at JAX's default. Measured on one L40S,
TF32 against `'highest'` moved the per-PMT predictor sum by 1.3e-3, the loss gradient by 2.3x
(with sign flips), the AD Fisher by 5.8x and the fitted vertex by 11.2 cm. A run that needs full
float32 accumulation sets `JAX_DEFAULT_MATMUL_PRECISION=highest` before JAX initialises, and should
say so with its results.
