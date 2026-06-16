# LUCiD architecture (merged engine)

This document describes LUCiD after the `unification` ↔ `refactor-v2` reconciliation:
**one differentiable forward engine with regime modes**, serving two co-equal lines —
inference/calibration (gradient-descent reconstruction, detector calibration) and
production (PhotonSim → v3 datasets) — and three co-equal first-class detector
families (water/WbLS Cherenkov tanks, ice/water string telescopes).

Units are **meters, nanoseconds, MeV** throughout.

## The one engine: `setup_event_simulator`

`lucid/simulation/simulator.py::setup_event_simulator(json_filename, ...)` is the central
assembly point. It reads a geometry JSON + a physics config, builds the detector geometry,
medium, propagator, photon source, and per-PMT response, and returns a **JIT-compiled
callable** `(ParticleParams|source|photon_data, DetectorParams, key) → hits`. Everything is
differentiable end-to-end (`lax.scan`/`vmap`/`jit`, DiCE soft weights).

Behaviour is selected by **regime flags / config**, not separate engines:

| Axis | Modes | Selected by |
|------|-------|-------------|
| **Geometry** | cylinder · sphere · box · **string (telescope)** | geom JSON `detector_type` (registry dispatch) |
| **Propagation** | surface (reflection) · **volume (per-DOM, no wall)** | `is_volume = isinstance(detector, StringTelescope)` |
| **Emission** | Cherenkov · **+ scintillation** | `medium.emission_processes` (+ `cherenkov_fraction` split) |
| **Optics** | scalar · wavelength-dependent (Rayleigh + Mie + g) | `wavelength_mode`, physics config |
| **Source** | SIREN track · calibration source · ROOT/PhotonSim data | `is_calibration` / `is_data` |
| **Readout** | aggregated · per_photon · realistic · moments · **per_segment** · waveform(_expected) · shotgun | `hit_mode` |

The surface (cylinder/sphere/box, Cherenkov-only, water) path is **byte-identical** to the
pre-merge forward — every regime addition is gated so it is never even traced for that path.
This is locked by `tests/reconciliation/test_tripwire.py`.

## Differentiable parameters: nested `DetectorParams`

`lucid/detector_params.py` — a JAX-pytree of physics-grouped sub-tuples (the `_SUBTUPLES`
registry drives generic tree-walks):

`scattering` (scatter_length, mie_scatter_length, g) · `absorption` · `reflection`
(wall/sensor; scalar or angular Schlick/Fresnel) · `response` (qe, spe_width, tts) ·
`per_pmt` (qe_corrections, gain, t0, walk) · `scintillation` (S, kB, C, tau_rise, tau_fall,
moyal_*). Each sub-tuple has normalize/denormalize + bounds, so any subset can be optimized
directly. Non-scintillating media leave the scintillation block at neutral 0 (no light) —
so adding it kept the water forward byte-identical.

## Geometry (registry-dispatched)

`lucid/geometry/registry.py` holds `@register_detector(name)`; `generate_detector(config)`
looks up the class. `DetectorGeometry.from_config` builds geometry + medium
(`make_medium(material)`) + propagator:

- **cylinder / sphere / box** → `lucid/propagation/*` surface propagators (grid-celled
  sensor lookup, soft overlap, reflection).
- **string** (`lucid/geometry/string.py::StringTelescope`, DOM positions from an NPZ) →
  `lucid/propagation/string/string_propagator.py` (skew-line ray↔string distance, top-K
  strings, DOM-bracket snap, per-DOM overlap + envelope-exit fallback).

## Emission sources

- **SIREN Cherenkov** (`lucid/sources/siren_rays.py` + `lucid/siren/core.py`): a
  sinusoidal net trained on PhotonSim photon tables over `(E, angle, s/s_max)`. The
  trained-model metadata carries the count+range model — `smax(E)` (track range) and
  `nphot(E)` (photon budget) — so each ray's intensity is `pmf × n_photons_fn(E)` directly,
  with no manual reweighting. Emission time `t0(d,E)` is a stretched-exponential cubic.
- **Scintillation** (`make_scintillation_surrogate_fn`): a dE/dx SIREN drives isotropic
  rays with Chou-quenched weights (`S/(1+kB·d+C·d²)`), a biexponential emission delay
  (`tau_rise/tau_fall`), and Moyal-sampled wavelengths. Built only when the medium
  scintillates; concatenated with Cherenkov rays.
- **Calibration sources** (`calibration_sources.py`): laser / isotropic, source passed at
  call time.
- **Data** (`event_io.py` recon path + `event_generation.py` production path): photons read
  from PhotonSim ROOT files.

## Config system

Two JSONs per detector in `config/`: `*_geom_config.json` (shape, sensors) and
`*_physics_config.json` (optical properties, composable — each property independently scalar
or λ-dependent, referencing `config/materials/*.json` + `config/pmt/*` QE curves). Material
JSONs carry the medium spec, including the `scintillation` block (light yield, timing,
Moyal spectrum) that WbLS inherits.

## The two lines on one engine

- **Inference / calibration** (`lucid/fitting/`, `lucid/optimization/`): build the
  simulator, take gradients of a loss (Poisson NLL, likelihood, factored) w.r.t.
  `DetectorParams` / `ParticleParams`, solve with Gauss-Newton+Schur / Adam. Fisher/CRB from
  autodiff. The data path loads ROOT photons via `pad_photon_data` (origins in **cm**, the
  simulator divides by 100).
- **Production** (`lucid/production/`): `lucid-run-job` drives GENIE → Geant4 macro →
  PhotonSim subprocess (`$PHOTONSIM_BIN`) → the LUCiD v3 writer. `event_generation.py` reads
  ROOT photons (meters), feeds the simulator with `hit_mode='per_segment'` (per-(segment,
  sensor) PE), and writes four parallel HDF5 files (`sensor/ hits/ edep/ labl/`). The
  modular data-gen layer (`root_reader, event_builder, v3_writer, seed_utils, …`) is
  self-contained; cluster scheduling lives in `cluster_common/`.

**Unit convention (unify-on-cm):** the simulator data boundary is **cm** (matching
`pad_photon_data`). `event_generation` works in meters and multiplies photon origins ×100 at
the simulator call, so both lines share one data impl.

## Invariants (enforced by `tests/`)

- **Water byte-identical** forward (scalar + wavelength) + AD-Fisher + leaf order —
  `tests/reconciliation/test_tripwire.py` (re-capture with `CAPTURE=1`).
- **Env-free forward**: zero `os.environ` reads in the forward/physics path
  (`test_no_env_reads_in_lucid_package`); `lucid/production/` is exempt (env-driven
  orchestration — `PHOTONSIM_BIN`, GENIE, cluster).
- **Differentiability**: scintillation Chou gradients correctly signed; volume optical-param
  gradients finite through multi-scatter (`test_scintillation.py`, `test_string_volume.py`);
  the new emitter's energy gradient matches FD (`test_emitter_energy_gradient.py`).

## Reconciliation status

Phases 1–4 are landed on `unification` (= `integration/unify-main`): merged DetectorParams,
emission dispatch, cubic t0, native SIREN emitter, scintillation, volume scatter + string
telescopes, modular sources, per_segment, and the production subsystem. The volume phase
function now carries the Rayleigh+Mie+g mixture (B-mie), with the single-step ice-optical
deposit gradient validated AD==FD. Remaining: the MULTI-STEP (K>1) ice optical-param
gradient is UNBIASED (the pathwise AD agrees with the common-random-number FD in the mean,
~0.3 sigma over ~80 keys) but VERY HIGH-VARIANCE — the discrete per-step DOM-candidate
selection (top_k / searchsorted bracket / argmin) injects large per-key gradient spikes
(per-key AD std ~50x the FD's, growing with K). So the ice/string reconstruction floor
(B-fit → R) is gated on VARIANCE REDUCTION (a soft/differentiable candidate selection, or
CRN/antithetic sampling), not on a bias fix; plus full GPU re-validation campaigns (recon
floor / calibration CRB) and an end-to-end production run (needs the external PhotonSim/GENIE
binaries). See `docs/RECONCILIATION_PLAN.md` for the full plan and per-phase findings.
