# Unification plan — simplified free/not, nested DetectorParams, full impact, phasing

## 1. "Free vs not" — simplify: a trainable SET, the rest is STRUCTURAL (drop the per-field registry)

The per-field registry (transform/gauge/prior/obs/free/marginalize per field) was over-engineered. Almost all of it is
determined by STRUCTURE, not a declaration:

- **free vs not** = a `set[str]` of trainable field names → `make_optimization_mask(params, trainable_fields)`
  ALREADY EXISTS in `detector_params.py` and yields the boolean pytree for masked optimization. Calibration / recon /
  joint = three different sets. That's the whole mechanism. Nothing else needed for "which is free".
- **marginalize (per-PMT Schur)** = STRUCTURAL: any leaf of shape `(NS,)` (i.e. living in the `per_pmt` group) is
  Schur-marginalized; everything else is a GN global. Determined by where it lives, not a flag.
- **gauge (mean-log-0 / mean-0)** = part of the per-PMT marginalization (the Schur block is gauge-fixed). Not per-field.
- **prior (curvature)** = STRUCTURAL: any `curve` leaf (a length-`n_ctrl` control-point array in the optical group)
  gets the 2nd-difference smoothness penalty. Determined by being a control-point curve, not a flag.
- **obs (which observable constrains it)** = NOT NEEDED. The Jacobian routes each param to whatever observable it
  affects automatically; the fit uses all active observables jointly.
- **transform (log vs linear)** = the one thing that's a genuine convention choice (see Q3). Either a uniform "log all
  positive leaves" convention or the existing normalize-to-[0,1] bounds — NOT a per-field table.

So the only per-fit inputs are: (a) `trainable_fields` set, (b) `active_observables` list, (c) which `reflection_fn`.
Everything else falls out of the param's GROUP and SHAPE. This is the simplification — the structure does the work.

## 2. Nested DetectorParams (tuples-in-tuples) — group so structure encodes handling

Nest by physics domain, with a designated `per_pmt` group (JAX handles nested NamedTuples natively; matches the DTRAX
`Species` nested-sub-tuple pattern):

```python
class DetectorParams(NamedTuple):
    scattering:  ScatteringParams     # rayleigh_amp, mie_amp(/mie_curve), g
    absorption:  AbsorptionParams     # abs_curve (control points)
    reflection:  ReflectionParams     # wall_R0_curve, wall_p, wall_fspec, cathode_nr_curve, cathode_nk, sensor_fspec
    response:    ResponseParams       # qe_amp, qe_curve, spe_width, tts        (global response)
    per_pmt:     PerPmtParams         # qe_corr(NS), gain(NS), t0(NS), walk(NS)  (the Schur block)
```
The fit reads it structurally: `per_pmt` → Schur-marginalized (gauge-fixed); the four global groups → GN; any
`*_curve` leaf → curvature prior. `trainable_fields` selects within. Bounds/normalize/save/load all already recurse via
`jax.tree.map` and `_fields`, so they keep working on the nested tree.

**What nesting BUYS:** (i) the per-PMT/global split — the single most important fit distinction — is now structural,
not a flag; (ii) grouping documents intent + lets you train a whole domain (`'reflection'`) at once; (iii) reflection
model choice maps to one sub-tuple. **What it COSTS:** every `dp.scatter_length` reference in the base sim becomes
`dp.scattering...`; the loaders/photon_step-unpacking change (Q1 weighs this).

## 3. Everything affected (full impact, esp. what nesting + new params touch)

| File / area | impact |
|---|---|
| `detector_params.py::DetectorParams` | restructure to nested + new leaves (amplitudes, curves, gain/t0/walk/w/tts); `default_bounds`/`create_default`/`normalize`/`mask`/`save`/`load` follow the nesting (mostly `tree.map`, low-touch) |
| `load_physics_config` / physics JSON schema | now reads curve arrays + per-PMT defaults + the new groups; `_project_missing_scalars` becomes "project curve from medium" |
| `simulation/simulator.py::_get_optical_arrays` | replaced by `optical_model(dp, medium, control_λ)`; unpacks the nested dp; ONE path (no wavelength_mode branch) |
| `simulation/photon_step.py` | takes per-photon `R0_λ, nr_λ` + calls `reflection_fn`; otherwise unchanged (already the validated DiCE step) |
| `sources/calibration_sources.py`, `siren_rays.py` | `Source.emit -> PhotonBatch(...,wavelengths)`; `Spectrum` types; the wavelength sampling moves here OUT of the simulator |
| make_hits (`sensor_response.py`) | becomes `ResponseModel` (Expected/Sample) with QE(λ)/qe_corr/gain/w/t0/walk; 5 hit_modes → observable methods (charge_mean, **charge_var (NEW)**, first_arrival) |
| `overlap.py` | reflection-rate use moves into `reflection_fn`; OVERLAP_RENORM stays |
| recon code (`LUCiD_recon/*`) | uses `setup_event_simulator` → affected only where it unpacks dp / sets optical; track-fit path unchanged; ParticleParams unchanged |
| NEW `fitting/` (gauss_newton, fisher, schur) | one GN+FD-Hessian+pos-clip+ridge; one CRB; reads `trainable_fields` + active observables; merges recon `gnrec` + calib `gn_fast` |
| config | a calibration config: detector + sources(+spectra) + trainable_fields + active_observables + reflection model + budget |

## 4. Phased plan (incremental, validate each phase against the mie_hunter numbers before wiring into base)

- **P0 (decisions).** Settle the questions below (structure, free-set granularity, opt-space, migration home).
- **P1 — params + optical_model + reflection_fn (standalone).** Nested `DetectorParams`, `optical_model`, the three
  `reflection_fn`s. Test in isolation: reproduce `wl_engine2`/`wlt_engine` per-photon optical + reflection numbers.
- **P2 — ResponseModel + observables.** Expected/Sample response; charge_mean, **charge_var**, first_arrival; QE(λ)/
  per-PMT/gain/w/t0/walk. Test: reproduce the Phase-2 QE↔gain split + the timing first-arrival.
- **P3 — sources + spectra.** `Source`/`Spectrum`/`PhotonBatch`; move λ-sampling out of the simulator. Test: laser +
  broadband iso emission matches.
- **P4 — fitting/ (GN+Schur+Fisher).** One optimizer reading `trainable_fields` + observables. Test: reproduce the
  flexible-curve fit (6 curves) + the full-calibration-vs-scale.
- **P5 — wire into `setup_event_simulator`.** Replace `_get_optical_arrays`/make_hits with the new components; commit
  the unified DiCE forward to base `lucid/`. Migrate recon onto it (verify vtx 7-8cm/dir 0.26°/E 0.26%).
- **P6 — joint fit.** `trainable_fields` = detector ∪ track → self-calibration.

Each phase is independently testable and reversible; the base sim isn't touched until P5.

## 5. Open questions (genuine choices — asked separately)
1. DetectorParams structure: nest-by-physics vs flat-plus-fields vs nest-by-fit-role.
2. Free/not granularity: a field-name set vs whole-group toggles.
3. Optimization space: log-space (fractional CRB) vs normalize-to-bounds (existing).
4. Migration home: new module first then wire in, vs refactor base DetectorParams in place.
