# DetectorParams (fittable pytree) vs args (config/runtime) — the delineation

## The litmus test
- **Would you ever take a gradient w.r.t. it / fit it?** → it's a **fittable pytree leaf** (DetectorParams, or
  SourceParams / ParticleParams for the other two fit modes).
- **Is it a configuration / runtime choice fixed for a given setup, or a known physical constant we never calibrate?**
  → it's an **arg** (passed to `setup_event_simulator` / the forward), not a pytree leaf.

Corollary: a leaf only earns a place in a fit pytree if some run might list it in `trainable_fields`. Everything that is
always-fixed (geometry, the medium *reference* shapes, the soft-overlap correction, the group velocity) is an arg.

## Three fittable pytrees (the optimization state) — `trainable_fields` selects across them
| pytree | holds | free in |
|---|---|---|
| **DetectorParams** | the detector's physical/response state (below) | calibration, joint |
| **SourceParams** | calibration source nuisances (laser dir, iso pos, per-source intensity) | calibration (optional), joint |
| **ParticleParams** | track (energy, position, direction, t0) | reconstruction, joint |

`free = DetectorParams` → calibration; `free = ParticleParams` → recon; `free = both` → joint self-calibration.
SourceParams is the calibration analogue of ParticleParams (the Task-2 source-nuisance work), used only when sources
are uncertain.

## DetectorParams (nested by physics; leaves = exactly the fittable detector DOF)
```
DetectorParams(
  scattering = ScatteringParams(rayleigh_amp[], mie_amp[]/mie_curve[n_ctrl], g[]),
  absorption = AbsorptionParams(abs_curve[n_ctrl]),
  reflection = ReflectionParams(wall_R0_curve[n_ctrl], wall_p[], wall_fspec[],
                                 cathode_nr_curve[n_ctrl], cathode_nk[], sensor_fspec[]),
  response   = ResponseParams(qe_amp[], qe_curve[n_ctrl], spe_width[], tts[]),
  per_pmt    = PerPmtParams(qe_corr[NS], gain[NS], t0[NS], walk[NS]),   # the Schur block (structural)
)
```
Notes / decisions per field:
- **tts, spe_width** ARE DetectorParams (`response`): they are calibrated PMT properties. (Today `TTS_NS` is a
  module-level env read — it MOVES here as a field; the data-gen and the timing observable both read `dp.response.tts`.)
- **g, cathode_nk** are physical and fittable-in-principle but usually FROZEN (hard to measure) — present as fields,
  default just not in `trainable_fields` (SK fixes n_imag).
- **reflection holds the SUPERSET** of model params; the chosen `reflection_fn` (an arg) reads the subset it needs
  (scalar→R0 constant, Schlick→R0_curve+p, Fresnel→nr_curve+nk; fspec for spec/diff). Unused fields stay frozen. One
  DetectorParams structure, model selects — no per-model pytree variant.
- **per_pmt** are all `(NS,)`; the fitter Schur-marginalizes this group structurally (gauge-fixed). `qe_corr`=rate
  efficiency, `gain`=charge-per-PE, `t0`=offset, `walk`=TQ-map slope.

## Args (config / runtime — NOT in any fit pytree)
| arg | why it's an arg, not a leaf |
|---|---|
| **geometry** (sensor positions, detector shape) | fixed per detector; via the geometry registry |
| **medium** (reference λ-shapes: Rayleigh 1/λ⁴, pure-water absorption, glass Sellmeier, bench QE) | the FIXED reference the DetectorParams deviations multiply; never fit |
| **control_λ** (the curve grid, e.g. SK 337/375/398/405/445) | parameterization metadata paired with the `*_curve` lengths; not a value to fit |
| **sources** (nominal laser/iso positions + spectra; or the SIREN net) | the run's emission setup; the fittable *nuisance* lives in SourceParams |
| **reflection_fn** | the reflection *model* (scalar/Schlick/Fresnel); its *params* are DetectorParams |
| **observables** (which of charge_mean / charge_var / first_arrival) | a fit-config list, not a physical value |
| **N (photons), K (bounces), temperature (soft overlap), rng/keys** | runtime/differentiability knobs |
| **response mode** (Expected model vs Sample truth) | which response, not a parameter |
| **OVERLAP_RENORM** | a soft-overlap correction *constant* (≈hard/soft total), depends on geometry+temperature, NOT a detector property → config (or computed at setup) |
| **n_group / speed_of_light** | known constant from the medium; we don't calibrate c (could be promoted to a frozen field only if ever needed) |
| **CYL_SQRT_EPS** and other numerical guards | numerical config, default 0 |
| **medium material name, wavelength band [lo,hi]** | source/medium config |

## Borderline calls (resolved)
- **TTS / SPE width** → DetectorParams (calibrated). Not env, not arg.
- **OVERLAP_RENORM** → arg/config (a differentiability correction, not a physical detector parameter). Computable from
  the hard/soft total at setup.
- **n_group (timing velocity)** → arg (known). Promote to a frozen DetectorParams field only if a future study fits it.
- **g, cathode n_imag** → DetectorParams fields, default frozen.
- **control_λ** → arg/config (the basis grid), must stay consistent with the curve field lengths.
- **Source position/intensity** → SourceParams (fittable nuisance) when uncertain; the *nominal* setup is an arg.

## The cleanup this implies for the canonical lucid (the module-level env reads → here)
Current module-level env reads become proper inputs:
- `TTS_NS` → `dp.response.tts` (a field).
- `OVERLAP_RENORM` → a setup arg / computed constant.
- `CYL_SQRT_EPS` → a setup arg (numerical), default 0.
- `PSTEP_DBG` / experimental `SIREN_*` / `*_NOTIME` → DELETE (refuted investigation scaffolding); keep `SIREN_IMPORTANCE`
  as a real (validated) emission option, exposed as an arg not an env flag.
- custom_vjp `nan_to_num` → opt-in arg + logged.

## One-line
A leaf is in a fit pytree iff some run might calibrate it; DetectorParams = the detector's physical+response DOF nested
by physics (with `per_pmt` the Schur block, `tts`/`spe_width` fields, reflection a superset the model picks from);
everything else — geometry, the medium reference shapes, control_λ, the source setup, the reflection-model choice, N/K/
temperature, OVERLAP_RENORM, n_group, numerical guards — is an arg.
